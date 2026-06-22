#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a "find similar poems" recommender from the Ganjoor dataset.

This script wires the Hugging Face dataset ``mabidan/ganjoor`` (classical
Persian poetry) into the project's flagship **Graph-LM** path and produces a
ready-to-query poem recommender:

  1. Load and (optionally) filter / sample the Ganjoor poems.
  2. Train a Persian Graph-LM on them — GCN graph encoder + gated graph-token
     fusion over a multi-relational word graph (co-occurrence, PMI, stem,
     subword). This is the project's "best mode".
  3. Embed every poem with the trained model and save a ``poem_index.pt`` next
     to the checkpoint.

The result lands in ``runs/ui/<name>/`` so the graphical UI (``app.py``) lists
it automatically in the "🪄 شعریار" playground tab, where you can paste a poem
and get the most similar poems back.

The defaults are tuned to finish in a few minutes on a laptop CPU: a capped
co-occurrence/PMI graph (``--graph-top-k``) and a large batch keep the per-step
GNN cheap. Pass ``--threads`` to use a fixed number of CPU cores.

Examples
--------
A quick, CPU-friendly demo (a few minutes, uses 6 cores)::

    python scripts/build_ganjoor_recommender.py --threads 6

Focus on a handful of poets::

    python scripts/build_ganjoor_recommender.py \
        --poets "حافظ,سعدی,مولوی,خیام,فردوسی" --limit 2000 --threads 6

Scale up for a serious model (slow on CPU; use a GPU)::

    python scripts/build_ganjoor_recommender.py \
        --limit 40000 --epochs 8 --d-model 256 --layers 4 \
        --relations cooccurrence,pmi,stem,subword --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the package importable when run as ``python scripts/...`` from the repo.
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

RUNS_DIR = BASE_DIR / "runs" / "ui"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a Graph-LM on Ganjoor and build a poem recommender.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--name", default="ganjoor-sheryar",
                   help="Checkpoint folder name under runs/ui/.")
    p.add_argument("--limit", type=int, default=800,
                   help="Maximum number of poems to use (0 = all ~119k).")
    p.add_argument("--poets", default="",
                   help="Comma-separated poet names to keep (empty = all poets).")
    p.add_argument("--min-chars", type=int, default=40,
                   help="Skip poems shorter than this many characters.")
    p.add_argument("--keep-prose", action="store_true",
                   help="Keep prose entries too (by default only verse-shaped "
                        "poems are kept, dropping mislabelled prose).")
    p.add_argument("--pc-removal", type=int, default=1,
                   help="Remove this many common/anisotropy directions from the "
                        "poem embeddings (0 disables). 1-2 is typical.")
    p.add_argument("--max-chars", type=int, default=1200,
                   help="Truncate poems longer than this many characters.")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--block-size", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=32,
                   help="Training batch size. On CPU the whole graph is "
                        "re-encoded per step, so a bigger batch = fewer steps "
                        "= much faster training.")
    p.add_argument("--vocab", type=int, default=2000,
                   help="Unigram tokenizer vocabulary size (pieces).")
    p.add_argument("--learning-rate", type=float, default=7e-4)
    p.add_argument("--encoder", default="gcn",
                   choices=["gcn", "gat", "graphsage", "rgcn", "none"],
                   help="Graph encoder (the GNN). 'none' disables the graph.")
    p.add_argument("--fusion", default="gated",
                   choices=["gated", "context_gated", "add"])
    p.add_argument("--relations", default="cooccurrence,pmi",
                   help="Comma-separated graph relations to build. cooccurrence "
                        "and pmi are fast; stem/subword are O(vocab^2) to build "
                        "and slow on CPU — add them only on small vocab / GPU.")
    p.add_argument("--window", type=int, default=3,
                   help="Co-occurrence window size (smaller = fewer edges).")
    p.add_argument("--graph-top-k", type=int, default=20,
                   help="Keep only the top-K strongest edges per node "
                        "(0 = no cap). Caps graph size so the per-step GNN on "
                        "CPU stays fast — the key CPU-speed lever.")
    p.add_argument("--min-count", type=int, default=2,
                   help="Drop graph nodes/edges below this co-occurrence count.")
    p.add_argument("--threads", type=int, default=0,
                   help="CPU threads for torch (0 = leave default / use the "
                        "OMP_NUM_THREADS environment variable).")
    p.add_argument("--index-batch-size", type=int, default=16,
                   help="Batch size when embedding poems for the index.")
    p.add_argument("--device", default="cpu", help="cpu or cuda.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _looks_like_poetry(raw: str) -> bool:
    """Heuristic: keep verse, drop prose (e.g. Qur'an tafsir mislabelled as a
    poem). Persian classical poetry is many short hemistich lines; prose is a
    few very long paragraph lines."""
    lines = [ln.strip() for ln in str(raw or "").split("\n") if ln.strip()]
    if len(lines) < 2:
        return False
    lengths = sorted(len(ln) for ln in lines)
    median = lengths[len(lengths) // 2]
    longest = lengths[-1]
    # Hemistichs are short; a very long line signals a prose paragraph.
    return median <= 60 and longest <= 110


def _clean_text(raw: str, max_chars: int) -> str:
    """Normalise a poem into a compact, single-spaced string for training."""
    text = str(raw or "").replace("‌", "‌").strip()
    # Collapse internal line breaks/whitespace; verse boundaries are not needed
    # for language-model training or bag-of-hidden-state embeddings.
    text = " ".join(text.split())
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text


def load_poems(args: argparse.Namespace) -> list[dict]:
    """Load, filter and sample poems from the Ganjoor dataset."""
    from datasets import load_dataset

    print("📥 Loading mabidan/ganjoor (first run downloads & caches it)…")
    dataset = load_dataset("mabidan/ganjoor", split="train")
    print(f"   dataset rows: {len(dataset):,}")

    wanted_poets = {p.strip() for p in args.poets.split(",") if p.strip()}
    if wanted_poets:
        dataset = dataset.filter(lambda r: str(r.get("poet", "")).strip() in wanted_poets)
        print(f"   after poet filter {sorted(wanted_poets)}: {len(dataset):,} rows")

    # Shuffle for variety, then take the requested number of poems.
    dataset = dataset.shuffle(seed=args.seed)

    poems: list[dict] = []
    for row in dataset:
        raw = row.get("text", "")
        if not args.keep_prose and not _looks_like_poetry(raw):
            continue
        text = _clean_text(raw, args.max_chars)
        if len(text) < args.min_chars:
            continue
        poems.append(
            {
                "id": str(row.get("id", "")),
                "poet": str(row.get("poet", "")).strip(),
                "poem": str(row.get("poem", "")).strip(),
                "cat": str(row.get("cat", "")).strip(),
                "text": text,
                # Keep the original (line-broken) form for nicer display.
                "display": str(row.get("text", "")).strip(),
            }
        )
        if args.limit and len(poems) >= args.limit:
            break

    print(f"✅ Selected {len(poems):,} poems "
          f"from {len({p['poet'] for p in poems})} poet(s).")
    return poems


def train_model(args: argparse.Namespace, output_dir: Path, corpus: list[str]) -> dict:
    """Train the Graph-LM (best mode) on the poem corpus."""
    from rakhshai_graph_nlp.lm.model import GraphLMConfig
    from rakhshai_graph_nlp.lm.trainer import LMTrainingConfig, train_graph_lm

    relations = [r.strip() for r in args.relations.split(",") if r.strip()]
    use_graph = args.encoder != "none"
    # Smaller corpora benefit from the low-data regularisers (augmentation,
    # curriculum, light contrastive + graph dropout). Large corpora do not.
    low_data = len(corpus) < 5000

    tcfg = LMTrainingConfig(
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        block_size=args.block_size,
        learning_rate=args.learning_rate,
        graph_relations=relations if use_graph else None,
        graph_relation_mode="embedding",
        graph_scope="document",
        graph_window_size=args.window,
        graph_min_count=args.min_count,
        graph_top_k=(args.graph_top_k or None),
        fusion_levels="token",
        checkpoint_metric="next_token",
        tokenizer_type="unigram",
        tokenizer_unigram_num_pieces=args.vocab,
        text_augmentation=low_data,
        curriculum_learning=low_data,
        contrastive_weight=0.05 if low_data else 0.0,
        node_dropout=0.05 if low_data else 0.0,
        edge_dropout=0.1 if low_data else 0.0,
        subgraph_sampling_ratio=0.9 if low_data else 1.0,
        early_stopping_patience=0,  # run every requested epoch
        device=args.device,
        seed=args.seed,
    )
    mcfg = GraphLMConfig(
        vocab_size=1,  # set from the tokenizer inside the trainer
        max_seq_len=args.block_size,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        dim_feedforward=args.d_model * 4,
        graph_encoder=args.encoder,
        fusion=args.fusion,
    )
    print(f"🎓 Training Graph-LM: encoder={args.encoder} fusion={args.fusion} "
          f"d_model={args.d_model} layers={args.layers} epochs={args.epochs} "
          f"vocab≈{args.vocab} relations={relations if use_graph else '—'}")
    started = time.perf_counter()
    metrics = train_graph_lm(
        corpus,
        training_config=tcfg,
        model_config=mcfg,
        graph_encoder=args.encoder,
        fusion=args.fusion,
    )
    elapsed = time.perf_counter() - started
    print(f"   ✓ trained in {elapsed:,.1f}s — best perplexity "
          f"{metrics.get('best_perplexity', float('nan')):.3f} "
          f"(epoch {metrics.get('best_epoch')})")
    return metrics


def build_index(args: argparse.Namespace, output_dir: Path, poems: list[dict]) -> Path:
    """Embed every poem with the trained model and save the index."""
    from rakhshai_graph_nlp.lm.poem_recommender import build_poem_index

    # Use the line-broken display text in the index so results render nicely,
    # while embedding is driven by the same cleaned text used for training.
    index_poems = [
        {**p, "text": p.get("display") or p["text"]} for p in poems
    ]

    def _progress(done: int, total: int) -> None:
        if done == total or done % (max(1, total // 10) * args.index_batch_size) < args.index_batch_size:
            print(f"   indexing… {done:,}/{total:,}")

    print("🧠 Building the poem embedding index…")
    started = time.perf_counter()
    index_path = build_poem_index(
        output_dir,
        index_poems,
        device=args.device,
        batch_size=args.index_batch_size,
        block_size=args.block_size,
        n_components=args.pc_removal,
        progress=_progress,
    )
    elapsed = time.perf_counter() - started
    print(f"   ✓ indexed {len(index_poems):,} poems in {elapsed:,.1f}s → "
          f"{index_path.relative_to(BASE_DIR)}")
    return index_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.threads and args.threads > 0:
        import os

        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
        import torch

        torch.set_num_threads(args.threads)
        print(f"🧵 Using {args.threads} CPU thread(s) for torch.")

    output_dir = RUNS_DIR / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    poems = load_poems(args)
    if len(poems) < 8:
        print("❌ Not enough poems to train a useful model. "
              "Relax --min-chars / --poets or raise --limit.")
        return 1

    corpus = [p["text"] for p in poems]
    # Save the exact training corpus next to the checkpoint so it is always
    # clear which texts the model learned from (one poem per line).
    corpus_path = output_dir / "corpus.txt"
    corpus_path.write_text("\n".join(corpus) + "\n", encoding="utf-8")
    print(f"📄 Training corpus saved → {corpus_path.relative_to(BASE_DIR)} "
          f"({len(corpus)} poems)")

    train_model(args, output_dir, corpus)
    build_index(args, output_dir, poems)

    print("\n🎉 Done.")
    print(f"   Checkpoint + index: {output_dir.relative_to(BASE_DIR)}")
    print("   Open the UI (./run_ui.sh) → tab «🪄 شعریار» → select this model "
          "→ paste a poem → «شبیه این را پیدا کن».")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
