"""Run a small controlled comparison of Rakhshai Persian tokenizers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rakhshai_graph_nlp.lm.model import GraphLMConfig
from rakhshai_graph_nlp.lm.trainer import LMTrainingConfig, train_graph_lm


def _load_corpus(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        corpus = [line.strip() for line in f if line.strip()]
    if not corpus:
        raise ValueError("corpus file is empty")
    return corpus


def _run_variant(
    corpus: list[str],
    *,
    output_dir: Path,
    tokenizer_type: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    run_dir = output_dir / tokenizer_type
    metrics = train_graph_lm(
        corpus,
        training_config=LMTrainingConfig(
            output_dir=str(run_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_ratio=args.validation_ratio,
            block_size=args.block_size,
            min_freq=args.min_freq,
            max_vocab_size=args.max_vocab_size,
            graph_window_size=args.graph_window_size,
            tokenizer_type=tokenizer_type,
            tokenizer_half_space=args.tokenizer_half_space,
            tokenizer_morph_splitting=args.tokenizer_morph_splitting,
            tokenizer_compound_verb_mode=args.tokenizer_compound_verb_mode,
            tokenizer_bpe_merges=args.tokenizer_bpe_merges,
            device=args.device,
            seed=args.seed,
        ),
        model_config=GraphLMConfig(
            vocab_size=1,
            max_seq_len=args.block_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dim_feedforward=args.dim_feedforward,
            graph_encoder=args.graph_encoder,
            fusion=args.fusion,
        ),
        graph_encoder=args.graph_encoder,
        fusion=args.fusion,
    )
    return {
        "tokenizer_type": tokenizer_type,
        "run_dir": str(run_dir),
        "best_validation_loss": metrics["best_validation_loss"],
        "best_perplexity": metrics["best_perplexity"],
        "tokenizer_stats": metrics.get("tokenizer_stats", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Rakhshai Persian tokenizer variants on one corpus."
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output-dir", default="runs/tokenizer-ablation")
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        default=["word", "char_chunk", "bpe"],
        choices=["word", "char_chunk", "bpe", "unigram"],
    )
    parser.add_argument("--graph-encoder", choices=["none", "gcn"], default="none")
    parser.add_argument("--fusion", choices=["gated", "context_gated", "add"], default="gated")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=None)
    parser.add_argument("--graph-window-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--dim-feedforward", type=int, default=64)
    parser.add_argument("--tokenizer-half-space", choices=["preserve", "split"], default="preserve")
    parser.add_argument("--tokenizer-morph-splitting", action="store_true")
    parser.add_argument(
        "--tokenizer-compound-verb-mode",
        choices=["none", "join"],
        default="none",
    )
    parser.add_argument("--tokenizer-bpe-merges", type=int, default=200)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    corpus = _load_corpus(Path(args.corpus))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        _run_variant(corpus, output_dir=output_dir, tokenizer_type=tokenizer_type, args=args)
        for tokenizer_type in args.tokenizers
    ]
    report = {
        "corpus": args.corpus,
        "seed": args.seed,
        "results": results,
    }
    report_path = output_dir / "tokenizer_comparison.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
