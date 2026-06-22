"""Graph-LM powered "find similar poems" recommender.

This module turns a trained Persian **Graph-LM** checkpoint into a semantic
poem search engine.  Each poem is embedded by running it through the full
Graph-LM (graph-fused token embeddings → Transformer hidden states), then mean
pooling the final hidden states.  Because the embeddings come from the graph
fusion path, similarity reflects the *graph-aware* representation the model
learned — not just surface word overlap.

The artefact produced is a ``poem_index.pt`` saved next to the checkpoint:

    {
        "embeddings": FloatTensor (num_poems, d_model),   # L2-normalised
        "meta": [{"poet": ..., "poem": ..., "cat": ..., "text": ...}, ...],
    }

At query time the user's poem is embedded the same way and compared to the
index by cosine similarity (a dot product on the normalised vectors).

Typical use::

    from rakhshai_graph_nlp.lm.poem_recommender import build_poem_index, PoemRecommender

    build_poem_index("runs/ui/ganjoor-sheryar", poems)        # one-off
    rec = PoemRecommender("runs/ui/ganjoor-sheryar")
    for hit in rec.search("الا یا ایها الساقی ...", top_k=5):
        print(hit["score"], hit["poet"], hit["poem"])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from .model import GraphCausalLM
from .tokenizer import PersianTokenizer

INDEX_FILENAME = "poem_index.pt"

# Keys we keep for every poem in the index. ``text`` is the searchable content;
# the rest are display/filter metadata coming straight from the Ganjoor dataset.
_META_KEYS = ("poet", "poem", "cat", "text", "id")


@dataclass
class _LoadedModel:
    """A checkpoint loaded and prepared for embedding (graph encoded once)."""

    model: GraphCausalLM
    tokenizer: PersianTokenizer
    device: torch.device
    graph_table: torch.Tensor | None
    block_size: int


def _resolve_block_size(model: GraphCausalLM, fallback: int) -> int:
    max_seq = int(getattr(model.config, "max_seq_len", fallback) or fallback)
    return max(8, min(max_seq, fallback))


def load_for_embedding(
    model_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    block_size: int = 128,
) -> _LoadedModel:
    """Load a Graph-LM checkpoint and pre-encode its graph once.

    The GNN output does not depend on the input tokens, so we run it a single
    time here to obtain a ``(vocab_size, d_model)`` graph-embedding table and
    reuse it for every poem instead of re-running the GNN per poem.
    """
    model_dir = Path(model_dir)
    torch_device = torch.device(device)
    model, tokenizer, _gen_cfg, _graph_cfg = GraphCausalLM.from_pretrained(
        model_dir, map_location=torch_device
    )
    model.to(torch_device)
    model.eval()

    graph_table: torch.Tensor | None = None
    graph_data, token_node_ids = GraphCausalLM.load_graph_artifacts(
        model_dir, map_location=torch_device
    )
    if graph_data is not None and token_node_ids is not None:
        graph_data = graph_data.to(torch_device)
        with torch.no_grad():
            # ``_precompute_graph_table`` encodes the whole graph once and maps
            # every token id to its graph embedding. ``subgraph`` is unused here
            # because the default fusion level is token-only.
            graph_table, _subgraph = model._precompute_graph_table(
                graph_data, token_node_ids.to(torch_device)
            )
    return _LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=torch_device,
        graph_table=graph_table,
        block_size=_resolve_block_size(model, block_size),
    )


@torch.no_grad()
def embed_texts(
    loaded: _LoadedModel,
    texts: Sequence[str],
    *,
    batch_size: int = 16,
    progress: callable | None = None,
) -> torch.Tensor:
    """Embed a list of texts into L2-normalised sentence vectors.

    Each text is encoded, run through the Graph-LM with the pre-computed graph
    table fused in at the token level, and the final hidden states are mean
    pooled over the non-padding positions.
    """
    model = loaded.model
    tokenizer = loaded.tokenizer
    device = loaded.device
    pad_id = int(model.config.pad_token_id)
    block_size = loaded.block_size

    vectors: list[torch.Tensor] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        encoded = []
        for text in batch:
            ids = tokenizer.encode(text or "", add_special_tokens=True)
            if not ids:
                ids = [tokenizer.bos_id, tokenizer.eos_id]
            encoded.append(ids[:block_size])
        max_len = max(len(ids) for ids in encoded)
        input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        input_ids = input_ids.to(device)

        graph_embeddings = None
        if loaded.graph_table is not None:
            graph_embeddings = F.embedding(
                input_ids.clamp_min(0), loaded.graph_table
            )
        # Pool the graph-fused *input* embeddings (token embedding + gated graph
        # fusion), NOT the Transformer's final hidden states. On a modestly
        # trained causal LM the deep hidden states are strongly anisotropic
        # (every sentence collapses to ~one direction, so cosine ≈ 1 for
        # everything); the fused input embeddings stay discriminative. Each
        # token vector is L2-normalised before pooling so frequent tokens do
        # not dominate the average.
        token_emb = model.token_embedding(input_ids)
        if graph_embeddings is not None:
            hidden = model.fusion(token_emb, graph_embeddings)
        else:
            hidden = token_emb
        hidden = F.normalize(hidden, p=2, dim=-1)
        mask = (input_ids != pad_id).unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1.0)
        pooled = summed / counts
        pooled = F.normalize(pooled, p=2, dim=-1)
        vectors.append(pooled.cpu())
        if progress is not None:
            progress(min(start + batch_size, total), total)
    if not vectors:
        return torch.empty((0, model.config.d_model))
    return torch.cat(vectors, dim=0)


def _fit_common_components(
    embeddings: torch.Tensor, n_components: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit the corpus mean and the top ``n_components`` principal directions.

    Removing these "common components" is the standard cure for embedding
    anisotropy (Arora et al. / Mu & Viswanath): the first few directions carry
    corpus-wide bias shared by every poem and crowd out the topical signal.
    """
    mean = embeddings.mean(dim=0)
    centered = embeddings - mean
    if n_components <= 0 or centered.shape[0] < 2:
        return mean, centered.new_zeros((0, centered.shape[1]))
    # Right singular vectors = principal directions of the row space.
    _u, _s, vt = torch.linalg.svd(centered, full_matrices=False)
    k = min(n_components, vt.shape[0])
    return mean, vt[:k].contiguous()


def _apply_common_components(
    embeddings: torch.Tensor, mean: torch.Tensor, components: torch.Tensor
) -> torch.Tensor:
    """Center, project out the common components, and re-normalise."""
    centered = embeddings - mean
    if components.numel():
        centered = centered - (centered @ components.t()) @ components
    return F.normalize(centered, p=2, dim=-1)


def _clean_meta(poem: dict) -> dict:
    meta = {key: poem.get(key, "") for key in _META_KEYS}
    meta["text"] = str(meta.get("text") or "").strip()
    return meta


def build_poem_index(
    model_dir: str | Path,
    poems: Sequence[dict],
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 16,
    block_size: int = 128,
    n_components: int = 1,
    progress: callable | None = None,
) -> Path:
    """Embed every poem with the checkpoint and save ``poem_index.pt``.

    Parameters
    ----------
    model_dir:
        A trained Graph-LM checkpoint directory.
    poems:
        A sequence of dicts with at least a ``text`` key (and optionally
        ``poet``, ``poem``, ``cat``, ``id``) — typically rows from the Ganjoor
        dataset.

    Returns
    -------
    Path to the written ``poem_index.pt``.
    """
    model_dir = Path(model_dir)
    cleaned = [_clean_meta(p) for p in poems if str(p.get("text") or "").strip()]
    if not cleaned:
        raise ValueError("no poems with non-empty text to index")

    loaded = load_for_embedding(model_dir, device=device, block_size=block_size)
    raw = embed_texts(
        loaded,
        [m["text"] for m in cleaned],
        batch_size=batch_size,
        progress=progress,
    )
    mean, components = _fit_common_components(raw, n_components)
    embeddings = _apply_common_components(raw, mean, components)
    index_path = model_dir / INDEX_FILENAME
    torch.save(
        {
            "embeddings": embeddings.contiguous(),
            "meta": cleaned,
            "d_model": int(loaded.model.config.d_model),
            "graph": loaded.graph_table is not None,
            "mean": mean.contiguous(),
            "components": components.contiguous(),
            "n_components": int(n_components),
        },
        index_path,
    )
    return index_path


class PoemRecommender:
    """Loads a checkpoint + its poem index and answers "find similar" queries."""

    def __init__(self, model_dir: str | Path, *, device: str | torch.device = "cpu"):
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / INDEX_FILENAME
        if not index_path.exists():
            raise FileNotFoundError(
                f"poem index not found: {index_path}. Build it with "
                "build_poem_index() (e.g. via scripts/build_ganjoor_recommender.py)."
            )
        payload = torch.load(index_path, map_location="cpu", weights_only=False)
        embeddings = payload["embeddings"]
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.as_tensor(embeddings)
        self.embeddings = embeddings.float()
        self.meta: list[dict] = list(payload.get("meta", []))
        self.has_graph: bool = bool(payload.get("graph", False))
        d_model = int(payload.get("d_model", self.embeddings.shape[1]))
        self.mean = payload.get("mean")
        if self.mean is None:
            self.mean = torch.zeros(d_model)
        self.components = payload.get("components")
        if self.components is None:
            self.components = torch.zeros((0, d_model))
        self._loaded = load_for_embedding(self.model_dir, device=device)

    @property
    def size(self) -> int:
        return int(self.embeddings.shape[0])

    def poets(self) -> list[str]:
        """Distinct poet names present in the index (for filtering)."""
        seen: list[str] = []
        for m in self.meta:
            name = str(m.get("poet") or "").strip()
            if name and name not in seen:
                seen.append(name)
        return sorted(seen)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        poet: str | None = None,
        exclude_identical: bool = True,
    ) -> list[dict]:
        """Return the ``top_k`` most similar poems to ``query``.

        Parameters
        ----------
        query:
            A poem or verse to search by (free Persian text).
        poet:
            If given, restrict results to this poet.
        exclude_identical:
            Drop a hit whose stored text is identical to the query (so pasting a
            poem that is already in the corpus does not just return itself).
        """
        query = (query or "").strip()
        if not query:
            return []
        if self.size == 0:
            return []
        query_raw = embed_texts(self._loaded, [query])  # (1, D), normalised
        query_vec = _apply_common_components(query_raw, self.mean, self.components)
        scores = (self.embeddings @ query_vec.squeeze(0)).numpy()

        order = scores.argsort()[::-1]
        normalized_query = " ".join(query.split())
        results: list[dict] = []
        for idx in order:
            meta = self.meta[int(idx)]
            if poet and str(meta.get("poet") or "").strip() != poet:
                continue
            if exclude_identical:
                stored = " ".join(str(meta.get("text") or "").split())
                if stored == normalized_query:
                    continue
            results.append({"score": float(scores[int(idx)]), **meta})
            if len(results) >= top_k:
                break
        return results


def list_poem_recommenders(runs_dir: str | Path) -> list[str]:
    """Find every checkpoint under ``runs_dir`` that has a built poem index.

    Returns paths relative to ``runs_dir``'s parent when possible, otherwise
    absolute paths, sorted for stable display.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    found: list[str] = []
    base = runs_dir.parent
    for index_path in runs_dir.rglob(INDEX_FILENAME):
        ckpt = index_path.parent
        if not (ckpt / "config.json").exists() or not (ckpt / "model.pt").exists():
            continue
        try:
            found.append(str(ckpt.relative_to(base)))
        except ValueError:
            found.append(str(ckpt))
    return sorted(found)
