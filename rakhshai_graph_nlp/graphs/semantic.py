"""Semantic graph construction.

This module builds semantic word graphs from explicit lexical relations,
FarsNet-style exports and, optionally, pre-computed word embeddings.

Example:

    >>> from rakhshai_graph_nlp.graphs.semantic import build_semantic_graph
    >>> relations = {"گربه": ["سگ"]}
    >>> graph = build_semantic_graph(["گربه", "سگ"], relations=relations)
    >>> graph.adjacency.sum()
    2.0
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Optional

import numpy as np

from .graph import Graph


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _add_relation(
    relations: dict[str, set[str]],
    source: object,
    target: object,
) -> None:
    src = str(source).strip()
    tgt = str(target).strip()
    if not src or not tgt or src == tgt:
        return
    relations.setdefault(src, set()).add(tgt)
    relations.setdefault(tgt, set()).add(src)


def _normalise_relation_mapping(
    relations: Mapping[str, Iterable[str]],
) -> dict[str, set[str]]:
    normalised: dict[str, set[str]] = {}
    for source, targets in relations.items():
        for target in targets:
            _add_relation(normalised, source, target)
    return normalised


def _relations_from_json(data: object) -> dict[str, set[str]]:
    relations: dict[str, set[str]] = {}

    if isinstance(data, Mapping):
        if "synsets" in data:
            return _relations_from_json(data["synsets"])
        for source, targets in data.items():
            if isinstance(targets, str):
                _add_relation(relations, source, targets)
            elif isinstance(targets, Iterable):
                for target in targets:
                    _add_relation(relations, source, target)
        return relations

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, Mapping):
                continue
            words = (
                item.get("words")
                or item.get("lemmas")
                or item.get("senses")
                or item.get("synonyms")
            )
            if isinstance(words, str):
                words = [part.strip() for part in words.split(",")]
            if isinstance(words, Iterable):
                word_list = [str(word).strip() for word in words if str(word).strip()]
                for i, source in enumerate(word_list):
                    for target in word_list[i + 1 :]:
                        _add_relation(relations, source, target)

            source = item.get("source") or item.get("word") or item.get("lemma")
            targets = item.get("targets") or item.get("related") or item.get("target")
            if source is not None and targets is not None:
                if isinstance(targets, str):
                    targets = [part.strip() for part in targets.split(",")]
                if isinstance(targets, Iterable):
                    for target in targets:
                        _add_relation(relations, source, target)

    return relations


def _relations_from_table(path: Path) -> dict[str, set[str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    relations: dict[str, set[str]] = {}
    synsets: dict[str, list[str]] = {}

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("FarsNet table must include a header row")
        for row in reader:
            source = (
                row.get("source")
                or row.get("src")
                or row.get("word")
                or row.get("lemma")
            )
            target = row.get("target") or row.get("dst") or row.get("related")
            if source and target:
                _add_relation(relations, source, target)
                continue

            synset_id = row.get("synset_id") or row.get("synset") or row.get("sid")
            word = row.get("word") or row.get("lemma") or row.get("sense")
            if synset_id and word:
                synsets.setdefault(synset_id, []).append(word)

    for words in synsets.values():
        for i, source in enumerate(words):
            for target in words[i + 1 :]:
                _add_relation(relations, source, target)
    return relations


def load_farsnet_relations(path: str | Path) -> dict[str, set[str]]:
    """Load lexical relations from a FarsNet-style JSON, CSV or TSV export.

    Supported JSON shapes include:

    - ``{"ماشین": ["خودرو", "اتومبیل"]}``
    - ``[{"words": ["ماشین", "خودرو"]}]``
    - ``{"synsets": [{"lemmas": ["پزشک", "دکتر"]}]}``

    Supported CSV/TSV columns include either ``source,target`` pairs or
    ``synset_id,word`` rows that are grouped into synonym relations.
    """

    farsnet_path = Path(path)
    suffix = farsnet_path.suffix.lower()
    if suffix == ".json":
        with farsnet_path.open(encoding="utf-8") as f:
            return _relations_from_json(json.load(f))
    if suffix in {".csv", ".tsv"}:
        return _relations_from_table(farsnet_path)
    raise ValueError("FarsNet export must be a JSON, CSV or TSV file")


def build_semantic_graph(
    words: Sequence[str],
    relations: Optional[Mapping[str, Iterable[str]]] = None,
    *,
    embedding_lookup: Optional[Mapping[str, Sequence[float]]] = None,
    similarity_threshold: float = 0.7,
    top_k: Optional[int] = None,
    relation_weight: float = 1.0,
) -> Graph:
    """Construct a semantic graph from a list of words.

    Parameters
    ----------
    words : Sequence[str]
        A sequence of words for which to build a semantic graph. Each
        word becomes a node in the graph.
    relations : Mapping[str, Iterable[str]], optional
        Mapping of a word to a collection of semantically related words
        (e.g. synonyms). Only relations where both words appear in the
        ``words`` sequence are added to the graph.
    embedding_lookup : Mapping[str, Sequence[float]], optional
        Optional embedding vectors keyed by word. When supplied, cosine
        similarity edges are added for word pairs above
        ``similarity_threshold``.
    similarity_threshold : float, optional
        Minimum cosine similarity for embedding-based edges. Defaults to
        ``0.7``.
    top_k : int, optional
        Keep at most the strongest ``top_k`` embedding neighbours for each word.
    relation_weight : float, optional
        Weight assigned to explicit lexical relations. Defaults to ``1.0``.

    Returns
    -------
    Graph
        A graph with the given words as nodes. Edges represent semantic
        relations supplied via ``relations`` and/or embedding similarity.
    """
    nodes = list(words)
    n = len(nodes)
    adjacency = np.zeros((n, n), dtype=float)
    index = {w: i for i, w in enumerate(nodes)}

    if relations:
        for src, targets in relations.items():
            if src not in index:
                continue
            src_idx = index[src]
            for tgt in targets:
                if tgt not in index:
                    continue
                tgt_idx = index[tgt]
                adjacency[src_idx, tgt_idx] = max(
                    adjacency[src_idx, tgt_idx],
                    relation_weight,
                )
                adjacency[tgt_idx, src_idx] = max(
                    adjacency[tgt_idx, src_idx],
                    relation_weight,
                )

    if embedding_lookup:
        vectors: dict[int, np.ndarray] = {}
        for word, idx in index.items():
            if word in embedding_lookup:
                vectors[idx] = np.asarray(embedding_lookup[word], dtype=float)

        neighbours: dict[int, list[tuple[int, float]]] = {i: [] for i in vectors}
        vector_items = list(vectors.items())
        for pos, (i, vec_i) in enumerate(vector_items):
            for j, vec_j in vector_items[pos + 1 :]:
                score = _cosine_similarity(vec_i, vec_j)
                if score >= similarity_threshold:
                    neighbours[i].append((j, score))
                    neighbours[j].append((i, score))

        for i, candidates in neighbours.items():
            ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
            if top_k is not None:
                ranked = ranked[:top_k]
            for j, score in ranked:
                adjacency[i, j] = max(adjacency[i, j], score)
                adjacency[j, i] = max(adjacency[j, i], score)

    return Graph(nodes=nodes, adjacency=adjacency)


def build_semantic_graph_from_farsnet(
    words: Sequence[str],
    farsnet_path: str | Path,
    *,
    embedding_lookup: Optional[Mapping[str, Sequence[float]]] = None,
    similarity_threshold: float = 0.7,
    top_k: Optional[int] = None,
    relation_weight: float = 1.0,
) -> Graph:
    """Build a semantic graph using a FarsNet-style lexical export."""

    relations = load_farsnet_relations(farsnet_path)
    return build_semantic_graph(
        words,
        relations=relations,
        embedding_lookup=embedding_lookup,
        similarity_threshold=similarity_threshold,
        top_k=top_k,
        relation_weight=relation_weight,
    )
