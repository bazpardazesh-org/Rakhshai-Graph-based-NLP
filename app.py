#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graphical user interface for the Rakhshai project (Rakhshai Graph-based NLP)
============================================================================

A professional, Persian, right-to-left (RTL) user interface that brings all of
the project's capabilities together in a single window:

  • Persian tokenizer (word/subword/BPE/Unigram) with half-space and morpheme splitting
  • Building and visualizing a multi-relational text graph
  • Training a graph language model (Graph-LM) with live per-epoch progress
  • Persian text generation with "graph memory"
  • Graph-based text classification (GCN / GraphSAGE / GAT)
  • Analytical tasks: summarization, content recommendation, hate-speech detection
  • "Full power" section: a step-by-step tour that showcases the project's full capability

Run:
    python app.py                      # on http://127.0.0.1:7860
    python app.py --share              # create a temporary public link
    python app.py --port 8080          # choose a custom port

This file is written defensively: every section catches errors and shows a Persian
message so the UI never crashes.
"""
from __future__ import annotations

import argparse
import html
import io
import json
import math
import tempfile
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Ensure the package is importable from the repository root
BASE_DIR = Path(__file__).resolve().parent

import gradio as gr  # noqa: E402

# ---------------------------------------------------------------------------
# Paths and default data
# ---------------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
RUNS_DIR = BASE_DIR / "runs"
UI_RUNS_DIR = RUNS_DIR / "ui"
UI_RUNS_DIR.mkdir(parents=True, exist_ok=True)

# A few Persian sample texts to quickly fill the input boxes
SAMPLE_TEXT = (
    "هوش مصنوعی زبان فارسی را با کمک گراف بهتر می‌فهمد.\n"
    "گراف دانش روابط میان واژه‌ها را نگه می‌دارد و معنا را غنی‌تر می‌کند.\n"
    "مدل زبانی گرافی رخشای برای پردازش متن فارسی طراحی شده است.\n"
    "دانشجویان در دانشگاه تهران دربارهٔ یادگیری ماشین پژوهش می‌کنند.\n"
    "پایتخت ایران تهران است و رودخانه‌های زیادی در شمال کشور جریان دارند."
)

SAMPLE_CLASSIFY = (
    "ورزش\tتیم ملی فوتبال ایران در جام جهانی بازی کرد و به پیروزی رسید.\n"
    "ورزش\tبازیکن والیبال با یک آبشار قدرتمند امتیاز گرفت.\n"
    "ورزش\tمسابقهٔ دوومیدانی فردا در ورزشگاه برگزار می‌شود.\n"
    "فناوری\tشرکت‌های فناوری پردازندهٔ جدید هوش مصنوعی را معرفی کردند.\n"
    "فناوری\tتلفن هوشمند تازه با دوربین بهتر روانهٔ بازار شد.\n"
    "فناوری\tنرم‌افزار جدید با کمک یادگیری ماشین کار می‌کند."
)

SAMPLE_HATE = (
    "نفرت\tاین گروه باید نابود شود و جایی میان ما ندارد.\n"
    "نفرت\tاز همهٔ آن‌ها متنفرم و باید بروند.\n"
    "عادی\tامروز هوای پاییزی بسیار دلپذیر و آرام بود.\n"
    "عادی\tبا دوستانم به کتابخانه رفتیم و کتاب خواندیم.\n"
    "عادی\tغذای امشب خوشمزه بود و همه راضی بودند."
)

RELATION_FA = {
    "cooccurrence": "هم‌آیی",
    "pmi": "PMI",
    "ppmi": "PPMI",
    "dependency": "وابستگی نحوی",
    "stem": "هم‌ریشگی",
    "subword": "زیرواژه",
    "semantic_similarity": "شباهت معنایی",
    "semantic": "شباهت معنایی",
    "word_document": "واژه‑سند",
    "topic_document": "موضوع‑سند",
    "document": "سند",
    "sentence": "جمله",
}

NODE_TYPE_FA = {
    "token": "واژه",
    "document": "سند",
    "topic": "موضوع",
    "sentence": "جمله",
    "context": "بافت",
}

# Color palette for relations (edges) and node types
RELATION_COLORS = {
    "cooccurrence": "#2563eb",
    "pmi": "#16a34a",
    "ppmi": "#0d9488",
    "dependency": "#dc2626",
    "stem": "#9333ea",
    "subword": "#ea580c",
    "semantic_similarity": "#db2777",
    "word_document": "#ca8a04",
    "topic_document": "#0891b2",
    "document": "#ca8a04",
    "sentence": "#64748b",
}
NODE_TYPE_COLORS = {
    "token": "#3b82f6",
    "document": "#f59e0b",
    "topic": "#06b6d4",
    "sentence": "#94a3b8",
    "context": "#a78bfa",
}


def _err_box(message: str, detail: str = "") -> str:
    """Persian error box."""
    detail_html = (
        f"<pre style='white-space:pre-wrap;text-align:left;direction:ltr;"
        f"font-size:12px;color:#7f1d1d'>{html.escape(detail)}</pre>"
        if detail
        else ""
    )
    return (
        "<div style='background:#fef2f2;border:1px solid #fca5a5;border-radius:12px;"
        "padding:14px 16px;color:#991b1b'>"
        f"<b>⚠️ خطا:</b> {html.escape(message)}{detail_html}</div>"
    )


def _info_box(message: str, kind: str = "info") -> str:
    palette = {
        "info": ("#eff6ff", "#bfdbfe", "#1e3a8a"),
        "ok": ("#f0fdf4", "#bbf7d0", "#166534"),
        "warn": ("#fffbeb", "#fde68a", "#92400e"),
    }[kind]
    return (
        f"<div style='background:{palette[0]};border:1px solid {palette[1]};"
        f"border-radius:12px;padding:14px 16px;color:{palette[2]}'>{message}</div>"
    )


def _help_card(intro: str, items, tip: str = "") -> str:
    """A friendly, plain-language help box aimed at non-technical users.

    intro : one or two sentences on what the section is for.
    items : list of (control name, plain-language explanation) pairs. Both may
            contain light, trusted HTML (e.g. <b>) — all values below are
            literals written by us, never user input, so they are inserted raw.
    tip   : optional closing note shown in a highlighted strip.
    """
    rows = "".join(
        "<div style='display:flex;gap:12px;align-items:flex-start;padding:9px 0;"
        "border-top:1px solid #e0e7ff'>"
        f"<div style='flex:0 0 175px;color:#3730a3;font-weight:700'>{name}</div>"
        f"<div style='flex:1;min-width:0;color:#334155;line-height:1.95'>{text}</div>"
        "</div>"
        for name, text in items
    )
    tip_html = (
        "<div style='margin-top:12px;background:#eef2ff;border-radius:10px;"
        f"padding:10px 14px;color:#3730a3;line-height:1.9'>💡 {tip}</div>"
        if tip
        else ""
    )
    return (
        "<div style='background:#fbfcff;border:1px solid #e0e7ff;border-radius:14px;"
        "padding:16px 18px'>"
        f"<div style='color:#475569;line-height:1.95;margin-bottom:6px'>{intro}</div>"
        f"{rows}{tip_html}</div>"
    )


# A consistent title for every section's help accordion.
HELP_TITLE = "📘 راهنمای ساده — هر گزینه یعنی چه؟ (برای همه، بدون دانش فنی)"


# ---------------------------------------------------------------------------
# Discover available corpora and checkpoints
# ---------------------------------------------------------------------------
def list_corpora() -> list[str]:
    items: list[str] = []
    if DATA_DIR.exists():
        for p in sorted(DATA_DIR.glob("*.txt")):
            items.append(str(p.relative_to(BASE_DIR)))
    return items


def list_checkpoints() -> list[str]:
    """Any folder that is a complete Graph-LM checkpoint."""
    found: list[str] = []
    if not RUNS_DIR.exists():
        return found
    for cfg in RUNS_DIR.rglob("config.json"):
        d = cfg.parent
        if (d / "model.pt").exists() and (d / "tokenizer.json").exists():
            # Only Graph-LM checkpoints (not classifiers)
            try:
                data = json.loads(cfg.read_text(encoding="utf-8"))
            except Exception:
                continue
            if "d_model" in data and "graph_encoder" in data:
                found.append(str(d.relative_to(BASE_DIR)))
    return sorted(found)


# ---------------------------------------------------------------------------
# Compute device (CPU / GPU) discovery — exposes the project's GPU support
# ---------------------------------------------------------------------------
def detect_devices() -> tuple[list[str], str]:
    """Return (available device choices, default). 'cuda' appears only when a
    CUDA GPU is actually available; every backend safely falls back to CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return ["cuda", "cpu"], "cuda"
    except Exception:
        pass
    return ["cpu"], "cpu"


def _device_note() -> str:
    choices, default = detect_devices()
    if "cuda" in choices:
        try:
            import torch

            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA"
        return _info_box(
            f"🟢 پردازندهٔ گرافیکی شناسایی شد: <b>{html.escape(str(name))}</b>. "
            "برای آموزش/تولید سریع‌تر «cuda» را برگزینید.",
            "ok",
        )
    return _info_box(
        "ℹ️ پردازندهٔ گرافیکی (CUDA) روی این دستگاه در دسترس نیست؛ همه‌چیز روی "
        "CPU اجرا می‌شود. روی سخت‌افزار دارای GPU، گزینهٔ «cuda» به‌صورت خودکار "
        "ظاهر می‌شود.",
        "info",
    )


# ---------------------------------------------------------------------------
# 1) Persian tokenizer
# ---------------------------------------------------------------------------
def run_tokenizer(text, tokenizer_type, half_space, morph, compound, unigram_pieces):
    try:
        from rakhshai_graph_nlp.lm.tokenizer import PersianTokenizer

        text = (text or "").strip()
        if not text:
            return _err_box("لطفاً متنی برای توکن‌سازی وارد کنید."), "", ""
        lines = [ln for ln in text.splitlines() if ln.strip()] or [text]
        tok = PersianTokenizer(
            tokenizer_type=tokenizer_type,
            keep_half_space=(half_space == "preserve"),
            morph_splitting=bool(morph),
            compound_verb_mode=("join" if compound else "none"),
            unigram_num_pieces=int(unigram_pieces),
            min_freq=1,
        )
        # Training is required for unigram/bpe; harmless for the others too
        tok.fit(lines)

        normalized = tok.normalize(text)
        tokens = tok.tokenize(text)
        ids = tok.encode(text, add_special_tokens=True)

        # Colored chips for the tokens
        chips = []
        for t in tokens:
            sub = t.startswith("##")
            label = html.escape(t)
            bg = "#ede9fe" if sub else "#dbeafe"
            border = "#c4b5fd" if sub else "#93c5fd"
            chips.append(
                f"<span style='display:inline-block;margin:3px;padding:5px 10px;"
                f"background:{bg};border:1px solid {border};border-radius:10px;"
                f"font-size:14px;color:#1e293b'>{label}</span>"
            )
        chips_html = (
            "<div style='line-height:2.2'>"
            f"<div style='color:#475569;margin-bottom:8px'>شمار توکن‌ها: "
            f"<b>{len(tokens)}</b> — اندازهٔ واژگان آموخته‌شده: "
            f"<b>{tok.vocab_size}</b> — توکن زیرواژه‌ای با ## مشخص شده است.</div>"
            + "".join(chips)
            + "</div>"
        )

        normalized_html = (
            "<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;"
            f"padding:12px;font-size:15px;color:#1e293b'>{html.escape(normalized)}</div>"
        )
        ids_str = " ".join(str(i) for i in ids)
        return chips_html, normalized_html, ids_str
    except Exception:
        return _err_box("توکن‌سازی ناموفق بود.", traceback.format_exc()), "", ""


# ---------------------------------------------------------------------------
# 2) Building and visualizing the text graph (self-contained SVG, no image deps)
# ---------------------------------------------------------------------------
def _short_label(name: str) -> str:
    name = name.replace("##", "")
    for prefix in ("document:", "topic:", "sentence:", "context:"):
        if name.startswith(prefix):
            return name.split(":", 1)[0][:1].upper() + name.split(":", 1)[1][:6]
    return name[:10]


def render_graph_svg(graph, max_nodes: int = 55, max_edges: int = 170, seed: int = 7) -> str:
    try:
        import networkx as nx
    except Exception:
        return _err_box("کتابخانهٔ networkx در دسترس نیست.")

    nodes = list(graph.nodes)
    ei = graph.edge_index
    ew = graph.edge_weight
    et = graph.edge_type
    node_types = graph.node_types or (["token"] * len(nodes))
    id2rel = {int(v): str(k) for k, v in graph.graph_config.get("edge_types", {}).items()}

    n_total_edges = int(ei.shape[1]) if getattr(ei, "size", 0) else 0
    if n_total_edges == 0 or len(nodes) == 0:
        return _info_box("گرافی برای نمایش ساخته نشد (متن کوتاه یا بدون یال).", "warn")

    deg = np.zeros(len(nodes), dtype=float)
    for k in range(n_total_edges):
        s, d = int(ei[0, k]), int(ei[1, k])
        w = abs(float(ew[k]))
        deg[s] += w
        deg[d] += w

    keep = sorted(range(len(nodes)), key=lambda i: deg[i], reverse=True)[:max_nodes]
    keepset = set(keep)

    edges = []
    for k in range(n_total_edges):
        s, d = int(ei[0, k]), int(ei[1, k])
        if s in keepset and d in keepset and s != d:
            rel = id2rel.get(int(et[k]), "") if et is not None else ""
            edges.append((s, d, float(ew[k]), rel))
    # Unique edges (max weight) sorted by weight
    best: dict[tuple[int, int], tuple[float, str]] = {}
    for s, d, w, rel in edges:
        key = (s, d) if s < d else (d, s)
        if key not in best or abs(w) > abs(best[key][0]):
            best[key] = (w, rel)
    edge_list = [(k[0], k[1], v[0], v[1]) for k, v in best.items()]
    edge_list.sort(key=lambda e: abs(e[2]), reverse=True)
    edge_list = edge_list[:max_edges]

    G = nx.Graph()
    G.add_nodes_from(keep)
    for s, d, w, rel in edge_list:
        G.add_edge(s, d)

    if G.number_of_nodes() == 0:
        return _info_box("گره‌ای برای نمایش وجود ندارد.", "warn")

    pos = nx.spring_layout(G, seed=seed, k=1.4 / math.sqrt(max(1, G.number_of_nodes())), iterations=120)

    W, H, M = 900, 600, 46
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    spanx = (maxx - minx) or 1.0
    spany = (maxy - miny) or 1.0

    def sx(x):
        return M + (x - minx) / spanx * (W - 2 * M)

    def sy(y):
        return M + (y - miny) / spany * (H - 2 * M)

    kept_deg = np.array([deg[i] for i in keep], dtype=float)
    dmax = kept_deg.max() or 1.0
    wmax = max((abs(e[2]) for e in edge_list), default=1.0) or 1.0

    parts = [
        f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' "
        "style='width:100%;height:auto;background:#0b1020;border-radius:14px'>"
    ]
    # Edges
    for s, d, w, rel in edge_list:
        x1, y1 = sx(pos[s][0]), sy(pos[s][1])
        x2, y2 = sx(pos[d][0]), sy(pos[d][1])
        color = RELATION_COLORS.get(rel, "#475569")
        width = 0.6 + 2.6 * (abs(w) / wmax)
        parts.append(
            f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
            f"stroke='{color}' stroke-width='{width:.2f}' stroke-opacity='0.55'/>"
        )
    # Nodes
    for i in keep:
        x, y = sx(pos[i][0]), sy(pos[i][1])
        r = 5 + 16 * (deg[i] / dmax)
        ntype = node_types[i] if i < len(node_types) else "token"
        color = NODE_TYPE_COLORS.get(ntype, "#3b82f6")
        label = html.escape(_short_label(nodes[i]))
        parts.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r:.1f}' fill='{color}' "
            f"fill-opacity='0.92' stroke='#0b1020' stroke-width='1.5'/>"
        )
        parts.append(
            f"<text x='{x:.1f}' y='{y - r - 3:.1f}' fill='#e2e8f0' font-size='12' "
            f"font-family='Vazirmatn,Tahoma,sans-serif' text-anchor='middle'>{label}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _graph_legend(graph) -> str:
    rels = graph.graph_config.get("enabled_relations", [])
    rel_counts = graph.graph_config.get("relation_edge_counts", {})
    chips = []
    for rel in rels:
        color = RELATION_COLORS.get(rel, "#475569")
        fa = RELATION_FA.get(rel, rel)
        cnt = rel_counts.get(rel, "")
        chips.append(
            f"<span style='display:inline-flex;align-items:center;gap:6px;margin:4px 6px'>"
            f"<span style='width:18px;height:5px;border-radius:3px;background:{color};"
            f"display:inline-block'></span>{html.escape(str(fa))}"
            f"<span style='color:#94a3b8'>({cnt})</span></span>"
        )
    node_counts = graph.graph_config.get("node_type_counts", {})
    node_chips = []
    for nt, cnt in node_counts.items():
        color = NODE_TYPE_COLORS.get(nt, "#3b82f6")
        fa = NODE_TYPE_FA.get(nt, nt)
        node_chips.append(
            f"<span style='display:inline-flex;align-items:center;gap:6px;margin:4px 6px'>"
            f"<span style='width:12px;height:12px;border-radius:50%;background:{color};"
            f"display:inline-block'></span>{html.escape(str(fa))}"
            f"<span style='color:#94a3b8'>({cnt})</span></span>"
        )
    return (
        "<div style='margin:10px 0'>"
        "<div style='color:#334155;margin-bottom:4px'><b>روابط (یال‌ها):</b></div>"
        f"<div>{''.join(chips)}</div>"
        "<div style='color:#334155;margin:8px 0 4px'><b>انواع گره:</b></div>"
        f"<div>{''.join(node_chips)}</div></div>"
    )


def _graph_importance_html(graph, top_n: int = 12) -> str:
    """Honest structural node importance: weighted degree + PageRank centrality.

    This is a real interpretability view (which tokens are most central in the
    text graph) computed from the graph itself — not the dummy explainer stub.
    """
    try:
        import networkx as nx
    except Exception:
        return ""
    nodes = list(graph.nodes)
    ei = graph.edge_index
    ew = graph.edge_weight
    n_edges = int(ei.shape[1]) if getattr(ei, "size", 0) else 0
    if n_edges == 0 or not nodes:
        return ""

    deg = np.zeros(len(nodes), dtype=float)
    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    for k in range(n_edges):
        s, d = int(ei[0, k]), int(ei[1, k])
        w = abs(float(ew[k]))
        if s == d:
            continue
        deg[s] += w
        deg[d] += w
        if G.has_edge(s, d):
            G[s][d]["weight"] += w
        else:
            G.add_edge(s, d, weight=w)
    try:
        pr = nx.pagerank(G, weight="weight")
    except Exception:
        pr = {i: deg[i] for i in range(len(nodes))}

    order = sorted(range(len(nodes)), key=lambda i: (pr.get(i, 0.0), deg[i]), reverse=True)
    order = [i for i in order if deg[i] > 0][:top_n]
    if not order:
        return ""
    prmax = max((pr.get(i, 0.0) for i in order), default=1.0) or 1.0
    rows = []
    for rank, i in enumerate(order, 1):
        score = pr.get(i, 0.0) / prmax
        bar = int(round(score * 100))
        label = html.escape(_short_label(nodes[i]))
        rows.append(
            f"<tr><td style='padding:4px 8px;color:#94a3b8;width:28px'>{rank}</td>"
            f"<td style='padding:4px 8px;color:#0f172a'><b>{label}</b></td>"
            f"<td style='padding:4px 8px;width:160px'>"
            f"<div style='background:#e2e8f0;border-radius:6px;height:10px'>"
            f"<div style='background:#4f46e5;width:{bar}%;height:10px;border-radius:6px'></div>"
            f"</div></td>"
            f"<td style='padding:4px 8px;text-align:left;direction:ltr;color:#475569'>"
            f"{pr.get(i, 0.0):.4f}</td></tr>"
        )
    return (
        "<div style='margin-top:12px'>"
        "<div style='color:#334155;margin-bottom:6px'><b>🎯 مهم‌ترین واژه‌ها "
        "(مرکزیت ساختاری):</b> بر پایهٔ PageRank و درجهٔ وزنی گراف — کدام واژه‌ها "
        "نقش کانونی در متن دارند.</div>"
        "<table style='width:100%;border-collapse:collapse;font-size:13px'>"
        "<tr><th style='text-align:right;padding:4px 8px;color:#475569'>#</th>"
        "<th style='text-align:right;padding:4px 8px;color:#475569'>واژه</th>"
        "<th style='text-align:right;padding:4px 8px;color:#475569'>اهمیت</th>"
        "<th style='text-align:left;padding:4px 8px;color:#475569'>PageRank</th></tr>"
        + "".join(rows)
        + "</table></div>"
    )


def build_graph_for_text(
    text, relations, window, weighting, scope, directed, min_count,
    semantic_method="distributional", linguistic_backend="auto",
):
    try:
        from rakhshai_graph_nlp.lm.tokenizer import PersianTokenizer
        from rakhshai_graph_nlp.lm.graph_builder import build_graph_lm_graph

        text = (text or "").strip()
        if not text:
            return _err_box("لطفاً متنی وارد کنید."), "", ""
        lines = [ln for ln in text.splitlines() if ln.strip()] or [text]
        if not relations:
            relations = ["cooccurrence", "pmi", "stem", "subword"]

        tok = PersianTokenizer(tokenizer_type="word", min_freq=1).fit(lines)
        graph = build_graph_lm_graph(
            lines,
            tok,
            window_size=int(window),
            min_count=int(min_count),
            weighting=weighting,
            directed=bool(directed),
            graph_scope=scope,
            graph_relations=list(relations),
            semantic_method=semantic_method,
            linguistic_backend=linguistic_backend,
        )
        svg = render_graph_svg(graph)
        legend = _graph_legend(graph) + _graph_importance_html(graph)
        cfg = graph.graph_config
        stats = (
            "<table style='width:100%;border-collapse:collapse;font-size:14px'>"
            + "".join(
                f"<tr><td style='padding:6px 10px;color:#475569'>{html.escape(k)}</td>"
                f"<td style='padding:6px 10px;text-align:left;direction:ltr'>"
                f"<b>{html.escape(str(v))}</b></td></tr>"
                for k, v in [
                    ("شمار گره‌ها", cfg.get("num_nodes")),
                    ("شمار یال‌ها", cfg.get("num_edges")),
                    ("روابط فعال", "، ".join(RELATION_FA.get(r, r) for r in cfg.get("enabled_relations", []))),
                    ("پشتوانهٔ وابستگی", cfg.get("dependency_backend")),
                    ("وزن‌دهی", cfg.get("weighting")),
                    ("دامنه", cfg.get("graph_scope")),
                ]
            )
            + "</table>"
        )
        return svg, legend, stats
    except Exception:
        return _err_box("ساخت گراف ناموفق بود.", traceback.format_exc()), "", ""


# ---------------------------------------------------------------------------
# 3) Training Graph-LM with live progress
# ---------------------------------------------------------------------------
def _safe_load_history(state_path: Path):
    try:
        import torch

        if not state_path.exists():
            return None
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        return state.get("history")
    except Exception:
        return None


def _perplexity_sparkline(history) -> str:
    pts = [(int(r["epoch"]), float(r.get("perplexity", 0.0))) for r in history if "perplexity" in r]
    if len(pts) < 1:
        return ""
    W, H, M = 460, 150, 28
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    spanx = (xmax - xmin) or 1
    spany = (ymax - ymin) or 1

    def sx(x):
        return M + (x - xmin) / spanx * (W - 2 * M)

    def sy(y):
        return H - M - (y - ymin) / spany * (H - 2 * M)

    poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
    dots = "".join(
        f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='3' fill='#2563eb'/>" for x, y in pts
    )
    return (
        f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' style='width:100%;"
        "max-width:480px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px'>"
        f"<polyline points='{poly}' fill='none' stroke='#2563eb' stroke-width='2'/>"
        f"{dots}"
        f"<text x='{M}' y='16' font-size='12' fill='#475569' font-family='Vazirmatn'>"
        f"سرگشتگی (Perplexity) — کمتر بهتر</text>"
        f"<text x='{W - M}' y='{H - 6}' font-size='11' fill='#94a3b8' text-anchor='end' "
        f"font-family='Vazirmatn'>دوره {xmax} → {ys[-1]:.2f}</text></svg>"
    )


def _history_rows(history):
    rows = []
    for r in history:
        rows.append(
            [
                r.get("epoch"),
                round(float(r.get("train_loss", 0)), 4),
                round(float(r.get("validation_loss", 0)), 4),
                round(float(r.get("perplexity", 0)), 3),
            ]
        )
    return rows


def _fusion_summary(metrics) -> str:
    fusion = metrics.get("fusion_stats") or {}
    if not fusion:
        return ""
    interesting = {
        "token_graph_share_mean": "سهم میانگین گراف در آمیزش (توکن)",
        "token_alpha_tanh": "ضریب دروازهٔ گراف (آلفا)",
        "token_text_gate_mean": "سهم میانگین متن در آمیزش (توکن)",
    }
    items = []
    for key, fa in interesting.items():
        if key in fusion:
            items.append(
                f"<tr><td style='padding:5px 10px;color:#475569'>{fa}</td>"
                f"<td style='padding:5px 10px;text-align:left;direction:ltr'>"
                f"<b>{float(fusion[key]):.4f}</b></td></tr>"
            )
    if not items:
        return ""
    return (
        "<div style='margin-top:10px'><b>آمار آمیزش گراف‑متن:</b>"
        "<table style='width:100%;border-collapse:collapse;font-size:13px'>"
        + "".join(items)
        + "</table></div>"
    )


def train_graph_lm_ui(
    corpus_text,
    corpus_file,
    encoder,
    fusion,
    relations,
    epochs,
    d_model,
    n_layers,
    n_heads,
    block_size,
    learning_rate,
    low_data,
    multitask_all,
    run_name,
    device="cpu",
    relation_mode="embedding",
    tokenizer_type="unigram",
    semantic_method="distributional",
    linguistic_backend="auto",
    checkpoint_metric="next_token",
    progress=gr.Progress(track_tqdm=False),
):
    """Generator that streams training progress live."""
    status_id = "<div></div>"
    try:
        # Prepare the corpus
        lines: list[str] = []
        if corpus_file:
            try:
                lines = [
                    ln.strip()
                    for ln in Path(corpus_file).read_text(encoding="utf-8").splitlines()
                    if ln.strip()
                ]
            except Exception:
                lines = []
        if not lines and corpus_text and corpus_text.strip():
            lines = [ln.strip() for ln in corpus_text.splitlines() if ln.strip()]
        if not lines:
            yield _err_box("پیکره خالی است. متنی وارد کنید یا فایلی بارگذاری کنید."), [], "", "", gr.update()
            return
        if len(lines) < 3:
            yield _err_box("برای آموزش حداقل ۳ خط متن لازم است."), [], "", "", gr.update()
            return

        if d_model % n_heads != 0:
            yield _err_box("«بُعد مدل» باید بر «شمار سرها» بخش‌پذیر باشد."), [], "", "", gr.update()
            return

        name = (run_name or "graph-lm").strip().replace(" ", "-") or "graph-lm"
        output_dir = UI_RUNS_DIR / name
        state_path = output_dir / "training_state.pt"
        # Clear the previous state so live progress is accurate
        try:
            if state_path.exists():
                state_path.unlink()
        except Exception:
            pass

        if not relations:
            relations = ["cooccurrence", "pmi", "stem", "word_document", "topic_document"]
        task_losses = "all" if multitask_all else "next_token,masked_token"

        holder: dict = {}

        def worker():
            try:
                import torch  # noqa: F401
                from rakhshai_graph_nlp.lm.model import GraphLMConfig
                from rakhshai_graph_nlp.lm.trainer import LMTrainingConfig, train_graph_lm

                tcfg = LMTrainingConfig(
                    output_dir=str(output_dir),
                    epochs=int(epochs),
                    block_size=int(block_size),
                    learning_rate=float(learning_rate),
                    graph_relations=list(relations) if encoder != "none" else None,
                    graph_relation_mode=relation_mode,
                    fusion_levels="token",
                    semantic_method=semantic_method,
                    linguistic_backend=linguistic_backend,
                    checkpoint_metric=checkpoint_metric,
                    task_losses=task_losses,
                    text_augmentation=bool(low_data),
                    curriculum_learning=bool(low_data),
                    contrastive_weight=0.05 if low_data else 0.0,
                    node_dropout=0.05 if low_data else 0.0,
                    edge_dropout=0.1 if low_data else 0.0,
                    subgraph_sampling_ratio=0.9 if low_data else 1.0,
                    early_stopping_patience=0,  # so that all epochs run
                    tokenizer_type=tokenizer_type,
                    device=device,
                    seed=0,
                )
                mcfg = GraphLMConfig(
                    vocab_size=1,
                    max_seq_len=int(block_size),
                    d_model=int(d_model),
                    n_heads=int(n_heads),
                    n_layers=int(n_layers),
                    dim_feedforward=int(d_model) * 4,
                    graph_encoder=encoder,
                    fusion=fusion,
                )
                metrics = train_graph_lm(
                    lines,
                    training_config=tcfg,
                    model_config=mcfg,
                    graph_encoder=encoder,
                    fusion=fusion,
                )
                holder["metrics"] = metrics
            except Exception:
                holder["error"] = traceback.format_exc()
            finally:
                holder["done"] = True

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        total = int(epochs)
        last_len = -1
        yield (
            _info_box(
                "🚀 آموزش آغاز شد… ساخت واژگان و گراف ممکن است چند ثانیه طول بکشد "
                "(پردازش روی CPU).",
                "info",
            ),
            [],
            "",
            "",
            gr.update(),
        )
        while not holder.get("done"):
            time.sleep(0.6)
            hist = _safe_load_history(state_path)
            if hist and len(hist) != last_len:
                last_len = len(hist)
                done = len(hist)
                last = hist[-1]
                status = _info_box(
                    f"⏳ در حال آموزش… دورهٔ <b>{done}</b> از <b>{total}</b> — "
                    f"خطای اعتبارسنجی: <b>{float(last.get('validation_loss', 0)):.4f}</b> — "
                    f"سرگشتگی: <b>{float(last.get('perplexity', 0)):.3f}</b>",
                    "info",
                )
                yield status, _history_rows(hist), _perplexity_sparkline(hist), "", gr.update()

        if holder.get("error"):
            yield _err_box("آموزش با خطا متوقف شد.", holder["error"]), [], "", "", gr.update()
            return

        metrics = holder.get("metrics", {})
        # Copy the corpus for graph memory during generation
        try:
            (output_dir / "corpus.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass

        hist = metrics.get("history", [])
        best_ppl = metrics.get("best_perplexity")
        tok_stats = metrics.get("tokenizer_stats", {})
        summary = (
            _info_box(
                "✅ آموزش کامل شد و بهترین نقطه‌بازرسی ذخیره شد.", "ok"
            )
            + "<div style='margin-top:10px'><table style='width:100%;border-collapse:collapse;"
            "font-size:14px'>"
            + "".join(
                f"<tr><td style='padding:6px 10px;color:#475569'>{k}</td>"
                f"<td style='padding:6px 10px;text-align:left;direction:ltr'><b>{v}</b></td></tr>"
                for k, v in [
                    ("بهترین سرگشتگی", f"{best_ppl:.3f}" if best_ppl else "—"),
                    ("بهترین دوره", metrics.get("best_epoch")),
                    ("اندازهٔ واژگان", tok_stats.get("vocab_size")),
                    ("نرخ ناشناخته (UNK) اعتبارسنجی", f"{tok_stats.get('validation_unk_rate', 0):.3f}"),
                    ("رمزگذار گراف", encoder),
                    ("مسیر نقطه‌بازرسی", str(output_dir.relative_to(BASE_DIR))),
                ]
            )
            + "</table></div>"
            + _fusion_summary(metrics)
            + _info_box(
                "اکنون به برگهٔ «✨ تولید متن» بروید و همین نقطه‌بازرسی را انتخاب کنید.",
                "info",
            )
        )
        yield (
            summary,
            _history_rows(hist),
            _perplexity_sparkline(hist),
            "",
            gr.update(choices=list_checkpoints(), value=str(output_dir.relative_to(BASE_DIR))),
        )
    except Exception:
        yield _err_box("آموزش ناموفق بود.", traceback.format_exc()), [], "", "", gr.update()


# ---------------------------------------------------------------------------
# 4) Text generation (using the CLI path for full parity)
# ---------------------------------------------------------------------------
def generate_text_ui(
    model_dir,
    prompt,
    max_new_tokens,
    min_new_tokens,
    temperature,
    top_k,
    repetition_penalty,
    graph_memory,
    gm_top_k,
    gm_depth,
    device="cpu",
):
    try:
        from rakhshai_graph_nlp.cli import _run_generate

        if not model_dir:
            return _err_box("ابتدا یک نقطه‌بازرسی انتخاب کنید (یا یکی را آموزش دهید)."), ""
        if not (prompt or "").strip():
            return _err_box("لطفاً یک «پیشوند» (prompt) فارسی وارد کنید."), ""

        abs_dir = (BASE_DIR / model_dir).resolve()
        if not (abs_dir / "config.json").exists():
            return _err_box(f"نقطه‌بازرسی یافت نشد: {model_dir}"), ""

        report_path = abs_dir / "_ui_memory_report.json"
        args = SimpleNamespace(
            model=str(abs_dir),
            prompt=prompt.strip(),
            max_new_tokens=int(max_new_tokens),
            min_new_tokens=int(min_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            graph_memory=("on" if graph_memory else "off"),
            graph_memory_top_k=int(gm_top_k),
            graph_memory_depth=int(gm_depth),
            graph_memory_max_edges=256,
            graph_memory_min_score=0.0,
            graph_memory_relation_weights=None,
            graph_memory_report_path=str(report_path),
            device=device,
        )
        generated = _run_generate(args)

        out_html = (
            "<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;"
            "padding:16px;font-size:17px;line-height:2;color:#14532d'>"
            f"{html.escape(generated)}</div>"
        )

        report_html = ""
        if report_path.exists():
            try:
                rep = json.loads(report_path.read_text(encoding="utf-8"))
                if rep.get("enabled"):
                    rows = [
                        ("گره‌های بازیابی‌شده", rep.get("retrieved_nodes")),
                        ("یال‌های بازیابی‌شده", rep.get("retrieved_edges")),
                        ("گره‌های بذر", rep.get("seed_nodes")),
                        ("پوشش پیشوند", f"{rep.get('coverage', 0):.2f}"),
                        ("گره‌های برتر", "، ".join(map(str, rep.get("top_nodes", [])[:8]))),
                    ]
                    report_html = (
                        "<div style='margin-top:12px'><b>🧠 گزارش حافظهٔ گرافی:</b>"
                        "<table style='width:100%;border-collapse:collapse;font-size:13px'>"
                        + "".join(
                            f"<tr><td style='padding:5px 10px;color:#475569'>{k}</td>"
                            f"<td style='padding:5px 10px;text-align:left;direction:rtl'>"
                            f"<b>{html.escape(str(v))}</b></td></tr>"
                            for k, v in rows
                        )
                        + "</table></div>"
                    )
                report_path.unlink()
            except Exception:
                pass
        return out_html, report_html
    except Exception:
        return _err_box("تولید متن ناموفق بود.", traceback.format_exc()), ""


# ---------------------------------------------------------------------------
# 5) Text classification
# ---------------------------------------------------------------------------
def _parse_labeled(text):
    texts, labels = [], []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if "\t" in ln:
            label, doc = ln.split("\t", 1)
        elif "|" in ln:
            label, doc = ln.split("|", 1)
        else:
            continue
        label, doc = label.strip(), doc.strip()
        if label and doc:
            labels.append(label)
            texts.append(doc)
    return texts, labels


def train_classifier_ui(train_text, model_type, epochs, state, device="cpu"):
    try:
        from rakhshai_graph_nlp.tasks.classification import TextGraphClassifier

        texts, labels = _parse_labeled(train_text)
        if len(texts) < 2 or len(set(labels)) < 2:
            return (
                _err_box("حداقل ۲ نمونه و ۲ برچسب متفاوت لازم است. قالب هر خط: برچسب⇥متن"),
                state,
            )
        clf = TextGraphClassifier(
            model=model_type,
            num_epochs=int(epochs),
            hidden_dim=32,
            device=device,
        )
        clf.fit(texts, labels)
        report = clf.evaluate(texts, labels)
        state = {"clf": clf, "labels": sorted(set(labels))}
        msg = (
            _info_box("✅ طبقه‌بند آموزش دید.", "ok")
            + "<div style='margin-top:8px'>"
            f"دقت روی دادهٔ آموزش: <b>{report['accuracy']:.3f}</b> — "
            f"F1 ماکرو: <b>{report['macro_f1']:.3f}</b> — "
            f"برچسب‌ها: <b>{html.escape('، '.join(sorted(set(labels))))}</b></div>"
            + _info_box("اکنون در کادر پایین، متن‌های تازه را برای پیش‌بینی وارد کنید.", "info")
        )
        return msg, state
    except Exception:
        return _err_box("آموزش طبقه‌بند ناموفق بود.", traceback.format_exc()), state


def predict_classifier_ui(test_text, state):
    try:
        if not state or "clf" not in state:
            return _err_box("ابتدا طبقه‌بند را آموزش دهید.")
        docs = [ln.strip() for ln in (test_text or "").splitlines() if ln.strip()]
        if not docs:
            return _err_box("متنی برای پیش‌بینی وارد کنید (هر خط یک متن).")
        preds = state["clf"].predict(docs)
        rows = "".join(
            f"<tr><td style='padding:8px 10px;border-bottom:1px solid #e2e8f0'>{html.escape(d)}</td>"
            f"<td style='padding:8px 10px;border-bottom:1px solid #e2e8f0'>"
            f"<span style='background:#dbeafe;color:#1e3a8a;border-radius:8px;padding:3px 10px'>"
            f"{html.escape(p)}</span></td></tr>"
            for d, p in zip(docs, preds)
        )
        return (
            "<table style='width:100%;border-collapse:collapse;font-size:14px'>"
            "<tr><th style='text-align:right;padding:8px 10px;color:#475569'>متن</th>"
            "<th style='text-align:right;padding:8px 10px;color:#475569'>برچسب پیش‌بینی‌شده</th></tr>"
            + rows
            + "</table>"
        )
    except Exception:
        return _err_box("پیش‌بینی ناموفق بود.", traceback.format_exc())


# ---------------------------------------------------------------------------
# 6) Analytical tasks
# ---------------------------------------------------------------------------
def summarize_ui(text, top_k, method):
    try:
        from rakhshai_graph_nlp.tasks.summarization import gat_summarise, textrank_summarise

        text = (text or "").strip()
        if not text:
            return _err_box("متنی برای خلاصه‌سازی وارد کنید.")
        if method == "textrank":
            summary = textrank_summarise(text, top_k=int(top_k))
        else:
            summary = gat_summarise(text, top_k=int(top_k))
        if not summary.strip():
            return _info_box("جمله‌ای برای خلاصه یافت نشد (متن کوتاه است).", "warn")
        sentences = [s.strip() for s in summary.split("\n") if s.strip()]
        items = "".join(
            f"<li style='margin:6px 0;padding:8px 12px;background:#f8fafc;border-radius:8px;"
            f"color:#1e293b'>{html.escape(s)}</li>"
            for s in sentences
        )
        return f"<ul style='list-style:none;padding:0'>{items}</ul>"
    except ImportError:
        return _err_box("برای خلاصه‌سازی به scikit-learn نیاز است: pip install scikit-learn")
    except Exception:
        return _err_box("خلاصه‌سازی ناموفق بود.", traceback.format_exc())


def recommend_ui(query, docs_text, top_k):
    try:
        from rakhshai_graph_nlp.tasks.recommendation import recommend_similar

        query = (query or "").strip()
        docs = [ln.strip() for ln in (docs_text or "").splitlines() if ln.strip()]
        if not query or len(docs) < 2:
            return _err_box("یک متن پرس‌وجو و حداقل ۲ سند (هر خط یک سند) وارد کنید.")
        results = recommend_similar(query, docs, top_k=int(top_k))
        rows = "".join(
            f"<tr><td style='padding:8px 10px;border-bottom:1px solid #e2e8f0;width:90px'>"
            f"<b>{score:.3f}</b></td>"
            f"<td style='padding:8px 10px;border-bottom:1px solid #e2e8f0'>{html.escape(docs[idx])}</td></tr>"
            for idx, score in results
        )
        return (
            "<table style='width:100%;border-collapse:collapse;font-size:14px'>"
            "<tr><th style='text-align:right;padding:8px 10px;color:#475569'>شباهت</th>"
            "<th style='text-align:right;padding:8px 10px;color:#475569'>سند</th></tr>"
            + rows
            + "</table>"
        )
    except ImportError:
        return _err_box("برای پیشنهاد محتوا به scikit-learn نیاز است: pip install scikit-learn")
    except Exception:
        return _err_box("پیشنهاد محتوا ناموفق بود.", traceback.format_exc())


def hate_speech_ui(train_text, test_text, model_type, device="cpu"):
    try:
        from rakhshai_graph_nlp.tasks.hate_speech import HateSpeechDetector

        texts, labels = _parse_labeled(train_text)
        if len(texts) < 2 or len(set(labels)) < 2:
            return _err_box("حداقل ۲ نمونه با دو برچسب (نفرت/عادی) لازم است. قالب: برچسب⇥متن")
        tests = [ln.strip() for ln in (test_text or "").splitlines() if ln.strip()]
        if not tests:
            return _err_box("متنی برای آزمون وارد کنید (هر خط یک متن).")
        det = HateSpeechDetector(model=model_type, num_epochs=120, device=device)
        det.fit(texts, labels)
        flags = det.predict(tests)
        rows = ""
        for doc, is_hate in zip(tests, flags):
            badge = (
                "<span style='background:#fee2e2;color:#991b1b;border-radius:8px;padding:3px 10px'>نفرت‌پراکنی</span>"
                if is_hate
                else "<span style='background:#dcfce7;color:#166534;border-radius:8px;padding:3px 10px'>عادی</span>"
            )
            rows += (
                f"<tr><td style='padding:8px 10px;border-bottom:1px solid #e2e8f0'>{html.escape(doc)}</td>"
                f"<td style='padding:8px 10px;border-bottom:1px solid #e2e8f0'>{badge}</td></tr>"
            )
        return (
            _info_box("✅ آشکارساز آموزش دید و پیش‌بینی انجام شد.", "ok")
            + "<table style='width:100%;border-collapse:collapse;font-size:14px;margin-top:8px'>"
            "<tr><th style='text-align:right;padding:8px 10px;color:#475569'>متن</th>"
            "<th style='text-align:right;padding:8px 10px;color:#475569'>نتیجه</th></tr>"
            + rows
            + "</table>"
        )
    except Exception:
        return _err_box("تشخیص نفرت‌پراکنی ناموفق بود.", traceback.format_exc())


# ---------------------------------------------------------------------------
# 7) "Full power" — step-by-step tour
# ---------------------------------------------------------------------------
def power_build_graph(corpus_text):
    lines = [ln.strip() for ln in (corpus_text or "").splitlines() if ln.strip()]
    if len(lines) < 3:
        return _err_box("برای این تور حداقل ۳ خط متن لازم است."), "", ""
    return build_graph_for_text(
        corpus_text,
        ["cooccurrence", "pmi", "dependency", "stem", "subword", "word_document", "topic_document"],
        4,
        "distance",
        "document",
        False,
        1,
    )


def power_train(corpus_text, state, progress=gr.Progress(track_tqdm=False)):
    """Full Graph-LM training with all features enabled, for the power tour."""
    for status, rows, spark, _extra, ckpt_update in train_graph_lm_ui(
        corpus_text,
        None,
        "gcn",
        "gated",
        ["cooccurrence", "pmi", "stem", "word_document", "topic_document"],
        6,  # epochs
        128,  # d_model
        2,  # n_layers
        4,  # n_heads
        96,  # block_size
        3e-4,
        True,  # low-data engine
        True,  # multitask all
        "power-tour",
    ):
        # Save the checkpoint into state when finished
        if isinstance(ckpt_update, dict) and ckpt_update.get("value"):
            state = {"checkpoint": ckpt_update["value"]}
        yield status, rows, spark, state


def power_generate(prompt, state):
    ckpt = (state or {}).get("checkpoint")
    if not ckpt:
        ckpt_list = [c for c in list_checkpoints() if "power-tour" in c]
        ckpt = ckpt_list[0] if ckpt_list else None
    if not ckpt:
        return _err_box("ابتدا مرحلهٔ ۳ (آموزش) را کامل کنید."), ""
    return generate_text_ui(ckpt, prompt, 80, 20, 0.8, 50, 1.2, True, 32, 1)


# ---------------------------------------------------------------------------
# 8) Social network analysis (GraphSAGE embeddings + communities + influence)
# ---------------------------------------------------------------------------
SAMPLE_SOCIAL = (
    "علی مریم\n"
    "علی رضا\n"
    "مریم رضا\n"
    "رضا سارا\n"
    "سارا نیما\n"
    "نیما سارا\n"
    "حسین زهرا\n"
    "حسین کاوه\n"
    "زهرا کاوه\n"
    "کاوه لیلا\n"
    "لیلا حسین\n"
    "رضا حسین"
)

_COMMUNITY_COLORS = [
    "#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c",
    "#0891b2", "#db2777", "#ca8a04", "#0d9488", "#4f46e5",
]


def _parse_social_edges(text):
    """Parse one edge per line: 'userA userB' (optionally a trailing weight)."""
    import re as _re

    edges = []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = [p for p in _re.split(r"[\t,>\-–—]+|\s+", ln) if p]
        if len(parts) < 2:
            continue
        u, v = parts[0], parts[1]
        w = 1.0
        if len(parts) >= 3:
            try:
                w = float(parts[2])
            except ValueError:
                w = 1.0
        if u != v:
            edges.append((u, v, w))
    return edges


def _render_social_svg(names, G, communities, deg, seed: int = 7) -> str:
    import networkx as nx

    if G.number_of_nodes() == 0:
        return _info_box("شبکه‌ای برای نمایش وجود ندارد.", "warn")
    node2comm = {}
    for ci, comm in enumerate(communities):
        for n in comm:
            node2comm[n] = ci
    pos = nx.spring_layout(
        G, seed=seed, k=1.5 / math.sqrt(max(1, G.number_of_nodes())), iterations=140
    )
    W, H, M = 900, 600, 56
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    spanx = (maxx - minx) or 1.0
    spany = (maxy - miny) or 1.0

    def sx(x):
        return M + (x - minx) / spanx * (W - 2 * M)

    def sy(y):
        return M + (y - miny) / spany * (H - 2 * M)

    dmax = max(deg.values()) or 1.0
    parts = [
        f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' "
        "style='width:100%;height:auto;background:#0b1020;border-radius:14px'>"
    ]
    for u, v in G.edges():
        x1, y1 = sx(pos[u][0]), sy(pos[u][1])
        x2, y2 = sx(pos[v][0]), sy(pos[v][1])
        parts.append(
            f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
            f"stroke='#475569' stroke-width='1.4' stroke-opacity='0.5'/>"
        )
    for n in G.nodes():
        x, y = sx(pos[n][0]), sy(pos[n][1])
        r = 7 + 18 * (deg.get(n, 0) / dmax)
        color = _COMMUNITY_COLORS[node2comm.get(n, 0) % len(_COMMUNITY_COLORS)]
        label = html.escape(str(names[n])[:12])
        parts.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r:.1f}' fill='{color}' "
            f"fill-opacity='0.92' stroke='#0b1020' stroke-width='1.5'/>"
        )
        parts.append(
            f"<text x='{x:.1f}' y='{y - r - 4:.1f}' fill='#e2e8f0' font-size='13' "
            f"font-family='Vazirmatn,Tahoma,sans-serif' text-anchor='middle'>{label}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def social_analysis_ui(edges_text, num_communities):
    try:
        import networkx as nx
        from rakhshai_graph_nlp.graphs.graph import Graph
        from rakhshai_graph_nlp.tasks.social_analysis import compute_social_embeddings

        edges = _parse_social_edges(edges_text)
        if len(edges) < 2:
            return (
                _err_box("حداقل ۲ یال لازم است. هر خط یک رابطه: «کاربرА کاربرB»."),
                "",
                "",
            )
        names = sorted({u for u, _, _ in edges} | {v for _, v, _ in edges})
        idx = {name: i for i, name in enumerate(names)}
        n = len(names)
        adjacency = np.zeros((n, n), dtype=float)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, v, w in edges:
            i, j = idx[u], idx[v]
            adjacency[i, j] = adjacency[j, i] = float(w)
            G.add_edge(i, j, weight=float(w))

        graph = Graph(nodes=list(range(n)), adjacency=adjacency, directed=False)
        # Identity (one-hot) node features — standard for featureless social graphs.
        features = np.eye(n, dtype=float)
        embeddings = compute_social_embeddings(graph, features, hidden_dims=[32, 16])

        # Community detection (real, graph-structural).
        try:
            comm_sets = list(
                nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
            )
        except Exception:
            comm_sets = [set(c) for c in nx.connected_components(G)]
        k = max(1, int(num_communities))
        if len(comm_sets) > k:
            comm_sets = sorted(comm_sets, key=len, reverse=True)
            comm_sets = comm_sets[: k - 1] + [set().union(*comm_sets[k - 1 :])]
        communities = [set(c) for c in comm_sets]

        deg = {i: float(adjacency[i].sum()) for i in range(n)}
        try:
            pr = nx.pagerank(G, weight="weight")
        except Exception:
            pr = {i: deg[i] for i in range(n)}

        svg = _render_social_svg(names, G, communities, deg)

        node2comm = {nn: ci for ci, comm in enumerate(communities) for nn in comm}
        order = sorted(range(n), key=lambda i: pr.get(i, 0.0), reverse=True)
        prmax = max(pr.values()) or 1.0
        infl_rows = []
        for rank, i in enumerate(order[:10], 1):
            color = _COMMUNITY_COLORS[node2comm.get(i, 0) % len(_COMMUNITY_COLORS)]
            bar = int(round(pr.get(i, 0.0) / prmax * 100))
            infl_rows.append(
                f"<tr><td style='padding:5px 8px;color:#94a3b8'>{rank}</td>"
                f"<td style='padding:5px 8px'><span style='display:inline-block;width:10px;"
                f"height:10px;border-radius:50%;background:{color};margin-left:6px'></span>"
                f"<b>{html.escape(str(names[i]))}</b></td>"
                f"<td style='padding:5px 8px;width:150px'><div style='background:#e2e8f0;"
                f"border-radius:6px;height:9px'><div style='background:{color};width:{bar}%;"
                f"height:9px;border-radius:6px'></div></div></td>"
                f"<td style='padding:5px 8px;text-align:left;direction:ltr;color:#475569'>"
                f"{pr.get(i, 0.0):.4f}</td></tr>"
            )
        influence_html = (
            "<div class='rk-card'><b>🏅 رتبه‌بندی نفوذ (PageRank)</b>"
            "<div style='color:#64748b;font-size:13px;margin:4px 0 8px'>کاربران کانونی "
            "شبکه — رنگ هر کاربر نشان‌دهندهٔ اجتماع اوست.</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:13px'>"
            "<tr><th style='text-align:right;padding:5px 8px;color:#475569'>#</th>"
            "<th style='text-align:right;padding:5px 8px;color:#475569'>کاربر</th>"
            "<th style='text-align:right;padding:5px 8px;color:#475569'>نفوذ</th>"
            "<th style='text-align:left;padding:5px 8px;color:#475569'>PageRank</th></tr>"
            + "".join(infl_rows)
            + "</table></div>"
        )

        comm_chips = []
        for ci, comm in enumerate(communities):
            color = _COMMUNITY_COLORS[ci % len(_COMMUNITY_COLORS)]
            members = "، ".join(html.escape(str(names[m])) for m in sorted(comm))
            comm_chips.append(
                f"<div style='margin:6px 0;padding:8px 12px;border-right:5px solid {color};"
                f"background:#f8fafc;border-radius:8px'><b>اجتماع {ci + 1}</b> "
                f"<span style='color:#64748b'>({len(comm)} عضو)</span><br>"
                f"<span style='color:#1e293b'>{members}</span></div>"
            )
        comm_html = (
            "<div class='rk-card'><b>👥 اجتماع‌های شناسایی‌شده "
            "(تفکیک‌پذیری مدولار)</b>"
            f"<div style='margin-top:8px'>{''.join(comm_chips)}</div>"
            "<div style='color:#64748b;font-size:13px;margin-top:10px'>"
            "تعبیه‌های گرهی با مدل <b>GraphSAGE</b> "
            f"(ابعاد خروجی: {embeddings.shape[1]}) محاسبه شد؛ اجتماع‌ها از ساختار "
            "شبکه استخراج شدند.</div></div>"
        )
        return svg, influence_html, comm_html
    except Exception:
        return _err_box("تحلیل شبکهٔ اجتماعی ناموفق بود.", traceback.format_exc()), "", ""


# ---------------------------------------------------------------------------
# 9) Dataset pipeline — graph node-classification benchmark
# ---------------------------------------------------------------------------
DEFAULT_DATASET = "benchmarks/persian_text_classification.csv"


def dataset_pipeline_ui(
    dataset_file, dataset_path, fmt, text_col, label_col,
    model_type, epochs, hidden_dim, window_size,
    train_ratio, val_ratio, test_ratio, device,
):
    try:
        from rakhshai_graph_nlp.cli import _run_dataset_pipeline

        path = None
        if dataset_file:
            path = str(dataset_file)
        elif dataset_path and dataset_path.strip():
            cand = (BASE_DIR / dataset_path.strip()).resolve()
            path = str(cand) if cand.exists() else dataset_path.strip()
        if not path or not Path(path).exists():
            return _err_box(
                "فایل دیتاست یافت نشد. یک فایل CSV/TSV/JSONL بارگذاری کنید یا مسیر "
                f"معتبری بدهید (پیش‌فرض: {DEFAULT_DATASET})."
            )

        total = float(train_ratio) + float(val_ratio) + float(test_ratio)
        if total <= 0:
            return _err_box("مجموع نسبت‌های آموزش/اعتبارسنجی/آزمون باید مثبت باشد.")

        out_dir = UI_RUNS_DIR / "dataset-pipeline"
        args = SimpleNamespace(
            dataset=path,
            dataset_format=fmt,
            text_column=(text_col or "text").strip(),
            label_column=(label_col or "label").strip(),
            train_ratio=float(train_ratio) / total,
            val_ratio=float(val_ratio) / total,
            test_ratio=float(test_ratio) / total,
            window_size=int(window_size),
            min_count=1,
            model=model_type,
            seed=0,
            hidden_dim=int(hidden_dim),
            epochs=int(epochs),
            learning_rate=1e-3,
            weight_decay=5e-4,
            dropout=0.5,
            gat_heads=4,
            device=device,
            output_dir=str(out_dir),
            report_path=str(out_dir / "metrics.json"),
            save_model=None,
        )
        report = _run_dataset_pipeline(args)

        head_rows = [
            ("دیتاست", Path(report.get("dataset", path)).name),
            ("معماری", report.get("model")),
            ("دستگاه", report.get("device")),
            ("شمار اسناد", report.get("num_documents")),
            ("شمار گره‌ها", report.get("num_nodes")),
            ("شمار کلاس‌ها", report.get("num_classes")),
        ]
        head_html = "".join(
            f"<tr><td style='padding:6px 10px;color:#475569'>{html.escape(str(k))}</td>"
            f"<td style='padding:6px 10px;text-align:left;direction:ltr'><b>{html.escape(str(v))}</b></td></tr>"
            for k, v in head_rows
        )

        split_fa = {"train": "آموزش", "validation": "اعتبارسنجی", "test": "آزمون"}

        def _fmt(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) else "—"

        split_rows = []
        for key in ("train", "validation", "test"):
            metrics = report.get("splits", {}).get(key, {}) or {}
            acc = _fmt(metrics.get("accuracy"))
            f1 = _fmt(metrics.get("macro_f1", metrics.get("f1")))
            split_rows.append(
                f"<tr><td style='padding:6px 10px'><b>{split_fa[key]}</b></td>"
                f"<td style='padding:6px 10px;text-align:left;direction:ltr'>{acc}</td>"
                f"<td style='padding:6px 10px;text-align:left;direction:ltr'>{f1}</td></tr>"
            )

        return (
            _info_box("✅ خط لولهٔ دیتاست اجرا شد و گزارش ساخته شد.", "ok")
            + "<div class='rk-card' style='margin-top:10px'><b>پیکربندی اجرا</b>"
            "<table style='width:100%;border-collapse:collapse;font-size:14px;margin-top:6px'>"
            + head_html
            + "</table></div>"
            + "<div class='rk-card' style='margin-top:10px'><b>📊 نتایج طبقه‌بندی گره</b>"
            "<table style='width:100%;border-collapse:collapse;font-size:14px;margin-top:6px'>"
            "<tr><th style='text-align:right;padding:6px 10px;color:#475569'>بخش</th>"
            "<th style='text-align:left;padding:6px 10px;color:#475569'>دقت</th>"
            "<th style='text-align:left;padding:6px 10px;color:#475569'>F1 ماکرو</th></tr>"
            + "".join(split_rows)
            + "</table></div>"
        )
    except ImportError:
        return _err_box("برای این بخش به scikit-learn نیاز است: pip install scikit-learn")
    except Exception:
        return _err_box("اجرای خط لولهٔ دیتاست ناموفق بود.", traceback.format_exc())


# ---------------------------------------------------------------------------
# UI appearance and layout
# ---------------------------------------------------------------------------
# The Persian font Vazirmatn is loaded via <head> so it always renders
HEAD_HTML = (
    "<link rel='preconnect' href='https://fonts.googleapis.com'>"
    "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
    "<link href='https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700"
    "&display=swap' rel='stylesheet'>"
)

# On page load, the whole document is set to right-to-left with Persian language
RTL_JS = """
() => {
  document.documentElement.setAttribute('dir', 'rtl');
  document.documentElement.setAttribute('lang', 'fa');
  document.body.setAttribute('dir', 'rtl');
}
"""

_FONT_STACK = "'Vazirmatn','Vazir','IRANSans','Segoe UI',Tahoma,sans-serif"

CUSTOM_CSS = f"""
html, body, .gradio-container {{ direction: rtl !important; }}
.gradio-container, .gradio-container * {{ font-family: {_FONT_STACK} !important; }}
/* Gradio wraps every Markdown/HTML component in a <div ... dir="ltr"> and marks
   the .prose body with dir="ltr". That HTML attribute forces Persian text to
   flow left-to-right, so punctuation, dashes, numbers, parentheses and emoji
   land on the wrong side. A CSS `direction` declaration beats the dir attribute,
   so we force RTL on those wrappers — everything inside then inherits RTL. */
.gradio-container [dir="ltr"] {{ direction: rtl !important; }}
/* Right-align and RTL-flow labels, headings, paragraphs, lists and our cards. */
.gradio-container label,
.gradio-container .label-wrap,
.gradio-container h1, .gradio-container h2, .gradio-container h3,
.gradio-container h4, .gradio-container h5, .gradio-container h6,
.gradio-container p, .gradio-container li, .gradio-container blockquote,
.gradio-container .md, .gradio-container .prose,
.gradio-container .rk-hero, .gradio-container .rk-card, .gradio-container .rk-step {{
  direction: rtl !important; text-align: right;
}}
.gradio-container textarea,
.gradio-container input[type=text],
.gradio-container input[type=number] {{ direction: rtl !important; text-align: right; }}
/* Tables */
.gradio-container table {{ direction: rtl !important; }}
.gradio-container th, .gradio-container td {{ text-align: right !important; }}
/* Lists with bullets on the right side */
.gradio-container ul, .gradio-container ol {{ padding-right: 1.4em; padding-left: 0; }}
/* Code, identifiers and explicitly LTR snippets stay left-aligned and
   monospaced. Declared AFTER the RTL rules above and marked !important so they
   win for <pre>/<code>/.rk-ltr even inside now-RTL Markdown. */
.gradio-container pre, .gradio-container code,
.gradio-container .rk-ltr, .gradio-container .rk-ltr * {{
  font-family: 'JetBrains Mono','Menlo','Consolas',monospace !important;
  direction: ltr !important; text-align: left !important;
}}
.rk-hero {{
  background: linear-gradient(135deg, #0b1020 0%, #1e293b 50%, #312e81 100%);
  border-radius: 20px; padding: 30px 28px; color: #e2e8f0; margin-bottom: 6px;
  text-align: right;
}}
.rk-hero h1 {{ font-size: 30px; margin: 0 0 8px; color: #fff; }}
.rk-hero p {{ font-size: 16px; color: #cbd5e1; margin: 4px 0; line-height: 1.9; }}
.rk-badge {{ display:inline-block; background:#1d4ed8; color:#fff; border-radius:999px;
  padding:4px 14px; font-size:13px; margin: 6px 6px 0 0; }}
.rk-card {{ background:#fff; border:1px solid #e2e8f0; border-radius:16px; padding:18px;
  text-align:right; color:#1e293b; }}
.rk-card b {{ color:#0f172a; }}
.rk-step {{ background:#fafafa; border:1px solid #e5e7eb; border-right:5px solid #4f46e5;
  border-radius:12px; padding:16px 18px; margin: 4px 0; text-align: right; color:#1e293b; }}
.rk-step h3 {{ margin:0 0 6px; color:#312e81; }}
.rk-step b {{ color:#0f172a; }}
footer {{ display:none !important; }}
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Vazirmatn"), "Tahoma", "sans-serif"],
).set(
    body_background_fill="#f1f5f9",
    block_radius="14px",
    button_primary_background_fill="*primary_600",
)

REL_CHOICES = [
    ("هم‌آیی (cooccurrence)", "cooccurrence"),
    ("PMI", "pmi"),
    ("PPMI", "ppmi"),
    ("وابستگی نحوی (dependency)", "dependency"),
    ("هم‌ریشگی (stem)", "stem"),
    ("زیرواژه (subword)", "subword"),
    ("شباهت معنایی (semantic)", "semantic_similarity"),
    ("واژه‑سند (word_document)", "word_document"),
    ("موضوع‑سند (topic_document)", "topic_document"),
]


# ---------------------------------------------------------------------------
# Optional supplementary packages — status report + one-click installation
# ---------------------------------------------------------------------------
# A single source of truth for both the install-status badges and the in-UI
# install buttons. "kind": "stanza_model" marks the Stanza Persian model, which
# is fetched with a different command than a normal pip install.
OPTIONAL_PACKAGES = [
    {"name": "scikit-learn", "module": "sklearn", "pip": "scikit-learn>=1.2",
     "desc": "معیارها، خلاصه‌سازی TF-IDF و گراف سند"},
    {"name": "stanza", "module": "stanza", "pip": "stanza>=1.6",
     "desc": "پردازش پیشرفتهٔ فارسی: برچسب نقش، تکواژه‌یابی و تجزیهٔ نحوی"},
    {"name": "faiss-cpu", "module": "faiss", "pip": "faiss-cpu>=1.7.4",
     "desc": "جست‌وجوی شباهت برداری سریع"},
    {"name": "مدل فارسی Stanza (fa)", "kind": "stanza_model",
     "install_label": "python -c \"import stanza; stanza.download('fa')\"",
     "desc": "مدل زبانی فارسی که Stanza برای تجزیهٔ نحوی به آن نیاز دارد"},
]


def _pkg_installed(pkg: dict) -> bool:
    """Whether an optional package (or the Stanza fa model) is available now."""
    import importlib.util

    if pkg.get("kind") == "stanza_model":
        return (Path.home() / "stanza_resources" / "fa").is_dir()
    return importlib.util.find_spec(pkg["module"]) is not None


def _install_argv(pkg: dict) -> list[str]:
    """The command (as an argv list) that installs this package."""
    import sys

    if pkg.get("kind") == "stanza_model":
        # Stanza has no runnable "stanza.download" module; the model is fetched
        # programmatically via stanza.download('fa').
        return [sys.executable, "-c", "import stanza; stanza.download('fa')"]
    return [sys.executable, "-m", "pip", "install", pkg["pip"]]


def _install_label(pkg: dict) -> str:
    """Human-readable install command shown in the UI."""
    return pkg.get("install_label") or f'pip install "{pkg["pip"]}"'


def _code_box(text: str) -> str:
    return (
        "<div style='direction:ltr;text-align:left;background:#0b1020;color:#e2e8f0;"
        "padding:8px 10px;border-radius:8px;margin-top:6px;font-family:monospace;"
        "font-size:13px;overflow:auto'>" + html.escape(text) + "</div>"
    )


def _pkg_status_badge(pkg: dict) -> str:
    """Status line for one package row: installed, or not-installed + command."""
    if _pkg_installed(pkg):
        return "<div style='color:#16a34a;font-weight:700'>✅ نصب است</div>"
    return (
        "<div style='color:#dc2626;font-weight:700'>❌ نصب نیست</div>"
        + _code_box(_install_label(pkg))
    )


def install_package(pkg: dict):
    """Install one optional package from the UI, streaming a Persian status.

    Yields (status_html, refreshed_badge_html). Runs pip (or the Stanza model
    downloader) in a subprocess so a failed install can never crash the UI.
    """
    import importlib.util
    import subprocess

    yield (
        _info_box(
            f"⏳ در حال نصب «{html.escape(pkg['name'])}»… بسته به سرعت اینترنت "
            "ممکن است چند دقیقه طول بکشد. لطفاً صبر کنید.",
            "info",
        ),
        _pkg_status_badge(pkg),
    )

    # The Persian model needs Stanza itself first.
    if pkg.get("kind") == "stanza_model" and importlib.util.find_spec("stanza") is None:
        yield (
            _info_box(
                "ابتدا باید پکیج «stanza» را نصب کنید، سپس مدل فارسی را دریافت کنید.",
                "warn",
            ),
            _pkg_status_badge(pkg),
        )
        return

    try:
        proc = subprocess.run(
            _install_argv(pkg),
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except Exception:
        yield (
            _err_box(f"اجرای دستور نصب «{pkg['name']}» ممکن نشد.", traceback.format_exc()),
            _pkg_status_badge(pkg),
        )
        return

    importlib.invalidate_caches()
    if proc.returncode == 0 and _pkg_installed(pkg):
        status = _info_box(f"✅ «{html.escape(pkg['name'])}» با موفقیت نصب شد.", "ok")
    elif proc.returncode == 0:
        status = _info_box(
            f"دستور نصب «{html.escape(pkg['name'])}» اجرا شد، اما برای فعال شدن کامل "
            "بهتر است رابط کاربری را یک‌بار ببندید و دوباره اجرا کنید.",
            "warn",
        )
    else:
        tail = (proc.stderr or proc.stdout or "").strip()[-1600:]
        status = _err_box(
            f"نصب «{html.escape(pkg['name'])}» ناموفق بود (کد خروج {proc.returncode}).",
            tail,
        )
    yield status, _pkg_status_badge(pkg)


def install_all_optional():
    """Install every optional package at once via ``pip install -e ".[all]"``.

    Yields a list of [status_html, *badge_html] matching the dependency outputs.
    """
    import importlib
    import subprocess
    import sys

    yield [
        _info_box(
            "⏳ در حال نصب همهٔ پکیج‌های اختیاری با دستور "
            "<code dir='ltr'>pip install -e \".[all]\"</code>… "
            "این کار ممکن است چند دقیقه طول بکشد.",
            "info",
        )
    ] + [_pkg_status_badge(p) for p in OPTIONAL_PACKAGES]

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".[all]"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except Exception:
        yield [_err_box("اجرای نصب گروهی ممکن نشد.", traceback.format_exc())] + [
            _pkg_status_badge(p) for p in OPTIONAL_PACKAGES
        ]
        return

    importlib.invalidate_caches()
    if proc.returncode == 0:
        status = _info_box(
            "✅ نصب گروهی انجام شد. برای دریافت «مدل فارسی Stanza» در صورت نیاز، دکمهٔ "
            "نصب همان ردیف را بزنید. ممکن است برای فعال شدن کامل، اجرای دوبارهٔ رابط لازم باشد.",
            "ok",
        )
    else:
        tail = (proc.stderr or proc.stdout or "").strip()[-1600:]
        status = _err_box(f"نصب گروهی ناموفق بود (کد خروج {proc.returncode}).", tail)
    yield [status] + [_pkg_status_badge(p) for p in OPTIONAL_PACKAGES]


def _recheck_packages():
    """Re-read install status for every optional package (after invalidation)."""
    import importlib

    importlib.invalidate_caches()
    return [_pkg_status_badge(p) for p in OPTIONAL_PACKAGES]


def _make_installer(pkg: dict):
    """Return a true generator function bound to one package (for button.click)."""

    def _installer():
        yield from install_package(pkg)

    return _installer


# ---------------------------------------------------------------------------
# 7) Poem playground — "find similar poems" with a trained Graph-LM recommender
# ---------------------------------------------------------------------------
# A trained recommender is expensive to load (model weights + graph + index),
# so keep one in memory per (checkpoint, device) and reuse it across requests.
_RECOMMENDER_CACHE: dict = {}


def list_recommender_choices() -> list[str]:
    """Checkpoints under runs/ that carry a built poem index (poem_index.pt)."""
    try:
        from rakhshai_graph_nlp.lm.poem_recommender import list_poem_recommenders

        return list_poem_recommenders(RUNS_DIR)
    except Exception:
        return []


def _get_recommender(model_dir: str, device: str):
    """Load (and cache) a PoemRecommender for the given checkpoint."""
    from rakhshai_graph_nlp.lm.poem_recommender import PoemRecommender

    key = f"{model_dir}|{device}"
    rec = _RECOMMENDER_CACHE.get(key)
    if rec is None:
        abs_dir = (BASE_DIR / model_dir).resolve()
        rec = PoemRecommender(abs_dir, device=device)
        _RECOMMENDER_CACHE[key] = rec
    return rec


def refresh_recommenders():
    """Rescan disk for available recommenders (for the refresh button)."""
    choices = list_recommender_choices()
    value = choices[0] if choices else None
    if not choices:
        msg = _info_box(
            "هنوز هیچ مدل پیشنهادگری ساخته نشده است. یک مدل بسازید (یک‌بار، در ترمینال):"
            "<pre style='direction:ltr;text-align:left;background:#0b1020;color:#e2e8f0;"
            "padding:12px;border-radius:10px;margin-top:8px;overflow:auto'>"
            "python scripts/build_ganjoor_recommender.py --threads 6</pre>"
            "سپس روی «🔄 تازه‌سازی فهرست» بزنید.",
            "warn",
        )
    else:
        msg = _info_box(
            f"✅ {len(choices)} مدل پیشنهادگر پیدا شد. یکی را انتخاب کنید و بارگذاری کنید.",
            "ok",
        )
    return gr.update(choices=choices, value=value), msg


def _count_model_params(model) -> int:
    """Total unique parameters (tied weights, e.g. lm_head/embedding, counted once)."""
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += int(p.numel())
    return total


def load_recommender_ui(model_dir, device="cpu"):
    """Load the selected recommender and report its size / poets / graph status."""
    try:
        if not model_dir:
            return (
                _info_box(
                    "هنوز مدلی ساخته نشده است. یک‌بار در ترمینال بسازید:"
                    "<pre style='direction:ltr;text-align:left;background:#0b1020;"
                    "color:#e2e8f0;padding:12px;border-radius:10px;margin-top:8px;"
                    "overflow:auto'>python scripts/build_ganjoor_recommender.py "
                    "--threads 6</pre>سپس «🔄 تازه‌سازی فهرست مدل‌ها» را بزنید.",
                    "warn",
                ),
                gr.update(choices=["همه"], value="همه"),
            )
        rec = _get_recommender(model_dir, device)
        poets = rec.poets()
        graph_note = (
            "با گراف (Graph-LM)" if rec.has_graph else "بدون گراف (پایه)"
        )
        sample_poets = "، ".join(poets[:12]) + ("، …" if len(poets) > 12 else "")

        cfg = rec._loaded.model.config
        params = _count_model_params(rec._loaded.model)
        corpus_file = BASE_DIR / model_dir / "corpus.txt"
        # `.rk-ltr` is the project's sanctioned left-to-right class (handled in
        # CUSTOM_CSS). We must NOT use a bare dir="ltr", because CUSTOM_CSS has a
        # rule `[dir="ltr"] { direction: rtl !important }` that would flip these
        # English paths/numbers back to RTL and clip them. White chips keep them
        # readable on both light and dark themes.
        chip = (
            "class='rk-ltr' style='display:inline-block;background:#ffffff;"
            "border:1px solid #c7d2fe;padding:1px 7px;border-radius:6px;"
            "word-break:break-all;max-width:100%;vertical-align:middle'"
        )
        corpus_value = (
            f"<code {chip}>{html.escape(model_dir)}/corpus.txt</code>"
            if corpus_file.exists()
            else "<i>ذخیره نشده — متن شعرها داخل poem_index.pt است</i>"
        )
        status = (
            _info_box("✅ مدل بارگذاری شد.", "ok")
            + "<div style='margin-top:8px;color:#334155;line-height:2'>"
            f"شمار شعرهای نمایه‌شده: <b>{rec.size:,}</b> — "
            f"شمار شاعران: <b>{len(poets)}</b> — حالت: <b>{graph_note}</b>"
            + (f"<br>شاعران: {html.escape(sample_poets)}" if sample_poets else "")
            + "</div>"
            + "<div style='margin-top:10px;background:#eef2ff;border:1px solid "
            "#c7d2fe;border-radius:10px;padding:12px 14px;color:#3730a3;"
            "line-height:2.4'>"
            f"<div>📁 پوشهٔ مدل: <code {chip}>{html.escape(model_dir)}</code></div>"
            f"<div>🧮 شمار پارامترها: <code {chip}>{params:,}</code> "
            f"(~{params / 1e6:.2f} میلیون)</div>"
            f"<div>📐 ابعاد: <code {chip}>d_model={cfg.d_model}, "
            f"layers={cfg.n_layers}, vocab={cfg.vocab_size}</code></div>"
            f"<div>📄 فایل پیکرهٔ آموزش: {corpus_value}</div>"
            "</div>"
        )
        return status, gr.update(choices=["همه"] + poets, value="همه")
    except Exception:
        return (
            _err_box("بارگذاری مدل ناموفق بود.", traceback.format_exc()),
            gr.update(choices=["همه"], value="همه"),
        )


def _poem_result_card(rank: int, hit: dict) -> str:
    score = float(hit.get("score", 0.0))
    poet = html.escape(str(hit.get("poet") or "—"))
    title = html.escape(str(hit.get("poem") or ""))
    cat = html.escape(str(hit.get("cat") or ""))
    text = html.escape(str(hit.get("text") or "")).replace("\n", "<br>")
    pct = max(0, min(100, int(round(score * 100))))
    meta_bits = " · ".join(b for b in [poet, title, cat] if b and b != "—")
    return (
        "<div style='background:#fbfcff;border:1px solid #e0e7ff;border-radius:14px;"
        "padding:14px 16px;margin-bottom:10px'>"
        "<div style='display:flex;justify-content:space-between;align-items:center;"
        "gap:10px;margin-bottom:8px'>"
        f"<div style='color:#3730a3;font-weight:800'>#{rank} — {meta_bits}</div>"
        "<div style='display:flex;align-items:center;gap:8px;flex:0 0 190px'>"
        "<div style='flex:1;background:#e2e8f0;border-radius:6px;height:9px'>"
        f"<div style='background:#4f46e5;width:{pct}%;height:9px;border-radius:6px'></div>"
        "</div>"
        f"<span style='color:#475569;font-size:13px;direction:ltr'>{score:.3f}</span>"
        "</div></div>"
        f"<div style='color:#0f172a;font-size:16px;line-height:2.1'>{text}</div>"
        "</div>"
    )


def find_similar_poems_ui(model_dir, query, top_k, poet, device="cpu"):
    try:
        if not model_dir:
            return _err_box("ابتدا یک مدل پیشنهادگر انتخاب و بارگذاری کنید.")
        if not (query or "").strip():
            return _err_box("لطفاً یک شعر یا بیت فارسی برای جست‌وجو وارد کنید.")
        rec = _get_recommender(model_dir, device)
        poet_filter = None if (not poet or poet == "همه") else poet
        hits = rec.search(
            query.strip(), top_k=int(top_k), poet=poet_filter
        )
        if not hits:
            return _info_box(
                "نتیجه‌ای یافت نشد. شاید صافی شاعر را روی «همه» بگذارید یا متن دیگری "
                "امتحان کنید.",
                "warn",
            )
        header = _info_box(
            f"🔎 {len(hits)} شعرِ نزدیک به متن شما (بر پایهٔ تعبیهٔ Graph-LM). "
            "عدد «شباهت» هرچه به ۱ نزدیک‌تر، شبیه‌تر.",
            "info",
        )
        cards = "".join(_poem_result_card(i, h) for i, h in enumerate(hits, 1))
        return header + "<div style='margin-top:12px'>" + cards + "</div>"
    except Exception:
        return _err_box("جست‌وجوی شعر ناموفق بود.", traceback.format_exc())


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="رخشای — رابط کاربری گراف‑NLP فارسی") as demo:
        # ---- Header ----
        gr.HTML(
            "<div class='rk-hero'>"
            "<h1>🕸️ رخشای — پردازش زبان فارسی مبتنی بر گراف</h1>"
            "<p>یک رابط کاربری کامل و فارسی برای دیدن تمام توان پروژه: توکنایزر فارسی، "
            "گراف چندرابطه‌ای متن، مدل زبانی گرافی (Graph-LM)، تولید متن با حافظهٔ گرافی، "
            "طبقه‌بندی و وظایف تحلیلی.</p>"
            "<div>"
            "<span class='rk-badge'>توکنایزر فارسی</span>"
            "<span class='rk-badge'>گراف چندرابطه‌ای</span>"
            "<span class='rk-badge'>Graph-LM</span>"
            "<span class='rk-badge'>حافظهٔ گرافی</span>"
            "<span class='rk-badge'>طبقه‌بندی گرافی</span>"
            "<span class='rk-badge'>خلاصه‌سازی</span>"
            "<span class='rk-badge'>شبکهٔ اجتماعی</span>"
            "<span class='rk-badge'>خط لولهٔ دیتاست</span>"
            "<span class='rk-badge'>پشتیبانی GPU</span>"
            "</div></div>"
        )

        # ---- Global compute device (shared by every tab that trains/generates) ----
        _dev_choices, _dev_default = detect_devices()
        with gr.Row():
            with gr.Column(scale=1):
                device_selector = gr.Dropdown(
                    _dev_choices,
                    value=_dev_default,
                    label="دستگاه پردازش",
                    info="cuda در صورت وجود GPU؛ در غیر این صورت CPU",
                )
            with gr.Column(scale=4):
                gr.HTML(_device_note())

        with gr.Tabs():
            # ============================ Home ============================
            with gr.Tab("🏠 خانه"):
                gr.Markdown(
                    "## به رخشای خوش آمدید\n"
                    "این پروژه زبان فارسی را با کمک **گراف** می‌فهمد. روابط میان واژه‌ها "
                    "(هم‌آیی، هم‌ریشگی، شباهت معنایی، وابستگی نحوی و …) در یک گراف نگه‌داری "
                    "می‌شود و یک مدل زبانی ترنسفورمری از این گراف برای فهم بهتر متن بهره می‌برد.\n\n"
                    "### از کجا شروع کنم؟\n"
                    "۱. برگهٔ **«✂️ توکنایزر»** را امتحان کنید تا ببینید متن فارسی چگونه شکسته می‌شود.\n\n"
                    "۲. در **«🕸️ گراف متن»** گرافِ یک متن را بسازید و ببینید.\n\n"
                    "۳. در **«🎓 آموزش Graph-LM»** یک مدل کوچک آموزش دهید (پیشرفت زنده دارد).\n\n"
                    "۴. در **«✨ تولید متن»** با همان مدل، متن فارسی بسازید.\n\n"
                    "۵. در **«🧠 شبکهٔ اجتماعی»** اجتماع‌ها و کاربران بانفوذ یک شبکه را کشف کنید.\n\n"
                    "۶. در **«📊 خط لولهٔ دیتاست»** یک دیتاست برچسب‌دار را با طبقه‌بندی گرافیِ گره ارزیابی کنید.\n\n"
                    "۷. و در پایان، برگهٔ **«🚀 با تمام قدرت»** شما را گام‌به‌گام تا حداکثر "
                    "توان پروژه همراهی می‌کند.\n\n"
                    "> 💡 **دستگاه پردازش** (CPU/GPU) را از بالای صفحه انتخاب کنید؛ روی "
                    "سخت‌افزار دارای GPU، گزینهٔ «cuda» خودکار ظاهر می‌شود.\n"
                )
                gr.HTML(
                    "<div class='rk-card'><b>راه‌اندازی سریع از خط فرمان:</b>"
                    "<pre style='direction:ltr;text-align:left;background:#0b1020;color:#e2e8f0;"
                    "padding:12px;border-radius:10px;overflow:auto'>"
                    "python app.py\n"
                    "# یا با لینک عمومی:\n"
                    "python app.py --share</pre></div>"
                )

            # ======================== Prerequisites ========================
            with gr.Tab("🧩 پیش‌نیازها"):
                gr.Markdown(
                    "## نصب پیش‌نیازها\n"
                    "پیش‌نیازهای **اصلی** و خودِ رابط کاربری به‌صورت خودکار نصب می‌شوند. "
                    "پکیج‌های زیر **اختیاری** هستند و هرکدام قابلیتی تازه را فعال می‌کنند؛ "
                    "فقط آن‌هایی را که نیاز دارید نصب کنید. کنار هر پکیج یک دکمهٔ **نصب** هست؛ "
                    "روی آن بزنید تا همان پکیج مستقیماً از داخل همین صفحه نصب شود (به ترمینال "
                    "نیازی نیست).\n\n"
                    "> ⚡ **برای استفاده از حداکثر توان پروژه، بهتر است همهٔ پیش‌نیازها نصب "
                    "باشند.** ساده‌ترین راه، دکمهٔ «نصب همهٔ پکیج‌های اختیاری» در پایین است.\n\n"
                    "> 🧠 **Stanza** پس از نصب، واقعاً به‌کار می‌رود: در برگه‌های «گراف متن» و "
                    "«آموزش Graph-LM» پشتوانهٔ زبانی را روی `auto` یا `stanza` بگذارید تا "
                    "روابط وابستگی نحوی و تکواژه‌یابی از Stanza استفاده کنند (نه فقط ابتکاری)."
                )
                with gr.Row():
                    install_all_btn = gr.Button(
                        "📦 نصب همهٔ پکیج‌های اختیاری", variant="primary"
                    )
                    dep_btn = gr.Button("🔄 بررسی دوبارهٔ وضعیت", variant="secondary")
                dep_status = gr.HTML()

                dep_badges = []
                for _pkg in OPTIONAL_PACKAGES:
                    with gr.Row(equal_height=True, elem_classes="rk-card"):
                        with gr.Column(scale=5):
                            gr.HTML(
                                "<div style='font-weight:700;font-size:15px;color:#0f172a'>"
                                f"{html.escape(_pkg['name'])}</div>"
                                "<div style='color:#64748b;font-size:13px;margin-top:3px'>"
                                f"{html.escape(_pkg['desc'])}</div>"
                            )
                        with gr.Column(scale=3, min_width=170):
                            _badge = gr.HTML(_pkg_status_badge(_pkg))
                        with gr.Column(scale=2, min_width=150):
                            _ins_btn = gr.Button(
                                "📥 نصب این مورد", variant="primary", size="sm"
                            )
                    dep_badges.append(_badge)
                    _ins_btn.click(_make_installer(_pkg), None, [dep_status, _badge])

                dep_btn.click(_recheck_packages, None, dep_badges)
                install_all_btn.click(
                    install_all_optional, None, [dep_status] + dep_badges
                )

            # ========================= Tokenizer =========================
            with gr.Tab("✂️ توکنایزر فارسی"):
                gr.Markdown(
                    "توکنایزر فارسی با نرمال‌سازی حروف عربی/فارسی، مدیریت **نیم‌فاصله**، "
                    "تجزیهٔ تکواژی و چهار الگوریتم زیرواژه‌ای."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        tk_text = gr.Textbox(
                            label="متن ورودی",
                            value=SAMPLE_TEXT,
                            lines=6,
                        )
                    with gr.Column(scale=2):
                        tk_type = gr.Dropdown(
                            ["word", "char_chunk", "bpe", "unigram"],
                            value="unigram",
                            label="نوع توکنایزر",
                            info="unigram برای فارسی پیشنهاد می‌شود (نرخ ناشناختهٔ کم)",
                        )
                        tk_half = gr.Radio(
                            ["preserve", "split"],
                            value="preserve",
                            label="نیم‌فاصله",
                            info="preserve = حفظ نیم‌فاصله، split = تبدیل به فاصله",
                        )
                        tk_morph = gr.Checkbox(label="تجزیهٔ تکواژی (پیشوند/پسوند)", value=False)
                        tk_compound = gr.Checkbox(label="چسباندن افعال مرکب", value=False)
                        tk_pieces = gr.Slider(
                            200, 8000, value=2000, step=100,
                            label="شمار قطعه‌های unigram",
                        )
                        tk_btn = gr.Button("توکن‌سازی", variant="primary")
                tk_out = gr.HTML(label="توکن‌ها")
                with gr.Accordion("متن نرمال‌شده و شناسه‌ها", open=False):
                    tk_norm = gr.HTML()
                    tk_ids = gr.Textbox(label="شناسه‌های توکن (با <bos>/<eos>)", lines=2)
                with gr.Accordion(HELP_TITLE, open=False):
                    gr.HTML(_help_card(
                        "این بخش متن فارسی شما را به تکه‌های کوچک («توکن») می‌شکند تا "
                        "هوش مصنوعی بتواند آن را بفهمد. هر تکه پایین صفحه نمایش داده می‌شود.",
                        [
                            ("نوع توکنایزر",
                             "روش بریدن متن. <b>word</b> یعنی برش ساده بر اساس فاصله؛ "
                             "<b>unigram</b> و <b>bpe</b> برش هوشمند به زیرواژه‌ها هستند که برای "
                             "فارسی بهتر کار می‌کنند و با کلمات ناآشنا بهتر کنار می‌آیند. اگر "
                             "مطمئن نیستید، <b>unigram</b> را نگه دارید."),
                            ("نیم‌فاصله",
                             "فاصلهٔ کوچک داخل یک کلمه، مثل «می‌رود» یا «کتاب‌ها». "
                             "<b>preserve</b> آن را دست‌نخورده نگه می‌دارد (پیشنهادی)، "
                             "<b>split</b> آن را به فاصلهٔ معمولی تبدیل می‌کند."),
                            ("تجزیهٔ تکواژی",
                             "جدا کردن پیشوند و پسوند از ریشهٔ کلمه (مثلاً «کتاب‌ها» ← «کتاب» + "
                             "«ها»). کمک می‌کند مدل بفهمد کلماتی که ریشهٔ مشترک دارند به هم مربوط‌اند."),
                            ("چسباندن افعال مرکب",
                             "فعل‌های دوتکه‌ای مثل «کار کردن» را به‌صورت یک واحد معنایی نگه می‌دارد "
                             "تا معنایشان گم نشود."),
                            ("شمار قطعه‌های unigram",
                             "چند تکهٔ متفاوت توکنایزر یاد بگیرد. عدد بزرگ‌تر = ریزبینی بیشتر ولی "
                             "کندتر؛ برای متن‌های کوتاه عدد کوچک کافی است."),
                        ],
                        "تکه‌هایی که با ## شروع می‌شوند، ادامهٔ یک کلمهٔ بزرگ‌ترند (نه یک کلمهٔ مستقل).",
                    ))
                tk_btn.click(
                    run_tokenizer,
                    [tk_text, tk_type, tk_half, tk_morph, tk_compound, tk_pieces],
                    [tk_out, tk_norm, tk_ids],
                )

            # ========================= Text graph =========================
            with gr.Tab("🕸️ گراف متن"):
                gr.Markdown(
                    "گراف چندرابطه‌ای متن را بسازید و ببینید. هر رنگ یال یک **نوع رابطه** و "
                    "هر رنگ گره یک **نوع گره** (واژه/سند/موضوع) است. اندازهٔ گره با درجهٔ آن "
                    "متناسب است."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        g_text = gr.Textbox(label="متن ورودی (هر خط یک سند)", value=SAMPLE_TEXT, lines=7)
                    with gr.Column(scale=2):
                        g_rel = gr.CheckboxGroup(
                            REL_CHOICES,
                            value=["cooccurrence", "pmi", "stem", "subword", "word_document"],
                            label="روابط",
                        )
                        g_window = gr.Slider(2, 10, value=4, step=1, label="اندازهٔ پنجرهٔ هم‌آیی")
                        g_weight = gr.Dropdown(
                            ["distance", "count", "pmi", "ppmi"], value="distance", label="وزن‌دهی"
                        )
                        g_scope = gr.Dropdown(
                            ["document", "sentence", "corpus"], value="document", label="دامنهٔ گراف"
                        )
                        g_directed = gr.Checkbox(label="گراف جهت‌دار", value=False)
                        g_mincount = gr.Slider(1, 5, value=1, step=1, label="کمینهٔ بسامد واژه")
                        g_semantic = gr.Dropdown(
                            ["distributional", "orthographic"],
                            value="distributional",
                            label="روش شباهت معنایی",
                            info="distributional = معنایی واقعی (PPMI)، orthographic = شباهت نگارشی",
                        )
                        g_ling = gr.Dropdown(
                            ["auto", "stanza", "heuristic"],
                            value="auto",
                            label="پشتوانهٔ زبانی (وابستگی/تکواژه)",
                            info="auto = Stanza در صورت نصب، وگرنه ابتکاری",
                        )
                        g_btn = gr.Button("ساخت گراف", variant="primary")
                g_svg = gr.HTML(label="نمای گراف")
                with gr.Row():
                    g_legend = gr.HTML()
                    g_stats = gr.HTML()
                with gr.Accordion(HELP_TITLE, open=False):
                    gr.HTML(_help_card(
                        "این بخش یک «گراف» می‌سازد: نقشه‌ای از کلمات (گره‌ها) و پیوندهای میان "
                        "آن‌ها (یال‌ها) که نشان می‌دهد کلمات متن شما چطور به هم مربوط‌اند.",
                        [
                            ("روابط",
                             "نوع پیوندهایی که بین کلمات کشیده می‌شود. مثلاً <b>هم‌آیی</b> یعنی دو "
                             "کلمه کنار هم آمده‌اند، <b>هم‌ریشگی</b> یعنی ریشهٔ مشترک دارند، "
                             "<b>شباهت معنایی</b> یعنی معنای نزدیک دارند. هر چه روابط بیشتری "
                             "انتخاب کنید، گراف غنی‌تر می‌شود."),
                            ("اندازهٔ پنجرهٔ هم‌آیی",
                             "چند کلمهٔ کنار هم «همسایه» حساب شوند. عدد بزرگ‌تر یعنی پیوندهای دورتر "
                             "هم دیده می‌شوند."),
                            ("وزن‌دهی",
                             "قدرت هر پیوند چطور حساب شود؛ مثلاً بر اساس فاصله یا شمار تکرار. روی "
                             "ضخامت خط‌های گراف اثر می‌گذارد."),
                            ("دامنهٔ گراف",
                             "گراف برای کل متن یک‌جا (<b>corpus</b>)، برای هر سند جدا "
                             "(<b>document</b>) یا برای هر جمله (<b>sentence</b>) ساخته شود."),
                            ("گراف جهت‌دار",
                             "اگر روشن باشد، ترتیب کلمات در پیوند مهم می‌شود («الف ← ب» با «ب ← الف» "
                             "فرق دارد). برای شروع، خاموش بماند ساده‌تر است."),
                            ("کمینهٔ بسامد واژه",
                             "کلماتی که کمتر از این تعداد در متن آمده‌اند نادیده گرفته می‌شوند تا "
                             "گراف شلوغ نشود."),
                            ("روش شباهت معنایی",
                             "<b>distributional</b> معنا را از روی همراهی کلمات حساب می‌کند "
                             "(دقیق‌تر)، <b>orthographic</b> فقط شباهت ظاهری حروف را می‌بیند."),
                            ("پشتوانهٔ زبانی",
                             "ابزار تحلیل دستور زبان فارسی. <b>auto</b> اگر Stanza نصب باشد از آن "
                             "استفاده می‌کند، وگرنه روش سادهٔ داخلی به‌کار می‌رود."),
                        ],
                        "اندازهٔ هر دایره در گراف نشان می‌دهد آن کلمه چقدر در متن نقش کانونی دارد؛ "
                        "پایین هم راهنمای رنگ‌ها و فهرست مهم‌ترین کلمات را می‌بینید.",
                    ))
                g_btn.click(
                    build_graph_for_text,
                    [g_text, g_rel, g_window, g_weight, g_scope, g_directed, g_mincount,
                     g_semantic, g_ling],
                    [g_svg, g_legend, g_stats],
                )

            # ======================= Graph-LM training =======================
            with gr.Tab("🎓 آموزش Graph-LM"):
                gr.Markdown(
                    "یک مدل زبانی گرافی کوچک آموزش دهید. پیشرفت **هر دوره به‌صورت زنده** "
                    "نمایش داده می‌شود. (اجرا روی CPU؛ برای دموی سریع، پیکرهٔ کوچک و چند دوره کافی است.)"
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        tr_text = gr.Textbox(
                            label="پیکرهٔ آموزش (هر خط یک نمونه)", value=SAMPLE_TEXT, lines=8
                        )
                        tr_file = gr.File(
                            label="یا یک فایل متنی بارگذاری کنید (.txt)", file_types=[".txt"], type="filepath"
                        )
                    with gr.Column(scale=2):
                        tr_encoder = gr.Dropdown(
                            ["gcn", "gat", "graphsage", "rgcn", "none"],
                            value="gcn",
                            label="رمزگذار گراف",
                            info="none = مدل پایه بدون گراف",
                        )
                        tr_fusion = gr.Dropdown(
                            ["gated", "context_gated", "add"], value="gated", label="نوع آمیزش گراف‑متن"
                        )
                        tr_rel = gr.CheckboxGroup(
                            REL_CHOICES,
                            value=["cooccurrence", "pmi", "stem", "word_document", "topic_document"],
                            label="روابط گراف",
                        )
                        tr_epochs = gr.Slider(1, 20, value=5, step=1, label="شمار دوره‌ها")
                        with gr.Accordion("تنظیمات پیشرفته", open=False):
                            tr_dmodel = gr.Slider(64, 256, value=128, step=64, label="بُعد مدل (d_model)")
                            tr_layers = gr.Slider(1, 4, value=2, step=1, label="شمار لایه‌ها")
                            tr_heads = gr.Slider(2, 8, value=4, step=2, label="شمار سرها")
                            tr_block = gr.Slider(32, 256, value=96, step=32, label="طول بلوک")
                            tr_lr = gr.Number(value=3e-4, label="نرخ یادگیری")
                            tr_lowdata = gr.Checkbox(
                                label="موتور آموزش کم‌داده (افزایش داده، حذف یال، …)", value=True
                            )
                            tr_multi = gr.Checkbox(label="همهٔ زیان‌های چندوظیفه‌ای", value=True)
                            tr_relmode = gr.Dropdown(
                                ["embedding", "bias", "rgcn"],
                                value="embedding",
                                label="حالت رابطه (relation mode)",
                                info="embedding = بردار به‌ازای هر رابطه (پیشنهادی)",
                            )
                            tr_tok = gr.Dropdown(
                                ["unigram", "word", "subword", "bpe", "char_chunk"],
                                value="unigram",
                                label="نوع توکنایزر",
                            )
                            tr_semantic = gr.Dropdown(
                                ["distributional", "orthographic"],
                                value="distributional",
                                label="روش شباهت معنایی",
                            )
                            tr_ling = gr.Dropdown(
                                ["auto", "stanza", "heuristic"],
                                value="auto",
                                label="پشتوانهٔ زبانی (وابستگی/تکواژه)",
                            )
                            tr_ckpt = gr.Dropdown(
                                ["next_token", "total"],
                                value="next_token",
                                label="معیار انتخاب نقطه‌بازرسی",
                            )
                            tr_name = gr.Textbox(label="نام اجرا", value="my-graph-lm")
                        tr_btn = gr.Button("شروع آموزش 🚀", variant="primary")
                tr_status = gr.HTML()
                tr_hist = gr.Dataframe(
                    headers=["دوره", "خطای آموزش", "خطای اعتبارسنجی", "سرگشتگی"],
                    label="تاریخچهٔ آموزش",
                    interactive=False,
                    row_count=(0, "dynamic"),
                )
                tr_spark = gr.HTML()
                tr_hidden = gr.HTML(visible=False)
                with gr.Accordion(HELP_TITLE, open=False):
                    gr.HTML(_help_card(
                        "این بخش یک مدل هوش مصنوعی کوچک را روی متن شما «آموزش» می‌دهد تا الگوهای "
                        "زبان فارسی را یاد بگیرد. آموزش روی CPU انجام می‌شود و پیشرفت هر دوره زنده "
                        "نمایش داده می‌شود. اگر تازه‌کارید، فقط متن را عوض کنید و دکمهٔ آموزش را "
                        "بزنید؛ بقیهٔ مقادیر پیش‌فرض معمولاً خوب‌اند.",
                        [
                            ("رمزگذار گراف",
                             "موتوری که دانش گراف را می‌خواند (<b>gcn</b>، <b>gat</b>، …). گزینهٔ "
                             "<b>none</b> یعنی مدل بدون کمک گراف آموزش ببیند — برای مقایسه مفید است."),
                            ("نوع آمیزش گراف‑متن",
                             "روش ترکیب دانش گراف با خودِ متن. <b>gated</b> به مدل اجازه می‌دهد خودش "
                             "تصمیم بگیرد چقدر به گراف تکیه کند (پیشنهادی)."),
                            ("روابط گراف",
                             "همان انواع پیوند بخش «گراف متن» که مدل در آموزش از آن‌ها بهره می‌برد."),
                            ("شمار دوره‌ها (epochs)",
                             "چند بار مدل کل متن را مرور و تمرین کند. بیشتر = یادگیری عمیق‌تر تا یک "
                             "حدی، اما کندتر."),
                            ("بُعد مدل (d_model)",
                             "اندازهٔ «مغز» مدل. بزرگ‌تر = توانمندتر ولی سنگین‌تر و کندتر. باید بر "
                             "«شمار سرها» بخش‌پذیر باشد."),
                            ("شمار لایه‌ها",
                             "عمق مدل. لایهٔ بیشتر = توان فهم بیشتر، اما نیاز به دادهٔ بیشتر و زمان "
                             "بیشتر."),
                            ("شمار سرها",
                             "چند «زاویهٔ نگاه» هم‌زمان مدل به متن داشته باشد؛ کمک می‌کند روابط "
                             "گوناگون را موازی ببیند."),
                            ("طول بلوک",
                             "حداکثر تعداد توکنی که مدل یک‌جا می‌بیند (اندازهٔ حافظهٔ کوتاه‌مدت مدل)."),
                            ("نرخ یادگیری",
                             "سرعت یادگیری مدل. خیلی زیاد = ناپایدار و پرشی، خیلی کم = بسیار کند. "
                             "مقدار پیش‌فرض معمولاً مناسب است."),
                            ("موتور آموزش کم‌داده",
                             "مجموعه‌ای از ترفندها (افزایش مصنوعی داده، حذف بخشی از گراف و …) که وقتی "
                             "متن کم است به یادگیری بهتر کمک می‌کند."),
                            ("همهٔ زیان‌های چندوظیفه‌ای",
                             "مدل هم‌زمان چند مهارت را تمرین می‌کند (مثل حدس کلمهٔ بعدی و کلمهٔ "
                             "پنهان‌شده) تا قوی‌تر شود."),
                            ("حالت رابطه",
                             "روش نمایش انواع پیوند برای مدل. <b>embedding</b> برای هر نوع رابطه یک "
                             "بردار جدا می‌سازد (پیشنهادی)."),
                            ("نوع توکنایزر",
                             "همان روش بریدن متن به تکه‌ها که در بخش «توکنایزر» توضیح داده شد."),
                            ("روش شباهت معنایی / پشتوانهٔ زبانی",
                             "مثل بخش «گراف متن»: نحوهٔ سنجش نزدیکی معنایی کلمات و ابزار تحلیل دستور "
                             "زبان فارسی."),
                            ("معیار انتخاب نقطه‌بازرسی",
                             "بر چه اساسی بهترین نسخهٔ مدل ذخیره شود. <b>next_token</b> یعنی بر پایهٔ "
                             "مهارت حدس کلمهٔ بعدی."),
                            ("نام اجرا",
                             "نامی برای ذخیرهٔ این مدل تا بعداً در بخش «تولید متن» همین مدل را پیدا "
                             "کنید."),
                        ],
                        "نمودار «سرگشتگی» (Perplexity) پایین صفحه هرچه پایین‌تر برود، یعنی مدل بهتر "
                        "یاد گرفته است. پس از پایان آموزش، به برگهٔ «✨ تولید متن» بروید.",
                    ))

            # ========================= Text generation =========================
            with gr.Tab("✨ تولید متن"):
                gr.Markdown(
                    "با یک نقطه‌بازرسی آموخته‌شده، متن فارسی بسازید. با روشن بودن **حافظهٔ "
                    "گرافی**، مدل ابتدا گره‌های مرتبط با پیشوند شما را از گراف بازیابی می‌کند."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        gen_model = gr.Dropdown(
                            choices=list_checkpoints(),
                            value=(list_checkpoints()[0] if list_checkpoints() else None),
                            label="نقطه‌بازرسی مدل",
                        )
                        gen_refresh = gr.Button("🔄 به‌روزرسانی فهرست")
                        gen_prompt = gr.Textbox(label="پیشوند (Prompt)", value="امروز در تهران", lines=2)
                        gen_btn = gr.Button("تولید متن ✨", variant="primary")
                    with gr.Column(scale=2):
                        gen_max = gr.Slider(10, 200, value=80, step=10, label="حداکثر توکن جدید")
                        gen_min = gr.Slider(0, 60, value=20, step=5, label="حداقل توکن جدید")
                        gen_temp = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="دما (Temperature)")
                        gen_topk = gr.Slider(1, 100, value=50, step=1, label="Top-k")
                        gen_rep = gr.Slider(1.0, 2.0, value=1.2, step=0.05, label="جریمهٔ تکرار")
                        gen_mem = gr.Checkbox(label="🧠 حافظهٔ گرافی", value=True)
                        gen_mem_topk = gr.Slider(4, 64, value=32, step=4, label="گره‌های برتر حافظه")
                        gen_mem_depth = gr.Slider(0, 3, value=1, step=1, label="عمق گسترش حافظه")
                gen_out = gr.HTML(label="متن تولیدشده")
                gen_report = gr.HTML()
                with gr.Accordion(HELP_TITLE, open=False):
                    gr.HTML(_help_card(
                        "این بخش با مدلی که آموزش دادید، متن فارسی تازه می‌سازد: شما شروع جمله را "
                        "می‌نویسید و مدل آن را ادامه می‌دهد.",
                        [
                            ("نقطه‌بازرسی مدل",
                             "کدام مدلِ آموخته‌شده استفاده شود. اگر تازه مدلی ساخته‌اید، با دکمهٔ "
                             "«🔄 به‌روزرسانی فهرست» آن را پیدا کنید."),
                            ("پیشوند (Prompt)",
                             "شروع متن که شما می‌نویسید؛ مدل از همین‌جا ادامه می‌دهد. هر چه روشن‌تر "
                             "باشد، نتیجه مرتبط‌تر است."),
                            ("حداکثر / حداقل توکن جدید",
                             "سقف و کفِ طول متن خروجی — یعنی مدل دست‌کم و حداکثر چند تکه (تقریباً "
                             "کلمه) تولید کند."),
                            ("دما (Temperature)",
                             "میزان خلاقیت مدل. عدد کم = محتاط و قابل‌پیش‌بینی، عدد زیاد = خلاق‌تر اما "
                             "گاهی پراکنده و بی‌ربط."),
                            ("Top-k",
                             "مدل کلمهٔ بعدی را فقط از میان k گزینهٔ محتمل برمی‌گزیند. کمتر = "
                             "محتاط‌تر، بیشتر = متنوع‌تر."),
                            ("جریمهٔ تکرار",
                             "جلوی تکرار خسته‌کنندهٔ کلمات را می‌گیرد. عدد بزرگ‌تر = تکرار کمتر."),
                            ("🧠 حافظهٔ گرافی",
                             "اگر روشن باشد، مدل پیش از نوشتن، کلمات مرتبط با پیشوند شما را از گراف "
                             "یادآوری می‌کند تا متن منسجم‌تر شود."),
                            ("گره‌های برتر حافظه / عمق گسترش حافظه",
                             "چند کلمهٔ مرتبط از گراف یادآوری شود، و جست‌وجو تا چند قدم دورتر ادامه "
                             "یابد."),
                        ],
                        "اگر متن خروجی خیلی تکراری شد، «دما» یا «جریمهٔ تکرار» را کمی بالا ببرید؛ اگر "
                        "بی‌ربط شد، «دما» را کم کنید.",
                    ))

                gen_refresh.click(
                    lambda: gr.update(choices=list_checkpoints()), None, gen_model
                )
                gen_btn.click(
                    generate_text_ui,
                    [
                        gen_model, gen_prompt, gen_max, gen_min, gen_temp, gen_topk,
                        gen_rep, gen_mem, gen_mem_topk, gen_mem_depth, device_selector,
                    ],
                    [gen_out, gen_report],
                )

                # Wire the training output into this tab after gen_model is defined
                tr_btn.click(
                    train_graph_lm_ui,
                    [
                        tr_text, tr_file, tr_encoder, tr_fusion, tr_rel, tr_epochs,
                        tr_dmodel, tr_layers, tr_heads, tr_block, tr_lr,
                        tr_lowdata, tr_multi, tr_name, device_selector,
                        tr_relmode, tr_tok, tr_semantic, tr_ling, tr_ckpt,
                    ],
                    [tr_status, tr_hist, tr_spark, tr_hidden, gen_model],
                )

            # ========================= Classification =========================
            with gr.Tab("🗂️ طبقه‌بندی متن"):
                gr.Markdown(
                    "طبقه‌بند مبتنی بر گراف (TextGCN). هر خطِ آموزش: **برچسب⇥متن** (با Tab یا |)."
                )
                cls_state = gr.State({})
                with gr.Row():
                    with gr.Column():
                        cls_train = gr.Textbox(
                            label="دادهٔ آموزش (برچسب⇥متن)", value=SAMPLE_CLASSIFY, lines=8
                        )
                        with gr.Row():
                            cls_model = gr.Dropdown(
                                ["gcn", "graphsage", "gat"], value="gcn", label="معماری"
                            )
                            cls_epochs = gr.Slider(50, 400, value=150, step=50, label="دوره‌ها")
                        cls_train_btn = gr.Button("آموزش طبقه‌بند", variant="primary")
                        cls_train_out = gr.HTML()
                    with gr.Column():
                        cls_test = gr.Textbox(
                            label="متن‌های تازه برای پیش‌بینی (هر خط یک متن)",
                            value="بازیکن فوتبال گل زد.\nرایانهٔ جدید با هوش مصنوعی کار می‌کند.",
                            lines=6,
                        )
                        cls_pred_btn = gr.Button("پیش‌بینی برچسب")
                        cls_pred_out = gr.HTML()
                with gr.Accordion(HELP_TITLE, open=False):
                    gr.HTML(_help_card(
                        "این بخش یاد می‌گیرد متن‌ها را در دسته‌های دلخواه شما (مثل «ورزش» یا "
                        "«فناوری») جای دهد. اول با چند نمونهٔ برچسب‌دار آموزش می‌بیند، بعد متن‌های "
                        "تازه را دسته‌بندی می‌کند.",
                        [
                            ("دادهٔ آموزش (برچسب⇥متن)",
                             "نمونه‌هایی که به مدل یاد می‌دهید. قالب هر خط: ابتدا برچسب، یک Tab، سپس "
                             "متن — مثل «ورزش⇥تیم ملی بازی کرد». دست‌کم ۲ دسته و چند نمونه لازم است."),
                            ("معماری",
                             "نوع شبکهٔ گرافی که برای یادگیری به‌کار می‌رود (<b>gcn</b>، "
                             "<b>graphsage</b>، <b>gat</b>). اگر مطمئن نیستید، <b>gcn</b> انتخاب "
                             "خوبی است."),
                            ("دوره‌ها",
                             "چند بار مدل روی نمونه‌ها تمرین کند. بیشتر = یادگیری دقیق‌تر تا یک حدی، "
                             "اما کندتر."),
                            ("متن‌های تازه برای پیش‌بینی",
                             "متن‌هایی که می‌خواهید مدل برچسبشان را حدس بزند؛ هر خط یک متن."),
                        ],
                        "برای نتیجهٔ بهتر، برای هر دسته چند نمونهٔ متنوع بدهید تا مدل تفاوت‌ها را خوب "
                        "یاد بگیرد.",
                    ))
                cls_train_btn.click(
                    train_classifier_ui,
                    [cls_train, cls_model, cls_epochs, cls_state, device_selector],
                    [cls_train_out, cls_state],
                )
                cls_pred_btn.click(predict_classifier_ui, [cls_test, cls_state], cls_pred_out)

            # ========================= Analytical tasks =========================
            with gr.Tab("🧰 وظایف تحلیلی"):
                with gr.Tabs():
                    with gr.Tab("📝 خلاصه‌سازی"):
                        sm_text = gr.Textbox(
                            label="متن", lines=8,
                            value=(
                                "هوش مصنوعی شاخه‌ای از علوم رایانه است. این فناوری به ماشین‌ها "
                                "توان یادگیری می‌دهد. یادگیری ماشین زیرشاخهٔ مهم آن است. "
                                "شبکه‌های عصبی الهام‌گرفته از مغز انسان هستند. گراف‌ها روابط "
                                "میان داده‌ها را نشان می‌دهند. پردازش زبان طبیعی به فهم متن کمک می‌کند."
                            ),
                        )
                        with gr.Row():
                            sm_method = gr.Radio(["textrank", "gat"], value="textrank", label="روش")
                            sm_topk = gr.Slider(1, 6, value=3, step=1, label="شمار جمله‌ها")
                            sm_btn = gr.Button("خلاصه کن", variant="primary")
                        sm_out = gr.HTML()
                        sm_btn.click(summarize_ui, [sm_text, sm_topk, sm_method], sm_out)
                        with gr.Accordion(HELP_TITLE, open=False):
                            gr.HTML(_help_card(
                                "این بخش از یک متن بلند، مهم‌ترین جمله‌ها را برمی‌گزیند و خلاصه‌ای "
                                "کوتاه می‌سازد.",
                                [
                                    ("روش",
                                     "<b>textrank</b> جمله‌های کلیدی را بر پایهٔ ارتباطشان با بقیه "
                                     "پیدا می‌کند (سریع و مطمئن)؛ <b>gat</b> از یک شبکهٔ گرافی برای "
                                     "همین کار استفاده می‌کند."),
                                    ("شمار جمله‌ها",
                                     "خلاصه شامل چند جمله باشد."),
                                ],
                                "هر چه متن ورودی طولانی‌تر و دارای جمله‌های کامل‌تر باشد، خلاصه دقیق‌تر "
                                "می‌شود.",
                            ))

                    with gr.Tab("🔎 پیشنهاد محتوا"):
                        rc_query = gr.Textbox(label="متن پرس‌وجو", value="یادگیری ماشین و هوش مصنوعی")
                        rc_docs = gr.Textbox(
                            label="اسناد (هر خط یک سند)", lines=7,
                            value=(
                                "شبکه‌های عصبی و یادگیری عمیق\n"
                                "آشپزی غذاهای سنتی ایرانی\n"
                                "الگوریتم‌های یادگیری ماشین\n"
                                "سفر به شمال ایران و طبیعت‌گردی\n"
                                "هوش مصنوعی در پزشکی"
                            ),
                        )
                        rc_topk = gr.Slider(1, 5, value=3, step=1, label="شمار پیشنهادها")
                        rc_btn = gr.Button("پیشنهاد بده", variant="primary")
                        rc_out = gr.HTML()
                        rc_btn.click(recommend_ui, [rc_query, rc_docs, rc_topk], rc_out)
                        with gr.Accordion(HELP_TITLE, open=False):
                            gr.HTML(_help_card(
                                "این بخش از میان چند سند، نزدیک‌ترین‌ها به متن پرس‌وجوی شما را پیدا "
                                "می‌کند — مثل یک پیشنهاددهندهٔ محتوا.",
                                [
                                    ("متن پرس‌وجو",
                                     "موضوع یا متنی که می‌خواهید شبیه‌ترین اسناد به آن را بیابید."),
                                    ("اسناد",
                                     "فهرست متن‌هایی که میانشان جست‌وجو می‌شود؛ هر خط یک سند."),
                                    ("شمار پیشنهادها",
                                     "چند سندِ نزدیک به‌عنوان نتیجه نشان داده شود."),
                                ],
                                "کنار هر نتیجه عددی به نام «شباهت» می‌بینید: هرچه بزرگ‌تر باشد، آن سند "
                                "به پرس‌وجوی شما نزدیک‌تر است.",
                            ))

                    with gr.Tab("🛡️ تشخیص نفرت‌پراکنی"):
                        hs_train = gr.Textbox(
                            label="دادهٔ آموزش (برچسب⇥متن — برچسب: نفرت/عادی)",
                            value=SAMPLE_HATE, lines=7,
                        )
                        hs_test = gr.Textbox(
                            label="متن‌های آزمون (هر خط یک متن)", lines=4,
                            value="آن‌ها باید از اینجا بروند و نابود شوند.\nامروز با خانواده پیک‌نیک رفتیم.",
                        )
                        hs_model = gr.Dropdown(["gcn", "graphsage", "gat"], value="gcn", label="معماری")
                        hs_btn = gr.Button("آموزش و تشخیص", variant="primary")
                        hs_out = gr.HTML()
                        hs_btn.click(
                            hate_speech_ui,
                            [hs_train, hs_test, hs_model, device_selector], hs_out,
                        )
                        with gr.Accordion(HELP_TITLE, open=False):
                            gr.HTML(_help_card(
                                "این بخش یاد می‌گیرد متن‌های «نفرت‌پراکن» را از متن‌های «عادی» تشخیص "
                                "دهد. اول با نمونه‌ها آموزش می‌بیند، بعد متن‌های آزمون را بررسی می‌کند.",
                                [
                                    ("دادهٔ آموزش",
                                     "نمونه‌های برچسب‌دار با قالب برچسب⇥متن، که برچسب یا «نفرت» است "
                                     "یا «عادی»."),
                                    ("متن‌های آزمون",
                                     "متن‌هایی که می‌خواهید وضعیتشان بررسی شود؛ هر خط یک متن."),
                                    ("معماری",
                                     "نوع شبکهٔ گرافی برای یادگیری. <b>gcn</b> گزینهٔ پیش‌فرض مناسبی "
                                     "است."),
                                ],
                                "این ابزار جنبهٔ آموزشی دارد و برای تصمیم‌های حساس واقعی به‌تنهایی قابل "
                                "اتکا نیست؛ نتیجه را همیشه با داوری انسانی بسنجید.",
                            ))

            # ==================== Social network analysis ====================
            with gr.Tab("🧠 شبکهٔ اجتماعی"):
                gr.Markdown(
                    "تحلیل شبکهٔ اجتماعی با **GraphSAGE**: تعبیه‌های گرهی محاسبه می‌شود، "
                    "**اجتماع‌ها** (با تفکیک‌پذیری مدولار) شناسایی و **کاربران بانفوذ** "
                    "(با PageRank) رتبه‌بندی می‌شوند.\n\n"
                    "هر خط یک رابطه است: «کاربرА کاربرB» (می‌توانید وزن اختیاری هم در ستون سوم "
                    "بدهید). این بخش روی CPU و با NumPy اجرا می‌شود."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        soc_text = gr.Textbox(
                            label="یال‌های شبکه (هر خط: کاربر۱ کاربر۲ [وزن])",
                            value=SAMPLE_SOCIAL, lines=10,
                        )
                    with gr.Column(scale=2):
                        soc_k = gr.Slider(1, 8, value=3, step=1, label="بیشینهٔ شمار اجتماع‌ها")
                        soc_btn = gr.Button("تحلیل شبکه 🧠", variant="primary")
                soc_svg = gr.HTML(label="نمای شبکه")
                with gr.Row():
                    soc_influence = gr.HTML()
                    soc_comm = gr.HTML()
                soc_btn.click(
                    social_analysis_ui, [soc_text, soc_k],
                    [soc_svg, soc_influence, soc_comm],
                )
                with gr.Accordion(HELP_TITLE, open=False):
                    gr.HTML(_help_card(
                        "این بخش یک شبکهٔ روابط (مثل دوستی میان افراد) را تحلیل می‌کند: گروه‌ها "
                        "(«اجتماع‌ها») و افراد بانفوذ را پیدا می‌کند.",
                        [
                            ("یال‌های شبکه",
                             "روابط شبکه؛ هر خط یک ارتباط میان دو نفر است، مثل «علی مریم». ستون سومِ "
                             "اختیاری می‌تواند وزن یا شدت رابطه باشد (مثلاً «علی مریم 3»)."),
                            ("بیشینهٔ شمار اجتماع‌ها",
                             "حداکثر چند گروه از شبکه استخراج شود. اگر عدد را کم بگذارید، گروه‌های "
                             "کوچک در هم ادغام می‌شوند."),
                        ],
                        "پایین، نقشهٔ رنگی شبکه (هر رنگ یک اجتماع)، فهرست گروه‌ها و رتبه‌بندی «نفوذ» "
                        "افراد را می‌بینید؛ دایرهٔ بزرگ‌تر یعنی فرد پرارتباط‌تر.",
                    ))

            # ===================== Dataset pipeline =====================
            with gr.Tab("📊 خط لولهٔ دیتاست"):
                gr.Markdown(
                    "یک دیتاست برچسب‌دار (CSV/TSV/JSONL) را با **طبقه‌بندی گرافیِ گره** "
                    "(به سبک TextGCN) ارزیابی کنید: گرافِ واژه‑سند ساخته می‌شود، اسناد به "
                    "آموزش/اعتبارسنجی/آزمون تقسیم می‌شوند و دقت/F1 گزارش می‌شود.\n\n"
                    f"اگر فایلی بارگذاری نکنید، از دیتاست نمونهٔ پروژه استفاده می‌شود: "
                    f"`{DEFAULT_DATASET}`."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        ds_file = gr.File(
                            label="بارگذاری دیتاست (.csv/.tsv/.jsonl)",
                            file_types=[".csv", ".tsv", ".jsonl", ".json"],
                            type="filepath",
                        )
                        ds_path = gr.Textbox(
                            label="یا مسیر دیتاست در پروژه", value=DEFAULT_DATASET
                        )
                        with gr.Row():
                            ds_text_col = gr.Textbox(label="نام ستون متن", value="text")
                            ds_label_col = gr.Textbox(label="نام ستون برچسب", value="label")
                            ds_fmt = gr.Dropdown(
                                ["auto", "csv", "tsv", "jsonl"], value="auto", label="قالب"
                            )
                    with gr.Column(scale=2):
                        ds_model = gr.Dropdown(
                            ["gcn", "graphsage", "gat"], value="gcn", label="معماری"
                        )
                        ds_epochs = gr.Slider(10, 200, value=40, step=10, label="دوره‌ها")
                        ds_hidden = gr.Slider(8, 128, value=32, step=8, label="بُعد نهان")
                        ds_window = gr.Slider(5, 30, value=20, step=5, label="پنجرهٔ هم‌آیی")
                        with gr.Row():
                            ds_train = gr.Slider(0.1, 0.9, value=0.7, step=0.05, label="نسبت آموزش")
                            ds_val = gr.Slider(0.05, 0.4, value=0.15, step=0.05, label="نسبت اعتبارسنجی")
                            ds_test = gr.Slider(0.05, 0.4, value=0.15, step=0.05, label="نسبت آزمون")
                        ds_btn = gr.Button("اجرای خط لوله 📊", variant="primary")
                ds_out = gr.HTML()
                with gr.Accordion(HELP_TITLE, open=False):
                    gr.HTML(_help_card(
                        "این بخش یک فایل دیتاست برچسب‌دار را می‌گیرد، خودکار آن را به گراف تبدیل "
                        "می‌کند، مدل را آموزش می‌دهد و دقت آن را گزارش می‌کند.",
                        [
                            ("بارگذاری دیتاست / مسیر دیتاست",
                             "فایل دادهٔ شما (CSV/TSV/JSONL). یا فایل را بارگذاری کنید یا مسیر آن در "
                             "پروژه را بنویسید. اگر خالی بماند، از دیتاست نمونهٔ پروژه استفاده می‌شود."),
                            ("نام ستون متن / برچسب",
                             "نام ستونی که «متن» در آن است و نام ستونی که «برچسب» (دسته) در آن است؛ "
                             "باید با سرستون‌های فایل شما یکی باشد."),
                            ("قالب",
                             "نوع فایل. <b>auto</b> خودش از روی پسوند فایل تشخیص می‌دهد."),
                            ("معماری",
                             "نوع شبکهٔ گرافی برای طبقه‌بندی (<b>gcn</b>/<b>graphsage</b>/<b>gat</b>)."),
                            ("دوره‌ها / بُعد نهان",
                             "شمار دفعات تمرین و اندازهٔ لایهٔ میانی مدل؛ بزرگ‌تر = توان بیشتر اما "
                             "کندتر."),
                            ("پنجرهٔ هم‌آیی",
                             "چند کلمهٔ کنار هم در ساخت گراف «همسایه» حساب شوند."),
                            ("نسبت آموزش / اعتبارسنجی / آزمون",
                             "داده به سه بخش تقسیم می‌شود: یادگیری، تنظیم، و سنجش نهایی. مجموع این "
                             "سه عدد هر چه باشد، خودکار به ۱۰۰٪ تبدیل می‌شود."),
                        ],
                        "نتیجهٔ نهایی «دقت» و «F1» را برای هر سه بخش نشان می‌دهد؛ معیار مهم، عملکرد "
                        "روی بخش «آزمون» است (داده‌ای که مدل در آموزش ندیده).",
                    ))
                ds_btn.click(
                    dataset_pipeline_ui,
                    [
                        ds_file, ds_path, ds_fmt, ds_text_col, ds_label_col,
                        ds_model, ds_epochs, ds_hidden, ds_window,
                        ds_train, ds_val, ds_test, device_selector,
                    ],
                    ds_out,
                )

            # ========================= Full power =========================
            # ================= Poem playground (شعریار) =================
            with gr.Tab("🪄 شعریار (پیشنهاد شعر)") as sheryar_tab:
                gr.Markdown(
                    "## شعریار — «شبیه این شعر را پیدا کن»\n"
                    "یک مدل **Graph-LM** که روی دیتاست شعر فارسی **گنجور** آموزش دیده، هر "
                    "شعر را به یک «بردار معنایی» تبدیل می‌کند. شعری را وارد کنید تا "
                    "نزدیک‌ترین شعرها از نظر **حال‌وهوا و معنا** (نه فقط کلمات مشترک) پیدا شوند.\n\n"
                    "> 🛠️ ساخت مدل (یک‌بار، در ترمینال؛ چند دقیقه روی CPU): "
                    "`python scripts/build_ganjoor_recommender.py --threads 6`"
                )
                _sh_choices = list_recommender_choices()
                with gr.Row():
                    with gr.Column(scale=3):
                        sh_model = gr.Dropdown(
                            _sh_choices,
                            value=(_sh_choices[0] if _sh_choices else None),
                            label="مدل پیشنهادگر (نقطه‌بازرسی دارای نمایهٔ شعر)",
                            info="مدل ساخته‌شده با اسکریپت گنجور",
                        )
                    with gr.Column(scale=1, min_width=160):
                        sh_refresh = gr.Button(
                            "🔄 تازه‌سازی فهرست مدل‌ها", variant="secondary"
                        )
                        gr.HTML(
                            "<div style='color:#64748b;font-size:12px;line-height:1.8;"
                            "margin-top:4px'>مدل خودکار بارگذاری می‌شود؛ این دکمه فقط "
                            "وقتی لازم است که مدلِ تازه‌ای ساخته‌اید.</div>"
                        )
                sh_status = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=3):
                        sh_query = gr.Textbox(
                            label="شعر یا بیت پرس‌وجو",
                            lines=4,
                            value=(
                                "بشنو این نی چون شکایت می‌کند\n"
                                "از جدایی‌ها حکایت می‌کند"
                            ),
                        )
                    with gr.Column(scale=2):
                        sh_topk = gr.Slider(1, 15, value=5, step=1, label="شمار نتایج")
                        sh_poet = gr.Dropdown(
                            ["همه"], value="همه",
                            label="صافی شاعر (اختیاری)",
                            info="خودکار با باز کردن تب پر می‌شود",
                        )
                        sh_btn = gr.Button("🔎 شبیه این را پیدا کن", variant="primary")
                sh_out = gr.HTML()

                # Auto-load the selected model when the tab is opened (Gradio does
                # not fire `.change` for the initial default value) and whenever
                # the user picks a different model — so the info box and the poet
                # filter populate without any extra click.
                sheryar_tab.select(
                    load_recommender_ui, [sh_model, device_selector],
                    [sh_status, sh_poet],
                )
                sh_model.change(
                    load_recommender_ui, [sh_model, device_selector],
                    [sh_status, sh_poet],
                )
                # Rescan disk for newly built models, then load the selection.
                sh_refresh.click(
                    refresh_recommenders, None, [sh_model, sh_status]
                ).then(
                    load_recommender_ui, [sh_model, device_selector],
                    [sh_status, sh_poet],
                )
                sh_btn.click(
                    find_similar_poems_ui,
                    [sh_model, sh_query, sh_topk, sh_poet, device_selector],
                    sh_out,
                )
                with gr.Accordion(HELP_TITLE, open=False):
                    gr.HTML(_help_card(
                        "این بخش از میان هزاران شعرِ نمایه‌شده، شبیه‌ترین‌ها به شعری که وارد "
                        "می‌کنید را پیدا می‌کند. «شباهت» را یک مدل گرافی می‌سنجد، نه جست‌وجوی "
                        "سادهٔ کلمه‌به‌کلمه.",
                        [
                            ("مدل پیشنهادگر",
                             "یک مدل آموخته‌شده روی دیتاست گنجور. اگر فهرست خالی است، با دستور "
                             "بالای صفحه یک‌بار آن را بسازید، سپس «تازه‌سازی فهرست» را بزنید."),
                            ("بارگذاری مدل",
                             "مدل و نمایهٔ شعرها را در حافظه می‌آورد و فهرست شاعران را پر می‌کند "
                             "(بار اول چند ثانیه)."),
                            ("شعر پرس‌وجو",
                             "شعر یا بیتی که می‌خواهید شبیه‌ترین شعرها به آن پیدا شوند."),
                            ("صافی شاعر",
                             "اگر فقط دنبال شعرهای یک شاعر خاص هستید، نام او را برگزینید؛ "
                             "وگرنه «همه»."),
                            ("شمار نتایج",
                             "چند شعرِ نزدیک نشان داده شود."),
                        ],
                        "کنار هر نتیجه نواری و عددی به نام «شباهت» می‌بینید: هرچه پُرتر و "
                        "نزدیک‌تر به ۱، آن شعر به شعر شما شبیه‌تر است.",
                    ))

            with gr.Tab("🚀 با تمام قدرت"):
                gr.Markdown(
                    "## تور گام‌به‌گام برای دیدن حداکثر توان رخشای\n"
                    "این بخش شما را از یک پیکرهٔ خام تا یک مدل زبانی گرافیِ آموخته‌شده و تولید "
                    "متن با حافظهٔ گرافی همراهی می‌کند. کافی است مرحله‌ها را به‌ترتیب اجرا کنید."
                )
                power_state = gr.State({})

                gr.HTML(
                    "<div class='rk-step'><h3>مرحلهٔ ۱ — پیکرهٔ خود را آماده کنید</h3>"
                    "متن فارسی شما (هر خط یک نمونه). می‌توانید همین نمونه را نگه دارید یا متن "
                    "خودتان را جایگزین کنید. هر چه خطوط بیشتر و مرتبط‌تر باشند، نتیجه بهتر است.</div>"
                )
                pw_corpus = gr.Textbox(label="پیکره", value=SAMPLE_TEXT, lines=8)

                gr.HTML(
                    "<div class='rk-step'><h3>مرحلهٔ ۲ — گراف چندرابطه‌ای را ببینید</h3>"
                    "اکنون گراف دانش متن شما با هفت رابطهٔ هم‌زمان (هم‌آیی، PMI، وابستگی نحوی، "
                    "هم‌ریشگی، زیرواژه، واژه‑سند و موضوع‑سند) ساخته می‌شود. این همان دانشی است "
                    "که مدل از آن بهره می‌برد.</div>"
                )
                pw_graph_btn = gr.Button("۲) ساخت و نمایش گراف", variant="primary")
                pw_svg = gr.HTML()
                with gr.Row():
                    pw_legend = gr.HTML()
                    pw_stats = gr.HTML()

                gr.HTML(
                    "<div class='rk-step'><h3>مرحلهٔ ۳ — آموزش کامل Graph-LM</h3>"
                    "یک مدل زبانی گرافی با رمزگذار GCN، آمیزش دروازه‌ای، همهٔ زیان‌های "
                    "چندوظیفه‌ای و «موتور آموزش کم‌داده» (افزایش داده، حذف گره/یال، یادگیری "
                    "متضاد و برنامهٔ درسی) آموزش می‌بیند. پیشرفت هر دوره زنده نشان داده می‌شود.</div>"
                )
                pw_train_btn = gr.Button("۳) آموزش با تمام قدرت 🚀", variant="primary")
                pw_status = gr.HTML()
                pw_hist = gr.Dataframe(
                    headers=["دوره", "خطای آموزش", "خطای اعتبارسنجی", "سرگشتگی"],
                    interactive=False, row_count=(0, "dynamic"),
                )
                pw_spark = gr.HTML()

                gr.HTML(
                    "<div class='rk-step'><h3>مرحلهٔ ۴ — تولید متن با حافظهٔ گرافی</h3>"
                    "با مدلی که خودتان آموزش دادید، متن فارسی بسازید. حافظهٔ گرافی روشن است: "
                    "مدل پیش از تولید، گره‌های مرتبط با پیشوند شما را از گراف بازیابی می‌کند و "
                    "گزارش آن را می‌بینید.</div>"
                )
                pw_prompt = gr.Textbox(label="پیشوند", value="مدل زبانی گرافی")
                pw_gen_btn = gr.Button("۴) تولید متن ✨", variant="primary")
                pw_gen_out = gr.HTML()
                pw_gen_report = gr.HTML()

                with gr.Accordion(HELP_TITLE, open=False):
                    gr.HTML(_help_card(
                        "این برگه یک «تور خودکار» است: تمام تنظیمات فنی از پیش روی بهترین حالت "
                        "قرار گرفته‌اند و شما فقط دو چیز را وارد می‌کنید و دکمه‌ها را به‌ترتیب "
                        "می‌زنید. نیازی به دانستن جزئیات فنی نیست.",
                        [
                            ("پیکره",
                             "متن فارسی شما (هر خط یک نمونه). هر چه خطوط بیشتر و مرتبط‌تر باشند، "
                             "نتیجه بهتر است. می‌توانید نمونه را نگه دارید یا متن خودتان را "
                             "جایگزین کنید."),
                            ("مرحلهٔ ۲ — ساخت گراف",
                             "نقشهٔ روابط کلمات متن شما را با هفت نوع پیوند هم‌زمان می‌سازد و "
                             "نمایش می‌دهد. همان دانشی که مدل از آن بهره می‌برد."),
                            ("مرحلهٔ ۳ — آموزش",
                             "یک مدل زبانی گرافی کامل را روی متن شما آموزش می‌دهد؛ پیشرفت هر دوره "
                             "زنده دیده می‌شود (روی CPU چند لحظه طول می‌کشد)."),
                            ("پیشوند",
                             "شروع جمله که مدل آن را در مرحلهٔ ۴ ادامه می‌دهد و متن تازه می‌سازد."),
                        ],
                        "حتماً مرحله‌ها را به‌ترتیب (۲ ← ۳ ← ۴) اجرا کنید؛ هر مرحله به نتیجهٔ مرحلهٔ "
                        "پیش از خود نیاز دارد. توضیح هر گزینهٔ فنی در راهنمای برگه‌های اختصاصی آمده است.",
                    ))

                pw_graph_btn.click(
                    power_build_graph, [pw_corpus], [pw_svg, pw_legend, pw_stats]
                )
                pw_train_btn.click(
                    power_train, [pw_corpus, power_state],
                    [pw_status, pw_hist, pw_spark, power_state],
                )
                pw_gen_btn.click(
                    power_generate, [pw_prompt, power_state], [pw_gen_out, pw_gen_report]
                )

        gr.HTML(
            "<div style='text-align:center;color:#94a3b8;margin-top:14px;font-size:13px'>"
            "رخشای — پردازش زبان فارسی مبتنی بر گراف</div>"
        )

        # On page load, set the whole document to right-to-left (the key RTL step)
        demo.load(fn=None, inputs=None, outputs=None, js=RTL_JS)
    return demo


def main():
    parser = argparse.ArgumentParser(description="رابط کاربری گرافیکی رخشای")
    parser.add_argument("--host", default="127.0.0.1", help="نشانی میزبان")
    parser.add_argument("--port", type=int, default=7860, help="پورت")
    parser.add_argument("--share", action="store_true", help="ساخت لینک عمومی موقت")
    args = parser.parse_args()

    demo = build_demo()
    demo.queue(default_concurrency_limit=2).launch(
        theme=THEME,
        css=CUSTOM_CSS,
        head=HEAD_HTML,
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
