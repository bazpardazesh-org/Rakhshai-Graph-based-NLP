"""Command line interface for Rakhshai Graph NLP."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .features.pyg_data import graph_to_data
from .features.tokenizer import tokenize
from .graphs.graph import Graph
from .graphs.text_graph import build_text_graph
from .metrics import accuracy, macro_f1
from .tasks.classification import train_node_classifier
from .utils.logging import setup_logger
from .utils.random import set_seed


def _synthetic_graph() -> tuple[Graph, np.ndarray, np.ndarray]:
    nodes = [0, 1, 2]
    A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    X = np.eye(3, dtype=float)
    labels = np.array([0, 1, 0])
    g = Graph(nodes=nodes, adjacency=A)
    return g, X, labels


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _load_text_dataset(
    path: str,
    *,
    text_column: str = "text",
    label_column: str = "label",
    dataset_format: str = "auto",
) -> tuple[list[str], list[str]]:
    """Load a labelled text classification dataset from CSV, TSV or JSONL."""

    dataset_path = Path(path)
    fmt = dataset_format
    if fmt == "auto":
        suffix = dataset_path.suffix.lower()
        if suffix == ".jsonl":
            fmt = "jsonl"
        elif suffix == ".tsv":
            fmt = "tsv"
        else:
            fmt = "csv"

    texts: list[str] = []
    labels: list[str] = []
    if fmt in {"csv", "tsv"}:
        delimiter = "\t" if fmt == "tsv" else ","
        with dataset_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if reader.fieldnames is None:
                raise ValueError("dataset file must include a header row")
            missing = {text_column, label_column} - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"dataset is missing required columns: {sorted(missing)}"
                )
            for row in reader:
                text = (row.get(text_column) or "").strip()
                label = (row.get(label_column) or "").strip()
                if text and label:
                    texts.append(text)
                    labels.append(label)
    elif fmt == "jsonl":
        with dataset_path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                try:
                    text = str(row[text_column]).strip()
                    label = str(row[label_column]).strip()
                except KeyError as exc:
                    raise ValueError(
                        f"dataset line {line_number} is missing field {exc.args[0]!r}"
                    ) from exc
                if text and label:
                    texts.append(text)
                    labels.append(label)
    else:
        raise ValueError("dataset_format must be one of: auto, csv, tsv, jsonl")

    if not texts:
        raise ValueError("dataset did not contain any labelled texts")
    return texts, labels


def _encode_labels(labels: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    label_to_id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    return np.array([label_to_id[label] for label in labels], dtype=int), label_to_id


def _split_indices(
    n_items: int,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("split ratios must be non-negative")
    if train_ratio + val_ratio + test_ratio <= 0:
        raise ValueError("at least one split ratio must be positive")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_items)
    if n_items == 1:
        return indices, np.array([], dtype=int), np.array([], dtype=int)

    total = train_ratio + val_ratio + test_ratio
    val_count = int(round(n_items * (val_ratio / total))) if val_ratio > 0 else 0
    test_count = int(round(n_items * (test_ratio / total))) if test_ratio > 0 else 0
    if val_ratio > 0 and n_items >= 3:
        val_count = max(1, val_count)
    if test_ratio > 0 and n_items >= 2:
        test_count = max(1, test_count)

    while val_count + test_count >= n_items:
        if val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break

    train_count = n_items - val_count - test_count
    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]
    test_idx = indices[train_count + val_count :]
    return train_idx, val_idx, test_idx


def _make_node_mask(n_nodes: int, node_indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(n_nodes, dtype=bool)
    mask[node_indices] = True
    return mask


def _evaluate_split(
    model: torch.nn.Module,
    data,
    labels: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    count = int(mask.sum())
    if count == 0:
        return {"count": 0, "accuracy": 0.0, "macro_f1": 0.0}
    preds = model.predict(data).cpu().numpy()
    y_true = labels[mask]
    y_pred = preds[mask]
    return {
        "count": count,
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
    }


def _run_dataset_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    texts, raw_labels = _load_text_dataset(
        args.dataset,
        text_column=args.text_column,
        label_column=args.label_column,
        dataset_format=args.dataset_format,
    )
    encoded_doc_labels, label_to_id = _encode_labels(raw_labels)
    tokenised = [tokenize(text) for text in texts]
    graph = build_text_graph(
        tokenised,
        window_size=args.window_size,
        min_count=args.min_count,
    )

    n_word_nodes = len(graph.nodes) - len(texts)
    doc_node_indices = np.arange(n_word_nodes, len(graph.nodes))
    train_docs, val_docs, test_docs = _split_indices(
        len(texts),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    train_nodes = doc_node_indices[train_docs]
    val_nodes = doc_node_indices[val_docs]
    test_nodes = doc_node_indices[test_docs]

    node_labels = np.zeros(len(graph.nodes), dtype=int)
    node_labels[doc_node_indices] = encoded_doc_labels
    train_mask = _make_node_mask(len(graph.nodes), train_nodes)
    features = np.eye(len(graph.nodes), dtype=float)

    model, losses = train_node_classifier(
        graph,
        node_labels,
        X=features,
        mask=train_mask,
        model_type=args.model,
        hidden_dim=args.hidden_dim,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        device=device,
        gat_heads=args.gat_heads,
    )
    data = graph_to_data(graph, features=features, labels=node_labels).to(device)

    val_mask = _make_node_mask(len(graph.nodes), val_nodes)
    test_mask = _make_node_mask(len(graph.nodes), test_nodes)
    report: dict[str, Any] = {
        "dataset": str(args.dataset),
        "model": args.model,
        "device": device,
        "num_documents": len(texts),
        "num_nodes": len(graph.nodes),
        "num_classes": len(label_to_id),
        "label_to_id": label_to_id,
        "splits": {
            "train": _evaluate_split(model, data, node_labels, train_mask),
            "validation": _evaluate_split(model, data, node_labels, val_mask),
            "test": _evaluate_split(model, data, node_labels, test_mask),
        },
        "final_loss": losses[-1] if losses else None,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        Path(args.report_path) if args.report_path else output_dir / "metrics.json"
    )
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report["report_path"] = str(report_path)

    if args.save_model:
        model_path = Path(args.save_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_type": args.model,
                "input_dim": features.shape[1],
                "hidden_dim": args.hidden_dim,
                "num_classes": len(label_to_id),
                "label_to_id": label_to_id,
                "metrics": report,
            },
            model_path,
        )
        report["model_path"] = str(model_path)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate graph-based Persian text classifiers, or run a "
            "small built-in smoke experiment when no dataset is provided."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        help="Path to a JSON config file whose keys match CLI option names",
        default=None,
    )
    parser.add_argument(
        "--dataset",
        help="CSV, TSV or JSONL labelled text dataset for training/evaluation",
        default=None,
    )
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "csv", "tsv", "jsonl"],
        default="auto",
        help="Dataset file format; auto detects from the file extension",
    )
    parser.add_argument("--text-column", default="text", help="Name of the text field")
    parser.add_argument(
        "--label-column",
        default="label",
        help="Name of the class label field",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Relative share of labelled documents used for training",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Relative share of labelled documents used for validation",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Relative share of labelled documents used for final testing",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/rgnn",
        help="Directory where metrics.json is written unless --report-path is set",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Explicit path for the JSON metrics report",
    )
    parser.add_argument(
        "--save-model",
        default=None,
        help="Optional path for saving the trained PyTorch model checkpoint",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="Token co-occurrence window used when building the text graph",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum token frequency required to keep a word node",
    )
    parser.add_argument(
        "--model",
        choices=["gcn", "graphsage", "gat"],
        default="gcn",
        help="Graph neural network architecture to train",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for splits")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level, such as DEBUG, INFO or WARNING",
    )
    parser.add_argument(
        "--log-to",
        choices=["wandb", "mlflow"],
        default=None,
        help="Optional experiment tracker used by the built-in smoke experiment",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=8,
        help="Hidden representation size in the GNN model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Optimizer L2 regularization strength",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability used by the GNN model",
    )
    parser.add_argument(
        "--gat-heads",
        type=int,
        default=4,
        help="Number of attention heads when --model gat is selected",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device; cuda falls back to cpu when unavailable",
    )
    config_args, _ = parser.parse_known_args(argv)
    parser.set_defaults(**_load_config(config_args.config))
    args = parser.parse_args(argv)

    logger = setup_logger(args.log_level)
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        args.device = "cpu"

    if args.dataset:
        report = _run_dataset_pipeline(args)
        val_metrics = report["splits"]["validation"]
        test_metrics = report["splits"]["test"]
        logger.info(
            "model=%s docs=%d val_accuracy=%.3f test_accuracy=%.3f report=%s",
            report["model"],
            report["num_documents"],
            val_metrics["accuracy"],
            test_metrics["accuracy"],
            report["report_path"],
        )
        return 0

    g, X, y = _synthetic_graph()
    model, _ = train_node_classifier(
        g,
        y,
        X=X,
        model_type=args.model,
        hidden_dim=args.hidden_dim,
        num_epochs=args.epochs,
        device=args.device,
    )
    data = graph_to_data(g, features=X, labels=y).to(args.device)
    preds = model.predict(data).cpu().numpy()
    acc = accuracy(y, preds)
    logger.info("model=%s accuracy=%.3f", args.model, acc)

    if args.log_to == "wandb":  # pragma: no cover - optional
        try:
            import wandb

            wandb.init(project="rgnn")
            wandb.log({"accuracy": acc})
        except Exception:  # pragma: no cover
            logger.warning("wandb not available")
    elif args.log_to == "mlflow":  # pragma: no cover
        try:
            import mlflow

            mlflow.log_metric("accuracy", acc)
        except Exception:  # pragma: no cover
            logger.warning("mlflow not available")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
