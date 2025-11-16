"""Command line interface for Rakhshai Graph NLP."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .features.pyg_data import graph_to_data
from .graphs.graph import Graph
from .metrics import accuracy
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a tiny experiment")
    parser.add_argument("--model", choices=["gcn", "graphsage", "gat"], default="gcn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-to", choices=["wandb", "mlflow"], default=None)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args(argv)

    logger = setup_logger(args.log_level)
    set_seed(args.seed)

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
