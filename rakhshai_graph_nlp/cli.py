"""Command line interface for Rakhshai Graph NLP."""

from __future__ import annotations

import argparse

import numpy as np

from .graphs.graph import Graph
from .models.gcn import GCNClassifier
from .models.gat import GATClassifier
from .models.graphsage import GraphSAGEClassifier
from .metrics import accuracy
from .utils.logging import setup_logger
from .utils.random import set_seed


MODELS = {
    "gcn": GCNClassifier,
    "gat": GATClassifier,
    "graphsage": GraphSAGEClassifier,
}


def _synthetic_graph() -> tuple[Graph, np.ndarray, np.ndarray]:
    nodes = [0, 1, 2]
    A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    X = np.eye(3, dtype=float)
    labels = np.array([0, 1, 0])
    g = Graph(nodes=nodes, adjacency=A)
    return g, X, labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a tiny experiment")
    parser.add_argument("--model", choices=MODELS.keys(), default="gcn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-to", choices=["wandb", "mlflow"], default=None)
    args = parser.parse_args(argv)

    logger = setup_logger(args.log_level)
    set_seed(args.seed)

    g, X, y = _synthetic_graph()
    model_cls = MODELS[args.model]
    model = model_cls(input_dim=X.shape[1], hidden_dim=4, num_classes=2)
    model.fit(g, X, y, num_epochs=10)
    preds = model.predict(g, X)
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
