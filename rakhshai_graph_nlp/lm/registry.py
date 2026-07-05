"""Run registry and reporting helpers for native LM training."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class RunRegistryConfig:
    run_dir: str
    command: Sequence[str] = field(default_factory=list)
    data_paths: Sequence[str] = field(default_factory=list)
    tokenizer_path: str | None = None
    checkpoint_dir: str | None = None
    notes: str = ""


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _git_hash(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def write_run_registry(config: RunRegistryConfig) -> dict[str, Any]:
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    model_config_path = run_dir / "config.json"
    registry = {
        "config": asdict(config),
        "git_hash": _git_hash(Path.cwd()),
        "metrics_hash": hash_file(metrics_path) if metrics_path.exists() else None,
        "model_config_hash": (
            hash_file(model_config_path) if model_config_path.exists() else None
        ),
        "data_hashes": {
            str(path): hash_file(path)
            for path in config.data_paths
            if Path(path).exists() and Path(path).is_file()
        },
        "tokenizer_hash": (
            hash_file(config.tokenizer_path)
            if config.tokenizer_path and Path(config.tokenizer_path).exists()
            else (
                hash_file(run_dir / "tokenizer.json")
                if (run_dir / "tokenizer.json").exists()
                else None
            )
        ),
        "checkpoint_hash": (
            hash_file(run_dir / "model.pt") if (run_dir / "model.pt").exists() else None
        ),
        "native_independence": {
            "uses_external_pretrained_lm": False,
            "uses_pretrained_embeddings": False,
            "uses_distillation": False,
            "uses_llm_synthetic_data": False,
            "uses_external_llm_judge": False,
        },
    }
    (run_dir / "run_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return registry


def build_run_report(run_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(run_dir)
    report: dict[str, Any] = {
        "run_dir": str(root),
        "files": sorted(path.name for path in root.iterdir()) if root.exists() else [],
        "native_independence": {
            "uses_external_pretrained_lm": False,
            "uses_pretrained_embeddings": False,
            "uses_distillation": False,
            "uses_llm_synthetic_data": False,
            "uses_external_llm_judge": False,
        },
    }
    for name in [
        "metrics.json",
        "config.json",
        "graph_config.json",
        "checkpoint_manifest.json",
        "run_registry.json",
        "sft_manifest.json",
        "native_eval_report.json",
    ]:
        path = root / name
        if path.exists():
            try:
                report[name.removesuffix(".json")] = json.loads(
                    path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                report[name.removesuffix(".json")] = {"path": str(path)}
    target = Path(output_path) if output_path else root / "run_report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

