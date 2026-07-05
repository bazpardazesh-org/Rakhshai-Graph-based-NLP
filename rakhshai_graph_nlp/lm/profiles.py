"""Named native Graph-LM model profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model import GraphLMConfig


@dataclass(frozen=True)
class ModelProfile:
    name: str
    d_model: int
    n_heads: int
    n_layers: int
    dim_feedforward: int
    block_size: int
    graph_hidden_dim: int
    graph_heads: int
    description: str


MODEL_PROFILES: dict[str, ModelProfile] = {
    "tiny-test": ModelProfile(
        name="tiny-test",
        d_model=32,
        n_heads=4,
        n_layers=1,
        dim_feedforward=96,
        block_size=32,
        graph_hidden_dim=32,
        graph_heads=2,
        description="Tiny profile for CI and smoke tests.",
    ),
    "125m": ModelProfile(
        name="125m",
        d_model=768,
        n_heads=12,
        n_layers=12,
        dim_feedforward=3072,
        block_size=1024,
        graph_hidden_dim=768,
        graph_heads=8,
        description="Small independent pretraining profile.",
    ),
    "350m": ModelProfile(
        name="350m",
        d_model=1024,
        n_heads=16,
        n_layers=24,
        dim_feedforward=4096,
        block_size=2048,
        graph_hidden_dim=1024,
        graph_heads=8,
        description="Mid-size independent pretraining profile.",
    ),
    "1b": ModelProfile(
        name="1b",
        d_model=2048,
        n_heads=16,
        n_layers=24,
        dim_feedforward=8192,
        block_size=2048,
        graph_hidden_dim=2048,
        graph_heads=8,
        description="Large single-node or small-cluster native profile.",
    ),
    "3b": ModelProfile(
        name="3b",
        d_model=2560,
        n_heads=20,
        n_layers=32,
        dim_feedforward=10240,
        block_size=4096,
        graph_hidden_dim=2560,
        graph_heads=8,
        description="Large native pretraining profile requiring distributed training.",
    ),
}

CONTEXT_PRESETS = {512, 1024, 2048, 4096}


def available_model_profiles() -> list[str]:
    return sorted(MODEL_PROFILES)


def build_graph_lm_config_from_profile(
    profile_name: str,
    *,
    vocab_size: int,
    overrides: dict[str, Any] | None = None,
) -> GraphLMConfig:
    """Create a `GraphLMConfig` from a named native profile."""

    key = profile_name.lower()
    if key not in MODEL_PROFILES:
        raise ValueError(
            f"unknown model profile {profile_name!r}; "
            f"available profiles: {', '.join(available_model_profiles())}"
        )
    profile = MODEL_PROFILES[key]
    values: dict[str, Any] = {
        "vocab_size": vocab_size,
        "max_seq_len": profile.block_size,
        "d_model": profile.d_model,
        "n_heads": profile.n_heads,
        "n_layers": profile.n_layers,
        "dim_feedforward": profile.dim_feedforward,
        "graph_hidden_dim": profile.graph_hidden_dim,
        "graph_heads": profile.graph_heads,
    }
    if overrides:
        values.update({key: value for key, value in overrides.items() if value is not None})
    if int(values["d_model"]) % int(values["n_heads"]) != 0:
        raise ValueError("profile override makes d_model indivisible by n_heads")
    if int(values["max_seq_len"]) not in CONTEXT_PRESETS and key != "tiny-test":
        raise ValueError("max_seq_len must be one of 512, 1024, 2048, 4096")
    return GraphLMConfig(**values)

