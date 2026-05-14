"""Hate-speech detection helpers for Persian text.

The module keeps a lightweight keyword helper for fast rule-based checks and
adds a trainable ``HateSpeechDetector`` built on the package's
``TextGraphClassifier`` pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from .classification import TextGraphClassifier


def contains_hate_speech(text: str, hate_terms: Iterable[str]) -> bool:
    """Check whether a text contains any hateful terms."""

    for term in hate_terms:
        if term in text:
            return True
    return False


class HateSpeechDetector:
    """Trainable Persian hate-speech detector.

    Labels are normalised to ``"hate"`` and ``"normal"`` so callers can pass
    booleans, integers or common text labels during training.
    """

    positive_label = "hate"
    negative_label = "normal"

    def __init__(
        self,
        *,
        model: str = "gcn",
        hidden_dim: int = 64,
        num_epochs: int = 200,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.classifier = TextGraphClassifier(
            model=model,  # type: ignore[arg-type]
            hidden_dim=hidden_dim,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            seed=seed,
        )

    @classmethod
    def _normalise_label(cls, label: str | int | bool) -> str:
        if isinstance(label, bool):
            return cls.positive_label if label else cls.negative_label
        label_text = str(label).strip().lower()
        if label_text in {"1", "true", "hate", "hateful", "toxic", "نفرت"}:
            return cls.positive_label
        if label_text in {"0", "false", "normal", "safe", "neutral", "عادی"}:
            return cls.negative_label
        return label_text

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[str | int | bool],
    ) -> "HateSpeechDetector":
        """Train the detector on labelled examples."""

        normalised = [self._normalise_label(label) for label in labels]
        self.classifier.fit(texts, normalised)
        return self

    def predict(self, texts: Sequence[str]) -> list[bool]:
        """Return ``True`` for texts predicted as hate speech."""

        labels = self.classifier.predict(texts)
        return [label == self.positive_label for label in labels]

    def predict_labels(self, texts: Sequence[str]) -> list[str]:
        """Return string labels such as ``"hate"`` or ``"normal"``."""

        return self.classifier.predict(texts)

    def evaluate(
        self,
        texts: Sequence[str],
        labels: Sequence[str | int | bool],
    ) -> dict[str, float | int]:
        """Evaluate the detector with accuracy and macro-F1."""

        normalised = [self._normalise_label(label) for label in labels]
        return self.classifier.evaluate(texts, normalised)

    def save(self, path: str | Path) -> None:
        """Save the trained detector."""

        self.classifier.save(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
    ) -> "HateSpeechDetector":
        """Load a trained detector."""

        detector = cls(device=device)
        detector.classifier = TextGraphClassifier.load(path, device=device)
        return detector
