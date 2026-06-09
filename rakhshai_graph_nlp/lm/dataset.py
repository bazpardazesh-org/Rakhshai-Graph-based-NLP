"""Datasets for next-token Persian language modelling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .tokenizer import PersianTokenizer


class LMDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Prepare ``input_ids`` and next-token ``target_ids`` windows."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: PersianTokenizer,
        *,
        block_size: int = 128,
        stride: int | None = None,
    ):
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.block_size = block_size
        self.stride = stride or block_size
        ids: list[int] = []
        for text in texts:
            encoded = tokenizer.encode(text, add_special_tokens=True)
            if len(encoded) > 1:
                ids.extend(encoded)
        if len(ids) < 2:
            raise ValueError("corpus must contain at least two language-model tokens")

        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for start in range(0, max(1, len(ids) - 1), self.stride):
            chunk = ids[start : start + block_size + 1]
            if len(chunk) < 2:
                continue
            input_ids = chunk[:-1]
            target_ids = chunk[1:]
            pad_len = block_size - len(input_ids)
            if pad_len > 0:
                input_ids = [*input_ids, tokenizer.pad_id]  # padding targets ignored
                target_ids = [*target_ids, -100]
                if pad_len > 1:
                    input_ids.extend([tokenizer.pad_id] * (pad_len - 1))
                    target_ids.extend([-100] * (pad_len - 1))
            self.examples.append(
                (
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(target_ids, dtype=torch.long),
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]


@dataclass
class LMLoaders:
    train: DataLoader
    validation: DataLoader | None


def build_lm_dataloaders(
    dataset: LMDataset,
    *,
    batch_size: int = 8,
    validation_ratio: float = 0.1,
    seed: int = 0,
) -> LMLoaders:
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    val_size = int(round(len(dataset) * validation_ratio))
    if len(dataset) > 1 and validation_ratio > 0:
        val_size = max(1, val_size)
    val_size = min(val_size, len(dataset) - 1)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    if val_size:
        train_data, val_data = random_split(
            dataset, [train_size, val_size], generator=generator
        )
        validation = DataLoader(val_data, batch_size=batch_size)
    else:
        train_data = dataset
        validation = None
    train = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    return LMLoaders(train=train, validation=validation)
