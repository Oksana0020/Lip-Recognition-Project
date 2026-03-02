"""
Utilities for splitting datasets into train/validation/test sets:
- random splitting for PyTorch-style datasets
- speaker-stratified splitting for sample dictionaries, ensuring all samples
  from the same speaker stay in the same split
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple, TypeVar

T = TypeVar("T")


def split_dataset_into_train_val_test(
    full_dataset: T,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> Tuple[T, T, T]:
    import torch

    total_samples = len(full_dataset)

    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size

    torch.manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
    )

    print(
        f"Dataset split: Train={train_size}, Val={val_size}, "
        f"Test={test_size}"
    )
    return train_dataset, val_dataset, test_dataset


def split_dataset_by_speakers(
    data_samples: List[Dict],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    print("\nSplitting dataset by speakers")

    speaker_to_samples: Dict[str, List[Dict]] = defaultdict(list)
    for sample in data_samples:
        speaker_id = str(sample.get("speaker_id", "unknown"))
        speaker_to_samples[speaker_id].append(sample)

    all_speakers = list(speaker_to_samples.keys())
    random.seed(random_seed)
    random.shuffle(all_speakers)

    total_speakers = len(all_speakers)
    train_speaker_count = int(total_speakers * train_ratio)
    validation_speaker_count = int(total_speakers * validation_ratio)

    train_speakers = all_speakers[:train_speaker_count]
    validation_speakers = all_speakers[
        train_speaker_count:train_speaker_count + validation_speaker_count
    ]
    test_speakers = all_speakers[
        train_speaker_count + validation_speaker_count:
    ]

    train_samples: List[Dict] = []
    validation_samples: List[Dict] = []
    test_samples: List[Dict] = []

    for speaker_id in train_speakers:
        train_samples.extend(speaker_to_samples[speaker_id])
    for speaker_id in validation_speakers:
        validation_samples.extend(speaker_to_samples[speaker_id])
    for speaker_id in test_speakers:
        test_samples.extend(speaker_to_samples[speaker_id])

    print(
        f"Train: {len(train_samples)} samples from {len(train_speakers)} "
        "speakers"
    )
    print(
        f"Validation: {len(validation_samples)} samples from "
        f"{len(validation_speakers)} speakers"
    )
    print(
        f"Test: {len(test_samples)} samples from {len(test_speakers)} speakers"
    )

    return train_samples, validation_samples, test_samples
