"""
Script to split preprocessed viseme/phoneme/word datasets into three sets:
train, validation and test.

This script creates a deterministic split of the dataset:
- Train: 7,000 videos
- Validation: 1,000 videos
- Test: 2,000 videos

The split is saved to JSON and can be applied to viseme,
phoneme or word-level datasets.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _allocate_counts_proportionally(
    class_counts: Dict[str, int],
    target_total: int,
) -> Dict[str, int]:
    """Allocate integer counts with largest-remainder method."""
    if target_total < 0:
        raise ValueError("target_total must be non-negative")

    total_available = sum(class_counts.values())
    if target_total > total_available:
        raise ValueError(
            "Requested target exceeds available samples: "
            f"target={target_total}, available={total_available}"
        )

    if total_available == 0:
        return {class_label: 0 for class_label in class_counts}

    exact = {
        class_label: (count / total_available) * target_total
        for class_label, count in class_counts.items()
    }

    allocated = {
        class_label: int(value)
        for class_label, value in exact.items()
    }

    assigned = sum(allocated.values())
    remainder = target_total - assigned

    remainder_order = sorted(
        class_counts.keys(),
        key=lambda class_label: (
            exact[class_label] - allocated[class_label],
            class_counts[class_label],
        ),
        reverse=True,
    )

    index = 0
    while remainder > 0 and remainder_order:
        class_label = remainder_order[index % len(remainder_order)]
        if allocated[class_label] < class_counts[class_label]:
            allocated[class_label] += 1
            remainder -= 1
        index += 1

    return allocated


def collect_all_samples(
    data_root: Path,
    allowed_extensions: Set[str],
) -> Dict[str, List[Path]]:
    """Collect samples organized by class label folder."""
    samples_by_class: Dict[str, List[Path]] = defaultdict(list)

    if not data_root.exists():
        raise ValueError(f"Data root directory does not exist: {data_root}")

    for class_folder in data_root.iterdir():
        if not class_folder.is_dir():
            continue

        class_label = class_folder.name
        for file_path in class_folder.iterdir():
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in allowed_extensions:
                continue
            samples_by_class[class_label].append(file_path)

    return dict(samples_by_class)


def create_stratified_split(
    samples_by_class: Dict[str, List[Path]],
    data_root: Path,
    train_size: int = 7000,
    val_size: int = 1000,
    test_size: int = 2000,
    random_seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Create stratified split with exact target sizes."""
    random.seed(random_seed)

    total_samples = sum(len(samples) for samples in samples_by_class.values())
    total_target = train_size + val_size + test_size

    if total_target <= 0:
        raise ValueError("train_size + val_size + test_size must be > 0")

    if total_target > total_samples:
        raise ValueError(
            "Requested split sizes exceed available samples: "
            f"requested={total_target}, available={total_samples}"
        )

    train_ratio = train_size / total_target
    val_ratio = val_size / total_target
    test_ratio = test_size / total_target

    print(f"\nTotal samples available: {total_samples:,}")
    print(
        "Target split: "
        f"{train_size:,} train / {val_size:,} val / {test_size:,} test"
    )
    print(
        "Split ratios: "
        f"{train_ratio:.1%} / {val_ratio:.1%} / {test_ratio:.1%}"
    )

    train_samples: List[str] = []
    val_samples: List[str] = []
    test_samples: List[str] = []

    class_to_relative_samples: Dict[str, List[str]] = {}
    for class_label, samples in sorted(samples_by_class.items()):
        class_samples = [str(p.relative_to(data_root)) for p in samples]
        random.shuffle(class_samples)
        class_to_relative_samples[class_label] = class_samples

    class_counts = {
        class_label: len(class_samples)
        for class_label, class_samples in class_to_relative_samples.items()
    }

    selected_per_class = _allocate_counts_proportionally(
        class_counts=class_counts,
        target_total=total_target,
    )
    train_per_class = _allocate_counts_proportionally(
        class_counts=selected_per_class,
        target_total=train_size,
    )

    remaining_after_train = {
        class_label: selected_per_class[class_label]
        - train_per_class[class_label]
        for class_label in selected_per_class
    }
    val_per_class = _allocate_counts_proportionally(
        class_counts=remaining_after_train,
        target_total=val_size,
    )

    test_per_class = {
        class_label: remaining_after_train[class_label]
        - val_per_class[class_label]
        for class_label in remaining_after_train
    }

    for class_label in sorted(class_to_relative_samples):
        class_samples = class_to_relative_samples[class_label]
        n_available = len(class_samples)
        n_selected = selected_per_class[class_label]
        n_train = train_per_class[class_label]
        n_val = val_per_class[class_label]
        n_test = test_per_class[class_label]

        selected = class_samples[:n_selected]
        train_samples.extend(selected[:n_train])
        val_samples.extend(selected[n_train:n_train + n_val])
        test_samples.extend(selected[n_train + n_val:n_train + n_val + n_test])

        print(
            f"  {class_label:>10}: {n_available:>5} available, "
            f"{n_selected:>5} selected → "
            f"{n_train:>4} train / {n_val:>4} val / {n_test:>4} test"
        )

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    return train_samples, val_samples, test_samples


def save_split_manifest(
    train_samples: List[str],
    val_samples: List[str],
    test_samples: List[str],
    output_path: Path,
    metadata: Dict,
) -> None:
    """Save the dataset split to a JSON manifest file."""
    manifest = {
        "metadata": metadata,
        "train": train_samples,
        "validation": val_samples,
        "test": test_samples,
        "counts": {
            "train": len(train_samples),
            "validation": len(val_samples),
            "test": len(test_samples),
            "total": len(train_samples) + len(val_samples) + len(test_samples),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSplit manifest saved to: {output_path}")
    print(f"  Train: {len(train_samples):,} samples")
    print(f"  Validation: {len(val_samples):,} samples")
    print(f"  Test: {len(test_samples):,} samples")
    print(f"  Total: {manifest['counts']['total']:,} samples")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split dataset into train/validation/test sets",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help=(
            "Path to dataset root directory (with class subfolders and "
            "segment files)"
        ),
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".npy,.json",
        help=(
            "Comma-separated file extensions to include "
            "(default: .npy,.json)"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="splits/split_manifest.json",
        help=(
            "Path to save split manifest JSON "
            "(default: splits/split_manifest.json)"
        ),
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=7000,
        help="Number of training samples (default: 7000)",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=1000,
        help="Number of validation samples (default: 1000)",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=2000,
        help="Number of test samples (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_path = Path(args.output)

    print("=" * 80)
    print("DATASET SPLITTING - TRAIN/VALIDATION/TEST")
    print("=" * 80)
    print(f"\nData root: {data_root}")
    print(f"Random seed: {args.seed}")

    allowed_extensions = {
        ext.strip().lower()
        for ext in args.extensions.split(",")
        if ext.strip()
    }
    if not allowed_extensions:
        print("\nERROR: --extensions must contain at least one extension")
        return

    print(
        "Included extensions: "
        f"{', '.join(sorted(allowed_extensions))}"
    )

    print("\nCollecting samples from dataset...")
    samples_by_class = collect_all_samples(
        data_root,
        allowed_extensions=allowed_extensions,
    )

    if not samples_by_class:
        print("\nERROR: No samples found in the dataset!")
        print(
            "Please check that "
            f"{data_root} contains class subfolders with matching files."
        )
        return

    print(f"Found {len(samples_by_class)} classes")

    print("\nCreating stratified split...")
    train_samples, val_samples, test_samples = create_stratified_split(
        samples_by_class,
        data_root=data_root,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_seed=args.seed,
    )

    metadata = {
        "data_root": str(data_root),
        "random_seed": args.seed,
        "extensions": sorted(allowed_extensions),
        "target_sizes": {
            "train": args.train_size,
            "validation": args.val_size,
            "test": args.test_size,
        },
        "num_classes": len(samples_by_class),
        "class_labels": sorted(samples_by_class.keys()),
    }

    save_split_manifest(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        output_path=output_path,
        metadata=metadata,
    )

    print("\n" + "=" * 80)
    print("SPLIT COMPLETE!")
    print("=" * 80)
    print(
        "\nSplit manifest can be used now for training, validation, "
        "and testing."
    )
    print("Load the JSON file to get the exact file lists for each split.")


if __name__ == "__main__":
    main()
