#!/usr/bin/env python
import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dataloader"))
sys.path.insert(0, str(REPO_ROOT / "utils"))

from physi_dataloader import (  # noqa: E402
    generate_physi_train_dataloader,
    generate_physi_val_test_dataloader,
)


FEATURE_SIZES = {"cgm": 6, "ecg": 12, "eeg": 128, "emg": 32}


def check_modality(modality, args):
    feature_size = FEATURE_SIZES[modality]
    train = generate_physi_train_dataloader(
        args.data_root,
        args.seq_len,
        args.missing_ratio,
        args.missing_pattern,
        batch_size=args.batch_size,
        modality=modality,
        stride=args.stride,
        proportion=0.5,
        max_files=args.max_files,
    )
    val = generate_physi_val_test_dataloader(
        args.data_root,
        args.seq_len,
        args.missing_ratio,
        args.missing_pattern,
        batch_size=args.batch_size,
        mode="val",
        modality=modality,
        stride=args.stride,
        proportion=0.5,
        max_files=args.max_files,
    )
    test = generate_physi_val_test_dataloader(
        args.data_root,
        args.seq_len,
        args.missing_ratio,
        args.missing_pattern,
        batch_size=args.batch_size,
        mode="test",
        modality=modality,
        stride=args.stride,
        proportion=0.5,
        max_files=args.max_files,
    )

    train_batch = next(iter(train))
    val_batch = next(iter(val))
    test_batch = next(iter(test))
    assert len(train_batch) == 7
    assert len(val_batch) == 5
    assert len(test_batch) == 5

    for tensor in train_batch:
        assert tuple(tensor.shape[1:]) == (args.seq_len, feature_size)
    for tensor in val_batch + test_batch:
        assert tuple(tensor.shape[1:]) == (args.seq_len, feature_size)

    print(
        f"{modality}: train_batches={len(train)} val_batches={len(val)} "
        f"test_batches={len(test)} train_tensor_shape={tuple(train_batch[0].shape)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/mnt/nvme2/kexin/Physi_post_processed")
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--stride", type=int, default=4096)
    parser.add_argument("--max_files", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--missing_ratio", type=float, default=0.2)
    parser.add_argument("--missing_pattern", choices=["point", "block"], default="point")
    parser.add_argument("--modalities", nargs="+", default=list(FEATURE_SIZES))
    args = parser.parse_args()

    for modality in args.modalities:
        check_modality(modality, args)


if __name__ == "__main__":
    main()
