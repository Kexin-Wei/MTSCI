from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils import sample_mask


MODALITY_DIRS = {
    "cgm": "CGM_metabonet_pp",
    "ecg": "ECG_physionet_pp",
    "eeg": "EEG_Zuco_pp",
    "emg": "EMG_emg2qwerty_pp",
}


def generate_physi_train_dataloader(
    dataset_path,
    seq_len,
    missing_ratio,
    missing_pattern,
    batch_size=4,
    modality="cgm",
    stride=None,
    proportion=0.8,
    seed=9101112,
    max_files=None,
    num_workers=0,
):
    windows, masks, groups = _load_split_windows(
        dataset_path, modality, seq_len, stride, "train", proportion, seed, max_files
    )
    rng = np.random.default_rng(seed)
    tensors = _make_train_tensors(windows, masks, groups, missing_ratio, missing_pattern, rng)
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def generate_physi_val_test_dataloader(
    dataset_path,
    seq_len,
    missing_ratio,
    missing_pattern="point",
    batch_size=4,
    mode="val",
    modality="cgm",
    stride=None,
    proportion=0.8,
    seed=None,
    max_files=None,
    num_workers=0,
):
    if mode not in {"val", "test"}:
        raise ValueError("mode must be val or test")
    seed = seed if seed is not None else (9101111 if mode == "val" else 9101110)
    windows, masks, _ = _load_split_windows(
        dataset_path, modality, seq_len, stride, mode, proportion, seed, max_files
    )
    rng = np.random.default_rng(seed)
    tensors = _make_eval_tensors(windows, masks, missing_ratio, missing_pattern, rng, mode)
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def _load_split_windows(dataset_path, modality, seq_len, stride, split, proportion, seed, max_files=None):
    if modality not in MODALITY_DIRS:
        raise ValueError(f"Unknown Physi modality {modality!r}")
    stride = int(stride or seq_len)
    root = Path(dataset_path) / MODALITY_DIRS[modality]
    if not root.exists():
        raise FileNotFoundError(f"Missing Physi directory: {root}")

    files = sorted(root.glob("*.npy"))
    if max_files is not None:
        files = files[:max_files]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    train_end = int(np.ceil(len(files) * proportion))
    val_end = train_end + int(np.floor((len(files) - train_end) / 2))
    if split == "train":
        selected = order[:train_end]
    elif split == "val":
        selected = order[train_end:val_end]
    else:
        selected = order[val_end:]

    windows, masks, groups = [], [], []
    for idx in sorted(selected):
        record = np.load(files[idx], allow_pickle=True).item()
        data = np.nan_to_num(record["data"].astype(np.float32, copy=False))
        observed = record.get("observed_mask", np.isfinite(data)).astype(np.float32, copy=False)
        length = int(record.get("length", data.shape[0]))
        for start in range(0, max(length - seq_len + 1, 0), stride):
            end = start + seq_len
            windows.append(data[start:end] * observed[start:end])
            masks.append(observed[start:end])
            groups.append(idx)

    if not windows:
        raise ValueError(f"No Physi windows for modality={modality}, split={split}, seq_len={seq_len}")
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(masks, dtype=np.float32),
        np.asarray(groups, dtype=np.int64),
    )


def _artificial_mask(shape, missing_ratio, missing_pattern, rng):
    if missing_pattern == "block":
        return sample_mask(
            shape=shape,
            p=0.0015,
            p_noise=missing_ratio,
            min_seq=12,
            max_seq=12 * 4,
            rng=rng,
        ).astype(np.float32)
    return sample_mask(
        shape=shape,
        p=0.0,
        p_noise=missing_ratio,
        min_seq=12,
        max_seq=12 * 4,
        rng=rng,
    ).astype(np.float32)


def _make_eval_tensors(windows, gt_mask, missing_ratio, missing_pattern, rng, mode):
    indicating_mask = _artificial_mask(windows.shape, missing_ratio, missing_pattern, rng)
    X = windows * (1 - indicating_mask)
    mask = gt_mask * (1 - indicating_mask)
    _print_stats(mode, gt_mask, indicating_mask, mask, missing_pattern)
    return (
        torch.from_numpy(X).float(),
        torch.from_numpy(mask).float(),
        torch.from_numpy(windows).float(),
        torch.from_numpy(gt_mask).float(),
        torch.from_numpy(indicating_mask).float(),
    )


def _make_train_tensors(windows, gt_mask, groups, missing_ratio, missing_pattern, rng):
    if windows.shape[0] < 2:
        raise ValueError("MTSCI Physi training needs at least two windows for adjacent prediction targets")

    same_record = groups[:-1] == groups[1:]
    current = windows[:-1][same_record]
    current_gt_mask = gt_mask[:-1][same_record]
    pred = windows[1:][same_record]
    pred_gt_mask = gt_mask[1:][same_record]
    if current.shape[0] == 0:
        raise ValueError("MTSCI Physi training found no adjacent window pairs within a recording")
    indicating_mask = _artificial_mask(current.shape, missing_ratio, missing_pattern, rng)
    X = current * (1 - indicating_mask)
    mask = current_gt_mask * (1 - indicating_mask)
    _print_stats("Train", current_gt_mask, indicating_mask, mask, missing_pattern)
    return (
        torch.from_numpy(X).float(),
        torch.from_numpy(mask).float(),
        torch.from_numpy(indicating_mask).float(),
        torch.from_numpy(current).float(),
        torch.from_numpy(current_gt_mask).float(),
        torch.from_numpy(pred).float(),
        torch.from_numpy(pred_gt_mask).float(),
    )


def _print_stats(name, gt_mask, indicating_mask, mask, missing_pattern):
    print(
        "{}: original missing ratio = {:.4f}, artificial missing ratio = {:.4f}, artificial missing pattern: {}, overall missing ratio = {:.4f}".format(
            name,
            1 - np.sum(gt_mask) / gt_mask.size,
            np.sum(indicating_mask) / indicating_mask.size,
            missing_pattern,
            1 - np.sum(mask) / mask.size,
        )
    )
