from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import librosa
import nibabel as nib
import numpy as np
import torch
from lightning import LightningDataModule
from sklearn.preprocessing import scale
from torch.utils.data import DataLoader, Dataset


class Sub:
    """Subject + загрузка fMRI. Путь через pathlib и data_root."""

    subs_with_fmri = [
        "04",
        "07",
        "08",
        "09",
        "11",
        "13",
        "14",
        "15",
        "16",
        "18",
        "22",
        "24",
        "27",
        "28",
        "29",
        "31",
        "35",
        "41",
        "43",
        "44",
        "45",
        "46",
        "47",
        "51",
        "52",
        "53",
        "55",
        "56",
        "60",
        "62",
    ]

    def __init__(self, number: str, data_root: str | Path = ".") -> None:
        if number not in Sub.subs_with_fmri:
            raise ValueError(f"У {number} испытуемого отсутствуют снимки фМРТ")

        self.number = number
        self.data_root = Path(data_root)

        self.path = (
            self.data_root
            / f"sub-{self.number}"
            / "ses-mri3t"
            / "func"
            / f"sub-{self.number}_ses-mri3t_task-film_run-1_bold.nii.gz"
        )

        scan = nib.load(str(self.path))
        data = scan.get_fdata().astype(np.float32)
        self.tensor = torch.from_numpy(data)


def get_audio_encoding(
    audio_path: str | Path,
    sr: int,
    n_mfcc: int,
) -> np.ndarray:
    audio_path = Path(audio_path)
    waveform, loaded_sr = librosa.load(str(audio_path), sr=sr)
    if loaded_sr != sr:
        waveform = librosa.resample(y=waveform, orig_sr=loaded_sr, target_sr=sr)

    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    mfcc = scale(mfcc, axis=1)
    return mfcc.T.astype(np.float32)


def _avg_pool_fmri(fmri_4d: torch.Tensor, coef: int) -> torch.Tensor:
    if coef <= 1:
        return fmri_4d
    pool = torch.nn.AvgPool3d(kernel_size=coef, stride=coef)
    return pool(fmri_4d.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)


def _build_pairs(
    dt: int,
    num_scans: int,
    nu: float,
    mu: float,
    n_video: int,
) -> list[tuple[int, int]]:
    n = n_video - int(mu * dt)
    start = int(mu * dt)
    pairs = [(int(i * nu / mu), start + i) for i in range(max(1, n))]
    return [
        (frame_idx, scan_idx)
        for frame_idx, scan_idx in pairs
        if 0 <= scan_idx < num_scans
    ]


class _PairsDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        if self.features.shape[0] != self.targets.shape[0]:
            raise ValueError("features and targets must have same length")

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.features[idx]), torch.from_numpy(self.targets[idx])


class AudioFmriDataModule(LightningDataModule):
    def __init__(
        self,
        sub: Sub,
        audio_features: np.ndarray,
        dt: int,
        coef: int,
        nu: float,
        mu: float,
        n_video: int,
        train_size: float,
        batch_size: int,
        num_workers: int,
        delta_target: bool,
        pin_memory: bool,
    ) -> None:
        super().__init__()
        self.nu = nu
        self.mu = mu
        self.n_video = n_video

        self.sub = sub
        self.audio_features = audio_features

        self.dt = int(dt)
        self.coef = int(coef)
        self.train_size = float(train_size)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.delta_target = bool(delta_target)
        self.pin_memory = bool(pin_memory)

        self._train_ds: Optional[_PairsDataset] = None
        self._val_ds: Optional[_PairsDataset] = None

        self.pooled_shape: Optional[Tuple[int, int, int]] = None
        self.train_min: Optional[float] = None
        self.train_max: Optional[float] = None

        self._y_val_full: Optional[np.ndarray] = None
        self._x_val_full: Optional[np.ndarray] = None

    @property
    def y_val_full(self) -> np.ndarray:
        if self._y_val_full is None:
            raise RuntimeError("DataModule is not set up yet")
        return self._y_val_full

    @property
    def x_val_full(self) -> np.ndarray:
        if self._x_val_full is None:
            raise RuntimeError("DataModule is not set up yet")
        return self._x_val_full

    def setup(self, stage: Optional[str] = None) -> None:
        fmri_4d = _avg_pool_fmri(self.sub.tensor, coef=self.coef)
        d1, d2, d3, t = fmri_4d.shape
        self.pooled_shape = (int(d1), int(d2), int(d3))

        pairs = _build_pairs(
            dt=self.dt, num_scans=int(t), nu=self.nu, mu=self.mu, n_video=self.n_video
        )
        features, targets = [], []

        for frame_idx, scan_idx in pairs:
            if not (0 <= frame_idx < self.audio_features.shape[0]):
                continue
            features.append(self.audio_features[frame_idx])
            targets.append(fmri_4d[..., scan_idx].reshape(-1).cpu().numpy())

        X = np.stack(features, axis=0).astype(np.float32)
        Y = np.stack(targets, axis=0).astype(np.float32)

        l_index = int(self.train_size * int(t))
        l_index = max(1, min(l_index, Y.shape[0] - 1))

        X_train, Y_train = X[:l_index], Y[:l_index]
        X_val, Y_val_full = X[l_index:], Y[l_index:]

        train_min = float(Y_train.min())
        train_max = float(Y_train.max())
        denom = (train_max - train_min) if train_max > train_min else 1.0

        Y_train_norm = (Y_train - train_min) / denom
        Y_val_norm_full = (Y_val_full - train_min) / denom

        self.train_min = train_min
        self.train_max = train_max
        self._y_val_full = Y_val_norm_full
        self._x_val_full = X_val

        if self.delta_target:
            Y_train_use = Y_train_norm[1:] - Y_train_norm[:-1]
            X_train_use = X_train[1:]
            Y_val_use = Y_val_norm_full[1:] - Y_val_norm_full[:-1]
            X_val_use = X_val[1:]
        else:
            Y_train_use, X_train_use = Y_train_norm, X_train
            Y_val_use, X_val_use = Y_val_norm_full, X_val

        self._train_ds = _PairsDataset(X_train_use, Y_train_use)
        self._val_ds = _PairsDataset(X_val_use, Y_val_use)

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            raise RuntimeError("Call setup() first")
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_ds is None:
            raise RuntimeError("Call setup() first")
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.val_dataloader()
