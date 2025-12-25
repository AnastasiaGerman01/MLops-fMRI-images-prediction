from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def save_object(obj: Any, filename: str | Path) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename: str | Path) -> Any:
    path = Path(filename)
    with path.open("rb") as inp:
        return pickle.load(inp)


def preprocess(v: np.ndarray) -> np.ndarray:
    denom = v.max() - v.min()
    denom = denom if denom != 0 else 1.0
    return (v - v.min()) / denom


def MSE(A: np.ndarray) -> float:
    m, n = A.shape
    return float((np.linalg.norm(A, "fro") ** 2) / (m * n))
