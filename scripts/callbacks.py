from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback


@dataclass
class MetricsPlotCallback(Callback):
    out_dir: Path
    prefix: str = "train"
    history: dict[str, list[float]] = field(default_factory=lambda: {})

    def _append(self, name: str, value: Any) -> None:
        try:
            v = float(value.detach().cpu().item())
        except Exception:
            try:
                v = float(value)
            except Exception:
                return
        self.history.setdefault(name, []).append(v)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        metrics = trainer.callback_metrics
        self._append("train_MSE", metrics.get("train_MSE"))
        self._append("val_MSE", metrics.get("val_MSE"))
        self._append("val_Pearson", metrics.get("val_Pearson"))

    def on_fit_end(self, trainer, pl_module) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        plots = [
            ("val_MSE.png", "val_MSE"),
            ("train_MSE.png", "train_MSE"),
            ("val_Pearson.png", "val_Pearson"),
        ]

        for filename, key in plots:
            values = self.history.get(key, [])
            if not values:
                continue
            plt.figure()
            plt.plot(range(1, len(values) + 1), values)
            plt.xlabel("epoch")
            plt.ylabel(key)
            plt.title(f"{self.prefix}: {key}")
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig(self.out_dir / filename, dpi=200)
            plt.close()
