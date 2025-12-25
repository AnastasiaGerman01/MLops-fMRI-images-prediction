from __future__ import annotations

from typing import Sequence

import torch
from lightning import LightningModule
from torchmetrics import MeanSquaredError


def _pearson_corr_per_sample(
    y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    y_hat_c = y_hat - y_hat.mean(dim=1, keepdim=True)
    y_c = y - y.mean(dim=1, keepdim=True)

    num = (y_hat_c * y_c).sum(dim=1)
    den = torch.sqrt((y_hat_c**2).sum(dim=1) * (y_c**2).sum(dim=1)).clamp_min(eps)
    return num / den


class LitMLPRegressor(LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        lr: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        layers: list[torch.nn.Module] = []
        prev_dim = int(input_dim)

        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(torch.nn.ReLU())
            if float(dropout) > 0.0:
                layers.append(torch.nn.Dropout(float(dropout)))
            prev_dim = int(hidden_dim)

        layers.append(torch.nn.Linear(prev_dim, int(output_dim)))
        self.net = torch.nn.Sequential(*layers)

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = torch.mean((y_hat - y) ** 2)

        self.log(
            "train_MSE",
            self.train_mse(y_hat, y),
            on_epoch=True,
            prog_bar=True,
            on_step=False,
        )
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        y_hat = self(x)
        pearson_batch = _pearson_corr_per_sample(y_hat, y).mean()

        self.log("val_MSE", self.val_mse(y_hat, y), on_epoch=True, prog_bar=True)
        self.log(
            "val_Pearson", pearson_batch, on_step=False, on_epoch=True, prog_bar=True
        )

    def predict_step(
        self, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )
