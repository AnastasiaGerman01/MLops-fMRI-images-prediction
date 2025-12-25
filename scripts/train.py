from __future__ import annotations

import shutil
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from scripts.callbacks import MetricsPlotCallback
from scripts.data_access import ensure_dvc_data
from scripts.dataloader import AudioFmriDataModule, Sub, get_audio_encoding
from scripts.logging_utils import flatten_dict, get_git_commit_id
from scripts.models import LitMLPRegressor


def _make_mlflow_logger(cfg: DictConfig, run_name: str) -> MLFlowLogger | None:
    if not bool(cfg.mlflow.enable):
        return None
    return MLFlowLogger(
        tracking_uri=str(cfg.mlflow.tracking_uri),
        experiment_name=str(cfg.mlflow.experiment_name),
        run_name=run_name,
    )


def _train_for_dt(
    cfg: DictConfig, x_audio, sub: Sub, dt: int, run_dir: Path
) -> tuple[float, Path]:
    dm = AudioFmriDataModule(
        nu=cfg.preprocess.pairs.nu,
        mu=cfg.preprocess.pairs.mu,
        n_video=cfg.preprocess.pairs.n_video,
        sub=sub,
        audio_features=x_audio,
        dt=int(dt),
        coef=int(cfg.preprocess.fmri.coef),
        train_size=float(cfg.split.train_size),
        batch_size=int(cfg.train_cfg.batch_size),
        num_workers=int(cfg.train_cfg.num_workers),
        delta_target=bool(cfg.target.delta),
        pin_memory=bool(cfg.train_cfg.pin_memory),
    )
    dm.setup()

    x0, y0 = next(iter(dm.train_dataloader()))

    model = LitMLPRegressor(
        input_dim=int(x0.shape[1]),
        output_dim=int(y0.shape[1]),
        hidden_dims=tuple(cfg.model.hidden_dims),
        dropout=float(cfg.model.dropout),
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
    )

    run_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"sub{cfg.data.subject_id}_dt{dt}"
    mlflow_logger = _make_mlflow_logger(cfg, run_name=run_name)

    plots_dir = Path(cfg.plots.dir) / run_name
    plot_cb = MetricsPlotCallback(out_dir=plots_dir, prefix=run_name)

    checkpoint_cb = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best",
        monitor="val_MSE",
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
    )

    trainer = L.Trainer(
        max_epochs=int(cfg.train_cfg.max_epochs),
        accelerator=cfg.train_cfg.accelerator,
        devices=cfg.train_cfg.devices,
        logger=mlflow_logger,
        callbacks=[checkpoint_cb, plot_cb],
        enable_checkpointing=True,
    )

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    flat = flatten_dict(cfg_dict if isinstance(cfg_dict, dict) else {})
    commit_id = get_git_commit_id()

    if mlflow_logger is not None:
        mlflow_logger.log_hyperparams(flat)
        mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "git_commit", commit_id)
        for k, v in dict(cfg.mlflow.tags).items():
            mlflow_logger.experiment.set_tag(mlflow_logger.run_id, str(k), str(v))

    trainer.fit(model, datamodule=dm)

    best_score = checkpoint_cb.best_model_score
    best_val_mse = (
        float(best_score.detach().cpu().item())
        if best_score is not None
        else float("inf")
    )
    best_ckpt_path = Path(checkpoint_cb.best_model_path)

    if mlflow_logger is not None:
        for png in plots_dir.glob("*.png"):
            mlflow_logger.experiment.log_artifact(
                mlflow_logger.run_id, str(png), artifact_path="plots"
            )

    return best_val_mse, best_ckpt_path


def train(cfg: DictConfig) -> None:
    ensure_dvc_data(
        data_path=cfg.data.path,
        remote_name=cfg.dvc.remote,
        timeout_seconds=cfg.dvc.timeout,
        jobs=cfg.dvc.jobs,
    )

    x_audio = get_audio_encoding(
        audio_path=Path(cfg.data.path) / cfg.data.audio_filename,
        sr=int(cfg.preprocess.audio.sr),
        n_mfcc=int(cfg.preprocess.audio.n_mfcc),
    )
    sub = Sub(cfg.data.subject_id, data_root=cfg.data.path)

    dt_min = int(cfg.preprocess.pairs.dt_min)
    dt_max = int(cfg.preprocess.pairs.dt_max)
    if dt_min > dt_max:
        raise ValueError("pairs.dt_min must be <= pairs.dt_max")

    artifacts_dir = Path(cfg.train_cfg.artifacts_dir)
    checkpoints_root = Path(cfg.train_cfg.checkpoint_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    best_dt: int | None = None
    best_val_mse = float("inf")
    best_ckpt: Path | None = None

    for dt in range(dt_min, dt_max + 1):
        run_dir = checkpoints_root / f"dt_{dt}"
        val_mse, ckpt_path = _train_for_dt(cfg, x_audio, sub, dt=dt, run_dir=run_dir)
        print(f"[dt={dt}] best val_MSE = {val_mse}  ckpt={ckpt_path}")
        if val_mse < best_val_mse and ckpt_path.exists():
            best_val_mse = val_mse
            best_dt = dt
            best_ckpt = ckpt_path

    if best_dt is None or best_ckpt is None:
        raise RuntimeError("dt search failed: no best checkpoint found")

    final_ckpt = artifacts_dir / "best.ckpt"
    shutil.copy2(best_ckpt, final_ckpt)
    (artifacts_dir / "best_dt.txt").write_text(str(best_dt))
    (artifacts_dir / "best_val_mse.txt").write_text(str(best_val_mse))

    print(f"BEST: dt={best_dt}, val_MSE={best_val_mse}, ckpt={final_ckpt}")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
