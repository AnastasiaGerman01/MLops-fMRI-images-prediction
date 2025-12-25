from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
import numpy as np
from omegaconf import DictConfig

import scripts.utils as utils
from scripts.data_access import ensure_dvc_data
from scripts.dataloader import AudioFmriDataModule, Sub, get_audio_encoding
from scripts.models import LitMLPRegressor
from scripts.visualizer import Visualizer


def infer(cfg: DictConfig) -> None:
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

    artifacts_dir = Path(cfg.infer_cfg.artifacts_dir)
    best_dt_path = artifacts_dir / "best_dt.txt"
    dt = (
        int(best_dt_path.read_text().strip())
        if best_dt_path.exists()
        else int(cfg.preprocess.pairs.dt_fallback)
    )

    dm = AudioFmriDataModule(
        nu=cfg.preprocess.pairs.nu,
        mu=cfg.preprocess.pairs.mu,
        n_video=cfg.preprocess.pairs.n_video,
        sub=sub,
        audio_features=x_audio,
        dt=int(dt),
        coef=int(cfg.preprocess.fmri.coef),
        train_size=float(cfg.split.train_size),
        batch_size=int(cfg.infer_cfg.batch_size),
        num_workers=int(cfg.infer_cfg.num_workers),
        delta_target=bool(cfg.target.delta),
        pin_memory=bool(cfg.infer_cfg.pin_memory),
    )
    dm.setup()

    ckpt_path = Path(cfg.infer_cfg.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    model = LitMLPRegressor.load_from_checkpoint(str(ckpt_path))

    trainer = L.Trainer(
        accelerator=cfg.infer_cfg.accelerator,
        devices=cfg.infer_cfg.devices,
        logger=False,
        enable_checkpointing=False,
    )

    preds = trainer.predict(model, datamodule=dm, ckpt_path=str(ckpt_path))
    delta_pred = np.concatenate([p.detach().cpu().numpy() for p in preds], axis=0)

    y_test_full = dm.y_val_full
    y_test_pred = y_test_full.copy()
    for t in range(1, y_test_full.shape[0]):
        y_test_pred[t] = y_test_full[0] + delta_pred[:t].sum(axis=0)

    mse_test = utils.MSE((y_test_pred[1:].T - y_test_full[1:].T))
    print("MSE_test:", mse_test)

    out_dir = Path(cfg.infer_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "y_test_full.npy", y_test_full)
    np.save(out_dir / "y_test_pred.npy", y_test_pred)
    (out_dir / "mse_test.txt").write_text(str(mse_test))

    if bool(cfg.infer_cfg.make_plots):

        class Compat:
            pass

        compat = Compat()
        compat.delta = bool(cfg.target.delta)
        compat.alpha = 0.0
        compat.dt = int(dt)
        compat.coef = int(cfg.preprocess.fmri.coef)
        compat.sub = sub
        compat._d1, compat._d2, compat._d3 = dm.pooled_shape
        compat.Y_test = y_test_full.T
        compat.Y_test_predicted = y_test_pred.T
        compat.deltaY_test_predicted = delta_pred.T
        compat.W = None

        viz = Visualizer(compat)
        viz.show_scan_slices(
            int(cfg.infer_cfg.viz_scan),
            int(cfg.infer_cfg.viz_dim),
            int(cfg.infer_cfg.viz_slice),
        )


@hydra.main(version_base=None, config_path="../configs", config_name="infer")
def main(cfg: DictConfig) -> None:
    infer(cfg)


if __name__ == "__main__":
    main()
