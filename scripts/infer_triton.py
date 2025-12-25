from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import scripts.utils as utils
from scripts.data_access import ensure_dvc_data
from scripts.dataloader import AudioFmriDataModule, Sub, get_audio_encoding
from scripts.visualizer import Visualizer


def _infer_triton_batches(
    cfg: DictConfig, x_batches: list[np.ndarray]
) -> list[np.ndarray]:
    try:
        import tritonclient.grpc as grpcclient
    except Exception as e:
        raise RuntimeError(
            "tritonclient is not installed. Install with:\n"
            '  uv add "tritonclient[grpc]"\n'
        ) from e

    client = grpcclient.InferenceServerClient(url=str(cfg.triton_server.url))

    model_name = str(cfg.triton_server.model_name)
    model_version = None
    if (
        "model_version" in cfg.triton_server
        and cfg.triton_server.model_version
        not in (
            None,
            "",
            "latest",
        )
    ):
        model_version = str(cfg.triton_server.model_version)

    input_name = str(getattr(cfg.triton_server, "input_name", "input"))
    output_name = str(getattr(cfg.triton_server, "output_name", "output"))

    outputs: list[np.ndarray] = []
    for xb in x_batches:
        xb = np.asarray(xb, dtype=np.float32)

        inp = grpcclient.InferInput(input_name, xb.shape, "FP32")
        inp.set_data_from_numpy(xb)

        out = grpcclient.InferRequestedOutput(output_name)

        res = client.infer(
            model_name=model_name,
            model_version=model_version,
            inputs=[inp],
            outputs=[out],
        )
        yb = res.as_numpy(output_name)
        if yb is None:
            raise RuntimeError(
                f"Triton response does not contain output '{output_name}'. "
                f"Check config.pbtxt output name."
            )
        outputs.append(yb)

    return outputs


def _load_dt_for_triton_model(cfg: DictConfig) -> int:

    if (
        "triton_repo_dir" in cfg.triton_server
        and cfg.triton_server.triton_repo_dir not in (None, "")
    ):
        if (
            "model_version" in cfg.triton_server
            and cfg.triton_server.model_version
            not in (
                None,
                "",
                "latest",
            )
        ):
            repo = Path(cfg.triton_server.triton_repo_dir)
            model_name = str(cfg.triton_server.model_name)
            model_version = str(cfg.triton_server.model_version)
            cand = repo / model_name / model_version / "best_dt.txt"
            if cand.exists():
                return int(cand.read_text().strip())

    artifacts_dir = Path(cfg.infer_triton_cfg.artifacts_dir)
    cand2 = artifacts_dir / "best_dt.txt"
    if cand2.exists():
        return int(cand2.read_text().strip())

    return int(cfg.preprocess.pairs.dt_fallback)


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

    sub = Sub(str(cfg.data.subject_id), data_root=cfg.data.path)

    dt = _load_dt_for_triton_model(cfg)

    dm = AudioFmriDataModule(
        nu=cfg.preprocess.pairs.nu,
        mu=cfg.preprocess.pairs.mu,
        n_video=cfg.preprocess.pairs.n_video,
        sub=sub,
        audio_features=x_audio,
        dt=int(dt),
        coef=int(cfg.preprocess.fmri.coef),
        train_size=float(cfg.split.train_size),
        batch_size=int(cfg.infer_triton_cfg.batch_size),
        num_workers=int(cfg.infer_triton_cfg.num_workers),
        delta_target=bool(cfg.target.delta),
        pin_memory=bool(cfg.infer_triton_cfg.pin_memory),
    )
    dm.setup()

    x_batches: list[np.ndarray] = []
    for x_t, _y_t in dm.predict_dataloader():
        x_batches.append(x_t.detach().cpu().numpy().astype(np.float32))

    pred_batches = _infer_triton_batches(cfg, x_batches=x_batches)
    delta_pred = np.concatenate(pred_batches, axis=0)

    y_test_full = dm.y_val_full

    if bool(cfg.target.delta):
        y_test_pred = y_test_full.copy()
        for t in range(1, y_test_full.shape[0]):
            y_test_pred[t] = y_test_full[0] + delta_pred[:t].sum(axis=0)
        mse_test = utils.MSE((y_test_pred[1:].T - y_test_full[1:].T))
    else:
        y_test_pred = delta_pred
        mse_test = utils.MSE((y_test_pred.T - y_test_full.T))

    print("dt used:", int(dt))
    print("MSE_test:", mse_test)

    out_dir = Path(cfg.infer_triton_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "y_test_full.npy", y_test_full)
    np.save(out_dir / "y_test_pred.npy", y_test_pred)
    (out_dir / "mse_test.txt").write_text(str(mse_test))
    (out_dir / "dt_used.txt").write_text(str(int(dt)))

    if bool(cfg.infer_triton_cfg.make_plots):

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
        compat.deltaY_test_predicted = delta_pred.T if bool(cfg.target.delta) else None
        compat.W = None

        viz = Visualizer(compat)
        viz.show_scan_slices(
            int(cfg.infer_triton_cfg.viz_scan),
            int(cfg.infer_triton_cfg.viz_dim),
            int(cfg.infer_triton_cfg.viz_slice),
        )


@hydra.main(version_base=None, config_path="../configs", config_name="infer_triton")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    infer(cfg)


if __name__ == "__main__":
    main()
