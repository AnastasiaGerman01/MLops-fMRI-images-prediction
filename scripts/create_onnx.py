from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from scripts.models import LitMLPRegressor


def _write_config_pbtxt(
    repo_dir: Path,
    model_name: str,
    input_dim: int,
    output_dim: int,
    max_batch_size: int,
) -> None:
    text = f"""
name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ {input_dim} ]
  }}
]

output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [ {output_dim} ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ 1, 8, 16, 32, 64 ]
  max_queue_delay_microseconds: 1000
}}
""".lstrip()

    (repo_dir / "config.pbtxt").write_text(text)


def _copy_best_dt(cfg: DictConfig, out_ver: Path) -> None:
    src: Path | None = None

    if "best_dt_path" in cfg.onnx and cfg.onnx.best_dt_path not in (None, ""):
        src = Path(cfg.onnx.best_dt_path)

    if (
        src is None
        and "artifacts_dir" in cfg.onnx
        and cfg.onnx.artifacts_dir not in (None, "")
    ):
        cand = Path(cfg.onnx.artifacts_dir) / "best_dt.txt"
        if cand.exists():
            src = cand

    if src is None and "infer" in cfg and "artifacts_dir" in cfg.infer:
        cand = Path(cfg.infer.artifacts_dir) / "best_dt.txt"
        if cand.exists():
            src = cand

    if src is None or not src.exists():
        raise FileNotFoundError(
            "best_dt.txt not found. Provide one of:\n"
            "- onnx.best_dt_path=/path/to/best_dt.txt\n"
            "- onnx.artifacts_dir=<dir_with_best_dt>\n"
            "- infer.artifacts_dir=<dir_with_best_dt>\n"
        )

    dt_str = src.read_text().strip()
    dt = int(dt_str)

    dst = out_ver / "best_dt.txt"
    dst.write_text(str(dt))
    print("Wrote best_dt:", dst)


def export_onnx(cfg: DictConfig) -> None:
    ckpt_path = Path(cfg.onnx.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_repo = Path(cfg.onnx.triton_repo_dir) / str(cfg.onnx.model_name)
    out_ver = out_repo / str(cfg.onnx.model_version)
    out_ver.mkdir(parents=True, exist_ok=True)

    onnx_path = out_ver / "model.onnx"

    model = LitMLPRegressor.load_from_checkpoint(str(ckpt_path))
    model.eval()
    model = model.to("cpu")

    input_dim = int(model.hparams.input_dim)
    output_dim = int(model.hparams.output_dim)
    dummy = torch.zeros(
        (int(cfg.onnx.dummy_batch_size), input_dim),
        dtype=torch.float32,
        device="cpu",
    )

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=int(cfg.onnx.opset),
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,
    )

    _write_config_pbtxt(
        repo_dir=out_repo,
        model_name=str(cfg.onnx.model_name),
        input_dim=input_dim,
        output_dim=output_dim,
        max_batch_size=int(cfg.onnx.max_batch_size),
    )

    _copy_best_dt(cfg, out_ver=out_ver)

    print("Exported ONNX:", onnx_path)
    print("Wrote Triton config:", out_repo / "config.pbtxt")


@hydra.main(version_base=None, config_path="../configs", config_name="create_onnx")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    export_onnx(cfg)


if __name__ == "__main__":
    main()
