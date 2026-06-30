"""Train the detector — fine-tune YOLO11 on our labelled dataset.

A thin wrapper around Ultralytics. Point it at the data.yaml that came out of labelling and it
trains a model, dropping the weights in runs/detect/<run>/weights/. The best.pt it produces is
what the live bot loads to spot enemies.

The defaults suit an 8 GB GPU (yolo11s, 1280px images, batch 4). If you run out of VRAM, drop
--imgsz to 960 or --batch to 2, or pass --batch -1 to let Ultralytics size the batch itself.
See [[python-venv-layout]] for the venv setup.

Run it from the ML venv (Python 3.12):
    venv-ml\\Scripts\\python.exe -m src.train.train
    venv-ml\\Scripts\\python.exe -m src.train.train --data data/datasets/cblox/data.yaml --epochs 150
"""
from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA = _ROOT / "data" / "datasets" / "cblox" / "data.yaml"

# 8 GB-VRAM-safe starting point. See module docstring for the OOM knobs.
_MODEL = "yolo11s.pt"
_IMGSZ = 1280
_BATCH = 4
_EPOCHS = 100


def train(data: Path, model: str, imgsz: int, batch: int, epochs: int, device: str,
          workers: int) -> None:
    """Fine-tune `model` on `data` (a YOLO data.yaml). Weights land in runs/detect/."""
    # Imported lazily so `--help` works without the heavy ML stack installed.
    from ultralytics import YOLO

    if not data.is_file():
        raise SystemExit(
            f"dataset not found: {data}\n"
            "run autolabel.py + hand-fix in Label Studio first (build plan steps 7-8)."
        )

    print(f"Training {model} on {data}")
    print(f"imgsz={imgsz}  batch={batch}  epochs={epochs}  device={device}  workers={workers}")

    yolo = YOLO(model)
    results = yolo.train(
        data=str(data),
        imgsz=imgsz,
        batch=batch,
        epochs=epochs,
        device=device,
        workers=workers,
    )
    save_dir = getattr(results, "save_dir", "runs/detect/<run>")
    print(f"done — weights in {save_dir}/weights/best.pt")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11 on the labeled dataset.")
    parser.add_argument("--data", type=Path, default=_DEFAULT_DATA,
                        help="path to the YOLO data.yaml (default: data/datasets/cblox/data.yaml)")
    parser.add_argument("--model", default=_MODEL, help="base weights (default: yolo11s.pt)")
    parser.add_argument("--imgsz", type=int, default=_IMGSZ, help="train image size (default: 1280)")
    parser.add_argument("--batch", type=int, default=_BATCH,
                        help="batch size; -1 = auto-fit largest (default: 4)")
    parser.add_argument("--epochs", type=int, default=_EPOCHS, help="epochs (default: 100)")
    parser.add_argument("--device", default="0", help="CUDA index '0' or 'cpu' (default: 0)")
    parser.add_argument("--workers", type=int, default=4,
                        help="dataloader workers; lower if workers crash on Windows (default: 4)")
    args = parser.parse_args()
    train(args.data, args.model, args.imgsz, args.batch, args.epochs, args.device, args.workers)


if __name__ == "__main__":
    _main()
