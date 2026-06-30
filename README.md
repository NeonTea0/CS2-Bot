# CS2-Bot

A vision-based FPS aimbot. It reads the screen, finds enemies with a YOLO model, and turns the
camera onto them with synthetic mouse input.

> Software input is detectable. Use a throwaway account, at your own risk.

## Requirements

- Windows 10 (build 2004+) or 11
- Python 3.12
- An NVIDIA GPU with CUDA for real-time inference (CPU works but is slow)
- Trained weights at `runs/detect/train-3/weights/best.pt` (included)

## Setup

```bash
# 1. create the env (Python 3.12)
py -V:3.12 -m venv venv-ml

# 2. install PyTorch first, from the CUDA index (cu128 for RTX 40/50-series)
venv-ml\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. install the rest
venv-ml\Scripts\python.exe -m pip install -r requirements-train.txt bettercam pywin32 opencv-python
```

PyTorch must come from the cu128 index *first* — the default PyPI build crashes with
"no kernel image" on newer cards.

## Run

Put the game on your **primary monitor**, then:

```bash
venv-ml\Scripts\python.exe -m src.bot.livedetect
```

Boxes are drawn in a transparent overlay on top of the game (hidden from screen capture, so it
never feeds back into the detector).

Useful flags:

```bash
--output 1          # capture a different monitor (0 = primary)
--threshold 0.3     # detection confidence (default 0.6)
--no-overlay        # draw in a normal cv2 window instead of the overlay
--no-aim            # detect/draw only, don't move the mouse
```

## Controls

`Z` - toggle auto-aim on/off
`Q` - quit

## Training (optional)

Retrain the detector on your own labelled data:

```bash
venv-ml\Scripts\python.exe -m src.train.train --data data/datasets/cblox/data.yaml --epochs 150
```

Weights land in `runs/detect/<run>/weights/best.pt`. Point `config.WEIGHTS` at the run you want.
