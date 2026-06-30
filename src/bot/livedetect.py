"""The main loop — capture a screen, find enemies, draw them, aim.

Run it from the ML venv (Python 3.12):
    venv-ml\\Scripts\\python.exe -m src.bot.livedetect
    venv-ml\\Scripts\\python.exe -m src.bot.livedetect --output 1 --conf 0.3
    venv-ml\\Scripts\\python.exe -m src.bot.livedetect --no-overlay   (separate cv2 window)

By default the boxes are painted in a transparent, click-through window laid over the game. That
window is hidden from screen capture, so the detector never sees its own boxes bleed into the
next frame. The game has to be on the primary monitor. Press Q anywhere (or Ctrl+C) to quit.

With --no-overlay the boxes go in an ordinary cv2 window instead — drag it to a second screen,
focus it, and press q to quit.

--output chooses which monitor to capture (0 = primary, 1 = second screen, ...). The model knows
two teams, ct and t; deciding which one counts as the enemy is the aimer's job.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import config, perception

# polled globally (GetAsyncKeyState) so they work while the GAME holds focus. Z = aim toggle, Q = quit
_TOGGLE_VK = 0x5A
_QUIT_VK = 0x51


# seconds per loop from config.DETECT_FPS (0 = uncapped)
def _detect_interval() -> float:
    return 1.0 / config.DETECT_FPS if config.DETECT_FPS > 0 else 0.0


# nap any leftover time so the loop holds `interval`; return new reference time
def _throttle(last: float, interval: float) -> float:
    if interval <= 0.0:
        return last
    dt = time.perf_counter() - last
    if dt < interval:
        time.sleep(interval - dt)
    return time.perf_counter()


# a key that flips a flag once per press (not while held)
class _Toggle:

    def __init__(self, vk: int, initial: bool) -> None:
        self._vk = vk
        self.on = initial
        self._was_down = False
        self.just_on = False

    def poll(self) -> bool:
        import win32api  # type: ignore

        # act only on the key-down edge, so one press = one flip
        self.just_on = False
        down = bool(win32api.GetAsyncKeyState(self._vk) & 0x8000)
        if down and not self._was_down:
            self.on = not self.on
            self.just_on = self.on
            print(f"auto-aim {'ON' if self.on else 'OFF'}")
        self._was_down = down
        return self.on


# drive the aimer while on; reset it on the OFF->ON edge
def _aim_tick(aimer, aim_toggle, dets, w: int, h: int, frame_id: int) -> None:
    if aimer is None or not aim_toggle.poll():
        return
    if aim_toggle.just_on:
        aimer.reset()
    aimer.update(dets, w / 2.0, h / 2.0, frame_id)


def _run_overlay(model, names, cam, threshold: float, aimer, aim_toggle) -> None:
    import win32api  # type: ignore

    from .overlay import Overlay

    overlay: Overlay | None = None
    interval = _detect_interval()
    last = time.perf_counter()
    # bumps once per real YOLO frame; the aimer keys off it
    frame_id = 0
    try:
        while True:
            # grab the latest frame (None until a new one is ready)
            frame = cam.grab()
            if frame is None:
                continue
            h, w = frame.shape[:2]

            # build the overlay to match resolution, once
            if overlay is None:
                overlay = Overlay(0, 0, w, h)

            # detect, then convert boxes to the shared Detection list
            results = model(frame, conf=threshold, iou=config.NMS_IOU, agnostic_nms=True,
                            verbose=False)
            frame_id += 1
            dets = perception.perceive(results[0].boxes, names, w, h)

            # draw and aim off the very same detections
            overlay.draw(perception.draw_boxes(dets, w, h))
            _aim_tick(aimer, aim_toggle, dets, w, h, frame_id)

            # Q quits, else hold frame rate
            if win32api.GetAsyncKeyState(_QUIT_VK) & 0x8000:
                break
            last = _throttle(last, interval)
    finally:
        # always tear the overlay down
        if overlay is not None:
            overlay.close()

def live_detect(weights: Path, device: int, output: int, threshold: float, use_aim: bool) -> None:
    import bettercam  # type: ignore
    import torch  # type: ignore
    from ultralytics import YOLO  # type: ignore

    # no weights -> bail with a clear message
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights} — train first.")

    # load model, move to GPU if present (else CPU)
    model = YOLO(str(weights))
    infer_dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(infer_dev)
    gpu_name = torch.cuda.get_device_name(0) if infer_dev != "cpu" else "CPU"
    print(f"inference on {infer_dev} ({gpu_name})")
    names = model.names

    # build the aimer only when aiming — also defers importing the actuator
    aimer = None
    aim_toggle = _Toggle(_TOGGLE_VK, initial=config.AIM_ENABLED)
    if use_aim:
        from .aim import Aimer
        aimer = Aimer()

    # open capture; device_idx = which GPU the monitor hangs off (list via bettercam.output_info())
    cam = bettercam.create(device_idx=device, output_idx=output)
    mode = "overlay (Q to quit)"
    aim_state = f"{'on' if aim_toggle.on else 'off'}, press Z to toggle" if use_aim else "off"
    print(f"capturing device {device} output {output}, threshold {threshold}, classes {names}, "
          f"render={mode}, aim={aim_state}.")
    try:
        _run_overlay(model, names, cam, threshold, aimer, aim_toggle)
    finally:
        del cam


def _main() -> None:
    p = argparse.ArgumentParser(description="Live YOLO detector + aimer on a captured monitor. "
                                            "Defaults come from config.py; flags override.")
    p.add_argument("--weights", type=Path, default=config.WEIGHTS,
                   help="model weights (default from config.py)")
    p.add_argument("--device", type=int, default=config.DEVICE,
                   help="GPU the target monitor is on, bettercam device_idx (0=dGPU, 1=iGPU)")
    p.add_argument("--output", type=int, default=config.OUTPUT,
                   help="output_idx within that device (0=first monitor on it)")
    p.add_argument("--threshold", type=float, default=config.THRESHOLD,
                   help="confidence threshold — only boxes >= this show")
    p.add_argument("--aim", action=argparse.BooleanOptionalAction, default=config.AIM_ENABLED,
                   help="snap the mouse onto the nearest target, then lock + auto-fire. Calibrate "
                        "the counts scale via src.bot.spintest. (--no-aim to force off)")
    args = p.parse_args()
    live_detect(args.weights, args.device, args.output, args.threshold, args.aim)


if __name__ == "__main__":
    _main()
