"""Every setting in one place.

These are the defaults the detector, overlay and aimer start from. Anything you pass on the
livedetect command line overrides what's here, so treat this as the sensible baseline rather
than the last word.
"""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

# Model weights (train-3 run, 2-class ct/t).
WEIGHTS: Path = _ROOT / "runs" / "detect" / "train-3" / "weights" / "best.pt"

# Capture target (bettercam). DEVICE = GPU the monitor hangs off (0=dGPU, 1=iGPU).
# OUTPUT = which monitor on that device (0 = first / primary).
DEVICE: int = 0
OUTPUT: int = 0

THRESHOLD: float = 0.6      # min confidence to show a box
NMS_IOU: float = 0.45  # merge boxes overlapping more than this (agnostic across ct/t)

# Cap the capture->YOLO loop rate. 0 = uncapped. The aimer paces off detections, so a low cap
# also slows re-aiming.
DETECT_FPS: int = 0

# True = transparent capture-excluded overlay on the game; False = separate cv2 window.
OVERLAY: bool = True

# Box colours (BGR) by class id. train-3 data.yaml order: 0 = ct, 1 = t.
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 128, 0),   # ct = blue-ish
    1: (0, 128, 255),   # t  = orange
}
FALLBACK_COLOR: tuple[int, int, int] = (0, 255, 0)

# --- Aim ------------------------------------------------------------------------------------
AIM_ENABLED: bool = True
# Class id to aim at (0=ct, 1=t); None = any class. Pick is always nearest the crosshair.
AIM_TARGET_CLASS: int | None = None

# Your in-game sensitivity. This is the ONE number you change per setup — everything else about
# how far the mouse moves is worked out from it. Higher sens means fewer counts per degree turned.
AIM_MOUSE_SENS: float = 3.15
# K ties counts, degrees, and sens together (K = counts/degree x sens) and stays the same no matter
# what your sens is. We measured it once with `spintest`; only redo that if the hardware or the
# game's look-mapping changes — NOT when you tweak sens above.
AIM_K: float = 45.52
# derived from the two above — leave it alone
AIM_COUNTS_PER_DEG: float = AIM_K / AIM_MOUSE_SENS

# Vertical aim point in the box: 0.0 = head (top), 0.5 = body (centre), 1.0 = feet.
AIM_AIMPOINT_FRAC: float = 0.5
# Within this many px of centre -> on target, send nothing (else it twitches forever).
AIM_DEADZONE_PX: int = 6
# Frames to skip after a CLOSED trim (on top of the new-frame gate) so the box catches up.
AIM_SETTLE_FRAMES: int = 1

# Sticky lock: keep the box within this radius of the last target instead of re-picking nearest
# every frame. Big enough to cover how far a target moves between frames.
AIM_LOCK_RADIUS_PX: float = 150.0
# Frames the locked target may stay missing before the lock drops and we re-acquire.
AIM_LOST_FRAMES: int = 8

# Horizontal FOV. Focal length is derived live from the captured frame WIDTH in aim.py
# (f = (frame_w/2)/tan(HFOV/2)), so px->angle (atan(e/f)) holds at any capture resolution.
AIM_HFOV_DEG: float = 105.41  # VERIFIED by angletest (2-sample fit, f=731px @1920w)

# CLOSED-loop zone = square centred on the crosshair, half-side = AIM_CLOSED_FRAC * cx.
# Outside it a target gets ONE open-loop angle snap per lock; inside it the closed loop trims.
AIM_CLOSED_FRAC: float = 1.0 / 3.0
# Blocking pause (ms) after the one-shot open snap so the big turn finishes before the next grab.
AIM_OPEN_DELAY_MS: float = 30.0
# Frames to skip after the open snap before closed may act. Must outlast capture+YOLO latency so
# closed reads a POST-move detection, not the pre-snap ghost. Raise if closed re-reads the snap angle.
AIM_OPEN_SETTLE_FRAMES: int = 3
# Glide: split each move into N raw sub-steps with AIM_MOVE_STEP_DELAY_MS between them (1 = teleport).
AIM_OPEN_MOVE_STEPS: int = 4
AIM_CLOSE_MOVE_STEPS: int = 2
AIM_MOVE_STEP_DELAY_MS: float = 3.0
# Fraction of the residual the closed loop moves each settled tick (0.5 = halve the error / tick).
AIM_TRIM_GAIN: float = 0.5

# Print per-tick aim math (error, sent dx/dy). Set False once it's tracking.
AIM_DEBUG: bool = True

# --- Auto-fire ------------------------------------------------------------------------------
AIM_AUTOFIRE: bool = True
AIM_FIRE_PX: int = 10              # fire only when within this many px of the target centre
AIM_FIRE_COOLDOWN_MS: float = 120.0  # min gap between shots (taps like a human, doesn't hold)
