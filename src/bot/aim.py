"""The aiming brain.

In this game the cursor is locked to the centre of the screen — you turn by moving the mouse,
not by sliding a pointer around. So to aim, we work out how far the enemy sits from the centre
and send the mouse the right amount to swing the view onto them.

The maths is two short hops, both calibrated for this exact setup rather than guessed:

    pixels off centre  ->  angle  ->  mouse counts

The one real headache is lag. The camera turns instantly, but the next frame from YOLO still
shows the enemy where they were a moment ago. React to that and we move twice and the view
shakes. Two safeguards stop it: we act only once per genuinely new frame, and we hold still for
a few frames after each move so the picture can catch up.

Aiming runs in two modes:
  - OPEN   — enemy far from centre: one decisive flick, then wait.
  - CLOSED — enemy near centre: small nudges to settle onto them, and fire once we're close.

After every move we also slide our lock to where the enemy should appear now the camera has
turned, so we keep tracking the same person instead of chasing where they used to be.

The deeper reasoning lives in bot_vault:
[[open-closed-aim-loops]], [[angle-space-state]], [[aim-sens-calibration]].
"""
from __future__ import annotations

import math
import time

from . import actuator, config


# strip detections down to (x, y, area), filtering by team if set
def _targets(dets) -> list[tuple[float, float, float]]:
    return [(d.cx, d.cy, d.area) for d in dets
            if config.AIM_TARGET_CLASS is None or d.cls == config.AIM_TARGET_CLASS]


# closest candidate to (x, y); squared distance, no sqrt
def _nearest(cands: list[tuple[float, float, float]], x: float,
             y: float) -> tuple[float, float, float]:
    return min(cands, key=lambda c: (c[0] - x) ** 2 + (c[1] - y) ** 2)


class Aimer:

    def __init__(self) -> None:
        # state carried between frames
        self._last_fire = 0.0
        self._acted_frame: int = -1
        self._settle_frames: int = 0
        self._lock: tuple[float, float] | None = None
        self._lost_frames: int = 0
        self._open_snapped: bool = False

    # wipe state clean (called when auto-aim flips ON)
    def reset(self) -> None:
        self._last_fire = 0.0
        self._acted_frame = -1
        self._settle_frames = 0
        self._lock = None
        self._lost_frames = 0
        self._open_snapped = False

    def _select(self, cands: list[tuple[float, float, float]], cx: float,
                cy: float) -> tuple[float, float]:
        # already locked: follow nearest box to last spot if within radius
        if self._lock is not None:
            lx, ly = self._lock
            near = _nearest(cands, lx, ly)
            if (near[0] - lx) ** 2 + (near[1] - ly) ** 2 <= config.AIM_LOCK_RADIUS_PX ** 2:
                self._lost_frames = 0
                self._lock = (near[0], near[1])
                return self._lock
            # nobody near it — hold the lock a few frames in case it's a blink
            self._lost_frames += 1
            if self._lost_frames <= config.AIM_LOST_FRAMES:
                return self._lock

        # no lock (or gave up): grab nearest to crosshair, start fresh
        tx, ty, _ = _nearest(cands, cx, cy)
        self._lock = (tx, ty)
        self._lost_frames = 0
        self._open_snapped = False
        return self._lock

    # re-aim the lock to where the enemy lands after this turn (undo our own swing)
    @staticmethod
    def _post_move_lock(cx: float, cy: float, ax: float, ay: float, dx: int, dy: int,
                        focal_px: float, counts_per_deg: float) -> tuple[float, float]:
        new_ax = math.radians(ax - dx / counts_per_deg)
        new_ay = math.radians(ay - dy / counts_per_deg)
        return cx + focal_px * math.tan(new_ax), cy + focal_px * math.tan(new_ay)

    def update(self, dets, cx: float, cy: float, frame_id: int) -> None:
        # same frame already handled — acting twice double-counts
        if frame_id == self._acted_frame:
            return
        self._acted_frame = frame_id

        # still settling from last move
        if self._settle_frames > 0:
            self._settle_frames -= 1
            return

        # nothing on screen — age the lock, drop it if gone too long
        cands = _targets(dets)
        if not cands:
            if self._lock is not None:
                self._lost_frames += 1
                if self._lost_frames > config.AIM_LOST_FRAMES:
                    self._lock = None
                    self._lost_frames = 0
                    self._open_snapped = False
            return

        # pick enemy, measure error (squared)
        now = time.perf_counter()
        tx, ty = self._select(cands, cx, cy)
        ex = tx - cx
        ey = ty - cy
        err2 = ex * ex + ey * ey

        # pixel error -> angle -> mouse counts
        focal_px = cx / math.tan(math.radians(config.AIM_HFOV_DEG) / 2.0)
        counts_per_deg = config.AIM_COUNTS_PER_DEG
        ax = math.degrees(math.atan(ex / focal_px))
        ay = math.degrees(math.atan(ey / focal_px))
        dx_full = int(round(counts_per_deg * ax))
        dy_full = int(round(counts_per_deg * ay))

        # dead-centre already — just fire
        if err2 <= config.AIM_DEADZONE_PX ** 2:
            self._maybe_fire(err2, now)
            return

        # OPEN: far from centre -> one big snap, re-aim lock, settle
        closed_half = config.AIM_CLOSED_FRAC * cx
        if not self._open_snapped and (abs(ex) > closed_half or abs(ey) > closed_half):
            if config.AIM_DEBUG:
                print(f"[aim] OPEN   detected ax={ax:+.2f} ay={ay:+.2f}deg  ->  move "
                      f"{dx_full / counts_per_deg:+.2f} {dy_full / counts_per_deg:+.2f}deg  "
                      f"(dx={dx_full} dy={dy_full})")
            if dx_full or dy_full:
                actuator.move_rel_steps(dx_full, dy_full, config.AIM_OPEN_MOVE_STEPS,
                                        config.AIM_MOVE_STEP_DELAY_MS / 1000.0)
                self._lock = self._post_move_lock(cx, cy, ax, ay, dx_full, dy_full,
                                                  focal_px, counts_per_deg)
                if config.AIM_OPEN_DELAY_MS > 0:
                    time.sleep(config.AIM_OPEN_DELAY_MS / 1000.0)
            self._open_snapped = True
            self._settle_frames = config.AIM_OPEN_SETTLE_FRAMES
            return

        # CLOSED: near centre -> trim a fraction of the error; full step if it rounds to zero
        self._open_snapped = True
        dx = int(round(config.AIM_TRIM_GAIN * dx_full))
        dy = int(round(config.AIM_TRIM_GAIN * dy_full))
        if not (dx or dy):
            dx, dy = dx_full, dy_full
        if config.AIM_DEBUG:
            print(f"[aim] CLOSED detected ax={ax:+.2f} ay={ay:+.2f}deg  ->  move "
                  f"{dx / counts_per_deg:+.2f} {dy / counts_per_deg:+.2f}deg  "
                  f"(dx={dx} dy={dy})  settle={self._settle_frames}")
        if dx or dy:
            actuator.move_rel_steps(dx, dy, config.AIM_CLOSE_MOVE_STEPS,
                                    config.AIM_MOVE_STEP_DELAY_MS / 1000.0)
            self._lock = self._post_move_lock(cx, cy, ax, ay, dx, dy, focal_px, counts_per_deg)
            self._settle_frames = config.AIM_SETTLE_FRAMES
            return

        # trim rounded to zero — take the shot
        self._maybe_fire(err2, now)

    # tap-fire when on target and cooldown elapsed
    def _maybe_fire(self, err2: float, now: float) -> None:
        if not config.AIM_AUTOFIRE or err2 > config.AIM_FIRE_PX ** 2:
            return
        if (now - self._last_fire) * 1000.0 < config.AIM_FIRE_COOLDOWN_MS:
            return
        actuator.shoot()
        self._last_fire = now
