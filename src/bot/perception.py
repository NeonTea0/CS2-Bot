"""Cleaning up what YOLO sees.

YOLO hands back rough boxes; this turns them into tidy Detection records the rest of the bot can
trust. Just as important, it's the single source of truth — the boxes drawn on screen and the
targets the aimer shoots at come from this one list, so what you see is always what it aims at.

Positions are exactly as captured. We don't adjust for the camera's own motion here; the aimer
takes care of that.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config


# one enemy this frame, in screen px; cx = box centre x, cy = aim point (AIM_AIMPOINT_FRAC down)
@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float
    area: float
    cls: int
    conf: float
    label: str


# YOLO boxes -> Detection list, both teams kept (aimer filters later)
def perceive(boxes, names: dict, w: int, h: int) -> list[Detection]:
    out: list[Detection] = []
    for b in boxes:
        # corners; drop the box if its centre is off-frame
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cx = (x1 + x2) / 2.0
        if not (0.0 <= cx <= w and 0.0 <= (y1 + y2) / 2.0 <= h):
            continue
        # class, confidence, vertical aim point, then bundle up
        cls = int(b.cls)
        conf = float(b.conf)
        cy = y1 + config.AIM_AIMPOINT_FRAC * (y2 - y1)
        out.append(Detection(x1, y1, x2, y2, cx, cy, (x2 - x1) * (y2 - y1), cls, conf,
                             f"{names.get(cls, cls)} {conf:.2f}"))
    return out


# Detections -> (x1, y1, x2, y2, colour, label) tuples for the renderers
def draw_boxes(dets: list[Detection], w: int, h: int) -> list[tuple]:
    drawn: list[tuple] = []
    for d in dets:
        col = config.CLASS_COLORS.get(d.cls, config.FALLBACK_COLOR)
        drawn.append((int(d.x1), int(d.y1), int(d.x2), int(d.y2), col, d.label))
    return drawn
