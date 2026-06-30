"""The bot's mouse.

Every camera turn and every shot the bot makes leaves through here. We send input straight to
Windows with SendInput, in relative mode with no acceleration, so the exact movement we ask for
is the movement the game receives — which is what a centre-locked FPS camera needs to aim true.

One caveat: this is software input, and software input can be spotted. The hardware route is
written up in the vault. Until then, play on a throwaway account.
"""
from __future__ import annotations

import ctypes
import random
import time

# Win32 handle + event-flag constants
_user32 = ctypes.windll.user32
_MOUSEEVENTF_MOVE = 0x0001
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
# extra-info field: 64-bit on 64-bit Python, else 32-bit
_ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32

# trigger hold (s) — tiny + randomised so it reads as a tap
_FIRE_HOLD_MIN, _FIRE_HOLD_MAX = 0.01, 0.025


# ctypes structs mirroring Win32 INPUT / MOUSEINPUT so SendInput accepts them
class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", _ULONG_PTR)]


class _INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT)]
    _anonymous_ = ("u",)
    _fields_ = [("type", ctypes.c_ulong), ("u", _U)]


# send one mouse event to Windows
def _send_mouse(flags: int, dx: int = 0, dy: int = 0) -> None:
    inp = _INPUT()
    inp.type = 0
    inp.mi = _MOUSEINPUT(dx, dy, 0, flags, 0, 0)
    _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))


# flick the camera by dx/dy in one shot (loop sets pacing)
def move_rel(dx: int, dy: int) -> None:
    _send_mouse(_MOUSEEVENTF_MOVE, dx, dy)


# glide the move over N hops so it sums exactly to dx/dy with no drift
def move_rel_steps(dx: int, dy: int, steps: int, step_delay: float = 0.0) -> None:
    if steps <= 1:
        move_rel(dx, dy)
        return

    # send only the delta from the running total each hop
    sent_x = sent_y = 0
    for i in range(1, steps + 1):
        want_x = int(round(dx * i / steps))
        want_y = int(round(dy * i / steps))
        step_x, step_y = want_x - sent_x, want_y - sent_y
        if step_x or step_y:
            move_rel(step_x, step_y)
            sent_x, sent_y = want_x, want_y
        # pause between hops (not after the last) to set glide speed
        if step_delay > 0 and i < steps:
            time.sleep(step_delay)


# one click: press, brief randomised hold, release
def shoot() -> None:
    _send_mouse(_MOUSEEVENTF_LEFTDOWN)
    time.sleep(random.uniform(_FIRE_HOLD_MIN, _FIRE_HOLD_MAX))
    _send_mouse(_MOUSEEVENTF_LEFTUP)
