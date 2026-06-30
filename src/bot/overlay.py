"""The see-through window that draws boxes over the game.

A borderless, always-on-top window covering the screen. It lets your clicks and key presses
fall straight through to the game underneath, treats pure-black pixels as fully transparent so
only the boxes show, and — the important part — is hidden from every screen-capture path. That
last bit means bettercam, OBS and PrintScreen all see a clean game frame, so the overlay never
feeds its own boxes back into the detector.

Drawing is double-buffered to keep it flicker-free: call draw() each frame with the boxes to
show, and close() when you're done.

Needs pywin32, and Windows 10 build 2004+ for the capture-exclusion (Windows 11 is fine).
"""
from __future__ import annotations

import ctypes

import win32api  # type: ignore
import win32con  # type: ignore
import win32gui  # type: ignore

# WDA_EXCLUDEFROMCAPTURE (win32 magic number): window renders to the screen but is
# omitted from all screen-capture paths. WDA_NONE = 0 would make it capturable again.
_WDA_EXCLUDEFROMCAPTURE = 0x11

# Pixels of this colour become fully transparent (COLORREF black). Detection colours
# must never be pure black or they'd vanish — the palette in livedetect is safe.
_KEY_COLOR = win32api.RGB(0, 0, 0)

_CLASS_NAME = "BotOverlayWnd"

# A box to draw: pixel corners, a BGR colour tuple (matches the cv2 path), and a label.
Box = tuple[int, int, int, int, tuple[int, int, int], str]


def _bgr_to_colorref(bgr: tuple[int, int, int]) -> int:
    b, g, r = bgr
    return win32api.RGB(r, g, b)


class Overlay:
    """A transparent topmost overlay covering a rectangle of the desktop."""

    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        self.width = width
        self.height = height
        hinst = win32api.GetModuleHandle(None)

        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = win32gui.DefWindowProc
        wc.hInstance = hinst
        wc.lpszClassName = _CLASS_NAME
        try:
            win32gui.RegisterClass(wc)
        except win32gui.error:
            pass  # class already registered this process — reuse it

        exstyle = (win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
                   | win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW)
        self.hwnd = win32gui.CreateWindowEx(
            exstyle, _CLASS_NAME, "overlay", win32con.WS_POPUP,
            x, y, width, height, 0, 0, hinst, None)

        # Colour-key transparency: every _KEY_COLOR pixel is see-through.
        win32gui.SetLayeredWindowAttributes(
            self.hwnd, _KEY_COLOR, 0, win32con.LWA_COLORKEY)

        # Hide the overlay from screen capture so it never leaks into a grabbed frame.
        ok = ctypes.windll.user32.SetWindowDisplayAffinity(
            self.hwnd, _WDA_EXCLUDEFROMCAPTURE)
        if not ok:
            print("WARN: SetWindowDisplayAffinity failed — overlay WILL show up in "
                  "captured frames (needs Win10 2004+).")

        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOWNOACTIVATE)
        win32gui.SetWindowPos(self.hwnd, win32con.HWND_TOPMOST, x, y, width, height,
                              win32con.SWP_NOACTIVATE)

    def draw(self, boxes: list[Box]) -> None:
        """Clear the overlay and draw the given boxes. Call once per frame."""
        hdc = win32gui.GetDC(self.hwnd)
        mem = win32gui.CreateCompatibleDC(hdc)
        bmp = win32gui.CreateCompatibleBitmap(hdc, self.width, self.height)
        win32gui.SelectObject(mem, bmp)

        # Fill the buffer with the key colour = transparent background.
        key_brush = win32gui.CreateSolidBrush(_KEY_COLOR)
        win32gui.FillRect(mem, (0, 0, self.width, self.height), key_brush)
        win32gui.DeleteObject(key_brush)

        hollow = win32gui.GetStockObject(win32con.NULL_BRUSH)  # unfilled rectangles
        win32gui.SetBkMode(mem, win32con.TRANSPARENT)          # text bg stays see-through

        for x1, y1, x2, y2, bgr, label in boxes:
            colorref = _bgr_to_colorref(bgr)
            pen = win32gui.CreatePen(win32con.PS_SOLID, 2, colorref)
            old_pen = win32gui.SelectObject(mem, pen)
            old_brush = win32gui.SelectObject(mem, hollow)
            win32gui.Rectangle(mem, x1, y1, x2, y2)
            win32gui.SelectObject(mem, old_pen)
            win32gui.SelectObject(mem, old_brush)
            win32gui.DeleteObject(pen)

            win32gui.SetTextColor(mem, colorref)
            ty = max(y1 - 16, 0)
            win32gui.DrawText(mem, label, -1, (x1, ty, x1 + 240, ty + 16),
                              win32con.DT_LEFT | win32con.DT_TOP
                              | win32con.DT_SINGLELINE | win32con.DT_NOCLIP)

        win32gui.BitBlt(hdc, 0, 0, self.width, self.height, mem, 0, 0, win32con.SRCCOPY)

        win32gui.DeleteObject(bmp)
        win32gui.DeleteDC(mem)
        win32gui.ReleaseDC(self.hwnd, hdc)
        win32gui.PumpWaitingMessages()  # keep the window responsive

    def close(self) -> None:
        win32gui.DestroyWindow(self.hwnd)
