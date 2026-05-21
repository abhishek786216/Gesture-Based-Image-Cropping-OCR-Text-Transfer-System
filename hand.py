"""
MouseCrop — Crop & Save  (Mouse / Touchpad version)
===================================================================
No camera, no MediaPipe — just your mouse or touchpad.

Controls
--------
  LEFT CLICK + DRAG          -> Draw a lasso to select a screen region
  RELEASE MOUSE BUTTON       -> Crop and save the selected region
  ESC                        -> Quit

Output
------
  Cropped images saved as  crop_YYYYMMDD_HHMMSS.png  next to this script.

Dependencies
------------
  pip install opencv-python mss numpy pyautogui
"""

from __future__ import annotations

import cv2
import numpy as np
import mss
import pyautogui
import os, time, datetime
from typing import List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
SELECT_COLOR   = (0, 210, 255)       # BGR cyan
GAP_FILL_PX    = 8                   # interpolate fast drags

SAVE_FOLDER    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputfolder")
os.makedirs(SAVE_FOLDER, exist_ok=True)
SCREEN_W, SCREEN_H = pyautogui.size()

# ══════════════════════════════════════════════════════════════════════════════
#  MODES
# ══════════════════════════════════════════════════════════════════════════════
HINT = "Drag to select | Release to Save | ESC=Quit"


# ══════════════════════════════════════════════════════════════════════════════
#  SELECT LASSO  (separate thin overlay, not permanent ink)
# ══════════════════════════════════════════════════════════════════════════════
class SelectLasso:
    def __init__(self):
        self.pts: List[Tuple[int, int]] = []
        self._last: Optional[Tuple[int, int]] = None

    def start(self, x, y):
        self.pts   = [(x, y)]
        self._last = (x, y)

    def extend(self, x, y):
        if self._last is None:
            return
        filled = _fill_gaps([self._last, (x, y)])
        self.pts.extend(filled[1:])
        self._last = (x, y)

    def finish(self) -> Optional[List[Tuple[int, int]]]:
        pts        = list(self.pts)
        self.pts   = []
        self._last = None
        return pts if len(pts) > 5 else None

    def draw_overlay(self, frame):
        """Draw the live lasso in cyan on the display frame."""
        if len(self.pts) < 2:
            return
        arr = np.array(self.pts, np.int32)
        cv2.polylines(frame, [arr], False, SELECT_COLOR, 2, cv2.LINE_AA)
        # closing line hint
        if len(self.pts) > 2:
            cv2.line(frame, self.pts[-1], self.pts[0], SELECT_COLOR, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  CROP & SAVE
# ══════════════════════════════════════════════════════════════════════════════
def crop_and_save(pts: List[Tuple[int, int]], sct: mss.mss) -> Optional[str]:
    if len(pts) < 3:
        return None

    arr         = np.array(pts, np.int32)
    x, y, w, h  = cv2.boundingRect(arr)
    pad = 8
    x  = max(0, x - pad);  y  = max(0, y - pad)
    w  = min(SCREEN_W - x, w + pad * 2)
    h  = min(SCREEN_H - y, h + pad * 2)

    if w < 5 or h < 5:
        return None

    region = {"top": y, "left": x, "width": w, "height": h}
    shot   = sct.grab(region)
    img    = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

    # Mask to drawn lasso shape
    mask    = np.zeros((h, w), np.uint8)
    shifted = arr - np.array([x, y])
    hull    = cv2.convexHull(shifted)
    cv2.fillPoly(mask, [hull], 255)

    result = np.full_like(img, 255)
    result[mask == 255] = img[mask == 255]

    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path     = os.path.join(SAVE_FOLDER, f"crop_{ts}.png")
    cv2.imwrite(path, result)
    print(f"\n  [Crop] SAVED -> {path}  ({w}x{h} px)\n")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER
# ══════════════════════════════════════════════════════════════════════════════
def _fill_gaps(pts, gap=GAP_FILL_PX):
    if len(pts) < 2:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts)):
        p0 = np.array(pts[i - 1], float)
        p1 = np.array(pts[i],     float)
        d  = np.linalg.norm(p1 - p0)
        if d > gap:
            for t in np.linspace(0, 1, int(d / gap) + 1)[1:]:
                out.append(tuple((p0 + t * (p1 - p0)).astype(int)))
        else:
            out.append(pts[i])
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
class MouseCropApp:

    WIN = "MouseCrop"

    def __init__(self):
        self.sct    = mss.mss()
        self._mon   = self.sct.monitors[1]
        self.lasso  = SelectLasso()

        self._mouse_down = False
        self._mx = 0
        self._my = 0

        self._banner_text  = ""
        self._banner_until = 0.0

        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(self.WIN, self._on_mouse)
        self._topmost_applied = False   # applied after first frame renders

    # ── mouse callback ────────────────────────────────────────────────────────
    def _on_mouse(self, event, x, y, flags, _param):
        self._mx, self._my = x, y

        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_down = True
            self.lasso.start(x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self._mouse_down:
            self.lasso.extend(x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self._mouse_down = False
            pts = self.lasso.finish()
            if pts:
                path = crop_and_save(pts, self.sct)
                if path:
                    fname = os.path.basename(path)
                    self._banner_text  = f"SAVED: {fname}"
                    self._banner_until = time.time() + 6

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        self._print_banner()
        while True:
            disp = self._grab_screen()

            # draw live lasso on top
            self.lasso.draw_overlay(disp)

            self._render_hud(disp)
            self._render_cursor(disp)
            self._render_banner(disp)
            self._render_hint(disp)

            cv2.imshow(self.WIN, disp)

            # Apply after first frame so the HWND actually exists
            if not self._topmost_applied:
                self._set_topmost()
                self._topmost_applied = True

            k = cv2.waitKey(1) & 0xFF
            if k == 27:                     # ESC
                break

        self.sct.close()
        cv2.destroyAllWindows()
        print("\nMouseCrop closed. Goodbye!")

    # ── screen grab ───────────────────────────────────────────────────────────
    def _grab_screen(self) -> np.ndarray:
        raw = self.sct.grab(self._mon)
        img = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)
        if img.shape[1] != SCREEN_W or img.shape[0] != SCREEN_H:
            img = cv2.resize(img, (SCREEN_W, SCREEN_H))
        return img

    # ── HUD ───────────────────────────────────────────────────────────────────
    def _render_hud(self, frame):
        col   = SELECT_COLOR
        label = "SELECT (drag lasso -> release = CROP & SAVE)"
        ov    = frame.copy()
        cv2.rectangle(ov, (8, 8), (620, 62), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (8, 8), (620, 62), col, 2)
        cv2.putText(frame, label, (16, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, col, 2, cv2.LINE_AA)

    # ── cursor ────────────────────────────────────────────────────────────────
    def _render_cursor(self, frame):
        x, y = self._mx, self._my
        cv2.drawMarker(frame, (x, y), SELECT_COLOR,
                        cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)

    # ── saved banner ──────────────────────────────────────────────────────────
    def _render_banner(self, frame):
        if not self._banner_text or time.time() >= self._banner_until:
            return
        bw, bh = min(SCREEN_W - 16, 760), 52
        bx, by = 8, SCREEN_H - bh - 50
        roi  = frame[by:by + bh, bx:bx + bw].copy()
        dark = np.full_like(roi, 10)
        cv2.addWeighted(dark, 0.82, roi, 0.18, 0, roi)
        frame[by:by + bh, bx:bx + bw] = roi
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 230, 80), 2)
        cv2.putText(frame, self._banner_text, (bx + 12, by + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 230, 80), 2)

    # ── hint bar ──────────────────────────────────────────────────────────────
    def _render_hint(self, frame):
        cv2.putText(frame, HINT,
                    (SCREEN_W // 2 - 150, SCREEN_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (70, 70, 70), 1)

    # ── always on top + exclude from screen capture (Windows) ───────────────
    def _set_topmost(self):
        try:
            import ctypes
            hwnd = ctypes.windll.user32.FindWindowW(None, self.WIN)
            # Always on top
            ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0003)
            # WDA_EXCLUDEFROMCAPTURE (0x11) — makes THIS window invisible to
            # mss / BitBlt screen grabs so we never capture our own overlay.
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x00000011)
            print("  [Init] Window excluded from screen capture.")
        except Exception as e:
            print(f"  [Init] SetWindowDisplayAffinity not available: {e}")
            print("  [Init] Falling back: grabbing non-primary monitor or desktop.")

    @staticmethod
    def _print_banner():
        print("\n" + "=" * 56)
        print("  MouseCrop  -  Crop & Save")
        print("=" * 56)
        print("  Left-drag      -> Draw a lasso to select a region")
        print("  Release mouse  -> Crop and save the selection")
        print("  ESC            -> Quit")
        print("  Crops saved as:  crop_YYYYMMDD_HHMMSS.png")
        print("=" * 56 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        MouseCropApp().run()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        input("\nPress Enter to exit.")