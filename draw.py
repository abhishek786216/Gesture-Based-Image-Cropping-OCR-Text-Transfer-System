"""
GestureDraw — Draw · Erase · Crop & Save
=========================================
GOAL (Phase 1):
  1. Draw on screen with INDEX finger
  2. Erase ink with ALL 4 fingers (open palm)
  3. Circle/select a region with INDEX+MIDDLE  ->  crop that screen region
     and SAVE it as a PNG in the same folder as this script.

Gestures
--------
  [INDEX only]      -> DRAW  (smooth green ink)
  [INDEX + MIDDLE]  -> SELECT  (close the drawn shape to auto-crop)
  [ALL 4 fingers]   -> ERASE  (palm wipe)
  [FIST / other]    -> STOP  (browse normally)
  C key             -> Clear all ink
  ESC               -> Quit

Output
------
  Cropped images saved as  crop_YYYYMMDD_HHMMSS.png  next to this script.

Dependencies (install once)
----------------------------
  pip install opencv-python mediapipe pyautogui mss numpy
  Download hand_landmarker.task from:
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
  Place it in the same folder as this script.
"""

from __future__ import annotations

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import pyautogui
import mss
import time, os, math, datetime
from collections import deque
from typing import List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — tweak these to suit your setup
# ══════════════════════════════════════════════════════════════════════════════
MODEL_FILENAME   = "hand_landmarker.task"   # must be next to this script
SAVE_FOLDER      = os.path.dirname(os.path.abspath(__file__))

BRUSH_SIZE       = 6          # drawing line thickness (px)
ERASER_RADIUS    = 70         # eraser circle radius (px)
INK_COLOR        = (0, 230, 110)      # BGR — bright green
SELECT_COLOR     = (0, 220, 255)      # BGR — cyan lasso outline
INK_OPACITY      = 0.88               # how visible the ink is (0-1)

EMA_ALPHA        = 0.30       # lower = smoother but laggier pointer
MAX_JUMP_PX      = 100        # ignore jumps larger than this (noise filter)
DEBOUNCE_FRAMES  = 10         # frames before mode switch is confirmed
DRAW_HOLD_FRAMES = 4          # extra frames before pen goes down (prevents ghost dots)
FINGER_UP_THRESH = 0.032      # y-delta to consider a finger "raised"

CLOSE_THRESH_PX  = 100        # how close the last point must be to first to auto-crop
GAP_FILL_PX      = 12         # interpolation step to fill fast-move gaps

PIP_W, PIP_H     = 260, 195   # picture-in-picture webcam preview size

SCREEN_W, SCREEN_H = pyautogui.size()


# ══════════════════════════════════════════════════════════════════════════════
#  MODES
# ══════════════════════════════════════════════════════════════════════════════
DRAW_MODE   = "DRAW"
STOP_MODE   = "STOP"
SELECT_MODE = "SELECT"
ERASE_MODE  = "ERASE"

HUD_LABELS = {
    DRAW_MODE:   "[INDEX only]       DRAW",
    STOP_MODE:   "[FIST]             STOP  -  browse freely",
    SELECT_MODE: "[INDEX+MIDDLE]     SELECT (close shape = CROP & SAVE)",
    ERASE_MODE:  "[ALL 4 fingers]    ERASE",
}
HUD_COLORS = {
    DRAW_MODE:   (0,   255,  80),
    STOP_MODE:   (140, 140, 140),
    SELECT_MODE: (0,   200, 255),
    ERASE_MODE:  (60,   90, 255),
}
GUIDE_LINES = [
    "INDEX only       = DRAW",
    "INDEX+MIDDLE     = SELECT (close=CROP)",
    "ALL 4 fingers    = ERASE",
    "FIST             = STOP / browse",
    "C = clear   ESC = quit",
]

# Hand skeleton connections
HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]


# ══════════════════════════════════════════════════════════════════════════════
#  GESTURE RECOGNISER
# ══════════════════════════════════════════════════════════════════════════════
class GestureRecogniser:
    """
    Classifies hand landmarks into a mode and provides smoothed
    index-fingertip and palm positions in SCREEN coordinates.
    """

    def __init__(self):
        self._buf    = deque(maxlen=DEBOUNCE_FRAMES)
        self._conf   = STOP_MODE
        self._hold   = 0
        self._sx     = -1.0
        self._sy     = -1.0

    # ── public properties ──────────────────────────────────────────────────
    @property
    def mode(self) -> str:
        return self._conf

    @property
    def finger_xy(self) -> Tuple[int, int]:
        if self._sx < 0:
            return (-1, -1)
        return (int(np.clip(self._sx, 0, SCREEN_W - 1)),
                int(np.clip(self._sy, 0, SCREEN_H - 1)))

    # ── update ─────────────────────────────────────────────────────────────
    def update(self, lm) -> str:
        """Call with mediapipe landmark list; returns current mode string."""
        raw  = self._classify(lm)
        mode = self._debounce(raw)

        # Extra stability gate — prevents accidental ink dots
        if mode == DRAW_MODE:
            self._hold += 1
            if self._hold < DRAW_HOLD_FRAMES:
                mode = STOP_MODE
        else:
            self._hold = 0

        # Smooth index fingertip to SCREEN coords
        rx = lm[8].x * SCREEN_W
        ry = lm[8].y * SCREEN_H
        if self._valid(rx, ry):
            self._sx = self._ema(self._sx, rx)
            self._sy = self._ema(self._sy, ry)

        return mode

    def reset(self):
        self._hold = 0
        self._sx = self._sy = -1.0
        self._debounce(STOP_MODE)

    # ── internals ──────────────────────────────────────────────────────────
    def _classify(self, lm) -> str:
        def up(tip, pip):
            return lm[tip].y < lm[pip].y - FINGER_UP_THRESH
        i = up(8, 6);   m = up(12, 10)
        r = up(16, 14); k = up(20, 18)
        if i and m and r and k:              return ERASE_MODE
        if i and not m and not r and not k:  return DRAW_MODE
        if i and m and not r and not k:      return SELECT_MODE
        return STOP_MODE

    def _debounce(self, raw: str) -> str:
        self._buf.append(raw)
        if self._buf.count(raw) >= DEBOUNCE_FRAMES:
            self._conf = raw
        return self._conf

    def _valid(self, nx, ny) -> bool:
        return self._sx < 0 or math.hypot(nx - self._sx, ny - self._sy) < MAX_JUMP_PX

    @staticmethod
    def _ema(prev, new, a=EMA_ALPHA):
        return new if prev < 0 else a * new + (1 - a) * prev


# ══════════════════════════════════════════════════════════════════════════════
#  INK CANVAS
# ══════════════════════════════════════════════════════════════════════════════
class InkCanvas:
    """
    BGRA overlay that lives at screen resolution.
    Handles drawing strokes, erasing, and repaint.
    """

    def __init__(self):
        self.layer: np.ndarray = np.zeros((SCREEN_H, SCREEN_W, 4), dtype=np.uint8)
        self.strokes: List[List[Tuple[int, int]]] = []
        self._cur: List[Tuple[int, int]] = []
        self._drawing = False
        self._last    = (0, 0)

    # ── drawing ────────────────────────────────────────────────────────────
    def start_stroke(self, x: int, y: int):
        self._drawing = True
        self._last    = (x, y)
        self._cur     = [(x, y)]

    def extend_stroke(self, x: int, y: int):
        if not self._drawing:
            return
        self._cur.append((x, y))
        cv2.line(self.layer, self._last, (x, y),
                 (*INK_COLOR, 255), BRUSH_SIZE, cv2.LINE_AA)
        self._last = (x, y)

    def finish_stroke(self) -> Optional[List[Tuple[int, int]]]:
        if not self._drawing:
            return None
        self._drawing = False
        pts = _fill_gaps(self._cur)
        if len(pts) > 1:
            self.strokes.append(pts)
        self._cur = []
        return pts if len(pts) > 1 else None

    @property
    def is_drawing(self) -> bool:
        return self._drawing

    # ── erasing ────────────────────────────────────────────────────────────
    def erase(self, cx: int, cy: int):
        cv2.circle(self.layer, (cx, cy), ERASER_RADIUS, (0, 0, 0, 0), -1)
        self.strokes = [
            s for s in self.strokes
            if not any(math.hypot(p[0] - cx, p[1] - cy) < ERASER_RADIUS for p in s)
        ]
        self._repaint()

    # ── composite ──────────────────────────────────────────────────────────
    def composite(self, screen: np.ndarray) -> np.ndarray:
        """Alpha-blend ink layer over a BGR screen capture."""
        alpha = self.layer[:, :, 3:4].astype(np.float32) / 255.0 * INK_OPACITY
        ink   = self.layer[:, :, :3].astype(np.float32)
        out   = screen.astype(np.float32) * (1 - alpha) + ink * alpha
        return out.astype(np.uint8)

    # ── reset ──────────────────────────────────────────────────────────────
    def clear(self):
        self.layer.fill(0)
        self.strokes.clear()
        self._cur     = []
        self._drawing = False

    # ── all points ─────────────────────────────────────────────────────────
    @property
    def all_points(self) -> List[Tuple[int, int]]:
        return [p for s in self.strokes for p in s]

    # ── internal ───────────────────────────────────────────────────────────
    def _repaint(self):
        self.layer.fill(0)
        for s in self.strokes:
            for i in range(1, len(s)):
                cv2.line(self.layer, s[i - 1], s[i],
                         (*INK_COLOR, 255), BRUSH_SIZE, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  CROP & SAVE
# ══════════════════════════════════════════════════════════════════════════════
def crop_and_save(all_pts: List[Tuple[int, int]], sct: mss.mss) -> Optional[str]:
    """
    Grab the REAL screen pixels inside the convex hull of all_pts,
    save as PNG, return the save path (or None on failure).
    """
    if len(all_pts) < 3:
        print("  [Crop] Not enough points.")
        return None

    arr         = np.array(all_pts, np.int32)
    x, y, w, h = cv2.boundingRect(arr)
    pad         = 6
    x  = max(0, x - pad);  y  = max(0, y - pad)
    w  = min(SCREEN_W - x, w + pad * 2)
    h  = min(SCREEN_H - y, h + pad * 2)

    if w < 5 or h < 5:
        print("  [Crop] Bounding box too small.")
        return None

    # Grab live screen region (NOT the ink layer)
    region = {"top": y, "left": x, "width": w, "height": h}
    shot   = sct.grab(region)
    img    = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

    # Mask to exact convex hull shape
    mask    = np.zeros((h, w), np.uint8)
    shifted = arr - np.array([x, y])
    hull    = cv2.convexHull(shifted)
    cv2.fillPoly(mask, [hull], 255)

    # White background outside the hull
    result     = np.full_like(img, 255)
    result[mask == 255] = img[mask == 255]

    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SAVE_FOLDER, f"crop_{ts}.png")
    cv2.imwrite(filename, result)
    print(f"\n  [Crop] SAVED -> {filename}  ({w}x{h} px)\n")
    return filename


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER
# ══════════════════════════════════════════════════════════════════════════════
def _fill_gaps(pts: List[Tuple[int, int]], gap: int = GAP_FILL_PX) -> List[Tuple[int, int]]:
    """Interpolate points to remove gaps caused by fast hand movement."""
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
class GestureDrawApp:

    WIN = "GestureDraw"

    def __init__(self):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"\n[FATAL] Model not found: {model_path}\n"
                "Download from:\n"
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task\n"
                "and place it next to this script.\n"
            )

        # MediaPipe hand landmarker
        self.detector = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.65,
                min_hand_presence_confidence=0.65,
                min_tracking_confidence=0.65,
            )
        )

        # Webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam (index 0).")

        self.sct        = mss.mss()
        self._mon       = self.sct.monitors[1]
        self.gesture    = GestureRecogniser()
        self.ink        = InkCanvas()
        self._prev_mode = STOP_MODE
        self._last_frame: Optional[np.ndarray] = None

        # Status banner (shown after crop)
        self._banner_text  = ""
        self._banner_until = 0.0

        # Window
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self._set_topmost()

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        self._print_banner()
        while True:
            # ── 1. webcam frame + hand detection ──────────────────────────
            ok, cam_frame = self.cap.read()
            if not ok:
                continue
            cam_frame = cv2.flip(cam_frame, 1)
            self._last_frame = cam_frame
            fh, fw = cam_frame.shape[:2]

            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
            )
            result = self.detector.detect_for_video(mp_img, int(time.time() * 1000))

            mode = STOP_MODE
            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                self._draw_skeleton(cam_frame, lm, fw, fh)
                mode = self.gesture.update(lm)
            else:
                self.gesture.reset()
                if self.ink.is_drawing:
                    self.ink.finish_stroke()

            # ── 2. update ink based on mode ───────────────────────────────
            ix, iy = self.gesture.finger_xy

            if mode == DRAW_MODE and ix >= 0:
                if not self.ink.is_drawing or self._prev_mode != DRAW_MODE:
                    self.ink.start_stroke(ix, iy)
                else:
                    self.ink.extend_stroke(ix, iy)

            elif mode == ERASE_MODE:
                if self.ink.is_drawing:
                    self.ink.finish_stroke()
                # Use palm (midpoint of wrist 0 + palm 9) for erase center
                if result.hand_landmarks:
                    lm = result.hand_landmarks[0]
                    px = int((lm[0].x + lm[9].x) / 2 * SCREEN_W)
                    py = int((lm[0].y + lm[9].y) / 2 * SCREEN_H)
                    px = int(np.clip(px, ERASER_RADIUS, SCREEN_W - ERASER_RADIUS))
                    py = int(np.clip(py, ERASER_RADIUS, SCREEN_H - ERASER_RADIUS))
                    self.ink.erase(px, py)

            elif mode == SELECT_MODE:
                if self.ink.is_drawing:
                    self.ink.finish_stroke()
                # Check if shape is closed -> crop
                if self._prev_mode == DRAW_MODE or self._prev_mode == SELECT_MODE:
                    pts = self.ink.all_points
                    if len(pts) > 10:
                        d = math.hypot(
                            pts[0][0] - pts[-1][0],
                            pts[0][1] - pts[-1][1],
                        )
                        if d < CLOSE_THRESH_PX:
                            # Draw convex hull in cyan before cropping
                            hull = cv2.convexHull(np.array(pts, np.int32))
                            c    = SELECT_COLOR
                            cv2.polylines(self.ink.layer, [hull], True,
                                          (c[0], c[1], c[2], 220), 3)
                            print("  [App] Shape closed — cropping screen region...")
                            path = crop_and_save(pts, self.sct)
                            if path:
                                fname = os.path.basename(path)
                                self._banner_text  = f"SAVED: {fname}"
                                self._banner_until = time.time() + 5
                            self.ink.clear()
            else:
                if self.ink.is_drawing:
                    self.ink.finish_stroke()

            self._prev_mode = mode

            # ── 3. compose display ────────────────────────────────────────
            screen = self._grab_screen()
            disp   = self.ink.composite(screen)
            self._render_hud(disp, mode)
            self._render_cursor(disp, mode, ix, iy,
                                result.hand_landmarks[0] if result.hand_landmarks else None)
            self._render_pip(disp)
            self._render_banner(disp)
            self._render_hint(disp)

            cv2.imshow(self.WIN, disp)

            # ── 4. keyboard ───────────────────────────────────────────────
            k = cv2.waitKey(1) & 0xFF
            if k == 27:          # ESC
                break
            elif k == ord("c"):
                self.ink.clear()
                self._banner_text = ""
                print("  Ink cleared.")

        # cleanup
        self.cap.release()
        self.detector.close()
        self.sct.close()
        cv2.destroyAllWindows()
        print("\nGestureDraw closed. Goodbye!")

    # ── screen grab ───────────────────────────────────────────────────────────
    def _grab_screen(self) -> np.ndarray:
        raw = self.sct.grab(self._mon)
        img = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)
        if img.shape[1] != SCREEN_W or img.shape[0] != SCREEN_H:
            img = cv2.resize(img, (SCREEN_W, SCREEN_H))
        return img

    # ── skeleton overlay on webcam pip ────────────────────────────────────────
    @staticmethod
    def _draw_skeleton(frame, lm, fw, fh):
        pts = [(int(lm[i].x * fw), int(lm[i].y * fh)) for i in range(21)]
        for a, b in HAND_CONN:
            cv2.line(frame, pts[a], pts[b], (0, 180, 0), 1)
        for pt in pts:
            cv2.circle(frame, pt, 3, (0, 255, 0), -1)

    # ── HUD overlay ───────────────────────────────────────────────────────────
    def _render_hud(self, frame, mode):
        col   = HUD_COLORS[mode]
        label = HUD_LABELS[mode]
        ov    = frame.copy()
        cv2.rectangle(ov, (8, 8), (530, 62), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (8, 8), (530, 62), col, 2)
        cv2.putText(frame, label, (16, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Strokes: {len(self.ink.strokes)}",
                    (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1)
        # Guide top-right
        for i, line in enumerate(GUIDE_LINES):
            cv2.putText(frame, line,
                        (SCREEN_W - 380, 20 + i * 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 80, 80), 1)

    # ── cursor ────────────────────────────────────────────────────────────────
    def _render_cursor(self, frame, mode, ix, iy, lm):
        if mode == DRAW_MODE and ix >= 0:
            cv2.circle(frame, (ix, iy), BRUSH_SIZE + 5, (0, 255, 100), 2, cv2.LINE_AA)
            cv2.circle(frame, (ix, iy), 2,              (0, 255, 100), -1)

        elif mode == ERASE_MODE and lm is not None:
            px = int(np.clip((lm[0].x + lm[9].x) / 2 * SCREEN_W,
                             ERASER_RADIUS, SCREEN_W - ERASER_RADIUS))
            py = int(np.clip((lm[0].y + lm[9].y) / 2 * SCREEN_H,
                             ERASER_RADIUS, SCREEN_H - ERASER_RADIUS))
            cv2.circle(frame, (px, py), ERASER_RADIUS, (60, 90, 255), 2)
            cv2.circle(frame, (px, py), 5,              (60, 90, 255), -1)

        elif mode == SELECT_MODE and ix >= 0:
            cv2.circle(frame, (ix, iy), BRUSH_SIZE + 5, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "SELECT", (ix + 12, iy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)

    # ── picture-in-picture webcam ──────────────────────────────────────────────
    def _render_pip(self, frame):
        if self._last_frame is None:
            return
        col = HUD_COLORS[self.gesture.mode]
        pip = cv2.resize(self._last_frame, (PIP_W, PIP_H))
        bx  = SCREEN_W - PIP_W - 14
        by  = SCREEN_H - PIP_H - 14
        cv2.rectangle(frame, (bx - 3, by - 3),
                      (bx + PIP_W + 3, by + PIP_H + 3), (20, 20, 20), 4)
        cv2.rectangle(frame, (bx - 1, by - 1),
                      (bx + PIP_W + 1, by + PIP_H + 1), col, 2)
        frame[by:by + PIP_H, bx:bx + PIP_W] = pip
        cv2.putText(frame, "Camera - show hand here",
                    (bx, by + PIP_H + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 80, 80), 1)

    # ── saved banner ──────────────────────────────────────────────────────────
    def _render_banner(self, frame):
        if not self._banner_text or time.time() >= self._banner_until:
            return
        bw = min(SCREEN_W - 16, 720)
        bh = 52
        bx = 8
        by = SCREEN_H - bh - 50
        roi = frame[by:by + bh, bx:bx + bw].copy()
        dark = np.full_like(roi, 10)
        cv2.addWeighted(dark, 0.82, roi, 0.18, 0, roi)
        frame[by:by + bh, bx:bx + bw] = roi
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 230, 80), 2)
        cv2.putText(frame, self._banner_text,
                    (bx + 12, by + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 230, 80), 2)

    # ── bottom hint ───────────────────────────────────────────────────────────
    def _render_hint(self, frame):
        cv2.putText(
            frame,
            "C = clear ink     ESC = quit     FIST = browse normally",
            (SCREEN_W // 2 - 290, SCREEN_H - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (62, 62, 62), 1,
        )

    # ── always on top (Windows only) ──────────────────────────────────────────
    def _set_topmost(self):
        try:
            import ctypes
            hwnd = ctypes.windll.user32.FindWindowW(None, self.WIN)
            ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0003)
        except Exception:
            pass

    # ── console banner ────────────────────────────────────────────────────────
    @staticmethod
    def _print_banner():
        print("\n" + "=" * 56)
        print("  GestureDraw  -  Phase 1: Draw · Erase · Crop & Save")
        print("=" * 56)
        print("  INDEX finger only   -> DRAW ink on screen")
        print("  INDEX + MIDDLE      -> SELECT  (close shape = CROP)")
        print("  ALL 4 fingers up    -> ERASE ink")
        print("  FIST / other        -> STOP  (browse normally)")
        print("  C = clear ink   ESC = quit")
        print("  Crops saved as:  crop_YYYYMMDD_HHMMSS.png")
        print("=" * 56 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        GestureDrawApp().run()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        input("\nPress Enter to exit.")