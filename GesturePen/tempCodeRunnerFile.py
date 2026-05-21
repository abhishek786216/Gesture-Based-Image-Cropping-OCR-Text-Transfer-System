"""
GestureCrop — Draw a lasso with your finger and save the crop
=============================================================
Gestures
--------
  [INDEX only]         -> DRAW  lasso on screen
  [INDEX + MIDDLE]     -> SAVE  crop of the drawn lasso region
  [ALL 4 fingers]      -> CLEAR lasso / cancel
  [FIST / other]       -> STOP  (do nothing / browse)
  ESC                  -> Quit

Output
------
  Saved as  crop_YYYYMMDD_HHMMSS.png  in  ./crops/  next to this script.

Dependencies
------------
  pip install opencv-python mediapipe mss numpy pyautogui
"""

from __future__ import annotations

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import mss
import pyautogui
import time, os, math, datetime
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these if needed
# ══════════════════════════════════════════════════════════════════════════════
MODEL_FILENAME   = "hand_landmarker.task"   # place next to this script

LASSO_COLOR      = (0, 220, 255)            # BGR cyan
LASSO_THICKNESS  = 3
SAVE_FOLDER      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crops")
os.makedirs(SAVE_FOLDER, exist_ok=True)

EMA_ALPHA        = 0.35
MAX_JUMP_PX      = 130
DEBOUNCE_FRAMES  = 10
DRAW_HOLD_FRAMES = 4
FINGER_UP_THRESH = 0.035
GAP_FILL_PX      = 10
PIP_W, PIP_H     = 260, 195

SCREEN_W, SCREEN_H = pyautogui.size()


# ══════════════════════════════════════════════════════════════════════════════
#  HAND SKELETON CONNECTIONS
# ══════════════════════════════════════════════════════════════════════════════
HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

# ══════════════════════════════════════════════════════════════════════════════
#  GESTURE MODES
# ══════════════════════════════════════════════════════════════════════════════
DRAW_MODE   = "DRAW"
SAVE_MODE   = "SAVE"
CLEAR_MODE  = "CLEAR"
STOP_MODE   = "STOP"

HUD_LABELS = {
    DRAW_MODE:  "[INDEX]         DRAW lasso",
    SAVE_MODE:  "[INDEX+MIDDLE]  SAVE crop",
    CLEAR_MODE: "[ALL 4]         CLEAR lasso",
    STOP_MODE:  "[FIST]          STOP",
}
HUD_COLORS = {
    DRAW_MODE:  (0,   220, 255),
    SAVE_MODE:  (0,   230,  80),
    CLEAR_MODE: (60,   90, 255),
    STOP_MODE:  (140, 140, 140),
}


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER — interpolate gaps in lasso
# ══════════════════════════════════════════════════════════════════════════════
def fill_gaps(pts: List[Tuple], gap: int = GAP_FILL_PX) -> List[Tuple]:
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
#  VISION  —  webcam + MediaPipe hand landmarks
# ══════════════════════════════════════════════════════════════════════════════
class VisionAgent:
    def __init__(self, model_path: str):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam (index 0).")

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
        self.last_frame: Optional[np.ndarray] = None
        self.landmarks = None
        self.fw = self.fh = 1

    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)
        self.last_frame = frame
        self.fh, self.fw = frame.shape[:2]

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        result = self.detector.detect_for_video(mp_img, int(time.time() * 1000))

        if result.hand_landmarks:
            self.landmarks = result.hand_landmarks[0]
            self._draw_skeleton(frame, self.landmarks)
        else:
            self.landmarks = None

    def _draw_skeleton(self, frame, lm):
        fw, fh = self.fw, self.fh
        pts = [(int(lm[i].x * fw), int(lm[i].y * fh)) for i in range(21)]
        for a, b in HAND_CONN:
            cv2.line(frame, pts[a], pts[b], (0, 180, 0), 1)
        for pt in pts:
            cv2.circle(frame, pt, 3, (0, 255, 0), -1)

    def release(self):
        self.cap.release()
        self.detector.close()


# ══════════════════════════════════════════════════════════════════════════════
#  GESTURE  —  classify + debounce + smooth fingertip position
# ══════════════════════════════════════════════════════════════════════════════
class GestureAgent:
    def __init__(self):
        self._buf   = deque(maxlen=DEBOUNCE_FRAMES)
        self._mode  = STOP_MODE
        self._hold  = 0
        self._sx = self._sy = -1

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def finger_xy(self) -> Tuple[int, int]:
        if self._sx < 0:
            return (-1, -1)
        return (int(np.clip(self._sx, 0, SCREEN_W - 1)),
                int(np.clip(self._sy, 0, SCREEN_H - 1)))

    def update(self, vision: VisionAgent) -> str:
        lm = vision.landmarks
        if lm is None:
            self._hold = 0
            self._sx = self._sy = -1
            return self._debounce(STOP_MODE)

        raw  = self._classify(lm)
        mode = self._debounce(raw)

        # Extra stability for DRAW — avoids accidental ink on mode transition
        if mode == DRAW_MODE:
            self._hold += 1
            if self._hold < DRAW_HOLD_FRAMES:
                mode = STOP_MODE
        else:
            self._hold = 0

        # Smooth index fingertip → screen coords
        rx = int(lm[8].x * SCREEN_W)
        ry = int(lm[8].y * SCREEN_H)
        if self._valid(rx, ry):
            a = EMA_ALPHA
            self._sx = rx if self._sx < 0 else int(a * rx + (1 - a) * self._sx)
            self._sy = ry if self._sy < 0 else int(a * ry + (1 - a) * self._sy)

        self._mode = mode
        return mode

    def _classify(self, lm) -> str:
        def up(tip, pip):
            return lm[tip].y < lm[pip].y - FINGER_UP_THRESH
        i = up(8, 6);  m = up(12, 10)
        r = up(16, 14); k = up(20, 18)
        if i and m and r and k:              return CLEAR_MODE
        if i and m and not r and not k:      return SAVE_MODE
        if i and not m and not r and not k:  return DRAW_MODE
        return STOP_MODE

    def _debounce(self, raw: str) -> str:
        self._buf.append(raw)
        if self._buf.count(raw) >= DEBOUNCE_FRAMES:
            self._mode = raw
        return self._mode

    def _valid(self, nx, ny) -> bool:
        return self._sx < 0 or math.hypot(nx - self._sx, ny - self._sy) < MAX_JUMP_PX


# ══════════════════════════════════════════════════════════════════════════════
#  LASSO  —  accumulates fingertip points and draws them
# ══════════════════════════════════════════════════════════════════════════════
class LassoAgent:
    def __init__(self):
        self.pts:   List[Tuple[int, int]] = []
        self._last: Optional[Tuple[int, int]] = None
        self._was_drawing = False

    def update(self, mode: str, fx: int, fy: int):
        if mode == DRAW_MODE and fx >= 0:
            if not self._was_drawing:
                # Start new lasso
                self.pts   = [(fx, fy)]
                self._last = (fx, fy)
            else:
                new_pts = fill_gaps([self._last, (fx, fy)])
                self.pts.extend(new_pts[1:])
                self._last = (fx, fy)
            self._was_drawing = True
        else:
            self._was_drawing = False

    def clear(self):
        self.pts   = []
        self._last = None

    def has_lasso(self) -> bool:
        return len(self.pts) > 5

    def draw_on(self, frame: np.ndarray):
        if len(self.pts) < 2:
            return
        arr = np.array(self.pts, np.int32)
        cv2.polylines(frame, [arr], False, LASSO_COLOR, LASSO_THICKNESS, cv2.LINE_AA)
        if len(self.pts) > 10:
            # Closing line hint
            cv2.line(frame, self.pts[-1], self.pts[0],
                     LASSO_COLOR, 1, cv2.LINE_AA)
        # Draw fingertip dot
        cv2.circle(frame, self.pts[-1], 6, LASSO_COLOR, -1)


# ══════════════════════════════════════════════════════════════════════════════
#  CROP & SAVE
# ══════════════════════════════════════════════════════════════════════════════
def crop_and_save(pts: List[Tuple[int, int]], sct: mss.mss) -> Optional[str]:
    if len(pts) < 3:
        return None

    arr        = np.array(pts, np.int32)
    x, y, w, h = cv2.boundingRect(arr)
    pad = 10
    x  = max(0, x - pad);  y  = max(0, y - pad)
    w  = min(SCREEN_W - x, w + pad * 2)
    h  = min(SCREEN_H - y, h + pad * 2)

    if w < 5 or h < 5:
        return None

    # Grab real screen pixels (NOT the overlay window)
    region = {"top": y, "left": x, "width": w, "height": h}
    shot   = sct.grab(region)
    img    = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

    # Mask to convex hull of lasso
    mask    = np.zeros((h, w), np.uint8)
    shifted = arr - np.array([x, y])
    hull    = cv2.convexHull(shifted)
    cv2.fillPoly(mask, [hull], 255)

    result = np.full_like(img, 255)
    result[mask == 255] = img[mask == 255]

    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    path = os.path.join(SAVE_FOLDER, f"crop_{ts}.png")
    cv2.imwrite(path, result)
    print(f"\n  [GestureCrop] SAVED -> {path}  ({w}x{h} px)\n")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
class GestureCropApp:

    WIN = "GestureCrop"

    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"\nModel not found: {model_path}\n"
                "Download from:\n"
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task\n"
                "and place it next to this script.\n"
            )

        self.sct     = mss.mss()
        self._mon    = self.sct.monitors[1]
        self.vision  = VisionAgent(model_path)
        self.gesture = GestureAgent()
        self.lasso   = LassoAgent()

        self._banner_text  = ""
        self._banner_until = 0.0
        self._prev_mode    = STOP_MODE
        self._topmost_done = False

        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.WIN, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        self._print_banner()
        while True:
            # 1. Update vision + gesture
            self.vision.tick()
            mode      = self.gesture.update(self.vision)
            fx, fy    = self.gesture.finger_xy

            # 2. Update lasso
            self.lasso.update(mode, fx, fy)

            # 3. Act on gesture transitions
            if mode == SAVE_MODE and self._prev_mode != SAVE_MODE:
                if self.lasso.has_lasso():
                    path = crop_and_save(self.lasso.pts, self.sct)
                    if path:
                        fname = os.path.basename(path)
                        self._banner_text  = f"SAVED: {fname}"
                        self._banner_until = time.time() + 6
                    else:
                        self._banner_text  = "Crop too small — try again"
                        self._banner_until = time.time() + 3
                    self.lasso.clear()
                else:
                    self._banner_text  = "No lasso drawn — use INDEX to draw first"
                    self._banner_until = time.time() + 3

            elif mode == CLEAR_MODE and self._prev_mode != CLEAR_MODE:
                self.lasso.clear()
                self._banner_text  = "Lasso cleared"
                self._banner_until = time.time() + 2
                print("  [GestureCrop] Lasso cleared.")

            self._prev_mode = mode

            # 4. Composite: grab screen → draw lasso overlay → HUD
            screen = self._grab_screen()
            self.lasso.draw_on(screen)
            self._render_cursor(screen, mode, fx, fy)
            self._render_hud(screen, mode)
            self._render_banner(screen)
            self._render_pip(screen, mode)
            self._render_hint(screen)

            cv2.imshow(self.WIN, screen)

            # Apply always-on-top + exclude from capture after first render
            if not self._topmost_done:
                self._set_topmost()
                self._topmost_done = True

            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break

        self.vision.release()
        self.sct.close()
        cv2.destroyAllWindows()
        print("\nGestureCrop closed. Goodbye!")

    # ── screen grab ───────────────────────────────────────────────────────────
    def _grab_screen(self) -> np.ndarray:
        raw = self.sct.grab(self._mon)
        img = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)
        if img.shape[1] != SCREEN_W or img.shape[0] != SCREEN_H:
            img = cv2.resize(img, (SCREEN_W, SCREEN_H))
        return img

    # ── cursor ────────────────────────────────────────────────────────────────
    def _render_cursor(self, frame, mode, fx, fy):
        if mode == DRAW_MODE and fx >= 0:
            cv2.circle(frame, (fx, fy), 10, LASSO_COLOR, 2, cv2.LINE_AA)
            cv2.circle(frame, (fx, fy), 3,  LASSO_COLOR, -1)

    # ── HUD bar (top-left) ────────────────────────────────────────────────────
    def _render_hud(self, frame, mode):
        col   = HUD_COLORS[mode]
        label = HUD_LABELS[mode]
        ov    = frame.copy()
        cv2.rectangle(ov, (8, 8), (540, 62), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (8, 8), (540, 62), col, 2)
        cv2.putText(frame, label, (16, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.76, col, 2, cv2.LINE_AA)
        pts_count = len(self.lasso.pts)
        cv2.putText(frame, f"Lasso points: {pts_count}",
                    (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (85, 85, 85), 1)

    # ── saved / status banner (bottom-left) ───────────────────────────────────
    def _render_banner(self, frame):
        if not self._banner_text or time.time() >= self._banner_until:
            return
        bw, bh = min(SCREEN_W - 16, 700), 52
        bx, by = 8, SCREEN_H - bh - 50
        roi    = frame[by:by + bh, bx:bx + bw].copy()
        dark   = np.full_like(roi, 10)
        cv2.addWeighted(dark, 0.82, roi, 0.18, 0, roi)
        frame[by:by + bh, bx:bx + bw] = roi
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 230, 80), 2)
        cv2.putText(frame, self._banner_text, (bx + 12, by + 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 80), 2)

    # ── PiP camera (bottom-right) ─────────────────────────────────────────────
    def _render_pip(self, frame, mode):
        cam = self.vision.last_frame
        if cam is None:
            return
        col  = HUD_COLORS[mode]
        pip  = cv2.resize(cam, (PIP_W, PIP_H))
        bx   = SCREEN_W - PIP_W - 14
        by   = SCREEN_H - PIP_H - 14
        cv2.rectangle(frame, (bx - 2, by - 2),
                      (bx + PIP_W + 2, by + PIP_H + 2), (20, 20, 20), 3)
        cv2.rectangle(frame, (bx - 1, by - 1),
                      (bx + PIP_W + 1, by + PIP_H + 1), col, 2)
        frame[by:by + PIP_H, bx:bx + PIP_W] = pip
        cv2.putText(frame, "Camera feed",
                    (bx, by + PIP_H + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (80, 80, 80), 1)

    # ── gesture guide (top-right) ─────────────────────────────────────────────
    def _render_hint(self, frame):
        guide = [
            "INDEX only      = DRAW lasso",
            "INDEX + MIDDLE  = SAVE crop",
            "ALL 4 fingers   = CLEAR lasso",
            "FIST            = STOP",
        ]
        for i, line in enumerate(guide):
            cv2.putText(frame, line,
                        (SCREEN_W - 360, 22 + i * 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, (75, 75, 75), 1)

        cv2.putText(frame,
                    "ESC = quit     crops saved in ./crops/",
                    (SCREEN_W // 2 - 210, SCREEN_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (65, 65, 65), 1)

    # ── Windows: always-on-top + exclude from screen capture ─────────────────
    def _set_topmost(self):
        try:
            import ctypes
            hwnd = ctypes.windll.user32.FindWindowW(None, self.WIN)
            # HWND_TOPMOST=-1, SWP_NOMOVE|SWP_NOSIZE=0x0003
            ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0003)
            # WDA_EXCLUDEFROMCAPTURE so the overlay is invisible to mss grabs
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x00000011)
            print("  [Init] Window set: always-on-top, excluded from capture.")
        except Exception as e:
            print(f"  [Init] Window flags not applied ({e}) — safe to continue.")

    @staticmethod
    def _print_banner():
        print("\n" + "=" * 55)
        print("  GestureCrop  —  Draw & Save with your hand")
        print("=" * 55)
        print("  INDEX only      -> Draw lasso on screen")
        print("  INDEX + MIDDLE  -> Save the crop")
        print("  ALL 4 fingers   -> Clear / cancel lasso")
        print("  FIST            -> Stop (do nothing)")
        print("  ESC             -> Quit")
        print(f"  Crops saved in: ./crops/")
        print("=" * 55 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        GestureCropApp().run()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        input("\nPress Enter to exit.")