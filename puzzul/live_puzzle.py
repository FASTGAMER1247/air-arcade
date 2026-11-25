"""
Live Puzzle Game
================
A webcam-based puzzle game where you capture a frame, 
divide it into a 3x3 grid, shuffle the tiles, then use 
pinch gestures to drag tiles back into place.

Controls:
  C - Capture frame (or use two-hand frame gesture + pinch)
  R - Reset / play again
  Q - Quit
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
import math
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
from enum import Enum
from collections import deque

# ─── Configuration ────────────────────────────────────────────────────────────
PINCH_THRESHOLD    = 0.05   # 5% of frame width
GRID_SIZE          = 3
NUM_TILES          = GRID_SIZE ** 2
FRAME_WIDTH        = 1280
FRAME_HEIGHT       = 720
TILE_SIZE          = 180
GRID_OFFSET_X      = 350
GRID_OFFSET_Y      = 150
FPS_TARGET         = 30
HAND_CONF          = 0.7
TRACK_CONF         = 0.5
SNAP_THRESHOLD     = 0.7    # 70% of tile size
FRAME_GESTURE_RATIO = 0.4   # 40% of frame width separation


# ─── Enums & Data Classes ─────────────────────────────────────────────────────
class GamePhase(Enum):
    WAITING = 0
    PUZZLE  = 2
    SOLVED  = 3


@dataclass
class Hand:
    landmarks:        np.ndarray
    handedness:       str
    index_finger_tip: Tuple[float, float] = (0, 0)
    thumb_tip:        Tuple[float, float] = (0, 0)
    pinch_distance:   float = 1.0
    is_pinching:      bool  = False
    palm_center:      Tuple[float, float] = (0, 0)
    is_open_palm:     bool  = False


@dataclass
class Tile:
    id:                int
    original_position: Tuple[int, int]
    current_position:  Tuple[int, int]
    image:             np.ndarray
    rect:              Tuple[int, int, int, int] = (0, 0, TILE_SIZE, TILE_SIZE)
    is_dragging:       bool = False
    drag_offset:       Tuple[float, float] = (0.0, 0.0)
    drag_x:            float = 0.0
    drag_y:            float = 0.0


# ─── Helpers ──────────────────────────────────────────────────────────────────
def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def point_in_rect(px, py, rx, ry, rw, rh):
    return rx <= px <= rx+rw and ry <= py <= ry+rh


def grid_to_screen(row, col):
    x = GRID_OFFSET_X + col * TILE_SIZE
    y = GRID_OFFSET_Y + row * TILE_SIZE
    return x, y


def put_text(img, text, pos, scale=0.7, color=(255,255,255), thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


# ─── Main Game Class ───────────────────────────────────────────────────────────
class LivePuzzle:
    def __init__(self):
        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # MediaPipe Tasks API (VIDEO mode avoids async callback deadlocks and reference cycles)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task')
        base_options = mp_python.BaseOptions(model_asset_path=model_path)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=HAND_CONF,
            min_hand_presence_confidence=TRACK_CONF,
            min_tracking_confidence=TRACK_CONF,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.perf_start_time = time.perf_counter()
        self.last_timestamp_ms = -1

        # State
        self.phase:     GamePhase = GamePhase.WAITING
        self.tiles:     List[Tile] = []
        self.start_time: float = 0.0
        self.elapsed:    float = 0.0
        self.palm_hold_start_time: float = 0.0

        # FPS
        self.fps_deque = deque(maxlen=30)
        self.fps_display = 0.0
        self.frame_count = 0

        # Frame gesture debounce
        self.frame_gesture_start = 0.0
        self.capture_cooldown    = 0.0

        # Drag state: only one tile dragged per hand
        self.dragging_tile: Optional[Tile] = None

        self.capture_rect = None

    # ── Tile creation ──────────────────────────────────────────────────────────
    def create_tiles(self, frame: np.ndarray, capture_rect: Tuple[int, int, int]):
        x0, y0, size = capture_rect
        h, w = frame.shape[:2]
        
        x1, y1 = max(0, x0), max(0, y0)
        x2, y2 = min(w, x0 + size), min(h, y0 + size)
        square = frame[y1:y2, x1:x2]
        
        if square.shape[0] != size or square.shape[1] != size:
            square = cv2.resize(square, (size, size))

        cell = size // GRID_SIZE
        self.tiles = []
        positions = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]

        for idx, (row, col) in enumerate(positions):
            patch = square[row*cell:(row+1)*cell, col*cell:(col+1)*cell]
            img   = cv2.resize(patch, (TILE_SIZE, TILE_SIZE))
            t = Tile(
                id=idx,
                original_position=(row, col),
                current_position=(row, col),
                image=img,
            )
            self.tiles.append(t)

        # Shuffle positions
        shuffled = positions.copy()
        while True:
            np.random.shuffle(shuffled)
            if all(shuffled[i] != positions[i] for i in range(NUM_TILES)):
                break
        for i, tile in enumerate(self.tiles):
            tile.current_position = shuffled[i]

        self._update_rects()

    def _update_rects(self):
        for tile in self.tiles:
            if not tile.is_dragging:
                x, y = grid_to_screen(*tile.current_position)
                tile.rect   = (x, y, TILE_SIZE, TILE_SIZE)
                tile.drag_x = float(x)
                tile.drag_y = float(y)

    # ── Snapping ───────────────────────────────────────────────────────────────
    def snap_tile(self, tile: Tile):
        cx = tile.drag_x + TILE_SIZE / 2
        cy = tile.drag_y + TILE_SIZE / 2

        best_dist = float('inf')
        best_pos  = tile.current_position

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                gx, gy = grid_to_screen(row, col)
                gcx    = gx + TILE_SIZE / 2
                gcy    = gy + TILE_SIZE / 2
                d      = dist((cx, cy), (gcx, gcy))
                if d < best_dist:
                    best_dist = d
                    best_pos  = (row, col)

        if best_dist < TILE_SIZE * SNAP_THRESHOLD:
            other = self._tile_at(best_pos, exclude=tile)
            if other:
                other.current_position = tile.current_position
            tile.current_position = best_pos

        tile.is_dragging = False
        self._update_rects()

    def _tile_at(self, pos, exclude=None):
        for t in self.tiles:
            if t is not exclude and t.current_position == pos:
                return t
        return None

    # ── Hand processing ────────────────────────────────────────────────────────
    def process_hands(self, results) -> List[Hand]:
        hands = []
        if not results.hand_landmarks or not results.handedness:
            return hands

        for hand_landmarks, handedness in zip(
            results.hand_landmarks,
            results.handedness
        ):
            if not handedness:
                continue
            lms = np.array([[l.x * FRAME_WIDTH, l.y * FRAME_HEIGHT, l.z]
                            for l in hand_landmarks])

            thumb_tip = (lms[4][0], lms[4][1])
            idx_tip   = (lms[8][0], lms[8][1])
            pd        = dist(thumb_tip, idx_tip)
            pinching  = pd < FRAME_WIDTH * PINCH_THRESHOLD

            palm_ids = [0, 1, 5, 9, 13, 17]
            palm_cx  = float(np.mean([lms[i][0] for i in palm_ids]))
            palm_cy  = float(np.mean([lms[i][1] for i in palm_ids]))

            wrist = lms[0]
            is_open = True
            for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
                tip_dist = dist((wrist[0], wrist[1]), (lms[tip_idx][0], lms[tip_idx][1]))
                pip_dist = dist((wrist[0], wrist[1]), (lms[pip_idx][0], lms[pip_idx][1]))
                if tip_dist < pip_dist:
                    is_open = False
                    break

            hands.append(Hand(
                landmarks=lms,
                handedness=handedness[0].category_name,
                index_finger_tip=idx_tip,
                thumb_tip=thumb_tip,
                pinch_distance=pd,
                is_pinching=pinching,
                palm_center=(palm_cx, palm_cy),
                is_open_palm=is_open,
            ))
        return hands

    def check_frame_gesture(self, hands: List[Hand]) -> bool:
        if len(hands) < 2:
            return False
        xs = [h.palm_center[0] for h in hands]
        return abs(xs[0] - xs[1]) > FRAME_WIDTH * FRAME_GESTURE_RATIO

    # ── Update dragging ────────────────────────────────────────────────────────
    def update_drag(self, hands: List[Hand]):
        any_pinching = any(h.is_pinching for h in hands)

        # If no hand is pinching, drop any dragged tile
        if not any_pinching and self.dragging_tile:
            self.snap_tile(self.dragging_tile)
            self.dragging_tile = None
            return

        for hand in hands:
            px, py = hand.index_finger_tip

            if hand.is_pinching:
                if self.dragging_tile is None:
                    # Try to pick up a tile
                    for tile in self.tiles:
                        rx, ry, rw, rh = tile.rect
                        if point_in_rect(px, py, rx, ry, rw, rh):
                            tile.is_dragging = True
                            tile.drag_offset = (px - tile.drag_x, py - tile.drag_y)
                            self.dragging_tile = tile
                            break
                else:
                    # Move dragged tile
                    tile = self.dragging_tile
                    tile.drag_x = px - tile.drag_offset[0]
                    tile.drag_y = py - tile.drag_offset[1]
                    tile.rect   = (int(tile.drag_x), int(tile.drag_y), TILE_SIZE, TILE_SIZE)
            else:
                if self.dragging_tile and not self.dragging_tile.is_dragging:
                    self.dragging_tile = None

    # ── Solved check ───────────────────────────────────────────────────────────
    def check_solved(self) -> bool:
        return all(t.current_position == t.original_position for t in self.tiles)

    # ── Rendering ──────────────────────────────────────────────────────────────
    def draw_ghost_grid(self, frame):
        overlay = frame.copy()
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x, y = grid_to_screen(row, col)
                cv2.rectangle(overlay, (x, y), (x+TILE_SIZE, y+TILE_SIZE), (60,50,40), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x, y = grid_to_screen(row, col)
                cv2.rectangle(frame, (x, y), (x+TILE_SIZE, y+TILE_SIZE), (100,90,80), 1)

    def draw_tiles(self, frame):
        # Draw non-dragging tiles first, then dragging on top
        for dragging in (False, True):
            for tile in self.tiles:
                if tile.is_dragging != dragging:
                    continue
                rx, ry = int(tile.drag_x), int(tile.drag_y)
                # Clamp to frame
                x1 = max(0, rx);  y1 = max(0, ry)
                x2 = min(FRAME_WIDTH,  rx + TILE_SIZE)
                y2 = min(FRAME_HEIGHT, ry + TILE_SIZE)
                if x2 <= x1 or y2 <= y1:
                    continue
                # Sub-image of tile to blit
                tx1 = x1 - rx;  ty1 = y1 - ry
                tx2 = tx1 + (x2-x1)
                ty2 = ty1 + (y2-y1)
                frame[y1:y2, x1:x2] = tile.image[ty1:ty2, tx1:tx2]
                # Border
                color = (250, 240, 230) if tile.is_dragging else (160, 150, 140)
                thick = 3 if tile.is_dragging else 1
                cv2.rectangle(frame, (rx, ry), (rx+TILE_SIZE, ry+TILE_SIZE), color, thick)

    def draw_hand(self, frame, hand: Hand):
        lms = hand.landmarks
        CONNECTIONS = [
            [0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12],
            [0,13,14,15,16], [0,17,18,19,20], [5,9,13,17]
        ]
        for chain in CONNECTIONS:
            for i in range(len(chain)-1):
                a = (int(lms[chain[i]][0]),   int(lms[chain[i]][1]))
                b = (int(lms[chain[i+1]][0]), int(lms[chain[i+1]][1]))
                cv2.line(frame, a, b, (200, 190, 180), 1)

        for i, lm in enumerate(lms):
            cx, cy = int(lm[0]), int(lm[1])
            color  = (220, 210, 200) if i in (4,8) else (150, 140, 130)
            cv2.circle(frame, (cx,cy), 4, color, -1)

        if hand.is_pinching:
            for tip in (hand.thumb_tip, hand.index_finger_tip):
                cv2.circle(frame, (int(tip[0]), int(tip[1])), 10, (230, 220, 210), 2)

    def draw_ui(self, frame, hands):
        # Cinematic top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 110), (20, 15, 10), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Title
        put_text(frame, "LIVE PUZZLE",
                 (FRAME_WIDTH//2 - 130, 50), 1.5, (230, 220, 210), 2)
        put_text(frame, "Frame it to Snap.  Pinch & Drag to Swap.",
                 (FRAME_WIDTH//2 - 235, 88), 0.5, (180, 170, 160), 1)

        if self.phase == GamePhase.WAITING:
            put_text(frame, "PHASE 1: CAPTURE",
                     (50, 145), 0.9, (200, 190, 180), 2)
            instructions = [
                "Use two hands to define your 3x3 grid",
                "Spread hands and pinch both to capture",
                "-- or --",
                "Press  'C'  to capture now",
            ]
            for i, txt in enumerate(instructions):
                put_text(frame, txt, (55, 185 + i*32), 0.55, (160, 150, 140), 1)

            # Frame-gesture indicator
            if len(hands) == 2 and self.check_frame_gesture(hands):
                put_text(frame, ">> FRAME GESTURE DETECTED - PINCH TO CAPTURE! <<",
                         (FRAME_WIDTH//2 - 310, FRAME_HEIGHT - 80),
                         0.65, (230, 220, 210), 2)

        elif self.phase == GamePhase.PUZZLE:
            put_text(frame, "PHASE 2: PUZZLE",
                     (FRAME_WIDTH - 280, 145), 0.9, (200, 190, 180), 2)
            put_text(frame, f"Time: {int(self.elapsed)}s",
                     (FRAME_WIDTH//2 - 60, 135), 1.0, (230, 220, 210), 2)
            info = ["Pinch to pick up a tile",
                    "Drag & drop to swap",
                    "Press 'R' to reset"]
            for i, txt in enumerate(info):
                put_text(frame, txt, (FRAME_WIDTH-300, 185+i*30), 0.5, (160, 150, 140), 1)

        elif self.phase == GamePhase.SOLVED:
            # Semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (FRAME_WIDTH//2-280, FRAME_HEIGHT//2-80),
                          (FRAME_WIDTH//2+280, FRAME_HEIGHT//2+90), (20, 15, 10), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            put_text(frame, "PUZZLE SOLVED!",
                     (FRAME_WIDTH//2 - 195, FRAME_HEIGHT//2 - 10),
                     1.8, (230, 220, 210), 2)
            put_text(frame, f"Time: {int(self.elapsed)}s",
                     (FRAME_WIDTH//2 - 70, FRAME_HEIGHT//2 + 50),
                     1.1, (200, 190, 180), 2)
                     
            if self.palm_hold_start_time > 0:
                time_held = time.time() - self.palm_hold_start_time
                time_left = max(1, 3 - int(time_held))
                put_text(frame, f"Resetting in {time_left}...",
                         (FRAME_WIDTH//2 - 150, FRAME_HEIGHT//2 + 85),
                         0.8, (0, 255, 255), 2)
            else:
                put_text(frame, "Hold both palms for 3s to reset  |  'Q' for Menu",
                         (FRAME_WIDTH//2 - 250, FRAME_HEIGHT//2 + 85),
                         0.55, (160, 150, 140), 1)

        # FPS
        put_text(frame, f"FPS: {self.fps_display:.0f}",
                 (10, FRAME_HEIGHT - 15), 0.6, (0,220,60), 1)

    # ── Reset ──────────────────────────────────────────────────────────────────
    def reset(self):
        self.phase        = GamePhase.WAITING
        self.tiles        = []
        self.elapsed      = 0.0
        self.dragging_tile = None
        self.palm_hold_start_time = 0.0

    def close(self):
        """Release camera, MediaPipe landmarker, and OpenCV windows."""
        if getattr(self, "landmarker", None) is not None:
            self.landmarker.close()
            self.landmarker = None
        if getattr(self, "cap", None) is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        try:
            if not self.cap or not self.cap.isOpened():
                print("Error: Could not open webcam for Live Puzzle.")
                return

            cv2.namedWindow("Live Puzzle", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Live Puzzle", FRAME_WIDTH, FRAME_HEIGHT)

            t_prev = time.time()

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time.perf_counter() - self.perf_start_time) * 1000)
                if timestamp_ms <= self.last_timestamp_ms:
                    timestamp_ms = self.last_timestamp_ms + 1
                self.last_timestamp_ms = timestamp_ms

                results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
                hands = self.process_hands(results)

                now = time.time()

                # FPS
                self.fps_deque.append(now - t_prev)
                t_prev = now
                self.frame_count += 1
                if self.frame_count % 30 == 0 and self.fps_deque:
                    avg = sum(self.fps_deque) / len(self.fps_deque)
                    self.fps_display = 1.0 / avg if avg > 0 else 0

                # ── Phase logic ─────────────────────────────────────────────────
                if self.phase == GamePhase.WAITING:
                    h, w = frame.shape[:2]
                    if len(hands) == 2:
                        p1 = hands[0].index_finger_tip
                        p2 = hands[1].index_finger_tip
                        cx = (p1[0] + p2[0]) / 2
                        cy = (p1[1] + p2[1]) / 2
                        size = max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
                        size = min(size, min(w, h))
                        x0 = max(0, min(int(cx - size/2), w - int(size)))
                        y0 = max(0, min(int(cy - size/2), h - int(size)))
                        self.capture_rect = (x0, y0, int(size))
                    else:
                        size = min(h, w)
                        x0 = (w - size) // 2
                        y0 = (h - size) // 2
                        self.capture_rect = (x0, y0, size)

                    frame_gesture = self.check_frame_gesture(hands)
                    dual_pinch    = (len(hands) == 2 and
                                     hands[0].is_pinching and hands[1].is_pinching)
                    if frame_gesture and dual_pinch and now > self.capture_cooldown:
                        self.create_tiles(frame, self.capture_rect)
                        self.phase      = GamePhase.PUZZLE
                        self.start_time = time.time()
                        self.capture_cooldown = now + 1.5

                elif self.phase == GamePhase.PUZZLE:
                    self.elapsed = time.time() - self.start_time
                    self.update_drag(hands)
                    if self.dragging_tile is None and self.check_solved():
                        self.phase = GamePhase.SOLVED

                elif self.phase == GamePhase.SOLVED:
                    if len(hands) == 2 and all(h.is_open_palm for h in hands):
                        if self.palm_hold_start_time == 0.0:
                            self.palm_hold_start_time = time.time()
                        elif time.time() - self.palm_hold_start_time >= 3.0:
                            self.reset()
                    else:
                        self.palm_hold_start_time = 0.0

                # ── Render ──────────────────────────────────────────────────────
                if self.phase == GamePhase.WAITING and self.capture_rect and len(hands) == 2:
                    x0, y0, size = self.capture_rect
                    cv2.rectangle(frame, (x0, y0), (x0+size, y0+size), (0, 255, 255), 2)
                    cell = size // GRID_SIZE
                    for i in range(1, GRID_SIZE):
                        cv2.line(frame, (x0 + i*cell, y0), (x0 + i*cell, y0 + size), (0, 255, 255), 1)
                        cv2.line(frame, (x0, y0 + i*cell), (x0 + size, y0 + i*cell), (0, 255, 255), 1)

                if self.phase in (GamePhase.PUZZLE, GamePhase.SOLVED):
                    self.draw_ghost_grid(frame)
                    self.draw_tiles(frame)

                for h in hands:
                    self.draw_hand(frame, h)

                self.draw_ui(frame, hands)

                cv2.imshow("Live Puzzle", frame)

                # ── Keyboard & window-close input ───────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key in (ord('c'), ord('C')) and self.phase == GamePhase.WAITING:
                    if self.capture_rect is None:
                        h, w = frame.shape[:2]
                        size = min(h, w)
                        x0 = (w - size) // 2
                        y0 = (h - size) // 2
                        self.capture_rect = (x0, y0, size)
                    self.create_tiles(frame, self.capture_rect)
                    self.phase      = GamePhase.PUZZLE
                    self.start_time = time.time()
                elif key in (ord('r'), ord('R')):
                    self.reset()

                try:
                    if cv2.getWindowProperty("Live Puzzle", cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.close()


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    game = LivePuzzle()
    game.run()
