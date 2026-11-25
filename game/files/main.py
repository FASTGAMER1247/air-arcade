"""
Air Hockey - 2 Player Air Hockey via Webcam Hand Tracking
Controls: Left hand = Player 1 (left paddle), Right hand = Player 2 (right paddle)
Index finger tip controls the paddle position
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request
import pygame
import numpy as np
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ─────────────────────────── CONFIG ───────────────────────────
GAME_W, GAME_H = 1280, 720
CAM_W, CAM_H   = 640, 360          # webcam display size (top strip)
PADDLE_R        = 40
BALL_R          = 20
BALL_SPEED_INIT = 80
BALL_SPEED_MAX  = 380
BALL_SPEED_INC  = 0                # speed-up per paddle hit
WIN_SCORE       = 7

# Neon palette
C_BG        = (5,   5,  15)
C_P1        = (0,  220, 255)       # cyan
C_P2        = (255,  60, 200)      # magenta
C_BALL      = (255, 240,  60)      # yellow
C_WHITE     = (255, 255, 255)
C_DARK      = (20,  20,  40)
C_GOAL_LINE = (80,  80, 120)


# ─────────────────────────── GLOW HELPER ──────────────────────
_glow_cache = {}

def draw_glow_circle(surf: pygame.Surface, color, center, radius, layers=4, alpha_step=40):
    key = (color, radius, layers, alpha_step)
    if key not in _glow_cache:
        total_r = radius + layers * 6 + 2
        glow_surf = pygame.Surface((total_r * 2, total_r * 2), pygame.SRCALPHA)
        for i in range(layers, 0, -1):
            r_off = i * 6
            alpha = alpha_step
            pygame.draw.circle(glow_surf, (*color, alpha),
                               (total_r, total_r),
                               radius + r_off)
        pygame.draw.circle(glow_surf, color, (total_r, total_r), radius)
        # bright core
        core_r = max(radius // 3, 4)
        bright = tuple(min(255, c + 100) for c in color)
        pygame.draw.circle(glow_surf, bright, (total_r, total_r), core_r)
        _glow_cache[key] = (glow_surf, total_r)
    
    glow_surf, total_r = _glow_cache[key]
    cx, cy = int(center[0]), int(center[1])
    surf.blit(glow_surf, (cx - total_r, cy - total_r))


def draw_glow_line(surf, color, p1, p2, width=3, layers=3):
    for i in range(layers, 0, -1):
        alpha = 30
        w = width + i * 4
        glow = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
        pygame.draw.line(glow, (*color, alpha), p1, p2, w)
        surf.blit(glow, (0, 0))
    pygame.draw.line(surf, color, p1, p2, width)


def draw_glow_rect(surf, color, rect, width=2, layers=3):
    for i in range(layers, 0, -1):
        alpha = 25
        r = pygame.Rect(rect.x - i*3, rect.y - i*3,
                        rect.w + i*6, rect.h + i*6)
        glow = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
        pygame.draw.rect(glow, (*color, alpha), r, width + i*2, border_radius=12)
        surf.blit(glow, (0, 0))
    pygame.draw.rect(surf, color, rect, width, border_radius=8)


# ─────────────────────────── PARTICLE ─────────────────────────
@dataclass
class Particle:
    x: float; y: float
    vx: float; vy: float
    life: float; max_life: float
    color: Tuple[int,int,int]
    size: float

    def update(self, dt):
        self.x  += self.vx * dt
        self.y  += self.vy * dt
        self.vy += 80 * dt          # gravity
        self.life -= dt

    @property
    def alive(self): return self.life > 0

    def draw(self, surf):
        if not self.alive: return
        s = max(1, int(self.size * (self.life / self.max_life)))
        pygame.draw.circle(surf, self.color, (int(self.x), int(self.y)), s)


def spawn_particles(particles, x, y, color, n=20, speed=200):
    for _ in range(n):
        angle = np.random.uniform(0, math.tau)
        spd   = np.random.uniform(speed*0.3, speed)
        life  = np.random.uniform(0.4, 0.9)
        particles.append(Particle(x, y,
                                  math.cos(angle)*spd, math.sin(angle)*spd,
                                  life, life, color,
                                  np.random.uniform(3, 8)))


# ─────────────────────────── BALL ─────────────────────────────
class Ball:
    def __init__(self):
        self.reset()

    def reset(self, direction=1):
        self.x  = GAME_W / 2
        self.y  = GAME_H / 2
        self.vx = 0
        self.vy = 0
        self.speed = 0
        self.trail: list[Tuple[float,float]] = []

    def update(self, dt, particles):
        # trail
        self.trail.append((self.x, self.y))
        if len(self.trail) > 18:
            self.trail.pop(0)

        self.x += self.vx * dt
        self.y += self.vy * dt

        # top / bottom walls
        if self.y - BALL_R < 0:
            self.y  = BALL_R
            self.vy = abs(self.vy)
            spawn_particles(particles, self.x, self.y, C_BALL, 8, 120)
        if self.y + BALL_R > GAME_H:
            self.y  = GAME_H - BALL_R
            self.vy = -abs(self.vy)
            spawn_particles(particles, self.x, self.y, C_BALL, 8, 120)

    def draw(self, surf):
        # trail
        for i, (tx, ty) in enumerate(self.trail):
            frac  = i / len(self.trail)
            r     = max(2, int(BALL_R * frac * 0.7))
            c     = tuple(int(col * frac) for col in C_BALL)
            pygame.draw.circle(surf, c, (int(tx), int(ty)), r)
        draw_glow_circle(surf, C_BALL, (self.x, self.y), BALL_R, layers=5)


# ─────────────────────────── PADDLE ───────────────────────────
class Paddle:
    def __init__(self, x, y, color, side):
        self.x = x; self.y = y
        self.color = color
        self.side  = side   # 'left' or 'right'
        self.vx = 0; self.vy = 0
        self.prev_x = x; self.prev_y = y
        self.hit_flash = 0.0

    def move_to(self, tx, ty, dt):
        if dt <= 0: dt = 0.001
        # clamp to own half
        if self.side == 'left':
            tx = np.clip(tx, PADDLE_R, GAME_W//2 - PADDLE_R)
        else:
            tx = np.clip(tx, GAME_W//2 + PADDLE_R, GAME_W - PADDLE_R)
        ty = np.clip(ty, PADDLE_R, GAME_H - PADDLE_R)
        
        self.vx = (tx - self.x) / dt
        self.vy = (ty - self.y) / dt
        self.x, self.y = tx, ty

    def check_ball_collision(self, ball: Ball, particles):
        dx = ball.x - self.x
        dy = ball.y - self.y
        dist = math.hypot(dx, dy)
        min_dist = PADDLE_R + BALL_R
        if dist < min_dist and dist > 0:
            nx, ny = dx/dist, dy/dist
            # push ball out to prevent sticking
            overlap = min_dist - dist + 1
            ball.x += nx * overlap
            ball.y += ny * overlap
            
            # relative velocity
            rvx = ball.vx - self.vx
            rvy = ball.vy - self.vy
            vel_along_normal = rvx * nx + rvy * ny
            
            if vel_along_normal < 0:
                # perfectly elastic collision against infinite mass paddle
                j = -2 * vel_along_normal
                ball.vx += j * nx
                ball.vy += j * ny
                
                # Add some speed bump and enforce speed limits
                cur = math.hypot(ball.vx, ball.vy)
                if cur > 0:
                    target_speed = max(BALL_SPEED_INIT, min(cur + BALL_SPEED_INC, BALL_SPEED_MAX))
                    scale = target_speed / cur
                    ball.vx *= scale
                    ball.vy *= scale
            
            self.hit_flash = 0.4
            spawn_particles(particles, ball.x, ball.y, self.color, 15, 180)
            return True
        return False

    def draw(self, surf):
        color = self.color
        if self.hit_flash > 0:
            t = self.hit_flash / 0.4
            color = tuple(min(255, int(c + (255-c)*t)) for c in color)
            self.hit_flash = max(0, self.hit_flash - 0.02)
        draw_glow_circle(surf, color, (self.x, self.y), PADDLE_R, layers=5)
        # ring
        pygame.draw.circle(surf, C_WHITE, (int(self.x), int(self.y)), PADDLE_R, 2)


# ─────────────────────────── HAND TRACKER ─────────────────────
import threading

class HandTracker:
    def __init__(self):
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading Hand Landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO)
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.left_pos:  Optional[Tuple[float,float]] = None
        self.right_pos: Optional[Tuple[float,float]] = None
        self.frame_rgb: Optional[np.ndarray] = None
        self._start_time = time.time()
        self.running = True
        
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - self._start_time) * 1000)
            results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            
            left = None
            right = None
            if results.hand_landmarks:
                # Collect all detected fingertip positions
                candidates = []
                for hand_landmarks in results.hand_landmarks:
                    tip = hand_landmarks[8]  # index finger tip
                    gx = tip.x * GAME_W
                    gy = tip.y * GAME_H
                    candidates.append((gx, gy))

                # After cv2.flip the image is a mirror, so screen position
                # directly reflects which side each player's hand is on.
                # Hands on LEFT half -> P1, hands on RIGHT half -> P2
                for gx, gy in candidates:
                    if gx < GAME_W / 2:
                        if left is None or abs(gx - GAME_W/2) < abs(left[0] - GAME_W/2):
                            left = (gx, gy)
                    else:
                        if right is None or abs(gx - GAME_W/2) < abs(right[0] - GAME_W/2):
                            right = (gx, gy)

            # alpha=0.75: responsive but stable
            alpha = 0.75
            if left:
                if self.left_pos:
                    self.left_pos = (self.left_pos[0] * (1-alpha) + left[0] * alpha,
                                     self.left_pos[1] * (1-alpha) + left[1] * alpha)
                else:
                    self.left_pos = left
            else:
                self.left_pos = None

            if right:
                if self.right_pos:
                    self.right_pos = (self.right_pos[0] * (1-alpha) + right[0] * alpha,
                                      self.right_pos[1] * (1-alpha) + right[1] * alpha)
                else:
                    self.right_pos = right
            else:
                self.right_pos = None
            self.frame_rgb = rgb


    def update(self):
        # Update is now handled asynchronously by _capture_loop
        pass

    def get_webcam_surface(self) -> Optional[pygame.Surface]:
        if self.frame_rgb is None:
            return None
        small = cv2.resize(self.frame_rgb, (CAM_W, CAM_H))
        surf  = pygame.surfarray.make_surface(np.transpose(small, (1, 0, 2)))
        return surf

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


# ─────────────────────────── FONT HELPERS ─────────────────────
def get_fonts():
    pygame.font.init()
    big   = pygame.font.SysFont('Courier New', 120, bold=True)
    med   = pygame.font.SysFont('Courier New',  48, bold=True)
    small = pygame.font.SysFont('Courier New',  28)
    return big, med, small


def render_neon_text(surf, font, text, color, cx, cy, glow_layers=3):
    base = font.render(text, True, color)
    for i in range(glow_layers, 0, -1):
        alpha = 40
        glow  = font.render(text, True, color)
        gs    = pygame.Surface(glow.get_size(), pygame.SRCALPHA)
        gs.fill((0,0,0,0))
        gs.blit(glow, (0,0))
        scaled = pygame.transform.smoothscale(
            gs, (gs.get_width()+i*8, gs.get_height()+i*8))
        scaled.set_alpha(alpha)
        surf.blit(scaled, (cx - scaled.get_width()//2, cy - scaled.get_height()//2))
    surf.blit(base, (cx - base.get_width()//2, cy - base.get_height()//2))


# ─────────────────────────── ARENA DRAW ───────────────────────
_arena_surf = None
def get_arena_surf():
    global _arena_surf
    if _arena_surf is None:
        _arena_surf = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
        mid_x = GAME_W // 2
        draw_glow_line(_arena_surf, C_GOAL_LINE, (mid_x, 0), (mid_x, GAME_H), 2, 2)
        for i in range(3, 0, -1):
            alpha = 20
            cs = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
            pygame.draw.circle(cs, (*C_GOAL_LINE, alpha), (mid_x, GAME_H//2), 100 + i*8, 3)
            _arena_surf.blit(cs, (0, 0))
        pygame.draw.circle(_arena_surf, C_GOAL_LINE, (mid_x, GAME_H//2), 100, 2)

        goal_w = 160
        for gx, color in [(0, C_P1), (GAME_W - goal_w, C_P2)]:
            r = pygame.Rect(gx, 0, goal_w, 6)
            pygame.draw.rect(_arena_surf, color, r)
            r2 = pygame.Rect(gx, GAME_H - 6, goal_w, 6)
            pygame.draw.rect(_arena_surf, color, r2)

        draw_glow_rect(_arena_surf, C_WHITE,
                       pygame.Rect(4, 4, GAME_W-8, GAME_H-8), 2, 2)
    return _arena_surf


def draw_arena(surf, score1, score2, font_big, font_med, font_small, cam_surf=None):
    if cam_surf:
        bg = pygame.transform.scale(cam_surf, (GAME_W, GAME_H))
        surf.blit(bg, (0, 0))
        overlay = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
        overlay.fill((5, 5, 15, 150))
        surf.blit(overlay, (0, 0))
    else:
        surf.fill(C_BG)

    # court lines
    surf.blit(get_arena_surf(), (0, 0))

    # scores
    render_neon_text(surf, font_big, str(score1), C_P1, GAME_W//4,   60)
    render_neon_text(surf, font_big, str(score2), C_P2, 3*GAME_W//4, 60)

    # player labels
    render_neon_text(surf, font_small, "P1  LEFT HAND",  C_P1, GAME_W//4,   GAME_H - 30)
    render_neon_text(surf, font_small, "RIGHT HAND  P2", C_P2, 3*GAME_W//4, GAME_H - 30)


# ─────────────────────────── SCREENS ──────────────────────────
def draw_start_screen(surf, font_big, font_med, font_small, cam_surf):
    surf.fill(C_BG)
    if cam_surf:
        surf.blit(cam_surf, (GAME_W//2 - CAM_W//2, GAME_H//2 - CAM_H//2 + 60))
    render_neon_text(surf, font_big, "AIR HOCKEY", C_BALL, GAME_W//2, 80)
    # Draw a decorative border
    border_rect = pygame.Rect(GAME_W//2 - 400, 160, 800, 300)
    draw_glow_rect(surf, C_GOAL_LINE, border_rect, width=2, layers=4)

    render_neon_text(surf, font_med, "Show BOTH hands to camera", C_WHITE, GAME_W//2, 210)
    render_neon_text(surf, font_small,
                     "Left hand = P1 (CYAN)   |   Right hand = P2 (MAGENTA)",
                     C_WHITE, GAME_W//2, 270)
    render_neon_text(surf, font_small,
                     f"First to {WIN_SCORE} wins!",
                     C_BALL, GAME_W//2, 320)
    render_neon_text(surf, font_med, "Press Q to Start", C_P1, GAME_W//2, GAME_H - 80)


def draw_goal_screen(surf, font_big, font_med, scorer, particles, dt):
    overlay = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 100))
    surf.blit(overlay, (0, 0))
    color = C_P1 if scorer == 1 else C_P2
    render_neon_text(surf, font_big, f"GOAL!  P{scorer}", color, GAME_W//2, GAME_H//2 - 40)
    render_neon_text(surf, font_med, "Get ready...", C_WHITE, GAME_W//2, GAME_H//2 + 60)


def draw_win_screen(surf, font_big, font_med, font_small, winner):
    surf.fill(C_BG)
    color = C_P1 if winner == 1 else C_P2
    render_neon_text(surf, font_big, f"PLAYER {winner}", color, GAME_W//2, GAME_H//2 - 80)
    render_neon_text(surf, font_big, "WINS!", color, GAME_W//2, GAME_H//2 + 20)
    render_neon_text(surf, font_med, "Q to Play Again | R to Reset | ESC for Menu", C_WHITE, GAME_W//2, GAME_H//2 + 130)


# ─────────────────────────── MAIN ─────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((GAME_W, GAME_H))
    pygame.display.set_caption("Air Hockey")
    clock  = pygame.time.Clock()

    font_big, font_med, font_small = get_fonts()
    tracker = HandTracker()

    score  = [0, 0]
    ball   = Ball()
    p1     = Paddle(GAME_W//4,     GAME_H//2, C_P1, 'left')
    p2     = Paddle(3*GAME_W//4,   GAME_H//2, C_P2, 'right')
    particles: list[Particle] = []

    STATE_START  = 'start'
    STATE_PLAY   = 'play'
    STATE_GOAL   = 'goal'
    STATE_WIN    = 'win'
    state        = STATE_START
    goal_timer   = 0.0
    GOAL_DELAY   = 2.2
    last_scorer  = 1
    winner       = 0

    # Background grid surface (static)
    grid_surf = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
    for gx in range(0, GAME_W, 60):
        pygame.draw.line(grid_surf, (255,255,255,8), (gx,0), (gx, GAME_H))
    for gy in range(0, GAME_H, 60):
        pygame.draw.line(grid_surf, (255,255,255,8), (0,gy), (GAME_W, gy))

    running = True
    while running:
        dt = clock.tick(240) / 1000.0
        tracker.update()
        cam_surf = tracker.get_webcam_surface()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_q:
                    if state in (STATE_START, STATE_WIN):
                        score = [0, 0]
                        ball.reset()
                        p1.x, p1.y = GAME_W//4,   GAME_H//2
                        p2.x, p2.y = 3*GAME_W//4, GAME_H//2
                        particles.clear()
                        state = STATE_PLAY
                if event.key == pygame.K_r:
                    score = [0, 0]
                    ball.reset()
                    p1.x, p1.y = GAME_W//4,   GAME_H//2
                    p2.x, p2.y = 3*GAME_W//4, GAME_H//2
                    particles.clear()
                    if state != STATE_START:
                        state = STATE_PLAY

        # ── START ──
        if state == STATE_START:
            draw_start_screen(screen, font_big, font_med, font_small, cam_surf)

        # ── PLAY ──
        elif state == STATE_PLAY:
            # move paddles from hand positions
            if tracker.left_pos:
                p1.move_to(*tracker.left_pos, dt)
            if tracker.right_pos:
                p2.move_to(*tracker.right_pos, dt)

            ball.update(dt, particles)
            p1.check_ball_collision(ball, particles)
            p2.check_ball_collision(ball, particles)

            # update particles
            for p in particles:
                p.update(dt)
            particles = [p for p in particles if p.alive]

            # scoring — ball passes goal line (left or right edge)
            scored = 0
            if ball.x - BALL_R < 0:
                scored = 2          # P2 scores (ball went past P1's side)
            elif ball.x + BALL_R > GAME_W:
                scored = 1          # P1 scores

            if scored:
                score[scored-1] += 1
                last_scorer = scored
                spawn_particles(particles, ball.x, ball.y, C_BALL, 40, 250)
                if score[scored-1] >= WIN_SCORE:
                    winner = scored
                    state  = STATE_WIN
                else:
                    ball.reset(direction=1 if scored==1 else -1)
                    goal_timer = GOAL_DELAY
                    state = STATE_GOAL

            # draw
            draw_arena(screen, score[0], score[1], font_big, font_med, font_small, cam_surf)
            screen.blit(grid_surf, (0, 0))

            # draw particles under paddles/ball
            for p in particles:
                p.draw(screen)
            p1.draw(screen)
            p2.draw(screen)
            ball.draw(screen)

            # hand tracking indicators
            if tracker.left_pos is None:
                render_neon_text(screen, font_small,
                                 "No left hand detected!", C_P1, GAME_W//4, GAME_H - 60)
            if tracker.right_pos is None:
                render_neon_text(screen, font_small,
                                 "No right hand detected!", C_P2, 3*GAME_W//4, GAME_H - 60)

        # ── GOAL ──
        elif state == STATE_GOAL:
            draw_arena(screen, score[0], score[1], font_big, font_med, font_small, cam_surf)
            screen.blit(grid_surf, (0, 0))
            for p in particles:
                p.update(dt)
                p.draw(screen)
            particles = [p for p in particles if p.alive]
            draw_goal_screen(screen, font_big, font_med, last_scorer, particles, dt)
            goal_timer -= dt
            if goal_timer <= 0:
                state = STATE_PLAY

        # ── WIN ──
        elif state == STATE_WIN:
            draw_win_screen(screen, font_big, font_med, font_small, winner)

        # FPS Counter
        fps = clock.get_fps()
        fps_text = font_small.render(f"FPS: {int(fps)}", True, (0, 255, 0))
        screen.blit(fps_text, (10, 10))

        pygame.display.flip()

    tracker.release()
    pygame.quit()


if __name__ == '__main__':
    main()
