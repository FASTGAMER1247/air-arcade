import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import os
import pygame
import numpy as np
import sys
import subprocess
import threading
import time
import math

# Configuration
WIDTH, HEIGHT = 1280, 720
HOVER_TIME_REQ = 1.5  # seconds

def get_fonts():
    pygame.font.init()
    big = pygame.font.SysFont('Courier New', 80, bold=True)
    med = pygame.font.SysFont('Courier New', 48, bold=True)
    small = pygame.font.SysFont('Courier New', 28)
    return big, med, small

class LauncherTracker:
    def __init__(self, model_path):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO)
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.hand_pos = None # x, y in pixels
        self.frame_rgb = None
        self.timestamp_ms = 0
        self.running = True
        
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                self.timestamp_ms += 33
                results = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

                current_pos = None
                if results.hand_landmarks:
                    # Just use the first hand found
                    hand = results.hand_landmarks[0]
                    tip = hand[8] # index finger tip
                    current_pos = (tip.x * WIDTH, tip.y * HEIGHT)

                # Smoothing
                if current_pos:
                    if self.hand_pos:
                        alpha = 0.6
                        self.hand_pos = (self.hand_pos[0]*(1-alpha) + current_pos[0]*alpha,
                                         self.hand_pos[1]*(1-alpha) + current_pos[1]*alpha)
                    else:
                        self.hand_pos = current_pos
                else:
                    self.hand_pos = None

                self.frame_rgb = rgb
        finally:
            if getattr(self, "landmarker", None) is not None:
                self.landmarker.close()
                self.landmarker = None

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=3.0)
        if getattr(self, "cap", None) is not None:
            self.cap.release()
            self.cap = None
        if not self.thread.is_alive() and getattr(self, "landmarker", None) is not None:
            self.landmarker.close()
            self.landmarker = None

def draw_button(surf, font_med, font_small, text, subtext, rect, color, hover_progress):
    # Create an alpha surface
    alpha_surf = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
    
    # Background dim
    pygame.draw.rect(alpha_surf, (20, 20, 30, 180), (0, 0, rect.w, rect.h), border_radius=15)
    
    # Draw hover fill from bottom to top
    if hover_progress > 0:
        fill_h = int(rect.h * hover_progress)
        fill_rect = pygame.Rect(0, rect.h - fill_h, rect.w, fill_h)
        pygame.draw.rect(alpha_surf, (*color, 80), fill_rect, border_radius=15)
        
    # Draw border
    border_color = color if hover_progress > 0 else (100, 100, 100)
    border_width = 6 if hover_progress > 0 else 3
    pygame.draw.rect(alpha_surf, border_color, (0, 0, rect.w, rect.h), width=border_width, border_radius=15)
    
    # Blit to main surface
    surf.blit(alpha_surf, (rect.x, rect.y))
    
    # Text
    txt_surf = font_med.render(text, True, color)
    surf.blit(txt_surf, (rect.centerx - txt_surf.get_width()//2, rect.centery - 20))
    sub_surf = font_small.render(subtext, True, (200, 200, 200))
    surf.blit(sub_surf, (rect.centerx - sub_surf.get_width()//2, rect.centery + 30))

def show_menu():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Gesture Game Launcher")
    clock = pygame.time.Clock()
    
    font_big, font_med, font_small = get_fonts()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "files", "hand_landmarker.task")
    if not os.path.exists(model_path):
        import urllib.request
        print("Downloading model...")
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", model_path)
    
    tracker = LauncherTracker(model_path)
    
    rect_hockey = pygame.Rect(140, 250, 450, 350)
    rect_puzzle = pygame.Rect(690, 250, 450, 350)
    
    hockey_hover = 0.0
    puzzle_hover = 0.0
    
    running = True
    selected_game = None
    
    try:
        while running:
            dt = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_x:
                        running = False
                        selected_game = None
                    if event.key == pygame.K_q:
                        selected_game = "hockey"
                        running = False

            # Draw camera background
            screen.fill((5, 5, 15))
            if tracker.frame_rgb is not None:
                cam_surf = pygame.surfarray.make_surface(np.transpose(tracker.frame_rgb, (1, 0, 2)))
                cam_surf = pygame.transform.scale(cam_surf, (WIDTH, HEIGHT))
                cam_surf.set_alpha(100) # dim webcam
                screen.blit(cam_surf, (0, 0))

            # Draw Title
            title = font_big.render("ARCADE LAUNCHER", True, (255, 255, 255))
            screen.blit(title, (WIDTH//2 - title.get_width()//2, 60))
            sub = font_small.render("Hover Hand OR Press Q: Air Hockey | X: Exit Launcher", True, (200, 200, 200))
            screen.blit(sub, (WIDTH//2 - sub.get_width()//2, 160))

            # Interaction logic
            pos = tracker.hand_pos
            hovered_hockey = False
            hovered_puzzle = False

            if pos:
                if rect_hockey.collidepoint(pos[0], pos[1]):
                    hovered_hockey = True
                elif rect_puzzle.collidepoint(pos[0], pos[1]):
                    hovered_puzzle = True

            if hovered_hockey:
                hockey_hover = min(1.0, hockey_hover + dt / HOVER_TIME_REQ)
                if hockey_hover >= 1.0:
                    selected_game = "hockey"
                    running = False
            else:
                hockey_hover = max(0, hockey_hover - dt * 2)

            if hovered_puzzle:
                puzzle_hover = min(1.0, puzzle_hover + dt / HOVER_TIME_REQ)
                if puzzle_hover >= 1.0:
                    selected_game = "puzzle"
                    running = False
            else:
                puzzle_hover = max(0, puzzle_hover - dt * 2)

            # Draw Buttons
            draw_button(screen, font_med, font_small, "AIR HOCKEY", "2 Player Action [Q]", rect_hockey, (0, 220, 255), hockey_hover)
            draw_button(screen, font_med, font_small, "LIVE PUZZLE", "Webcam Jigsaw", rect_puzzle, (255, 60, 200), puzzle_hover)

            # Draw Cursor
            if pos:
                pygame.draw.circle(screen, (255, 255, 255), (int(pos[0]), int(pos[1])), 15)
                pygame.draw.circle(screen, (200, 200, 200), (int(pos[0]), int(pos[1])), 20, 2)

                # Optional: draw progress on cursor itself
                active_hover = max(hockey_hover, puzzle_hover)
                if active_hover > 0:
                    pygame.draw.circle(screen, (255, 255, 0), (int(pos[0]), int(pos[1])), int(15 + 20 * active_hover), 3)

            pygame.display.flip()
    except KeyboardInterrupt:
        selected_game = None
    finally:
        tracker.release()
        pygame.quit()
    return selected_game

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add game directories to sys.path for direct imports
    files_dir = os.path.join(script_dir, "files")
    puzzul_dir = os.path.join(script_dir, "puzzul")
    
    if files_dir not in sys.path:
        sys.path.append(files_dir)
    if puzzul_dir not in sys.path:
        sys.path.append(puzzul_dir)

    try:
        while True:
            game = show_menu()
            if not game:
                break  # Exit if no game selected (e.g. user closed launcher)

            if game == "hockey":
                import main as hockey_game
                # Use importlib to reload in case of changes, though usually not needed for this flow
                import importlib
                importlib.reload(hockey_game)

                os.chdir(files_dir)
                try:
                    hockey_game.main()
                except Exception as e:
                    print(f"Error running Hockey: {e}")
                finally:
                    os.chdir(script_dir)
            else:
                import live_puzzle as puzzle_game
                import importlib
                importlib.reload(puzzle_game)

                os.chdir(puzzul_dir)
                game_instance = None
                try:
                    game_instance = puzzle_game.LivePuzzle()
                    game_instance.run()
                except Exception as e:
                    print(f"Error running Puzzle: {e}")
                finally:
                    if game_instance is not None:
                        game_instance.close()
                    os.chdir(script_dir)
    except KeyboardInterrupt:
        print("\nExiting launcher.")

if __name__ == '__main__':
    main()
