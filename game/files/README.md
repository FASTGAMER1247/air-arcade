# 🏒 Air Hockey

A 2-player air hockey game controlled entirely by your hands via webcam!

## How It Works

| Hand | Player | Color |
|------|--------|-------|
| Left hand  | Player 1 | 🩵 Cyan    |
| Right hand | Player 2 | 💜 Magenta |

Your **index finger tip** controls the paddle. Move your hand to deflect the ball into the opponent's goal.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> On macOS with Apple Silicon:
> ```bash
> pip install opencv-python mediapipe pygame numpy
> ```

### 2. Run the game

```bash
python main.py
```

### 3. Controls

| Action | Key |
|--------|-----|
| Start / Restart | `SPACE` |
| Quit | `ESC` |

## Gameplay Rules

- First to **7 goals** wins
- Ball speeds up with every paddle hit (max speed × 3)
- Each player is locked to their own half
- Score when the ball crosses the opponent's left/right edge

## Features

- 🤚 Real-time hand tracking (MediaPipe, up to 2 hands)
- ✨ Neon glow effects on ball, paddles, and court lines
- 💥 Particle explosion on every hit and goal
- 🔥 Ball trail effect
- 📹 Live webcam feed inset (top-right corner)
- 🏟️ Retro-neon arena with grid overlay

## Tips

- Make sure you have **good lighting** for best hand detection
- Keep both hands **clearly visible** to the camera
- Hold your hand in a natural, open position — the index finger tip is tracked
- Position yourself about **1–2 metres** from the webcam

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera not found | Change `cv2.VideoCapture(0)` to `(1)` or `(2)` in `main.py` |
| Only one hand detected | Improve lighting, keep hands within frame |
| Laggy performance | Lower `GAME_W, GAME_H` to `960, 540` in config section |
| `mediapipe` install fails | Try `pip install mediapipe --pre` |
