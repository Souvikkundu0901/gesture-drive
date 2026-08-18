# 🎮 Virtual Steering Wheel — Extended (Nitro + Handbrake)

Control any arrow-key browser/PC racing game using just your hands and a webcam — no hardware needed.

This is a personal extension of [jayesh-cmd/virtual-steering-wheel](https://github.com/jayesh-cmd/virtual-steering-wheel), built on top of the original MediaPipe hand-tracking base with two extra gestures for nitro and handbrake, plus a couple of small fixes.

---

## What's in this folder

| File | Purpose |
|---|---|
| `steering_wheel_extended.py` | Main script — run this |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## How It Works

Hold both fists toward the camera like you're gripping a steering wheel. Tilt your hands to steer.

```
Both hands level     →  Straight (no key pressed)
Tilt LEFT            →  ← LEFT arrow key
Tilt RIGHT           →  → RIGHT arrow key
Both hands FIST      →  ↑ UP arrow key      (accelerate)
Both hands OPEN      →  ↓ DOWN arrow key    (brake)
✌️  Peace sign        →  SHIFT               (nitro)
👍 Thumbs up          →  SPACE               (handbrake)
Remove hands          →  All keys released instantly
```

Peace sign and thumbs up work on **either hand** and are debounced — a gesture has to hold for a few consecutive frames to trigger, so a stray misread frame won't spam the key.

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Run it

```bash
python3 steering_wheel_extended.py
```

Press **Q** in the camera window to quit — this releases all held keys automatically.

---

## Platform Notes

**macOS:** Grant camera permission before running.
`System Settings → Privacy & Security → Camera → enable Terminal (or your Python launcher)`

**Windows:** No manual change needed — the script auto-detects the OS and picks the right camera backend. You may see a one-time Windows Security popup asking if Python can access your camera — click **Allow**.

---

## Config (top of `steering_wheel_extended.py`)

| Setting | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | `0` = built-in webcam, `1`/`2` = external USB camera |
| `DEAD_ZONE_DEG` | `12` | Degrees of tilt to ignore at center (prevents jitter) |
| `FLIP_CAMERA` | `True` | Mirror the feed (selfie view) |
| `GRACE_FRAMES` | `8` | Frames to wait before releasing steer/throttle keys when hands disappear |
| `GESTURE_HOLD_FRAMES` | `3` | Consecutive frames a gesture must be held to trigger |
| `GESTURE_RELEASE_FRAMES` | `3` | Consecutive frames a gesture must be absent to release |
| `NITRO_KEY` | `Shift` | Key pressed for the peace-sign gesture |
| `HANDBRAKE_KEY` | `Space` | Key pressed for the thumbs-up gesture |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `[ERROR] Cannot open camera` | Change `CAMERA_INDEX` to `0`, `1`, or `2` |
| Steering is reversed | Toggle `FLIP_CAMERA = False` |
| Keys stuck after removing hands | Hands must be fully out of frame for ~8 frames |
| Nitro/handbrake won't trigger | Hold the gesture clearly and steadily — it needs 3 consecutive frames to register |
| Nitro/handbrake fires accidentally | Lower lighting/motion blur can confuse finger detection — try a well-lit, plain background |
| Low FPS / laggy | Lower camera resolution in the script or close other apps |

---

## Known Trade-off

Peace sign and thumbs up both count as a "closed-ish" hand under the base fist/open throttle logic. So doing one of those gestures on one hand while the other hand is a genuine fist will *also* trigger accelerate alongside nitro/handbrake. In practice nitro-while-accelerating makes sense; handbrake-while-accelerating is a little odd but harmless — the game just receives both keys. Can be tightened later with stricter mutual exclusion if needed.

---

## Works With Any Game That Uses Arrow Keys

- Google Chrome Dinosaur game
- Trackmania
- TORCS
- Hill Climb Racing (browser)
- Any browser/PC racing game using arrow keys, Shift, and Space

---

— Souvik
