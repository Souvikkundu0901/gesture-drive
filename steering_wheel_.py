import os
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import platform
import atexit
import signal
import ctypes
from pynput.keyboard import Key, Controller

CAMERA_INDEX        = 0         # 0 = built-in webcam. Change to 1/2 for external cam
DEAD_ZONE_DEG        = 12
RELEASE_ZONE_DEG     = 6
SOFT_ZONE_DEG        = 25
FLIP_CAMERA          = True
SHOW_ANGLE           = True
MIN_DETECTION_CONF   = 0.7
MIN_TRACKING_CONF    = 0.5
GRACE_FRAMES         = 8
OPEN_FINGER_THRESH   = 3

# --- Gesture config (Safe keys to prevent Windows/Chrome lockups) ---
NITRO_KEY            = 'n'         # peace sign -> nitro ('n' instead of Shift to avoid Sticky Keys)
HANDBRAKE_KEY        = Key.space   # thumbs up  -> handbrake
GESTURE_HOLD_FRAMES  = 3           # consecutive frames required to trigger a gesture
GESTURE_RELEASE_FRAMES = 3         # consecutive frames required to release it
THUMB_EXTEND_RATIO   = 0.5         # thumb-tip-to-index-mcp distance / palm-size threshold

CLR_WHEEL    = (80, 200, 255)
CLR_LEFT     = (60, 120, 255)
CLR_RIGHT    = (50, 220, 140)
CLR_NEUTRAL  = (200, 200, 200)
CLR_TEXT     = (255, 255, 255)
CLR_ACCENT   = (0, 180, 255)
CLR_HAND_L   = (255, 130, 60)
CLR_HAND_R   = (60, 230, 130)
CLR_ACCEL    = (50, 220, 100)
CLR_BRAKE    = (0, 60, 255)
CLR_NITRO    = (255, 210, 0)
CLR_HANDBRK  = (0, 140, 255)

keyboard = Controller()

# Keys that this program is allowed to control.
CONTROLLED_KEYS = (
    Key.left, Key.right, Key.up, Key.down,
    Key.shift, Key.shift_l, Key.shift_r, Key.space,
    NITRO_KEY, HANDBRAKE_KEY,
)

def windows_force_keyup():
    """Forces Windows OS to clear any virtual keydown states for all controlled keys and modifiers."""
    if platform.system() == "Windows":
        try:
            user32 = ctypes.windll.user32
            # VK_SHIFT=0x10, VK_CONTROL=0x11, VK_MENU=0x12, VK_SPACE=0x20, VK_LEFT=0x25, VK_UP=0x26, VK_RIGHT=0x27, VK_DOWN=0x28, VK_LSHIFT=0xA0, VK_RSHIFT=0xA1, 'N'=0x4E, 'B'=0x42
            vk_codes = [0x10, 0x11, 0x12, 0x20, 0x25, 0x26, 0x27, 0x28, 0xA0, 0xA1, 0x4E, 0x42]
            for vk in vk_codes:
                user32.keybd_event(vk, 0, 0x0002, 0)  # KEYEVENTF_KEYUP = 0x0002
        except Exception:
            pass

def force_release_keyboard():
    """Release every key this program can control, even if our state is wrong."""
    for key in CONTROLLED_KEYS:
        try:
            keyboard.release(key)
        except Exception:
            pass
    windows_force_keyup()

# If the previous run was interrupted, clear any modifier/control-key state
# before starting a new run.
force_release_keyboard()
atexit.register(force_release_keyboard)

def handle_shutdown_signal(signum, frame):
    """Best-effort cleanup for Ctrl+C / termination signals."""
    force_release_keyboard()
    raise KeyboardInterrupt

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

class LandmarkWrapper:
    def __init__(self, landmarks):
        self.landmark = landmarks

class DetectionResultWrapper:
    def __init__(self, multi_hand_landmarks, multi_handedness):
        self.multi_hand_landmarks = multi_hand_landmarks
        self.multi_handedness = multi_handedness

class HandednessWrapper:
    class Classification:
        def __init__(self, label):
            self.label = label
    def __init__(self, label):
        self.classification = [self.Classification(label)]

class HandsDetector:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.use_legacy = True
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_hands = mp.solutions.hands
        else:
            self.use_legacy = False
            model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
            if not os.path.exists(model_path):
                print("[INFO] Downloading hand landmarker model...")
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                urllib.request.urlretrieve(url, model_path)
                print("[INFO] Model downloaded successfully.")

            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, rgb_frame):
        if self.use_legacy:
            return self.hands.process(rgb_frame)
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.detector.detect(mp_image)
            multi_hand_landmarks = []
            multi_handedness = []
            if result.hand_landmarks and result.handedness:
                for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    multi_hand_landmarks.append(LandmarkWrapper(landmarks))
                    label = handedness[0].category_name
                    multi_handedness.append(HandednessWrapper(label))
            return DetectionResultWrapper(multi_hand_landmarks, multi_handedness)

    def draw_landmarks(self, frame, hand_landmarks):
        if self.use_legacy:
            conn_style = self.mp_drawing.DrawingSpec(color=(80, 80, 100), thickness=1)
            landmark_style = self.mp_drawing.DrawingSpec(color=(200, 200, 255), thickness=1, circle_radius=2)
            self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS, landmark_style, conn_style)
        else:
            h, w = frame.shape[:2]
            lm_list = hand_landmarks.landmark
            for start_idx, end_idx in HAND_CONNECTIONS:
                p1 = (int(lm_list[start_idx].x * w), int(lm_list[start_idx].y * h))
                p2 = (int(lm_list[end_idx].x * w), int(lm_list[end_idx].y * h))
                cv2.line(frame, p1, p2, (80, 80, 100), 1)
            for lm in lm_list:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 2, (200, 200, 255), -1)

    def close(self):
        if self.use_legacy:
            self.hands.close()
        else:
            self.detector.close()



def is_open_hand(hand_landmarks):
    FINGER_TIPS = [8, 12, 16, 20]
    FINGER_PIPS = [6, 10, 14, 18]
    extended = sum(
        1 for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y
    )
    return extended >= OPEN_FINGER_THRESH


def get_finger_states(hand_landmarks):
    """Returns (thumb, index, middle, ring, pinky) extended booleans."""
    lm = hand_landmarks.landmark

    # Non-thumb fingers: tip above pip (in image y) means extended
    index  = lm[8].y  < lm[6].y
    middle = lm[12].y < lm[10].y
    ring   = lm[16].y < lm[14].y
    pinky  = lm[20].y < lm[18].y

    # Thumb: use distance from thumb tip to index MCP, normalized by palm size
    palm_size = math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y) + 1e-6
    thumb_spread = math.hypot(lm[4].x - lm[5].x, lm[4].y - lm[5].y) / palm_size
    thumb = thumb_spread > THUMB_EXTEND_RATIO

    return thumb, index, middle, ring, pinky


def classify_gesture(hand_landmarks):
    """Returns 'PEACE', 'THUMBS_UP', or None for gesture-recognition purposes."""
    thumb, index, middle, ring, pinky = get_finger_states(hand_landmarks)

    if index and middle and not ring and not pinky:
        return "PEACE"
    if thumb and not index and not middle and not ring and not pinky:
        return "THUMBS_UP"
    return None


class SteeringController:
    def __init__(self):
        self.keys_held = {
            Key.left: False, Key.right: False, Key.up: False, Key.down: False,
            NITRO_KEY: False, HANDBRAKE_KEY: False,
        }
        self.angle_history = []
        self.HISTORY_LEN   = 1

        # gesture debounce counters
        self.nitro_hit    = 0
        self.nitro_miss   = 0
        self.brake_hit    = 0
        self.brake_miss   = 0
        self.nitro_active = False
        self.handbrake_active = False

    def _press(self, key):
        if not self.keys_held[key]:
            keyboard.press(key)
            self.keys_held[key] = True

    def _release(self, key):
        if self.keys_held[key]:
            keyboard.release(key)
            self.keys_held[key] = False

    def release_all(self):
        # Release according to our state first...
        for key in list(self.keys_held.keys()):
            try:
                keyboard.release(key)
            except Exception:
                pass
            self.keys_held[key] = False

        # ...then force-release every controlled key in case our state became
        # out of sync because the process was interrupted or an exception
        # happened between press() and updating keys_held.
        force_release_keyboard()

        self.angle_history.clear()
        self.nitro_hit = self.nitro_miss = self.brake_hit = self.brake_miss = 0
        self.nitro_active = self.handbrake_active = False

    def smooth_angle(self, raw_angle):
        self.angle_history.append(raw_angle)
        if len(self.angle_history) > self.HISTORY_LEN:
            self.angle_history.pop(0)
        return float(np.mean(self.angle_history))

    def update_steer(self, left_wrist, right_wrist):
        dx = right_wrist[0] - left_wrist[0]
        dy = right_wrist[1] - left_wrist[1]

        raw_angle_rad = math.atan2(dy, dx)
        raw_angle_deg = math.degrees(raw_angle_rad)
        angle = self.smooth_angle(raw_angle_deg)

        direction = "STRAIGHT"
        if angle < -DEAD_ZONE_DEG:
            direction = "LEFT"
        elif angle > DEAD_ZONE_DEG:
            direction = "RIGHT"
        elif self.keys_held[Key.left] and angle > -RELEASE_ZONE_DEG:
            direction = "STRAIGHT"
        elif self.keys_held[Key.right] and angle < RELEASE_ZONE_DEG:
            direction = "STRAIGHT"

        strength = 0.0
        if direction == "LEFT":
            strength = min(1.0, (abs(angle) - DEAD_ZONE_DEG) / (SOFT_ZONE_DEG - DEAD_ZONE_DEG))
            self._press(Key.left)
            self._release(Key.right)
        elif direction == "RIGHT":
            strength = min(1.0, (abs(angle) - DEAD_ZONE_DEG) / (SOFT_ZONE_DEG - DEAD_ZONE_DEG))
            self._press(Key.right)
            self._release(Key.left)
        else:
            self._release(Key.left)
            self._release(Key.right)

        return angle, direction, strength

    def update_throttle(self, left_open, right_open):
        both_open = left_open and right_open
        both_fist = (not left_open) and (not right_open)

        if both_fist:
            self._press(Key.up)
            self._release(Key.down)
            return "ACCEL"
        elif both_open:
            self._press(Key.down)
            self._release(Key.up)
            return "BRAKE"
        else:
            self._release(Key.up)
            self._release(Key.down)
            return "NEUTRAL"

    def update_gestures(self, gestures_this_frame):
        """gestures_this_frame: set of gestures ('PEACE'/'THUMBS_UP') seen across all visible hands."""
        # --- Nitro (peace sign) ---
        if "PEACE" in gestures_this_frame:
            self.nitro_hit += 1
            self.nitro_miss = 0
        else:
            self.nitro_miss += 1
            self.nitro_hit = 0

        if not self.nitro_active and self.nitro_hit >= GESTURE_HOLD_FRAMES:
            self.nitro_active = True
            self._press(NITRO_KEY)
        elif self.nitro_active and self.nitro_miss >= GESTURE_RELEASE_FRAMES:
            self.nitro_active = False
            self._release(NITRO_KEY)

        # --- Handbrake (thumbs up) ---
        if "THUMBS_UP" in gestures_this_frame:
            self.brake_hit += 1
            self.brake_miss = 0
        else:
            self.brake_miss += 1
            self.brake_hit = 0

        if not self.handbrake_active and self.brake_hit >= GESTURE_HOLD_FRAMES:
            self.handbrake_active = True
            self._press(HANDBRAKE_KEY)
        elif self.handbrake_active and self.brake_miss >= GESTURE_RELEASE_FRAMES:
            self.handbrake_active = False
            self._release(HANDBRAKE_KEY)

        return self.nitro_active, self.handbrake_active


def draw_steering_wheel(frame, center, angle_deg, direction, strength):
    h, w = frame.shape[:2]
    radius = int(min(w, h) * 0.10)
    cx, cy = center

    color = CLR_NEUTRAL
    if direction == "LEFT":
        color = CLR_LEFT
    elif direction == "RIGHT":
        color = CLR_RIGHT

    cv2.circle(frame, (cx + 3, cy + 3), radius, (0, 0, 0), 4)
    cv2.circle(frame, (cx, cy), radius, color, 3)

    for sa in [0, 120, 240]:
        rad = math.radians(sa - angle_deg)
        x1 = int(cx + radius * 0.4 * math.cos(rad))
        y1 = int(cy - radius * 0.4 * math.sin(rad))
        x2 = int(cx + radius * 0.95 * math.cos(rad))
        y2 = int(cy - radius * 0.95 * math.sin(rad))
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    cv2.circle(frame, (cx, cy), 6, color, -1)

    if direction != "STRAIGHT":
        start_a = -30 if direction == "RIGHT" else 150
        end_a   =  30 if direction == "RIGHT" else 210
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_a, end_a, color, 5)


def draw_hud(frame, angle, direction, strength, throttle_mode, both_hands_visible,
             left_open, right_open, fps, nitro_active, handbrake_active, is_paused=False):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 190), (w, h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    bar_w = int(w * 0.5)
    bar_h = 14
    bar_x = (w - bar_w) // 2
    bar_y = h - 140
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)

    mid = bar_x + bar_w // 2
    cv2.rectangle(frame, (mid - 2, bar_y - 4), (mid + 2, bar_y + bar_h + 4), (180, 180, 180), -1)

    fill_len = int((bar_w // 2) * strength)
    if direction == "LEFT" and fill_len > 0:
        cv2.rectangle(frame, (mid - fill_len, bar_y), (mid, bar_y + bar_h), CLR_LEFT, -1)
    elif direction == "RIGHT" and fill_len > 0:
        cv2.rectangle(frame, (mid, bar_y), (mid + fill_len, bar_y + bar_h), CLR_RIGHT, -1)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    dir_color = CLR_LEFT if direction == "LEFT" else (CLR_RIGHT if direction == "RIGHT" else CLR_NEUTRAL)
    cv2.putText(frame, "<- LEFT",  (bar_x, bar_y - 10),               font, 0.45, CLR_LEFT,  1)
    cv2.putText(frame, "RIGHT ->", (bar_x + bar_w - 80, bar_y - 10),  font, 0.45, CLR_RIGHT, 1)
    cv2.putText(frame, direction,  (mid - 30, bar_y + bar_h + 28),    font, 0.8,  dir_color, 2)

    if SHOW_ANGLE:
        cv2.putText(frame, f"{angle:+.1f} deg", (bar_x, h - 110), font, 0.55, CLR_TEXT, 1)

    throttle_color = CLR_ACCEL if throttle_mode == "ACCEL" else (CLR_BRAKE if throttle_mode == "BRAKE" else CLR_NEUTRAL)
    throttle_label = {
        "ACCEL":   "ACCEL [UP]",
        "BRAKE":   "BRAKE [DOWN]",
        "NEUTRAL": "NEUTRAL",
    }[throttle_mode]

    cv2.rectangle(frame, (bar_x, h - 95), (bar_x + bar_w, h - 72), (30, 30, 40), -1)
    cv2.rectangle(frame, (bar_x, h - 95), (bar_x + bar_w, h - 72), throttle_color, 2)
    cv2.putText(frame, throttle_label, (bar_x + 10, h - 78), font, 0.65, throttle_color, 2)

    # --- Gesture status row (nitro / handbrake) ---
    gest_y1, gest_y2 = h - 65, h - 42
    half_w = bar_w // 2 - 5

    nitro_color = CLR_NITRO if nitro_active else (60, 60, 70)
    cv2.rectangle(frame, (bar_x, gest_y1), (bar_x + half_w, gest_y2), (30, 30, 40), -1)
    cv2.rectangle(frame, (bar_x, gest_y1), (bar_x + half_w, gest_y2), nitro_color, 2)
    cv2.putText(frame, "NITRO [N]" if nitro_active else "nitro: peace [N]",
                (bar_x + 8, gest_y2 - 6), font, 0.5, nitro_color, 2 if nitro_active else 1)

    hb_color = CLR_HANDBRK if handbrake_active else (60, 60, 70)
    hb_x = bar_x + bar_w - half_w
    cv2.rectangle(frame, (hb_x, gest_y1), (hb_x + half_w, gest_y2), (30, 30, 40), -1)
    cv2.rectangle(frame, (hb_x, gest_y1), (hb_x + half_w, gest_y2), hb_color, 2)
    cv2.putText(frame, "HANDBRAKE [SPACE]" if handbrake_active else "brake: thumbs up",
                (hb_x + 8, gest_y2 - 6), font, 0.45, hb_color, 2 if handbrake_active else 1)

    l_label = "OPEN" if left_open else "FIST"
    r_label = "OPEN" if right_open else "FIST"
    l_color = CLR_BRAKE if left_open else CLR_ACCEL
    r_color = CLR_BRAKE if right_open else CLR_ACCEL
    cv2.putText(frame, f"L:{l_label}", (bar_x + bar_w + 10, h - 130), font, 0.5, l_color, 1)
    cv2.putText(frame, f"R:{r_label}", (bar_x + bar_w + 10, h - 110), font, 0.5, r_color, 1)

    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 90, 30), font, 0.55, CLR_ACCENT, 1)

    if is_paused:
        # Prominent PAUSE warning at the top center
        cv2.rectangle(frame, (w // 2 - 190, 10), (w // 2 + 190, 48), (0, 0, 180), -1)
        cv2.putText(frame, "PAUSED (Press P to Resume)", (w // 2 - 175, 36), font, 0.65, (255, 255, 255), 2)
    else:
        status       = "BOTH HANDS DETECTED" if both_hands_visible else "SHOW BOTH HANDS"
        status_color = (60, 220, 60) if both_hands_visible else (0, 80, 255)
        cv2.putText(frame, status, (10, 30), font, 0.55, status_color, 1)
        cv2.putText(frame, "[P]=Pause [ESC]=Quit", (10, 55), font, 0.45, (180, 180, 180), 1)

    draw_steering_wheel(frame, (w - 80, h - 100), angle, direction, strength)


def draw_hand_connection(frame, lw, rw):
    lx, ly = lw
    rx, ry = rw
    cv2.line(frame, (lx, ly), (rx, ry), (30, 100, 200), 8)
    cv2.line(frame, (lx, ly), (rx, ry), CLR_ACCENT, 2)
    cv2.circle(frame, (lx, ly), 10, CLR_HAND_L, -1)
    cv2.circle(frame, (rx, ry), 10, CLR_HAND_R, -1)
    cv2.circle(frame, (lx, ly), 13, CLR_HAND_L, 2)
    cv2.circle(frame, (rx, ry), 13, CLR_HAND_R, 2)
    mx = (lx + rx) // 2
    my = (ly + ry) // 2
    cv2.circle(frame, (mx, my), 7, CLR_WHEEL, -1)


def main():
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        print("  -> macOS: System Settings > Privacy & Security > Camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    controller = SteeringController()

    # Best-effort cleanup if the program receives Ctrl+C or a termination signal.
    try:
        signal.signal(signal.SIGINT, handle_shutdown_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handle_shutdown_signal)
    except Exception:
        pass

    hands = HandsDetector(
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF,
    )

    window_name = "Virtual Steering Wheel (Extended)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    prev_time         = time.time()
    angle             = 0.0
    direction         = "STRAIGHT"
    strength          = 0.0
    throttle_mode     = "NEUTRAL"
    left_open         = False
    right_open        = False
    lost_frames       = 0
    nitro_active      = False
    handbrake_active  = False
    is_paused         = False

    print("=" * 55)
    print("  Virtual Steering Wheel (Extended)  |  ESC or Q to quit")
    print("=" * 55)
    print("  FIST        = Accelerate (UP)     OPEN  = Brake (DOWN)")
    print("  PEACE SIGN  = Nitro (N)           THUMBS UP = Handbrake (SPACE)")
    print("  Tilt hands LEFT/RIGHT to steer — works in any mode")
    print("  P = Pause/Resume controller (releases all keys)")
    print("  R = Emergency release of ALL controlled keys")
    print("  ESC / Q = Stop and close cleanly")
    print("=" * 55)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            if FLIP_CAMERA:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            both_visible = False
            gestures_this_frame = set()

            if results.multi_hand_landmarks and results.multi_handedness:
                hand_data = {}

                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label

                    hands.draw_landmarks(frame, hand_landmarks)

                    wrist  = hand_landmarks.landmark[0]
                    wx     = int(wrist.x * w)
                    wy     = int(wrist.y * h)
                    opened = is_open_hand(hand_landmarks)
                    hand_data[label] = (wrist.x, wrist.y, wx, wy, opened)

                    gesture = classify_gesture(hand_landmarks)
                    if gesture:
                        gestures_this_frame.add(gesture)

                if not is_paused:
                    if "Left" in hand_data and "Right" in hand_data:
                        both_visible = True
                        lost_frames  = 0

                        lx_n, ly_n, lx_px, ly_px, left_open  = hand_data["Left"]
                        rx_n, ry_n, rx_px, ry_px, right_open = hand_data["Right"]

                        draw_hand_connection(frame, (lx_px, ly_px), (rx_px, ry_px))
                        angle, direction, strength = controller.update_steer((lx_n, ly_n), (rx_n, ry_n))
                        throttle_mode = controller.update_throttle(left_open, right_open)
                    else:
                        lost_frames += 1
                        if lost_frames >= GRACE_FRAMES:
                            controller.release_all()
                            angle, direction, strength = 0.0, "STRAIGHT", 0.0
                            throttle_mode = "NEUTRAL"
                            left_open = right_open = False

                    # gestures are evaluated from whichever hands are visible, even if only one
                    nitro_active, handbrake_active = controller.update_gestures(gestures_this_frame)
                else:
                    # When paused, keep keys released
                    controller.release_all()
                    angle, direction, strength = 0.0, "STRAIGHT", 0.0
                    throttle_mode = "NEUTRAL"
                    left_open = right_open = False
                    nitro_active = handbrake_active = False
            else:
                lost_frames += 1
                if lost_frames >= GRACE_FRAMES or is_paused:
                    controller.release_all()
                    angle, direction, strength = 0.0, "STRAIGHT", 0.0
                    throttle_mode = "NEUTRAL"
                    left_open = right_open = False
                    nitro_active = handbrake_active = False
                else:
                    nitro_active, handbrake_active = controller.update_gestures(set())

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw_hud(frame, angle, direction, strength, throttle_mode, both_visible,
                     left_open, right_open, fps, nitro_active, handbrake_active, is_paused=is_paused)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            # P = Pause/Resume toggle
            if key in (ord('p'), ord('P')):
                is_paused = not is_paused
                controller.release_all()
                force_release_keyboard()
                nitro_active = False
                handbrake_active = False
                throttle_mode = "NEUTRAL"
                direction = "STRAIGHT"
                strength = 0.0
                print(f"[INFO] Controller {'PAUSED' if is_paused else 'RESUMED'}.")

            # R = emergency keyboard reset
            if key in (ord('r'), ord('R')):
                controller.release_all()
                force_release_keyboard()
                nitro_active = False
                handbrake_active = False
                throttle_mode = "NEUTRAL"
                direction = "STRAIGHT"
                strength = 0.0
                print("[INFO] Emergency keyboard reset: all controlled keys released.")

            # ESC (27) or Q = quit
            if key in (ord('q'), ord('Q'), 27):
                break

    finally:
        # Always release keyboard input before closing the camera/window.
        controller.release_all()
        force_release_keyboard()
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Stopped. All controlled keys released.")


if __name__ == "__main__":
    main()
