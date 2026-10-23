import cv2
import mediapipe as mp
import pyautogui
import time
import math
last_lock_time = 0

pyautogui.FAILSAFE = False

# ==============================
# Camera Setup
# ==============================
cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

# ==============================
# MediaPipe Setup
# ==============================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

# ==============================
# Screen Size
# ==============================
screen_w, screen_h = pyautogui.size()

# ==============================
# Variables
# ==============================
prev_y = 0
frame_count = 0

# EMA Smoothing
smooth_x = 0
smooth_y = 0
alpha = 0.2

# Gesture HUD
gesture_text = "None"

# Lock System
locked = False

# Cooldowns
last_screenshot = 0




# ==============================
# Finger Detection Function
# ==============================
def fingers_up(hand_landmarks):

    fingers = []

    # Thumb
    if hand_landmarks[4].x < hand_landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other Fingers
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):

        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            fingers.append(1)

        else:
            fingers.append(0)

    return fingers


# ==============================
# Main Loop
# ==============================
while True:

    success, frame = cap.read()

    if not success:
        continue

    frame_count += 1

    # Skip alternate frames for performance
    if frame_count % 2 != 0:
        continue

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    gesture_text = "Moving"

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = int(time.time() * 1000)

    result = detector.detect_for_video(mp_image, timestamp)

    if result.hand_landmarks:

        for hand_landmarks in result.hand_landmarks:

            # ==============================
            # Landmark Points
            # ==============================
            index = hand_landmarks[8]
            thumb = hand_landmarks[4]
            middle = hand_landmarks[12]

            index_x = int(index.x * w)
            index_y = int(index.y * h)

            thumb_x = int(thumb.x * w)
            thumb_y = int(thumb.y * h)

            middle_x = int(middle.x * w)
            middle_y = int(middle.y * h)

            # ==============================
            # Finger Detection
            # ==============================
            fingers = fingers_up(hand_landmarks)

            total_fingers = fingers.count(1)

            # ==============================
            # Draw Landmarks
            # ==============================
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)

            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), -1)

            # ==============================
            # LOCK / UNLOCK SYSTEM
            # ==============================

            if 'last_lock_time' not in globals():
                last_lock_time = 0

            current_time = time.time()

            # ==========================================
            # LOCK / UNLOCK SYSTEM
            # ==========================================

            current_time = time.time()

            # Distance between thumb and pinky
            pinky = hand_landmarks[20]

            pinky_x = int(pinky.x * w)
            pinky_y = int(pinky.y * h)

            thumb_pinky_dist = math.hypot(
                thumb_x - pinky_x,
                thumb_y - pinky_y
            )

            # ------------------------------
            # LOCK -> only pinky up
            # ------------------------------
            if fingers == [0,0,0,0,1]:

                if not locked and current_time - last_lock_time > 2:

                    locked = True

                    gesture_text = "SYSTEM LOCKED"

                    last_lock_time = current_time

            # ------------------------------
            # UNLOCK -> thumb + pinky touch
            # ------------------------------
            if locked:

                # Thumb and pinky close
                if thumb_pinky_dist < 70:

                    if current_time - last_lock_time > 2:

                        locked = False

                        gesture_text = "SYSTEM UNLOCKED"

                        last_lock_time = current_time

                        # ==========================================
            # STOP ALL GESTURES WHEN LOCKED
            # ==========================================

            if locked:

                gesture_text = "SYSTEM LOCKED"

                cv2.putText(
                    frame,
                    "SYSTEM LOCKED",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                # Show screen
                cv2.imshow("AI Virtual Mouse", frame)

                # ESC key
                if cv2.waitKey(1) & 0xFF == 27:
                    break

                # Skip all gestures
                continue

            # ==============================
            # Cursor Movement
            # ==============================
            screen_x = screen_w * index.x
            screen_y = screen_h * index.y

            # EMA Smoothing
            smooth_x = alpha * screen_x + (1 - alpha) * smooth_x
            smooth_y = alpha * screen_y + (1 - alpha) * smooth_y

            pyautogui.moveTo(smooth_x, smooth_y)

            # ==============================
            # LEFT CLICK
            # Index + Thumb Close
            # ==============================
            dist_click = math.hypot(
                index_x - thumb_x,
                index_y - thumb_y
            )

            if dist_click < 20:

                gesture_text = "LEFT CLICK"

                pyautogui.click()

                time.sleep(0.15)

            # ==============================
            # RIGHT CLICK
            # Thumb + Middle Close
            # ==============================
            dist_right = math.hypot(
                thumb_x - middle_x,
                thumb_y - middle_y
            )

            if dist_right < 20:

                gesture_text = "RIGHT CLICK"

                pyautogui.rightClick()

                time.sleep(0.15)

            # ==============================
            # SCROLL
            # Index + Middle Close
            # ==============================
            dist_scroll = math.hypot(
                index_x - middle_x,
                index_y - middle_y
            )

            if dist_scroll < 20:

                gesture_text = "SCROLL"

                if prev_y != 0:

                    if index_y - prev_y > 1:
                        pyautogui.scroll(-200)

                    elif prev_y - index_y > 1:
                        pyautogui.scroll(200)

                prev_y = index_y

            else:
                prev_y = 0

            # ==============================
            # SCREENSHOT
            # All Fingers Closed
            # ==============================

            # Detect closed fist
            if fingers == [0,0,0,0,0]:

                current_time = time.time()

                # Cooldown
                if current_time - last_screenshot > 2:

                    filename = f"screenshot_{int(current_time)}.png"

                    pyautogui.screenshot(filename)

                    gesture_text = "SCREENSHOT TAKEN"

                    last_screenshot = current_time

           

    # ==============================
    # Gesture HUD
    # ==============================
    cv2.rectangle(
        frame,
        (10, 10),
        (400, 60),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Gesture : {gesture_text}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # ==============================
    # Display Window
    # ==============================
    cv2.imshow("AI Virtual Mouse", frame)

    # ESC Key to Exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==============================
# Cleanup
# ==============================
cap.release()

cv2.destroyAllWindows()