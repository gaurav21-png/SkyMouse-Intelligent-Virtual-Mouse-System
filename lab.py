import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np

last_lock_time = 0
pyautogui.FAILSAFE = False

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, 640) 
cap.set(4, 480)

# MediaPipe Setup
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

# Screen Size
screen_w, screen_h = pyautogui.size()

# Variables
prev_y = 0
frame_count = 0
smooth_x = 0
smooth_y = 0
alpha = 0.35
gesture_text = "None"
locked = False
last_screenshot = 0
frameR = 100 

# Finger Detection Function
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

# Main Loop
while True:
    success, frame = cap.read()

    if not success:
        continue

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    gesture_text = "Moving"

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, timestamp)

    cv2.rectangle(frame, (frameR, frameR), (w - frameR, h - frameR), (255, 0, 255), 2)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            # Landmark Points
            index = hand_landmarks[8]
            thumb = hand_landmarks[4]
            middle = hand_landmarks[12]

            index_x = int(index.x * w)
            index_y = int(index.y * h)

            thumb_x = int(thumb.x * w)
            thumb_y = int(thumb.y * h)

            middle_x = int(middle.x * w)
            middle_y = int(middle.y * h)

            # Finger Detection
            # fingers[0] = Thumb, fingers[1] = Index, fingers[2] = Middle, fingers[3] = Ring, fingers[4] = Pinky
            fingers = fingers_up(hand_landmarks)
            total_fingers = fingers.count(1)

            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), -1)
            cv2.circle(frame, (middle_x, middle_y), 10, (0, 0, 255), -1)

            # LOCK / UNLOCK SYSTEM
            if 'last_lock_time' not in globals():
                last_lock_time = 0

            current_time = time.time()
            pinky = hand_landmarks[20]
            pinky_x = int(pinky.x * w)
            pinky_y = int(pinky.y * h)

            thumb_pinky_dist = math.hypot(thumb_x - pinky_x, thumb_y - pinky_y)

            if fingers == [0,0,0,0,1]:
                if not locked and current_time - last_lock_time > 2:
                    locked = True
                    gesture_text = "SYSTEM LOCKED"
                    last_lock_time = current_time

            if locked:
                if thumb_pinky_dist < 70:
                    if current_time - last_lock_time > 2:
                        locked = False
                        gesture_text = "SYSTEM UNLOCKED"
                        last_lock_time = current_time

            if locked:
                gesture_text = "SYSTEM LOCKED"
                cv2.putText(frame, "SYSTEM LOCKED", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.imshow("AI Virtual Mouse", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # Cursor Movement 
            screen_x = np.interp(index_x, (frameR, w - frameR), (0, screen_w))
            screen_y = np.interp(index_y, (frameR, h - frameR), (0, screen_h))

            smooth_x = alpha * screen_x + (1 - alpha) * smooth_x
            smooth_y = alpha * screen_y + (1 - alpha) * smooth_y

            pyautogui.moveTo(smooth_x, smooth_y, _pause=False)

            # LEFT & RIGHT CLICK (Strict State)
            # All fingers OPEN
            if fingers == [1, 1, 1, 1, 1]:
                
                # 2D Distance for Left Click (Middle + Index)
                dist_left = math.hypot(thumb_x - index_x, thumb_y - index_y)
                
                # 2D Distance for Right Click (Thumb + Index)
                dist_right = math.hypot(index_x - middle_x, index_y - middle_y)

                # TRIGGER LEFT CLICK
                if dist_left < 30:
                    gesture_text = "LEFT CLICK"
                    pyautogui.click()
                    time.sleep(0.1)
                
                # TRIGGER RIGHT CLICK
                elif dist_right < 30:
                    gesture_text = "RIGHT CLICK"
                    pyautogui.rightClick()
                    time.sleep(0.1)

            # SCROLL (Strict State)
            # Pinky DOWN, Middle + Index UP
            elif fingers[4] == 0 and fingers[1] == 1 and fingers[2] == 1:
                
                # Wait for Index and Middle to connect before scrolling (2D Distance)
                dist_scroll = math.hypot(index_x - middle_x, index_y - middle_y)
                
                if dist_scroll < 30:
                    gesture_text = "SCROLL"
                    if prev_y != 0:
                        if index_y - prev_y > 1:
                            pyautogui.scroll(-200)
                        elif prev_y - index_y > 1:
                            pyautogui.scroll(200)
                    prev_y = index_y
                else:
                    prev_y = 0
            else:
                prev_y = 0 # Instantly reset scroll logic if hand is not in shape

            # SCREENSHOT
            if fingers == [0,0,0,0,0]:
                current_time = time.time()
                if current_time - last_screenshot > 2:
                    filename = f"screenshot_{int(current_time)}.png"
                    pyautogui.screenshot(filename)
                    gesture_text = "SCREENSHOT TAKEN"
                    last_screenshot = current_time

    # Gesture HUD
    cv2.rectangle(frame, (10, 10), (400, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Gesture : {gesture_text}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display Window
    cv2.imshow("AI Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()