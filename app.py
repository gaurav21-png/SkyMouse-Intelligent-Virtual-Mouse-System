import customtkinter as ctk
import threading

# ------------------ VIRTUAL MOUSE CODE ------------------ #
def run_app():
    import cv2
    import mediapipe as mp
    import pyautogui
    import time
    import math

    pyautogui.FAILSAFE = False

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # MediaPipe setup
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1
    )

    detector = HandLandmarker.create_from_options(options)

    screen_w, screen_h = pyautogui.size()
    prev_y = 0
    prev_zoom_dist = 0
    frame_count = 0

    while True:

        frame_count += 1
        if frame_count % 3!= 0:
            continue

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.flip(frame,1)

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp = int(time.time()*1000)

        result = detector.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:

            for hand_landmarks in result.hand_landmarks:

                index = hand_landmarks[8]
                thumb = hand_landmarks[4]
                middle = hand_landmarks[12]

                index_x = int(index.x * w)
                index_y = int(index.y * h)

                thumb_x = int(thumb.x * w)
                thumb_y = int(thumb.y * h)

                middle_x = int(middle.x * w)
                middle_y = int(middle.y * h)

                # Draw points
                cv2.circle(frame,(index_x,index_y),10,(0,255,0),-1)
                cv2.circle(frame,(thumb_x,thumb_y),10,(255,0,0),-1)

                # Cursor movement
                screen_x = screen_w * index.x
                screen_y = screen_h * index.y

                pyautogui.moveTo(screen_x, screen_y, duration=0)

                # Left click
                dist_click = math.hypot(index_x-thumb_x,index_y-thumb_y)

                if dist_click < 20:
                    pyautogui.click()
                    time.sleep(0.1)

                # Right click
                dist_right = math.hypot(thumb_x-middle_x,thumb_y-middle_y)

                if dist_right < 20:
                    pyautogui.rightClick()
                    time.sleep(0.1)

                # Scroll gesture
                dist_scroll = math.hypot(index_x - middle_x, index_y - middle_y)

                if dist_scroll < 20:   # fingers close → scroll mode
                    
                    if prev_y != 0:
                        if index_y - prev_y > 1:
                            pyautogui.scroll(-200)   # scroll down
                        
                        elif prev_y - index_y > 1:
                            pyautogui.scroll(200)    # scroll up

                    prev_y = index_y

                else:
                    prev_y = 0
        cv2.imshow("AI Virtual Mouse", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break        


    cap.release()
    cv2.destroyAllWindows()


# ------------------ UI CODE ------------------ #

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

def start_app():
    status_label.configure(text="Status: Running", text_color="green")
    threading.Thread(target=run_app).start()

def stop_app():
    status_label.configure(text="Status: Stopped", text_color="red")

app = ctk.CTk()
app.title("AI Virtual Mouse")
app.geometry("600x400")

# Title
title = ctk.CTkLabel(app, text="AI Virtual Mouse", font=("Arial", 24, "bold"))
title.pack(pady=20)

# Main frame
frame = ctk.CTkFrame(app)
frame.pack(pady=10, padx=20, fill="both", expand=True)

# Controls
control_frame = ctk.CTkFrame(frame)
control_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

ctk.CTkLabel(control_frame, text="Controls", font=("Arial", 18)).pack(pady=10)

start_btn = ctk.CTkButton(control_frame, text="Start", command=start_app)
start_btn.pack(pady=10)

stop_btn = ctk.CTkButton(control_frame, text="Stop", command=stop_app)
stop_btn.pack(pady=10)

exit_btn = ctk.CTkButton(control_frame, text="Exit", command=app.destroy)
exit_btn.pack(pady=10)

# Info panel
info_frame = ctk.CTkFrame(frame)
info_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

ctk.CTkLabel(info_frame, text="Instructions", font=("Arial", 18)).pack(pady=10)

instructions = """
👆 Move Index Finger → Move Cursor
🤏 Index + Thumb → Left Click
✌ Thumb + Middle → Right Click
👇 Move fingers → Scroll
"""

ctk.CTkLabel(info_frame, text=instructions, justify="left").pack(pady=10)

# Status
status_label = ctk.CTkLabel(app, text="Status: Stopped", font=("Arial", 12))
status_label.pack(pady=10)

app.mainloop()