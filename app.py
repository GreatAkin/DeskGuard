import customtkinter as ctk
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import sys
from collections import deque

# ----------------- Audio Alert -----------------
def play_alert_sound():
    def _beep():
        try:
            if sys.platform.startswith("win"):
                import winsound
                winsound.Beep(1000, 200)
                winsound.Beep(1400, 300)
            elif sys.platform == "darwin":
                import os
                os.system("afplay /System/Library/Sounds/Ping.aiff")
            else:
                print("\a", end="", flush=True)
        except Exception:
            print("\a", end="", flush=True)

    threading.Thread(target=_beep, daemon=True).start()


# ----------------- Landmarks & Camera Setup -----------------
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# From desk_guard.py
KEY_LANDMARKS = {
    "nose": mp_pose.PoseLandmark.NOSE,
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_ear": mp_pose.PoseLandmark.LEFT_EAR,
    "right_ear": mp_pose.PoseLandmark.RIGHT_EAR,
}

# Eye landmarks for EAR calculation
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    coords = []
    for idx in eye_indices:
        lm = landmarks[idx]
        coords.append(np.array([lm.x * img_w, lm.y * img_h]))

    p1, p2, p3, p4, p5, p6 = coords
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def find_working_camera(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                return cap, i
            cap.release()
    return None, -1


# ----------------- Main GUI Application -----------------
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class PosturaNodeApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("PosturaNode")
        self.geometry("350x660")
        self.resizable(False, False)
        self.attributes("-topmost", True)

        # ---------------- State Variables ----------------
        self.is_slouching = False
        self.slouch_start_time = None
        self.calibrated_distance = None
        self.calibrate_flag = False

        self.phone_pickup_count = 0
        
        # Blink rate tracking
        self.current_bpm = 16
        self.min_healthy_bpm = 10
        self.eye_alert_triggered = False

        self.alert_window = None
        self.is_running = True

        # ---------------- Posture Section ----------------
        self.status_indicator = ctk.CTkLabel(
            self, text="CALIBRATING...", font=ctk.CTkFont(size=16, weight="bold"), text_color="#F39C12"
        )
        self.status_indicator.pack(pady=(16, 0))

        self.score_value = ctk.CTkLabel(
            self, text="--", font=ctk.CTkFont(size=50, weight="bold")
        )
        self.score_value.pack(pady=(0, 6))

        self.calibrate_btn = ctk.CTkButton(
            self,
            text="Calibrate Neutral Posture",
            command=self.request_calibration,
            fg_color="#2980B9",
            hover_color="#3498DB",
        )
        self.calibrate_btn.pack(pady=4, padx=20, fill="x")

        # ---------------- Metrics Frame ----------------
        metrics_frame = ctk.CTkFrame(self)
        metrics_frame.pack(padx=20, pady=12, fill="both", expand=True)

        # 1. Phone Pickups Counter
        phone_title = ctk.CTkLabel(metrics_frame, text="Phone Pickups", font=ctk.CTkFont(size=13, weight="bold"))
        phone_title.pack(pady=(6, 0))

        self.phone_label = ctk.CTkLabel(
            metrics_frame, text="0 times", font=ctk.CTkFont(size=20, weight="bold"), text_color="#3498DB"
        )
        self.phone_label.pack(pady=(0, 6))

        # 2. Blink Rate Tracker
        blink_title = ctk.CTkLabel(metrics_frame, text="Blink Rate (Healthy: 12-20 BPM)", font=ctk.CTkFont(size=13, weight="bold"))
        blink_title.pack(pady=(6, 0))

        self.blink_label = ctk.CTkLabel(
            metrics_frame, text="Calculating...", font=ctk.CTkFont(size=20, weight="bold"), text_color="#4CAF50"
        )
        self.blink_label.pack(pady=(0, 2))

        self.diag_label = ctk.CTkLabel(
            metrics_frame, text="EAR: -- | Blinks: 0", font=ctk.CTkFont(size=11), text_color="#888888"
        )
        self.diag_label.pack(pady=(0, 8))

        # ---------------- Controls ----------------
        self.phone_btn = ctk.CTkButton(
            self,
            text="+1 Phone Pickup (Mock)",
            command=self.increment_phone_pickups,
            height=28,
            fg_color="#2C3E50",
            hover_color="#34495E",
        )
        self.phone_btn.pack(pady=4, padx=20, fill="x")

        self.demo_switch = ctk.CTkSwitch(
            self, text="Demo Mode (Instant Alert)", command=self.trigger_demo_alert
        )
        self.demo_switch.pack(pady=10)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Launch periodic check and unified vision worker
        self.check_intervention_loop()
        self.start_vision_worker()

    # ---------------- Vision Background Thread ----------------
    def start_vision_worker(self):
        worker = threading.Thread(target=self._run_vision_pipeline, daemon=True)
        worker.start()

    def _run_vision_pipeline(self):
        cap, index = find_working_camera()
        if cap is None:
            self.after(0, self.status_indicator.configure, {"text": "CAMERA NOT FOUND", "text_color": "#F44336"})
            return

        EAR_THRESHOLD = 0.21
        CONSEC_FRAMES = 2
        ROLLING_WINDOW_SEC = 60.0

        frame_counter = 0
        total_blinks = 0
        blink_timestamps = deque()

        with mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as pose, mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as face_mesh:

            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.03)
                    continue

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                current_time = time.time()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # --- 1. Blink Extraction (Face Mesh) ---
                face_results = face_mesh.process(rgb_frame)
                current_ear = 0.0
                bpm = 0.0

                if face_results.multi_face_landmarks:
                    mesh_points = face_results.multi_face_landmarks[0].landmark
                    left_ear = calculate_ear(mesh_points, LEFT_EYE_INDICES, w, h)
                    right_ear = calculate_ear(mesh_points, RIGHT_EYE_INDICES, w, h)
                    current_ear = (left_ear + right_ear) / 2.0

                    if current_ear < EAR_THRESHOLD:
                        frame_counter += 1
                    else:
                        if frame_counter >= CONSEC_FRAMES:
                            total_blinks += 1
                            blink_timestamps.append(current_time)
                        frame_counter = 0

                    while blink_timestamps and current_time - blink_timestamps[0] > ROLLING_WINDOW_SEC:
                        blink_timestamps.popleft()

                    if len(blink_timestamps) > 0:
                        window_duration = min(ROLLING_WINDOW_SEC, current_time - blink_timestamps[0] + 1e-5)
                        bpm = (len(blink_timestamps) / window_duration) * 60.0

                # --- 2. Posture Extraction (desk_guard Landmarks) ---
                pose_results = pose.process(rgb_frame)
                is_slouch = False
                posture_score = 98

                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark
                    points = {}

                    for name, idx in KEY_LANDMARKS.items():
                        lm = landmarks[idx.value]
                        if lm.visibility > 0.5:
                            points[name] = (int(lm.x * w), int(lm.y * h))

                    if "left_shoulder" in points and "right_shoulder" in points:
                        mid_shoulder_y = (points["left_shoulder"][1] + points["right_shoulder"][1]) / 2.0
                        
                        head_y = None
                        if "nose" in points:
                            head_y = points["nose"][1]
                        elif "left_ear" in points and "right_ear" in points:
                            head_y = (points["left_ear"][1] + points["right_ear"][1]) / 2.0

                        if head_y is not None:
                            head_vert_dist = mid_shoulder_y - head_y

                            if self.calibrate_flag or self.calibrated_distance is None:
                                self.calibrated_distance = head_vert_dist
                                self.calibrate_flag = False

                            if head_vert_dist < (self.calibrated_distance * 0.82):
                                is_slouch = True
                                ratio = max(0.0, head_vert_dist / self.calibrated_distance)
                                posture_score = int(np.clip(ratio * 90, 30, 65))
                            else:
                                is_slouch = False
                                posture_score = int(np.clip(90 + (head_vert_dist / self.calibrated_distance) * 10, 85, 100))

                # Safe dispatch to UI thread
                self.after(
                    0,
                    self.update_live_ui,
                    round(bpm, 1),
                    round(current_ear, 2),
                    total_blinks,
                    is_slouch,
                    posture_score,
                )

                time.sleep(0.02)

        cap.release()

    # ---------------- Thread-Safe UI Dispatch ----------------
    def request_calibration(self):
        self.calibrate_flag = True

    def update_live_ui(self, bpm: float, ear: float, blinks: int, is_slouch: bool, score: int):
        # 1. Update Posture Display
        if self.calibrated_distance is None:
            self.status_indicator.configure(text="SIT STRAIGHT & CALIBRATE", text_color="#F39C12")
            self.score_value.configure(text="--")
        elif is_slouch:
            self.status_indicator.configure(text="SLOUCHING DETECTED", text_color="#F44336")
            self.score_value.configure(text=str(score))
            if not self.is_slouching:
                self.is_slouching = True
                self.slouch_start_time = time.time()
        else:
            self.status_indicator.configure(text="GOOD POSTURE", text_color="#4CAF50")
            self.score_value.configure(text=str(score))
            self.is_slouching = False
            self.slouch_start_time = None

        # 2. Update Blink Rate & Low Blink Pop-up Trigger
        self.current_bpm = bpm
        self.diag_label.configure(text=f"EAR: {ear:.2f} | Blinks: {blinks}")

        if bpm < self.min_healthy_bpm and blinks >= 2:
            self.blink_label.configure(text=f"{bpm:.0f} BPM (Low!)", text_color="#F44336")
            if not self.eye_alert_triggered:
                self.eye_alert_triggered = True
                self.show_intervention_alert(
                    title_text="Rest Your Eyes & Get Up! 👀",
                    message_text=f"Your blink rate dropped to {bpm:.0f} BPM (Healthy: 12-20).\nStand up and look 20 feet away for 20 seconds.",
                )
        else:
            self.blink_label.configure(text=f"{bpm:.0f} BPM", text_color="#4CAF50")
            if bpm >= self.min_healthy_bpm:
                self.eye_alert_triggered = False

    def increment_phone_pickups(self):
        """Call this function when phone detection triggers."""
        self.phone_pickup_count += 1
        self.phone_label.configure(text=f"{self.phone_pickup_count} times")

    def trigger_demo_alert(self):
        if self.demo_switch.get():
            self.show_intervention_alert(
                title_text="Demo Alert 🧘",
                message_text="Instant demo triggered!\nStand up and stretch.",
            )

    def check_intervention_loop(self):
        # 45-second slouch intervention rule (use 5 for testing)
        if self.is_slouching and self.slouch_start_time:
            if time.time() - self.slouch_start_time >= 45:
                self.show_intervention_alert(
                    title_text="Time for a Stretch Break! 🧘",
                    message_text="You have been slouching for 45 seconds.\nSit up straight and roll your shoulders.",
                )
                self.slouch_start_time = None

        self.after(1000, self.check_intervention_loop)

    def show_intervention_alert(self, title_text: str, message_text: str):
        if self.alert_window is not None and self.alert_window.winfo_exists():
            return

        play_alert_sound()

        self.alert_window = ctk.CTkToplevel(self)
        self.alert_window.title("Health Intervention Alert")
        self.alert_window.geometry("500x300")
        self.alert_window.attributes("-topmost", True)

        self.alert_window.update_idletasks()
        x = (self.alert_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (self.alert_window.winfo_screenheight() // 2) - (300 // 2)
        self.alert_window.geometry(f"+{x}+{y}")

        title = ctk.CTkLabel(self.alert_window, text=title_text, font=ctk.CTkFont(size=22, weight="bold"))
        title.pack(pady=(45, 12))

        msg = ctk.CTkLabel(self.alert_window, text=message_text, font=ctk.CTkFont(size=14), justify="center")
        msg.pack(pady=10)

        dismiss_btn = ctk.CTkButton(
            self.alert_window, text="Got it, taking a break", command=self.alert_window.destroy, width=170, height=36
        )
        dismiss_btn.pack(pady=20)

    def on_closing(self):
        self.is_running = False
        self.destroy()


if __name__ == "__main__":
    app = PosturaNodeApp()
    app.mainloop()