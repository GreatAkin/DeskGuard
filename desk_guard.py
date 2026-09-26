import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

# Landmarks we care about, mapped to MediaPipe's landmark indices
KEY_LANDMARKS = {
    "nose": mp_pose.PoseLandmark.NOSE,
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_ear": mp_pose.PoseLandmark.LEFT_EAR,
    "right_ear": mp_pose.PoseLandmark.RIGHT_EAR,
}

# Which pairs of our key points to connect with lines
SKELETON_CONNECTIONS = [
    ("left_ear", "left_shoulder"),
    ("right_ear", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
]

def find_working_camera(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                return cap, i
            cap.release()
    return None, -1

def main():
    cap, index = find_working_camera()
    if cap is None:
        print("Error: No working camera found.")
        return

    print(f"Using camera index {index}. Press 'q' to quit.")

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                h, w, _ = frame.shape
                landmarks = results.pose_landmarks.landmark

                # Compute pixel coordinates for each key point (only if visible enough)
                points = {}
                for name, idx in KEY_LANDMARKS.items():
                    lm = landmarks[idx.value]
                    if lm.visibility > 0.5:
                        points[name] = (int(lm.x * w), int(lm.y * h))

                # Draw skeleton lines first (so dots/labels sit on top)
                for start_name, end_name in SKELETON_CONNECTIONS:
                    if start_name in points and end_name in points:
                        cv2.line(frame, points[start_name], points[end_name], (0, 255, 0), 2)

                # Draw joints + labels
                for name, (x_px, y_px) in points.items():
                    cv2.circle(frame, (x_px, y_px), 6, (0, 0, 255), -1)
                    cv2.putText(
                        frame, name, (x_px + 8, y_px - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                    )

            cv2.imshow("Pose Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
