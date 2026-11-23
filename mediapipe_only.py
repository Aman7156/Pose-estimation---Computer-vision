import cv2
import mediapipe as mp
import time

VIDEO_IN = "clip1.mp4"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(VIDEO_IN)

total_time = 0
processed_frames = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    start_time = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = pose.process(rgb)

    if result.pose_landmarks:

        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    end_time = time.time()

    total_time += (end_time - start_time)

    processed_frames += 1

    cv2.imshow("MediaPipe Only", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

avg_time = total_time / processed_frames
fps = 1 / avg_time

print("\nPerformance Results:")
print("Frames:", processed_frames)
print("Average time:", round(avg_time, 4))
print("FPS:", round(fps, 2))