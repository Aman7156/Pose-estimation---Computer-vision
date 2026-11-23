import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import mediapipe as mp
import csv
import numpy as np
from ultralytics import YOLO
from collections import OrderedDict


class CentroidTracker:
    def __init__(self, max_disappeared=20):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]

    def update(self, centroids):
        if not centroids:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        if not self.objects:
            for c in centroids:
                self.register(c)
        else:
            ids  = list(self.objects.keys())
            prev = list(self.objects.values())
            D    = np.linalg.norm(
                np.array(prev)[:, None] - np.array(centroids)[None, :], axis=2
            )
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_r, used_c = set(), set()

            for r, c in zip(rows, cols):
                if r in used_r or c in used_c:
                    continue
                oid = ids[r]
                self.objects[oid] = centroids[c]
                self.disappeared[oid] = 0
                used_r.add(r)
                used_c.add(c)

            for r in set(range(len(prev))) - used_r:
                self.disappeared[ids[r]] += 1
                if self.disappeared[ids[r]] > self.max_disappeared:
                    self.deregister(ids[r])

            for c in set(range(len(centroids))) - used_c:
                self.register(centroids[c])

        return self.objects


VIDEO_IN  = "clip3.mp4"
VIDEO_OUT = "output_pose.mp4"
CSV_OUT   = "keypoints.csv"

YOLO_CONF     = 0.35
POSE_CONF_DET = 0.5
POSE_CONF_TRK = 0.5
CROP_PAD      = 30
MIN_BOX_H     = 40

COLOURS = [
    (0, 255, 120), (255, 100, 50), (50, 180, 255),
    (255, 220, 0), (180, 0, 255), (0, 200, 200),
    (255, 80, 180), (80, 255, 200), (200, 120, 0),
    (100, 100, 255),
]

def player_colour(pid):
    return COLOURS[pid % len(COLOURS)]


print("Loading YOLOv8n...")
yolo = YOLO("yolov8n.pt")

print("Loading MediaPipe Pose...")
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=POSE_CONF_DET,
    min_tracking_confidence=POSE_CONF_TRK,
    smooth_landmarks=True,
)

cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open {VIDEO_IN}")

W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS   = cap.get(cv2.CAP_PROP_FPS) or 25.0
TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {W}×{H} @ {FPS:.1f} fps — {TOTAL} frames")

out     = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
tracker = CentroidTracker(max_disappeared=20)

frame_idx = 0

with open(CSV_OUT, "w", newline="") as f:
    csv_writer = csv.writer(f)

    header = ["frame", "player_id", "box_x1", "box_y1", "box_x2", "box_y2"]
    for i in range(33):
        header += [f"x{i}", f"y{i}", f"z{i}", f"vis{i}"]
    csv_writer.writerow(header)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        yolo_results = yolo(frame, classes=[0], conf=YOLO_CONF, verbose=False)[0]

        boxes     = []
        centroids = []

        for box in yolo_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (y2 - y1) < MIN_BOX_H:
                continue
            boxes.append((x1, y1, x2, y2))
            centroids.append(((x1 + x2) // 2, (y1 + y2) // 2))

        objects = tracker.update(centroids)

        for x1, y1, x2, y2 in boxes:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if not objects:
                continue

            pid = min(objects, key=lambda k: np.linalg.norm(
                np.array(objects[k]) - np.array((cx, cy))
            ))
            colour = player_colour(pid)

            cx1 = max(0, x1 - CROP_PAD)
            cy1 = max(0, y1 - CROP_PAD)
            cx2 = min(W, x2 + CROP_PAD)
            cy2 = min(H, y2 + CROP_PAD)
            crop = frame[cy1:cy2, cx1:cx2]

            if crop.size == 0:
                continue

            cH, cW = crop.shape[:2]
            scale = 1.0
            if cH < 200:
                scale = 200 / cH
                crop  = cv2.resize(crop, (int(cW * scale), int(cH * scale)),
                                   interpolation=cv2.INTER_LINEAR)

            scaled_cW = int(cW * scale)
            scaled_cH = int(cH * scale)

            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result   = pose.process(rgb_crop)

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)
            cv2.putText(frame, f"P{pid}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

            if result.pose_landmarks:
                lms = result.pose_landmarks.landmark

                for lm in lms:
                    px = int(cx1 + lm.x * scaled_cW / scale)
                    py = int(cy1 + lm.y * scaled_cH / scale)
                    if lm.visibility > 0.4:
                        cv2.circle(frame, (px, py), 3, colour, -1)

                for conn in mp_pose.POSE_CONNECTIONS:
                    a, b = conn
                    la, lb = lms[a], lms[b]
                    if la.visibility > 0.4 and lb.visibility > 0.4:
                        pax = int(cx1 + la.x * scaled_cW / scale)
                        pay = int(cy1 + la.y * scaled_cH / scale)
                        pbx = int(cx1 + lb.x * scaled_cW / scale)
                        pby = int(cy1 + lb.y * scaled_cH / scale)
                        cv2.line(frame, (pax, pay), (pbx, pby), colour, 1)

                row = [frame_idx, pid, x1, y1, x2, y2]
                for lm in lms:
                    abs_x = cx1 + lm.x * scaled_cW / scale
                    abs_y = cy1 + lm.y * scaled_cH / scale
                    row += [round(abs_x / W, 5), round(abs_y / H, 5),
                            round(lm.z, 5), round(lm.visibility, 3)]
                csv_writer.writerow(row)
            else:
                csv_writer.writerow(
                    [frame_idx, pid, x1, y1, x2, y2] + [0] * (33 * 4)
                )

        hud = f"Frame {frame_idx}/{TOTAL}  |  Players detected: {len(boxes)}"
        cv2.putText(frame, hud, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        out.write(frame)
        cv2.imshow("SoccerNet Pose Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\nDone. Output: {VIDEO_OUT} | CSV: {CSV_OUT}")
