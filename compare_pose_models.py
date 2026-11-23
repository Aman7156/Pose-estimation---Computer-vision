import argparse
import csv
import math
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
N_LANDMARKS = 33


class OneEuroFilter:
    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.3, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(freq, cutoff):
        te = 1.0 / freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.freq, self.dcutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(self.freq, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class KeypointSmoother:
    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.3):
        self.filters = [
            [OneEuroFilter(freq, mincutoff, beta) for _ in range(2)]
            for _ in range(N_LANDMARKS)
        ]

    def smooth(self, points_xy):
        out = np.empty_like(points_xy)
        for i in range(N_LANDMARKS):
            out[i, 0] = self.filters[i][0](points_xy[i, 0])
            out[i, 1] = self.filters[i][1](points_xy[i, 1])
        return out


def iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class IouTracker:
    def __init__(self, iou_threshold=0.3, max_missed=15):
        self.tracks = {}
        self.missed = {}
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

    def update(self, boxes):
        assigned = {}
        used_tracks = set()
        for box in boxes:
            best_id, best_iou = None, self.iou_threshold
            for tid, tbox in self.tracks.items():
                if tid in used_tracks:
                    continue
                score = iou(box, tbox)
                if score > best_iou:
                    best_id, best_iou = tid, score
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            assigned[best_id] = box
            used_tracks.add(best_id)

        for tid in list(self.tracks.keys()):
            if tid not in assigned:
                self.missed[tid] = self.missed.get(tid, 0) + 1
                if self.missed[tid] > self.max_missed:
                    self.tracks.pop(tid, None)
                    self.missed.pop(tid, None)

        for tid, box in assigned.items():
            self.tracks[tid] = box
            self.missed[tid] = 0

        return list(assigned.items())


def laplacian_sharpness(gray_crop):
    if gray_crop.size == 0:
        return 0.0
    return cv2.Laplacian(gray_crop, cv2.CV_64F).var()


def guess_failure_causes(box, all_boxes, frame_shape, gray_frame,
                          blur_thresh=60.0, min_box_px=70, occlusion_iou=0.15):
    causes = []
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1

    if min(h, w) < min_box_px:
        causes.append("low_resolution")

    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(frame_shape[1], x2), min(frame_shape[0], y2)
    crop = gray_frame[y1c:y2c, x1c:x2c]
    if laplacian_sharpness(crop) < blur_thresh:
        causes.append("motion_blur")

    for other in all_boxes:
        if other is box:
            continue
        if iou(box, other) > occlusion_iou:
            causes.append("occlusion")
            break

    if len(all_boxes) > 1:
        causes.append("multiple_players")

    return causes or ["unknown"]


TRACK_COLORS = [
    (66, 135, 245), (245, 66, 173), (66, 245, 132), (245, 197, 66),
    (170, 66, 245), (245, 66, 66), (66, 245, 224), (150, 150, 150),
]


def color_for(track_id):
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def draw_skeleton(frame, points_xy, color, thickness=2):
    for a, b in POSE_CONNECTIONS:
        pa, pb = points_xy[a], points_xy[b]
        if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
            continue
        cv2.line(frame, tuple(pa.astype(int)), tuple(pb.astype(int)), color, thickness)
    for pt in points_xy:
        if np.any(np.isnan(pt)):
            continue
        cv2.circle(frame, tuple(pt.astype(int)), 2, color, -1)


def main():
    ap = argparse.ArgumentParser(description="Phase 3: optimised, smoothed, analysed skeletal tracking")
    ap.add_argument("--source", required=True, help="0 for webcam, or path to a video file")
    ap.add_argument("--yolo-model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--outdir", default="phase3_outputs")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--mincutoff", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.3)

    ap.add_argument("--resize-scale", type=float, default=1.0,
                     help="Scale factor applied to each frame before detection/pose, "
                          "e.g. 0.5 = half resolution. Used to study the speed/accuracy trade-off.")
    ap.add_argument("--model-complexity", type=int, default=0, choices=[0, 1, 2],
                     help="MediaPipe Pose model_complexity: 0=lite, 1=full, 2=heavy. "
                          "Used to compare lightweight vs heavier pose models.")
    ap.add_argument("--run-name", default=None,
                     help="Label for this configuration, used in the shared comparison_results.csv. "
                          "Defaults to a name built from resize-scale and model-complexity.")
    ap.add_argument("--comparison-csv", default="comparison_results.csv",
                     help="Shared CSV that accumulates one summary row per run, across multiple invocations.")
    args = ap.parse_args()

    run_name = args.run_name or f"scale{args.resize_scale}_complexity{args.model_complexity}"

    os.makedirs(args.outdir, exist_ok=True)
    source = 0 if args.source == "0" else args.source

    print(f"[phase3] run: {run_name}")
    print(f"[phase3] loading YOLO detector ({args.yolo_model}) ...")
    detector = YOLO(args.yolo_model)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"[phase3] could not open source: {args.source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    proc_w = max(1, int(frame_w * args.resize_scale))
    proc_h = max(1, int(frame_h * args.resize_scale))

    out_video_path = os.path.join(args.outdir, f"annotated_{run_name}.mp4")
    writer = cv2.VideoWriter(
        out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (frame_w, frame_h)
    )

    csv_path = os.path.join(args.outdir, f"frame_logs_{run_name}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "track_id", "hit", "causes",
        "raw_jitter", "smoothed_jitter", "frame_time_s"
    ])

    tracker = IouTracker()
    smoothers = {}
    raw_prev = {}
    smooth_prev = {}

    shared_pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=args.model_complexity,
        min_detection_confidence=0.5,
    )

    miss_cause_counts = defaultdict(int)
    total_detections = 0
    total_hits = 0
    raw_jitters, smoothed_jitters = [], []
    frame_idx = 0
    t_start = time.time()
    fps_hist = []

    print("[phase3] running. Press 'q' or Esc in the video window to stop early.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if args.max_frames and frame_idx > args.max_frames:
            break

        t0 = time.time()

        if args.resize_scale != 1.0:
            proc_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        else:
            proc_frame = frame

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = detector(proc_frame, classes=[0], conf=args.conf, verbose=False)[0]
        raw_boxes = [tuple(map(int, b)) for b in results.boxes.xyxy.cpu().numpy()] if len(results.boxes) else []

        inv_scale = 1.0 / args.resize_scale if args.resize_scale != 1.0 else 1.0
        boxes = [tuple(int(v * inv_scale) for v in b) for b in raw_boxes]

        tracked = tracker.update(boxes)

        active_ids = set()
        for track_id, box in tracked:
            active_ids.add(track_id)
            total_detections += 1
            x1, y1, x2, y2 = box
            pad = 15
            cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
            cx2, cy2 = min(frame_w, x2 + pad), min(frame_h, y2 + pad)
            crop = frame[cy1:cy2, cx1:cx2]

            if track_id not in smoothers:
                smoothers[track_id] = KeypointSmoother(
                    freq=src_fps, mincutoff=args.mincutoff, beta=args.beta
                )

            hit = False
            causes = []
            raw_jit = smooth_jit = float("nan")

            if crop.size > 0:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                res = shared_pose.process(rgb)

                if res.pose_landmarks:
                    hit = True
                    total_hits += 1
                    ch, cw = crop.shape[:2]
                    raw_pts = np.array(
                        [[lm.x * cw + cx1, lm.y * ch + cy1] for lm in res.pose_landmarks.landmark],
                        dtype=np.float64,
                    )
                    smooth_pts = smoothers[track_id].smooth(raw_pts)

                    if track_id in raw_prev:
                        raw_jit = float(np.linalg.norm(raw_pts - raw_prev[track_id], axis=1).mean())
                        smooth_jit = float(np.linalg.norm(smooth_pts - smooth_prev[track_id], axis=1).mean())
                        raw_jitters.append(raw_jit)
                        smoothed_jitters.append(smooth_jit)
                    raw_prev[track_id] = raw_pts
                    smooth_prev[track_id] = smooth_pts

                    col = color_for(track_id)
                    draw_skeleton(frame, smooth_pts, col, thickness=2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 1)
                    cv2.putText(frame, f"ID {track_id}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            if not hit:
                causes = guess_failure_causes(box, boxes, frame.shape, gray_frame)
                for c in causes:
                    miss_cause_counts[c] += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"MISS: {','.join(causes)}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            csv_writer.writerow([frame_idx, track_id, int(hit), "|".join(causes), raw_jit, smooth_jit, round(time.time() - t0, 4)])

        for tid in list(smoothers.keys()):
            if tid not in active_ids and tracker.missed.get(tid, 999) > tracker.max_missed:
                smoothers.pop(tid, None)
                raw_prev.pop(tid, None)
                smooth_prev.pop(tid, None)

        dt = time.time() - t0
        fps_hist.append(1.0 / dt if dt > 0 else 0.0)
        if len(fps_hist) > 30:
            fps_hist.pop(0)
        live_fps = sum(fps_hist) / len(fps_hist)

        hit_rate = (total_hits / total_detections * 100) if total_detections else 0.0
        overlay = f"[{run_name}] frame {frame_idx} | fps {live_fps:.1f} | persons {len(tracked)} | hit-rate {hit_rate:.1f}%"
        cv2.putText(frame, overlay, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        writer.write(frame)

        if frame_idx % 15 == 0 or frame_idx == 1:
            print(f"[phase3] {overlay} | misses-so-far {total_detections - total_hits}")

        if not args.no_display:
            cv2.imshow(f"Phase 3 - {run_name}", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[phase3] stopped by user.")
                break

    elapsed = time.time() - t_start
    cap.release()
    writer.release()
    csv_file.close()
    shared_pose.close()
    if not args.no_display:
        cv2.destroyAllWindows()

    avg_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    avg_raw_jit = float(np.mean(raw_jitters)) if raw_jitters else float("nan")
    avg_smooth_jit = float(np.mean(smoothed_jitters)) if smoothed_jitters else float("nan")
    jitter_reduction = (
        (1 - avg_smooth_jit / avg_raw_jit) * 100
        if raw_jitters and avg_raw_jit > 0 else float("nan")
    )
    hit_rate = (total_hits / total_detections * 100) if total_detections else 0.0
    total_misses = total_detections - total_hits

    lines = []
    lines.append("=" * 60)
    lines.append(f"PHASE 3 SUMMARY - {run_name}")
    lines.append("=" * 60)
    lines.append(f"Resize scale:            {args.resize_scale}")
    lines.append(f"Pose model complexity:   {args.model_complexity}")
    lines.append(f"Frames processed:        {frame_idx}")
    lines.append(f"Elapsed time (s):        {elapsed:.1f}")
    lines.append(f"Average FPS:             {avg_fps:.2f}")
    lines.append(f"Total person-detections: {total_detections}")
    lines.append(f"Pose hits:               {total_hits}  ({hit_rate:.1f}%)")
    lines.append(f"Pose misses:             {total_misses}  ({100 - hit_rate:.1f}%)")
    lines.append("")
    lines.append("-- Temporal smoothing (1-Euro filter) --")
    lines.append(f"Average raw jitter (px/frame):      {avg_raw_jit:.3f}")
    lines.append(f"Average smoothed jitter (px/frame): {avg_smooth_jit:.3f}")
    lines.append(f"Jitter reduction:                   {jitter_reduction:.1f}%")
    lines.append("")
    lines.append("-- Miss-cause breakdown (a miss can have >1 cause) --")
    for cause, count in sorted(miss_cause_counts.items(), key=lambda kv: -kv[1]):
        pct = (count / total_misses * 100) if total_misses else 0.0
        lines.append(f"  {cause:<18} {count:>6}   ({pct:.1f}% of misses)")
    lines.append("=" * 60)

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    summary_path = os.path.join(args.outdir, f"summary_{run_name}.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")

    print(f"\n[phase3] annotated video: {out_video_path}")
    print(f"[phase3] per-frame log:   {csv_path}")
    print(f"[phase3] summary:         {summary_path}")

    comparison_row = {
        "run_name": run_name,
        "resize_scale": args.resize_scale,
        "model_complexity": args.model_complexity,
        "frames_processed": frame_idx,
        "elapsed_s": round(elapsed, 1),
        "avg_fps": round(avg_fps, 2),
        "total_detections": total_detections,
        "hits": total_hits,
        "hit_rate_pct": round(hit_rate, 1),
        "avg_raw_jitter_px": round(avg_raw_jit, 3) if raw_jitters else "",
        "avg_smoothed_jitter_px": round(avg_smooth_jit, 3) if raw_jitters else "",
        "jitter_reduction_pct": round(jitter_reduction, 1) if raw_jitters else "",
    }
    write_header = not os.path.exists(args.comparison_csv)
    with open(args.comparison_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comparison_row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(comparison_row)
    print(f"[phase3] appended summary row to: {args.comparison_csv}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            causes, counts = zip(*sorted(miss_cause_counts.items(), key=lambda kv: -kv[1])) if miss_cause_counts else ((), ())
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(causes, counts, color="#e05a5a")
            ax.set_title(f"Pose-tracking miss causes - {run_name}")
            ax.set_ylabel("count")
            fig.tight_layout()
            plot_path = os.path.join(args.outdir, f"miss_causes_{run_name}.png")
            fig.savefig(plot_path)
            print(f"[phase3] miss-cause plot: {plot_path}")
        except Exception as e:
            print(f"[phase3] WARNING: could not save plot ({e}). Summary/CSV/video are unaffected.")


if __name__ == "__main__":
    main()