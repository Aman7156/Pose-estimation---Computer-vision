import argparse
import os
import cv2
import pandas as pd

def pick_frames(log_df, hit_col="hit", cause_col="causes"):
    picks = {}

    hits = log_df[log_df[hit_col] == 1]
    if not hits.empty:
        picks["clean_hit"] = int(hits.iloc[len(hits) // 2]["frame"])

    misses = log_df[log_df[hit_col] == 0]
    for cause in ["low_resolution", "multiple_players", "occlusion", "motion_blur"]:
        rows = misses[misses[cause_col].fillna("").str.contains(cause)]
        if not rows.empty:
            picks[f"miss_{cause}"] = int(rows.iloc[len(rows) // 2]["frame"])

    return picks

def extract_frames(video_path, frame_numbers, outdir, prefix):
    os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video_path}")

    saved = {}
    for label, frame_no in frame_numbers.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no - 1))
        ok, frame = cap.read()
        if not ok:
            print(f"  could not read frame {frame_no} for {label}, skipping")
            continue
        out_path = os.path.join(outdir, f"{prefix}_{label}_frame{frame_no}.jpg")
        cv2.imwrite(out_path, frame)
        saved[label] = out_path
        print(f"  saved {label} (frame {frame_no}) -> {out_path}")

    cap.release()
    return saved

def main():
    ap = argparse.ArgumentParser(description="Pick and extract representative frames for Phase 5 qualitative analysis")
    ap.add_argument("--video", required=True, help="path to the annotated video, e.g. phase3_outputs/annotated_clip3_full_res_lite.mp4")
    ap.add_argument("--log-csv", required=True, help="matching frame log csv, e.g. phase3_outputs/frame_logs_clip3_full_res_lite.csv")
    ap.add_argument("--outdir", default="phase5_frames")
    ap.add_argument("--prefix", default="run")
    args = ap.parse_args()

    df = pd.read_csv(args.log_csv)
    picks = pick_frames(df)

    if not picks:
        print("no usable hit or miss rows found in this log")
        return

    print(f"picked frames for {args.prefix}:")
    for label, frame_no in picks.items():
        print(f"  {label}: frame {frame_no}")

    extract_frames(args.video, picks, args.outdir, args.prefix)
    print(f"\ndone. Images saved in {args.outdir}/")


if __name__ == "__main__":
    main()