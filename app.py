import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

RUNS = {
    "Baseline": {
        "video": "output_pose_web.mp4",
        "keypoints_csv": "keypoints.csv",
        "format": "keypoints",  
    },
    "Phase 3 - clip1, full res, lite model": {
        "video": "phase3_outputs/annotated_clip1_full_res_lite.mp4",
        "keypoints_csv": "phase3_outputs/frame_logs_clip1_full_res_lite.csv",
        "format": "framelog",
    },
    "Phase 3 - clip1, half res, lite model": {
        "video": "phase3_outputs/annotated_clip1_half_res_lite.mp4",
        "keypoints_csv": "phase3_outputs/frame_logs_clip1_half_res_lite.csv",
        "format": "framelog",
    },
    "Phase 3 - clip1, full res, full model": {
        "video": "phase3_outputs/annotated_clip1_full_res_full.mp4",
        "keypoints_csv": "phase3_outputs/frame_logs_clip1_full_res_full.csv",
        "format": "framelog",
    },
    "Phase 3 - clip2, full res, lite model": {
        "video": "phase3_outputs/annotated_clip2_full_res_lite.mp4",
        "keypoints_csv": "phase3_outputs/frame_logs_clip2_full_res_lite.csv",
        "format": "framelog",
    },
    "Phase 3 - clip2, half res, lite model": {
        "video": "phase3_outputs/annotated_clip2_half_res_lite.mp4",
        "keypoints_csv": "phase3_outputs/frame_logs_clip2_half_res_lite.csv",
        "format": "framelog",
    },
    "Phase 3 - clip2, full res, full model": {
        "video": "phase3_outputs/annotated_clip2_full_res_full.mp4",
        "keypoints_csv": "phase3_outputs/frame_logs_clip2_full_res_full.csv",
        "format": "framelog",
    },
    "Phase 3 - clip3, full res, lite model": {
        "video": "phase3_outputs/annotated_clip3_full_res_lite.mp4",
        "keypoints_csv": "phase3_outputs/frame_logs_clip3_full_res_lite.csv",
        "format": "framelog",
    },
    "Phase 3 - clip3, half res, lite model": {
        "video": "phase3_outputs/annotated_clip3_half_res_lite.mp4",
        "keypoints_csv": "phase3_outputs/frame_logs_clip3_half_res_lite.csv",
        "format": "framelog",
    },
    "Phase 3 - clip3, full res, full model": {
        "video": "phase3_outputs/annotated_clip3_full_res_full.mp4",
        "keypoints_csv": "phase3_outputs/frame_logs_clip3_full_res_full.csv",
        "format": "framelog",
    },
}

JOINTS_OF_INTEREST = {
    "Left Hip": 23,
    "Right Hip": 24,
    "Left Knee": 25,
    "Right Knee": 26,
    "Left Shoulder": 11,
    "Right Shoulder": 12,
}

VISIBILITY_WARN_THRESHOLD = 0.5   
MIN_DETECTION_RATE_WARN = 0.7     


@st.cache_data
def load_keypoints(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df


def compute_quality_stats(df, run_format="keypoints"):

    empty = {
        "frames_with_detections": 0,
        "avg_players_per_frame": 0.0,
        "avg_visibility": np.nan,
        "hit_rate": np.nan,
        "top_miss_causes": None,
    }
    if df is None or df.empty:
        return empty

    if run_format == "framelog":
        total = len(df)
        hits = df["hit"].sum() if "hit" in df.columns else np.nan
        hit_rate = (hits / total * 100) if total else np.nan
        frames_with_detections = df["frame"].nunique()
        avg_players_per_frame = df.groupby("frame")["track_id"].nunique().mean()

        top_causes = None
        if "causes" in df.columns:
            misses = df[df["hit"] == 0]["causes"].dropna()
            if not misses.empty:
                exploded = misses.str.split("|").explode()
                top_causes = exploded.value_counts()

        return {
            "frames_with_detections": frames_with_detections,
            "avg_players_per_frame": avg_players_per_frame,
            "avg_visibility": np.nan,  
            "hit_rate": hit_rate,
            "top_miss_causes": top_causes,
        }

    frames_with_detections = df["frame"].nunique()
    avg_players_per_frame = df.groupby("frame")["player_id"].nunique().mean()
    vis_cols = [c for c in df.columns if c.startswith("vis")]
    avg_visibility = df[vis_cols].mean().mean() if vis_cols else np.nan

    return {
        "frames_with_detections": frames_with_detections,
        "avg_players_per_frame": avg_players_per_frame,
        "avg_visibility": avg_visibility,
        "hit_rate": np.nan,
        "top_miss_causes": None,
    }


def get_joint_xy(df, joint_index, player_id=None):
    x_col, y_col = f"x{joint_index}", f"y{joint_index}"
    if x_col not in df.columns or y_col not in df.columns:
        return None

    sub = df if player_id is None else df[df["player_id"] == player_id]
    sub = sub.sort_values("frame")
    return sub[["frame", x_col, y_col]].rename(columns={x_col: "x", y_col: "y"})

st.set_page_config(page_title="Skeletal Tracking Demonstrator", layout="wide")
st.title("⚽ Skeletal Tracking — HCI Demonstrator")
st.caption("Phase 4 — visual inspection and quality feedback for the pose-estimation pipeline")

st.sidebar.header("Controls")

available_runs = [name for name, paths in RUNS.items()
                   if os.path.exists(paths["keypoints_csv"]) or os.path.exists(paths["video"])]

if not available_runs:
    st.warning(
        "No run files found yet. Edit the `RUNS` dictionary at the top of app.py "
        "to point at your actual video and keypoints CSV paths, then reload."
    )
    available_runs = list(RUNS.keys()) 

selected_run = st.sidebar.selectbox("Select run / configuration", available_runs)
run_paths = RUNS[selected_run]

df = load_keypoints(run_paths["keypoints_csv"])
run_format = run_paths.get("format", "keypoints")
id_col = "player_id" if run_format == "keypoints" else "track_id"

if df is not None and not df.empty:
    min_frame, max_frame = int(df["frame"].min()), int(df["frame"].max())
    frame_range = st.sidebar.slider(
        "Frame range to inspect",
        min_value=min_frame, max_value=max_frame,
        value=(min_frame, min(min_frame + 100, max_frame)),
    )
    ids = sorted(df[id_col].unique().tolist()) if id_col in df.columns else []
    selected_player = st.sidebar.selectbox("Player / Track ID", ["All"] + ids)
    selected_player = None if selected_player == "All" else selected_player

    if run_format == "keypoints":
        joint_name = st.sidebar.selectbox("Joint to plot over time", list(JOINTS_OF_INTEREST.keys()))
    else:
        joint_name = None
        st.sidebar.caption("This run logs hit/miss + jitter, not joint positions — "
                            "joint trajectory plot isn't available for this format.")
else:
    frame_range, selected_player, joint_name = None, None, None

col_video, col_stats = st.columns([1.2, 1])

with col_video:
    st.subheader("1. Video playback with skeleton overlay")
    if os.path.exists(run_paths["video"]):
        st.video(run_paths["video"])
    else:
        st.info(f"Video not found at `{run_paths['video']}`. "
                f"Update the path in the RUNS config at the top of app.py.")

with col_stats:
    st.subheader("4. Tracking quality feedback")
    stats = compute_quality_stats(df, run_format)

    st.metric("Frames with detections", stats["frames_with_detections"])
    st.metric("Avg players/tracks detected / frame",
               f"{stats['avg_players_per_frame']:.2f}" if not np.isnan(stats["avg_players_per_frame"]) else "—")

    if run_format == "framelog":
        st.metric("Pose hit rate",
                   f"{stats['hit_rate']:.1f}%" if not np.isnan(stats["hit_rate"]) else "—")
        if df is not None and not df.empty and stats["top_miss_causes"] is not None:
            st.caption("Top miss causes:")
            st.dataframe(stats["top_miss_causes"].rename("count"), use_container_width=True)
        if df is None or df.empty:
            st.warning("No frame log data loaded for this run — cannot assess quality.")
        elif not np.isnan(stats["hit_rate"]) and stats["hit_rate"] < 70:
            st.warning(f"Pose hit rate ({stats['hit_rate']:.1f}%) is below 70% — "
                       f"a large share of detected players had no usable pose.")
        else:
            st.success("Pose hit rate is within an acceptable range.")
    else:
        st.metric("Avg keypoint visibility",
                   f"{stats['avg_visibility']:.2f}" if not np.isnan(stats["avg_visibility"]) else "—")
        if df is None or df.empty:
            st.warning("No keypoints data loaded for this run — cannot assess quality.")
        elif stats["avg_visibility"] < VISIBILITY_WARN_THRESHOLD:
            st.warning(
                f"Average visibility ({stats['avg_visibility']:.2f}) is below the "
                f"{VISIBILITY_WARN_THRESHOLD} threshold — many keypoints may be occluded "
                f"or unreliable in this run."
            )
        else:
            st.success("Average keypoint visibility is within an acceptable range.")

st.divider()

st.subheader("2. Visualisation of tracked joints over time")

if df is None or df.empty:
    st.info("Load a keypoints CSV to see joint trajectories here.")
elif run_format == "framelog":
    st.caption("This run (frame_logs.csv) logs hit/miss + jitter per track, not joint x/y positions. "
               "Showing hit-rate over time instead.")
    windowed = df[(df["frame"] >= frame_range[0]) & (df["frame"] <= frame_range[1])]
    if selected_player is not None:
        windowed = windowed[windowed["track_id"] == selected_player]

    hit_by_frame = windowed.groupby("frame")["hit"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(hit_by_frame.index, hit_by_frame.values)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Hit rate (%)")
    ax.set_title(f"Pose hit rate over frames ({selected_run})")
    st.pyplot(fig)

    if "smoothed_jitter" in windowed.columns and windowed["smoothed_jitter"].notna().any():
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(windowed["frame"], windowed["raw_jitter"], label="raw jitter", alpha=0.6)
        ax2.plot(windowed["frame"], windowed["smoothed_jitter"], label="smoothed jitter", alpha=0.9)
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Jitter (px/frame)")
        ax2.set_title("Raw vs smoothed jitter")
        ax2.legend()
        st.pyplot(fig2)

    with st.expander("Show raw values for this range"):
        st.dataframe(windowed, use_container_width=True)
else:
    joint_idx = JOINTS_OF_INTEREST[joint_name]
    traj = get_joint_xy(df, joint_idx, selected_player)

    if traj is None or traj.empty:
        st.info("No trajectory data available for this joint/player combination.")
    else:
        traj_in_range = traj[(traj["frame"] >= frame_range[0]) & (traj["frame"] <= frame_range[1])]

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(traj_in_range["frame"], traj_in_range["x"], label="x")
        ax.plot(traj_in_range["frame"], traj_in_range["y"], label="y")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Normalized position")
        ax.set_title(f"{joint_name} position over frames ({selected_run})")
        ax.legend()
        st.pyplot(fig)

        with st.expander("Show raw values for this range"):
            st.dataframe(traj_in_range, use_container_width=True)

st.divider()

st.subheader("Bonus: run statistics / summary files")

def build_baseline_summary(kp_df):
    if kp_df is None or kp_df.empty:
        return None
    frames_processed = kp_df["frame"].nunique()
    total_rows = len(kp_df)
    avg_players = kp_df.groupby("frame")["player_id"].nunique().mean()

    vis_cols = [c for c in kp_df.columns if c.startswith("vis")]
    x_cols = [f"x{i}" for i in range(33) if f"x{i}" in kp_df.columns]
    y_cols = [f"y{i}" for i in range(33) if f"y{i}" in kp_df.columns]

    zero_mask = kp_df[vis_cols].sum(axis=1) == 0
    zero_rows = zero_mask.sum()
    miss_rate = (zero_rows / total_rows * 100) if total_rows else float("nan")
    hit_rate = 100 - miss_rate

    vis_data = kp_df[vis_cols].replace(0, np.nan)
    avg_visibility = vis_data.mean().mean()

    jitters = []
    hits_df = kp_df[~zero_mask].sort_values(["player_id", "frame"])
    for pid, g in hits_df.groupby("player_id"):
        pts = g[x_cols + y_cols].to_numpy()
        if len(pts) > 1:
            diffs = np.diff(pts, axis=0)
            xy_diffs = diffs.reshape(diffs.shape[0], -1, 2)
            per_frame_jit = np.linalg.norm(xy_diffs, axis=2).mean(axis=1)
            jitters.extend(per_frame_jit.tolist())
    avg_jitter = float(np.mean(jitters)) if jitters else float("nan")

    miss_rows = kp_df[zero_mask]
    causes = defaultdict(int)
    total_misses = len(miss_rows)
    if total_misses:
        box_h = miss_rows["box_y2"] - miss_rows["box_y1"]
        box_w = miss_rows["box_x2"] - miss_rows["box_x1"]
        causes["low_resolution"] = int((np.minimum(box_h, box_w) < 70).sum())
        players_per_frame = kp_df.groupby("frame")["player_id"].nunique()
        multi_frames = set(players_per_frame[players_per_frame > 1].index)
        causes["multiple_players"] = int(miss_rows["frame"].isin(multi_frames).sum())

    lines = []
    lines.append("=" * 60)
    lines.append("BASELINE SUMMARY (computed from keypoints.csv)")
    lines.append("=" * 60)
    lines.append(f"Frames processed:        {frames_processed}")
    lines.append(f"Total detection rows:    {total_rows}")
    lines.append(f"Avg players/frame:       {avg_players:.2f}")
    lines.append(f"Pose hit rate:           {hit_rate:.1f}%  (rows with at least one non-zero landmark)")
    lines.append(f"Pose miss rate:          {miss_rate:.1f}%  (all-zero rows, i.e. no landmarks found)")
    lines.append(f"Avg keypoint visibility: {avg_visibility:.3f}  (excludes zero/missing landmarks)")
    lines.append("")
    lines.append("-- Jitter (no smoothing applied in baseline) --")
    lines.append(f"Average raw jitter (px/frame): {avg_jitter:.3f}")
    lines.append("")
    lines.append("-- Miss-cause breakdown (a miss can have >1 cause) --")
    for cause, count in sorted(causes.items(), key=lambda kv: -kv[1]):
        pct = (count / total_misses * 100) if total_misses else 0.0
        lines.append(f"  {cause:<18} {count:>6}   ({pct:.1f}% of misses)")
    lines.append("=" * 60)
    return "\n".join(lines)


col_a, col_b = st.columns(2)
with col_a:
    st.caption("Baseline summary")
    baseline_df = load_keypoints("keypoints.csv")
    baseline_summary = build_baseline_summary(baseline_df)
    if baseline_summary:
        st.text(baseline_summary)
    else:
        st.caption("Not found at `keypoints.csv` in the working directory.")

    with st.expander("Baseline `statistics.csv` (per-joint detail)"):
        if os.path.exists("statistics.csv"):
            st.dataframe(pd.read_csv("statistics.csv"), use_container_width=True)
        else:
            st.caption("Not found at `statistics.csv` in the working directory.")

PHASE3_RUN_TO_SLUG = {
    "Phase 3 - clip1, full res, lite model": "clip1_full_res_lite",
    "Phase 3 - clip1, half res, lite model": "clip1_half_res_lite",
    "Phase 3 - clip1, full res, full model": "clip1_full_res_full",
    "Phase 3 - clip2, full res, lite model": "clip2_full_res_lite",
    "Phase 3 - clip2, half res, lite model": "clip2_half_res_lite",
    "Phase 3 - clip2, full res, full model": "clip2_full_res_full",
    "Phase 3 - clip3, full res, lite model": "clip3_full_res_lite",
    "Phase 3 - clip3, half res, lite model": "clip3_half_res_lite",
    "Phase 3 - clip3, full res, full model": "clip3_full_res_full",
}

with col_b:
    st.caption(f"Summary for: {selected_run}")
    slug = PHASE3_RUN_TO_SLUG.get(selected_run)
    summary_path = f"phase3_outputs/summary_{slug}.txt" if slug else None

    if summary_path and os.path.exists(summary_path):
        with open(summary_path) as f:
            st.text(f.read())
    elif summary_path:
        st.caption(f"Not found at `{summary_path}`.")
    else:
        st.caption("Select a Phase 3 run above to see its summary here.")

st.divider()
st.caption("All twelve logged configurations (comparison_results.csv)")
if os.path.exists("comparison_results.csv"):
    st.dataframe(pd.read_csv("comparison_results.csv"), use_container_width=True)
else:
    st.caption("Not found at `comparison_results.csv` in the working directory.")