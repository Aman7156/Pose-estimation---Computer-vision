OUTDIR="comparison_outputs"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg not found."
    echo "Install it first:  brew install ffmpeg"
    exit 1
fi

shopt -s nullglob 
files=("$OUTDIR"/annotated_*.mp4)

if [ ${#files[@]} -eq 0 ]; then
    echo "No files matching $OUTDIR/annotated_*.mp4 found. Nothing to do."
    exit 0
fi

echo "Found ${#files[@]} video(s) to check/re-encode:"
for f in "${files[@]}"; do
    echo "  - $f"
done
echo ""

for f in "${files[@]}"; do
    codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name \
            -of csv=p=0 "$f" 2>/dev/null || echo "unknown")

    if [ "$codec" == "h264" ]; then
        echo "OK (already H.264): $f"
        continue
    fi

    echo "Re-encoding ($codec -> h264): $f"
    tmp_file="${f%.mp4}_h264_tmp.mp4"
    ffmpeg -y -loglevel error -i "$f" -vcodec libx264 -pix_fmt yuv420p "$tmp_file"
    mv "$tmp_file" "$f"
    echo "  done."
done

echo ""
echo "All videos in $OUTDIR/ are now H.264. Restart Streamlit to pick up the changes:"
echo "  (Ctrl+C the running app, then)  streamlit run app.py"