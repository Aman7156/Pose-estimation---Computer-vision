# Real-Time Athlete Pose Estimation

A real-time computer vision system that tracks an athlete's body landmarks on live video at **30 FPS** and streams the results to a coaching dashboard.

Built in Python and PyTorch at the University of Messina (Nov 2025 – Feb 2026), and deployed with Docker and Kubernetes.

<!-- Add a short GIF or screenshot here: it is the first thing a recruiter looks at. -->
<!-- ![Demo](docs/demo.gif) -->

---

## What it does

- Reads live video from a webcam or a video file
- Detects the athlete and estimates their body landmarks (joints such as shoulders, elbows, hips, knees) frame by frame
- Runs at around 30 frames per second, fast enough for live coaching
- Streams the landmark data to a dashboard, so a coach can follow the athlete's movement as it happens

## Why it matters

Coaches usually judge technique by eye or by rewatching recordings. This system gives them body-position data live, while the athlete is training, so they can correct movement on the spot.

---

## How it works

```
Camera / video  ->  Frame capture  ->  Pose model  ->  Landmarks  ->  Dashboard
   (OpenCV)                          (PyTorch)        (x, y, confidence)   (live stream)
```

1. **Capture:** OpenCV reads frames from the camera or a video file.
2. **Pose estimation:** each frame goes through the pose model, which returns the position of each body landmark.  
   <!-- EDIT: name the model you used, e.g. "MediaPipe Pose", "YOLOv8-Pose" or your own PyTorch model. -->
3. **Streaming:** the landmarks are sent to the coaching dashboard in real time.  
   <!-- EDIT: say how, e.g. "over WebSockets" or "through a FastAPI endpoint". -->
4. **Deployment:** the whole pipeline runs in Docker containers, managed with Kubernetes.

---

## Tech stack

| Area | Tools |
|---|---|
| Language | Python |
| Deep learning | PyTorch |
| Computer vision | OpenCV <!-- EDIT: add MediaPipe / YOLO if you used them --> |
| Deployment | Docker, Kubernetes |

---

## Getting started

### Requirements

- Python 3.10 or newer <!-- EDIT: change to the version you used -->
- A webcam or a sample video file
- Docker (optional, for the container setup)

### Run locally

```bash
git clone https://github.com/Aman7156/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
python main.py --source 0          # 0 = webcam
python main.py --source video.mp4  # or a video file
```

<!-- EDIT: replace main.py and the flags with your real file names and options. -->

### Run with Docker

```bash
docker build -t pose-estimation .
docker run --rm -p 8000:8000 pose-estimation
```

<!-- EDIT: match the port and image name to your Dockerfile. -->

---

## Project structure

```
.
├── src/            # capture, pose model and streaming code
├── dashboard/      # coaching dashboard
├── k8s/            # Kubernetes manifests
├── Dockerfile
├── requirements.txt
└── README.md
```

<!-- EDIT: update this tree to match your real folders. -->

---

## Results

- Runs in real time at about **30 FPS** on live video
- Tracks athlete body landmarks frame by frame and streams them to the dashboard

<!-- EDIT (optional): add the hardware you tested on (e.g. "laptop CPU" or "NVIDIA GPU"),
     and any accuracy numbers you actually measured. Only add numbers you can explain in an interview. -->

---

## Documentation

The full technical documentation (40+ pages) was written for an academic journal submission at the University of Messina.

<!-- EDIT: link it here only if you are allowed to share it publicly. -->

---

## What I learned

- Keeping a deep learning model fast enough for live video
- Turning model output into data that non-technical users (coaches) can follow
- Packaging and running an ML system with Docker and Kubernetes

---

## Author

**Aman Kumar Singh**  
M.Sc. Data Science and Engineering, Politecnico di Torino  
[LinkedIn](https://linkedin.com/in/aman-kumar-singh-4a632a1a0) · [GitHub](https://github.com/Aman7156)
