# 🚦 Traffic Analytics Pipeline

  > *Real-time vehicle detection, multi-object tracking, and unique-vehicle counting from
     traffic footage. YOLOv8 + ByteTrack + dual-line crossing dedup + CSV export, in one 
    command.*

    [![🤗 Live Demo](https://img.shields.io/badge/🤗_Live_Demo-Try_it_now-yellow?style=for-th
  e-badge)](https://huggingface.co/spaces/Ctrlescflyy/traffic-analytics)

    ![Python](https://img.shields.io/badge/Python-3.12-3776AB)
  ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C)
  ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111F68)
  ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8)
  ![Gradio](https://img.shields.io/badge/Gradio-coming_soon-FF7C00)

  ---

  ## 🎥 Demo

  <img width="600" height="338" alt="demo" 
  src="https://github.com/user-attachments/assets/31a8042e-874f-4888-a0f2-0c5e811f71b7" 
  />
  
  > _Live counter overlay, persistent vehicle IDs, two virtual counting lines visible 
  (yellow). Every box is tracked frame-to-frame; the counter ticks up only when a new 
  unique vehicle is detected._
  
  ---

  ## ✨ Features

  ### 🎯 Vehicle Detection
  - YOLOv8n (nano) pretrained on COCO — fast, lightweight
  - Filtered to relevant vehicle classes: `car`, `truck`, `bus`, `motorcycle`
  (pedestrians/bicycles excluded for clean vehicle counts)
  - Confidence threshold tuned to suppress false positives in dense scenes

  ### 🆔 Multi-Object Tracking
  - **ByteTrack** integration via `model.track(persist=True)`
  - Persistent ID per vehicle across the entire clip — same car keeps the same number
  - Handles short occlusions: IDs re-associated when an object briefly disappears
  - Verified: a single vehicle tracked across **200+ consecutive frames** in the test
  clip

  ### 🎥 Annotated Video Output
  - Every frame rendered with bounding boxes, class labels, and tracking IDs
  - Live counter overlay shows running unique-vehicle count
  - Output saved to `data/traffic_tracked.mp4` at original resolution + fps
  - Powered by `cv2.VideoWriter` + Ultralytics' `result.plot()` helper

  ### 🧮 Counting & Dedup
  - **Two perpendicular virtual lines** placed across the intersection — one horizontal,
  one vertical
  - Each vehicle counted **once per line crossed** (dedup via per-line ID sets)
  - **Unique-vehicle count** as the headline metric (union of both lines): captures every
   distinct vehicle that passed through, regardless of trajectory
  - **Turning vehicles** identified separately (crossed both lines = passed through the
  intersection in two segments)
  - CSV export of every crossing event for downstream analysis: `track_id, class, line,
  direction, frame, timestamp_sec`

  ### 🌐 Live Demo ✅
    - **[Try the live demo →](https://huggingface.co/spaces/Ctrlescflyy/traffic-analytics)**
  on HuggingFace Spaces
    - Upload any traffic video → get back annotated output + crossings CSV
    - Built with Gradio, deployed on free-tier HF Spaces (CPU)

  ### 📊 Mini Dashboard _(in progress)_
  - Time-series of vehicle counts per minute
  - Per-class bar chart
  - Total counters across the whole run 

  ---

  ## 📊 Metrics

  Validated against frame-by-frame manual ground truth on the test clip (2208 frames, ~44
   sec, single-camera diagonal SE intersection view).

  | Metric | Auto | Manual | Result |
  |---|---|---|---|
  | **Unique vehicles detected** | 25 | 26 | **96.2% accuracy** |
  | Turning vehicles (crossed both lines) | 5 | — | auxiliary signal |
  | **Recall on real movements** | — | — | **100%** — every real vehicle captured |

  ### Camera context
  
  This clip was filmed from a **corner-angle camera**, so all real traffic flows
  diagonally (southeast) across the frame. The script's per-line directional counts are
  computed correctly but — given the camera angle — they don't map to true cardinal flow
  directions for this video. **"Unique vehicles" is the most honest headline metric for 
  this camera setup.**

  A top-down or perpendicular-mounted camera would produce meaningful per-direction
  breakdowns; for this clip, the unique-vehicle count is the metric to trust.

  ### Known limitations

  - **Phantom north counts**: ByteTrack ID jitter on stationary bounding boxes near the
  counting line occasionally triggers false crossings. A minimum-motion threshold (e.g.,
  require ≥10px of movement between frames) would eliminate these in v2.
  - **Class mis-labeling**: YOLOv8n (nano, COCO-pretrained) occasionally confuses
  trucks/buses and motorcycles from oblique camera angles. Vehicle *detection* accuracy
  is high; vehicle-*class* accuracy is moderate. A fine-tuned model on
  intersection-specific data would resolve this — out of scope for this sprint.
  
  ---

  ## 🛠️  Tech Stack

  | Layer | Technology |
  |---|---|
  | **Language** | Python 3.12 |
  | **Detection** | Ultralytics YOLOv8n (PyTorch backend) |
  | **Tracking** | ByteTrack |
  | **Video I/O** | OpenCV |
  | **Demo UI** _(coming)_ | Gradio |
  | **Deployment** _(coming)_ | HuggingFace Spaces |

  ---
  
  ## 📁 Project Structure

  ```
  traffic_detector/
  ├── traffic-sdc/
  │   ├── data/
  │   │   ├── traffic.mp4              # Input video (gitignored)
  │   │   ├── traffic_tracked.mp4      # Annotated output (gitignored)
  │   │   └── crossings.csv            # Per-event crossing log (gitignored)
  │   ├── src/
  │   │   └── detector.py              # Detection + tracking + counting + CSV writer
  │   ├── requirements.txt
  │   └── venv/                        # Local virtualenv (gitignored)
  ├── yolov8n.pt                       # Pretrained weights (gitignored)
  ├── .gitignore
  └── README.md
  ```
  
  ---

  ## 🚀 Getting Started

  ### Prerequisites
  - Python 3.10+
  - A traffic video file at `traffic-sdc/data/traffic.mp4`

  ### Installation
  
  ```bash
  git clone https://github.com/Jaya242/traffic_detector.git
  cd traffic_detector/traffic-sdc
  python3 -m venv venv
  source venv/bin/activate
  pip install ultralytics opencv-python gradio
  ```

  ### Run
  
  ```bash
  python src/detector.py
  ```

  Outputs saved to `traffic-sdc/data/`:
  - `traffic_tracked.mp4` — annotated video with counting lines + live counter overlay
  - `crossings.csv` — per-vehicle crossing event log

  Terminal prints the final unique-vehicle count and accuracy vs the manually-tuned
  ground truth.
  
  ---

  ## 🗺️  Roadmap

  - [x] YOLOv8 vehicle detection
  - [x] ByteTrack persistent-ID tracking
  - [x] Annotated video output with `cv2.VideoWriter`
  - [x] Dual-line crossing counter + CSV export
  - [x] Unique-vehicle dedup + turning-vehicle detection
  - [x] Accuracy metric vs manual ground-truth count (96.2%)
  - [x] Gradio app on HuggingFace Spaces (live URL)
  - [ ] Mini analytics dashboard
  - [ ] Minimum-motion threshold to eliminate phantom counts
  - [ ] Class fine-tuning on intersection-specific data
  - [ ] FastAPI + Docker deployment

  ---
  
  ## 📧 Contact

  **Jaya Arora**
  - 📧 jayaarora2402@gmail.com
  - 💼 [LinkedIn](https://www.linkedin.com/in/jaya-arora-6892a93a0/)
  - 🐙 [GitHub](https://github.com/Jaya242)

