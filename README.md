
  # 🚦 Traffic Analytics Pipeline

  > *Real-time intersection movement analytics — detects, tracks, and counts vehicles across all 4
  directions from raw traffic footage. YOLOv8 + ByteTrack + per-line CSV export, in one command.*

  ![Python](https://img.shields.io/badge/Python-3.12-3776AB)
  ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C)
  ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111F68)
  ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8)
  ![Gradio](https://img.shields.io/badge/Gradio-coming_soon-FF7C00)
  
  ---

  ## 🎥 Demo


<img width="600" height="338" alt="demo" src="https://github.com/user-attachments/assets/31a8042e-874f-4888-a0f2-0c5e811f71b7" />



  > _Annotated output video attached above — vehicles with persistent IDs tracked across frames._

  ---
  
  ## ✨ Features

  ### 🎯 Vehicle Detection
  - YOLOv8n (nano) pretrained on COCO — fast, lightweight
  - Filtered to relevant classes: `car`, `truck`, `bus`, `motorcycle`
  - Confidence threshold tuned to suppress false positives in dense scenes
  
  ### 🆔 Multi-Object Tracking
  - **ByteTrack** integration via `model.track(persist=True)`
  - Persistent ID per vehicle across the entire clip — same car keeps the same number
  - Handles short occlusions: IDs re-associated when an object briefly disappears
  - Verified: a single vehicle tracked across **200+ consecutive frames** in the test clip

  ### 🎥 Annotated Video Output 
  - Every frame rendered with bounding boxes, class labels, and tracking IDs
  - Output saved to `data/traffic_tracked.mp4` at original resolution + fps
  - Powered by `cv2.VideoWriter` + Ultralytics' `result.plot()` helper
  
  ### 🧮 Counting & Analytics =
  - Virtual line-crossing logic — count each track ID once when it crosses the line
  - Per-direction, per-class breakdown
  - CSV export of every counted event

  ### 🌐 Live Demo _(in progress)_
  - Gradio app: upload any traffic video → get back annotated output + CSV
  - Deployed on **HuggingFace Spaces** for a public live URL

  ### 📊 Mini Dashboard _(in progress)_
  - Time-series of vehicle counts per minute
  - Per-class bar chart
  - Total counters across the whole run

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
  │   │   └── traffic_tracked.mp4      # Annotated output (gitignored)
  │   ├── src/
  │   │   └── detector.py              # Detection + tracking + video writer
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

  The annotated video saves to `traffic-sdc/data/traffic_tracked.mp4`.
  
  ---

  ## 🗺️  Roadmap

  - [x] YOLOv8 vehicle detection                 
  - [x] ByteTrack persistent-ID tracking
  - [x] Annotated video output with `cv2.VideoWriter`
  - [x] Line-crossing counter + CSV export
  - [ ] Per-class breakdown
  - [ ] Accuracy metric vs manual ground-truth count
  - [ ] Gradio app on HuggingFace Spaces (live URL)
  - [ ] Mini analytics dashboard
  - [ ] FastAPI + Docker deployment

  ---

  ## 📧 Contact
  
  **Jaya Arora**
  - 📧 jayaarora2402@gmail.com  
  - 💼 [LinkedIn](https://www.linkedin.com/in/jaya-arora-6892a93a0/)
  - 🐙 [GitHub](https://github.com/Jaya242)


