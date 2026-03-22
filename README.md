#  Multi-Person Detection in Verification Videos

## Problem
Detect whether a verification session video contains more than one person,
which is a signal for potential coercion or incentivised fraud.

## Approach

### Core idea
We use a **pretrained YOLOv8 person detector** and aggregate its per-frame detections
to a video-level binary label. This is a deliberate design choice: person detection
is a well-solved problem (YOLO trained on COCO), and our dataset is too small to
fine-tune reliably.

### Frame sampling
Two strategies are implemented:
- **Uniform**: sample a frame every N seconds (reproducible, simple baseline)
- **Scene-change**: sample frames where significant visual change is detected
  (more likely to capture a second person briefly entering the scene)

### Aggregation
Frame-level detections are aggregated to a video-level prediction:
- **any** (default): label=1 if ANY frame has >1 person detected
- **threshold**: label=1 if >X% of frames have >1 person (more robust to noise)

## Project structure

```
veriff_assignment/
├── data_utils.py      # Video loading and frame sampling
├── detector.py        # Person detection + frame aggregation
├── evaluate.py        # Metrics and report generation
├── predict.py         # CLI entry point
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

YOLOv8 weights (`yolov8n.pt`) download automatically on first run.

## Usage

### Evaluate on labeled dataset
```bash
python predict.py --mode evaluate \
    --video_dir ./videos \
    --label_file labels.txt \
    --sampling uniform \
    --aggregation threshold \
    --output predictions.txt
```

### Predict on a new unlabeled video
```bash
python predict.py --mode predict --video_path ./new_video.mp4 --aggregation threshold 
```

### Compare sampling strategies
```bash
python predict.py --mode compare \
    --video_dir ./videos \
    --label_file labels.txt --aggregation threshold 
```

## Evaluation notes

We report precision, recall, and F1 rather than accuracy, because:
1. Classes may be imbalanced
2. False negatives (missed coercion) carry higher cost than false positives

