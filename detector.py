"""
detector.py
-----------
Person detection on video frames using a pretrained YOLOv8 model.
Aggregates frame-level detections into a video-level binary prediction.

Design rationale:
- We use a pretrained detector rather than training from scratch because:
  (a) The dataset is tiny (19 videos)
  (b) Person detection is a well-solved problem on COCO
  (c) Fine-tuning on actual Veriff session data is the right next step at scale
- Aggregation strategy: threshold 0.2.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
from data_utils import VideoSample


@dataclass
class FrameResult:
    frame_idx: int
    person_count: int
    confidence_scores: List[float]


@dataclass
class VideoResult:
    video_id: str
    true_label: int
    predicted_label: int
    predicted_prob: float  # fraction of frames with >1 person
    frame_results: List[FrameResult] = field(default_factory=list)
    max_persons_detected: int = 0


class MultiPersonDetector:
    """
    Wraps a pretrained YOLOv8 model for multi-person detection in videos.
    
    Aggregation options:
    - "any": label=1 if ANY frame has >1 person (high recall, lower precision)
    - "majority": label=1 if MAJORITY of frames have >1 person (more robust to noise)
    - "threshold": label=1 if fraction of frames > threshold
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        aggregation: str = "threshold",
        majority_threshold: float = 0.2  # used only when aggregation="threshold"
    ):
        self.confidence_threshold = confidence_threshold
        self.aggregation = aggregation
        self.majority_threshold = majority_threshold
        self.model = self._load_model()

    def _load_model(self):
        """Load pretrained YOLOv8 nano — fast and sufficient for person detection."""
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")  # auto-downloads on first run
            print("YOLOv8n loaded successfully.")
            return model
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

    def detect_persons_in_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Run YOLO on a single frame and count detected persons.
        
        YOLO class 0 = person (COCO dataset convention).
        """
        results = self.model(frame, verbose=False)
        
        person_confidences = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls == 0 and conf >= self.confidence_threshold:  # class 0 = person
                        person_confidences.append(conf)

        return FrameResult(
            frame_idx=0,  # updated by caller
            person_count=len(person_confidences),
            confidence_scores=person_confidences
        )

    def predict_video(self, sample: VideoSample) -> VideoResult:
        """
        Predict video-level label by aggregating frame-level detections.
        
        Aggregation strategies explained:
        - "any": flag if a second person appears even briefly (catches coercion)
        - "threshold": require second person in >X% of frames (reduces false positives
          from passersby or reflections)
        """
        if sample.frames is None or len(sample.frames) == 0:
            print(f"Warning: no frames for {sample.video_id}")
            return VideoResult(
                video_id=sample.video_id,
                true_label=sample.label,
                predicted_label=0,
                predicted_prob=0.0
            )

        frame_results = []
        multi_person_frame_count = 0

        for idx, frame in enumerate(sample.frames):
            result = self.detect_persons_in_frame(frame)
            result.frame_idx = idx
            frame_results.append(result)
            if result.person_count > 1:
                multi_person_frame_count += 1

        fraction_multi = multi_person_frame_count / len(frame_results)
        max_persons = max(r.person_count for r in frame_results)

        # Aggregation
        if self.aggregation == "any":
            predicted_label = 1 if multi_person_frame_count > 0 else 0
        elif self.aggregation == "majority":
            predicted_label = 1 if fraction_multi >= 0.5 else 0
        elif self.aggregation == "threshold":
            predicted_label = 1 if fraction_multi >= self.majority_threshold else 0
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return VideoResult(
            video_id=sample.video_id,
            true_label=sample.label,
            predicted_label=predicted_label,
            predicted_prob=fraction_multi,
            frame_results=frame_results,
            max_persons_detected=max_persons
        )

    def predict_dataset(self, samples: List[VideoSample]) -> List[VideoResult]:
        """Run prediction on all videos in the dataset."""
        results = []
        for sample in samples:
            print(f"Processing {sample.video_id}...")
            result = self.predict_video(sample)
            results.append(result)
            print(f"  → predicted={result.predicted_label}, "
                  f"true={result.true_label}, "
                  f"multi-person frames={result.predicted_prob:.1%}, "
                  f"max persons={result.max_persons_detected}")
        return results
