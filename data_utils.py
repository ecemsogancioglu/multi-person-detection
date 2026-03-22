"""
data_utils.py
-------------
Video loading, frame sampling strategies, and dataset preparation
for the Veriff multi-person detection task.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class VideoSample:
    video_id: str
    video_path: str
    label: int  # 0 = single person, 1 = multiple people
    frames: np.ndarray = None  # shape: (N, H, W, C)


def load_labels(label_file: str) -> Dict[str, int]:
    """
    Parse the label text file into a dict: {video_id: label}.
    Handles varying whitespace between video_id and label.
    """
    labels = {}
    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("video") or line.lower().startswith("label"):
                continue  # skip header if present
            parts = line.split()
            if len(parts) >= 2:
                video_id = parts[0]
                label = int(parts[1])
                labels[video_id] = label
    return labels


def sample_frames_uniform(
    video_path: str,
    interval_seconds: float = 1.0,
    max_frames: int = 30
) -> np.ndarray:
    """
    Sample frames at a fixed time interval.
    
    Args:
        video_path: Path to the .mp4 file
        interval_seconds: Time between sampled frames in seconds
        max_frames: Maximum number of frames to return (cap for long videos)
    
    Returns:
        np.ndarray of shape (N, H, W, C) in BGR format
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25.0  # fallback if FPS undetectable

    frame_interval = max(1, int(fps * interval_seconds))
    frames = []
    frame_idx = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return np.array(frames) if frames else np.array([])


def sample_frames_scene_change(
    video_path: str,
    diff_threshold: float = 5.0,
    min_interval_frames: int = 10,
    max_frames: int = 30
) -> np.ndarray:
    """
    Sample frames when significant visual change is detected.
    
    More likely to capture the moment a second person enters the scene
    compared to uniform sampling, which may miss brief appearances.
    
    Args:
        video_path: Path to the .mp4 file
        diff_threshold: Mean absolute pixel difference to trigger a new sample
        min_interval_frames: Minimum frames between samples (avoid burst sampling)
        max_frames: Maximum number of frames to return
    
    Returns:
        np.ndarray of shape (N, H, W, C) in BGR format
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    prev_gray = None
    frames_since_last_sample = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            # Always take the first frame
            frames.append(frame)
        else:
            diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
            if diff > diff_threshold and frames_since_last_sample >= min_interval_frames:
                frames.append(frame)
                frames_since_last_sample = 0

        prev_gray = gray
        frames_since_last_sample += 1

    cap.release()

    # If scene-change found too few frames, pad with uniform sampling
    if len(frames) < 3:
        return sample_frames_uniform(video_path, interval_seconds=1.0, max_frames=max_frames)

    return np.array(frames)


def load_dataset(
    video_dir: str,
    label_file: str,
    sampling_strategy: str = "uniform",
    interval_seconds: float = 1.0,
    max_frames: int = 30
) -> List[VideoSample]:
    """
    Load all labeled videos and sample frames from each.

    Args:
        video_dir: Directory containing .mp4 files
        label_file: Path to the labels .txt file
        sampling_strategy: "uniform" or "scene_change"
        interval_seconds: Used only for uniform sampling
        max_frames: Max frames per video

    Returns:
        List of VideoSample objects with frames loaded
    """
    labels = load_labels(label_file)
    video_dir = Path(video_dir)
    samples = []

    for video_id, label in labels.items():
        video_path = video_dir / f"{video_id}.mp4"
        if not video_path.exists():
            print(f"Warning: video not found for {video_id}, skipping.")
            continue

        if sampling_strategy == "uniform":
            frames = sample_frames_uniform(
                str(video_path),
                interval_seconds=interval_seconds,
                max_frames=max_frames
            )
        elif sampling_strategy == "scene_change":
            frames = sample_frames_scene_change(
                str(video_path),
                max_frames=max_frames
            )
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

        sample = VideoSample(
            video_id=video_id,
            video_path=str(video_path),
            label=label,
            frames=frames
        )
        samples.append(sample)
        print(f"Loaded {video_id}: label={label}, frames sampled={len(frames)}")

    print(f"\nDataset loaded: {len(samples)} videos")
    print(f"Label distribution: {sum(s.label for s in samples)} positive, "
          f"{sum(1 - s.label for s in samples)} negative")

    return samples
