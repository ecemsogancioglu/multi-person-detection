"""
predict.py
----------
CLI entry point for the Veriff multi-person detection system.

Usage examples:

  # Evaluate on labeled dataset:
  python predict.py --mode evaluate \
      --video_dir ./videos \
      --label_file labels.txt \
      --sampling uniform \
      --aggregation threshold

  # Predict on a single new video:
  python predict.py --mode predict \
      --video_path ./new_video.mp4 --aggregation threshold

  # Compare sampling strategies:
  python predict.py --mode compare \
      --video_dir ./videos \
      --label_file labels.txt --aggregation threshold
"""

import argparse
from data_utils import load_dataset, sample_frames_uniform, sample_frames_scene_change, VideoSample
from detector import MultiPersonDetector
from evaluate import print_report, save_predictions


def evaluate_mode(args):
    """Load dataset, run detector, print full evaluation report."""
    print(f"Loading dataset from {args.video_dir}...")
    samples = load_dataset(
        video_dir=args.video_dir,
        label_file=args.label_file,
        sampling_strategy=args.sampling,
        interval_seconds=args.interval,
        max_frames=args.max_frames
    )

    detector = MultiPersonDetector(
        confidence_threshold=args.confidence,
        aggregation=args.aggregation
    )

    results = detector.predict_dataset(samples)
    print_report(results)

    if args.output:
        save_predictions(results, args.output)


def predict_mode(args):
    """Run inference on a single unlabeled video."""
    print(f"Running inference on {args.video_path}...")

    frames = sample_frames_uniform(args.video_path, interval_seconds=1.0, max_frames=30)
    sample = VideoSample(
        video_id="input",
        video_path=args.video_path,
        label=-1,  # unknown
        frames=frames
    )

    detector = MultiPersonDetector(confidence_threshold=args.confidence)
    result = detector.predict_video(sample)

    print(f"\nResult: {'MULTIPLE PEOPLE DETECTED' if result.predicted_label == 1 else 'SINGLE PERSON'}")
    print(f"Frames with multiple people: {result.predicted_prob:.1%}")
    print(f"Max persons detected in a frame: {result.max_persons_detected}")


def compare_mode(args):
    """Compare uniform vs scene-change sampling on the same dataset."""
    print("=== Comparing sampling strategies ===\n")

    for strategy in ["uniform", "scene_change"]:
        print(f"\n--- Strategy: {strategy} ---")
        samples = load_dataset(
            video_dir=args.video_dir,
            label_file=args.label_file,
            sampling_strategy=strategy,
            max_frames=args.max_frames
        )
        detector = MultiPersonDetector(aggregation="threshold", majority_threshold=0.2)
        results = detector.predict_dataset(samples)
        print_report(results)


def main():
    parser = argparse.ArgumentParser(
        description="Veriff multi-person detection in verification videos"
    )
    parser.add_argument("--mode", choices=["evaluate", "predict", "compare"],
                        default="evaluate", help="Run mode")

    # Dataset args
    parser.add_argument("--video_dir", type=str, default="./videos",
                        help="Directory containing .mp4 files")
    parser.add_argument("--label_file", type=str, default="labels.txt",
                        help="Path to labels text file")
    parser.add_argument("--video_path", type=str,
                        help="Path to single video (predict mode only)")

    # Sampling args
    parser.add_argument("--sampling", choices=["uniform", "scene_change"],
                        default="uniform", help="Frame sampling strategy")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between frames (uniform sampling)")
    parser.add_argument("--max_frames", type=int, default=30,
                        help="Max frames to sample per video")

    # Detection args
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="YOLO confidence threshold for person detection")
    parser.add_argument("--aggregation", choices=["any", "majority", "threshold"],
                        default="threshold", help="How to aggregate frame predictions to video level")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save predictions CSV")

    args = parser.parse_args()

    if args.mode == "evaluate":
        evaluate_mode(args)
    elif args.mode == "predict":
        if not args.video_path:
            parser.error("--video_path required for predict mode")
        predict_mode(args)
    elif args.mode == "compare":
        compare_mode(args)


if __name__ == "__main__":
    main()
