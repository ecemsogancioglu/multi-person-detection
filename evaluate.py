"""
evaluate.py
-----------
Evaluation metrics for the multi-person detection task.

Why not just accuracy?
- Dataset is small and likely imbalanced (more single-person videos)
- In production, false negatives (missed coercion) carry higher cost than
  false positives (flagging a legitimate session for review)
- We report precision, recall, F1 and discuss the tradeoff explicitly
"""

from typing import List
from detector import VideoResult


def compute_metrics(results: List[VideoResult]) -> dict:
    """
    Compute classification metrics from video-level predictions.
    
    Returns dict with accuracy, precision, recall, F1, plus
    confusion matrix components.
    """
    true_labels = [r.true_label for r in results]
    pred_labels = [r.predicted_label for r in results]

    tp = sum(t == 1 and p == 1 for t, p in zip(true_labels, pred_labels))
    tn = sum(t == 0 and p == 0 for t, p in zip(true_labels, pred_labels))
    fp = sum(t == 0 and p == 1 for t, p in zip(true_labels, pred_labels))
    fn = sum(t == 1 and p == 0 for t, p in zip(true_labels, pred_labels))

    total = len(results)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_samples": total
    }


def print_report(results: List[VideoResult]):
    """Print a human-readable evaluation report."""
    metrics = compute_metrics(results)

    print("\n" + "=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)
    print(f"Total videos evaluated: {metrics['n_samples']}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positive  (correctly flagged multi-person): {metrics['tp']}")
    print(f"  True Negative  (correctly passed single-person): {metrics['tn']}")
    print(f"  False Positive (single-person flagged as multi): {metrics['fp']}")
    print(f"  False Negative (multi-person missed):            {metrics['fn']}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}  ← most important in production")
    print(f"  F1 Score:  {metrics['f1']:.3f}")
    print("\nNote: With only 19 samples, all metrics have high variance.")
    print("These numbers are directional, not conclusive.")

    print("\nPer-video breakdown:")
    print(f"{'Video':<12} {'True':>6} {'Pred':>6} {'Multi-frames':>14} {'Max persons':>12}")
    print("-" * 55)
    for r in results:
        status = "✓" if r.true_label == r.predicted_label else "✗"
        print(f"{r.video_id:<12} {r.true_label:>6} {r.predicted_label:>6} "
              f"{r.predicted_prob:>13.1%} {r.max_persons_detected:>12} {status}")
    print("=" * 50)


def save_predictions(results: List[VideoResult], output_path: str):
    """Save predictions to a text file for submission."""
    with open(output_path, "w") as f:
        f.write("video_id\ttrue_label\tpredicted_label\tmulti_person_frame_fraction\n")
        for r in results:
            f.write(f"{r.video_id}\t{r.true_label}\t"
                    f"{r.predicted_label}\t{r.predicted_prob:.4f}\n")
    print(f"\nPredictions saved to {output_path}")
