#!/usr/bin/env python3
"""
Compares a freshly-retrained model's metrics against the currently
committed baseline and fails (non-zero exit) if test accuracy dropped by
more than --tolerance. This is the gate the retrain workflow runs before
it will open a PR - a retrain that makes the model worse never reaches
main, automatically or otherwise.

Usage:
    python scripts/check_model_regression.py \
        --baseline baseline_metrics.json \
        --candidate predictor/ml_model/model_metrics.json \
        --tolerance 0.03
"""
import argparse
import json
import sys


def best_test_accuracy(metrics: dict) -> float:
    best = metrics["best_model"]
    return metrics["results"][best]["test_accuracy"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.03,
        help="Allowed accuracy drop before this is treated as a regression (default 0.03 = 3 percentage points).",
    )
    args = parser.parse_args()

    with open(args.baseline) as f:
        baseline = json.load(f)
    with open(args.candidate) as f:
        candidate = json.load(f)

    baseline_acc = best_test_accuracy(baseline)
    candidate_acc = best_test_accuracy(candidate)
    delta = candidate_acc - baseline_acc

    print(f"Baseline  ({baseline['best_model']}): test_accuracy = {baseline_acc:.4f}")
    print(f"Candidate ({candidate['best_model']}): test_accuracy = {candidate_acc:.4f}")
    print(f"Delta: {delta:+.4f}  (tolerance: -{args.tolerance:.4f})")

    if delta < -args.tolerance:
        print(
            f"REGRESSION: candidate accuracy dropped by {-delta:.4f}, "
            f"more than the allowed {args.tolerance:.4f}. Failing."
        )
        return 1

    print("OK: no unacceptable regression.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
