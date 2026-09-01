"""
evaluate.py
===========
Benchmark harness for SAGE-HAR.

What this script does, step by step:
  1. Generates a synthetic tri-axial accelerometer stream for three subject
     profiles: Calm, Tremor/Parkinsonian, and Restless. Each profile spends
     most windows "at rest" with occasional bursts of "activity", but the
     three differ in how noisy rest looks and how often activity happens --
     this is what makes Tremor a genuinely hard case for the gate (high
     resting noise looks a lot like real motion).
  2. Runs every window through SageGate and tallies how many windows the
     gate would have skipped.
  3. Compares two "classifiers" running on top of the gate:
       - ALWAYS-ON: a HAR model runs on every single window.
       - GATED:     a HAR model runs only when the gate is ACTIVE; while
                     the gate is INACTIVE it defaults to predicting REST,
                     since that is exactly what an INACTIVE gate is
                     asserting -- it is never right to assume "still
                     active" just because that's what the last real
                     inference happened to say.
     Both classifiers are modeled as a simple noisy oracle (97% per-window
     accuracy) so the comparison isolates the accuracy *cost of skipping*
     rather than the accuracy of any particular model architecture. In
     practice GATED comes out essentially tied with (often fractionally
     ahead of) ALWAYS-ON: skipping removes hundreds of trivially-correct
     rest windows from the classifier's exposure to its own 3% error rate,
     and the only real cost is a handful of missed windows at the very
     start of each activity burst (the gate's detection latency).
  4. Prints a summary table and writes per-subject rows to results.csv.

Only the standard library is used (random, csv, statistics) plus our own
sage_gate module -- no PyTorch required to run this benchmark.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from typing import List, Tuple

from sage_gate import GateConfig, SageGate

Sample = Tuple[float, float, float]

WINDOW_LEN = 40          # samples per window (~0.8s at 50Hz)
WINDOWS_PER_SUBJECT = 600
MODEL_ACCURACY = 0.97    # per-window accuracy of the stand-in HAR classifier


# ---------------------------------------------------------------------------
# Step 1: Synthetic multi-subject data generator
# ---------------------------------------------------------------------------
@dataclass
class Profile:
    """Describes how a synthetic subject behaves."""

    name: str
    rest_noise_std: float          # how jittery the sensor is while resting
    active_amplitude: float        # how big the signal swings during motion
    activation_chance: float       # per-window chance of starting a motion bout
    rest_bout_range: Tuple[int, int]
    active_bout_range: Tuple[int, int]


PROFILES = {
    "calm": Profile("Calm", rest_noise_std=0.015, active_amplitude=2.2,
                     activation_chance=0.02, rest_bout_range=(40, 120), active_bout_range=(15, 45)),
    "tremor": Profile("Tremor/Parkinsonian", rest_noise_std=0.12, active_amplitude=1.8,
                       activation_chance=0.03, rest_bout_range=(30, 90), active_bout_range=(10, 35)),
    "restless": Profile("Restless", rest_noise_std=0.05, active_amplitude=2.6,
                         activation_chance=0.10, rest_bout_range=(10, 35), active_bout_range=(20, 70)),
}


def generate_subject_stream(
    profile: Profile, num_windows: int, rng: random.Random
) -> Tuple[List[List[Sample]], List[int]]:
    """Return (windows, labels) for one synthetic subject.

    label = 1 while the subject is "active" (moving), 0 while "at rest".
    This label plays the role of ground truth activity for both the gate's
    skip-rate bookkeeping and the F1 comparison below.
    """
    windows: List[List[Sample]] = []
    labels: List[int] = []

    is_active = False
    bout_left = rng.randint(*profile.rest_bout_range)
    phase = 0.0

    for _ in range(num_windows):
        bout_left -= 1
        if bout_left <= 0:
            if is_active:
                is_active = False
                bout_left = rng.randint(*profile.rest_bout_range)
            elif rng.random() < profile.activation_chance * WINDOW_LEN:
                is_active = True
                bout_left = rng.randint(*profile.active_bout_range)
            else:
                bout_left = rng.randint(*profile.rest_bout_range)

        window: List[Sample] = []
        for _ in range(WINDOW_LEN):
            if is_active:
                phase += 0.35
                amp = profile.active_amplitude
                window.append((
                    amp * (0.7 * rng.random() - 0.35) + rng.gauss(0, amp * 0.4) + amp * _sin(phase),
                    rng.gauss(0, amp * 0.3),
                    1.0 + rng.gauss(0, amp * 0.3),
                ))
            else:
                s = profile.rest_noise_std
                window.append((rng.gauss(0, s), rng.gauss(0, s), 1.0 + rng.gauss(0, s)))
        windows.append(window)
        labels.append(1 if is_active else 0)

    return windows, labels


def _sin(x: float) -> float:
    # tiny local helper so evaluate.py doesn't need `import math` just for one call
    import math
    return math.sin(x)


# ---------------------------------------------------------------------------
# Step 2 & 3: Run the gate, then simulate always-on vs. gated classification
# ---------------------------------------------------------------------------
def classify(true_label: int, rng: random.Random) -> int:
    """Stand-in for a real HAR model's per-window prediction: correct with
    probability MODEL_ACCURACY, otherwise flips to the other class."""
    if rng.random() < MODEL_ACCURACY:
        return true_label
    return 1 - true_label


def macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    """Macro-averaged F1 across the two classes {0, 1}."""
    f1_scores = []
    for cls in (0, 1):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores)


@dataclass
class SubjectResult:
    subject: str
    windows: int
    har_calls: int
    skip_rate_pct: float
    power_saved_pct: float
    f1_gated: float
    f1_always_on: float


def estimate_power_saved_pct(windows: int, har_calls: int) -> float:
    """Rough relative energy model: a HAR inference costs far more than the
    gate's own variance computation, which costs a little more than pure
    idle. Numbers are illustrative, not measured on real hardware."""
    HAR_COST, GATE_COST, IDLE_COST = 1.0, 0.015, 0.001
    skipped = windows - har_calls
    gated_energy = har_calls * (HAR_COST + GATE_COST) + skipped * (GATE_COST + IDLE_COST)
    always_on_energy = windows * HAR_COST
    if always_on_energy == 0:
        return 0.0
    return max(0.0, 1 - gated_energy / always_on_energy) * 100


def run_subject(profile_key: str, profile: Profile, seed: int) -> SubjectResult:
    rng = random.Random(seed)
    windows, labels = generate_subject_stream(profile, WINDOWS_PER_SUBJECT, rng)

    gate = SageGate(GateConfig())
    decisions = gate.step_batch(windows)

    har_calls = sum(1 for d in decisions if d.run_model)
    skip_rate_pct = (1 - har_calls / len(decisions)) * 100

    # ALWAYS-ON: model runs every window.
    always_on_pred = [classify(label, rng) for label in labels]

    # GATED: model runs only on ACTIVE windows. While the gate is INACTIVE it
    # is, by definition, asserting "this person is resting" -- so the correct
    # default prediction there is REST (0), not whatever the classifier last
    # said. (An earlier version of this harness carried the last *active*
    # prediction forward through every subsequent rest window too, so one
    # correctly-classified "active" reading from hundreds of windows back
    # kept being treated as still-true indefinitely. That's not how a real
    # deployed gate behaves and it tanked the gated F1 score for no honest
    # reason -- fixed by defaulting to REST whenever the gate is closed.)
    gated_pred: List[int] = [
        classify(label, rng) if d.run_model else 0
        for d, label in zip(decisions, labels)
    ]

    return SubjectResult(
        subject=f"{profile.name} #{seed}",
        windows=len(decisions),
        har_calls=har_calls,
        skip_rate_pct=skip_rate_pct,
        power_saved_pct=estimate_power_saved_pct(len(decisions), har_calls),
        f1_gated=macro_f1(labels, gated_pred),
        f1_always_on=macro_f1(labels, always_on_pred),
    )


# ---------------------------------------------------------------------------
# Step 4: Run everything, print a summary, write results.csv
# ---------------------------------------------------------------------------
def main() -> None:
    subjects_per_profile = 3
    results: List[SubjectResult] = []

    # Subject seeds are derived from a single fixed master seed rather than
    # Python's built-in hash() -- string hashing is randomized per-process
    # by default (PYTHONHASHSEED), so hash(("calm", 0)) gives a different
    # number every run. That silently made this "benchmark" unreproducible:
    # nobody could rerun it and get the numbers printed in a report. A
    # dedicated random.Random(master_seed) instance fixes that.
    master_seed = 20240521
    seed_rng = random.Random(master_seed)

    for profile_key, profile in PROFILES.items():
        for i in range(subjects_per_profile):
            seed = seed_rng.randint(0, 2**31 - 1)
            results.append(run_subject(profile_key, profile, seed))

    print(f"{'Subject':<24}{'Windows':>9}{'HAR calls':>11}{'Skip %':>9}"
          f"{'Power %':>10}{'F1 gated':>10}{'F1 always':>11}")
    print("-" * 84)
    for r in results:
        print(f"{r.subject:<24}{r.windows:>9}{r.har_calls:>11}{r.skip_rate_pct:>8.1f}%"
              f"{r.power_saved_pct:>9.1f}%{r.f1_gated:>10.3f}{r.f1_always_on:>11.3f}")

    n = len(results)
    avg_skip = sum(r.skip_rate_pct for r in results) / n
    avg_power = sum(r.power_saved_pct for r in results) / n
    avg_f1_gated = sum(r.f1_gated for r in results) / n
    avg_f1_always = sum(r.f1_always_on for r in results) / n

    print("-" * 84)
    print(f"Average across {n} subjects: skip={avg_skip:.1f}%  power_saved={avg_power:.1f}%  "
          f"f1_gated={avg_f1_gated:.3f}  f1_always_on={avg_f1_always:.3f}")

    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "windows", "har_calls", "skip_rate_pct",
                          "power_saved_pct", "f1_gated", "f1_always_on"])
        for r in results:
            writer.writerow([r.subject, r.windows, r.har_calls, f"{r.skip_rate_pct:.2f}",
                              f"{r.power_saved_pct:.2f}", f"{r.f1_gated:.4f}", f"{r.f1_always_on:.4f}"])
    print("\nWrote results.csv")


if __name__ == "__main__":
    main()
