"""
train.py
========
Actually trains har_cnn.py's HAR1DCNN on a real dataset: UCI-HAR
("Human Activity Recognition Using Smartphones"), 30 subjects wearing a
phone on their waist, 6 activity classes, already windowed by the dataset
authors into 128-sample chunks at 50Hz (2.56s windows, 50% overlap).

This is a genuine train/validate/test run -- real gradient descent, real
held-out accuracy -- not a shape-check smoke test. Steps:

  1. Load the "total_acc" (x, y, z) inertial signals -- raw accelerometer
     readings including gravity, matching what sage_gate.py's SVM expects
     downstream. Each row in these files is already one 128-sample window.
  2. Split UCI-HAR's own train/ folder further into train/val (90/10) so we
     can watch for overfitting during training; UCI-HAR's test/ folder is
     held out completely and only touched once, at the very end.
  3. Train HAR1DCNN with cross-entropy loss and the Adam optimizer for a
     fixed number of epochs, printing train/val loss and accuracy each
     epoch, and keeping the checkpoint with the best validation accuracy.
  4. Load that best checkpoint and report final test accuracy + macro-F1
     on data the model has never seen.
  5. Save the trained weights to data/processed/har_cnn_uci.pt.

Run:
    pip install -r requirements.txt
    python train.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from har_cnn import HARConv1D

DATA_DIR = Path("data/raw/uci_har")
CHECKPOINT_PATH = Path("data/processed/har_cnn_uci.pt")
ACTIVITY_NAMES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.1
SEED = 42


# ---------------------------------------------------------------------------
# Step 1: Load UCI-HAR's raw inertial signal windows into tensors
# ---------------------------------------------------------------------------
def _load_signal_file(path: Path) -> torch.Tensor:
    """Each line in these files is one 128-sample window for one axis.
    Returns a (num_windows, 128) tensor."""
    rows = []
    with open(path) as f:
        for line in f:
            rows.append([float(v) for v in line.split()])
    return torch.tensor(rows, dtype=torch.float32)


def load_split(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load one of UCI-HAR's 'train' or 'test' splits.

    Returns:
        X: (num_windows, 3, 128) float tensor -- 3 channels are total_acc x,y,z
        y: (num_windows,) long tensor of class indices in [0, 5]
    """
    signal_dir = DATA_DIR / split / "Inertial Signals"
    x_axis = _load_signal_file(signal_dir / f"total_acc_x_{split}.txt")
    y_axis = _load_signal_file(signal_dir / f"total_acc_y_{split}.txt")
    z_axis = _load_signal_file(signal_dir / f"total_acc_z_{split}.txt")

    # Stack the 3 axes into channels: (num_windows, 3, window_len)
    X = torch.stack([x_axis, y_axis, z_axis], dim=1)

    label_file = "y_train.txt" if split == "train" else "y_test.txt"
    with open(DATA_DIR / split / label_file) as f:
        # UCI-HAR labels are 1-indexed (1..6); shift to 0-indexed for PyTorch.
        labels = [int(line.strip()) - 1 for line in f if line.strip()]
    y = torch.tensor(labels, dtype=torch.long)

    return X, y


# ---------------------------------------------------------------------------
# Step 2 & 3: Train
# ---------------------------------------------------------------------------
def run_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer=None) -> Tuple[float, float]:
    """One pass over `loader`. If `optimizer` is given, trains; otherwise
    just evaluates. Returns (average_loss, accuracy)."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for X_batch, y_batch in loader:
            if is_training:
                optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == y_batch).sum().item()
            total += X_batch.size(0)

    return total_loss / total, correct / total


def macro_f1(y_true: List[int], y_pred: List[int], num_classes: int) -> float:
    """Macro-averaged F1 across all classes, stdlib only."""
    scores = []
    for c in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        scores.append(f1)
    return sum(scores) / len(scores)


def main() -> None:
    torch.manual_seed(SEED)

    if not DATA_DIR.exists():
        raise SystemExit(
            f"Expected UCI-HAR data at {DATA_DIR}/ (train/ and test/ with "
            "Inertial Signals/ + y_*.txt inside). Download the dataset from "
            "https://archive.ics.uci.edu/dataset/240 and place it there."
        )

    print("Loading UCI-HAR...")
    X_train_full, y_train_full = load_split("train")
    X_test, y_test = load_split("test")
    print(f"  train+val windows: {X_train_full.shape[0]}   test windows: {X_test.shape[0]}")
    print(f"  window shape: {tuple(X_train_full.shape[1:])}  (channels, window_len)")

    full_dataset = TensorDataset(X_train_full, y_train_full)
    val_size = int(len(full_dataset) * VAL_FRACTION)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
    )
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = HARConv1D(in_channels=3, num_classes=len(ACTIVITY_NAMES))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model parameters: {num_params:,}")
    print(f"\nTraining for {EPOCHS} epochs (batch_size={BATCH_SIZE}, lr={LEARNING_RATE})...\n")
    print(f"{'epoch':>5}  {'train_loss':>10}  {'train_acc':>9}  {'val_loss':>10}  {'val_acc':>9}  {'time':>6}")

    best_val_acc = 0.0
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        elapsed = time.time() - t0

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            marker = "  <- saved (best val_acc)"

        print(f"{epoch:5d}  {train_loss:10.4f}  {train_acc:8.2%}  {val_loss:10.4f}  {val_acc:8.2%}  "
              f"{elapsed:5.1f}s{marker}")

    # --- Final evaluation on the held-out test set, using the BEST checkpoint ---
    print(f"\nLoading best checkpoint (val_acc={best_val_acc:.2%}) for final test evaluation...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y_batch.tolist())

    test_acc = sum(1 for t, p in zip(all_labels, all_preds) if t == p) / len(all_labels)
    test_f1 = macro_f1(all_labels, all_preds, num_classes=len(ACTIVITY_NAMES))

    print(f"\n{'='*50}")
    print(f"FINAL TEST RESULTS (never seen during training)")
    print(f"{'='*50}")
    print(f"Test accuracy: {test_acc:.2%}")
    print(f"Test macro-F1: {test_f1:.4f}")
    print(f"{'='*50}")

    print(f"\nPer-class breakdown:")
    print(f"{'class':<20}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}")
    for c, name in enumerate(ACTIVITY_NAMES):
        tp = sum(1 for t, p in zip(all_labels, all_preds) if t == c and p == c)
        fp = sum(1 for t, p in zip(all_labels, all_preds) if t != c and p == c)
        fn = sum(1 for t, p in zip(all_labels, all_preds) if t == c and p != c)
        support = sum(1 for t in all_labels if t == c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"{name:<20}{precision:>10.2%}{recall:>9.2%}{f1:>8.3f}{support:>9}")

    print(f"\nSaved trained weights to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
