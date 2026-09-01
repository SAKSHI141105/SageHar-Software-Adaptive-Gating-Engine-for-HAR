"""
har_cnn.py
==========
A standard, small 1D-CNN classifier for Human Activity Recognition.

This is the "expensive" model that SAGE-HAR is trying to avoid running
unnecessarily. It's intentionally simple -- three convolutional blocks
followed by global average pooling and a linear classifier head -- which
is a common, well-understood architecture for HAR on windowed IMU data.

Input shape:  (batch_size, 3, window_size)
                        ^     ^
                        |     +-- one accelerometer window, e.g. 128 samples
                        +-- 3 channels: x, y, z axes
Output shape: (batch_size, num_classes)  -- raw logits (no softmax applied)

Requires PyTorch: `pip install torch`
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HARConv1D(nn.Module):
    """Three Conv1d -> BatchNorm -> ReLU blocks, each halving the time
    dimension with a MaxPool, followed by global average pooling and a
    linear output layer.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 6, base_channels: int = 32) -> None:
        super().__init__()

        # --- Block 1: 3 channels in -> base_channels out -----------------
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        # --- Block 2: base_channels -> base_channels * 2 ------------------
        self.block2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        # --- Block 3: base_channels * 2 -> base_channels * 4 ---------------
        self.block3 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        # Global average pooling collapses the time dimension to length 1,
        # regardless of the input window size, so this model works with any
        # window_size without changing the architecture.
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)

        # A small classifier head on top of the pooled features.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(base_channels * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch_size, 3, window_size) -> logits: (batch_size, num_classes)"""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        return self.classifier(x)


if __name__ == "__main__":
    # Quick shape check: run one fake batch through the model.
    batch_size, channels, window_size, num_classes = 8, 3, 128, 6

    model = HARConv1D(in_channels=channels, num_classes=num_classes)
    dummy_input = torch.randn(batch_size, channels, window_size)

    logits = model(dummy_input)
    print(f"Input shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    assert logits.shape == (batch_size, num_classes)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
