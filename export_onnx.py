"""
export_onnx.py
===============
Exports the trained har_cnn_uci.pt checkpoint to ONNX so the dashboard
(index.html) can run the real trained model in-browser via onnxruntime-web,
instead of only ever showing a synthetic simulation.

Run:
    python train.py          # if data/processed/har_cnn_uci.pt doesn't exist yet
    python export_onnx.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from har_cnn import HARConv1D
from train import ACTIVITY_NAMES

CHECKPOINT_PATH = Path("data/processed/har_cnn_uci.pt")
ONNX_PATH = Path("data/processed/har_cnn_uci.onnx")


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise SystemExit(f"No trained checkpoint at {CHECKPOINT_PATH}. Run `python train.py` first.")

    model = HARConv1D(in_channels=3, num_classes=len(ACTIVITY_NAMES))
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    dummy_input = torch.randn(1, 3, 128)

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        input_names=["sensor_window"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,  # the newer dynamo-based exporter needs the extra
                        # `onnxscript` package; the classic TorchScript-based
                        # exporter needs nothing beyond torch itself.
    )

    size_kb = ONNX_PATH.stat().st_size / 1024
    print(f"Exported {ONNX_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
