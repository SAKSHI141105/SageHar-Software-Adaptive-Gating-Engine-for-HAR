"""Tests that the checked-in ONNX export actually matches the PyTorch
checkpoint it was exported from -- guards against the dashboard's "Real
Model Replay" mode silently going stale if the model is retrained and
`python export_onnx.py` isn't rerun to match.

Skips automatically if the checkpoint/export/onnxruntime aren't present,
so this doesn't break the suite in a fresh checkout that hasn't trained
yet -- see test_train.py's UCI_HAR_PRESENT skip for the same pattern.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
onnxruntime = pytest.importorskip("onnxruntime")
np = pytest.importorskip("numpy")

from har_cnn import HARConv1D

CHECKPOINT_PATH = Path("data/processed/har_cnn_uci.pt")
ONNX_PATH = Path("data/processed/har_cnn_uci.onnx")

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_PATH.exists() and ONNX_PATH.exists()),
    reason="trained checkpoint or ONNX export not present",
)


class TestOnnxExportMatchesCheckpoint:
    def test_onnx_predictions_match_pytorch(self):
        model = HARConv1D(in_channels=3, num_classes=6)
        model.load_state_dict(torch.load(CHECKPOINT_PATH))
        model.eval()

        session = onnxruntime.InferenceSession(str(ONNX_PATH))
        input_name = session.get_inputs()[0].name

        torch.manual_seed(0)
        x = torch.randn(5, 3, 128)
        with torch.no_grad():
            torch_out = model(x).numpy()

        # The checked-in export was made with a fixed batch size of 1.
        onnx_outputs = [session.run(None, {input_name: x[i:i + 1].numpy()})[0] for i in range(5)]
        onnx_out = np.concatenate(onnx_outputs, axis=0)

        assert (torch_out.argmax(axis=1) == onnx_out.argmax(axis=1)).all()
        assert np.abs(torch_out - onnx_out).max() < 1e-3
