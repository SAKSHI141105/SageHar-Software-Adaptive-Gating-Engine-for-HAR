"""Guards the last step of the deployment chain: web_real_data.js's
embedded base64 model must actually match the current ONNX export.

Without this, a retrain + re-export with a forgotten
`generate_real_data_js.py` step would silently ship a stale model to the
dashboard while every other test (which checks .pt vs .onnx, not
.onnx vs web_real_data.js) stays green.
"""

import base64
import hashlib
import re
from pathlib import Path

import pytest

ONNX_PATH = Path("data/processed/har_cnn_uci.onnx")
JS_PATH = Path("web_real_data.js")

pytestmark = pytest.mark.skipif(
    not (ONNX_PATH.exists() and JS_PATH.exists()),
    reason="ONNX export or web_real_data.js not present",
)


def _extract_embedded_model_bytes() -> bytes:
    content = JS_PATH.read_text()
    match = re.search(r'REAL_MODEL_BASE64 = "([^"]+)"', content)
    assert match, "REAL_MODEL_BASE64 constant not found in web_real_data.js"
    return base64.b64decode(match.group(1))


class TestWebRealDataMatchesOnnxExport:
    def test_embedded_model_bytes_match_onnx_file(self):
        onnx_bytes = ONNX_PATH.read_bytes()
        embedded_bytes = _extract_embedded_model_bytes()

        onnx_hash = hashlib.sha256(onnx_bytes).hexdigest()
        embedded_hash = hashlib.sha256(embedded_bytes).hexdigest()

        assert embedded_hash == onnx_hash, (
            "web_real_data.js's embedded model is stale -- rerun "
            "`python generate_real_data_js.py` after any retrain/re-export."
        )
