"""Tests for har_cnn.py -- the 1D-CNN architecture (shape/plumbing only;
these do not train anything, see tests/test_train.py for that)."""

import pytest

torch = pytest.importorskip("torch")

from har_cnn import HARConv1D


class TestHARConv1DShapes:
    def test_output_shape_matches_num_classes(self):
        model = HARConv1D(in_channels=3, num_classes=6)
        x = torch.randn(8, 3, 128)
        logits = model(x)
        assert logits.shape == (8, 6)

    def test_batch_size_of_one_works(self):
        model = HARConv1D(in_channels=3, num_classes=6)
        x = torch.randn(1, 3, 128)
        logits = model(x)
        assert logits.shape == (1, 6)

    def test_works_with_different_window_lengths(self):
        # Global average pooling means the architecture should not care
        # about window_len, unlike a plain Flatten-based classifier head.
        model = HARConv1D(in_channels=3, num_classes=6)
        for window_len in (64, 128, 200):
            x = torch.randn(2, 3, window_len)
            logits = model(x)
            assert logits.shape == (2, 6)

    def test_rejects_wrong_input_rank(self):
        model = HARConv1D(in_channels=3, num_classes=6)
        with pytest.raises(Exception):
            model(torch.randn(3, 128))  # missing batch dimension


class TestHARConv1DParameters:
    def test_has_trainable_parameters(self):
        model = HARConv1D(in_channels=3, num_classes=6)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert num_params > 0

    def test_gradients_flow_on_backward_pass(self):
        model = HARConv1D(in_channels=3, num_classes=6)
        x = torch.randn(4, 3, 128)
        y = torch.randint(0, 6, (4,))
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None for g in grads)
        assert any(g.abs().sum().item() > 0 for g in grads)
