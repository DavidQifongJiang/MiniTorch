"""Exercise the CNN without downloading MNIST or requiring demo dependencies."""

import numpy as np

import minitorch
from examples.mnist_cnn_demo import BACKEND, Network
from minitorch.models import from_numpy


def test_cnn_partial_batch_inference_and_training():
    model = Network()
    images = from_numpy(np.random.default_rng(42).random((3, 1, 28, 28)), BACKEND)
    model.eval()
    with minitorch.no_grad():
        first = model(images)
        second = model(images)
    assert first.shape == (3, 10)
    assert not first.requires_grad()
    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())
    np.testing.assert_allclose(first.exp().sum(1).to_numpy(), np.ones((3, 1)), atol=1e-10)
    model.train()
    labels = from_numpy(np.eye(10)[[1, 3, 5]], BACKEND)
    loss = -(model(images) * labels).sum(1).mean()
    loss.backward()
    assert all(p.value.grad is not None and np.isfinite(p.value.grad.to_numpy()).all() for p in model.parameters())
