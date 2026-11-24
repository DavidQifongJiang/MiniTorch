"""Regression tests for optimized execution and numerical edge cases."""

import os
import random

import numpy as np
import pytest

import minitorch
from minitorch.models import MLP, binary_loss, from_numpy, get_backend


@pytest.fixture(params=["cpu"] + (["cuda"] if os.getenv("MINITORCH_RUN_CUDA_TESTS") == "true" else []))
def backend(request):
    return get_backend(request.param)


def test_gradient_flags_and_no_grad():
    x = minitorch.tensor([2.0], requires_grad=False)
    assert not (x * x).requires_grad()
    x.requires_grad_(True)
    with minitorch.no_grad():
        assert not (x * x).requires_grad()
        with minitorch.no_grad():
            assert not (x + 1).requires_grad()
    y = x * x
    y.backward()
    assert x.grad[0] == 4.0
    x.requires_grad_(False)
    assert not (x * x).requires_grad()
    with pytest.raises(RuntimeError):
        with minitorch.no_grad():
            raise RuntimeError("restore context")
    assert minitorch.is_grad_enabled()


def test_reference_backend_matmul():
    a = minitorch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    b = minitorch.tensor([[2., 1.], [0., 3.], [1., 4.]], requires_grad=True)
    result = a @ b
    np.testing.assert_array_equal(result.to_numpy(), [[5., 19.], [14., 43.]])
    result.sum().backward()
    np.testing.assert_array_equal(a.grad.to_numpy(), [[3., 3., 5.], [3., 3., 5.]])
    np.testing.assert_array_equal(b.grad.to_numpy(), [[5., 5.], [7., 7.], [9., 9.]])


@pytest.mark.parametrize("shape", [(31, 33, 35), (33, 65, 17), (128, 128, 128), (256, 256, 256), (512, 512, 512)])
def test_matmul_tile_boundaries(backend, shape):
    m, k, n = shape
    rng = np.random.default_rng(12)
    a, b = rng.normal(size=(m, k)), rng.normal(size=(k, n))
    actual = (from_numpy(a, backend) @ from_numpy(b, backend)).to_numpy()
    # einsum supplies an independent reference without loading a second BLAS
    # OpenMP runtime in environments that also import PyTorch.
    np.testing.assert_allclose(actual, np.einsum("ik,kj->ij", a, b), rtol=1e-9, atol=1e-9)


def test_batched_transposed_matmul_gradients(backend):
    rng = np.random.default_rng(13)
    a, b = rng.normal(size=(2, 7, 5)), rng.normal(size=(1, 9, 7))
    x, y = from_numpy(a, backend), from_numpy(b, backend)
    x.requires_grad_(True)
    y.requires_grad_(True)
    z = x.permute(0, 2, 1) @ y.permute(0, 2, 1)
    expected = np.einsum("...ki,...jk->...ij", a, b)
    np.testing.assert_allclose(z.to_numpy(), expected, atol=1e-10)
    z.sum().backward()
    upstream = np.ones_like(expected)
    np.testing.assert_allclose(x.grad.to_numpy(), np.einsum("...mn,...nk->...km", upstream, b), atol=1e-10)
    expected_b = np.einsum("...km,...mn->...nk", a, upstream).sum(axis=0, keepdims=True)
    np.testing.assert_allclose(y.grad.to_numpy(), expected_b, atol=1e-10)


@pytest.mark.parametrize("width", [1, 31, 33, 255, 257, 1023, 1024, 1025, 2051])
def test_reduction_full_dimension(backend, width):
    values = np.random.default_rng(width).normal(size=(3, width))
    x = from_numpy(values, backend)
    np.testing.assert_allclose(x.sum(1).to_numpy(), values.sum(1, keepdims=True), atol=1e-10)
    np.testing.assert_allclose(x.permute(1, 0).sum(0).to_numpy(), values.T.sum(0, keepdims=True), atol=1e-10)
    np.testing.assert_allclose(minitorch.max(x, 1).to_numpy(), values.max(1, keepdims=True), atol=1e-10)


def test_broadcast_and_layout(backend):
    a = np.arange(24.0).reshape(2, 3, 4)
    x = from_numpy(a, backend).permute(2, 0, 1)
    y = from_numpy(np.array([1.0, 2.0, 3.0]), backend)
    np.testing.assert_array_equal((x * y + 2).to_numpy(), a.transpose(2, 0, 1) * [1, 2, 3] + 2)
    assert x._tensor._storage is x.permute(1, 2, 0)._tensor._storage
    contiguous = x.contiguous()
    assert contiguous.view(24)._tensor._storage is contiguous._tensor._storage
    assert isinstance(x.to_numpy(), np.ndarray)
    if backend.cuda:
        assert hasattr(x._tensor._storage, "__cuda_array_interface__")
        assert hasattr((x + y)._tensor._storage, "__cuda_array_interface__")


def test_large_permuted_elementwise_and_reduction(backend):
    values = np.random.default_rng(44).normal(size=(129, 257))
    bias = np.arange(129, dtype=np.float64)
    x = from_numpy(values, backend).permute(1, 0)
    y = from_numpy(bias, backend)
    np.testing.assert_allclose((-x + y).to_numpy(), -values.T + bias, atol=1e-10)
    np.testing.assert_allclose(x.sum(1).to_numpy(), values.T.sum(1, keepdims=True), atol=1e-10)


def test_stable_softmax_and_gradient(backend):
    values = np.array([[1000., 1001., 999.], [-1001., -999., -1000.]])
    x = from_numpy(values, backend)
    x.requires_grad_(True)
    shifted = values - values.max(1, keepdims=True)
    expected = np.exp(shifted) / np.exp(shifted).sum(1, keepdims=True)
    np.testing.assert_allclose(minitorch.softmax(x, 1).to_numpy(), expected, atol=1e-12)
    result = minitorch.logsoftmax(x, 1)
    np.testing.assert_allclose(result.to_numpy(), np.log(expected), atol=1e-12)
    result.sum().backward()
    np.testing.assert_allclose(x.grad.to_numpy(), 1 - 3 * expected, atol=1e-10)


def test_dropout_scaling_and_backend(backend):
    random.seed(42)
    x = from_numpy(np.ones((128,)), backend)
    x.requires_grad_(True)
    y = minitorch.dropout(x, 0.5)
    assert set(np.unique(y.to_numpy())) == {0.0, 2.0}
    assert y.backend is backend
    y.sum().backward()
    np.testing.assert_array_equal(x.grad.to_numpy(), y.to_numpy())
    assert minitorch.dropout(x, 0.5, ignore=True) is x
    with pytest.raises(ValueError):
        minitorch.dropout(x, -0.1)


def test_optimizer_keeps_parameters_as_leaves(backend):
    model = MLP(4, backend, seed=7)
    x = from_numpy(np.array([[0.1, 0.3], [0.8, 0.2]]), backend)
    y = from_numpy(np.array([[0.0], [1.0]]), backend)
    optim = minitorch.SGD(model.parameters(), 0.01)
    for _ in range(3):
        optim.zero_grad()
        binary_loss(model(x), y).backward()
        optim.step()
        assert all(p.value.is_leaf() for p in model.parameters())
    assert x.grad is None
    with minitorch.no_grad():
        assert not model(x).requires_grad()
