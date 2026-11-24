"""Optional PyTorch oracle; run with pip install '.[dev,benchmark]'."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import minitorch
from minitorch.models import MLP, binary_loss, from_numpy, get_backend, make_dataset


@pytest.mark.parametrize("backend_name", ["cpu"] + (["cuda"] if os.getenv("MINITORCH_RUN_CUDA_TESTS") == "true" else []))
@pytest.mark.parametrize("hidden,batch", [(4, 7), (33, 35)])
def test_forward_backward_and_sgd_match_pytorch(backend_name, hidden, batch):
    backend = get_backend(backend_name)
    model = MLP(hidden, backend, seed=15)
    x, y = make_dataset(batch, seed=16)
    inputs, labels = from_numpy(x, backend), from_numpy(y, backend)
    parameters = {name: torch.tensor(p.value.to_numpy().tolist(), dtype=torch.float64, requires_grad=True)
                  for name, p in model.named_parameters()}
    expected = torch.tensor(x.tolist(), dtype=torch.float64)
    for i in range(1, 4):
        expected = expected @ parameters[f"layer{i}.weights"] + parameters[f"layer{i}.bias"]
        expected = expected.relu() if i < 3 else expected.sigmoid()
    actual = model(inputs)
    np.testing.assert_allclose(actual.to_numpy(), expected.detach().tolist(), rtol=1e-9, atol=1e-10)
    target = torch.tensor(y.tolist(), dtype=torch.float64)
    probability = expected * target + (expected - 1) * (target - 1)
    expected_loss = -(probability + 1e-6).log().mean()
    loss = binary_loss(actual, labels)
    expected_loss.backward()
    loss.backward()
    for name, p in model.named_parameters():
        np.testing.assert_allclose(p.value.grad.to_numpy(), parameters[name].grad.tolist(), rtol=1e-8, atol=1e-10)
    minitorch.SGD(model.parameters(), 0.05).step()
    for name, p in model.named_parameters():
        updated = parameters[name].detach() - 0.05 * parameters[name].grad
        np.testing.assert_allclose(p.value.to_numpy(), updated.tolist(), rtol=1e-9, atol=1e-10)
