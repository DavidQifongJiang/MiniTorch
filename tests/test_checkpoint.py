import json

import numpy as np
import pytest

from minitorch.checkpoint import load_checkpoint, save_checkpoint
from minitorch.models import MLP, from_numpy, get_backend, make_dataset
from minitorch.training import evaluate, train


def test_checkpoint_roundtrip_and_independent_evaluation(tmp_path):
    model, report = train(dataset="simple", points=17, hidden=4, epochs=2, batch_size=8, validation_points=19)
    x, y = make_dataset(23, 100, "simple")
    before = model(from_numpy(x, model.backend)).to_numpy()
    path = tmp_path / "model.npz"
    save_checkpoint(model, path, report)
    restored, metadata = load_checkpoint(path)
    np.testing.assert_array_equal(restored(from_numpy(x, restored.backend)).to_numpy(), before)
    assert metadata == report
    metrics = evaluate(restored, x, y, batch_size=7)
    assert metrics["examples"] == 23
    assert metrics["accuracy"] == float(((before > 0.5) == y).mean())
    assert not restored.training


def test_checkpoint_rejects_invalid_parameter(tmp_path):
    model = MLP(4)
    path = tmp_path / "bad.npz"
    save_checkpoint(model, path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["layer1.weights"] = np.zeros((100, 100))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="shape or dtype"):
        load_checkpoint(path)


def test_cli_train_evaluate(tmp_path, capsys):
    from minitorch.cli import main
    path, report = tmp_path / "cli.npz", tmp_path / "run.json"
    assert main(["train", "--dataset", "simple", "--points", "16", "--hidden", "4", "--epochs", "1",
                 "--validation-points", "17", "--threads", "2", "--checkpoint", str(path), "--report", str(report)]) == 0
    assert json.loads(report.read_text())["validation_points"] == 17
    capsys.readouterr()
    assert main(["evaluate", str(path), "--points", "19", "--threads", "2"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["examples"] == 19
    assert result["seed"] == 1339
