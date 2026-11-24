"""Reproducible synthetic classification with separate validation data."""

import time

import numpy as np

from .autodiff import no_grad
from .models import MLP, binary_loss, from_numpy, get_backend, make_dataset
from .optim import SGD


def evaluate(model, x, y, batch_size=256):
    """Evaluate every example, including a final partial batch."""
    if batch_size < 1 or len(x) == 0 or len(x) != len(y):
        raise ValueError("Evaluation needs matching nonempty data and a positive batch size")
    was_training = model.training
    model.eval()
    correct, loss_sum = 0, 0.0
    try:
        with no_grad():
            for start in range(0, len(x), batch_size):
                target = y[start:start + batch_size]
                prediction = model(from_numpy(x[start:start + batch_size], model.backend)).to_numpy()
                probability = prediction * target + (prediction - 1) * (target - 1)
                loss_sum += float(-(np.log(probability + 1e-6)).sum())
                correct += int(((prediction > 0.5) == target).sum())
    finally:
        if was_training:
            model.train()
    return {"examples": len(x), "accuracy": correct / len(x), "loss": loss_sum / len(x)}


def train(*, dataset="xor", points=1000, hidden=16, epochs=100, batch_size=64,
          learning_rate=0.1, seed=1337, backend="cpu", validation_points=1000, log=None):
    """Train an MLP and return the model plus a serializable run record."""
    if epochs < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive and finite")
    target = get_backend(backend)
    model = MLP(hidden, target, seed)
    x, y = make_dataset(points, seed, dataset)
    validation_x, validation_y = make_dataset(validation_points, seed + 1, dataset)
    # Batch storage is created once. Shuffle batch order deterministically each epoch.
    batches = [(from_numpy(x[i:i + batch_size], target), from_numpy(y[i:i + batch_size], target))
               for i in range(0, points, batch_size)]
    optimizer = SGD(model.parameters(), learning_rate)
    history = []
    start = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch + 1).permutation(len(batches))
        for index in order:
            inputs, labels = batches[index]
            optimizer.zero_grad()
            binary_loss(model(inputs), labels).backward()
            optimizer.step()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            metrics = {"epoch": epoch + 1, "validation": evaluate(model, validation_x, validation_y, batch_size)}
            history.append(metrics)
            if log is not None:
                log(metrics)
    report = {"dataset": dataset, "points": points, "hidden": hidden, "epochs": epochs,
              "batch_size": batch_size, "learning_rate": learning_rate, "seed": seed,
              "validation_seed": seed + 1, "validation_points": validation_points,
              "backend": backend, "dtype": "float64", "history": history,
              "elapsed_seconds_including_compilation_and_validation": time.perf_counter() - start}
    return model, report
