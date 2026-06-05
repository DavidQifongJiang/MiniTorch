import random
import time
from collections import defaultdict

import torch

import minitorch


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class Network(torch.nn.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        h = self.layer1.forward(x).relu()
        h = self.layer2.forward(h).relu()
        return self.layer3.forward(h).sigmoid()


class Linear(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weight = torch.nn.Parameter(2 * (torch.rand((in_size, out_size)) - 0.5))
        self.bias = torch.nn.Parameter(2 * (torch.rand((out_size,)) - 0.5))

    def forward(self, x):
        return x @ self.weight + self.bias


class TorchTrain:
    def __init__(
        self,
        hidden_layers,
        batch_size=None,
        preload_batches=False,
        evaluate=True,
        collect_timing=False,
    ):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)
        self.batch_size = batch_size
        self.preload_batches = preload_batches
        self.evaluate = evaluate
        self.collect_timing = collect_timing
        self.last_timing = {}

    def run_one(self, x):
        return self.model.forward(torch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(torch.tensor(X)).detach()

    def train(
        self,
        data,
        learning_rate,
        max_epochs=500,
        log_fn=default_log_fn,
    ):
        self.model = Network(self.hidden_layers)
        self.max_epochs = max_epochs
        model = self.model
        batch_size = self.batch_size or data.N

        losses = []
        timings = defaultdict(float)
        preloaded_batches = None
        total_epoch_time = 0.0

        if self.preload_batches:
            start = time.perf_counter()
            examples = list(zip(data.X, data.y))
            random.shuffle(examples)
            X_shuf, y_shuf = zip(*examples)
            preloaded_batches = [
                (
                    torch.tensor(
                        X_shuf[i : i + batch_size],
                        dtype=torch.float32,
                        requires_grad=True,
                    ),
                    torch.tensor(y_shuf[i : i + batch_size], dtype=torch.float32),
                )
                for i in range(0, len(X_shuf), batch_size)
            ]
            timings["data_prepare_seconds"] += time.perf_counter() - start

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.perf_counter()
            total_loss = 0.0
            if preloaded_batches is None:
                shuffled = list(zip(data.X, data.y))
                random.shuffle(shuffled)
                X_shuf, y_shuf = zip(*shuffled)
                batch_iter = range(0, len(X_shuf), batch_size)
            else:
                batch_iter = list(preloaded_batches)
                random.shuffle(batch_iter)

            for item in batch_iter:
                if preloaded_batches is None:
                    i = item
                    start = time.perf_counter()
                    X_batch = torch.tensor(
                        X_shuf[i : i + batch_size], requires_grad=True
                    )
                    y_batch = torch.tensor(y_shuf[i : i + batch_size])
                    timings["data_prepare_seconds"] += time.perf_counter() - start
                else:
                    X_batch, y_batch = item

                start = time.perf_counter()
                out = model.forward(X_batch).view(len(y_batch))
                timings["forward_seconds"] += time.perf_counter() - start

                start = time.perf_counter()
                probs = (out * y_batch) + (out - 1.0) * (y_batch - 1.0)
                loss = -probs.log().sum()
                loss.view(1).backward()
                total_loss += loss.reshape(-1).item()
                timings["backward_seconds"] += time.perf_counter() - start

                start = time.perf_counter()
                for p in model.parameters():
                    if p.grad is not None:
                        p.data = p.data - learning_rate * (
                            p.grad / float(len(y_batch))
                        )
                        p.grad.zero_()
                timings["optimizer_seconds"] += time.perf_counter() - start

            losses.append(total_loss)

            if self.evaluate and (epoch % 10 == 0 or epoch == max_epochs):
                start = time.perf_counter()
                with torch.no_grad():
                    out = model.forward(torch.tensor(data.X)).view(data.N)
                    y = torch.tensor(data.y)
                    pred = out > 0.5
                    correct = ((y == 1) * pred).sum() + ((y == 0) * (~pred)).sum()
                timings["evaluation_seconds"] += time.perf_counter() - start
                log_fn(epoch, total_loss, correct.item(), losses)

            total_epoch_time += time.perf_counter() - epoch_start

        timings["total_epoch_seconds"] = total_epoch_time
        timings["epochs"] = max_epochs
        self.last_timing = dict(timings) if self.collect_timing else {}


if __name__ == "__main__":
    PTS = 250
    HIDDEN = 10
    RATE = 0.5
    TorchTrain(HIDDEN).train(minitorch.datasets["Xor"](PTS), RATE)
