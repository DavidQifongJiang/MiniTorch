import random
import time

import numba

import minitorch


FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = (
    minitorch.TensorBackend(minitorch.CudaOps) if numba.cuda.is_available() else None
)


def default_log_fn(epoch, total_loss, correct, losses, time_elapsed):
    print(
        f"Epoch {epoch}: time={time_elapsed:.2f}s "
        f"loss={total_loss:.4f} correct={correct}"
    )


def RParam(*shape, backend):
    r = minitorch.rand(shape, backend=backend) - 0.5
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden, backend):
        super().__init__()
        self.layer1 = Linear(2, hidden, backend)
        self.layer2 = Linear(hidden, hidden, backend)
        self.layer3 = Linear(hidden, 1, backend)

    def forward(self, x):
        h = self.layer1.forward(x).relu()
        h = self.layer2.forward(h).relu()
        return self.layer3.forward(h).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size, backend):
        super().__init__()
        self.weights = RParam(in_size, out_size, backend=backend)
        self.bias = minitorch.Parameter(
            minitorch.zeros((out_size,), backend=backend) + 0.1
        )
        self.out_size = out_size

    def forward(self, x):
        x = x.view(*x.shape, 1)
        weight = self.weights.value.view(1, *self.weights.value.shape)
        bias = self.bias.value.view(1, self.out_size)
        return (x * weight).sum(1).view(x.shape[0], self.out_size) + bias


class FastTrain:
    def __init__(self, hidden_layers, backend=FastTensorBackend, batch_size=10):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers, backend)
        self.backend = backend
        self.batch_size = batch_size

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x], backend=self.backend))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X, backend=self.backend))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.model = Network(self.hidden_layers, self.backend)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        batch = self.batch_size
        losses = []
        total_epoch_time = 0.0

        for epoch in range(max_epochs):
            start_time = time.time()
            total_loss = 0.0
            shuffled = list(zip(data.X, data.y))
            random.shuffle(shuffled)
            X_shuf, y_shuf = zip(*shuffled)

            for i in range(0, len(X_shuf), batch):
                optim.zero_grad()
                X = minitorch.tensor(X_shuf[i : i + batch], backend=self.backend)
                y = minitorch.tensor(y_shuf[i : i + batch], backend=self.backend)

                out = self.model.forward(X).view(y.shape[0])
                prob = (out * y) + (out - 1.0) * (y - 1.0)
                loss = -prob.log()
                (loss / y.shape[0]).sum().view(1).backward()
                total_loss = loss.sum().view(1)[0]
                optim.step()

            losses.append(total_loss)
            time_elapsed = time.time() - start_time
            total_epoch_time += time_elapsed

            if epoch % 10 == 0 or epoch == max_epochs - 1:
                X = minitorch.tensor(data.X, backend=self.backend)
                y = minitorch.tensor(data.y, backend=self.backend)
                out = self.model.forward(X).view(y.shape[0])
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses, time_elapsed)

        print(
            "Average time per epoch: "
            f"{total_epoch_time / max_epochs:.2f}s for {max_epochs} epochs"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--PTS", type=int, default=50, help="number of points")
    parser.add_argument("--HIDDEN", type=int, default=10, help="number of hiddens")
    parser.add_argument("--RATE", type=float, default=0.05, help="learning rate")
    parser.add_argument("--BACKEND", default="cpu", help="cpu or gpu")
    parser.add_argument("--DATASET", default="simple", help="simple, split, or xor")
    parser.add_argument("--BATCH", type=int, default=10, help="batch size")

    args = parser.parse_args()

    if args.DATASET == "xor":
        data = minitorch.datasets["Xor"](args.PTS)
    elif args.DATASET == "split":
        data = minitorch.datasets["Split"](args.PTS)
    else:
        data = minitorch.datasets["Simple"](args.PTS)

    if args.BACKEND == "gpu":
        if GPUBackend is None:
            raise RuntimeError("CUDA backend requested, but CUDA is not available.")
        backend = GPUBackend
    else:
        backend = FastTensorBackend

    FastTrain(args.HIDDEN, backend=backend, batch_size=args.BATCH).train(
        data, args.RATE
    )
