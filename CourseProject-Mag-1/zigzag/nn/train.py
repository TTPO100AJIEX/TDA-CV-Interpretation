import numpy
import torch
import pandas
import cvtda.utils
import cvtda.logging
import torch.utils.data
import cvtda.neural_network
from cvtda.classification import estimate_quality


def train(
    model: torch.nn.Module,
    train_dl: torch.utils.data.DataLoader,
    test_dl: torch.utils.data.DataLoader,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    random_state: int = 42,
    device: torch.device = cvtda.neural_network.default_device,
) -> pandas.DataFrame:
    cvtda.utils.set_random_seed(random_state)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    history = []
    pbar = cvtda.logging.logger().pbar(range(epochs), desc="Training")
    for epoch in pbar:
        model.train()
        for X, y in train_dl:
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(X.to(device)), y.to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        test_preds, test_targets = [], []
        for X, y in test_dl:
            test_targets.extend(y.tolist())
            with torch.no_grad():
                test_preds.extend(model(X.to(device)).cpu().tolist())
        metrics = estimate_quality(numpy.array(test_preds), numpy.array(test_targets))
        cvtda.logging.logger().set_pbar_postfix(pbar, metrics)
        history.append(metrics)
    return pandas.DataFrame(history)
