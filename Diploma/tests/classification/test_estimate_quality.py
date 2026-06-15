import math

import numpy
import torch
import pytest

import cvtda.utils
import cvtda.classification


@pytest.mark.parametrize(["num_classes"], [pytest.param(2), pytest.param(10), pytest.param(100)])
def test_estimate_quality(num_classes):
    cvtda.utils.set_random_seed(42)
    metrics = cvtda.classification.estimate_quality(
        torch.nn.functional.softmax(torch.rand((50, num_classes)), dim=1).numpy(),
        numpy.random.randint(0, num_classes, size=50),
    )
    for key, value in metrics.items():
        assert not math.isnan(value)
        assert isinstance(value, float)
        assert value >= 0 and value <= 1
