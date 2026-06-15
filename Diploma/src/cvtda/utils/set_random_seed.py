import os
import random

import numpy
import torch


def set_random_seed(seed: int = 42):
    """
    Seed the random numbers generators in `random`, `numpy`, and `torch`.

    Parameters
    ----------
    seed : ``int``
        The seed to set. Defaults to `42`.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
