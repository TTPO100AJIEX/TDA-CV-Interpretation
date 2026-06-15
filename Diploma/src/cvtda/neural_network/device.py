"""
Defines 'default_device' for operations with :mod:`torch`.
- If cuda is available, uses cuda.
- Else, if mps is available, uses mps.
- Otherwise, falls back to cpu.
"""

import torch


if torch.cuda.is_available():
    default_device = torch.device("cuda")
elif torch.mps.is_available():
    default_device = torch.device("mps")
else:
    default_device = torch.device("cpu")
