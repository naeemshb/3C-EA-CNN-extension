import os
import sys

import pytest
import torch

FINAL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if FINAL_DIR not in sys.path:
    sys.path.insert(0, FINAL_DIR)

torch.set_num_threads(2)


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def final_dir():
    return FINAL_DIR
