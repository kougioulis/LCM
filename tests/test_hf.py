"""
Quick test for loading a pretrained LCM from the Hugging Face Hub.
Downloads nkougioulis/LCM-2.5M (~10 MB); skipped if the Hub is unreachable.
"""

import pytest
import torch

from lcm_pytorch.modules.lcm_module import LCMModule
from lcm_pytorch.utils.utils import lagged_batch_crosscorrelation


def test_from_pretrained_hub():
    try:
        model = LCMModule.from_pretrained("nkougioulis/LCM-2.5M").eval()
    except OSError as e:  # network errors from huggingface_hub
        pytest.skip(f"Hugging Face Hub unreachable: {e}")

    hp = model.hparams # max sequence length, num of variables
    x = torch.randn(1, hp.max_seq_len, hp.n_vars)
    with torch.no_grad():
        out = model((x, lagged_batch_crosscorrelation(x, hp.max_lag)))

    assert out.shape == (1, hp.n_vars, hp.n_vars, hp.max_lag)
