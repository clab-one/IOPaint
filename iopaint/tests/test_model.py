import pytest
import torch

from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy
from iopaint.tests.utils import assert_equal, get_config, current_dir, check_device


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
@pytest.mark.parametrize(
    "strategy", [HDStrategy.ORIGINAL, HDStrategy.RESIZE, HDStrategy.CROP]
)
def test_lama(device, strategy):
    check_device(device)
    model = ModelManager(name="lama", device=device)
    assert_equal(
        model,
        get_config(strategy=strategy),
        f"lama_{strategy[0].upper() + strategy[1:]}_result.png",
    )

    fx = 1.3
    assert_equal(
        model,
        get_config(strategy=strategy),
        f"lama_{strategy[0].upper() + strategy[1:]}_fx_{fx}_result.png",
        fx=1.3,
    )
