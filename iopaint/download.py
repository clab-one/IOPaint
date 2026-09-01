import os
from typing import List

from loguru import logger

from iopaint.const import DEFAULT_MODEL_DIR
from iopaint.schema import ModelType, ModelInfo


def cli_download_model(model: str):
    """지우기 모델 하나를 내려받는다.

    FOLIO fork: HuggingFace 의 diffusion 파이프라인을 받아오던 경로를 들어냈다.
    남은 모델은 GitHub release 의 torch.jit 아카이브 하나뿐이다.
    """
    from iopaint.model import models

    if model not in models:
        raise NotImplementedError(
            f"Unsupported model: {model}. Available models: {list(models.keys())}"
        )

    logger.info(f"Downloading {model}...")
    models[model].download()
    logger.info("Done.")


def scan_models() -> List[ModelInfo]:
    """내려받아 둔 지우기 모델을 훑는다."""
    from iopaint.model import models

    model_dir = os.getenv("XDG_CACHE_HOME", DEFAULT_MODEL_DIR)
    logger.debug(f"Scanning erase models in {model_dir}")

    return [
        ModelInfo(name=name, path=name, model_type=ModelType.INPAINT)
        for name, m in models.items()
        if m.is_erase_model and m.is_downloaded()
    ]
