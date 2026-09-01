from typing import List, Dict

import torch
from loguru import logger
import numpy as np

from iopaint.download import scan_models
from iopaint.helper import switch_mps_device
from iopaint.model import models
from iopaint.model.utils import torch_gc
from iopaint.schema import InpaintRequest, ModelInfo


class ModelManager:
    """지우기 모델 하나를 들고 있는다.

    FOLIO fork: controlnet/brushnet/powerpaint/lcm-lora 전환 경로를 전부
    들어냈다. 남은 모델이 하나뿐이라 `switch` 는 사실상 재적재이고,
    스케줄러·프롬프트·시드 같은 diffusion 상태는 존재하지 않는다.
    """

    def __init__(self, name: str, device: torch.device, **kwargs):
        self.name = name
        self.device = device
        self.kwargs = kwargs
        self.available_models: Dict[str, ModelInfo] = {}
        self.scan_models()
        self.model = self.init_model(name, device, **kwargs)

    @property
    def current_model(self) -> ModelInfo:
        return self.available_models[self.name]

    def init_model(self, name: str, device, **kwargs):
        logger.info(f"Loading model: {name}")
        if name not in self.available_models:
            raise NotImplementedError(
                f"Unsupported model: {name}. Available models: {list(self.available_models.keys())}"
            )
        if name not in models:
            raise NotImplementedError(f"Unsupported model: {name}")

        model_info = self.available_models[name]
        return models[name](device, **{**kwargs, "model_info": model_info})

    @torch.inference_mode()
    def __call__(self, image, mask, config: InpaintRequest):
        """
        Args:
            image: [H, W, C] RGB
            mask: [H, W, 1] 255 means area to repaint
            config:

        Returns:
            BGR image
        """
        return self.model(image, mask, config).astype(np.uint8)

    def scan_models(self) -> List[ModelInfo]:
        available_models = scan_models()
        self.available_models = {it.name: it for it in available_models}
        return available_models

    def switch(self, new_name: str):
        if new_name == self.name:
            return

        old_name = self.name
        self.name = new_name
        try:
            del self.model
            torch_gc()
            self.model = self.init_model(
                new_name, switch_mps_device(new_name, self.device), **self.kwargs
            )
        except Exception as e:
            self.name = old_name
            logger.info(f"Switch model from {old_name} to {new_name} failed, rollback")
            self.model = self.init_model(
                old_name, switch_mps_device(old_name, self.device), **self.kwargs
            )
            raise e
