import hashlib
from typing import List

import numpy as np
import torch
from loguru import logger

from iopaint.helper import download_model
from iopaint.plugins.base_plugin import BasePlugin
# 표는 iopaint.weights 한 곳에만 둔다 - 사본을 만들면 어긋나도
# 조용히 두 번 받을 뿐이라 드러나지 않는다.
from iopaint.weights import SEGMENT_ANYTHING_MODELS
from iopaint.plugins.segment_anything import SamPredictor, sam_model_registry
from iopaint.plugins.segment_anything.predictor_hq import SamHQPredictor
from iopaint.plugins.segment_anything2.build_sam import build_sam2
from iopaint.plugins.segment_anything2.sam2_image_predictor import SAM2ImagePredictor
from iopaint.schema import RunPluginRequest

# 从小到大


class InteractiveSeg(BasePlugin):
    name = "InteractiveSeg"
    support_gen_mask = True

    def __init__(self, model_name, device):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self._init_session(model_name)

    def _init_session(self, model_name: str):
        model_path = download_model(
            SEGMENT_ANYTHING_MODELS[model_name]["url"],
            SEGMENT_ANYTHING_MODELS[model_name]["md5"],
        )
        logger.info(f"SegmentAnything model path: {model_path}")
        if "sam_hq" in model_name:
            self.predictor = SamHQPredictor(
                sam_model_registry[model_name](checkpoint=model_path).to(self.device)
            )
        elif model_name.startswith("sam2"):
            sam2_model = build_sam2(
                model_name, ckpt_path=model_path, device=self.device
            )
            self.predictor = SAM2ImagePredictor(sam2_model)
        else:
            self.predictor = SamPredictor(
                sam_model_registry[model_name](checkpoint=model_path).to(self.device)
            )
        self.prev_img_md5 = None

    def switch_model(self, new_model_name):
        if self.model_name == new_model_name:
            return

        logger.info(
            f"Switching InteractiveSeg model from {self.model_name} to {new_model_name}"
        )
        self._init_session(new_model_name)
        self.model_name = new_model_name

    def gen_mask(self, rgb_np_img, req: RunPluginRequest) -> np.ndarray:
        img_md5 = hashlib.md5(req.image.encode("utf-8")).hexdigest()
        return self.forward(rgb_np_img, req.clicks, img_md5)

    @torch.inference_mode()
    def forward(self, rgb_np_img, clicks: List[List], img_md5: str):
        input_point = []
        input_label = []
        for click in clicks:
            x = click[0]
            y = click[1]
            input_point.append([x, y])
            input_label.append(click[2])

        if img_md5 and img_md5 != self.prev_img_md5:
            self.prev_img_md5 = img_md5
            self.predictor.set_image(rgb_np_img)

        masks, _, _ = self.predictor.predict(
            point_coords=np.array(input_point),
            point_labels=np.array(input_label),
            multimask_output=False,
        )
        mask = masks[0].astype(np.uint8) * 255
        return mask
