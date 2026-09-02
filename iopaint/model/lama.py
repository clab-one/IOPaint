import os

import cv2
import numpy as np
import torch

from iopaint.weights import ERASE_WEIGHTS
from iopaint.helper import (
    norm_img,
    get_cache_path_by_url,
    load_jit_model,
    download_model,
)
from iopaint.schema import InpaintRequest
from .base import InpaintModel

# 출처는 iopaint.weights 한 곳에만 둔다. 사본을 만들면 어긋나도 조용히 두 번
# 받을 뿐이라 드러나지 않는다. 환경변수 우회는 그대로 남긴다 - 다른 미러를
# 쓰는 경로가 upstream 부터 있었다.
LAMA_MODEL_URL = os.environ.get("LAMA_MODEL_URL", ERASE_WEIGHTS["lama"]["url"])
LAMA_MODEL_MD5 = os.environ.get("LAMA_MODEL_MD5", ERASE_WEIGHTS["lama"]["md5"])


class LaMa(InpaintModel):
    name = "lama"
    pad_mod = 8
    is_erase_model = True

    @staticmethod
    def download():
        download_model(LAMA_MODEL_URL, LAMA_MODEL_MD5)

    def init_model(self, device, **kwargs):
        self.model = load_jit_model(LAMA_MODEL_URL, device, LAMA_MODEL_MD5).eval()

    @staticmethod
    def is_downloaded() -> bool:
        return os.path.exists(get_cache_path_by_url(LAMA_MODEL_URL))

    def forward(self, image, mask, config: InpaintRequest):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W]
        return: BGR IMAGE
        """
        image = norm_img(image)
        mask = norm_img(mask)

        mask = (mask > 0) * 1
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        inpainted_image = self.model(image, mask)

        cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        return cur_res
