import abc
from typing import Optional

import cv2
import torch
import numpy as np
from loguru import logger

from iopaint.helper import (
    boxes_from_mask,
    resize_max_size,
    pad_img_to_modulo,
    switch_mps_device,
)
from iopaint.schema import InpaintRequest, HDStrategy


class InpaintModel:
    name = "base"
    min_size: Optional[int] = None
    pad_mod = 8
    pad_to_square = False
    is_erase_model = False

    def __init__(self, device, **kwargs):
        """

        Args:
            device:
        """
        device = switch_mps_device(self.name, device)
        self.device = device
        self.init_model(device, **kwargs)

    @abc.abstractmethod
    def init_model(self, device, **kwargs): ...

    @staticmethod
    @abc.abstractmethod
    def is_downloaded() -> bool:
        return False

    @abc.abstractmethod
    def forward(self, image, mask, config: InpaintRequest):
        """Input images and output images have same size
        images: [H, W, C] RGB
        masks: [H, W, 1] 255 为 masks 区域
        return: BGR IMAGE
        """
        ...

    @staticmethod
    def download(): ...

    def _pad_forward(self, image, mask, config: InpaintRequest):
        origin_height, origin_width = image.shape[:2]
        pad_image = pad_img_to_modulo(
            image, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size
        )
        pad_mask = pad_img_to_modulo(
            mask, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size
        )

        # logger.info(f"final forward pad size: {pad_image.shape}")

        image, mask = self.forward_pre_process(image, mask, config)

        result = self.forward(pad_image, pad_mask, config)
        result = result[0:origin_height, 0:origin_width, :]

        result, image, mask = self.forward_post_process(result, image, mask, config)

        if config.keep_unmasked_area:
            mask = mask[:, :, np.newaxis]
            result = result * (mask / 255) + image[:, :, ::-1] * (1 - (mask / 255))
        return result

    def forward_pre_process(self, image, mask, config):
        return image, mask

    def forward_post_process(self, result, image, mask, config):
        return result, image, mask

    @torch.no_grad()
    def __call__(self, image, mask, config: InpaintRequest):
        """
        images: [H, W, C] RGB, not normalized
        masks: [H, W]
        return: BGR IMAGE
        """
        inpaint_result = None
        # logger.info(f"hd_strategy: {config.hd_strategy}")
        if config.hd_strategy == HDStrategy.CROP:
            if max(image.shape) > config.hd_strategy_crop_trigger_size:
                logger.info("Run crop strategy")
                boxes = self._bounded_boxes(mask)
                crop_result = []
                for box in boxes:
                    crop_image, crop_box = self._run_box(image, mask, box, config)
                    crop_result.append((crop_image, crop_box))

                inpaint_result = image[:, :, ::-1]
                for crop_image, crop_box in crop_result:
                    x1, y1, x2, y2 = crop_box
                    inpaint_result[y1:y2, x1:x2, :] = crop_image

        elif config.hd_strategy == HDStrategy.RESIZE:
            if max(image.shape) > config.hd_strategy_resize_limit:
                origin_size = image.shape[:2]
                downsize_image = resize_max_size(
                    image, size_limit=config.hd_strategy_resize_limit
                )
                downsize_mask = resize_max_size(
                    mask, size_limit=config.hd_strategy_resize_limit
                )

                logger.info(
                    f"Run resize strategy, origin size: {image.shape} forward size: {downsize_image.shape}"
                )
                inpaint_result = self._pad_forward(
                    downsize_image, downsize_mask, config
                )

                # only paste masked area result
                inpaint_result = cv2.resize(
                    inpaint_result,
                    (origin_size[1], origin_size[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
                original_pixel_indices = mask < 127
                inpaint_result[original_pixel_indices] = image[:, :, ::-1][
                    original_pixel_indices
                ]

        if inpaint_result is None:
            inpaint_result = self._pad_forward(image, mask, config)

        return inpaint_result

    #: 한 요청이 돌릴 수 있는 크롭 추론의 최대 개수.
    #:
    #: 왜 필요한가 - `boxes_from_mask` 는 마스크의 연결요소 하나당 박스
    #: 하나를 만들고 개수에 상한이 없었다. 4000x4000 사진에 10픽셀 간격
    #: 점 마스크(PNG 로 수백 KB)를 보내면 박스가 16만 개가 되고, 박스마다
    #: 추론이 한 번 돌아 **요청 하나가 몇 시간 동안 코어를 물었다.**
    #:
    #: 그 동안 `ModelManager._model_lock` 이 잡혀 다른 모든 지우기가 멈추고,
    #: 입장 제어 슬롯도 계속 점유된다. `/healthz` 는 async 라 200 을
    #: 유지하므로 쿠버네티스는 아무 조치도 하지 않는다. 클라이언트가 연결을
    #: 끊어도 스레드풀에서 도는 sync 핸들러는 취소되지 않는다.
    #:
    #: 16 인 이유: 사람이 손으로 지우는 대상은 보통 한두 개고, 얼굴 여러 개나
    #: 흩어진 잡티를 한 번에 지우는 경우를 생각해도 이 정도면 넉넉하다.
    MAX_CROP_BOXES = 16

    def _bounded_boxes(self, mask):
        """크롭 박스를 상한 안으로 줄인다.

        넘치면 거절하지 않고 **하나로 합친다.** 거절하면 "점을 많이 찍으면
        실패하는" 편집기가 되는데, 사용자는 왜 실패했는지 알 수 없다.
        전체를 감싸는 박스 하나로 돌리면 결과는 조금 거칠어도 나온다 -
        어차피 그만큼 흩어진 마스크는 문맥이 뒤섞여 있다.
        """
        boxes = boxes_from_mask(mask)
        if len(boxes) <= self.MAX_CROP_BOXES:
            return boxes

        logger.warning(
            f"크롭 박스 {len(boxes)}개 → 하나로 합친다 (상한 {self.MAX_CROP_BOXES})"
        )
        import numpy as np

        stacked = np.concatenate(boxes, axis=0).reshape(-1, 4)
        return [
            np.array(
                [
                    stacked[:, 0].min(),
                    stacked[:, 1].min(),
                    stacked[:, 2].max(),
                    stacked[:, 3].max(),
                ]
            )
        ]

    def _crop_box(self, image, mask, box, config: InpaintRequest):
        """

        Args:
            image: [H, W, C] RGB
            mask: [H, W, 1]
            box: [left,top,right,bottom]

        Returns:
            BGR IMAGE, (l, r, r, b)
        """
        box_h = box[3] - box[1]
        box_w = box[2] - box[0]
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        img_h, img_w = image.shape[:2]

        w = box_w + config.hd_strategy_crop_margin * 2
        h = box_h + config.hd_strategy_crop_margin * 2

        _l = cx - w // 2
        _r = cx + w // 2
        _t = cy - h // 2
        _b = cy + h // 2

        l = max(_l, 0)
        r = min(_r, img_w)
        t = max(_t, 0)
        b = min(_b, img_h)

        # try to get more context when crop around image edge
        if _l < 0:
            r += abs(_l)
        if _r > img_w:
            l -= _r - img_w
        if _t < 0:
            b += abs(_t)
        if _b > img_h:
            t -= _b - img_h

        l = max(l, 0)
        r = min(r, img_w)
        t = max(t, 0)
        b = min(b, img_h)

        crop_img = image[t:b, l:r, :]
        crop_mask = mask[t:b, l:r]

        # logger.info(f"box size: ({box_h},{box_w}) crop size: {crop_img.shape}")

        return crop_img, crop_mask, [l, t, r, b]

    def _run_box(self, image, mask, box, config: InpaintRequest):
        """

        Args:
            image: [H, W, C] RGB
            mask: [H, W, 1]
            box: [left,top,right,bottom]

        Returns:
            BGR IMAGE
        """
        crop_img, crop_mask, [l, t, r, b] = self._crop_box(image, mask, box, config)

        return self._pad_forward(crop_img, crop_mask, config), [l, t, r, b]
