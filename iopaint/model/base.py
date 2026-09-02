import abc
from typing import Optional

import cv2
import torch
import numpy as np
from loguru import logger

from iopaint.helper import (
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
    #: `boxes_from_mask` 는 마스크의 연결요소 하나당 박스 하나를 만들고
    #: 개수에 상한이 없었다. 박스마다 추론이 한 번 돌아 **요청 하나가 몇
    #: 시간 동안 코어를 물었다.** 그 동안 `ModelManager._model_lock` 이
    #: 잡혀 다른 모든 지우기가 멈추고, `/healthz` 는 async 라 200 을
    #: 유지하므로 쿠버네티스는 아무 조치도 하지 않는다.
    #:
    #: 16 인 이유: 손으로 지우는 대상은 보통 한두 개고, 얼굴 여러 개나
    #: 흩어진 잡티를 한 번에 지우는 경우를 생각해도 넉넉하다.
    MAX_CROP_BOXES = 16

    #: 연결요소를 분석할 최대 변 길이.
    #:
    #: **개수를 세는 일 자체가 공격 표면이었다.** 처음엔 개수를 확인해
    #: 넘치면 합치도록 고쳤는데, 그 확인에 닿기 전에
    #: `boxes_from_mask` 의 `cv2.findContours` 가 모든 윤곽을 만든다.
    #: 실측(6000² 체커보드, 1픽셀 성분 9백만 개):
    #:
    #:     findContours                41.6s   RSS +6.1GB
    #:     connectedComponentsWithStats 4.0s   RSS +12.5GB
    #:     이 방식(2048 로 축소)         0.02s   RSS +35MB
    #:
    #: 둘 다 16Gi 한계를 위협하고 그 동안 락을 쥔다. 그래서 개수를 세는
    #: 것도 묶는다 - 축소한 마스크에서만 분석하고 좌표를 되돌린다.
    #:
    #: 정밀도 손실은 무해하다. `_crop_box` 가 어차피 margin 128 을 더하고,
    #: 축소는 성분을 **합치는** 방향으로만 틀린다(INTER_AREA + >0 문턱).
    #: 합치는 것은 우리의 폴백이므로 안전한 쪽으로 틀린다.
    BOX_ANALYSIS_MAX_SIDE = 2048

    def _bounded_boxes(self, mask):
        """크롭 박스를 개수와 분석 비용 양쪽에서 묶는다."""
        import cv2
        import numpy as np

        height, width = mask.shape[:2]
        scale = min(1.0, self.BOX_ANALYSIS_MAX_SIDE / max(height, width))
        if scale < 1.0:
            small = cv2.resize(
                mask,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = mask

        # `> 0` 이다. `> 127` 로 하면 축소로 옅어진 얇은 마스크가 사라진다.
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            (small > 0).astype(np.uint8), connectivity=8
        )
        components = count - 1  # 0번은 배경
        if components <= 0:
            return []

        inverse = 1.0 / scale if scale < 1.0 else 1.0

        def to_full(x0, y0, x1, y1):
            """축소 좌표를 원본으로 되돌린다.

            **바깥으로 넓힌다.** 안쪽으로 깎으면 마스크의 끝이 크롭 밖에
            남아 그 부분이 지워지지 않는다 - 결과에 원본 조각이 남는다.
            """
            return np.array(
                [
                    max(0, int(np.floor(x0 * inverse))),
                    max(0, int(np.floor(y0 * inverse))),
                    min(width, int(np.ceil(x1 * inverse))),
                    min(height, int(np.ceil(y1 * inverse))),
                ]
            ).astype(int)

        boxes = stats[1:]  # x, y, w, h, area
        if components > self.MAX_CROP_BOXES:
            # 넘치면 **거절하지 않고 하나로 합친다.** 거절하면 "점을 많이
            # 찍으면 실패하는" 편집기가 되는데 사용자는 이유를 알 수 없다.
            # 어차피 그만큼 흩어진 마스크는 문맥이 뒤섞여 있다.
            logger.warning(
                f"크롭 박스 {components}개 → 하나로 합친다 "
                f"(상한 {self.MAX_CROP_BOXES})"
            )
            return [
                to_full(
                    boxes[:, 0].min(),
                    boxes[:, 1].min(),
                    (boxes[:, 0] + boxes[:, 2]).max(),
                    (boxes[:, 1] + boxes[:, 3]).max(),
                )
            ]

        return [
            to_full(b[0], b[1], b[0] + b[2], b[1] + b[3]) for b in boxes
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
