"""요청 하나가 서비스를 세울 수 있는가.

FOLIO fork 가 추가한 시험이다. 독립 보안 검토에서 나온 critical 2건을 고정한다.

두 결함의 공통점은 **작은 요청이 큰 일을 시킨다**는 것이다. 인증을 붙여도
정상 사용자 한 명의 실수로 같은 일이 벌어진다 - 그래서 인증보다 먼저다.
"""

import io

import numpy as np
import pytest
from PIL import Image

from iopaint.helper import MAX_IMAGE_PIXELS, decode_base64_to_image, load_img
from iopaint.model.base import InpaintModel


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


# --- 압축 폭탄 ---------------------------------------------------------------


def test_oversized_image_is_refused_before_pixels_are_made():
    """상한을 넘는 이미지는 디코드 전에 거절된다.

    Pillow 기본값은 89.5M 픽셀에서 경고만 내고 약 179M 을 넘어야 예외다.
    즉 178M 픽셀짜리 PNG 가 통과한다 - 전부 같은 색이면 압축본은 수백 KB 고
    디코드하면 RGB 배열만 537MB 다.

    상한 바로 위 크기로 시험한다. 178M 픽셀 PNG 를 실제로 만들면 이 시험이
    그 자체로 무거워진다 - 검사하는 것은 "상한이 걸리는가" 지 Pillow 의
    한계가 아니다.
    """
    side = int(MAX_IMAGE_PIXELS**0.5) + 64
    oversized = _png(np.zeros((side, side), np.uint8))

    # 압축본은 작다. 그것이 이 공격의 요점이다.
    assert len(oversized) < 2_000_000, "표본이 너무 크다 - 시험이 공격을 대표하지 않는다"

    with pytest.raises(ValueError, match="픽셀은 상한"):
        load_img(oversized)


def test_oversized_base64_image_is_refused():
    """base64 경로도 같은 상한을 받는다.

    두 디코더가 있고 한쪽만 막으면 다른 쪽으로 들어온다.
    """
    side = int(MAX_IMAGE_PIXELS**0.5) + 64
    import base64

    encoded = "data:image/png;base64," + base64.b64encode(
        _png(np.zeros((side, side), np.uint8))
    ).decode()

    with pytest.raises(ValueError, match="픽셀은 상한"):
        decode_base64_to_image(encoded)


def test_normal_photo_still_decodes():
    """정상 사진은 그대로 통과한다.

    상한을 너무 낮게 잡으면 기능이 죽는다. 4000x3000(12MP)은 흔한 크기다.
    """
    image, _, _ = load_img(_png(np.zeros((3000, 4000, 3), np.uint8)), return_info=True)
    assert image.shape[:2] == (3000, 4000)


# --- 마스크 연결요소 증폭 ----------------------------------------------------


class _CountingModel(InpaintModel):
    """추론 횟수만 센다. 실제 모델을 적재하지 않는다."""

    name = "counting"
    pad_mod = 8
    min_size = None

    def __init__(self):  # upstream __init__ 은 가중치를 받는다
        self.calls = 0

    def forward(self, image, mask, config):
        self.calls += 1
        return image[:, :, ::-1]

    @staticmethod
    def is_downloaded() -> bool:
        return True


def _scattered_mask(side: int, step: int) -> np.ndarray:
    """서로 떨어진 점 마스크. 연결요소가 점 개수만큼 생긴다."""
    mask = np.zeros((side, side), np.uint8)
    mask[::step, ::step] = 255
    return mask


def test_scattered_mask_does_not_multiply_inference():
    """흩어진 마스크가 추론 횟수를 늘리지 못한다.

    `boxes_from_mask` 는 연결요소 하나당 박스 하나를 만들고 개수 상한이
    없었다. 박스마다 추론이 한 번 돌아 **요청 하나가 몇 시간 동안 코어를
    물었다.** 그 동안 ModelManager._model_lock 이 잡혀 다른 모든 지우기가
    멈추고, /healthz 는 async 라 200 을 유지하므로 쿠버네티스는 아무 조치도
    하지 않는다.
    """
    from iopaint.schema import HDStrategy, InpaintRequest

    side = 1200  # crop 트리거(800)를 넘겨야 이 경로로 온다
    image = np.zeros((side, side, 3), np.uint8)
    mask = _scattered_mask(side, step=40)

    # 이 마스크가 실제로 상한을 넘는 연결요소를 만드는지 먼저 확인한다.
    # 안 넘으면 이 시험은 아무것도 검사하지 않는다.
    from iopaint.helper import boxes_from_mask

    assert len(boxes_from_mask(mask)) > InpaintModel.MAX_CROP_BOXES, "표본이 상한을 넘지 않는다"

    model = _CountingModel()
    model(image, mask, InpaintRequest(hd_strategy=HDStrategy.CROP))

    assert model.calls <= InpaintModel.MAX_CROP_BOXES, (
        f"추론이 {model.calls}회 돌았다 - 상한 {InpaintModel.MAX_CROP_BOXES} 를 넘었다"
    )


def test_few_boxes_are_left_alone():
    """상한 아래면 박스별로 그대로 돈다.

    합치는 것은 넘칠 때의 처리다. 항상 합치면 여러 물체를 지울 때 결과가
    나빠진다 - 떨어진 두 물체 사이의 멀쩡한 픽셀까지 다시 그린다.
    """
    from iopaint.schema import HDStrategy, InpaintRequest

    side = 1200
    image = np.zeros((side, side, 3), np.uint8)
    mask = np.zeros((side, side), np.uint8)
    mask[100:200, 100:200] = 255
    mask[900:1000, 900:1000] = 255

    model = _CountingModel()
    model(image, mask, InpaintRequest(hd_strategy=HDStrategy.CROP))
    assert model.calls == 2, f"박스 2개인데 추론 {model.calls}회"
