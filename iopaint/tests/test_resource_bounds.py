"""요청 하나가 서비스를 세울 수 있는가.

FOLIO fork 가 추가한 시험이다. 독립 보안 검토에서 나온 critical 2건을 고정한다.

두 결함의 공통점은 **작은 요청이 큰 일을 시킨다**는 것이다. 인증을 붙여도
정상 사용자 한 명의 실수로 같은 일이 벌어진다 - 그래서 인증보다 먼저다.
"""

import io
import pathlib
import tempfile

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
    model = _CountingModel()
    assert len(model._bounded_boxes(mask)) == 1, "표본이 합치기 경로로 가지 않는다"

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


def test_component_analysis_cost_is_bounded():
    """성분을 세는 일 자체가 묶여 있는가.

    처음 고칠 때는 개수만 확인했다. 그런데 그 확인에 닿기 전에
    `cv2.findContours` 가 모든 윤곽을 만든다 - 6000² 체커보드(1픽셀 성분
    9백만 개)에서 41.6초와 6.1GB 였다. 그 동안 _model_lock 을 쥐고 있으므로
    다른 모든 지우기가 멈춘다.

    지금은 축소한 마스크에서만 분석한다. 이 시험은 그 비용이 실제로 묶여
    있는지 시간과 메모리로 본다.
    """
    import resource
    import time

    def rss_mb() -> float:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)

    side = 6000
    mask = np.zeros((side, side), np.uint8)
    mask[::2, ::2] = 255  # 성분 9백만 개

    model = _CountingModel()
    before = rss_mb()
    started = time.monotonic()
    boxes = model._bounded_boxes(mask)
    elapsed = time.monotonic() - started
    grew = rss_mb() - before

    assert len(boxes) >= 1
    # 넉넉한 상한이다. findContours 경로는 41.6초·6.1GB 였다.
    assert elapsed < 5.0, f"분석에 {elapsed:.1f}초 걸렸다 - 묶이지 않았다"
    assert grew < 1500, f"RSS 가 {grew:.0f}MiB 늘었다 - 묶이지 않았다"


def test_merged_box_covers_the_whole_mask():
    """합친 박스가 마스크를 전부 덮는가.

    축소 좌표를 되돌릴 때 안쪽으로 깎으면 마스크의 끝이 크롭 밖에 남아
    **그 부분이 지워지지 않는다** - 결과에 원본 조각이 남고, 사용자는
    "덜 지워졌다" 고 본다. 바깥으로 넓혀야 한다.
    """
    side = 4000
    mask = np.zeros((side, side), np.uint8)
    # 모서리에 점을 흩어 상한을 넘기고, 극단 좌표를 포함시킨다.
    mask[7:9, 11:13] = 255
    mask[side - 9 : side - 7, side - 13 : side - 11] = 255
    mask[::37, ::37] = 255

    model = _CountingModel()
    boxes = model._bounded_boxes(mask)
    assert len(boxes) == 1, "합치기 경로가 아니다"

    x0, y0, x1, y1 = boxes[0]
    ys, xs = np.nonzero(mask > 0)
    assert x0 <= xs.min() and x1 >= xs.max() + 1, f"x 범위가 마스크를 덮지 않는다 ({x0},{x1})"
    assert y0 <= ys.min() and y1 >= ys.max() + 1, f"y 범위가 마스크를 덮지 않는다 ({y0},{y1})"


def test_session_create_rejects_pixel_bomb_before_taking_a_slot():
    """압축 폭탄으로 세션 슬롯을 고갈시킬 수 있는가.

    바이트 상한만 보면 막히지 않는다 - 6400² 단색 PNG 는 20KB 도 안 된다.
    erase 에서 413 을 주는 것으로는 늦다: 그때는 이미 슬롯을 잡았고, 폭탄
    32개면 TTL 20분 동안 정상 사용자가 세션을 못 만든다. 인증이 없으므로
    누구나 할 수 있다.

    그래서 **슬롯을 잡기 전에** 헤더만 읽어 거절해야 한다.
    """
    from iopaint.session import SessionError, SessionStore

    side = 6400  # 40.96M 픽셀 > 상한 40M
    buf = io.BytesIO()
    Image.new("RGB", (side, side), (7, 7, 7)).save(buf, "PNG")
    bomb = buf.getvalue()
    assert len(bomb) < 1_000_000, f"폭탄이 커서 바이트 상한에 걸린다 ({len(bomb)})"

    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(root=pathlib.Path(tmp), max_sessions=4, ttl=600)
        for _ in range(6):  # 슬롯 수보다 많이 시도한다
            with pytest.raises(SessionError) as excinfo:
                store.create(bomb, "png")
            assert excinfo.value.status == 413, f"413 이 아니라 {excinfo.value.status}"
        assert store.stats()["sessions"] == 0, "거절했는데 슬롯을 먹었다"

        # 정상 이미지는 그대로 통과해야 한다.
        ok = io.BytesIO()
        Image.new("RGB", (64, 64), (1, 2, 3)).save(ok, "PNG")
        s = store.create(ok.getvalue(), "png")
        assert s.alive
        assert store.stats()["sessions"] == 1


def test_session_create_rejects_garbage_without_taking_a_slot():
    """이미지가 아닌 것도 슬롯을 먹지 못한다."""
    from iopaint.session import SessionError, SessionStore

    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(root=pathlib.Path(tmp), max_sessions=2, ttl=600)
        with pytest.raises(SessionError) as excinfo:
            store.create(b"not an image at all", "png")
        assert excinfo.value.status == 400, f"400 이 아니라 {excinfo.value.status}"
        assert store.stats()["sessions"] == 0
