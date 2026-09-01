"""추론 분배 계약.

두 노드의 속도가 20배 차이 난다(i7m 2870ms vs M4 267ms @800²). 그래서
라운드로빈은 틀린 답이고, 남은 시간을 추정해 고른다. 동시에 M4 는 클러스터
밖이라 언제든 사라질 수 있으므로 **고르되 의존하지 않는다**.

여기서 고정하는 것:
  - 더 빨리 끝나는 쪽을 고른다. 빠른 노드가 밀리면 느린 쪽으로 넘어간다.
  - 원격이 죽으면 로컬이 처리한다. 요청은 실패하지 않는다.
  - ROI 왕복이 마스크 밖을 한 픽셀도 바꾸지 않는다.
"""

import numpy as np
import pytest

from iopaint.scheduler import (
    LocalBackend,
    Scheduler,
    Stats,
    composite_roi,
    square_roi,
)


class _Fake:
    def __init__(self, name, seed_ms, healthy=True, fail=False):
        self.name = name
        self.stats = Stats(seed_ms=seed_ms)
        self._healthy = healthy
        self._fail = fail
        self.calls = 0

    def healthy(self):
        return self._healthy

    def erase(self, image, mask):
        self.calls += 1
        if self._fail:
            raise RuntimeError("노드가 답하지 않는다")
        return np.full_like(image, 7)


def _sched(remote_seed=300.0, local_seed=2900.0, **kw):
    return Scheduler(_Fake("local", local_seed), _Fake("coreml", remote_seed, **kw))


# --- 고르기 -----------------------------------------------------------------


def test_prefers_the_faster_node():
    s = _sched()
    assert s.choose().name == "coreml"


def test_falls_back_when_the_fast_node_is_backed_up():
    """빠른 노드도 줄이 길면 느려진다.

    coreml 300ms 에 4건이 밀려 있으면 1500ms, local 2900ms 보다 빠르다.
    10건이면 3300ms 라 local 이 낫다. 라운드로빈이면 이 판단이 없다.
    """
    s = _sched()
    s.remote.stats.inflight = 4
    assert s.choose().name == "coreml"

    s.remote.stats.inflight = 10
    assert s.choose().name == "local"


def test_unhealthy_remote_is_not_chosen():
    s = _sched(healthy=False)
    assert s.choose().name == "local"


def test_no_remote_configured_uses_local():
    s = Scheduler(_Fake("local", 2900.0), None)
    assert s.choose().name == "local"
    assert s.stats() == {"local": {"p50_ms": 2900, "inflight": 0}}


def test_measured_latency_overrides_the_seed():
    """씨앗값은 관측이 없을 때만 쓴다. 노드가 느려지면 트래픽이 준다."""
    s = _sched()
    for _ in range(8):
        s.remote.stats.observe(5000.0)  # M4 가 실제로 느려졌다
    assert s.choose().name == "local"


# --- 폴백 -------------------------------------------------------------------


def test_remote_failure_falls_back_to_local():
    """편집 세션이 노드 하나 때문에 깨지면 안 된다 - 계획 16 절."""
    img = np.zeros((16, 16, 3), np.uint8)
    mask = np.zeros((16, 16), np.uint8)
    s = _sched(fail=True)

    out, used = s.erase(img, mask)

    assert used == "local"
    assert s.local.calls == 1
    assert (out == 7).all()


def test_failure_marks_remote_unhealthy():
    """한 번 실패하면 health 재확인 전까지 보내지 않는다."""
    img = np.zeros((8, 8, 3), np.uint8)
    s = _sched(fail=True)
    s.erase(img, np.zeros((8, 8), np.uint8))
    assert s.remote._healthy is False


def test_inflight_returns_to_zero_after_failure():
    """실패해도 카운터가 새면 안 된다 - 새면 그 노드는 영원히 '바쁨'이다."""
    s = _sched(fail=True)
    s.erase(np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8), np.uint8))
    assert s.remote.stats.inflight == 0
    assert s.local.stats.inflight == 0


# --- ROI --------------------------------------------------------------------


def test_roi_is_square_and_inside_the_image():
    mask = np.zeros((300, 400), np.uint8)
    mask[100:140, 200:210] = 255

    y0, y1, x0, x1 = square_roi(mask, (300, 400))

    assert y1 - y0 == x1 - x0, "정사각형이 아니다"
    assert 0 <= y0 and y1 <= 300 and 0 <= x0 and x1 <= 400
    # 마스크를 품어야 한다
    assert y0 <= 100 and y1 >= 140 and x0 <= 200 and x1 >= 210


def test_roi_gives_context_around_the_mask():
    """마스크에 딱 맞게 자르면 LaMa 가 채울 근거가 없다."""
    mask = np.zeros((512, 512), np.uint8)
    mask[200:240, 200:240] = 255

    y0, y1, _, _ = square_roi(mask, (512, 512))

    assert y1 - y0 > 40, "여백이 없다"


def test_roi_of_a_mask_touching_the_edge_stays_inside():
    mask = np.zeros((100, 100), np.uint8)
    mask[0:10, 0:10] = 255

    y0, y1, x0, x1 = square_roi(mask, (100, 100))

    assert y0 >= 0 and x0 >= 0 and y1 <= 100 and x1 <= 100


def test_empty_mask_has_no_roi():
    assert square_roi(np.zeros((10, 10), np.uint8), (10, 10)) is None


def test_composite_touches_only_the_masked_pixels():
    """ROI 왕복이 마스크 밖을 바꾸면 안 된다.

    800 으로 줄였다 늘리면 ROI 전체에 리샘플링 자국이 남는다. 마스크 안쪽만
    되돌려야 사진의 나머지가 원본 그대로다.
    """
    rng = np.random.default_rng(5)
    img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), np.uint8)
    mask[20:30, 20:30] = 255
    roi = square_roi(mask, (64, 64))
    filled = np.full((roi[1] - roi[0], roi[3] - roi[2], 3), 200, np.uint8)

    out = composite_roi(img, mask, roi, filled)

    assert (out[mask > 127] == 200).all(), "마스크 안이 안 채워졌다"
    assert (out[mask <= 127] == img[mask <= 127]).all(), "마스크 밖이 바뀌었다"


def test_local_backend_records_its_own_latency():
    calls = []

    class MM:
        def __call__(self, image, mask, req):
            calls.append(req)
            return np.zeros_like(image)

    b = LocalBackend(MM(), lambda: "REQ", seed_ms=1234.0)
    assert b.stats.p50 == 1234.0

    b.erase(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), np.uint8))

    assert calls == ["REQ"]
    assert b.stats.p50 < 1234.0, "실측이 씨앗값을 밀어내지 않았다"


def test_local_backend_returns_rgb_not_bgr():
    """경계 계약은 RGB 다.

    LaMa 는 BGR 을 돌려준다. 백엔드가 그대로 넘기면 호출자는 로컬에서만 맞는
    변환을 하게 되고, 원격(Core ML) 경로의 R 과 B 가 뒤집힌다. 회색조나 대칭
    색으로는 절대 안 잡히므로 **채널마다 다른 값**을 쓴다.
    """
    bgr = np.zeros((2, 2, 3), np.uint8)
    bgr[..., 0] = 10  # B
    bgr[..., 1] = 20  # G
    bgr[..., 2] = 30  # R

    class MM:
        def __call__(self, image, mask, req):
            return bgr

    out = LocalBackend(MM(), lambda: None).erase(
        np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2), np.uint8)
    )

    assert tuple(out[0, 0]) == (30, 20, 10), f"RGB 로 나오지 않았다: {out[0, 0]}"


def test_stats_reports_both_nodes():
    s = _sched()
    body = s.stats()
    assert body["local"]["p50_ms"] == 2900
    assert body["coreml"] == {"p50_ms": 300, "inflight": 0, "healthy": True}
