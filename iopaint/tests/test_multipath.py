"""여러 경로에서 동시에 들어올 때의 계약.

경로가 하나일 때는 입장 제어만으로 충분했다. 경로가 늘면 두 가지가 깨진다.

1. 입장 제어를 검증보다 먼저 잡으면 무효 요청이 자리를 먹는다.
   실측(clab-cluster, capacity 9): inpaint 6 + 없는 플러그인 6 을 동시에
   보냈더니 무효 요청 4건이 슬롯을 차지하고 진짜 지우기 1건이 503 을 맞았다.

2. 모델 교체는 입장 제어 밖에 있는데 `del self.model` 로 시작한다.
   추론과 겹치면 진행 중인 요청이 사라진 객체를 잡는다.

두 경우 모두 요청이 실패하는 것 자체가 아니라, **다른 요청을 망가뜨리는 것**이
문제다. 여기서 고정하는 것이 그 경계다.
"""

import threading
import time

import pytest
from fastapi import HTTPException

from iopaint.admission import Admission


class _Recorder:
    """입장 제어가 실제로 잡혔는지 센다."""

    def __init__(self):
        self.entered = 0
        self._real = Admission(inflight=1, max_queue=0)

    def admit(self):
        self.entered += 1
        return self._real.admit()


class _FakePlugin:
    support_gen_image = False
    support_gen_mask = False


def _api_with(plugins):
    """라우팅·모델 없이 검증 경계만 시험한다."""
    from iopaint.api import Api

    api = Api.__new__(Api)
    api.plugins = plugins
    api.admission = _Recorder()
    api._plugin_lock = threading.Lock()
    return api


def test_unknown_plugin_is_rejected_before_taking_a_slot():
    api = _api_with({})
    req = type("R", (), {"name": "RealESRGAN"})()

    with pytest.raises(HTTPException) as e:
        api.api_run_plugin_gen_image(req)

    assert e.value.status_code == 422
    assert api.admission.entered == 0, "무효 요청이 자리를 먹었다"


def test_unsupported_capability_is_rejected_before_taking_a_slot():
    api = _api_with({"RemoveBG": _FakePlugin()})
    req = type("R", (), {"name": "RemoveBG"})()

    # gen_image 로 검사한다. gen_mask 라우트는 지웠다 - 앱은 Vision
    # 온디바이스 마스크를 쓴다. 불변식("문 앞에서 거절, 슬롯 점유 0")은
    # 남은 라우트에서도 같아야 한다.
    with pytest.raises(HTTPException) as e:
        api.api_run_plugin_gen_image(req)

    assert e.value.status_code == 422
    assert api.admission.entered == 0


# --- 모델 교체 경쟁 ---------------------------------------------------------


class _SlowModel:
    def __init__(self, tag):
        self.tag = tag

    def __call__(self, image, mask, config):
        time.sleep(0.3)
        import numpy as np

        return np.full((2, 2, 3), self.tag, dtype=np.uint8)


def _manager_with(tag):
    from iopaint.model_manager import ModelManager

    m = ModelManager.__new__(ModelManager)
    m.name = "a"
    m.device = "cpu"
    m.kwargs = {}
    m.available_models = {}
    m._model_lock = threading.Lock()
    m.model = _SlowModel(tag)
    return m


def test_inference_arriving_mid_switch_does_not_see_a_missing_model(monkeypatch):
    """교체 도중에 도착한 추론이 사라진 모델을 잡지 않는다.

    위험 구간은 `del self.model` 과 새 모델 적재 완료 사이다. 적재는 느리므로
    (LaMa 는 초 단위) 그 사이에 도착한 요청은 실제로 존재하지 않는 속성을
    읽는다. 진행 중인 추론은 이미 객체 참조를 들고 있어 영향이 없다 - 무너지는
    쪽은 뒤에 오는 요청이다.
    """
    m = _manager_with(7)
    monkeypatch.setattr("iopaint.model_manager.switch_mps_device", lambda n, d: d)
    monkeypatch.setattr("iopaint.model_manager.torch_gc", lambda: None)

    def slow_load(name, device, **kw):
        time.sleep(0.3)  # 실제 적재가 느린 구간을 재현한다
        return _SlowModel(9)

    monkeypatch.setattr(m, "init_model", slow_load)

    def switcher():
        m.switch("b")

    t = threading.Thread(target=switcher)
    t.start()
    time.sleep(0.05)  # 모델이 지워진 뒤, 새 모델이 오기 전
    img = m(None, None, None)  # 락이 없으면 AttributeError
    t.join()

    assert img[0, 0, 0] == 9, "교체 완료 전의 모델로 답했다"
    assert m.name == "b"


# --- 플러그인 교체 경쟁 -----------------------------------------------------


class _SwappablePlugin:
    """`switch_model` 이 내부 모델을 갈아치우는 upstream 플러그인의 모양."""

    support_gen_image = True
    support_gen_mask = True

    def __init__(self):
        self.model = "old"
        self.observed = []

    def switch_model(self, name):
        self.model = None  # 교체 구간 - 여기서 gen_image 가 들어오면 안 된다
        time.sleep(0.3)
        self.model = name

    def gen_image(self, img, req):
        self.observed.append(self.model)
        import numpy as np

        return np.zeros((2, 2, 3), dtype=np.uint8)


# `/api/v1/switch_plugin_model` 라우트를 지웠으므로 그 경쟁 시험도 지웠다.
#
# `_plugin_lock` 은 남긴다. 이제 그것을 잡는 곳은 gen_image 하나뿐이지만,
# 플러그인의 CPU 작업을 직렬화하는 역할이 그대로 남아 있다 - 여러 요청이
# 동시에 RealESRGAN·GFPGAN 을 돌리면 코어를 서로 뺏는다.
#
# 런타임 모델 교체를 다시 열 거면 이 시험도 되살릴 것. git 이력에 있다.

class _OverlapWatch:
    """동시에 몇 개가 몸통에 들어와 있었는지 최대값을 센다."""

    def __init__(self, delay=0.02):
        self.delay = delay
        self.peak = 0
        self._cur = 0
        self._guard = threading.Lock()

    def work(self):
        with self._guard:
            self._cur += 1
            self.peak = max(self.peak, self._cur)
        time.sleep(self.delay)
        with self._guard:
            self._cur -= 1


def _hammer(fn, n=6):
    ts = [threading.Thread(target=fn) for _ in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]



def test_model_calls_never_overlap_whatever_inflight_says():
    """`ModelManager.__call__` 은 겹치지 않는다.

    세션 경로와 무상태 /api/v1/inpaint 가 **둘 다** 여기로 모인다. 여기가
    겹치면 OMP_NUM_THREADS=8 짜리 추론이 여러 벌 돌아 서로 스로틀만 건다.
    """
    import numpy as np

    w = _OverlapWatch()
    m = _manager_with(1)

    class _Watched:
        def __call__(self, image, mask, config):
            w.work()
            return np.zeros((2, 2, 3), dtype=np.uint8)

    m.model = _Watched()
    _hammer(lambda: m(None, None, None))

    assert w.peak == 1, (
        f"로컬 추론이 {w.peak} 벌 겹쳤다. --inflight 가 1 이 아니게 된 뒤로는 "
        "_model_lock 만이 CPU 를 지킨다 - 좁히면 과부하 사고가 재발한다."
    )


def test_plugin_runs_never_overlap_whatever_inflight_says():
    """플러그인 실행도 겹치지 않는다.

    wave 2 의 SAM 이 바로 이 경로다. 지금 배포는 plugins:[] 라 비어 있어
    아무 일도 없지만, 모델을 붙이는 순간 CPU 를 쓰는 두 번째 경로가 된다.
    """
    import base64
    import io

    import numpy as np
    from PIL import Image

    w = _OverlapWatch()

    class _Heavy:
        support_gen_image = True
        support_gen_mask = True

        def gen_image(self, img, req):
            w.work()
            return np.zeros((2, 2, 3), dtype=np.uint8)

        gen_mask = gen_image

    api = _api_with({"heavy": _Heavy()})
    # 입장 제어는 원래대로 5 를 허용한다 - 막는 것이 락뿐임을 보이려는
    # 시험이므로 여기서 직렬화되면 아무것도 검증하지 못한다.
    api.admission = Admission(inflight=5, max_queue=8)
    api.config = type("C", (), {"quality": 95})()

    buf = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(buf, "PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    req = type("R", (), {"name": "heavy", "image": b64})()

    # **진짜 핸들러를 부른다.** 테스트가 락을 직접 잡으면 핸들러에서 락이
    # 사라져도 통과한다 - 아무것도 지키지 못하는 검사가 된다.
    _hammer(lambda: api.api_run_plugin_gen_image(req))

    assert w.peak == 1, (
        f"플러그인이 {w.peak} 벌 겹쳤다. _plugin_lock 이 CPU 게이트를 겸한다 - "
        "wave 2 에서 SAM 을 붙이면 이것이 유일한 보호다."
    )
