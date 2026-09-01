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

    with pytest.raises(HTTPException) as e:
        api.api_run_plugin_gen_mask(req)

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


def test_plugin_swap_does_not_expose_a_half_swapped_plugin(monkeypatch):
    """플러그인 모델 교체 중에 도착한 요청이 반쯤 바뀐 상태를 보지 않는다.

    inpaint 는 ModelManager 안에서 막았지만 플러그인은 upstream 클래스라
    Api 가 막아야 한다. wave 3~5 에서 RealESRGAN·GFPGAN·SAM 이 붙으면
    실제로 부딪히는 경로다.
    """
    from iopaint.api import Api

    plugin = _SwappablePlugin()
    api = _api_with({"P": plugin})
    monkeypatch.setattr("iopaint.api.torch_gc", lambda: None)
    monkeypatch.setattr(
        "iopaint.api.decode_base64_to_image", lambda *a, **k: (None, None, {}, "png")
    )
    monkeypatch.setattr("iopaint.api.pil_to_bytes", lambda *a, **k: b"")
    monkeypatch.setattr("iopaint.api.concat_alpha_channel", lambda img, a: img)
    monkeypatch.setattr("iopaint.api.cv2.cvtColor", lambda img, code: img)
    api.config = type("C", (), {"quality": 95, "remove_bg_model": None})()

    t = threading.Thread(
        target=lambda: Api.api_switch_plugin_model(
            api, type("R", (), {"plugin_name": "P", "model_name": "new"})()
        )
    )
    t.start()
    time.sleep(0.05)  # 모델이 None 인 구간
    Api.api_run_plugin_gen_image(api, type("R", (), {"name": "P", "image": ""})())
    t.join()

    assert plugin.observed == ["new"], f"반쯤 바뀐 플러그인을 봤다: {plugin.observed}"
