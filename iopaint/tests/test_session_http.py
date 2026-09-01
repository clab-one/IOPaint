"""세션 API 의 HTTP 계약.

저장소 단위 시험은 락과 TTL 을 보지만, 실제로 쓰이는 모양 - 업로드하고,
마스크를 여러 번 보내고, 결과를 받고, 지우는 - 은 라우팅과 미들웨어를 지나야
한다. 여기서는 모델을 가짜로 바꿔 그 경로만 본다.

특히 고정하는 것:
  - 두 번째 지우기가 첫 번째 **결과** 위에서 일어난다.
  - 동시에 넣어도 편집이 유실되지 않는다.
  - 지우는 도중에 삭제가 와도 500 이 나지 않는다.
"""

import io
import threading
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image


def _png(color, size=(32, 32)) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, "PNG")
    return b.getvalue()


def _mask(size=(32, 32)) -> bytes:
    m = np.zeros((size[1], size[0]), np.uint8)
    m[8:16, 8:16] = 255
    b = io.BytesIO()
    Image.fromarray(m).save(b, "PNG")
    return b.getvalue()


class _CountingModel:
    """지울 때마다 밝기를 한 단계 올린다 - 결과가 누적되는지 보려고."""

    def __init__(self):
        self.calls = 0
        self.delay = 0.0
        self._guard = threading.Lock()

    def __call__(self, image, mask, config):
        with self._guard:
            self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        # LaMa 와 같은 규약으로 답한다: **BGR**. 입력은 RGB 로 들어오므로
        # 뒤집어서 돌려줘야 실제 모델과 같은 모양이 된다. 이걸 대칭 색으로
        # 두면 R/B 뒤집힘 버그가 테스트를 그대로 통과한다.
        rgb = np.clip(image.astype(np.int32) + 10, 0, 255).astype(np.uint8)
        return rgb[..., ::-1].copy()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FOLIO_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("FOLIO_MAX_SESSIONS", "4")

    from iopaint.api import Api
    from iopaint.schema import ApiConfig

    model = _CountingModel()
    monkeypatch.setattr(Api, "_build_plugins", lambda self: {})
    monkeypatch.setattr(Api, "_build_model_manager", lambda self: model)

    app = FastAPI()
    cfg = ApiConfig.model_construct(
        inflight=1, max_queue=8, quality=95, output_dir=None
    )
    api = Api(app, cfg)
    try:
        yield TestClient(app, raise_server_exceptions=False), model, api
    finally:
        api._reaper.stop()
        api.sessions.close()


def _create(c, image=None):
    r = c.post(
        "/v1/edit-sessions",
        files={"file": ("p.png", image or _png((10, 10, 10)), "image/png")},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _erase(c, sid, size=(32, 32)):
    return c.post(
        f"/v1/edit-sessions/{sid}/erase",
        files={"mask": ("m.png", _mask(size), "image/png")},
    )


def _pixel(c, sid):
    r = c.get(f"/v1/edit-sessions/{sid}/result")
    assert r.status_code == 200, r.text
    return tuple(np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))[0, 0])


def test_create_returns_an_id_and_ttl(client):
    c, _, _ = client
    r = c.post(
        "/v1/edit-sessions", files={"file": ("p.png", _png((0, 0, 0)), "image/png")}
    )
    assert r.status_code == 201
    body = r.json()
    assert body["id"] and body["ttl_seconds"] > 0


def test_erase_only_needs_the_mask(client):
    """원본을 다시 올리지 않는다 - 그게 세션의 요점이다."""
    c, model, _ = client
    sid = _create(c)

    r = _erase(c, sid)

    assert r.status_code == 200, r.text
    assert r.json()["ops"] == 1
    assert model.calls == 1


def test_operations_accumulate_on_the_working_image(client):
    """두 번째 지우기는 첫 번째 결과 위에서 일어난다.

    가짜 모델이 호출마다 +10 을 하므로 두 번이면 +20 이다. 원본 위에서
    다시 계산했다면 +10 에 머문다.
    """
    c, _, _ = client
    sid = _create(c, _png((10, 10, 10)))

    assert _erase(c, sid).status_code == 200
    assert _erase(c, sid).status_code == 200

    assert _pixel(c, sid) == (30, 30, 30)


def test_concurrent_erases_on_one_session_all_land(client):
    """같은 세션에 동시에 넣어도 하나도 잃지 않는다.

    직렬화가 없으면 넷이 같은 working 을 읽고 각자 써서 +10 에 그친다.
    """
    c, _, _ = client
    sid = _create(c, _png((0, 0, 0)))

    codes = []

    def erase():
        codes.append(_erase(c, sid).status_code)

    threads = [threading.Thread(target=erase) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert codes == [200] * 4, codes
    assert _pixel(c, sid) == (40, 40, 40)


def test_delete_during_erase_does_not_break_either(client):
    """지우는 도중 삭제가 와도 둘 다 정상적으로 끝난다.

    삭제가 세션 락을 잡지 않으면, 진행 중인 지우기가 사라진 디렉터리에 쓰면서
    500 을 낸다. 삭제는 진행 중인 연산이 끝나기를 기다린 뒤에 지운다.
    """
    c, model, api = client
    sid = _create(c, _png((0, 0, 0)))
    model.delay = 0.4

    erase_result = {}

    def erase():
        r = _erase(c, sid)
        erase_result["code"] = r.status_code

    t = threading.Thread(target=erase)
    t.start()
    time.sleep(0.15)  # 추론이 락을 잡은 뒤

    r = c.delete(f"/v1/edit-sessions/{sid}")
    t.join()

    assert r.status_code == 204
    assert erase_result["code"] == 200, "진행 중이던 지우기가 깨졌다"
    assert c.get(f"/v1/edit-sessions/{sid}/result").status_code == 404
    assert len(api.sessions) == 0


def test_erase_waiting_behind_a_delete_gets_404(client):
    """락을 기다리는 사이 세션이 사라지면 404 다 - 500 이 아니다."""
    c, model, api = client
    sid = _create(c, _png((0, 0, 0)))
    model.delay = 0.3

    codes = []

    def erase():
        codes.append(_erase(c, sid).status_code)

    first = threading.Thread(target=erase)
    first.start()
    time.sleep(0.1)
    second = threading.Thread(target=erase)  # 락 뒤에 줄을 선다
    second.start()
    time.sleep(0.05)

    c.delete(f"/v1/edit-sessions/{sid}")
    first.join()
    second.join()

    assert 500 not in codes, codes
    assert sorted(codes) == [200, 404], codes


def test_result_of_unknown_session_is_404(client):
    c, _, _ = client
    assert c.get("/v1/edit-sessions/zzz/result").status_code == 404


def test_delete_makes_the_session_gone(client):
    c, _, _ = client
    sid = _create(c)
    assert c.delete(f"/v1/edit-sessions/{sid}").status_code == 204
    assert c.get(f"/v1/edit-sessions/{sid}/result").status_code == 404
    assert _erase(c, sid).status_code == 404
    assert c.delete(f"/v1/edit-sessions/{sid}").status_code == 404


def test_mask_size_mismatch_is_400(client):
    c, _, _ = client
    sid = _create(c, _png((0, 0, 0), size=(32, 32)))
    assert _erase(c, sid, size=(16, 16)).status_code == 400


def test_session_limit_is_enforced_over_http(client):
    c, _, _ = client
    for _ in range(4):
        _create(c)
    r = c.post(
        "/v1/edit-sessions", files={"file": ("p.png", _png((0, 0, 0)), "image/png")}
    )
    assert r.status_code == 503


def test_colors_survive_the_round_trip(client):
    """채널이 뒤집히지 않는다.

    가짜 모델은 실제 LaMa 처럼 BGR 로 답한다. 스케줄러 경계에서 RGB 로
    정규화되고 API 가 다시 변환하지 않아야 원래 색이 나온다. 한때 백엔드마다
    규약이 달라 M4 경로만 R/B 가 뒤집혔었다 - 대칭 색으로는 안 잡힌다.
    """
    c, _, _ = client
    sid = _create(c, _png((90, 40, 10)))  # R≠G≠B

    assert _erase(c, sid).status_code == 200

    assert _pixel(c, sid) == (100, 50, 20), "채널이 뒤집혔다"


def test_healthz_reports_session_state(client):
    """세션 수와 reaper 생사를 밖에서 볼 수 있어야 한다.

    tmpfs 는 노드 RAM 이다. 몇 개가 살아 있는지 모르면 사진이 쌓이는 것을
    아무도 모른다. reaper 는 데몬 스레드라 조용히 죽고, Python 3.12 는 스레드
    이름을 OS 에 붙이지 않아 /proc 으로도 확인되지 않는다 - 프로세스가 스스로
    답하는 수밖에 없다.
    """
    c, _, api = client

    body = c.get("/healthz").json()["sessions"]
    assert body == {
        "sessions": 0,
        "max_sessions": 4,
        "ttl_seconds": int(api.sessions.ttl),
        "reaper": True,
    }

    sid = _create(c)
    assert c.get("/healthz").json()["sessions"]["sessions"] == 1

    c.delete(f"/v1/edit-sessions/{sid}")
    assert c.get("/healthz").json()["sessions"]["sessions"] == 0

    api._reaper.stop()
    assert c.get("/healthz").json()["sessions"]["reaper"] is False


def test_sessions_from_different_clients_do_not_interfere(client):
    """서로 다른 세션은 독립이다 - 한쪽 편집이 다른 쪽에 보이지 않는다."""
    c, _, _ = client
    a = _create(c, _png((0, 0, 0)))
    b = _create(c, _png((100, 100, 100)))

    _erase(c, a)
    _erase(c, a)
    _erase(c, b)

    assert _pixel(c, a) == (20, 20, 20)
    assert _pixel(c, b) == (110, 110, 110)
