"""입장 제어의 HTTP 표면 계약.

`test_admission.py` 는 Admission 객체를 본다. 그것만으로는 부족했다 -
upstream 의 예외 핸들러가 헤더를 버려서 503 에 Retry-After 가 붙지 않는 것을
객체 테스트는 통과시켰고 실제 요청만 잡아냈다.

그래서 여기서는 **프로덕션 `api_middleware` 를 그대로 설치한다.** 핸들러 코드를
테스트에 복사하면 그 복사본만 검증하게 되고, 실제 핸들러가 다시 헤더를 버려도
초록으로 남는다 - 회귀를 못 잡는 테스트다.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from iopaint.admission import Admission
from iopaint.api import api_middleware


def _app() -> FastAPI:
    app = FastAPI()
    # 프로덕션과 같은 예외 처리·CORS 를 붙인다.
    api_middleware(app)

    admission = Admission(inflight=1, max_queue=0)

    @app.get("/free")
    def free():
        with admission.admit():
            return {"ok": True}

    @app.get("/busy")
    def busy():
        # 자리를 먼저 채운 뒤 한 번 더 들어가려 한다 - 반드시 거절이다.
        with admission.admit():
            with admission.admit():
                return {"ok": True}

    return app


def test_busy_response_carries_retry_after():
    client = TestClient(_app(), raise_server_exceptions=False)

    assert client.get("/free").status_code == 200

    r = client.get("/busy")
    assert r.status_code == 503
    # 이 헤더가 없으면 클라이언트는 즉시 재시도하고 거절만 늘린다.
    # upstream 핸들러는 status 와 body 만 돌려줘서 이걸 잃었다.
    assert r.headers.get("Retry-After") == "5"


def test_slot_is_released_after_rejection():
    """거절이 자리를 갉아먹지 않는다. 그러면 용량이 서서히 0 이 된다."""
    client = TestClient(_app(), raise_server_exceptions=False)

    for _ in range(3):
        assert client.get("/busy").status_code == 503
    assert client.get("/free").status_code == 200
