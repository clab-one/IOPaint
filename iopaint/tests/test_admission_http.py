"""입장 제어의 HTTP 표면 계약.

test_admission.py 는 Admission 객체를 본다. 그것만으로는 부족했다 -
upstream 의 예외 핸들러가 헤더를 버려서 503 에 Retry-After 가 붙지 않는 것을
객체 테스트는 통과시켰고 실제 요청만 잡아냈다.
"""

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from iopaint.admission import Admission


def _app_with_handler() -> FastAPI:
    """api_middleware 와 같은 방식으로 예외를 처리하는 최소 앱."""
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.exception_handler(HTTPException)
    async def handler(request, e: HTTPException):
        return JSONResponse(
            status_code=e.status_code,
            content=jsonable_encoder({"detail": e.detail}),
            headers=getattr(e, "headers", None),
        )

    admission = Admission(inflight=1, max_queue=0)

    @app.get("/slow")
    def slow():
        with admission.admit():
            return {"ok": True}

    @app.get("/always-busy")
    def busy():
        # 자리를 먼저 채운 뒤 한 번 더 들어가려 한다 - 반드시 거절이다.
        with admission.admit():
            with admission.admit():
                return {"ok": True}

    return app


def test_busy_response_carries_retry_after():
    client = TestClient(_app_with_handler(), raise_server_exceptions=False)

    assert client.get("/slow").status_code == 200

    r = client.get("/always-busy")
    assert r.status_code == 503
    # 이 헤더가 없으면 클라이언트는 즉시 재시도하고 거절만 늘린다.
    assert r.headers.get("Retry-After") == "5"
