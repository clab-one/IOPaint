"""노출된 경로가 정확히 이 목록인가.

FOLIO fork 가 추가한 시험이다.

## 왜 필요한가

upstream 은 13개 라우트를 열어 뒀고 FOLIO 앱은 그중 5개만 쓴다. 남은 8개는
기능이 아니라 공격 표면이었다 - `adjust_mask` 는 150바이트 요청 하나로
컨테이너를 OOMKill 할 수 있었고(`kernel_size` 무제한), `/api/v1/model` 은
30바이트로 추론을 전면 정지시킬 수 있었다.

지우는 것으로 끝나지 않는다. **다시 늘어나는 것을 막아야 한다.** upstream 을
다시 당기거나 기능을 급히 붙일 때 라우트는 조용히 되살아나고, 그때는 아무도
"이게 왜 열려 있지" 를 묻지 않는다.

그래서 목록을 계약으로 만든다. 늘리려면 이 파일을 함께 고쳐야 하고, 고치는
순간 리뷰에서 보인다.

## /docs 를 놓친 이야기

라우트 8개를 지우고 "5개만 남았다" 고 확인했는데, 임시 스크립트가 자기
`FastAPI()` 를 만들어 검사했다. 배포는 `cli.py` 가 만든 앱으로 뜨고 거기엔
`/docs`·`/redoc`·`/openapi.json` 이 열려 있었다 - 라우트를 줄여 놓고 그
5개의 스키마를 게시하는 셈이었다.

그래서 이 시험은 `iopaint.cli.make_app()` 을 부른다. 배포와 같은 것을 검사
하지 않는 시험은 통과해도 아무것도 보장하지 않는다.
"""

import pytest
from fastapi import FastAPI

import iopaint.api as api_module
from iopaint.cli import make_app
from iopaint.schema import ApiConfig

#: 앱이 실제로 부르는 것. `folio-core` 의 `Sources/FolioRemote/FolioAIClient.swift`
#: 가 이 다섯 개만 쓴다.
CLIENT_ROUTES = {
    "/v1/edit-sessions",
    "/v1/edit-sessions/{sid}/erase",
    "/v1/edit-sessions/{sid}/result",
    "/v1/edit-sessions/{sid}",
    "/api/v1/run_plugin_gen_image",
}

#: kubelet 이 부른다. Ingress 라우터에서는 가려야 하지만 라우트 자체는 있어야
#: 한다 - `deploy/k8s/deployment.yaml` 의 startup/readiness/liveness 프로브가
#: 이 경로를 직접 친다.
PROBE_ROUTES = {"/healthz", "/readyz"}


@pytest.fixture
def app(monkeypatch) -> FastAPI:
    """배포와 같은 앱. 모델과 플러그인은 대역으로 바꾼다."""

    class _Manager:
        name = "lama"
        available_models = {"lama": object()}

        def __call__(self, *a, **k):  # 호출되지 않아야 한다
            raise AssertionError("이 시험은 추론하지 않는다")

    monkeypatch.setattr(api_module.Api, "_build_model_manager", lambda self: _Manager())
    monkeypatch.setattr(api_module.Api, "_build_plugins", lambda self: {})

    instance = make_app()
    config = ApiConfig.model_construct(
        model="lama", inflight=1, max_queue=1, quality=95, output_dir=None
    )
    built = api_module.Api(instance, config)
    try:
        yield instance
    finally:
        built.sessions.close()


def registered(app: FastAPI) -> set[str]:
    return {route.path for route in app.routes if hasattr(route, "path")}


def test_exposed_routes_are_exactly_the_allowlist(app):
    """열린 경로가 목록과 정확히 같은가.

    부분집합이 아니라 **일치**를 본다. 늘어난 것도 줄어든 것도 잡아야 한다 -
    줄어들면 앱이 깨지고 늘어나면 표면이 넓어진다.
    """
    assert registered(app) == CLIENT_ROUTES | PROBE_ROUTES


def test_docs_and_schema_are_not_served(app):
    """문서 UI 와 OpenAPI 스키마가 없는가.

    이것을 켜 두면 남은 5개의 필드 이름·타입·기본값이 그대로 공개된다.
    라우트를 줄인 이유의 절반이 사라진다.
    """
    paths = registered(app)
    for leaked in ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"):
        assert leaked not in paths, f"{leaked} 가 열려 있다"


def test_removed_routes_have_no_handlers_left(app):
    """지운 라우트의 핸들러가 남아 있지 않은가.

    라우트 등록만 지우고 메서드를 남기면 다음 사람이 한 줄로 되살린다.
    그때 입력 상한과 인증은 함께 오지 않는다.
    """
    for dead in (
        "api_adjust_mask",
        "api_inpaint",
        "api_current_model",
        "api_switch_model",
        "api_switch_plugin_model",
        "api_run_plugin_gen_mask",
        "api_save_image",
        "api_server_config",
    ):
        assert not hasattr(api_module.Api, dead), f"{dead} 가 아직 있다"
