import os
import threading
import time
import traceback
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch

try:
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
    torch._C._jit_set_profiling_mode(False)
except:
    pass

import uvicorn
from PIL import Image
from fastapi import APIRouter, FastAPI, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from loguru import logger

from iopaint.admission import Admission
from iopaint.helper import (
    decode_base64_to_image,
    load_img,
    pil_to_bytes,
    numpy_to_bytes,
    concat_alpha_channel,
    gen_frontend_mask,
    adjust_mask,
)
from iopaint.model.utils import torch_gc
from iopaint.session import Reaper, SessionError, SessionStore
from iopaint.model_manager import ModelManager
from iopaint.plugins import build_plugins, RealESRGANUpscaler, InteractiveSeg
from iopaint.plugins.base_plugin import BasePlugin
from iopaint.plugins.remove_bg import RemoveBG
from iopaint.schema import (
    ApiConfig,
    ServerConfigResponse,
    SwitchModelRequest,
    InpaintRequest,
    RunPluginRequest,
    PluginInfo,
    AdjustMaskRequest,
    RemoveBGModel,
    SwitchPluginModelRequest,
    ModelInfo,
    InteractiveSegModel,
    RealESRGANModel,
)


def api_middleware(app: FastAPI):
    rich_available = False
    try:
        if os.environ.get("WEBUI_RICH_EXCEPTIONS", None) is not None:
            import anyio  # importing just so it can be placed on silent list
            import starlette  # importing just so it can be placed on silent list
            from rich.console import Console

            console = Console()
            rich_available = True
    except Exception:
        pass

    def handle_exception(request: Request, e: Exception):
        err = {
            "error": type(e).__name__,
            "detail": vars(e).get("detail", ""),
            "body": vars(e).get("body", ""),
            "errors": str(e),
        }
        if not isinstance(
            e, HTTPException
        ):  # do not print backtrace on known httpexceptions
            message = f"API error: {request.method}: {request.url} {err}"
            if rich_available:
                print(message)
                console.print_exception(
                    show_locals=True,
                    max_frames=2,
                    extra_lines=1,
                    suppress=[anyio, starlette],
                    word_wrap=False,
                    width=min([console.width, 200]),
                )
            else:
                traceback.print_exc()
        # FOLIO 변경: 헤더를 버리지 않는다.
        # upstream 핸들러는 status 와 body 만 돌려줘서 503 의 Retry-After 가
        # 사라졌다. 클라이언트가 언제 다시 와야 하는지 모르면 즉시 재시도해
        # 거절만 늘린다 - 입장 제어를 넣은 이유를 스스로 무너뜨린다.
        return JSONResponse(
            status_code=vars(e).get("status_code", 500),
            content=jsonable_encoder(err),
            headers=getattr(e, "headers", None),
        )

    @app.middleware("http")
    async def exception_handling(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            return handle_exception(request, e)

    @app.exception_handler(Exception)
    async def fastapi_exception_handler(request: Request, e: Exception):
        return handle_exception(request, e)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, e: HTTPException):
        return handle_exception(request, e)

    cors_options = {
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "allow_origins": ["*"],
        "allow_credentials": True,
        "expose_headers": ["X-Seed"],
    }
    app.add_middleware(CORSMiddleware, **cors_options)


class Api:
    def __init__(self, app: FastAPI, config: ApiConfig):
        self.app = app
        self.config = config
        self.router = APIRouter()
        # 플러그인 객체의 수명을 지킨다. `switch_plugin_model` 은 플러그인이
        # 들고 있는 모델을 통째로 갈아치우는데, 그 사이 `gen_image`/`gen_mask`
        # 가 같은 객체를 쓰고 있으면 반쯤 바뀐 상태를 읽는다. inpaint 쪽은
        # ModelManager 안에서 막았지만 플러그인은 upstream 클래스라 여기서
        # 막는다 - 그래야 fork 가 upstream 을 계속 당겨올 수 있다.
        #
        # (upstream 의 queue_lock 은 diffusion 경로와 함께 사라져 쓰이지
        #  않고 있었다. 이름을 정확한 것으로 바꿔 되살린다.)
        self._plugin_lock = threading.Lock()
        # 무한 동시성이 컨테이너를 죽인다 - iopaint/admission.py 참조.
        self.admission = Admission(
            inflight=config.inflight, max_queue=config.max_queue
        )
        api_middleware(self.app)

        self.plugins = self._build_plugins()
        self.model_manager = self._build_model_manager()

        # 편집 세션 (AI-003). tmpfs 에 산다 - 배포가 medium: Memory 로 붙인다.
        # 사진 바이트는 디스크에도 DB 에도 남기지 않는다.
        self.sessions = SessionStore(
            os.environ.get("FOLIO_SESSION_DIR", "/tmp/folio-sessions"),
            ttl=float(os.environ.get("FOLIO_SESSION_TTL", 20 * 60)),
            max_sessions=int(os.environ.get("FOLIO_MAX_SESSIONS", 32)),
        )
        self._reaper = Reaper(self.sessions)
        self._reaper.start()

        # fmt: off
        self.add_api_route("/api/v1/server-config", self.api_server_config, methods=["GET"],
                           response_model=ServerConfigResponse)
        self.add_api_route("/api/v1/model", self.api_current_model, methods=["GET"], response_model=ModelInfo)
        self.add_api_route("/api/v1/model", self.api_switch_model, methods=["POST"], response_model=ModelInfo)
        self.add_api_route("/api/v1/inpaint", self.api_inpaint, methods=["POST"])
        self.add_api_route("/api/v1/switch_plugin_model", self.api_switch_plugin_model, methods=["POST"])
        self.add_api_route("/api/v1/run_plugin_gen_mask", self.api_run_plugin_gen_mask, methods=["POST"])
        self.add_api_route("/api/v1/run_plugin_gen_image", self.api_run_plugin_gen_image, methods=["POST"])
        self.add_api_route("/api/v1/adjust_mask", self.api_adjust_mask, methods=["POST"])
        self.add_api_route("/api/v1/save_image", self.api_save_image, methods=["POST"])
        # 편집 세션 (AI-003). 사진을 한 번만 올리고 그 뒤로는 마스크만 보낸다.
        # 무상태 /api/v1/inpaint 는 그대로 둔다 - 한 번 쓰고 마는 호출에는
        # 세션이 과하고, 기존 계약을 깰 이유가 없다.
        self.add_api_route("/v1/edit-sessions", self.api_session_create, methods=["POST"])
        self.add_api_route("/v1/edit-sessions/{sid}/erase", self.api_session_erase, methods=["POST"])
        self.add_api_route("/v1/edit-sessions/{sid}/result", self.api_session_result, methods=["GET"])
        self.add_api_route("/v1/edit-sessions/{sid}", self.api_session_delete, methods=["DELETE"])
        # 프로브 전용. async 라 이벤트 루프에서 직접 답한다 - 추론이 점유한
        # 스레드풀을 거치지 않는다. 이전 배포는 프로브가 /api/v1/server-config
        # 를 쳤고, 그게 모델 매니저를 건드리는 sync 엔드포인트라 부하 중에
        # 굶어 죽으면서 멀쩡한 컨테이너가 kill 됐다.
        self.add_api_route("/healthz", self.api_healthz, methods=["GET"])
        self.add_api_route("/readyz", self.api_readyz, methods=["GET"])
        # fmt: on

    def add_api_route(self, path: str, endpoint, **kwargs):
        return self.app.add_api_route(path, endpoint, **kwargs)

    def api_save_image(self, file: UploadFile):
        # Sanitize filename to prevent path traversal
        safe_filename = Path(file.filename).name  # Get just the filename component

        # Construct the full path within output_dir
        output_path = self.config.output_dir / safe_filename

        # Ensure output directory exists
        if not self.config.output_dir or not self.config.output_dir.exists():
            raise HTTPException(
                status_code=400,
                detail="Output directory not configured or doesn't exist",
            )

        # Read and write the file
        origin_image_bytes = file.file.read()
        with open(output_path, "wb") as fw:
            fw.write(origin_image_bytes)

    # --- 편집 세션 (AI-003) ------------------------------------------------
    #
    # 계약은 하나다: **같은 세션의 연산은 순서가 있다.** 두 번째 지우기는 첫
    # 번째 결과 위에서 일어난다. 무상태 /api/v1/inpaint 로는 그 순서를 서버가
    # 알 수 없어 클라이언트가 결과를 받아 다시 올려야 했고, 왕복마다 전체
    # 이미지가 오갔다.
    #
    # 직렬화는 세션 단위다. 서로 다른 세션은 서로를 기다리지 않는다.

    def _session_or_http(self, sid: str):
        try:
            return self.sessions.get(sid)
        except SessionError as e:
            raise HTTPException(status_code=e.status, detail=e.detail)

    def api_session_create(self, file: UploadFile):
        """원본을 한 번 올린다. 그 뒤로는 마스크만 보낸다."""
        data = file.file.read()
        ext = (Path(file.filename or "").suffix.lstrip(".") or "png").lower()
        try:
            s = self.sessions.create(data, ext)
        except SessionError as e:
            raise HTTPException(status_code=e.status, detail=e.detail)
        return JSONResponse(
            {"id": s.id, "ttl_seconds": int(self.sessions.ttl)}, status_code=201
        )

    def api_session_erase(self, sid: str, mask: UploadFile):
        """현재 working 이미지에서 마스크 영역을 지운다.

        입장 제어는 세션 락 **밖**에서 잡는다. 안에서 잡으면 같은 세션을 기다리는
        요청이 슬롯을 쥔 채로 줄을 서서, 다른 세션의 작업까지 굶는다.
        """
        s = self._session_or_http(sid)
        mask_bytes = mask.file.read()

        with self.admission.admit():
            with s.lock:
                # 락을 기다리는 동안 세션이 삭제됐을 수 있다. 삭제는 같은
                # 락을 잡고 내리므로, 여기 들어왔다는 것은 그 경쟁이 끝났다는
                # 뜻이다 - 결과를 다시 확인한다.
                if not s.alive or not s.working.exists():
                    raise HTTPException(status_code=404, detail="session not found")
                image, alpha_channel, infos = load_img(
                    s.working.read_bytes(), return_info=True
                )
                mask_np, _ = load_img(mask_bytes, gray=True)
                mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)[1]
                if image.shape[:2] != mask_np.shape[:2]:
                    raise HTTPException(
                        400,
                        detail=f"Image size({image.shape[:2]}) and mask size({mask_np.shape[:2]}) not match.",
                    )

                req = InpaintRequest(image="", mask="")
                start = time.time()
                rgb_np_img = self.model_manager(image, mask_np, req)
                elapsed = (time.time() - start) * 1000
                torch_gc()

                rgb_np_img = cv2.cvtColor(rgb_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                rgb_res = concat_alpha_channel(rgb_np_img, alpha_channel)
                out = pil_to_bytes(
                    Image.fromarray(rgb_res), ext="png", quality=self.config.quality,
                    infos=infos,
                )
                # 원자적으로 바꾼다. 도중에 죽어도 반쯤 쓰인 working 이 남지 않는다.
                tmp = s.working.with_suffix(".png.partial")
                tmp.write_bytes(out)
                os.replace(tmp, s.working)
                s.ops += 1
                logger.info(f"session {sid}: erase #{s.ops} {elapsed:.0f}ms")

        return JSONResponse({"id": sid, "ops": s.ops, "ms": round(elapsed)})

    def api_session_result(self, sid: str):
        """현재 working 을 그대로 준다.

        락을 잡지 않는다. working 은 `os.replace` 로만 갱신되므로 읽는 쪽은
        항상 어떤 연산의 완결된 결과를 본다 - 반쯤 쓰인 파일은 존재하지 않는다.
        진행 중인 지우기를 기다리지 않는 편이 낫다: 클라이언트가 원하는 것은
        "지금까지의 결과"이고, 자기 요청의 결과가 필요하면 그 응답을 이미
        받은 뒤다.
        """
        s = self._session_or_http(sid)
        if not s.alive or not s.working.exists():
            raise HTTPException(status_code=404, detail="session not found")
        return Response(content=s.working.read_bytes(), media_type="image/png")

    def api_session_delete(self, sid: str):
        """사용자가 `Done` 을 눌렀다. 사진을 즉시 지운다."""
        try:
            self.sessions.delete(sid)
        except SessionError as e:
            raise HTTPException(status_code=e.status, detail=e.detail)
        return Response(status_code=204)

    async def api_healthz(self):
        """살아 있는가. 바쁨은 죽음이 아니다 - 큐가 가득 차도 200 이다."""
        return JSONResponse(
            {
                "ok": True,
                "in_system": self.admission.present,
                "capacity": self.admission.inflight + self.admission.max_queue,
                # 세션은 tmpfs 에 산다 - 노드 RAM 이다. 몇 개가 살아 있는지
                # 밖에서 볼 수 없으면 사진이 쌓이는 것을 아무도 모른다.
                # reaper 는 데몬 스레드라 조용히 죽을 수 있고, Python 3.12 는
                # 스레드 이름을 OS 에 붙이지 않아 /proc 으로도 확인이 안 된다.
                "sessions": {**self.sessions.stats(), "reaper": self._reaper.alive},
            }
        )

    async def api_readyz(self):
        """요청을 받을 수 있는가.

        적재 여부만 본다. 혼잡을 not-ready 로 답하면 엔드포인트가 Service 에서
        빠지고, 그 순간 남은 요청까지 connection refused 가 된다 - 혼잡을
        장애로 승격시키는 짓이다. 혼잡은 503 으로 답할 일이지 이탈할 일이 아니다.
        """
        ready = self.model_manager is not None
        return JSONResponse({"ready": ready}, status_code=200 if ready else 503)

    def api_current_model(self) -> ModelInfo:
        return self.model_manager.current_model

    def api_switch_model(self, req: SwitchModelRequest) -> ModelInfo:
        if req.name == self.model_manager.name:
            return self.model_manager.current_model
        self.model_manager.switch(req.name)
        return self.model_manager.current_model

    def api_switch_plugin_model(self, req: SwitchPluginModelRequest):
        if req.plugin_name in self.plugins:
            with self._plugin_lock:
                self.plugins[req.plugin_name].switch_model(req.model_name)
            if req.plugin_name == RemoveBG.name:
                self.config.remove_bg_model = req.model_name
            if req.plugin_name == RealESRGANUpscaler.name:
                self.config.realesrgan_model = req.model_name
            if req.plugin_name == InteractiveSeg.name:
                self.config.interactive_seg_model = req.model_name
            torch_gc()

    def api_server_config(self) -> ServerConfigResponse:
        plugins = []
        for it in self.plugins.values():
            plugins.append(
                PluginInfo(
                    name=it.name,
                    support_gen_image=it.support_gen_image,
                    support_gen_mask=it.support_gen_mask,
                )
            )

        return ServerConfigResponse(
            plugins=plugins,
            modelInfos=self.model_manager.scan_models(),
            removeBGModel=self.config.remove_bg_model,
            removeBGModels=RemoveBGModel.values(),
            realesrganModel=self.config.realesrgan_model,
            realesrganModels=RealESRGANModel.values(),
            interactiveSegModel=self.config.interactive_seg_model,
            interactiveSegModels=InteractiveSegModel.values(),
            enableAutoSaving=self.config.output_dir is not None,
        )

    def api_inpaint(self, req: InpaintRequest):
        with self.admission.admit():
            return self._api_inpaint(req)

    def _api_inpaint(self, req: InpaintRequest):
        image, alpha_channel, infos, ext = decode_base64_to_image(req.image)
        mask, _, _, _ = decode_base64_to_image(req.mask, gray=True)
        logger.info(f"image ext: {ext}")

        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        if image.shape[:2] != mask.shape[:2]:
            raise HTTPException(
                400,
                detail=f"Image size({image.shape[:2]}) and mask size({mask.shape[:2]}) not match.",
            )

        start = time.time()
        rgb_np_img = self.model_manager(image, mask, req)
        logger.info(f"process time: {(time.time() - start) * 1000:.2f}ms")
        torch_gc()

        rgb_np_img = cv2.cvtColor(rgb_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        rgb_res = concat_alpha_channel(rgb_np_img, alpha_channel)

        res_img_bytes = pil_to_bytes(
            Image.fromarray(rgb_res),
            ext=ext,
            quality=self.config.quality,
            infos=infos,
        )

        return Response(content=res_img_bytes, media_type=f"image/{ext}")

    def _require_plugin(self, name: str, *, gen_image: bool):
        """자리를 잡기 전에 거절할 수 있는 것은 먼저 거절한다.

        입장 제어를 검증보다 먼저 잡으면 없는 플러그인을 부르는 요청도 자리를
        차지한다. 실측: inpaint 6 + 없는 플러그인 6 을 동시에 넣었더니 무효
        요청 4건이 슬롯을 먹고 진짜 지우기 1건이 503 을 맞았다. 사전형 조회는
        비용이 없으니 문 앞에서 끝낸다.
        """
        if name not in self.plugins:
            raise HTTPException(status_code=422, detail="Plugin not found")
        supported = (
            self.plugins[name].support_gen_image
            if gen_image
            else self.plugins[name].support_gen_mask
        )
        if not supported:
            kind = "image" if gen_image else "mask"
            raise HTTPException(
                status_code=422, detail=f"Plugin does not support output {kind}"
            )

    def api_run_plugin_gen_image(self, req: RunPluginRequest):
        self._require_plugin(req.name, gen_image=True)
        with self.admission.admit():
            return self._api_run_plugin_gen_image(req)

    def _api_run_plugin_gen_image(self, req: RunPluginRequest):
        ext = "png"
        rgb_np_img, alpha_channel, infos, _ = decode_base64_to_image(req.image)
        with self._plugin_lock:
            bgr_or_rgba_np_img = self.plugins[req.name].gen_image(rgb_np_img, req)
        torch_gc()

        if bgr_or_rgba_np_img.shape[2] == 4:
            rgba_np_img = bgr_or_rgba_np_img
        else:
            rgba_np_img = cv2.cvtColor(bgr_or_rgba_np_img, cv2.COLOR_BGR2RGB)
            rgba_np_img = concat_alpha_channel(rgba_np_img, alpha_channel)

        return Response(
            content=pil_to_bytes(
                Image.fromarray(rgba_np_img),
                ext=ext,
                quality=self.config.quality,
                infos=infos,
            ),
            media_type=f"image/{ext}",
        )

    def api_run_plugin_gen_mask(self, req: RunPluginRequest):
        self._require_plugin(req.name, gen_image=False)
        with self.admission.admit():
            return self._api_run_plugin_gen_mask(req)

    def _api_run_plugin_gen_mask(self, req: RunPluginRequest):
        rgb_np_img, _, _, _ = decode_base64_to_image(req.image)
        with self._plugin_lock:
            bgr_or_gray_mask = self.plugins[req.name].gen_mask(rgb_np_img, req)
        torch_gc()
        res_mask = gen_frontend_mask(bgr_or_gray_mask)
        return Response(
            content=numpy_to_bytes(res_mask, "png"),
            media_type="image/png",
        )

    def api_adjust_mask(self, req: AdjustMaskRequest):
        mask, _, _, _ = decode_base64_to_image(req.mask, gray=True)
        mask = adjust_mask(mask, req.kernel_size, req.operate)
        return Response(content=numpy_to_bytes(mask, "png"), media_type="image/png")

    def launch(self):
        self.app.include_router(self.router)
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            timeout_keep_alive=999999999,
        )

    def _build_plugins(self) -> Dict[str, BasePlugin]:
        return build_plugins(
            self.config.enable_interactive_seg,
            self.config.interactive_seg_model,
            self.config.interactive_seg_device,
            self.config.enable_remove_bg,
            self.config.remove_bg_device,
            self.config.remove_bg_model,
            self.config.enable_realesrgan,
            self.config.realesrgan_device,
            self.config.realesrgan_model,
            self.config.enable_gfpgan,
            self.config.gfpgan_device,
            self.config.enable_restoreformer,
            self.config.restoreformer_device,
            self.config.no_half,
        )

    def _build_model_manager(self):
        return ModelManager(
            name=self.config.model,
            device=torch.device(self.config.device),
            no_half=self.config.no_half,
            low_mem=self.config.low_mem,
        )
