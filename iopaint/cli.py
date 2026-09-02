import os
from pathlib import Path
from typing import Optional

import typer
from fastapi import FastAPI
from loguru import logger
from typer import Option
from typer_config import use_json_config

from iopaint.const import *
from iopaint.runtime import setup_model_dir, dump_environment_info, check_device
from iopaint.schema import InteractiveSegModel, Device, RealESRGANModel, RemoveBGModel

# FOLIO fork: install-plugins-packages / download / list / run(batch) /
# start-web-config 를 들어냈다. 서버는 헤드리스로만 뜬다.
def make_app():
    """배포와 같은 설정의 FastAPI 앱.

    시험이 이것을 부른다. `FastAPI()` 를 시험이 따로 만들면 배포와 다른
    앱을 검사하게 되고, 실제로 그렇게 해서 docs 노출을 못 잡았다.

    문서 UI 와 스키마를 끈다. 기본값으로 두면 /docs · /redoc ·
    /openapi.json 이 열린다. 라우트를 13개에서 5개로 줄여 놓고 그 5개의
    스키마를 인터넷에 게시하면 감축의 절반이 무의미하다 - 필드 이름·타입·
    기본값이 그대로 나간다.
    """
    from fastapi import FastAPI

    return FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


typer_app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)


@typer_app.callback()
def main():
    """FOLIO fork of IOPaint - headless erase service.

    명령이 `start` 하나만 남았지만 콜백을 둬서 subcommand 형태를 유지한다.
    없으면 typer 가 단일 명령 앱으로 접어버려 `iopaint start` 가 깨진다.
    """


@typer_app.command(help="Start IOPaint server")
@use_json_config()
def start(
    host: str = Option("127.0.0.1"),
    port: int = Option(8080),
    model: str = Option(
        DEFAULT_MODEL, help=f"Erase models: [{', '.join(AVAILABLE_MODELS)}]."
    ),
    model_dir: Path = Option(
        DEFAULT_MODEL_DIR,
        help=MODEL_DIR_HELP,
        dir_okay=True,
        file_okay=False,
        callback=setup_model_dir,
    ),
    low_mem: bool = Option(False, help="Enable attention slicing to save memory."),
    no_half: bool = Option(False, help=NO_HALF_HELP),
    device: Device = Option(Device.cpu),
    inflight: int = Option(1, help=INFLIGHT_HELP),
    max_queue: int = Option(8, help=MAX_QUEUE_HELP),
    output_dir: Optional[Path] = Option(
        None, help=OUTPUT_DIR_HELP, dir_okay=True, file_okay=False
    ),
    quality: int = Option(100, help=QUALITY_HELP),
    enable_interactive_seg: bool = Option(False, help=INTERACTIVE_SEG_HELP),
    interactive_seg_model: InteractiveSegModel = Option(
        InteractiveSegModel.sam2_1_tiny, help=INTERACTIVE_SEG_MODEL_HELP
    ),
    interactive_seg_device: Device = Option(Device.cpu),
    enable_remove_bg: bool = Option(False, help=REMOVE_BG_HELP),
    remove_bg_device: Device = Option(Device.cpu, help=REMOVE_BG_DEVICE_HELP),
    remove_bg_model: RemoveBGModel = Option(RemoveBGModel.briaai_rmbg_1_4),
    enable_realesrgan: bool = Option(False, help=REALESRGAN_HELP),
    realesrgan_device: Device = Option(Device.cpu),
    realesrgan_model: RealESRGANModel = Option(RealESRGANModel.realesr_general_x4v3),
    enable_gfpgan: bool = Option(False, help=GFPGAN_HELP),
    gfpgan_device: Device = Option(Device.cpu),
    enable_restoreformer: bool = Option(False, help=RESTOREFORMER_HELP),
    restoreformer_device: Device = Option(Device.cpu),
):
    dump_environment_info()
    device = check_device(device)
    remove_bg_device = check_device(remove_bg_device)
    realesrgan_device = check_device(realesrgan_device)
    gfpgan_device = check_device(gfpgan_device)

    if output_dir:
        output_dir = output_dir.expanduser().absolute()
        logger.info(f"Image will be saved to {output_dir}")
        if not output_dir.exists():
            logger.info(f"Create output directory {output_dir}")
            output_dir.mkdir(parents=True)

    model_dir = model_dir.expanduser().absolute()

    from iopaint.download import cli_download_model, scan_models

    scanned_models = scan_models()
    if model not in [it.name for it in scanned_models]:
        logger.info(f"{model} not found in {model_dir}, try to downloading")
        cli_download_model(model)

    from iopaint.api import Api
    from iopaint.schema import ApiConfig

    app = make_app()

    api_config = ApiConfig(
        host=host,
        port=port,
        model=model,
        no_half=no_half,
        low_mem=low_mem,
        device=device,
        output_dir=output_dir,
        inflight=inflight,
        max_queue=max_queue,
        quality=quality,
        enable_interactive_seg=enable_interactive_seg,
        interactive_seg_model=interactive_seg_model,
        interactive_seg_device=interactive_seg_device,
        enable_remove_bg=enable_remove_bg,
        remove_bg_device=remove_bg_device,
        remove_bg_model=remove_bg_model,
        enable_realesrgan=enable_realesrgan,
        realesrgan_device=realesrgan_device,
        realesrgan_model=realesrgan_model,
        enable_gfpgan=enable_gfpgan,
        gfpgan_device=gfpgan_device,
        enable_restoreformer=enable_restoreformer,
        restoreformer_device=restoreformer_device,
    )
    print(api_config.model_dump_json(indent=4))
    api = Api(app, api_config)
    api.launch()
