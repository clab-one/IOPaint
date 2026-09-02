from enum import Enum
from pathlib import Path
from typing import Optional, Literal, List

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    # FOLIO fork: diffusion 계열(DIFFUSERS_SD/SDXL/OTHER)을 들어냈다. 남은 것은
    # 지우기 모델 하나뿐이므로 타입도 하나다.
    INPAINT = "inpaint"  # LaMa


class ModelInfo(BaseModel):
    name: str
    path: str
    model_type: ModelType


class Choices(str, Enum):
    @classmethod
    def values(cls):
        return [member.value for member in cls]


class RealESRGANModel(Choices):
    realesr_general_x4v3 = "realesr-general-x4v3"
    RealESRGAN_x4plus = "RealESRGAN_x4plus"
    RealESRGAN_x4plus_anime_6B = "RealESRGAN_x4plus_anime_6B"


class RemoveBGModel(Choices):
    briaai_rmbg_1_4 = "briaai/RMBG-1.4"
    briaai_rmbg_2_0 = "briaai/RMBG-2.0"
    # models from https://github.com/danielgatis/rembg
    u2net = "u2net"
    u2netp = "u2netp"
    u2net_human_seg = "u2net_human_seg"
    u2net_cloth_seg = "u2net_cloth_seg"
    silueta = "silueta"
    isnet_general_use = "isnet-general-use"
    birefnet_general = "birefnet-general"
    birefnet_general_lite = "birefnet-general-lite"
    birefnet_portrait = "birefnet-portrait"
    birefnet_dis = "birefnet-dis"
    birefnet_hrsod = "birefnet-hrsod"
    birefnet_cod = "birefnet-cod"
    birefnet_massive = "birefnet-massive"


class Device(Choices):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


class InteractiveSegModel(Choices):
    vit_b = "vit_b"
    vit_l = "vit_l"
    vit_h = "vit_h"
    sam_hq_vit_b = "sam_hq_vit_b"
    sam_hq_vit_l = "sam_hq_vit_l"
    sam_hq_vit_h = "sam_hq_vit_h"
    mobile_sam = "mobile_sam"

    sam2_tiny = "sam2_tiny"
    sam2_small = "sam2_small"
    sam2_base = "sam2_base"
    sam2_large = "sam2_large"

    sam2_1_tiny = "sam2_1_tiny"
    sam2_1_small = "sam2_1_small"
    sam2_1_base = "sam2_1_base"
    sam2_1_large = "sam2_1_large"


class PluginInfo(BaseModel):
    name: str
    support_gen_image: bool = False
    support_gen_mask: bool = False


class HDStrategy(str, Enum):
    # Use original image size
    ORIGINAL = "Original"
    # Resize the longer side of the image to a specific size(hd_strategy_resize_limit),
    # then do inpainting on the resized image. Finally, resize the inpainting result to the original size.
    # The area outside the mask will not lose quality.
    RESIZE = "Resize"
    # Crop masking area(with a margin controlled by hd_strategy_crop_margin) from the original image to do inpainting
    CROP = "Crop"


class ApiConfig(BaseModel):
    host: str
    port: int
    model: str
    no_half: bool
    low_mem: bool
    device: Device
    output_dir: Optional[Path]
    # 입장 제어. iopaint/admission.py 참조.
    inflight: int
    max_queue: int
    quality: int
    enable_interactive_seg: bool
    interactive_seg_model: InteractiveSegModel
    interactive_seg_device: Device
    enable_remove_bg: bool
    remove_bg_device: Device
    remove_bg_model: str
    enable_realesrgan: bool
    realesrgan_device: Device
    realesrgan_model: RealESRGANModel
    enable_gfpgan: bool
    gfpgan_device: Device
    enable_restoreformer: bool
    restoreformer_device: Device


class InpaintRequest(BaseModel):
    """지우기 요청.

    FOLIO fork: prompt·sampler·seed·controlnet·outpainting 등 diffusion 전용
    필드를 전부 들어냈다. 남은 것은 지우기 모델이 실제로 읽는 값뿐이다.
    """

    image: Optional[str] = Field(None, description="base64 encoded image")
    mask: Optional[str] = Field(None, description="base64 encoded mask")

    hd_strategy: str = Field(
        HDStrategy.CROP,
        description="Different way to preprocess image, only used by erase models(e.g. lama)",
    )
    hd_strategy_crop_trigger_size: int = Field(
        800,
        description="Crop trigger size for hd_strategy=CROP, if the longer side of the image is larger than this value, use crop strategy",
    )
    hd_strategy_crop_margin: int = Field(
        128, description="Crop margin for hd_strategy=CROP"
    )
    hd_strategy_resize_limit: int = Field(
        1280, description="Resize limit for hd_strategy=RESIZE"
    )

    keep_unmasked_area: bool = Field(
        True, description="Keep unmasked area unchanged"
    )


class RunPluginRequest(BaseModel):
    name: str
    image: str = Field(..., description="base64 encoded image")
    clicks: List[List[int]] = Field(
        [], description="Clicks for interactive seg, [[x,y,0/1], [x2,y2,0/1]]"
    )
    scale: float = Field(2.0, description="Scale for upscaling")



