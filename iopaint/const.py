import os

# torch.jit 로 추적된 LaMa 는 mps 에서 돌지 않는다. helper.switch_mps_device 가
# 이 목록을 보고 cpu 로 되돌린다 - Apple 실리콘 가속은 Core ML worker 쪽이 맡는다.
MPS_UNSUPPORT_MODELS = ["lama"]

DEFAULT_MODEL = "lama"
AVAILABLE_MODELS = ["lama"]

DEFAULT_MODEL_DIR = os.path.abspath(
    os.getenv("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
)

MODEL_DIR_HELP = f"""
Model download directory (by setting XDG_CACHE_HOME environment variable), by default model download to {DEFAULT_MODEL_DIR}
"""

NO_HALF_HELP = "Using full precision model. If your generate result is always black or green, use this argument."

QUALITY_HELP = """
Quality of image encoding, 0-100. Default is 95, higher quality will generate larger file size.
"""

OUTPUT_DIR_HELP = """
Result images will be saved to output directory automatically.
"""

INTERACTIVE_SEG_HELP = "Enable interactive segmentation using Segment Anything."
INTERACTIVE_SEG_MODEL_HELP = "Model size: mobile_sam < vit_b < vit_l < vit_h. Bigger model size means better segmentation but slower speed."
REMOVE_BG_HELP = "Enable remove background plugin."
REMOVE_BG_DEVICE_HELP = "Device for remove background plugin. 'cuda' only supports briaai models(briaai/RMBG-1.4 and briaai/RMBG-2.0)"
REALESRGAN_HELP = "Enable realesrgan super resolution"
GFPGAN_HELP = "Enable GFPGAN face restore. To also enhance background, use with --enable-realesrgan"
RESTOREFORMER_HELP = "Enable RestoreFormer face restore. To also enhance background, use with --enable-realesrgan"
