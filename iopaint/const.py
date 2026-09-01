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

INFLIGHT_HELP = """
동시에 돌릴 추론 수. 기본 1.
OMP_NUM_THREADS 가 이미 컨테이너 CPU 할당량을 전부 쓰도록 맞춰져 있으므로
두 개를 겹치면 스레드가 코어의 배수가 되어 둘 다 느려진다.
"""

MAX_QUEUE_HELP = """
대기열 상한. 기본 8. 넘으면 503 + Retry-After 로 즉시 거절한다.
무한정 기다리게 하면 프로브가 굶어 컨테이너가 죽는다.
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
