import gc

import torch

# FOLIO fork: 이 파일에는 diffusers 스케줄러 어댑터와 StyleGAN 계열 연산
# (upfirdn2d, conv2d_resample, bias_act 등)이 1000줄 넘게 있었다. 전자는
# diffusion 모델이, 후자는 MAT/FcF/MIGAN 이 쓰던 것으로 둘 다 들어냈다.
# 남은 것은 지우기 경로가 실제로 부르는 하나뿐이다.


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
