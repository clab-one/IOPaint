"""가중치 출처 표. FOLIO fork 가 추가한 파일이다.

**이 모듈은 아무것도 import 하지 않는다.**

그 제약이 목적이다. CI 의 `validate` 잡은 torch 도 loguru 도 설치하지 않고
매니페스트와 이 표만 대조한다 - 몇 초에 끝나고, 실패하면 build 가 아예
시작되지 않아 digest 가 바뀌지 않는다. 여기에 무거운 import 를 하나라도
들이면 그 잡이 못 돌고, 못 도는 가드는 없는 가드다.

## 왜 플러그인이 아니라 여기 있나

처음엔 반대였다 - 플러그인이 자기 표를 들고 `iopaint.prefetch` 가 읽었다.
표는 한 벌이었지만 읽는 쪽이 torch 를 끌고 와서 validate 가 깨졌다.

그래서 뒤집었다. 표는 여기 있고 플러그인이 읽어 간다. 사본이 늘지 않으면서
가벼운 쪽이 무거운 쪽에 기대지 않는다.

## 고칠 때

여기 값을 고치면 플러그인이 받는 것도 같이 바뀐다. **사본을 만들지 말 것** -
어긋나도 조용히 두 번 받을 뿐이라 드러나지 않는다.

이 파일을 손으로 옮겨 적다가 md5 를 깨뜨린 적이 있다. 값은 원본에서
복사하거나 프로그램으로 뽑을 것.
"""

#: 지우기 모델. 이것만 플러그인이 아니라 핵심 경로다.
ERASE_WEIGHTS = {
    "lama": {
        "url": "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
        "md5": "e3aa4aaa15225a33ec84f9f4bc47e500",
    },
}

#: SAM 계열. 배포는 mobile_sam - CPU 뿐이고 대화형이라 vit_h(2.4GB)는 못 쓴다.
SEGMENT_ANYTHING_MODELS = {'vit_b': {'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
           'md5': '01ec64d29a2fca3f0661936605ae66f8'},
 'vit_l': {'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
           'md5': '0b3195507c641ddb6910d2bb5adee89c'},
 'vit_h': {'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
           'md5': '4b8939a88964f0f4ff5f5b2642c598a6'},
 'mobile_sam': {'url': 'https://github.com/Sanster/models/releases/download/MobileSAM/mobile_sam.pt',
                'md5': 'f3c0d8cda613564d499310dab6c812cd'},
 'sam_hq_vit_b': {'url': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth',
                  'md5': 'c6b8953247bcfdc8bb8ef91e36a6cacc'},
 'sam_hq_vit_l': {'url': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth',
                  'md5': '08947267966e4264fb39523eccc33f86'},
 'sam_hq_vit_h': {'url': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth',
                  'md5': '3560f6b6a5a6edacd814a1325c39640a'},
 'sam2_tiny': {'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt',
               'md5': '99eacccce4ada0b35153d4fd7af05297'},
 'sam2_small': {'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt',
                'md5': '7f320dbeb497330a2472da5a16c7324d'},
 'sam2_base': {'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt',
               'md5': '09dc5a3d7719f64aaea1d37341ef26f2'},
 'sam2_large': {'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt',
                'md5': '08083462423be3260cd6a5eef94dc01c'},
 'sam2_1_tiny': {'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',
                 'md5': '6aa6761c9da74fbaa74b4c790a0a2007'},
 'sam2_1_small': {'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt',
                  'md5': '51713b3d1994696d27f35f9c6de6f5ef'},
 'sam2_1_base': {'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt',
                 'md5': 'ec7bd7d23d280d5e3cfa45984c02eda5'},
 'sam2_1_large': {'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt',
                  'md5': '2b30654b6112c42a115563c638d238d9'}}

#: RealESRGAN. 구조(scale, 신경망)는 플러그인이 안다 - 여기는 출처만.
REAL_ESRGAN_WEIGHTS = {'realesr-general-x4v3': {'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
                          'md5': '91a7644643c884ee00737db24e478156'},
 'RealESRGAN_x4plus': {'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                       'md5': '99ec365d4afad750833258a1a24f44ca'},
 'RealESRGAN_x4plus_anime_6B': {'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
                                'md5': 'd58ce384064ec1591c2ea7b79dbf47ba'}}

#: 얼굴 복원. 값은 각 플러그인 소스에서 그대로 가져왔다.
FACE_RESTORE_WEIGHTS = {
    "GFPGANv1.4": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "md5": "94d735072630ab734561130a47bc44f8",
    },
    "RestoreFormer": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth",
        "md5": "eaeeff6c4a1caa1673977cb374e6f699",
    },
}

#: GFPGAN 이 얼굴을 찾고 잘라내는 데 쓰는 보조 모델. 본 가중치와 별개로
#: facexlib 이 같은 checkpoints 디렉터리에 받는다 - 빠뜨리면 얼굴 복원을
#: 처음 부르는 순간 200MB 를 받는다.
#:
#: md5 는 upstream 이 공표하지 않아 받아서 계산했다. 값이 바뀌면 prefetch 가
#: 지우고 다시 받으므로, 그때는 upstream 이 파일을 갈아 끼운 것이다.
FACEXLIB_WEIGHTS = {
    "detection_Resnet50_Final": {
        "url": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "md5": "bce939bc22d8cec91229716dd932e56e",
    },
    "parsing_parsenet": {
        "url": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        "md5": "33d9956898d4fa637c30eda7faa28496",
    },
}

#: Hugging Face 저장소에서 오는 것. URL 이 아니라 repo+파일로 부르고, 캐시도
#: torch 쪽이 아니라 $XDG_CACHE_HOME/huggingface 다. 그래서 prefetch 가 다른
#: 갈래로 처리한다 - md5 대신 hub 자체의 무결성 검사에 맡긴다.
HF_WEIGHTS = {
    "rmbg-1.4": {"repo": "briaai/RMBG-1.4", "file": "model.pth"},
    "rmbg-2.0": {"repo": "briaai/RMBG-2.0", "file": "model.safetensors"},
}
