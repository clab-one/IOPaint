"""모델 가중치 선인출.

FOLIO fork 가 추가한 파일이다. upstream 에는 없다.

왜 필요한가 - 가중치를 이미지에 구우면 wave 3~5(RealESRGAN, GFPGAN,
RestoreFormer, SAM, RemoveBG)를 붙이는 순간 이미지가 GB 급이 된다.
그래서 가중치는 PVC 에 두고 이미지는 코드만 담는다. 이 모듈이 파드 기동 전
initContainer 에서 돌며 PVC 를 채운다.

`helper.download_model` 은 파일이 있으면 그냥 넘어간다 - md5 를 다시 보지
않는다. 이미지에 구워 넣을 때는 빌드가 원자적이라 문제가 없었지만, PVC 는
다르다. 다운로드 도중 파드가 죽으면 잘린 파일이 남고 그 뒤로는 영원히
"있다"고 판정된다. 그래서 여기서는 **있으면 md5 를 검사하고, 어긋나면 지우고
다시 받는다.**

받는 것이 유일한 방법은 아니다. 전환기에는 이미지에 이미 가중치가 구워져
있는데, PVC 를 `/models` 에 마운트하는 순간 그 레이어가 **가려진다**. 그래서
initContainer 에서는 PVC 를 다른 경로(`/pvc`)에 붙이고 이미지의 baked 경로를
`--seed-from` 으로 알려준다. 그러면 네트워크 없이 로컬 복사로 PVC 가 찬다.
bake 를 걷어낸 뒤에는 seed 가 없으니 자연히 다운로드로 떨어진다.

    python -m iopaint.prefetch lama
    python -m iopaint.prefetch --seed-from /models lama
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

from loguru import logger


def _specs() -> dict[str, tuple[str, str]]:
    """모델 이름 → (url, md5).

    플러그인 가중치도 여기 있다. 플러그인은 생성 시점에 스스로 받지만,
    그 시점이 파드 기동 중이라 **첫 기동이 다운로드만큼 길어진다.** 여기
    등록해 두면 initContainer 가 미리 채우고, 기동은 md5 확인만 한다.

    이름은 플러그인이 아니라 가중치 단위다 - 같은 플러그인이 모델을 바꿔
    끼우므로(`--interactive-seg-model`) 배포가 고른 것만 받으면 된다.
    """
    from iopaint.model.lama import LAMA_MODEL_MD5, LAMA_MODEL_URL
    from iopaint.plugins.interactive_seg import SEGMENT_ANYTHING_MODELS
    from iopaint.plugins.realesrgan import REAL_ESRGAN_MODELS

    specs = {"lama": (LAMA_MODEL_URL, LAMA_MODEL_MD5)}

    # SAM 계열. 표를 그대로 쓴다 - 여기 URL 을 옮겨 적으면 플러그인이 받는
    # 것과 어긋날 수 있고, 어긋나도 조용히 두 번 받을 뿐이라 안 드러난다.
    for key, spec in SEGMENT_ANYTHING_MODELS.items():
        specs[f"sam:{key}"] = (spec["url"], spec["md5"])

    # 키가 enum 이라 값으로 눕힌다 - CLI 가 넘기는 문자열과 같아야
    # `--interactive-seg-model` 처럼 배포 인자에서 그대로 쓸 수 있다.
    for key, spec in REAL_ESRGAN_MODELS.items():
        specs[f"realesrgan:{key.value}"] = (spec["url"], spec["model_md5"])

    return specs


def _seed_candidate(seed_root: str, path: str) -> str:
    """캐시 경로를 seed 루트 아래의 같은 자리로 옮겨 본다.

    `get_cache_path_by_url` 은 `$XDG_CACHE_HOME/torch/hub/checkpoints/<파일>`
    을 준다. seed 이미지도 같은 규칙으로 구워졌으므로 접미부만 붙이면 된다.
    맞는지는 어차피 md5 로 확인하니, 틀린 추측은 다운로드로 떨어질 뿐이다.
    """
    return os.path.join(seed_root, "torch", "hub", "checkpoints", os.path.basename(path))


def prefetch(name: str, seed_from: str | None = None) -> str:
    from iopaint.helper import download_model, get_cache_path_by_url, md5sum

    specs = _specs()
    if name not in specs:
        raise SystemExit(f"unknown model: {name}. known: {sorted(specs)}")

    url, expected = specs[name]
    path = get_cache_path_by_url(url)

    if os.path.exists(path):
        actual = md5sum(path)
        if actual == expected:
            logger.info(f"{name}: 이미 있음 ({path}, md5 확인)")
            return path
        # 잘렸거나 다른 파일이다. 두고 가면 영원히 "있다"로 판정된다.
        logger.warning(f"{name}: md5 불일치 {actual} != {expected}, 지우고 다시 받는다")
        os.remove(path)

    if seed_from:
        src = _seed_candidate(seed_from, path)
        if os.path.exists(src) and md5sum(src) == expected:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # 같은 파일시스템이 아니므로 rename 은 못 쓴다. 임시 이름으로 받아
            # 다 옮긴 뒤에 제자리로 넘긴다 - 중간에 죽어도 잘린 파일이 최종
            # 경로에 남지 않는다.
            tmp = f"{path}.partial"
            shutil.copyfile(src, tmp)
            os.replace(tmp, path)
            logger.info(f"{name}: 이미지에서 복사 ({src} -> {path})")
            return path
        logger.info(f"{name}: seed 없음/불일치 ({src}), 받는다")

    download_model(url, expected)
    logger.info(f"{name}: 준비됨 ({path})")
    return path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="iopaint.prefetch")
    ap.add_argument("models", nargs="*", default=["lama"])
    ap.add_argument(
        "--seed-from",
        metavar="DIR",
        help="이미지에 구워진 캐시 루트. 있으면 받지 않고 복사한다.",
    )
    args = ap.parse_args(argv)

    logger.info(f"XDG_CACHE_HOME={os.environ.get('XDG_CACHE_HOME', '(unset)')}")
    for name in args.models or ["lama"]:
        prefetch(name, seed_from=args.seed_from)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
