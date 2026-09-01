"""이미지와 매니페스트가 가중치 출처를 두고 합의하는가.

이 저장소는 코드와 배포 매니페스트가 한 곳에 있고 push 가 곧 배포다. 그래서
`Dockerfile` 과 `deploy/k8s/deployment.yaml` 이 서로 다른 가정을 하면 바로
클러스터에 나타난다.

실제로 네 번 났다. 마지막은 이랬다 - digest 만 되돌리고 그 이미지를 만든
Dockerfile 은 그대로 뒀다. 다음 push 에서 CI 가 가중치 없는 이미지를 다시
만들었고, PVC 를 붙이지 않는 Deployment 가 그것을 받아 매 기동마다 205MB 를
받게 됐다.

사람이 매번 두 파일을 대조하는 것으로는 못 막는다. 불변식은 하나다:

    런타임에 가중치가 존재할 경로가 **적어도 하나** 있어야 한다.
      - 이미지에 구워져 있거나 (Dockerfile 의 다운로드 단계)
      - PVC 를 /models 에 붙이고 initContainer 가 채우거나

둘 다 없으면 파드는 기동할 때마다 인터넷에서 받는다 - egress 가 막히면 아예
뜨지 못한다. 그 상태를 여기서 막는다.
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "Dockerfile"
DEPLOYMENT = ROOT / "deploy" / "k8s" / "deployment.yaml"

WEIGHT_MOUNT = "/models"


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return DOCKERFILE.read_text()


@pytest.fixture(scope="module")
def deployment() -> dict:
    return yaml.safe_load(DEPLOYMENT.read_text())


def _bakes_weights(dockerfile: str) -> bool:
    """빌드 중에 가중치를 이미지 레이어에 넣는가.

    지금은 `RUN python -m iopaint.prefetch lama` 한 줄이다 - 런타임과 같은
    코드를 써야 2단계의 `--seed-from` 이 같은 자리를 찾는다. 직접 받는
    형태도 인정한다.
    """
    for line in dockerfile.splitlines():
        s = line.strip()
        if s.startswith("#") or not s.startswith("RUN"):
            continue
        if "iopaint.prefetch" in s:
            return True
        if "download" in s and ("lama" in s or "model" in s):
            return True
    return False


def _pod_spec(deployment: dict) -> dict:
    return deployment["spec"]["template"]["spec"]


def _mounts_weights_volume(deployment: dict) -> bool:
    """주 컨테이너가 가중치 경로에 볼륨을 붙이는가."""
    for c in _pod_spec(deployment).get("containers", []):
        for m in c.get("volumeMounts", []):
            if m.get("mountPath") == WEIGHT_MOUNT:
                return True
    return False


def _prefetch_init(deployment: dict) -> dict | None:
    for c in _pod_spec(deployment).get("initContainers", []) or []:
        joined = " ".join(c.get("command", []) + c.get("args", []))
        if "iopaint.prefetch" in joined:
            return c
    return None


def test_weights_have_at_least_one_source(dockerfile, deployment):
    """이미지에 굽거나 PVC 를 채우거나 - 둘 중 하나는 있어야 한다."""
    baked = _bakes_weights(dockerfile)
    provisioned = _mounts_weights_volume(deployment) and _prefetch_init(deployment)

    assert baked or provisioned, (
        "가중치 출처가 없다. Dockerfile 에 굽는 단계도 없고 Deployment 가 "
        f"{WEIGHT_MOUNT} 에 볼륨을 붙이며 initContainer 로 채우지도 않는다. "
        "이대로 배포하면 파드가 매 기동마다 인터넷에서 받는다."
    )


def test_prefetch_init_container_has_the_module_to_run(deployment):
    """initContainer 가 부르는 모듈이 이미지에 실제로 들어가는가.

    prefetch 를 부르는 매니페스트를, 그 모듈이 없는 이미지에 적용해 Init:Error
    로 서비스를 내린 적이 있다.
    """
    init = _prefetch_init(deployment)
    if init is None:
        pytest.skip("prefetch initContainer 없음")

    assert (ROOT / "iopaint" / "prefetch.py").exists(), (
        "initContainer 가 iopaint.prefetch 를 부르는데 모듈이 저장소에 없다"
    )


def test_prefetch_init_does_not_mount_over_its_own_seed(deployment):
    """initContainer 가 이미지의 baked 경로를 PVC 로 덮어쓰지 않는가.

    PVC 를 /models 에 붙이면 이미지에 구워진 레이어가 가려진다. 그러면
    --seed-from 이 가리키는 자리에 아무것도 없다 - 복사가 아니라 다운로드가
    되고, 전환기의 안전판이 사라진다.
    """
    init = _prefetch_init(deployment)
    if init is None:
        pytest.skip("prefetch initContainer 없음")

    joined = " ".join(init.get("command", []) + init.get("args", []))
    if "--seed-from" not in joined:
        pytest.skip("seed 를 쓰지 않음")

    seed = joined.split("--seed-from", 1)[1].split()[0]
    covered = [m.get("mountPath") for m in init.get("volumeMounts", [])]
    assert seed not in covered, (
        f"initContainer 가 seed 경로 {seed} 를 볼륨으로 덮었다 - "
        "이미지의 baked 가중치가 가려져 복사할 것이 없다"
    )
