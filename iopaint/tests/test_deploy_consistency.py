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


def test_coreml_url_comes_with_client_certificates(deployment):
    """원격 노드를 켰으면 인증서도 붙어 있어야 한다.

    M4 워커는 mTLS 로만 답한다. URL 만 주고 인증서를 안 붙이면 모든 요청이
    핸드셰이크에서 실패하고, 스케줄러는 매번 로컬로 되돌린다 - 조용히 느려질
    뿐 아무도 모른다. 가중치 출처와 같은 종류의 어긋남이다.
    """
    spec = _pod_spec(deployment)
    volumes = {v["name"]: v for v in spec.get("volumes", [])}

    for c in spec.get("containers", []):
        env = {e["name"]: e.get("value") for e in c.get("env", [])}
        if "FOLIO_COREML_URL" not in env:
            continue

        mounts = {m["mountPath"]: m for m in c.get("volumeMounts", [])}
        for var in ("FOLIO_COREML_CA", "FOLIO_COREML_CERT", "FOLIO_COREML_KEY"):
            path = env.get(var)
            assert path, f"{var} 가 없다 - 기본 경로에 기대면 조용히 어긋난다"

            # mountPath 문자열이 있는지가 아니라, 그 경로를 덮는 마운트가
            # 실제로 **이 시크릿**인지 본다. emptyDir 이나 ConfigMap 을
            # /pki 에 붙여도 통과하던 검사는 아무것도 보장하지 않았다.
            covering = [mp for mp in mounts if path.startswith(mp.rstrip("/") + "/")]
            assert covering, f"{var}={path} 를 덮는 volumeMount 가 없다"

            vol = volumes.get(mounts[max(covering, key=len)]["name"], {})
            secret = vol.get("secret", {}).get("secretName")
            assert secret == "folio-coreml-client", (
                f"{var}={path} 가 시크릿에서 오지 않는다 (실제: {vol}). "
                "mTLS 핸드셰이크가 매번 실패하고 조용히 로컬로 되돌아간다."
            )

    _assert_two_nodes_are_not_serialized(spec)


def _assert_two_nodes_are_not_serialized(spec):
    """원격 노드를 켰으면 입장 제어가 그것을 직렬화하면 안 된다.

    --inflight=1 은 로컬 CPU 를 지키려고 넣은 값인데, 그 일은 이제
    LocalBackend 의 슬롯이 맡는다. 여기 1 을 그대로 두면 로컬 CPU 를 쓰지도
    않는 원격 호출까지 같은 세마포어에 걸려, 노드를 하나 더 붙여도 처리량이
    한 톨도 늘지 않는다 - 실측 conc=1..8 전부 1.12 req/s.
    """
    for c in spec.get("containers", []):
        env = {e["name"] for e in c.get("env", [])}
        if "FOLIO_COREML_URL" not in env:
            continue
        flags = [a for a in c.get("args", []) if a.startswith("--inflight=")]
        assert flags, "--inflight 가 명시돼 있지 않다"
        n = int(flags[0].split("=", 1)[1])
        assert n >= 2, (
            f"--inflight={n} 이면 원격 노드가 로컬과 같은 줄에 선다. "
            "노드를 붙인 의미가 사라진다."
        )


def test_liveness_probe_outlasts_its_own_handler(deployment):
    """liveness 시간 초과가 핸들러 최악 지연보다 길어야 한다.

    kubelet 기본은 1초다. /healthz 가 그보다 오래 걸릴 수 있으면 멀쩡한
    컨테이너가 3회 만에 재시작된다 - 원격 노드 하나가 사라졌을 때 로컬로
    폴백하기는커녕 파드가 죽는, 정반대의 결과였다.
    """
    for c in _pod_spec(deployment).get("containers", []):
        probe = c.get("livenessProbe")
        if not probe:
            continue
        assert probe.get("timeoutSeconds", 1) >= 3, (
            "livenessProbe.timeoutSeconds 가 없거나 너무 짧다. "
            "kubelet 기본 1초로는 여유가 없다."
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
