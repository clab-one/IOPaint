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


# --- 플러그인 -----------------------------------------------------------------

#: 배포 인자에서 플러그인을 켜는 플래그 → 그 플러그인이 필요로 하는 가중치를
#: `iopaint.prefetch` 이름으로 돌려주는 함수.
#:
#: 모델을 고르는 플래그가 따로 있으면 그것을 읽는다 - 기본값과 다른 모델을
#: 켜 놓고 기본값만 프리페치하면 파드가 기동 중에 받는다.
_PLUGIN_WEIGHTS = {
    "--enable-interactive-seg": lambda a: f"sam:{_flag(a, '--interactive-seg-model', 'vit_l')}",
    "--enable-realesrgan": lambda a: f"realesrgan:{_flag(a, '--realesrgan-model', 'realesr-general-x4v3')}",
    # 얼굴 복원은 본 가중치 하나로 끝나지 않는다. facexlib 이 검출·파싱
    # 모델을 따로 받으므로 셋 다 있어야 기동 중 다운로드가 없다.
    "--enable-gfpgan": lambda a: "face:GFPGANv1.4",
    "--enable-restoreformer": lambda a: "face:RestoreFormer",
    "--enable-remove-bg": lambda a: f"hf:{_HF_ALIAS[_flag(a, '--remove-bg-model', 'briaai/RMBG-1.4')]}",
}

#: 얼굴 검출·파싱은 어느 복원 모델을 켜든 함께 필요하다.
_FACE_HELPERS = ("facexlib:detection_Resnet50_Final", "facexlib:parsing_parsenet")

#: CLI 가 받는 모델 이름 → prefetch 가 부르는 이름.
_HF_ALIAS = {"briaai/RMBG-1.4": "rmbg-1.4", "briaai/RMBG-2.0": "rmbg-2.0"}


def _flag(args, name, default):
    for a in args:
        if a.startswith(f"{name}="):
            return a.split("=", 1)[1]
    return default


def test_enabled_plugins_have_their_weights_prefetched(deployment):
    """켠 플러그인의 가중치는 initContainer 가 미리 받아야 한다.

    플러그인은 생성 시점에 가중치를 스스로 받는다. 그 시점이 파드 기동
    중이라, 여기 빠뜨리면 첫 기동이 다운로드만큼 길어진다. egress 가 막힌
    환경이면 아예 뜨지 못한다.

    사람이 두 목록을 대조하는 것으로는 못 막는다 - 켤 때는 인자만 보고,
    initContainer 는 화면 위쪽에 있어 눈에 안 들어온다.
    """
    spec = _pod_spec(deployment)
    prefetched = set()
    for c in spec.get("initContainers", []):
        prefetched |= set(_model_args(c.get("command", [])))

    for c in spec.get("containers", []):
        args = c.get("args", [])
        for flag, weight_of in _PLUGIN_WEIGHTS.items():
            if flag not in args:
                continue
            wants = [weight_of(args)]
            if flag in ("--enable-gfpgan", "--enable-restoreformer"):
                wants += list(_FACE_HELPERS)
            for want in wants:
                _assert_prefetched(flag, want, prefetched)


def _assert_prefetched(flag, want, prefetched):
            assert want in prefetched, (
                f"{flag} 을 켰는데 {want!r} 가 프리페치 목록에 없다. "
                f"현재 목록: {sorted(prefetched)}. "
                "이대로면 파드가 기동 중에 가중치를 받는다."
            )


def test_prefetch_knows_every_weight_the_manifest_asks_for(deployment):
    """매니페스트가 부르는 이름을 prefetch 가 알아야 한다.

    모르는 이름을 주면 initContainer 가 SystemExit 으로 죽고 파드는
    Init:Error 에 멈춘다. 오타 하나로 서비스가 안 뜬다.
    """
    known = _known_weight_names()
    for c in _pod_spec(deployment).get("initContainers", []):
        for name in _model_args(c.get("command", [])):
            assert name in known, (
                f"prefetch 가 모르는 이름: {name!r}. 아는 것: {sorted(known)}"
            )


#: `python -m iopaint.prefetch` 뒤에 값을 받는 옵션들. 그 값은 모델 이름이
#: 아니므로 건너뛴다. 이름 목록을 하드코딩해 거르면 경로가 바뀔 때마다
#: 시험이 엉뚱한 곳에서 깨진다.
_VALUED_OPTS = {"--seed-from"}


def _model_args(command: list[str]) -> list[str]:
    """initContainer command 에서 모델 이름만 골라낸다."""
    rest = command[command.index("iopaint.prefetch") + 1 :] if "iopaint.prefetch" in command else []
    names, skip = [], False
    for token in rest:
        if skip:
            skip = False
            continue
        if token in _VALUED_OPTS:
            skip = True
        elif not token.startswith("-"):
            names.append(token)
    return names


def _known_weight_names() -> set[str]:
    """`iopaint.prefetch` 가 받아들이는 이름 전부.

    **`iopaint.weights` 만 읽는다.** prefetch 를 import 하면 loguru 와 torch 가
    따라온다. 이 파일은 CI 의 validate 잡이 pytest+pyyaml 만 깔고 돌리는
    것이라 그 순간 잡이 죽는다 - 실제로 한 번 죽였다. 못 도는 가드는 없는
    가드고, 그러면 build 가 검사 없이 통과한다.

    두 모듈이 같은 표를 보므로 여기서 이름을 다시 만들어도 어긋나지 않는다.
    """
    from iopaint import weights

    names = set(weights.ERASE_WEIGHTS)
    for prefix, table in (
        ("sam", weights.SEGMENT_ANYTHING_MODELS),
        ("realesrgan", weights.REAL_ESRGAN_WEIGHTS),
        ("face", weights.FACE_RESTORE_WEIGHTS),
        ("facexlib", weights.FACEXLIB_WEIGHTS),
        ("hf", weights.HF_WEIGHTS),
    ):
        names |= {f"{prefix}:{k}" for k in table}
    return names


def test_guard_sees_the_same_names_prefetch_accepts():
    """가벼운 가드와 실제 prefetch 가 같은 이름 집합을 봐야 한다.

    가드는 torch 를 피하려고 이름을 스스로 만든다. 그 목록이 prefetch 가
    실제로 받아들이는 것과 갈라지면, 매니페스트가 통과하고도 initContainer 가
    죽는다 - 가드가 있는데 못 막는 최악의 형태다.
    """
    pytest.importorskip("torch", reason="무거운 쪽은 로컬과 build 잡에서만 검사한다")
    from iopaint import weights
    from iopaint.prefetch import _specs

    # hf: 는 URL 갈래가 아니라 _prefetch_hf 가 따로 받는다. _specs 에 없는
    # 것이 정상이므로 그만큼 빼고 견준다.
    hf = {f"hf:{k}" for k in weights.HF_WEIGHTS}
    assert _known_weight_names() - hf == set(_specs())
