"""가중치 선인출 계약.

`helper.download_model` 은 파일이 있으면 md5 를 보지 않고 넘어간다. 가중치를
이미지에 구울 때는 빌드가 원자적이라 문제가 없었지만, PVC 로 옮기면 다운로드
도중 파드가 죽었을 때 잘린 파일이 남고 그 뒤로는 영원히 "있다"로 판정된다.

prefetch 는 그 구멍을 막는다. 여기서 고정하는 것이 그 동작이다.
"""

import hashlib
from pathlib import Path

import pytest

from iopaint import prefetch as prefetch_mod


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
    """네트워크를 타지 않는 가짜 모델 하나."""
    cached = tmp_path / "fake.pt"
    url = "https://example.invalid/fake.pt"
    good = b"weights"
    digest = hashlib.md5(good).hexdigest()

    monkeypatch.setattr(prefetch_mod, "_specs", lambda: {"fake": (url, digest)})
    monkeypatch.setattr(
        "iopaint.helper.get_cache_path_by_url", lambda _u: str(cached)
    )

    downloads = []

    def fake_download(u, md5=None):
        downloads.append(u)
        cached.write_bytes(good)
        return str(cached)

    monkeypatch.setattr("iopaint.helper.download_model", fake_download)
    return cached, good, downloads


def test_downloads_when_missing(fake_model):
    cached, good, downloads = fake_model
    prefetch_mod.prefetch("fake")
    assert cached.read_bytes() == good
    assert len(downloads) == 1


def test_skips_when_present_and_intact(fake_model):
    cached, good, downloads = fake_model
    cached.write_bytes(good)
    prefetch_mod.prefetch("fake")
    # 이미 온전하면 다시 받지 않는다. 재시작마다 205MB 를 받으면 안 된다.
    assert downloads == []


def test_replaces_corrupt_file(fake_model):
    """잘린 파일을 지우고 다시 받는다.

    upstream 의 download_model 은 여기서 그냥 넘어가고, 그러면 파드는 손상된
    가중치로 영원히 뜬다.
    """
    cached, good, downloads = fake_model
    cached.write_bytes(b"trunc")  # 다운로드 중 죽은 흔적

    prefetch_mod.prefetch("fake")

    assert len(downloads) == 1
    assert cached.read_bytes() == good


def test_unknown_model_fails_loudly(fake_model):
    with pytest.raises(SystemExit):
        prefetch_mod.prefetch("nope")


# --- seed 경로 -------------------------------------------------------------
#
# PVC 를 `/models` 에 마운트하면 이미지에 구워진 가중치 레이어가 가려진다.
# 그래서 initContainer 는 PVC 를 다른 자리에 붙이고 이미지의 baked 경로를
# seed 로 준다. 이게 PVC 전환의 안전판이다 - 네트워크가 아예 필요 없다.


def _seed_dir(root: Path, name: str, payload: bytes) -> Path:
    d = root / "torch" / "hub" / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(payload)
    return root


def test_copies_from_seed_instead_of_downloading(fake_model, tmp_path):
    cached, good, downloads = fake_model
    seed = _seed_dir(tmp_path / "baked", "fake.pt", good)

    prefetch_mod.prefetch("fake", seed_from=str(seed))

    assert cached.read_bytes() == good
    assert downloads == [], "seed 가 있는데 받았다"


def test_falls_back_to_download_when_seed_absent(fake_model, tmp_path):
    """bake 를 걷어낸 뒤의 상태 - seed 가 없으면 그냥 받는다."""
    cached, good, downloads = fake_model

    prefetch_mod.prefetch("fake", seed_from=str(tmp_path / "empty"))

    assert len(downloads) == 1
    assert cached.read_bytes() == good


def test_rejects_a_corrupt_seed(fake_model, tmp_path):
    """seed 도 믿지 않는다. md5 가 어긋나면 받는다."""
    cached, good, downloads = fake_model
    seed = _seed_dir(tmp_path / "baked", "fake.pt", b"corrupt")

    prefetch_mod.prefetch("fake", seed_from=str(seed))

    assert len(downloads) == 1
    assert cached.read_bytes() == good


def test_seed_copy_leaves_no_partial_file(fake_model, tmp_path):
    """복사 중 죽어도 최종 경로에 잘린 파일이 남지 않는다."""
    cached, good, downloads = fake_model
    seed = _seed_dir(tmp_path / "baked", "fake.pt", good)

    prefetch_mod.prefetch("fake", seed_from=str(seed))

    assert not list(cached.parent.glob("*.partial"))
