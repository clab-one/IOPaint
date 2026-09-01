"""편집 세션 계약.

세션이 존재하는 이유는 두 가지다 - 사진을 한 번만 올리는 것, 그리고 **연산에
순서를 주는 것**. 무상태 `/api/v1/inpaint` 로는 두 번째 지우기가 첫 번째 결과
위에서 일어나는지 서버가 알 수 없다.

여기서 고정하는 것은 그 순서와, tmpfs 를 지키는 상한들이다.
"""

import threading
import time

import pytest

from iopaint.session import Reaper, SessionError, SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "sessions", ttl=60, max_sessions=3, max_bytes=1024)


def test_create_then_get_roundtrip(store):
    s = store.create(b"jpegbytes", "jpg")
    assert store.get(s.id).id == s.id
    assert s.working.read_bytes() == b"jpegbytes"
    assert (s.dir / "original.jpg").read_bytes() == b"jpegbytes"


def test_unknown_session_is_404(store):
    with pytest.raises(SessionError) as e:
        store.get("nope")
    assert e.value.status == 404


def test_delete_removes_the_photo_immediately(store):
    """`Done` 은 즉시 지운다 - TTL 을 기다리지 않는다."""
    s = store.create(b"x", "png")
    d = s.dir
    store.delete(s.id)
    assert not d.exists()
    with pytest.raises(SessionError):
        store.get(s.id)


def test_oversized_upload_is_rejected(store):
    """한 장이 tmpfs 를 다 먹지 못하게 한다."""
    with pytest.raises(SessionError) as e:
        store.create(b"y" * 2048, "png")
    assert e.value.status == 413
    assert len(store) == 0


def test_session_count_is_bounded(store):
    """tmpfs 는 노드 RAM 이다. 거절이 노드를 죽이는 것보다 낫다."""
    for _ in range(3):
        store.create(b"x", "png")
    with pytest.raises(SessionError) as e:
        store.create(b"x", "png")
    assert e.value.status == 503


def test_ttl_reaps_abandoned_sessions(tmp_path):
    """클라이언트가 사라지는 것은 예외가 아니라 정상 경로다."""
    store = SessionStore(tmp_path / "s", ttl=0.05)
    s = store.create(b"x", "png")
    d = s.dir
    time.sleep(0.1)

    assert store.reap() == 1
    assert not d.exists()
    assert len(store) == 0


def test_touch_keeps_an_active_session_alive(tmp_path):
    """쓰고 있는 세션은 TTL 로 사라지면 안 된다."""
    store = SessionStore(tmp_path / "s", ttl=0.2)
    s = store.create(b"x", "png")
    for _ in range(4):
        time.sleep(0.08)
        store.get(s.id)  # 만질 때마다 수명이 갱신된다
    assert store.reap() == 0
    assert len(store) == 1


def test_close_leaves_no_photos_behind(tmp_path):
    store = SessionStore(tmp_path / "s")
    dirs = [store.create(b"x", "png").dir for _ in range(3)]
    store.close()
    assert not any(d.exists() for d in dirs)
    assert len(store) == 0


# --- 순서 -------------------------------------------------------------------


def test_same_session_operations_serialize(store):
    """같은 세션의 두 연산이 겹치지 않는다.

    이게 없으면 둘 다 현재 working 을 읽고 각자 쓰고, 나중에 쓴 쪽이 앞의
    편집을 지운다. 락이 그 경우 뒤에 온 쪽을 앞의 결과 위에서 돌게 만든다.
    """
    s = store.create(b"x", "png")
    overlap = []
    active = 0
    guard = threading.Lock()

    def op():
        nonlocal active
        with s.lock:
            with guard:
                active += 1
                overlap.append(active)
            time.sleep(0.05)
            with guard:
                active -= 1

    threads = [threading.Thread(target=op) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max(overlap) == 1, f"같은 세션 연산이 겹쳤다: {overlap}"


def test_different_sessions_do_not_block_each_other(store):
    """서로 다른 세션은 서로를 기다리지 않는다."""
    a = store.create(b"x", "png")
    b = store.create(b"y", "png")

    started = threading.Event()
    release = threading.Event()

    def hold():
        with a.lock:
            started.set()
            release.wait(2)

    t = threading.Thread(target=hold)
    t.start()
    assert started.wait(1)

    # a 가 잡혀 있어도 b 는 즉시 잡힌다
    assert b.lock.acquire(timeout=0.5), "다른 세션이 막혔다"
    b.lock.release()
    release.set()
    t.join()


def test_reaper_thread_cleans_without_requests(tmp_path):
    """요청이 없어도 치운다 - 사진을 들고 있는 시간은 짧을수록 좋다."""
    store = SessionStore(tmp_path / "s", ttl=0.05)
    reaper = Reaper(store, interval=0.05)
    s = store.create(b"x", "png")
    reaper.start()
    try:
        deadline = time.time() + 3
        while time.time() < deadline and s.dir.exists():
            time.sleep(0.05)
        assert not s.dir.exists(), "reaper 가 치우지 않았다"
    finally:
        reaper.stop()
