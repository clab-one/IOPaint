"""편집 세션 저장소.

FOLIO fork 가 추가한 모듈이다. upstream 에는 없다.

왜 필요한가 - upstream 의 `/api/v1/inpaint` 는 무상태다. 요청마다 원본 전체를
올리고 결과를 받는다. 사진 한 장을 여러 번 고치는 편집기에서는 그게 매번
수 MB 를 다시 올린다는 뜻이고, 더 나쁜 것은 **순서가 정의되지 않는다**는
점이다. 두 번째 지우기가 첫 번째 결과 위에서 일어나야 하는지 원본 위에서
일어나야 하는지 서버가 알 방법이 없다. 클라이언트가 결과를 받아 다시 올리는
방식으로 흉내낼 수는 있지만, 그러면 왕복마다 전체 이미지가 오간다.

세션은 그 둘을 서버 쪽에 둔다.

    POST   /v1/edit-sessions            원본 1회 업로드 -> id
    POST   /v1/edit-sessions/{id}/erase 마스크만 -> working 갱신
    GET    /v1/edit-sessions/{id}/result 현재 working
    DELETE /v1/edit-sessions/{id}       즉시 삭제

## 어디에 사는가

tmpfs(`FOLIO_SESSION_DIR`, 배포에서 `medium: Memory` 12Gi)다. 사진은 디스크에
남지 않는다. TTL 이 지나거나 클라이언트가 `Done` 하면 사라진다. DB 에 사진
바이트를 넣지 않는다 - 계획 11 절.

## 순서

한 세션의 연산은 **직렬화한다**. 같은 세션에 지우기 두 개가 동시에 오면 둘 다
현재 working 을 읽고 각자의 결과를 쓰려 하고, 나중에 쓴 쪽이 앞의 편집을
지운다. 세션마다 락을 두어 그 경우 뒤에 온 쪽이 앞의 결과 위에서 돈다.
서로 다른 세션은 서로를 기다리지 않는다.

락은 자원이므로 세션과 함께 사라져야 한다 - 세션 수만큼만 존재한다.

## 무엇을 막는가

- 세션 수 상한: tmpfs 는 노드 RAM 이다. 무한히 만들게 두면 노드가 죽는다.
- 원본 크기 상한: 한 장이 tmpfs 를 다 먹지 못하게 한다.
- TTL: 클라이언트가 `Done` 을 부르지 않고 사라지는 경우가 정상 경로다.
"""

from __future__ import annotations

import os
import secrets
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

# 세션 하나가 들고 있는 파일. 이름을 고정해 두면 디렉터리만 지우면 끝난다.
ORIGINAL = "original"
WORKING = "working.png"


class SessionError(Exception):
    """호출자가 HTTP 상태로 옮길 수 있는 실패."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclass
class Session:
    id: str
    dir: Path
    created: float
    touched: float
    ops: int = 0
    # 이 세션의 연산을 직렬화한다. 세션과 수명을 같이한다.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # 삭제는 이 락을 잡고 내린다. 진행 중인 연산은 락 안에서 이 값을 다시
    # 확인해야 한다 - 락을 기다리는 동안 세션이 사라졌을 수 있다.
    alive: bool = True

    @property
    def working(self) -> Path:
        return self.dir / WORKING


class SessionStore:
    """세션 디렉터리를 만들고, 재우고, 치운다.

    스레드 안전하다 - FastAPI 의 sync 핸들러는 스레드풀에서 돈다.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        *,
        ttl: float = 20 * 60,
        max_sessions: int = 32,
        max_bytes: int = 64 * 1024 * 1024,
    ):
        self.root = Path(root)
        self.ttl = ttl
        self.max_sessions = max_sessions
        self.max_bytes = max_bytes
        self._sessions: dict[str, Session] = {}
        self._guard = threading.Lock()  # _sessions 사전 자체를 지킨다
        self.root.mkdir(parents=True, exist_ok=True)

    # --- 수명 -------------------------------------------------------------

    def create(self, image: bytes, ext: str) -> Session:
        if len(image) > self.max_bytes:
            raise SessionError(
                413, f"image too large: {len(image)} > {self.max_bytes} bytes"
            )

        self.reap()  # 자리를 만들 수 있으면 만들고 센다

        with self._guard:
            if len(self._sessions) >= self.max_sessions:
                # tmpfs 는 노드 RAM 이다. 거절이 노드를 죽이는 것보다 낫다.
                raise SessionError(503, "too many sessions")
            sid = secrets.token_urlsafe(9)
            now = time.time()
            s = Session(id=sid, dir=self.root / sid, created=now, touched=now)
            self._sessions[sid] = s

        s.dir.mkdir(parents=True, exist_ok=True)
        (s.dir / f"{ORIGINAL}.{ext}").write_bytes(image)
        s.working.write_bytes(image)
        logger.info(f"session {sid}: 생성 ({len(image)} bytes, .{ext})")
        return s

    def get(self, sid: str) -> Session:
        with self._guard:
            s = self._sessions.get(sid)
            if s is None:
                raise SessionError(404, "session not found")
            s.touched = time.time()
            return s

    def _retire(self, s: Session, why: str) -> None:
        """세션 하나를 내린다. **반드시 세션 락을 잡고 지운다.**

        레지스트리에서 빼는 것만으로는 부족하다. 지우기가 진행 중이면 그
        스레드는 이미 `Session` 을 들고 있고 곧 working 을 쓴다. 디렉터리를
        먼저 지우면 사라진 자리에 쓰게 되고, 그 요청은 500 으로 끝난다.

        락을 잡으면 진행 중인 연산이 끝난 뒤에 지운다. 추론 한 번만큼 기다리는
        대신, 삭제가 조용히 성공하고 사진은 확실히 사라진다.
        """
        with s.lock:
            s.alive = False
            shutil.rmtree(s.dir, ignore_errors=True)
        logger.info(f"session {s.id}: {why} (연산 {s.ops}회)")

    def delete(self, sid: str) -> None:
        with self._guard:
            s = self._sessions.pop(sid, None)
        if s is None:
            raise SessionError(404, "session not found")
        # 레지스트리에서 이미 빠졌으므로 새 연산은 시작될 수 없다(get 이 404).
        # 남은 것은 진행 중인 연산 하나뿐이고, 락이 그것을 기다린다.
        self._retire(s, "삭제")

    def reap(self) -> int:
        """TTL 이 지난 세션을 치운다.

        클라이언트가 `Done` 을 부르지 않고 사라지는 것은 예외가 아니라 정상
        경로다 - 앱이 죽거나, 전화가 오거나, 네트워크가 끊긴다.
        """
        cutoff = time.time() - self.ttl
        with self._guard:
            dead = [s for s in self._sessions.values() if s.touched < cutoff]
            for s in dead:
                self._sessions.pop(s.id, None)
        for s in dead:
            self._retire(s, "TTL 만료로 정리")
        return len(dead)

    # --- 관찰 -------------------------------------------------------------

    def __len__(self) -> int:
        with self._guard:
            return len(self._sessions)

    def stats(self) -> dict:
        with self._guard:
            return {
                "sessions": len(self._sessions),
                "max_sessions": self.max_sessions,
                "ttl_seconds": int(self.ttl),
            }

    def close(self) -> None:
        """전부 치운다. 프로세스가 내려갈 때 사진을 남기지 않는다."""
        with self._guard:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for s in sessions:
            self._retire(s, "종료로 정리")


class Reaper:
    """주기적으로 `reap` 을 부르는 데몬 스레드.

    요청이 올 때만 치우면 아무도 안 오는 동안 tmpfs 가 붙잡혀 있다. 사진을
    들고 있는 시간은 짧을수록 좋다.
    """

    def __init__(self, store: SessionStore, interval: float = 60.0):
        self.store = store
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="session-reaper", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.store.reap()
            except Exception:  # 청소가 서버를 죽이면 안 된다
                logger.exception("session reaper failed")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
