"""입장 제어.

FOLIO fork 가 추가한 파일이다. upstream 에는 없다.

왜 필요한가 - 실측으로 드러났다. 512² 요청 100건을 동시에 넣으면:

    ok 6/100, 나머지 94건 소실
    Readiness probe failed: context deadline exceeded
    Liveness  probe failed: context deadline exceeded
    Killing: Container iopaint failed liveness probe, will be restarted

무한 동시성이 원인이다. Starlette 은 sync 엔드포인트를 스레드풀에서 돌리고,
그 스레드 하나하나가 torch 추론을 띄운다. 컨테이너 CPU 할당량은 8코어인데
OMP_NUM_THREADS=8 짜리 추론이 여러 개 겹치면 스레드가 코어의 몇 배가 되고,
주기마다 스로틀에 걸려 전부 같이 느려진다. 프로브까지 같은 스레드풀에서
굶어 죽으면서 쿠버네티스가 멀쩡한 컨테이너를 죽인다.

그래서 두 가지를 강제한다.

    in-flight 상한   추론은 한 번에 하나. OMP_NUM_THREADS 가 이미 코어를
                     전부 쓰도록 맞춰져 있으므로 두 개를 겹치면 손해다.
    큐 상한          기다리는 줄에도 끝이 있다. 넘으면 즉시 503 + Retry-After.
                     무한정 기다리게 하는 것보다 거절이 정직하다.

계획 16 절의 "bounded queue → Retry-After" 가 이것이다.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

from fastapi import HTTPException
from loguru import logger


class Admission:
    """동시 추론 수와 대기열 길이를 제한한다."""

    def __init__(self, inflight: int = 1, max_queue: int = 8, retry_after: int = 5):
        if inflight < 1:
            raise ValueError("inflight >= 1")
        if max_queue < 0:
            raise ValueError("max_queue >= 0")
        self._sem = threading.BoundedSemaphore(inflight)
        self._capacity = inflight + max_queue
        self._lock = threading.Lock()
        self._present = 0
        self._retry_after = retry_after
        self.inflight = inflight
        self.max_queue = max_queue

    @property
    def present(self) -> int:
        """실행 중 + 대기 중."""
        with self._lock:
            return self._present

    @contextmanager
    def admit(self):
        """자리를 잡고 들어간다. 자리가 없으면 503."""
        with self._lock:
            if self._present >= self._capacity:
                logger.warning(
                    f"admission full: {self._present}/{self._capacity}, rejecting"
                )
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"busy: {self._present} in system, capacity {self._capacity}"
                    ),
                    headers={"Retry-After": str(self._retry_after)},
                )
            self._present += 1

        try:
            # 여기서 기다린다. 큐 상한을 이미 통과했으므로 무한 대기가 아니다.
            self._sem.acquire()
            try:
                yield
            finally:
                self._sem.release()
        finally:
            with self._lock:
                self._present -= 1
