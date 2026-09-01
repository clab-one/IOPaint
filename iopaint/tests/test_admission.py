"""입장 제어 계약.

실측으로 드러난 결함을 고정한다 - 512² 100건 동시 요청에서 6/100 만 성공하고
liveness 가 굶어 컨테이너가 kill 됐다. 원인은 무한 동시성이었다.
"""

import threading
import time

import pytest
from fastapi import HTTPException

from iopaint.admission import Admission


def test_rejects_beyond_capacity():
    """용량(inflight + max_queue)을 넘으면 즉시 503 이다. 무한 대기가 아니다."""
    admission = Admission(inflight=1, max_queue=2)
    passed, rejected = [], []

    def hit(i):
        try:
            with admission.admit():
                time.sleep(0.3)
                passed.append(i)
        except HTTPException as e:
            rejected.append(e)

    threads = [threading.Thread(target=hit, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(passed) == 3
    assert len(rejected) == 3
    assert all(e.status_code == 503 for e in rejected)
    # 클라이언트가 언제 다시 올지 알아야 한다.
    assert all(e.headers.get("Retry-After") for e in rejected)


def test_serialises_inflight():
    """inflight=1 이면 두 요청이 겹치지 않는다.

    OMP_NUM_THREADS 가 컨테이너 CPU 할당량을 전부 쓰도록 맞춰져 있어서,
    두 추론이 겹치면 스레드가 코어의 배수가 되고 둘 다 느려진다.
    """
    admission = Admission(inflight=1, max_queue=8)
    concurrent = 0
    peak = 0
    lock = threading.Lock()

    def hit(_):
        nonlocal concurrent, peak
        with admission.admit():
            with lock:
                concurrent += 1
                peak = max(peak, concurrent)
            time.sleep(0.05)
            with lock:
                concurrent -= 1

    threads = [threading.Thread(target=hit, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert peak == 1


def test_slot_is_released_on_error():
    """핸들러가 터져도 자리를 반납한다. 안 그러면 용량이 서서히 0 이 된다."""
    admission = Admission(inflight=1, max_queue=0)

    with pytest.raises(RuntimeError):
        with admission.admit():
            raise RuntimeError("boom")

    assert admission.present == 0
    with admission.admit():
        pass


def test_rejects_invalid_config():
    with pytest.raises(ValueError):
        Admission(inflight=0)
    with pytest.raises(ValueError):
        Admission(inflight=1, max_queue=-1)
