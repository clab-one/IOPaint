"""추론을 어느 노드에서 돌릴지 고른다.

FOLIO fork 가 추가한 모듈이다. upstream 에는 없다.

두 개의 실행 자원이 있다.

    i7m  (클러스터 안)   torch CPU    800² 한 장에 2870ms
    M4   (LAN 건너)      Core ML      800² 한 장에 142ms, 왕복 포함 267ms

M4 가 20배 빠르지만 클러스터 밖에 있다. 꺼질 수도 있고, 네트워크가 끊길
수도 있다. 그래서 **선택은 하되 의존하지 않는다** - M4 가 답하지 않으면
i7m 이 그대로 처리한다. 사용자는 느려질 뿐 실패하지 않는다.

## 고르는 방법

라운드로빈은 틀린 답이다. 두 노드의 속도가 20배 차이 나므로 번갈아 보내면
느린 쪽이 병목이 된다. 남은 시간을 추정해서 더 빨리 끝나는 쪽을 고른다.

    estimatedFinish = (진행 중 + 대기) × 최근 p50

p50 은 실측으로 채운다. 처음에는 관측이 없으므로 씨앗값을 두고, 돌기
시작하면 실제 값이 그것을 밀어낸다. 노드가 느려지면(썸 스로틀, 다른 부하)
p50 이 올라가고 자연히 트래픽이 줄어든다 - 별도의 부하 조절이 필요 없다.

## M4 는 800×800 만 받는다

Core ML 모델을 그 크기로 변환했다. 그래서 원본을 통째로 보내지 않는다 -
마스크 bbox 로 정사각 ROI 를 떠서 800 으로 맞춰 보내고, 돌아온 결과를 원래
크기로 되돌려 **마스크 안쪽만** 원본에 합성한다. 마스크 밖은 한 픽셀도
건드리지 않는다. 리샘플링 자국이 사진 전체에 번지는 것을 막는 유일한 방법이다.

원본 사진 자체가 LAN 을 건너지 않는다는 뜻이기도 하다 - 계획 14 절.
"""

from __future__ import annotations

import os
import ssl
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np
from loguru import logger

TILE = 800  # M4 Core ML 모델의 고정 입력 크기. 워커가 422 로 거절한다.


class Backend(Protocol):
    """## 색 계약 — RGB in, RGB out.

    이걸 명시하지 않아 한 번 틀렸다. LaMa 는 BGR 을 돌려주고 Core ML 워커는
    PNG(=디코드하면 BGR)를 돌려주는데, 두 경로가 서로 다른 규약으로 나오면
    호출자는 어느 한쪽에서만 맞는 변환을 하게 된다 - M4 경로의 R 과 B 가
    뒤집혔다. 대칭인 회색조 테스트로는 잡히지 않는다.

    그래서 변환은 **각 백엔드 안에서** 끝낸다. 경계를 넘는 배열은 언제나
    RGB 다. 호출자는 다시 변환하지 않는다.
    """

    name: str
    #: 이 백엔드가 지키는 자원의 동시 사용 상한. 로컬은 CPU 라 1, 원격은
    #: 네트워크라 여러 개.
    slots: threading.BoundedSemaphore

    def healthy(self) -> bool:
        """**I/O 금지.** 요청 경로와 `/healthz` 에서 불린다."""
        ...

    def erase(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray: ...


@dataclass
class Stats:
    """최근 지연을 들고 있는다.

    평균이 아니라 p50 을 쓴다. 한 번의 이상치가 라우팅을 오래 왜곡하면 안 된다.
    """

    seed_ms: float
    window: int = 16
    _samples: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    inflight: int = 0

    def observe(self, ms: float) -> None:
        with self._lock:
            self._samples.append(ms)
            if len(self._samples) > self.window:
                self._samples.pop(0)

    def enter(self) -> None:
        """줄에 선다. **슬롯을 잡기 전에** 부른다 - `estimated_ms` 가
        "앞에 몇 개 있는가" 를 뜻하려면 대기 중인 것도 세야 한다."""
        with self._lock:
            self.inflight += 1

    def leave(self) -> None:
        with self._lock:
            self.inflight -= 1

    @property
    def p50(self) -> float:
        with self._lock:
            if not self._samples:
                return self.seed_ms
            s = sorted(self._samples)
            return s[len(s) // 2]

    def estimated_ms(self) -> float:
        """지금 보내면 언제 끝나는가."""
        return max(1, self.inflight + 1) * self.p50


def square_roi(
    mask: np.ndarray, shape: tuple[int, int], margin: float = 0.25
) -> tuple[int, int, int, int] | None:
    """마스크를 감싸는 정사각 영역. (y0, y1, x0, x1)

    여백을 준다 - LaMa 는 주변 문맥을 보고 채우므로 마스크에 딱 맞게 자르면
    채울 근거가 없어진다.
    """
    ys, xs = np.nonzero(mask > 127)
    if ys.size == 0:
        return None

    h, w = shape
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    side = max(y1 - y0, x1 - x0)
    side = int(side * (1 + 2 * margin))
    side = max(32, min(side, min(h, w)))  # 이미지를 벗어나는 정사각형은 없다

    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    ty0 = min(max(0, cy - side // 2), h - side)
    tx0 = min(max(0, cx - side // 2), w - side)
    return ty0, ty0 + side, tx0, tx0 + side


def composite_roi(
    image: np.ndarray, mask: np.ndarray, roi: tuple[int, int, int, int], filled: np.ndarray
) -> np.ndarray:
    """ROI 결과를 마스크 안쪽에만 되돌린다.

    ROI 전체를 덮어쓰면 800 으로 줄였다 늘린 흔적이 마스크 밖에도 남는다.
    지우기는 마스크 안에서만 일어나야 한다.
    """
    y0, y1, x0, x1 = roi
    out = image.copy()
    sub = out[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1] > 127
    sub[m] = filled[m]
    return out


class LocalBackend:
    """클러스터 안의 torch CPU. 언제나 있다."""

    name = "local"

    def __init__(
        self, model_manager, request_factory, seed_ms: float = 2900.0, slots: int = 1
    ):
        self._mm = model_manager
        self._request = request_factory
        self.stats = Stats(seed_ms=seed_ms)
        # **여기가 원래 입장 제어가 지키려던 자원이다.** OMP_NUM_THREADS 가
        # 컨테이너 할당량(8코어)을 이미 다 쓰도록 맞춰져 있어 두 벌을 겹치면
        # 서로 스로틀만 걸린다 - 512² 100건 동시 요청에 컨테이너가 죽었던
        # 사고의 원인이다(admission.py).
        #
        # 제한을 그것이 지키는 자원 옆에 둔다. 전역 세마포어로 모든 추론을
        # 직렬화하면 로컬 CPU 와 무관한 원격 호출까지 줄을 서서, 노드를
        # 하나 더 붙여도 처리량이 늘지 않는다 - 실측 1.12 req/s 고정.
        self.slots = threading.BoundedSemaphore(slots)

    def healthy(self) -> bool:
        return True

    def erase(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        t = time.time()
        out = self._mm(image, mask, self._request())
        self.stats.observe((time.time() - t) * 1000)
        # LaMa 는 BGR 을 돌려준다. 경계 계약은 RGB 이므로 여기서 맞춘다 -
        # 호출자가 백엔드마다 다른 변환을 하게 두지 않는다.
        return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2RGB)


class CoreMLBackend:
    """LAN 건너의 M4. 빠르지만 없어질 수 있다."""

    name = "coreml"

    def __init__(
        self,
        base_url: str,
        *,
        ca: str,
        cert: str,
        key: str,
        # 20 초는 너무 관대했다. 원격이 얼어붙는 순간 진행 중이던 요청 하나가
        # 타임아웃을 다 쓰고서야 로컬로 넘어간다 - 실측 21,014ms. 요청은
        # 성공하지만 그 지연이면 앱이 먼저 포기한다.
        #
        # 5 초는 정상 왕복(p50 290ms)의 17배, 슬롯이 꽉 찬 최악(약 1.5초)의
        # 3배다. 여유는 충분하고 이탈 감지는 4배 빨라진다.
        timeout: float = 5.0,
        seed_ms: float = 300.0,
        probe_timeout: float = 3.0,
        # 4 다. 1 로 줄여 봤다가 되돌렸다 - 재현 금지용으로 근거를 남긴다.
        #
        # 줄인 논거는 이랬다. 워커를 직접 때리면 겹쳐도 처리량이 거의 안 는다:
        #
        #     conc=1  5.43 req/s  p50  184ms      conc=4  6.30 req/s  p50  634ms
        #     conc=2  6.03 req/s  p50  330ms      conc=8  6.53 req/s  p50 1220ms
        #
        # Core ML 안에서 이미 직렬화되니 겹치는 것은 대기열을 워커 안에 숨겨
        # 관측 p50 만 부풀린다(경합 없는 184ms 대 /healthz 의 1110ms). 그
        # 부풀린 값이 estimated_ms = (inflight+1) x p50 을 밀어올려 느린
        # 로컬로 트래픽을 돌린다 - 그러니 슬롯을 1 로 두어 줄을 이쪽에
        # 세워야 한다.
        #
        # 클러스터 경유로 실측했더니 **양쪽 다 나빠졌다** (3회 일관):
        #
        #                 slots=4              slots=1
        #     conc=1      1.10 req/s           1.09 req/s
        #     conc=2      1.52                 1.41
        #     conc=4      2.18                 1.80
        #     conc=8      2.22  p50 2526ms     1.91  p50 4150ms
        #
        # 두 가지를 틀렸다.
        #
        # 하나. 워커 직접 측정이 실제 경로를 대표하지 않았다. 이 경로의 요청
        # 하나는 900ms 인데 추론은 그중 300ms 뿐이다 - 나머지는 ROI 추출,
        # PNG 인코드/디코드, 세션 직렬화, 네트워크다. 슬롯이 하나면 그 일이
        # 추론과 겹치지 못한다. 워커 직접 측정에서는 그 일을 클라이언트가
        # 했기 때문에 겹침의 값어치가 작아 보였다.
        #
        # 둘. p50 이 부풀어 로컬로 넘어가는 것은 결함이 아니라 부하 분산이다.
        # 느려진 노드에서 트래픽이 빠지는 것이 추정식의 목적이다. slots=1 은
        # conc=4 에서 12건 전부를 원격에 몰아 로컬을 놀렸고, 그래서 느렸다.
        #
        # 겹침이 가속기를 놀게 하지 않는 한 슬롯은 넉넉한 편이 낫다.
        slots: int = 4,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.stats = Stats(seed_ms=seed_ms)
        self.slots = threading.BoundedSemaphore(slots)
        self._ctx = ssl.create_default_context(cafile=ca)
        self._ctx.load_cert_chain(cert, key)
        # 워커 인증서는 IP SAN 으로 발급됐다. 호스트명 검증은 그대로 둔다 -
        # CA 가 우리 것이고 SAN 에 그 IP 가 들어 있다.
        self._healthy = False
        self._probe_timeout = probe_timeout

    # --- 건강 ---------------------------------------------------------------

    def healthy(self) -> bool:
        """**I/O 를 하지 않는다.** 배경 프로버가 갱신한 값을 읽기만 한다.

        여기서 왕복하면 안 되는 이유는 지연이 아니라 정지다. `/healthz` 는
        `async` 엔드포인트라 이벤트 루프에서 직접 돈다 - 그 안에서 블로킹
        `urlopen` 을 하면 원격이 응답하지 않는 동안 **이벤트 루프 전체가**
        멈춘다. 그러면 liveness 프로브(kubelet 기본 1초)가 시간 초과하고,
        3회면 멀쩡한 컨테이너가 재시작된다. 원격 노드 하나가 사라졌을 뿐인데
        로컬 폴백은커녕 파드가 죽는다 - 계획 16 절이 약속한 것의 정반대다.

        블랙홀(RST 가 아니라 무응답)일 때가 최악이다. 연결 거부는 즉시
        실패하지만 응답 없는 호스트는 타임아웃까지 붙잡는다.
        """
        return self._healthy

    def probe_once(self) -> bool:
        """실제로 왕복한다. 배경 스레드에서만 부른다."""
        try:
            import urllib.request

            with urllib.request.urlopen(
                f"{self.base_url}/internal/v1/health",
                timeout=self._probe_timeout,
                context=self._ctx,
            ) as r:
                ok = r.status == 200
        except Exception as e:
            if self._healthy:
                logger.warning(f"coreml 노드 이탈: {e}")
            ok = False
        else:
            if ok and not self._healthy:
                logger.info("coreml 노드 복귀")
        self._healthy = ok
        return ok

    # --- 추론 ---------------------------------------------------------------

    def erase(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        roi = square_roi(mask, image.shape[:2])
        if roi is None:
            return image  # 지울 것이 없다
        y0, y1, x0, x1 = roi

        crop = image[y0:y1, x0:x1]
        cmask = mask[y0:y1, x0:x1]
        side = y1 - y0

        # 800 으로 맞춘다. 마스크는 최근접으로 - 보간하면 경계가 회색이 되고
        # 워커 쪽 이진화에서 마스크가 번진다.
        r_img = cv2.resize(crop, (TILE, TILE), interpolation=cv2.INTER_AREA)
        r_mask = cv2.resize(cmask, (TILE, TILE), interpolation=cv2.INTER_NEAREST)

        t = time.time()
        out800 = self._post(r_img, r_mask)
        self.stats.observe((time.time() - t) * 1000)

        filled = cv2.resize(out800, (side, side), interpolation=cv2.INTER_CUBIC)
        return composite_roi(image, mask, roi, filled)

    def _post(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import urllib.request

        body, ctype = _multipart(
            [
                ("image", "roi.png", _png(image)),
                ("mask", "mask.png", _png(mask)),
            ]
        )
        req = urllib.request.Request(
            f"{self.base_url}/internal/v1/lama", data=body, method="POST"
        )
        req.add_header("Content-Type", ctype)
        with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as r:
            raw = r.read()
        got = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if got is None:
            raise RuntimeError("coreml 노드가 이미지가 아닌 것을 답했다")
        return cv2.cvtColor(got, cv2.COLOR_BGR2RGB)


def _png(arr: np.ndarray) -> bytes:
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("PNG 인코딩 실패")
    return buf.tobytes()


def _multipart(parts: list[tuple[str, str, bytes]]) -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    out = bytearray()
    for field_name, filename, data in parts:
        out += (
            f"--{b}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode()
        out += data + b"\r\n"
    out += f"--{b}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={b}"


class Scheduler:
    """더 빨리 끝나는 쪽으로 보낸다. 실패하면 로컬로 되돌린다."""

    def __init__(self, local: Backend, remote: Backend | None = None):
        self.local = local
        self.remote = remote
        # 원격을 가진 쪽이 그 생사 확인도 가진다. 만들되 띄우지는 않는다 -
        # 테스트가 스레드 없이 스케줄러를 쓸 수 있어야 한다.
        self.prober = Prober(remote) if isinstance(remote, CoreMLBackend) else None

    def start(self) -> None:
        """배경 프로버를 띄운다. 원격이 없으면 아무 일도 하지 않는다."""
        if self.prober is not None:
            self.prober.start()

    def choose(self) -> Backend:
        if self.remote is None or not self.remote.healthy():
            return self.local
        if self.remote.stats.estimated_ms() <= self.local.stats.estimated_ms():
            return self.remote
        return self.local

    def erase(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, str]:
        """(결과, 실제로 쓴 백엔드 이름).

        원격이 실패하면 로컬로 되돌린다. 사용자 편집 세션이 노드 하나 때문에
        깨지면 안 된다 - 계획 16 절.
        """
        chosen = self.choose()
        if chosen is self.local:
            return self._run(self.local, image, mask), self.local.name

        try:
            return self._run(chosen, image, mask), chosen.name
        except Exception as e:
            logger.warning(f"coreml 실패, 로컬로 되돌린다: {e}")
            chosen._healthy = False  # 다음 health 확인까지 보내지 않는다
            return self._run(self.local, image, mask), self.local.name

    @staticmethod
    def _run(backend: Backend, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # 줄부터 선다. 슬롯을 기다리는 동안에도 inflight 에 잡혀야 뒤따르는
        # 요청의 `estimated_ms` 가 이 대기를 본다 - 그래야 원격이 밀릴 때
        # 자연스럽게 로컬로 넘어간다.
        backend.stats.enter()
        try:
            with backend.slots:
                return backend.erase(image, mask)
        finally:
            backend.stats.leave()

    def stats(self) -> dict:
        out = {
            "local": {
                "p50_ms": round(self.local.stats.p50),
                "inflight": self.local.stats.inflight,
            }
        }
        if self.remote is not None:
            out["coreml"] = {
                "p50_ms": round(self.remote.stats.p50),
                "inflight": self.remote.stats.inflight,
                # 캐시된 값이다. 이 호출은 네트워크를 건드리지 않는다.
                "healthy": self.remote.healthy(),
                # 프로버가 죽으면 위 값이 마지막 판정에 얼어붙는다. 그러면
                # 원격이 죽어도 계속 보내고 매번 폴백해 두 배로 느려진다.
                "prober": self.prober.alive if self.prober is not None else False,
            }
        return out


class Prober:
    """원격 백엔드의 생사를 배경에서 갱신하는 데몬 스레드.

    Reaper 와 같은 이유로 스레드다: 요청이 올 때만 확인하면 그 확인 비용이
    요청 지연이 되고, 더 나쁘게는 `/healthz` 처럼 **이벤트 루프에서 도는**
    호출자가 원격 타임아웃만큼 통째로 멈춘다.

    `alive` 를 밖에서 볼 수 있게 둔다 - 데몬 스레드는 조용히 죽고, 죽으면
    `healthy` 가 마지막 값에 영원히 얼어붙는다. 그 상태로 원격이 죽으면
    모든 요청이 원격으로 갔다가 폴백하느라 두 배로 느려진다.
    """

    def __init__(self, backend: CoreMLBackend, interval: float = 5.0):
        self.backend = backend
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self._thread is not None:
            return
        # 첫 판정은 동기로 한 번 낸다. 안 그러면 기동 직후 interval 동안
        # 멀쩡한 원격을 죽은 것으로 보고 전부 로컬로 보낸다.
        self.backend.probe_once()
        self._thread = threading.Thread(
            target=self._run, name="coreml-prober", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.backend.probe_once()
            except Exception as e:  # 스레드가 죽으면 판정이 얼어붙는다
                logger.warning(f"coreml 프로브 실패: {e}")


def build_remote() -> CoreMLBackend | None:
    """환경이 갖춰져 있을 때만 원격을 만든다.

    인증서가 없으면 원격은 없다 - 로컬만으로 완전히 동작해야 한다. 개발
    노트북과 CI 에서도 같은 코드가 돌아야 하기 때문이다.
    """
    url = os.environ.get("FOLIO_COREML_URL")
    if not url:
        return None
    ca = os.environ.get("FOLIO_COREML_CA", "/pki/ca.crt")
    cert = os.environ.get("FOLIO_COREML_CERT", "/pki/client.crt")
    key = os.environ.get("FOLIO_COREML_KEY", "/pki/client.key")
    missing = [p for p in (ca, cert, key) if not os.path.exists(p)]
    if missing:
        logger.warning(f"coreml 인증서 없음 {missing} - 로컬만 쓴다")
        return None
    logger.info(f"coreml 백엔드 등록: {url}")
    return CoreMLBackend(url, ca=ca, cert=cert, key=key)
