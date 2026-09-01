# folio-iopaint — 헤드리스 지우기 서비스.
#
# 웹 UI 와 diffusion 을 들어낸 fork 라 프런트엔드 빌드 단계가 없다.
# torch 는 CPU 휠만 받는다 - clab-cluster 노드에 GPU 가 없고 CUDA 런타임은
# 쓰지도 않으면서 이미지를 수 GB 불린다.
#
# 가중치는 아직 이미지에 굽는다(205MB). 최종 목표는 PVC 지만, PVC 를
# /models 에 마운트하는 순간 이 레이어가 가려지므로 전환은 세 걸음이다:
#
#   1) 이미지에 prefetch 모듈을 넣는다. bake 유지.          <- 지금
#   2) Deployment 가 PVC 를 붙이고 initContainer 가 이 레이어에서 복사한다.
#      (PVC 를 /pvc 에, seed 를 /models 로 - 그래서 가려지지 않는다)
#   3) bake 를 걷어낸다. PVC 는 이미 차 있어 md5 확인만 한다.
#
# 각 걸음이 단독으로 안전해야 한다. 겹치는 구간 없이 갈아타려다 네 번 깨뜨렸다.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/models

# opencv-python 의 런타임 의존. slim 에는 없다.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch 를 먼저, 별도 레이어로. 가장 크고 가장 덜 바뀐다.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2" torchvision

COPY requirements.txt setup.py README.md ./
COPY iopaint ./iopaint
RUN pip install -e .

RUN useradd --create-home --uid 10001 folio && mkdir -p /models && chown -R folio:folio /models
USER folio

# 가중치를 구워 넣는다. 빌드는 원자적이라 잘린 파일이 남지 않는다.
# prefetch 를 그대로 쓴다 - 런타임과 같은 코드가 같은 자리에 놓아야
# 2단계의 --seed-from 이 그 자리를 찾는다.
RUN python -m iopaint.prefetch lama

EXPOSE 8080

# 프로브 전용 경로. 작업 스레드풀을 거치지 않는다.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if b'true' in urllib.request.urlopen('http://127.0.0.1:8080/readyz',timeout=4).read() else 1)"

ENTRYPOINT ["iopaint", "start"]
CMD ["--model=lama", "--device=cpu", "--host=0.0.0.0", "--port=8080", "--model-dir=/models"]
