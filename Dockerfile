# folio-iopaint — 헤드리스 지우기 서비스.
#
# 웹 UI 와 diffusion 을 들어낸 fork 라 프런트엔드 빌드 단계가 없다.
# torch 는 CPU 휠만 받는다 - clab-cluster 노드에 GPU 가 없고 CUDA 런타임은
# 쓰지도 않으면서 이미지를 수 GB 불린다.
#
# 가중치는 이미지에 굽지 않는다. wave 3~5(RealESRGAN·GFPGAN·SAM·RemoveBG)를
# 붙이면 GB 급이 되기 때문이다. PVC 에 두고 initContainer 가 채운다
# (iopaint/prefetch.py). 이미지는 코드만 담는다.

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

EXPOSE 8080

# 프로브 전용 경로. 작업 스레드풀을 거치지 않는다.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if b'true' in urllib.request.urlopen('http://127.0.0.1:8080/readyz',timeout=4).read() else 1)"

ENTRYPOINT ["iopaint", "start"]
CMD ["--model=lama", "--device=cpu", "--host=0.0.0.0", "--port=8080", "--model-dir=/models"]
