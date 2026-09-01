# folio-iopaint

FOLIO 의 AI 편집 서버. [Sanster/IOPaint](https://github.com/Sanster/IOPaint) 를
fork 해 **웹 UI 와 diffusion 계열을 전부 들어내고 지우기(LaMa) 하나만 남긴**
브랜치다 (`folio-vendor`). 라이선스는 upstream 그대로 Apache-2.0.

```
iopaint start --model=lama --device=cpu --host=0.0.0.0 --port=8080
```

정적 파일을 서빙하지 않는다. `/` 는 404 다.

## 남은 API

| 메서드 | 경로 | 하는 일 |
|---|---|---|
| `POST` | `/api/v1/inpaint` | 이미지 + 마스크 → 지운 이미지 |
| `GET` | `/api/v1/server-config` | 적재된 모델·플러그인 |
| `GET` `POST` | `/api/v1/model` | 현재 모델 조회·전환 |
| `POST` | `/api/v1/run_plugin_gen_mask` | 플러그인 마스크 생성 (segmentation) |
| `POST` | `/api/v1/run_plugin_gen_image` | 플러그인 이미지 생성 (upscale, restore, remove-bg) |
| `POST` | `/api/v1/switch_plugin_model` | 플러그인 모델 전환 |
| `POST` | `/api/v1/adjust_mask` | 마스크 확장/축소/반전 |
| `POST` | `/api/v1/save_image` | 출력 디렉터리에 저장 |

FOLIO 의 edit-session API(`/v1/edit-sessions/...`)는 이 위에 AI-003 에서
올린다. 여기 있는 것은 upstream 계약 그대로다.

## 들어낸 것

| | 왜 |
|---|---|
| `web_app/` 정적 프런트엔드, `/` 마운트 | FOLIO 의 화면은 iOS 앱이다 |
| `web_config.py` (gradio), `installer.py` | 서버에 설정 UI·플러그인 설치기를 두지 않는다 |
| `file_manager/`, `--input` 디렉터리 모드, `/api/v1/inputimage` | 서버는 사진을 보관하지 않는다. 세션은 tmpfs 다 |
| Stable Diffusion · SDXL · ControlNet · BrushNet · PowerPaint · AnyText · Kandinsky · PaintByExample · InstructPix2Pix | 생성이 아니라 지우기다 |
| LDM · ZITS · MAT · FcF · MIGAN · OpenCV2 · Manga · AnimeLaMa · anime_seg | 지우기 모델을 하나로 고정한다. 품질 기준은 LaMa |
| socket.io `diffusion_progress` 스트림 | LaMa 는 단일 forward 라 흘려보낼 step 이 없다 |
| `InpaintRequest` 의 prompt·sampler·seed·croper·extender·lcm-lora 등 | 지우기 경로가 읽지 않는 필드 |
| `model/utils.py` 의 스케줄러 어댑터와 StyleGAN 연산 1000여 줄 | 위 모델들이 쓰던 것 |
| `iopaint download` · `list` · `run`(배치) · `start-web-config` | 서버는 헤드리스로만 뜬다 |

`InpaintRequest.sd_keep_unmasked_area` 는 `keep_unmasked_area` 로 바꿨다.
diffusion 이 없는데 `sd_` 접두사를 남길 이유가 없다.

## 남긴 플러그인 계층

`interactive_seg`(SAM/SAM-HQ/SAM2) · `realesrgan` · `gfpgan` ·
`restoreformer` · `remove_bg` 는 그대로 뒀다. FOLIO 의 wave 2-5
(자동 물체 선택 → 화질 개선 → 얼굴 복원 → 배경 제거)가 이 위에 붙는다.

## 검증

```
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -e .
.venv/bin/python -m pytest iopaint/tests --ignore=iopaint/tests/test_plugins.py -k "not cuda and not mps"
```

`test_plugins.py` 는 플러그인 모델 가중치를 내려받으므로 wave 2 이후에 돈다.

## 출처

- upstream: `Sanster/IOPaint` `main` `61a759fb` (2025-04-29, 아카이브됨)
- 참고: `daraskme/IOpaint` `modernize-2026` `27b6ea66` — fork 가 아니라 별도
  업로드라 merge 계보가 없다. 현대화 커밋은 참고만 한다
- 라이선스: Apache-2.0 (`LICENSE`)
