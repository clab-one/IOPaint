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

여기에 FOLIO 의 edit-session API 를 얹었다:

| 메서드 | 경로 | 하는 일 |
|---|---|---|
| `POST` | `/v1/edit-sessions` | 원본을 tmpfs 에 올리고 세션을 연다 |
| `POST` | `/v1/edit-sessions/{sid}/erase` | 직전 결과 위에 지운다 (세션 단위 직렬화) |
| `GET` | `/v1/edit-sessions/{sid}/result` | 현재 결과 |
| `DELETE` | `/v1/edit-sessions/{sid}` | 닫는다 |
| `GET` | `/healthz` `/readyz` | 입장 제어·세션·백엔드 상태 |

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

## 플러그인 (wave 2-5, 배포됨)

| 플러그인 | 모델 | 하는 일 | 클러스터 실측 (512²) |
|---|---|---|---|
| `InteractiveSeg` | `mobile_sam` | 점 하나 → 마스크 | 785ms |
| `RemoveBG` | `briaai/RMBG-1.4` | 배경 제거 (RGBA) | 1019ms |
| `RealESRGAN` | `realesr-general-x4v3` | 확대 (x2 기준) | 2001ms |
| `GFPGAN` | `GFPGANv1.4` | 얼굴 복원 | 2255ms |

`restoreformer` 는 코드만 남기고 켜지 않았다. GFPGAN 과 같은 자리를 노리고
둘 다 켤 이유가 없다.

큰 모델을 고르지 않은 이유는 하나다 - 이 노드는 CPU 뿐이다. SAM 은 vit_h
대신 mobile_sam(2.4GB → 40MB), RealESRGAN 은 x4plus 대신 general-x4v3
(64MB → 5MB)다. 품질이 모자라면 각각 vit_b, x4plus 가 다음 후보고 바꾸는 데는
배포 인자와 프리페치 목록 두 줄이면 된다.

### 가중치는 한 곳에서만 온다

`iopaint/weights.py` 가 URL 과 md5 를 전부 들고 있고, 플러그인과
`iopaint.prefetch` 가 거기서 읽는다. **그 모듈은 아무것도 import 하지
않는다** - CI 의 `validate` 잡이 torch 없이 매니페스트를 대조해야 하기
때문이다. 못 도는 가드는 없는 가드다.

가중치는 이미지가 아니라 PVC(`folio-ai-models`, 20Gi)에 있고 initContainer 가
채운다. 현재 927MB.

`deploy/k8s/deployment.yaml` 에서 플러그인을 켜면 프리페치 목록에도 넣어야
한다. 빠뜨리면 파드가 **기동 중에** 받는다. `test_deploy_consistency.py` 가
그 어긋남을 CI 에서 떨어뜨린다 - 얼굴 복원처럼 가중치가 셋인 경우도 안다
(GFPGANv1.4 + facexlib 검출·파싱).

## 검증

```
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -e .
.venv/bin/python -m pytest iopaint/tests --ignore=iopaint/tests/test_plugins.py -k "not cuda and not mps"
```

`test_plugins.py` 는 플러그인 가중치를 내려받으므로 기본에서 뺀다.

CI 는 두 잡이다. `validate` 가 pytest+pyyaml 만 깔고 배포 불변식을 몇 초에
검사하고, 통과해야 `build` 가 돈다 - 실패하면 이미지가 만들어지지 않으므로
digest 가 바뀌지 않는다. 그 잡에서 도는 시험은 무거운 것을 import 하면 안
된다.

## 출처

- upstream: `Sanster/IOPaint` `main` `61a759fb` (2025-04-29, 아카이브됨)
- 참고: `daraskme/IOpaint` `modernize-2026` `27b6ea66` — fork 가 아니라 별도
  업로드라 merge 계보가 없다. 현대화 커밋은 참고만 한다
- 라이선스: Apache-2.0 (`LICENSE`)
