# RVC Cover Maker

내 목소리 RVC 모델을 **학습**하고, 그 모델로 유튜브 곡을 **커버**로 만드는 로컬 웹 도구.
학습 탭(데이터 → `.pth`) + 커버 탭(링크 → 커버), 두 모드.

> ⚠️ 변환 결과물의 사용 책임은 본인에게 있습니다. 개인·비공개 용도 한정.

## 빠른 시작

기본 모델로 **Tom Holland**(공개 데모 음성)가 함께 설치돼, 클론 후 바로 커버를 만들 수 있습니다.
자기 목소리는 학습 탭에서 추가하세요.

**로컬 (macOS):**
```bash
git clone https://github.com/Thingamajigbo/rvc-cover-maker.git && cd rvc-cover-maker
# 사전: brew install uv ffmpeg sox
bash setup.sh                                   # 두 venv + 엔진 + Tom Holland (~10분)
.venv/bin/python -m uvicorn main:app --app-dir backend --port 8000   # → localhost:8000
```

**빌린 GPU 서버 (RunPod 등) — 터미널에 한 줄:**
```bash
curl -fsSL https://raw.githubusercontent.com/Thingamajigbo/rvc-cover-maker/main/gpu-bootstrap.sh | bash
```
HTTP 8000 노출 후 `https://<팟ID>-8000.proxy.runpod.net` 접속. 서버에선 📁파일 업로드 사용
(유튜브는 데이터센터 IP 차단). 처음 한 번은 환경 자동구성으로 ~15분. 다 쓰면 팟 **Stop**.

밑바닥부터 재현하는 전체 빌드 가이드는 [`BUILD_SYSTEM_PROMPT.md`](BUILD_SYSTEM_PROMPT.md) — Claude에 넣으면 됩니다.

## 현재 상태

- ✅ **Phase 0**: 백엔드 뼈대 + 로컬 컴퓨트 검증 (FastAPI, device 자동감지)
- ✅ **사전**: AICoverGen 엔진 설치 (Mac/CPU 적응, `setup.sh`로 재현)
- ✅ **Phase 1**: 커버 end-to-end (유튜브→분리→RVC→합치기) 검증
- ✅ **Phase 2~3**: 웹 도구 MVP — 큐+진행률, 모델 업로드/선택, 가사(LRCLIB)
- ✅ **프로덕션화**: SaaS급 UI 재설계(**Resona**), 하네스 엔지니어링(스톨/취소/타임아웃/세분화 진행률)
- ✅ **Phase 4~5**: 목소리 학습 탭 — 데이터검사·전처리→f0→feature→학습→인덱스, 커버 탭과 GPU 공유, pytest 25개 GREEN
  - 학습 엔진(RVC-WebUI)은 커버 venv 재사용. **실제 학습은 GPU 권장**(CPU는 매우 느림). 전처리까지 로컬 검증 완료, 풀 학습은 GPU에서.

### 하네스 (80% 멈춤 해결)
RVC 단계의 tqdm은 `\r`로 출력 → 기존엔 80%에서 멈춰 보였음. `backend/harness.py`가
`\r`/`\n` 모두 파싱해 `n/6` 세그먼트를 80→95%로 흘리고, 스톨 감지·취소·하드 타임아웃을 제공.

### 테스트
```bash
.venv/bin/python -m pytest tests/ -q      # 25 passed (하네스/API/학습, 엔진은 목킹)
```

### 디바이스 주의 (Apple Silicon)
MPS(Metal)는 이 RVC 코드에서 **데드락** → 엔진은 **CPU 강제**(`RVC_FORCE_CPU=1`).
65초 곡 ≈ 140초(분리가 병목). 더 빠르게: CUDA GPU 서버로 이동(코드 그대로, device 자동감지).

## 환경

- macOS / Apple Silicon (로컬). GPU 서버로 옮길 때도 동일 코드 (device 자동감지: cuda/mps/cpu)
- Python 3.10 (uv가 관리, 시스템 파이썬 안 건드림)

## 실행

```bash
cd ~/rvc-cover-maker
cp .env.example .env            # 최초 1회
.venv/bin/python -m uvicorn main:app --app-dir backend --port 8000
# → 브라우저로 http://127.0.0.1:8000 접속
#   커버 탭: 유튜브 링크 + 모델 선택 + 피치 → [커버 생성]
```

### API
- `POST /api/jobs` `{youtube_url, model_name, pitch, index_rate, protect, rms_mix_rate, f0_method, output_format}` → `{job_id}`
- `GET /api/jobs/{id}` → `{status, step, progress, logs, result}` (1.5초 폴링)
- `GET /api/models` · `POST /api/models` (multipart: name + .pth/.index/.zip)
- `GET /api/lyrics?title=&artist=` (LRCLIB) · `GET /api/gpu`

컴퓨트 장치만 빠르게 확인:

```bash
.venv/bin/python backend/device.py
```

## 구조

```
backend/   FastAPI (main.py, device.py, 이후 jobs/train/lyrics/models)
frontend/  학습/커버 탭 UI (이후)
datasets/  학습용 목소리 데이터 (gitignore)
models/    .pth/.index (gitignore)
outputs/   생성된 커버 (gitignore)
external/  RVC-WebUI, AICoverGen (setup.sh가 설치, gitignore)
```
