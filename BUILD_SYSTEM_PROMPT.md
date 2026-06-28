# RVC 커버 메이커 — 빌드 시스템 프롬프트

> 이 문서 전체를 Claude(Claude Code 권장)에게 시스템 프롬프트/첫 메시지로 넣으면, 처음 보는
> 사람도 똑같은 결과물을 만들 수 있도록 작성됨. 우리가 실제로 부딪힌 함정과 해결책이 전부 들어있다.
> Claude는 이 가이드를 "이미 검증된 사실"로 취급하고, 추측으로 우회하지 말 것.

---

## 0. 너의 임무

내 목소리(또는 임의 인물 목소리)를 **RVC 모델로 학습**하고, 그 목소리로 **노래 커버**를 만드는
**로컬 우선 + 필요시 GPU 서버** 웹도구를 만든다. 두 개의 탭:

- **커버 탭**: 유튜브 링크 또는 오디오 파일 → 보컬분리 → RVC 음성변환 → 믹스 → 완성 커버
- **학습 탭**: 오디오 데이터셋 업로드 → 전처리 → f0 → feature → train → faiss index → 커버용 모델 등록

**핵심 방침**
- 코드는 디바이스 자동감지(`cuda`/`mps`/`cpu`)로 작성 → 로컬(Mac)에서 만들고 GPU 서버로 옮겨도 재작성 0.
- 1인 사용 로컬 도구다. **과설계 금지**(YAGNI). 인메모리 큐·폴링이면 충분, DB/Redis/k8s 쓰지 마라.
- 무거운 두 엔진은 **건드리지 말고 subprocess로 호출**한다. 우리 코드와 dep이 충돌하므로 격리한다.

---

## 1. 아키텍처 (이대로 따라라)

```
[브라우저: vanilla JS] ──HTTP──> [FastAPI 백엔드] ──subprocess──> [엔진 venv]
   index.html/app.js/style.css        backend/*.py                ├ AICoverGen (커버)
                                   인메모리 큐 + 단일 워커          └ RVC-WebUI (학습)
```

- **두 개의 격리된 venv** (dep 충돌 때문에 필수):
  - 백엔드 `.venv`: fastapi + torch 2.5 + numpy 1.26
  - 엔진 `external/AICoverGen/.venv`: torch 2.2.2(Mac) / CUDA(서버) + numpy 1.23.5 + gradio 3.39 + fairseq
  - 학습 엔진(RVC-WebUI)은 **AICoverGen venv를 재사용**한다(같은 torch/fairseq/faiss). 추가 dep은 `av`/tensorboard뿐.
- **Python 3.10을 `uv`로 격리**한다. 시스템 파이썬(3.13 등) 건드리지 마라: `uv venv --python 3.10 .venv`.
- **백엔드가 엔진을 CLI subprocess로 호출**한다(임포트 X). 출력 스트림을 파싱해 진행률을 만든다.

### 파일 구조
```
backend/
  main.py        FastAPI 엔드포인트 (커버/학습/모델/가사/gpu)
  jobs.py        인메모리 큐 + 단일 워커 + 취소/재시도/스톨
  aicovergen.py  커버 엔진 subprocess 래퍼 + engine_lock + env
  train.py       학습 5단계 subprocess (preprocess→f0→feature→train→index)
  harness.py     stream_process: \r/\n 둘 다 파싱, 스톨/취소/타임아웃
  device.py      cuda/mps/cpu 자동감지 리포트
  models.py      모델 목록/업로드 (rvc_models/<name>/*.pth[+*.index])
  lyrics.py      LRCLIB 가사 검색
frontend/        index.html / app.js / style.css (커버·학습 탭 전환)
patches/         aicovergen-mac.patch, rvc-webui-mac.patch (디바이스/호환 패치)
tests/           pytest (큐/하네스/학습 데이터검사) — 변경 후 항상 GREEN 유지
requirements.txt 백엔드 전용 (엔진 dep은 setup.sh가 따로)
setup.sh         전체 재현 설치 (두 venv + 엔진 clone + 패치 + 모델 다운로드)
run.sh           서버 시작 (서버 환경에선 반드시 이걸로)
```

---

## 2. 스택 (정확한 핀 — 바꾸지 마라, 이유 있음)

**백엔드 `requirements.txt`**
```
fastapi==0.115.6
uvicorn[standard]==0.34.0
python-multipart==0.0.20      # 파일 업로드
requests==2.32.3
python-dotenv==1.0.1
numpy==1.26.4                 # <2 여야 옛 RVC 코드가 np.int 등에서 안 깨짐
soundfile==0.14.0             # 학습 데이터 무음/클리핑 분석
faiss-cpu==1.8.0              # retrieval index
packaging>=23                 # faiss가 packaging.version.Version 임포트
torch==2.5.1                  # 디바이스 감지 + 서버에선 cuDNN9를 엔진에 빌려줌
torchaudio==2.5.1
```

**엔진 venv (setup.sh가 설치)**
```
torch==2.2.2 torchaudio==2.2.2   # Mac/MPS. 서버는 CUDA 빌드로 치환
numpy==1.23.5                    # 맨 마지막에 다시 핀(다른 패키지가 끌어올림)
fairseq==0.12.2                  # ⚠️ git 태그에서 빌드 (아래 함정 1)
gradio==3.39.0 gradio_client==0.3.0
faiss-cpu==1.8.0 librosa==0.9.1 scipy==1.11.1 soundfile==0.12.1
praat-parselmouth pedalboard==0.7.7 pydub pyworld torchcrepe yt-dlp sox
onnxruntime (CPU) / onnxruntime-gpu (nvidia-smi 있으면)
av tensorboard tensorboardX     # 학습(RVC-WebUI)용
```

**엔진 repo**
- 커버: `https://github.com/SociallyIneptWeeb/AICoverGen`
- 학습: `https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI` (= RVC-WebUI)
- 사전학습(40k v2): HuggingFace `lj1995/VoiceConversionWebUI`의 `pretrained_v2/f0G40k.pth`, `f0D40k.pth`

---

## 3. ⚠️ 함정 모음 (이게 이 문서의 핵심 — 전부 우리가 실제로 당한 것)

### 함정 1 — fairseq 0.12.2는 PyPI sdist로 빌드 실패
PyPI sdist에 C++ 소스(`balanced_assignment.cpp`)가 빠져있다. **git 태그에서 빌드**해라:
```bash
python -m pip install --no-build-isolation \
  "fairseq @ git+https://github.com/facebookresearch/fairseq.git@v0.12.2"
```
전제: 먼저 `pip<24.1`, `setuptools<60`, `wheel`, `numpy==1.23.5`, `cython`, `torch==2.2.2` 설치. 그리고 **uv의 빌더는 fairseq C++ glob에서 실패**하니 venv 안에 진짜 pip을 넣고 `python -m pip`로 빌드해라.

### 함정 2 — macOS: torch+faiss+sklearn이 각자 libomp.dylib 번들 → OpenMP 데드락
RVC의 CPU conv(rmvpe)가 OpenMP 배리어에서 **0% CPU로 무한 행**. 짧은 곡은 운으로 통과, 긴 곡은 확정 멈춤.
`KMP_DUPLICATE_LIB_OK`/`OMP_NUM_THREADS`로는 **못 막는다.** 진짜 해결 = **libomp 단일화**:
```bash
SP=".venv/lib/python3.10/site-packages"
TORCH_OMP="$(cd "$SP" && pwd)/torch/.dylibs/libomp.dylib"
for p in faiss sklearn; do
  d="$SP/$p/.dylibs/libomp.dylib"
  [ -f "$d" ] && [ ! -L "$d" ] && mv "$d" "$d.bak" && ln -s "$TORCH_OMP" "$d"
done
```
(faiss/sklearn의 libomp를 torch 것으로 심링크 → 런타임 하나만 로드.) 추가 방어: 엔진 env에
`KMP_DUPLICATE_LIB_OK=TRUE` + `*_NUM_THREADS=4`, 그리고 **engine_lock**(fcntl.flock, 머신 전역 단일 엔진).

### 함정 3 — MPS(Apple GPU)는 이 옛 RVC 코드에서 데드락
MPS로 돌리면 8초 클립도 10분+ 안 끝남. **Mac에선 CPU 강제**해라(`RVC_FORCE_CPU=1` 기본 → 패치에서
`torch.backends.mps.is_available=lambda:False`). 디바이스 자동감지 코드는 그대로 둬서 **CUDA 서버로 옮기면 자동으로 GPU 사용**.

### 함정 4 — 업로드 mp3/m4a를 soundfile이 못 읽음
데이터검사(`soundfile`)가 mp3/m4a 디코드 못 한다. 도착 즉시 **ffmpeg로 wav 변환**(`_ensure_wav`). `ffmpeg`/`sox`는 시스템 패키지(brew/apt)로 깔아야 한다.

### 함정 5 — matplotlib 3.10이 `tostring_rgb()` 제거
RVC `utils.py`가 `tostring_rgb()` 호출 → `buffer_rgba()`로 패치(`rvc-webui-mac.patch`).

### 함정 6 — extract_f0_rmvpe.py가 `device="cuda"` 하드코딩
Mac에서 깨짐 → `cuda` 가용 시 cuda, 아니면 cpu로 패치.

### 함정 7 — RVC train.py 체크포인트 누적으로 디스크 폭발
`-l 0`이면 epoch마다 G/D 체크포인트가 쌓여 디스크가 찬다 → **`-l 1`**(최신만 유지). `-se 5`(5ep마다 저장), `-sw 1`.

### 함정 8 — 학습이 click_train의 보조 파일을 요구
`train.py`로 직접 호출하면 `filelist.txt` + `config.json`이 없어서 실패. WebUI의 click_train이 만드는 걸
**복제**해서 미리 만들어라(`_prepare_train_files`). 무음 참조 2줄(mute)도 filelist에 추가.

### 함정 9 — train.py가 최종 `.pth`를 상대경로 `assets/weights/<name>.pth`로 저장
이 디렉터리 없으면 100ep 다 돌고 **마지막 저장에서 깨진다**(`Parent directory assets/weights does not exist`).
setup.sh에서 `mkdir -p .../assets/weights` 미리 만들어라. (이미 학습이 끝났는데 이 에러가 나면 재학습 말고
G 체크포인트에서 추출: `process_ckpt.extract_small_model(G경로, name, "40k", 1, info, "v2")` → `assets/weights/<name>.pth` 생성.)

### 함정 10 — 진행률이 80%에서 멈춰 보임
RVC tqdm이 `\r`로 출력 → 줄단위 read가 막힘. `harness.stream_process`가 **`\r`/`\n` 둘 다** 파싱하게 만들고,
RVC 구간(0.80~0.95)을 tqdm "n/m"으로 채워라. + 스톨 감지/취소/하드 타임아웃.

---

## 4. GPU 서버(RunPod) 플레이북 — 우리가 제일 많이 당한 곳

> 서버는 데이터센터 IP라 행동이 로컬과 다르다. 아래는 전부 코드/스크립트에 박아라.

### 4-1. 배포
- 이미지 CUDA 버전이 머신과 안 맞으면 컨테이너가 안 뜬다(예: cu128 이미지 + 12.7 머신) → 호환 이미지 선택.
- HTTP 8000 노출, `--host 0.0.0.0`. **SSH는 "SSH over exposed TCP"** (`root@IP -p PORT -i key`)가 안정적
  (웹터미널/Jupyter는 잘 죽음). **재시작하면 SSH 포트가 바뀐다** — Connect 다이얼로그에서 새 명령 받아라.
- Linux/CUDA 추가 버그: ① fairseq git태그 빌드(함정1) ② `tensorboard`/`tensorboardX` 누락 ③ feature device
  하드코딩 → 자동감지 ④ `-l 1`(함정7) ⑤ `packaging` 누락.

### 4-2. 유튜브 다운로드 막힘
데이터센터 IP는 유튜브 봇차단("Sign in to confirm..."). **서버에선 📁파일 업로드 전용**. 유튜브는 로컬(Mac)에서만.

### 4-3. ⭐ 네트워크 볼륨(MooseFS)이 진짜 적이다
RunPod의 `/workspace`는 MooseFS 네트워크 볼륨이다. **큰 파일 읽기/쓰기가 간헐적으로 FUSE에서 멈추고**
(`torch.save`·libsndfile·사전학습 모델 읽기가 행), **쿼터도 빡빡**(~20G인데 CUDA torch venv 두 개로 참).

**근본 해결 = 무거운 건 로컬 컨테이너 디스크로:**
- 중간 산출물(커버 stem, 학습 feature/checkpoint)은 `run.sh`가 로컬(`/tmp/rvc_scratch`)로 심링크.
- **재다운로드 가능한 무거운 트리**(venv 2개 + 사전학습/베이스 모델 `RVC-WebUI/assets`)도 로컬 디스크로
  심링크. `/workspace`엔 코드 + 학습된 목소리 모델 + 출력만 남긴다. → 18.8G가 1.4G로 줄고 멈춤도 사라진다.
- 단, 컨테이너 디스크는 **팟 재시작 시 전부 비워진다**(venv뿐 아니라 `uv`·`ffmpeg`·`sox`까지). 그래서
  `run.sh`를 **자가복구 부트스트랩**으로: venv 없으면 → apt(ffmpeg/sox) + uv 재설치 + setup.sh 재빌드(~15분) + 서버.
  대가: 재시작 후 첫 기동이 느림. 이게 "볼륨 최대 감축"의 트레이드오프다.
- **GPU 보컬분리**(onnxruntime-gpu)는 cuDNN9이 필요한데 엔진 venv엔 없다 → **백엔드 torch(2.5+cu124)의
  cuDNN9를 `LD_LIBRARY_PATH`로 빌려준다**(새 설치 X, 쿼터 회피). nvidia-smi 있으면 onnxruntime-gpu 설치.

### 4-4. 셸 함정 (자동화할 때)
- ssh 원격 명령 문자열에 `run.sh`가 들어있는데 `pkill -f run.sh`를 같이 쓰면 **자기 ssh 세션을 죽인다**.
  pkill 패턴이 자기 명령줄과 겹치지 않게 하라(PID로 죽이거나 정확한 이름 사용).
- 멈춘 학습 워커의 cmdline은 `python -c from multiprocessing.spawn...`이라 `pkill -f train.py`로 안 잡힌다. PID로 죽여라.
- MooseFS가 wedge되면 `dd`/`du` 같은 큰 읽기가 D-state(취소불가)로 ssh를 멈춘다. 큰 읽기 테스트 자제, 메타데이터(ls)로 확인.
- 긴 작업은 `nohup ... >log 2>&1 &`로 detached 실행 후 별도 세션에서 로그 폴링. 재시작에도 살리려면 로그파일 기준으로 추적.

---

## 5. 빌드 순서 (이 순서로)

```
1. 도구 확인: uv, ffmpeg, sox  → verify: command -v 통과
2. 백엔드 venv + requirements   → verify: uvicorn import 됨
3. AICoverGen clone + 엔진 venv (함정1·2 적용) + 베이스 모델 다운로드
                                → verify: src/download_models.py 성공
4. 패치 적용 (디바이스/호환)     → verify: git apply --reverse --check 통과(멱등)
5. 커버 1회 end-to-end          → verify: 공개 데모 모델로 wav 1개 생성
6. RVC-WebUI clone (+av/tensorboard) + 사전학습 다운로드 + assets/weights mkdir(함정9)
7. 학습 5단계 + faiss index     → verify: 짧은 데이터셋으로 .pth + .index 생성, 커버탭 목록에 등장
8. 프론트(2탭) + 큐/하네스 + pytest GREEN
```

setup.sh는 **멱등**하게(`[ -x .venv/bin/python ] ||` 식) 작성해 재실행 안전하게.

---

## 6. 실행

```bash
# 로컬 (Mac) — 유튜브 가능
.venv/bin/python -m uvicorn main:app --app-dir backend --port 8000   # → localhost:8000

# 서버 (RunPod) — 반드시 이걸로 (로컬 심링크·자가복구 포함). 파일 업로드만.
bash run.sh        # HOST/PORT/RVC_LOCAL/RVC_SCRATCH 환경변수로 조절
```

---

## 7. 완료 기준 (이게 충족돼야 끝)

- [ ] 커버: 파일 업로드 → 진행률이 0→100% 끊김없이 → 재생/다운로드 가능한 wav
- [ ] 학습: 데이터셋 업로드 → 5단계 진행 → `<name>.pth` + `.index` 생성 → `/api/models`에 등장
- [ ] 큐: 한 번에 한 작업(단일 워커), 취소/재시도/스톨 표시 동작
- [ ] `pytest tests/` 전부 GREEN
- [ ] 서버: GPU 사용 확인(`/api/gpu` → `cuda`), 네트워크 볼륨 사용량 최소, 재시작 후 `bash run.sh`로 자동 복구

---

## 8. 운영 메모

- 모델 등록 = `external/AICoverGen/rvc_models/<name>/`에 `<name>.pth`(+`*.index`) 넣으면 끝(`models.py`가 폴더 스캔).
- 저작권 경고 UI는 띄우되, 생성물 배포 책임은 사용자에게 있다고 명시(서비스는 도구).
- GPU 서버는 시간당 과금($0.7/hr 수준) — **다 쓰면 반드시 Stop**.
- 참고 구현(우리 repo): `github.com/Thingamajigbo/rvc-cover-maker` (코드만 공개, 모델/데이터 gitignore).
```
```
