# 설치 및 실행 방법 (GPU 머신 기준)

욕설·비속어를 순화하는 모델을 Kanana-1.5-2.1B-Instruct + LoRA로 학습한다.
베이스 모델은 `kakaocorp/kanana-1.5-2.1b-instruct-2505` (Apache-2.0, Llama 구조)를 쓴다.
학습은 12GB GPU 머신에서 한다. CPU만 있는 PC에서는 데이터 준비까지만 가능하다.

## 한 번에 실행
clone 후 `run.bat` 더블클릭, 또는:
```bash
python run_all.py                      # venv, torch(CUDA 자동 선택), 패키지, 데이터, 스모크 테스트
python run_all.py --skip-smoke         # 스모크 테스트 건너뛰기
python run_all.py --torch-tag cu121    # torch CUDA 태그 직접 지정 (자동 선택이 안 될 때)
```
끝난 단계는 건너뛰고, 실패하면 그 자리에서 멈춘다. 첫 실행은 Kanana 베이스 모델(약 4.6GB)을 받느라 오래 걸린다.
본 학습과 평가는 포함되지 않으니 이어서 직접 실행한다:
```bash
.venv\Scripts\python src/train.py      # Linux: .venv/bin/python
.venv\Scripts\python src/evaluate.py
.venv\Scripts\python src/infer.py "씨발 존나 덥네"
```
Git Bash에서는 경로를 `.venv/Scripts/python`처럼 슬래시로 쓴다.

## 단계별 수동 실행
```bash
python -m venv .venv && .venv\Scripts\activate
nvidia-smi                                                    # 우측 상단 CUDA 버전 확인
pip install torch --index-url https://download.pytorch.org/whl/cu121   # 버전은 CUDA에 맞게
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"    # True 여야 함

python src/prepare_data.py                                    # 데이터 생성 (수십 초)
python src/train.py --max_steps 20 --max_samples 200 --output_dir outputs/smoke   # 스모크
python src/train.py                                           # 본 학습
python src/evaluate.py
```

## 데이터
- `prepare_data.py`가 HF에서 KOTOX와 K/DA를 받아 `data/raw/`에 저장하고, `data/processed/{train,valid,test}.jsonl`을 만든다.
- `data/`와 `outputs/`는 `.gitignore`에 들어 있어 git에 안 올라간다. clone한 곳에서 이 명령을 한 번 돌려 데이터를 새로 만든다.
- 시드 고정이라 어느 머신에서 돌려도 같은 split이 나온다. 기준 결과: train 20,205 / valid 1,898 / test 1,866쌍 (합성 쌍 포함).
- 합성 쌍(`type=synthetic`): `configs/lexicon.tsv`의 욕설 어휘를 정상 문장에 끼우거나 고정 틀에 넣어 만든다. 공개 데이터에 없는 욕설을 모델이 지우지 않고 순화하게 하려는 용도다. 15%는 한 문장에 욕설 2~3개를 넣은 쌍(`type=multi`)으로, 첫 번째 욕만 고치고 나머지를 남기는 문제를 줄이려는 것이다. 어휘를 늘리려면 이 파일에 줄을 추가하고 `data/` 폴더를 지운 뒤 `prepare_data.py`를 다시 돌린다. 끄려면 `data.augment.enabled: false`.
- 규칙 변형 쌍(`data.obfuscate`, `type=obf_aug`): `src/obfuscate.py`가 욕 문장의 입력을 KOTOX처럼 변형(받침 추가, 된소리, y모음, 원문자, 자모 분리, 한자·로마자, 숫자·X, 기호, 띄어쓰기 없애기)하고 정답은 그대로 둔다. train에만 넣어서 KOTOX 변형 test로 효과를 잰다.
- identity 늘리기(`data.identity_extra`): LLM 쌍의 정답을 identity로도 넣고, train의 identity를 `train_repeat`번 반복한다. 깨끗한 문장을 고치거나 뒷부분을 지우는 걸 줄이려는 용도다. train에만 적용해서 valid/test는 이전 결과와 비교할 수 있다.

## 학습 설정 (`configs/kanana_lora.yaml`)
- LoRA r=16, lr 2e-4, 3 epoch, 실질 배치 32 (`batch_size 8 × grad_accum 4`).
- OOM(메모리 부족)이면 `batch_size: 4`, `grad_accum: 8`로 바꾼다.
- bf16을 지원하는 GPU면 bf16, 아니면 fp16으로 자동 선택된다.
- 프롬프트(`model.prompt`)도 이 파일에 있다.

## 결과물
- `outputs/kanana-lora/best/` : LoRA 어댑터 + 토크나이저 (수십 MB). 베이스 모델은 HF 캐시에 따로 있다.
- `outputs/kanana-lora/train_log.json` : loss 기록.
- `outputs/kanana-lora/eval_test.json` : type별 chrF, BLEU, 정상문 유지율, 욕설 잔존율.
- `outputs/kanana-lora/eval_test_all.csv` : 테스트 전 행의 입력/정답/출력. 엑셀에서 `type`으로 필터링해서 직접 읽어본다.
- `outputs/kanana-lora/moderate_test.json`, `moderate_test_all.csv` : `src/eval_moderate.py` 결과. 봇 판단(`src/moderate.py`: (1차 필터) → 순화 → 자기 재검사(결과가 안 바뀔 때까지 다시 순화) → 단어 목록 안전장치)을 테스트셋에 돌려서 pass/rewrite/block 비율과 실제로 보이는 문장에 욕이 남은 비율(leak)을 잰다. 설정은 config의 `bot`. `--no-prefilter`로 1차 필터 없이 모든 문장을 모델에 넣은 경우와 비교할 수 있다.
- 학습 중간에 죽으면 `best/`가 없고 `checkpoints/`에만 남는다. 100 step마다 저장되며 `train.py --resume`으로 이어서 학습한다.
- 데이터를 다시 만들었으면 옛 `checkpoints/`는 지우고 처음부터 학습한다 (`--resume`은 옛 데이터 기준 체크포인트를 잇는다).
- 모델은 git에 올리지 말고 `best/` 폴더를 직접 복사하거나 HF Hub에 따로 올린다.

## 문제 해결
- 모델 로딩 에러: `configs/kanana_lora.yaml`의 `model.name`과 `lora.target_modules`가 짝이 맞는지 확인한다 (Llama 구조: q/k/v/o/gate/up/down_proj).
- `hidden size (1792) is not a multiple of the number of attention heads (24)`: transformers 5.4+의 config 검사가 Kanana를 잘못 막는 것이다. `src/llm_common.py` 맨 위의 우회 코드가 import 되는지 확인한다 (스크립트에서 `transformers`로 모델을 직접 부르기 전에 `llm_common`을 먼저 import).
- `nvidia-smi`를 못 찾음: NVIDIA 드라이버가 없거나 NVIDIA GPU가 아닌 것이다. 드라이버 설치 후 재부팅한다.

## 사용 전 확인
- K/DA 데이터는 CC-BY-NC-4.0 (비상업). 사용 전 저자 폼 작성을 요청하고 있다: https://forms.gle/WDECRuuH328jw93L6
- KOTOX는 MIT, 베이스 모델 Kanana-1.5는 Apache-2.0이다.
- `data/llm_pairs/`의 LLM 생성 쌍은 Qwen(Apache-2.0)으로 만든다. 출력물로 다른 모델 학습을 막는 라이선스(EXAONE 등)의 모델로는 만들지 않는다.
- K/DA가 비상업이라 학습한 모델은 CC-BY-NC-4.0으로 공개한다 (무료 공개·무료 봇은 가능, 상업적 사용은 불가). 상업적으로 쓰려면 K/DA를 빼고 다시 학습한다.
