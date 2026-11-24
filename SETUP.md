# 설치 및 실행 방법 (GPU 머신 기준)

욕설·비속어를 순화하는 모델을 `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct` + LoRA로 학습한다.
학습은 12GB GPU 머신에서 한다. CPU만 있는 PC에서는 데이터 준비까지만 가능하다.

## 한 번에 실행
clone 후 `run.bat` 더블클릭, 또는:
```bash
python run_all.py                      # venv, torch(CUDA 자동 선택), 패키지, 데이터, 스모크 테스트
python run_all.py --skip-smoke         # 스모크 테스트 건너뛰기
python run_all.py --torch-tag cu121    # torch CUDA 태그 직접 지정 (자동 선택이 안 될 때)
```
끝난 단계는 건너뛰고, 실패하면 그 자리에서 멈춘다. 첫 실행은 EXAONE 베이스 모델(약 5GB)을 받느라 오래 걸린다.
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
- 시드 고정이라 어느 머신에서 돌려도 같은 split이 나온다. 기준 결과: train 14,588 / valid 1,658 / test 1,629쌍.

## 학습 설정 (`configs/exaone_lora.yaml`)
- LoRA r=16, lr 2e-4, 3 epoch, 실질 배치 32 (`batch_size 8 × grad_accum 4`).
- OOM(메모리 부족)이면 `batch_size: 4`, `grad_accum: 8`로 바꾼다.
- bf16을 지원하는 GPU면 bf16, 아니면 fp16으로 자동 선택된다.
- 프롬프트(`model.prompt`)도 이 파일에 있다.

## 결과물
- `outputs/exaone-lora/best/` : LoRA 어댑터 + 토크나이저 (수십 MB). 베이스 모델은 HF 캐시에 따로 있다.
- `outputs/exaone-lora/train_log.json` : loss 기록.
- `outputs/exaone-lora/eval_test.json` : type별 chrF, BLEU, 정상문 유지율, 욕설 잔존율.
- `outputs/exaone-lora/eval_test_all.csv` : 테스트 전 행의 입력/정답/출력. 엑셀에서 `type`으로 필터링해서 직접 읽어본다.
- 학습 중간에 죽으면 `best/`가 없고 `checkpoints/`에만 남는다.
- 모델은 git에 올리지 말고 `best/` 폴더를 직접 복사하거나 HF Hub에 따로 올린다.

## 문제 해결
- 모델 로딩 에러: `transformers` 버전부터 확인한다. EXAONE 3.5 커스텀 코드는 5.x용으로 갱신된 판이 필요하다.
- `nvidia-smi`를 못 찾음: NVIDIA 드라이버가 없거나 NVIDIA GPU가 아닌 것이다. 드라이버 설치 후 재부팅한다.

## 사용 전 확인
- K/DA 데이터는 CC-BY-NC-4.0 (비상업). 사용 전 저자 폼 작성을 요청하고 있다: https://forms.gle/WDECRuuH328jw93L6
- KOTOX는 MIT, EXAONE은 비상업(연구) 라이선스다.
