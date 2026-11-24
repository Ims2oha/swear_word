# 설치 및 실행 방법 (GPU 머신 기준)

학습은 12GB GPU 머신에서 한다. 이 PC(CPU)에서는 데이터 준비까지만 가능하다.

## 한 번에 실행
clone 후 `run.bat` 더블클릭, 또는:
```bash
python run_all.py              # venv, torch(CUDA 자동 선택), 데이터, 스모크, 학습, 평가
python run_all.py --skip-smoke # 스모크 테스트 건너뛰기
```
끝난 단계는 건너뛰고, 실패하면 그 자리에서 멈춘다. 아래는 단계별 수동 실행 방법이다.

## 1. 환경 만들기
```bash
git clone <이 폴더를 옮긴 곳>   # 또는 폴더째 복사
cd 욕감지\ 봇
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/Mac: source .venv/bin/activate
```

## 2. torch 설치 (CUDA 버전에 맞게)
`nvidia-smi` 우측 상단의 CUDA Version을 확인하고 맞는 걸 설치한다.
```bash
# CUDA 12.1 예시
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8 예시
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
설치 확인:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # True 여야 함
```

## 3. 나머지 패키지
```bash
pip install -r requirements.txt
```
`requirements.txt`의 `torch` 줄은 2번에서 이미 깔았으면 그대로 통과한다.

## 4. 데이터 준비 (CPU로도 가능, 인터넷 필요)
```bash
python src/prepare_data.py
```
- HF에서 KOTOX와 K/DA CSV를 받아 `data/raw/`에 저장.
- `data/processed/{train,valid,test}.jsonl`, `stats.json` 생성.
- `data/`와 `outputs/`는 `.gitignore`에 들어 있어서 git에 안 올라간다. GPU 머신에서 clone한 뒤 이 명령을 한 번 돌려 데이터를 새로 만든다 (수십 초).
- 시드 고정이라 어느 머신에서 돌려도 같은 split이 나온다. 기준 결과: train 14,588 / valid 1,658 / test 1,629쌍.

## 5. 학습
스모크 테스트 먼저 (몇 분):
```bash
python src/train.py --max_steps 50 --max_samples 500 --output_dir outputs/smoke
```
loss가 줄고 `outputs/smoke/best`가 생기면 정상. 이후 전체 학습:
```bash
python src/train.py
```
- 설정은 `configs/kobart.yaml` (lr 5e-5, epoch 10, batch 32, fp16 자동).
- 메모리 부족(OOM)이면 `train.batch_size`를 16으로 낮춘다.
- 결과: `outputs/kobart-detox/best/` (모델 + 토크나이저), `train_log.json`.

## 6. 평가
```bash
python src/evaluate.py
```
- `outputs/kobart-detox/eval_test.json` : type별 chrF, BLEU, 정상문 유지율, 욕설 잔존율.
- `outputs/kobart-detox/eval_test_samples.csv` : 100개 샘플 (직접 읽어서 확인).

## 7. 추론
```bash
python src/infer.py "씨발 존나 덥네"
python src/infer.py            # 대화형, 빈 줄이면 종료
```
학습한 `outputs/kobart-detox/best/` 폴더만 복사하면 CPU PC에서도 추론 가능하다. 모델은 git에 올리지 말고 직접 복사하거나 HF Hub 같은 곳에 따로 올린다.

## 사용 전 확인
- K/DA 데이터는 CC-BY-NC-4.0 (비상업). 사용 전 저자 폼 작성을 요청하고 있다: https://forms.gle/WDECRuuH328jw93L6
- KOTOX는 MIT.
