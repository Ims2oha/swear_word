"""한 번에 실행: venv -> torch(CUDA) -> 패키지 -> 데이터 -> 스모크 -> 학습 -> 평가.

python run_all.py [--skip-smoke]
이미 끝난 단계(venv, torch, 데이터)는 건너뛴다. 실패하면 그 자리에서 멈춘다.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
# (최소 드라이버 CUDA 버전, torch wheel 태그) - 높은 것부터 시도
TORCH_TAGS = [((13, 0), "cu130"), ((12, 8), "cu128"), ((12, 6), "cu126"),
              ((12, 4), "cu124"), ((12, 1), "cu121"), ((11, 8), "cu118")]


def step(msg):
    print(f"\n{'=' * 60}\n[{msg}]\n{'=' * 60}", flush=True)


def run(cmd, check=True, **kw):
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=env, **kw)
    if check and r.returncode != 0:
        sys.exit(f"\n실패 (exit {r.returncode}): {' '.join(str(c) for c in cmd)}")
    return r


def cuda_ok() -> bool:
    return PY.exists() and run(
        [PY, "-c", "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"],
        check=False, capture_output=True).returncode == 0


def driver_cuda():
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return (int(m[1]), int(m[2])) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-smoke", action="store_true", help="스모크 테스트 건너뛰기")
    args = ap.parse_args()

    step("1/6 가상환경")
    if not PY.exists():
        run([sys.executable, "-m", "venv", VENV])
    run([PY, "-m", "pip", "install", "-q", "--upgrade", "pip"])

    step("2/6 torch (CUDA)")
    if cuda_ok():
        print("CUDA torch 이미 설치됨, 건너뜀")
    else:
        ver = driver_cuda()
        if ver is None:
            sys.exit("nvidia-smi 를 못 찾음. NVIDIA 드라이버가 설치된 GPU 컴퓨터에서 실행해라.")
        print(f"드라이버 CUDA {ver[0]}.{ver[1]}")
        for need, tag in TORCH_TAGS:
            if ver < need:
                continue
            r = run([PY, "-m", "pip", "install", "torch", "--index-url",
                     f"https://download.pytorch.org/whl/{tag}"], check=False)
            if r.returncode == 0 and cuda_ok():
                break
            print(f"{tag} 실패, 더 낮은 버전 시도")
        if not cuda_ok():
            sys.exit("torch 에서 CUDA 를 못 씀. 드라이버를 업데이트하거나 수동으로 torch 를 설치해라.")

    step("3/6 패키지")
    run([PY, "-m", "pip", "install", "-q", "-r", "requirements.txt"])

    step("4/6 데이터 준비")
    if (ROOT / "data/processed/train.jsonl").exists():
        print("data/processed 이미 있음, 건너뜀 (다시 만들려면 data 폴더 삭제)")
    else:
        run([PY, "src/prepare_data.py"])

    if not args.skip_smoke:
        step("5/6 스모크 테스트 (50 step)")
        run([PY, "src/train.py", "--max_steps", "50", "--max_samples", "500",
             "--output_dir", "outputs/smoke"])
    else:
        print("\n5/6 스모크 테스트 건너뜀")

    step("6/6 본 학습 + 평가")
    run([PY, "src/train.py"])
    run([PY, "src/evaluate.py"])

    print("\n끝. 모델: outputs/kobart-detox/best/")
    print("평가: outputs/kobart-detox/eval_test.json, eval_test_samples.csv (샘플은 직접 읽어봐라)")
    print('추론: .venv 활성화 후  python src/infer.py "씨발 존나 덥네"')


if __name__ == "__main__":
    main()
