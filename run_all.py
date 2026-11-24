"""한 번에 실행: venv -> torch(CUDA) -> 패키지 -> 데이터 -> 스모크 테스트.

python run_all.py [--skip-smoke] [--torch-tag cu121]
이미 끝난 단계(venv, torch, 데이터)는 건너뛴다. 실패하면 그 자리에서 멈춘다.
본 학습과 평가는 포함하지 않는다 (README의 SETUP.md 참고).
"""
import argparse
import os
import re
import shutil
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


def find_smi():
    """PATH에 없어도 Windows 기본 설치 위치를 확인한다."""
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for c in (r"C:\Windows\System32\nvidia-smi.exe",
              r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
        if Path(c).exists():
            return c
    return None


def driver_cuda():
    smi = find_smi()
    if not smi:
        return None
    out = subprocess.run([smi], capture_output=True, text=True).stdout
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return (int(m[1]), int(m[2])) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-smoke", action="store_true", help="스모크 테스트 건너뛰기")
    ap.add_argument("--torch-tag", help="torch wheel 태그 직접 지정 (예: cu121, cu126, cu128)")
    args = ap.parse_args()

    step("1/5 가상환경")
    if not PY.exists():
        run([sys.executable, "-m", "venv", VENV])
    run([PY, "-m", "pip", "install", "-q", "--upgrade", "pip"])

    step("2/5 torch (CUDA)")
    if cuda_ok():
        print("CUDA torch 이미 설치됨, 건너뜀")
    else:
        ver = driver_cuda()
        if args.torch_tag:
            tags = [args.torch_tag]
        elif ver is None:
            print("nvidia-smi 를 못 찾음 -> cu126 으로 한 번만 시도")
            tags = ["cu126"]
        else:
            print(f"드라이버 CUDA {ver[0]}.{ver[1]}")
            tags = [tag for need, tag in TORCH_TAGS if ver >= need]
        for tag in tags:
            r = run([PY, "-m", "pip", "install", "torch", "--index-url",
                     f"https://download.pytorch.org/whl/{tag}"], check=False)
            if r.returncode == 0 and cuda_ok():
                break
            print(f"{tag} 실패" + (", 더 낮은 버전 시도" if tag != tags[-1] else ""))
        if not cuda_ok():
            sys.exit(
                "\ntorch 에서 CUDA 를 못 씀. 확인할 것:\n"
                "  1) NVIDIA GPU 가 맞는지 (AMD/인텔 GPU 면 이 방식으로는 안 됨)\n"
                "  2) NVIDIA 드라이버 설치 후 재부팅 (https://www.nvidia.com/drivers)\n"
                "  3) 드라이버는 있는데 안 되면 태그 직접 지정: python run_all.py --torch-tag cu121")

    step("3/5 패키지")
    run([PY, "-m", "pip", "install", "-q", "-r", "requirements.txt"])

    step("4/5 데이터 준비")
    if (ROOT / "data/processed/train.jsonl").exists():
        print("data/processed 이미 있음, 건너뜀 (다시 만들려면 data 폴더 삭제)")
    else:
        run([PY, "src/prepare_data.py"])

    if not args.skip_smoke:
        step("5/5 스모크 테스트 (50 step)")
        run([PY, "src/train.py", "--max_steps", "50", "--max_samples", "500",
             "--output_dir", "outputs/smoke"])
    else:
        print("\n5/5 스모크 테스트 건너뜀")

    py = ".venv\\Scripts\\python" if os.name == "nt" else ".venv/bin/python"
    print("\n준비 끝. 본 학습과 평가는 직접 실행:")
    print(f"  {py} src/train.py")
    print(f"  {py} src/evaluate.py")


if __name__ == "__main__":
    main()
