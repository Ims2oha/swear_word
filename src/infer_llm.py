"""LoRA 모델 추론. 예: python src/infer_llm.py "씨@발 존나 덥네" """
import argparse
import sys

import torch
import yaml

from common import ROOT
from llm_common import load_for_inference, rewrite

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="*")
    ap.add_argument("--adapter", default=str(ROOT / "outputs" / "exaone-lora" / "best"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "exaone_lora.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_for_inference(cfg, args.adapter, device)

    if args.texts:
        for t, o in zip(args.texts, rewrite(model, tok, cfg["model"], args.texts, device)):
            print(f"{t} -> {o}")
    else:
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            if not line.strip():
                break
            print(rewrite(model, tok, cfg["model"], [line], device)[0])


if __name__ == "__main__":
    main()
