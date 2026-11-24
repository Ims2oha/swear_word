"""LoRA 모델 평가. 지표는 KoBART 평가(evaluate.py)와 같은 함수를 써서 그대로 비교된다.

python src/evaluate_llm.py [--split test] [--max_samples 0]
출력: outputs/exaone-lora/eval_{split}.json, eval_{split}_all.csv (모든 행의 입력/정답/출력)
"""
import argparse
import csv
import json
import sys

import torch
import yaml

from common import ROOT, read_jsonl
from evaluate import TYPES, print_table, score
from llm_common import load_for_inference, rewrite

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "exaone_lora.yaml"))
    ap.add_argument("--adapter", default=str(ROOT / "outputs" / "exaone-lora" / "best"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    rows = read_jsonl(ROOT / cfg["data"]["processed_dir"] / f"{args.split}.jsonl")
    if args.max_samples:
        rows = rows[: args.max_samples]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_for_inference(cfg, args.adapter, device)
    outs = rewrite(model, tok, cfg["model"], [r["input"] for r in rows], device, cfg["eval"]["batch_size"])

    results = {"overall": score(rows, outs)}
    for t in TYPES:
        sub = [(r, o) for r, o in zip(rows, outs) if r["type"] == t]
        if sub:
            results[t] = score([r for r, _ in sub], [o for _, o in sub])

    out_dir = ROOT / cfg["train"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval_{args.split}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / f"eval_{args.split}_all.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "input", "target", "output"])
        for r, o in zip(rows, outs):
            w.writerow([r["type"], r["input"], r["target"], o])

    print_table(results)
    print(f"saved: {out_dir / f'eval_{args.split}.json'}")
    print(f"saved: {out_dir / f'eval_{args.split}_all.csv'}")


if __name__ == "__main__":
    main()
