"""LoRA 모델 평가.

python src/evaluate.py [--split test] [--max_samples 0]
출력: outputs/kanana-lora/eval_{split}.json, eval_{split}_all.csv (모든 행의 입력/정답/출력)
지표(type별): chrf, bleu, exact, profanity_out/in, n
"""
import argparse
import csv
import json
import sys

import sacrebleu
import torch
import yaml

from common import ROOT, read_jsonl
from llm_common import load_for_inference, rewrite

sys.stdout.reconfigure(encoding="utf-8")

TYPES = ["toxic", "obfuscated", "identity", "synthetic", "llm"]
PROFANITY = ["씨발", "시발", "ㅅㅂ", "ㅆㅂ", "병신", "ㅂㅅ", "개새끼", "좆", "존나", "졸라", "지랄", "썅", "미친놈", "미친년"]


def profanity_rate(texts):
    return sum(1 for t in texts if any(w in t for w in PROFANITY)) / len(texts)


def score(rows, outs):
    n = len(rows)
    if n == 0:
        return {"chrf": 0.0, "bleu": 0.0, "exact": 0.0, "profanity_out": 0.0, "profanity_in": 0.0, "n": 0}
    refs = [r["target"] for r in rows]
    return {
        "chrf": sacrebleu.corpus_chrf(outs, [refs]).score,
        "bleu": sacrebleu.corpus_bleu(outs, [refs], tokenize="char").score,
        "exact": sum(1 for o, ref in zip(outs, refs) if o == ref) / n,
        "profanity_out": profanity_rate(outs),
        "profanity_in": profanity_rate([r["input"] for r in rows]),
        "n": n,
    }


def print_table(results):
    print(f"{'group':<11}{'n':>7}{'chrf':>9}{'bleu':>9}{'exact':>9}{'prof_out':>10}{'prof_in':>10}")
    for name, m in results.items():
        print(
            f"{name:<11}{m['n']:>7}{m['chrf']:>9.2f}{m['bleu']:>9.2f}"
            f"{m['exact']:>9.4f}{m['profanity_out']:>10.4f}{m['profanity_in']:>10.4f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "kanana_lora.yaml"))
    ap.add_argument("--adapter", default=str(ROOT / "outputs" / "kanana-lora" / "best"))
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
