"""봇 판단(src/moderate.py)을 테스트셋에 돌려서 실제로 사람들한테 보이는 결과를 잰다.

python src/eval_moderate.py [--split test] [--no-prefilter] [--no-obfuscation-check] [--no-self-check] [--max_passes 3] [--min_similarity 30]
출력: outputs/kanana-lora/moderate_{split}.json, moderate_{split}_all.csv

지표(type별)
  pass/rewrite/block : 각 판단 비율
  leak    : 사람들한테 보이는 문장(pass=원문, rewrite=순화문, block=없음)에 PROFANITY 가 남은 비율
  calls   : 문장당 평균 모델 호출 수 (1차 필터에서 걸러지면 0)
identity 는 pass 가 아닌 비율이 곧 '깨끗한 메시지를 건드린' 비율이다.
"""
import argparse
import csv
import json
import sys
from collections import Counter

import torch
import yaml

from common import ROOT, read_jsonl
from evaluate import TYPES
from llm_common import load_for_inference
from moderate import Moderator
from profanity import PROFANITY

sys.stdout.reconfigure(encoding="utf-8")


def visible(text, d):
    return {"pass": text, "rewrite": d.output, "block": ""}[d.action]


def summarize(rows, decs):
    n = len(rows)
    acts = Counter(d.action for d in decs)
    leak = sum(1 for r, d in zip(rows, decs) if any(w in visible(r["input"], d) for w in PROFANITY))
    return {"n": n, **{a: acts[a] / n for a in ("pass", "rewrite", "block")},
            "leak": leak / n, "calls": sum(d.passes for d in decs) / n,
            "reasons": dict(Counter(d.reason for d in decs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "kanana_lora.yaml"))
    ap.add_argument("--adapter", default=str(ROOT / "outputs" / "kanana-lora" / "best"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--no-prefilter", action="store_true", help="1차 필터 없이 모든 문장을 모델에 넣기")
    ap.add_argument("--no-obfuscation-check", action="store_true", help="1차 필터에서 변형 흔적 검사 끄기")
    ap.add_argument("--no-self-check", action="store_true", help="자기 재검사 끄기 (목록 욕설이 남을 때만 다시 순화)")
    ap.add_argument("--max_passes", type=int, default=None)
    ap.add_argument("--min_similarity", type=float, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    rows = read_jsonl(ROOT / cfg["data"]["processed_dir"] / f"{args.split}.jsonl")
    if args.max_samples:
        rows = rows[: args.max_samples]

    override = {"batch_size": cfg["eval"]["batch_size"]}
    if args.no_prefilter:
        override["prefilter"] = False
    if args.no_obfuscation_check:
        override["obfuscation_check"] = False
    if args.no_self_check:
        override["self_check"] = False
    if args.max_passes is not None:
        override["max_passes"] = args.max_passes
    if args.min_similarity is not None:
        override["min_similarity"] = args.min_similarity

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_for_inference(cfg, args.adapter, device)
    mod = Moderator.from_config(cfg, model, tok, device, **override)
    decs = mod.decide([r["input"] for r in rows])

    results = {"settings": {"prefilter": mod.prefilter, "obfuscation_check": mod.obfuscation_check,
                            "self_check": mod.self_check,
                            "max_passes": mod.max_passes,
                            "min_similarity": mod.min_similarity},
               "overall": summarize(rows, decs)}
    for t in TYPES:
        sub = [(r, d) for r, d in zip(rows, decs) if r["type"] == t]
        if sub:
            results[t] = summarize([r for r, _ in sub], [d for _, d in sub])

    out_dir = ROOT / cfg["train"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"moderate_{args.split}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / f"moderate_{args.split}_all.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "action", "reason", "passes", "similarity", "input", "target", "output"])
        for r, d in zip(rows, decs):
            w.writerow([r["type"], d.action, d.reason, d.passes, f"{d.similarity:.1f}", r["input"], r["target"],
                        d.output])

    print(f"settings: {results['settings']}")
    print(f"{'group':<11}{'n':>6}{'pass':>8}{'rewrite':>9}{'block':>8}{'leak':>8}{'calls':>7}")
    for name, m in results.items():
        if name == "settings":
            continue
        print(f"{name:<11}{m['n']:>6}{m['pass']:>8.3f}{m['rewrite']:>9.3f}{m['block']:>8.3f}"
              f"{m['leak']:>8.4f}{m['calls']:>7.2f}")
    print("block 사유:", dict(Counter(d.reason for d in decs if d.action == "block")))
    print(f"saved: {out_dir / f'moderate_{args.split}.json'}")
    print(f"saved: {out_dir / f'moderate_{args.split}_all.csv'}")


if __name__ == "__main__":
    main()
