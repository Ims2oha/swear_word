"""KoBART 순화 모델 평가 스크립트. 예: python src/evaluate.py --split test """
import argparse
import csv
import json
import random
import sys

import sacrebleu
import torch
import yaml
from transformers import BartForConditionalGeneration

from common import PAD_ID, ROOT, decode, encode_source, load_tokenizer, read_jsonl

sys.stdout.reconfigure(encoding="utf-8")

TYPES = ["toxic", "obfuscated", "identity"]
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


def generate_batch(model, tok, rows, device, max_source_len, max_target_len, beams):
    ids_list = [encode_source(tok, r["input"], max_source_len) for r in rows]
    width = max(len(ids) for ids in ids_list)
    input_ids = [ids + [PAD_ID] * (width - len(ids)) for ids in ids_list]
    attention_mask = [[1] * len(ids) + [0] * (width - len(ids)) for ids in ids_list]
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(device)
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=beams,
            max_length=max_target_len,
            early_stopping=True,
        )
    return [decode(tok, row.tolist()) for row in generated]


def print_table(results):
    print(f"{'group':<11}{'n':>7}{'chrf':>9}{'bleu':>9}{'exact':>9}{'prof_out':>10}{'prof_in':>10}")
    for name, m in results.items():
        print(
            f"{name:<11}{m['n']:>7}{m['chrf']:>9.2f}{m['bleu']:>9.2f}"
            f"{m['exact']:>9.4f}{m['profanity_out']:>10.4f}{m['profanity_in']:>10.4f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "kobart.yaml"))
    ap.add_argument("--model", default=str(ROOT / "outputs" / "kobart-detox" / "best"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--beams", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    max_source_len = cfg["model"]["max_source_len"]
    max_target_len = cfg["model"]["max_target_len"]
    beams = args.beams if args.beams is not None else cfg["eval"]["num_beams"]
    batch_size = cfg["eval"]["batch_size"]

    rows = read_jsonl(ROOT / cfg["data"]["processed_dir"] / f"{args.split}.jsonl")
    if args.max_samples:
        rows = rows[: args.max_samples]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(args.model)
    model = BartForConditionalGeneration.from_pretrained(args.model).to(device).eval()

    outs = [None] * len(rows)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["input"]))
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch_outs = generate_batch(model, tok, [rows[i] for i in idx], device, max_source_len, max_target_len, beams)
        for i, o in zip(idx, batch_outs):
            outs[i] = o

    results = {"overall": score(rows, outs)}
    for t in TYPES:
        sub = [(r, o) for r, o in zip(rows, outs) if r["type"] == t]
        if sub:
            results[t] = score([r for r, _ in sub], [o for _, o in sub])

    out_dir = ROOT / cfg["train"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"eval_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    pairs = list(zip(rows, outs))
    rng = random.Random(cfg["seed"])
    samples = rng.sample(pairs, 100) if len(pairs) > 100 else pairs
    with open(out_dir / f"eval_{args.split}_samples.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "input", "target", "output"])
        for r, o in samples:
            w.writerow([r["type"], r["input"], r["target"], o])

    print_table(results)
    print(f"saved: {out_dir / f'eval_{args.split}.json'}")
    print(f"saved: {out_dir / f'eval_{args.split}_samples.csv'}")


if __name__ == "__main__":
    main()
