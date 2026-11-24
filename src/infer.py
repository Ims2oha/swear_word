"""KoBART 순화 모델 추론 스크립트. 예: python src/infer.py "씨@발 존나 덥네" """
import argparse
import sys

import torch
import yaml
from transformers import BartForConditionalGeneration

from common import PAD_ID, ROOT, decode, encode_source, load_tokenizer
from preprocess import clean_input

sys.stdout.reconfigure(encoding="utf-8")


def rewrite(model, tok, texts, device, max_source_len, max_target_len, beams):
    """texts를 배치로 순화. 가장 긴 길이에 맞춰 PAD 오른쪽 패딩."""
    ids_list = [encode_source(tok, clean_input(t), max_source_len) for t in texts]
    max_len = max(len(ids) for ids in ids_list)
    input_ids, attention_mask = [], []
    for ids in ids_list:
        pad = max_len - len(ids)
        input_ids.append(ids + [PAD_ID] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="*")
    ap.add_argument("--model", default=str(ROOT / "outputs" / "kobart-detox" / "best"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "kobart.yaml"))
    ap.add_argument("--beams", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    max_source_len = cfg["model"]["max_source_len"]
    max_target_len = cfg["model"]["max_target_len"]
    beams = args.beams if args.beams is not None else cfg["eval"]["num_beams"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(args.model)
    model = BartForConditionalGeneration.from_pretrained(args.model).to(device).eval()

    if args.texts:
        outs = rewrite(model, tok, args.texts, device, max_source_len, max_target_len, beams)
        for t, o in zip(args.texts, outs):
            print(f"{t} -> {o}")
    else:
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            if not line.strip():
                break
            print(rewrite(model, tok, [line], device, max_source_len, max_target_len, beams)[0])


if __name__ == "__main__":
    main()
