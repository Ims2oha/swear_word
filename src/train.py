"""KoBART 전체 미세조정 (유해문 -> 순화문).

python src/train.py [--config ...] [--max_steps 50] [--max_samples 200] [--output_dir ...]
--max_steps / --max_samples 는 스모크 테스트용.
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import sacrebleu
import torch
import yaml
from torch.utils.data import Dataset
from transformers import (BartForConditionalGeneration, EarlyStoppingCallback,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)

from common import PAD_ID, ROOT, decode, encode_source, encode_target, load_tokenizer, read_jsonl


class PairDataset(Dataset):
    def __init__(self, rows, tok, max_src, max_tgt):
        self.items = [(encode_source(tok, r["input"], max_src), encode_target(tok, r["target"], max_tgt))
                      for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        src, tgt = self.items[i]
        return {"input_ids": src, "labels": tgt}


def collate(batch):
    ms, mt = max(len(b["input_ids"]) for b in batch), max(len(b["labels"]) for b in batch)
    ids = [b["input_ids"] + [PAD_ID] * (ms - len(b["input_ids"])) for b in batch]
    mask = [[1] * len(b["input_ids"]) + [0] * (ms - len(b["input_ids"])) for b in batch]
    lab = [b["labels"] + [-100] * (mt - len(b["labels"])) for b in batch]
    return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask), "labels": torch.tensor(lab)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "kobart.yaml"))
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    t, m = cfg["train"], cfg["model"]
    out = Path(args.output_dir) if args.output_dir else ROOT / t["output_dir"]
    set_seed(cfg["seed"])

    proc = ROOT / cfg["data"]["processed_dir"]
    rows = {s: read_jsonl(proc / f"{s}.jsonl") for s in ("train", "valid")}
    if args.max_samples:
        rng = random.Random(cfg["seed"])
        rows = {s: rng.sample(r, min(len(r), args.max_samples)) for s, r in rows.items()}

    tok = load_tokenizer(m["name"])
    train_ds = PairDataset(rows["train"], tok, m["max_source_len"], m["max_target_len"])
    valid_ds = PairDataset(rows["valid"], tok, m["max_source_len"], m["max_target_len"])
    model = BartForConditionalGeneration.from_pretrained(m["name"])

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        hyps = [decode(tok, x) for x in preds]
        refs = [decode(tok, x) for x in p.label_ids]
        return {"chrf": sacrebleu.corpus_chrf(hyps, [refs]).score,
                "exact": float(np.mean([h == r for h, r in zip(hyps, refs)]))}

    use_cuda = torch.cuda.is_available()
    fp16 = use_cuda if t["fp16"] == "auto" else bool(t["fp16"])
    steps_per_epoch = math.ceil(len(train_ds) / t["batch_size"])
    total = args.max_steps if args.max_steps > 0 else steps_per_epoch * t["epochs"]
    strategy = {"eval_strategy": "steps", "save_strategy": "steps",
                "eval_steps": args.max_steps, "save_steps": args.max_steps} if args.max_steps > 0 else \
               {"eval_strategy": "epoch", "save_strategy": "epoch"}

    targs = Seq2SeqTrainingArguments(
        output_dir=str(out / "checkpoints"),
        learning_rate=t["lr"], weight_decay=t["weight_decay"],
        num_train_epochs=t["epochs"], max_steps=args.max_steps,
        per_device_train_batch_size=t["batch_size"], per_device_eval_batch_size=t["batch_size"],
        warmup_steps=int(t["warmup_ratio"] * total),
        fp16=fp16, predict_with_generate=True,
        generation_max_length=m["max_target_len"], generation_num_beams=1,
        load_best_model_at_end=True, metric_for_best_model="chrf", greater_is_better=True,
        save_total_limit=2, logging_steps=50, report_to="none",
        remove_unused_columns=False, dataloader_num_workers=0, seed=cfg["seed"],
        **strategy,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=valid_ds,
        data_collator=collate, processing_class=tok, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=t["patience"])],
    )
    trainer.train()

    best = out / "best"
    trainer.save_model(str(best))
    tok.save_pretrained(str(best))
    (out / "train_log.json").write_text(
        json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved best model -> {best}")


if __name__ == "__main__":
    main()
