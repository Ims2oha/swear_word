"""Kanana-1.5-2.1B + LoRA 순화 학습.

python src/train.py [--config configs/kanana_lora.yaml] [--max_steps 30] [--max_samples 200] [--resume]
결과: outputs/kanana-lora/best/ (LoRA 어댑터 + 토크나이저, 수십 MB)
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from common import ROOT, read_jsonl
from llm_common import encode_example, load_base, load_tokenizer, pick_dtype


class LMDataset(Dataset):
    def __init__(self, rows, tok, model_cfg):
        self.items = [encode_example(tok, model_cfg, r) for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, labels = self.items[i]
        return {"input_ids": ids, "labels": labels}


def make_collate(pad_id):
    def collate(batch):
        w = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [pad_id] * (w - len(b["input_ids"])) for b in batch]
        mask = [[1] * len(b["input_ids"]) + [0] * (w - len(b["input_ids"])) for b in batch]
        lab = [b["labels"] + [-100] * (w - len(b["labels"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask),
                "labels": torch.tensor(lab)}
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "kanana_lora.yaml"))
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--resume", action="store_true", help="마지막 체크포인트에서 이어서 학습")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    m, l, t = cfg["model"], cfg["lora"], cfg["train"]
    out = Path(args.output_dir) if args.output_dir else ROOT / t["output_dir"]
    set_seed(cfg["seed"])

    proc = ROOT / cfg["data"]["processed_dir"]
    rows = {s: read_jsonl(proc / f"{s}.jsonl") for s in ("train", "valid")}
    if args.max_samples:
        rng = random.Random(cfg["seed"])
        rows = {s: rng.sample(r, min(len(r), args.max_samples)) for s, r in rows.items()}

    tok = load_tokenizer(m["name"], m["trust_remote_code"])
    train_ds, valid_ds = LMDataset(rows["train"], tok, m), LMDataset(rows["valid"], tok, m)

    dtype = pick_dtype(t["precision"])
    print(f"dtype={dtype}, train={len(train_ds)}, valid={len(valid_ds)}")
    model = load_base(m, dtype)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=l["r"], lora_alpha=l["alpha"], lora_dropout=l["dropout"],
        target_modules=l["target_modules"], bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    eff_bs = t["batch_size"] * t["grad_accum"]
    total = args.max_steps if args.max_steps > 0 else math.ceil(len(train_ds) / eff_bs) * t["epochs"]
    strategy = ({"eval_strategy": "steps", "save_strategy": "steps",
                 "eval_steps": args.max_steps, "save_steps": args.max_steps}
                if args.max_steps > 0 else
                {"eval_strategy": "steps", "save_strategy": "steps",   # 중간에 꺼져도 --resume 으로 이어가기 위해
                 "eval_steps": t["save_steps"], "save_steps": t["save_steps"]})

    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        learning_rate=t["lr"], weight_decay=t["weight_decay"], lr_scheduler_type="cosine",
        num_train_epochs=t["epochs"], max_steps=args.max_steps,
        per_device_train_batch_size=t["batch_size"], per_device_eval_batch_size=cfg["eval"]["batch_size"],
        gradient_accumulation_steps=t["grad_accum"], warmup_steps=int(t["warmup_ratio"] * total),
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2, logging_steps=20, report_to="none",
        remove_unused_columns=False, dataloader_num_workers=0, seed=cfg["seed"],
        **strategy)
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=valid_ds,
        data_collator=make_collate(tok.pad_token_id), processing_class=tok,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=t["patience"])])
    last = get_last_checkpoint(str(out / "checkpoints")) if args.resume else None
    if args.resume:
        print(f"이어서 학습: {last}" if last else "체크포인트가 없어 처음부터 시작")
    trainer.train(resume_from_checkpoint=last)

    best = out / "best"
    trainer.save_model(str(best))          # peft 모델이라 어댑터만 저장된다
    tok.save_pretrained(str(best))
    (out / "train_log.json").write_text(
        json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved adapter -> {best}")


if __name__ == "__main__":
    main()
