"""LLM(EXAONE) + LoRA 공통: 프롬프트, 인코딩, 로딩, 배치 생성.

프롬프트는 모델의 chat template 을 그대로 쓴다.
  [|system|][|endofturn|]\n[|user|]{지시문 + 문장}\n[|assistant|]{순화문}[|endofturn|]
학습할 때 [|assistant|] 뒤(순화문 + 종료 토큰)에만 loss 를 건다.
"""
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from preprocess import clean_input


def pick_dtype(precision: str):
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_tokenizer(name_or_path, trust_remote_code: bool):
    tok = AutoTokenizer.from_pretrained(str(name_or_path), trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _expose_embeddings(model):
    """EXAONE 커스텀 코드에는 get_input_embeddings 가 없어 peft/체크포인팅이 실패한다.
    임베딩 층(transformer.wte)의 위치를 직접 알려준다."""
    inner = getattr(model, "transformer", None)
    if inner is None or not hasattr(inner, "wte"):
        return
    emb = inner.wte
    for m in (model, inner):
        m.get_input_embeddings = lambda emb=emb: emb


def load_base(model_cfg: dict, dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"], trust_remote_code=model_cfg["trust_remote_code"], dtype=dtype)
    _expose_embeddings(model)
    return model


def build_prompt(tok, template: str, text: str) -> str:
    content = template.replace("{text}", clean_input(text))
    return tok.apply_chat_template([{"role": "user", "content": content}],
                                   add_generation_prompt=True, tokenize=False)


def encode_example(tok, model_cfg: dict, row: dict):
    """(input_ids, labels). labels 는 프롬프트 부분을 -100 으로 가린다."""
    prompt_ids = tok(build_prompt(tok, model_cfg["prompt"], row["input"]),
                     add_special_tokens=False)["input_ids"]
    answer_ids = tok(row["target"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    limit = model_cfg["max_seq_len"]
    return (prompt_ids + answer_ids)[:limit], ([-100] * len(prompt_ids) + answer_ids)[:limit]


@torch.no_grad()
def _generate_batch(model, tok, prompts, max_new_tokens, device):
    enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    width, pad = max(len(e) for e in enc), tok.pad_token_id
    ids = torch.tensor([[pad] * (width - len(e)) + e for e in enc]).to(device)        # 왼쪽 패딩
    mask = torch.tensor([[0] * (width - len(e)) + [1] * len(e) for e in enc]).to(device)
    out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                         do_sample=False, eos_token_id=tok.eos_token_id, pad_token_id=pad)
    return [tok.decode(o[width:], skip_special_tokens=True).strip() for o in out]


def rewrite(model, tok, model_cfg: dict, texts, device, batch_size: int = 16):
    """texts -> 순화문 리스트 (입력 순서 유지). 길이순 정렬해서 패딩 낭비를 줄인다."""
    prompts = [build_prompt(tok, model_cfg["prompt"], t) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    outs = [None] * len(texts)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        res = _generate_batch(model, tok, [prompts[i] for i in idx], model_cfg["max_new_tokens"], device)
        for i, o in zip(idx, res):
            outs[i] = o
    return outs


def load_for_inference(cfg: dict, adapter_dir, device):
    """베이스 모델 + LoRA 어댑터를 합쳐서 로드. (model, tok) 반환."""
    m = cfg["model"]
    dtype = pick_dtype(cfg["train"].get("precision", "auto"))
    base = load_base(m, dtype)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload().to(device).eval()
    adapter_dir = Path(adapter_dir)
    tok_src = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else m["name"]
    return model, load_tokenizer(tok_src, m["trust_remote_code"])
