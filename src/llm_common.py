"""LLM(Kanana) + LoRA 공통: 프롬프트, 인코딩, 로딩, 배치 생성.

프롬프트는 모델의 chat template 을 그대로 쓴다 (Llama 3 형식).
  <|start_header_id|>user<|end_header_id|>\n\n{지시문 + 문장}<|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>\n\n{순화문}<|eot_id|>
학습할 때 assistant 헤더 뒤(순화문 + 종료 토큰 <|eot_id|> = eos)에만 loss 를 건다.
"""
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

from preprocess import clean_input

# Kanana-1.5 는 head_dim(128)을 config 에 따로 적어서 hidden_size(1792)가 head 수(24)로 안 나눠떨어진다.
# transformers 5.4+ 의 config 검사가 head_dim 을 무시하고 이걸 에러로 막아서, 그 검사만 뺀다.
LlamaConfig.__class_validators__ = [v for v in getattr(LlamaConfig, "__class_validators__", [])
                                    if v.__name__ != "validate_architecture"]


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


def load_base(model_cfg: dict, dtype):
    return AutoModelForCausalLM.from_pretrained(
        model_cfg["name"], trust_remote_code=model_cfg["trust_remote_code"], dtype=dtype)


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
def _generate_batch(model, tok, prompts, max_new_tokens, device, n=1, temperature=None, top_p=None):
    """prompts -> 문장 리스트. n>1 이면 프롬프트마다 n개씩 샘플링해서 프롬프트 순서대로 이어 붙인다."""
    enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    width, pad = max(len(e) for e in enc), tok.pad_token_id
    ids = torch.tensor([[pad] * (width - len(e)) + e for e in enc]).to(device)        # 왼쪽 패딩
    mask = torch.tensor([[0] * (width - len(e)) + [1] * len(e) for e in enc]).to(device)
    gen = ({"do_sample": True, "temperature": temperature, "top_p": top_p or 1.0, "num_return_sequences": n}
           if temperature else {"do_sample": False})
    out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                         eos_token_id=tok.eos_token_id, pad_token_id=pad, **gen)
    return [tok.decode(o[width:], skip_special_tokens=True).strip() for o in out]


def _run(model, tok, model_cfg, texts, device, batch_size, n=1, **gen):
    """길이순 정렬해서 배치로 생성 (패딩 낭비를 줄인다). 입력 순서대로 문장 n개씩 리스트로 돌려준다."""
    prompts = [build_prompt(tok, model_cfg["prompt"], t) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    outs = [None] * len(texts)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        res = _generate_batch(model, tok, [prompts[i] for i in idx], model_cfg["max_new_tokens"], device, n=n, **gen)
        for k, i in enumerate(idx):
            outs[i] = res[k * n:(k + 1) * n]
    return outs


def rewrite(model, tok, model_cfg: dict, texts, device, batch_size: int = 16):
    """texts -> 순화문 리스트 (입력 순서 유지, greedy)."""
    return [o[0] for o in _run(model, tok, model_cfg, texts, device, batch_size)]


def rewrite_samples(model, tok, model_cfg: dict, texts, device, n=4, temperature=0.8, top_p=0.95,
                    batch_size: int = 16):
    """texts -> 문장마다 순화 후보 n개 (샘플링). greedy 결과가 실패했을 때 다른 순화를 찾는 용도."""
    return _run(model, tok, model_cfg, texts, device, max(1, batch_size // n), n=n,
                temperature=temperature, top_p=top_p)


def load_for_inference(cfg: dict, adapter_dir, device):
    """베이스 모델 + LoRA 어댑터를 합쳐서 로드. (model, tok) 반환."""
    m = cfg["model"]
    dtype = pick_dtype(cfg["train"].get("precision", "auto"))
    base = load_base(m, dtype)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload().to(device).eval()
    adapter_dir = Path(adapter_dir)
    tok_src = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else m["name"]
    return model, load_tokenizer(tok_src, m["trust_remote_code"])
