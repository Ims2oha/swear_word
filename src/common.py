"""토크나이징·디코딩·jsonl 공통 함수.

KoBART(gogamza/kobart-base-v2) 토크나이저는 특수 토큰을 자동으로 안 붙인다.
vocab: <s>=0, </s>=1, <pad>=3. 모델 config: decoder_start_token_id=eos=1.
  인코더 입력 = <s> 문장 </s>
  라벨        = 문장 </s>        (decoder 입력은 모델이 [</s>] + 라벨[:-1]로 자동 생성)
"""
import json
from pathlib import Path

from transformers import PreTrainedTokenizerFast

BOS_ID, EOS_ID, PAD_ID = 0, 1, 3
ROOT = Path(__file__).resolve().parent.parent


def load_tokenizer(name_or_path: str) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast.from_pretrained(name_or_path)


def _ids(tok, text: str):
    return tok(text, add_special_tokens=False, return_token_type_ids=False)["input_ids"]


def encode_source(tok, text: str, max_len: int):
    return [BOS_ID] + _ids(tok, text)[: max_len - 2] + [EOS_ID]


def encode_target(tok, text: str, max_len: int):
    return _ids(tok, text)[: max_len - 1] + [EOS_ID]


def decode(tok, ids) -> str:
    """생성 결과/라벨 디코딩. 맨 앞 decoder_start(</s>)는 건너뛰고, 그 뒤 첫 </s>에서 끊음."""
    out = []
    for i in ids:
        i = int(i)
        if i == EOS_ID and out:
            break
        if i in (BOS_ID, EOS_ID, PAD_ID) or i < 0:
            continue
        out.append(i)
    return tok.decode(out, skip_special_tokens=True).strip()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
