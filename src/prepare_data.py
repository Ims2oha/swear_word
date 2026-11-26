"""K/DA + KOTOX 순화 쌍 데이터 준비.

python src/prepare_data.py [--config configs/kanana_lora.yaml]
출력: data/processed/{train,valid,test}.jsonl, data/processed/stats.json
필드: input, target, source(kotox|kda|synthetic|llm), type(toxic|obfuscated|identity|synthetic|llm), group(정상 문장, split 단위)
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from huggingface_hub import hf_hub_download

from augment import load_lexicon, make_synthetic
from common import ROOT, read_jsonl, write_jsonl
from obfuscate import obfuscate
from preprocess import clean_input, clean_target
from profanity import profanity_terms

# LLM 거절·메타 문구 (일반 한국어 "할 수 없다" 같은 표현은 넣지 않음)
META = re.compile(
    "|".join(map(re.escape, [
        "죄송합니다", "죄송하지만", "언어 모델", "언어모델", "AI로서", "AI 어시스턴트", "인공지능",
        "도와드릴 수 없", "제공할 수 없", "as an ai", "i cannot", "i can't", "i'm sorry", "sorry",
    ])),
    re.I,
)
SPLIT_RANK = {"train": 0, "valid": 1, "test": 2}


def fetch(repo: str, filename: str, raw_dir: Path) -> Path:
    local = raw_dir / repo.replace("/", "__")
    return Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", local_dir=str(local)))


def read_csv(path: Path, required, name: str, drops: Counter) -> pd.DataFrame:
    bad = []
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, engine="python",
                     on_bad_lines=lambda row: bad.append(1))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[{name}] 컬럼 없음: {missing} (실제 컬럼: {list(df.columns)})")
    if bad:
        drops[f"{name}:bad_csv_line"] += len(bad)
    print(f"[{name}] rows={len(df)} bad_lines={len(bad)}")
    return df


def build(df: pd.DataFrame, source: str, specs, hint, cfg_d, drops: Counter):
    """specs: [(컬럼, type)] 순서대로 쌍을 만든다. identity 는 neutral->neutral."""
    recs = []
    for row in df.to_dict("records"):
        neutral = clean_target(row.get("neutral"))
        for col, typ in specs:
            if typ == "identity":
                inp = clean_input(row.get("neutral"))
                tgt = inp
            else:
                inp = clean_input(row.get(col))
                tgt = neutral
            if not inp or not tgt:
                drops["empty"] += 1
            elif not (cfg_d["min_len"] <= len(inp) <= cfg_d["max_len"]
                      and cfg_d["min_len"] <= len(tgt) <= cfg_d["max_len"]):
                drops["length"] += 1
            elif META.search(inp) or META.search(tgt):
                drops["meta_phrase"] += 1
            elif typ != "identity" and inp == tgt:
                drops["input_equals_target"] += 1
            else:
                recs.append(dict(input=inp, target=tgt, source=source, type=typ,
                                 group=neutral, hint=hint))
    return recs


def hash_split(group: str, seed: int, ratios) -> str:
    h = int(hashlib.md5(f"{seed}:{group}".encode("utf-8")).hexdigest(), 16) % 10000 / 10000
    return "train" if h < ratios[0] else ("valid" if h < ratios[0] + ratios[1] else "test")


def cap_identity(rows, frac: float, rng: random.Random):
    ident = [r for r in rows if r["type"] == "identity"]
    rest = [r for r in rows if r["type"] != "identity"]
    keep = int(frac / (1 - frac) * len(rest))
    if len(ident) > keep:
        ident = rng.sample(ident, keep)
    out = rest + ident
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "kanana_lora.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    d, seed = cfg["data"], cfg["seed"]
    raw_dir, out_dir = ROOT / d["raw_dir"], ROOT / d["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    drops = Counter()

    # 1) 다운로드 + 쌍 만들기
    kotox_cols = ["neutral", "toxic", "toxic_obfuscated_texts"]
    kotox_specs = [("toxic", "toxic"), ("toxic_obfuscated_texts", "obfuscated"), ("neutral", "identity")]
    recs = []
    for split, fname in d["kotox_files"].items():
        df = read_csv(fetch(d["kotox_repo"], fname, raw_dir), kotox_cols, f"kotox/{split}", drops)
        recs += build(df, "kotox", kotox_specs, split, d, drops)
    df = read_csv(fetch(d["kda_repo"], d["kda_file"], raw_dir), ["neutral", "implicit"], "kda", drops)
    recs += build(df, "kda", [("implicit", "toxic"), ("neutral", "identity")], None, d, drops)

    # 2) 중복 제거 (input, target) - KOTOX 먼저
    seen, uniq = set(), []
    for r in recs:
        k = (r["input"], r["target"])
        if k in seen:
            drops["duplicate"] += 1
            continue
        seen.add(k)
        uniq.append(r)

    # 3) split: 같은 정상 문장(group)은 같은 split. KOTOX 기존 split 우선 (test > valid > train)
    group_split = {}
    for r in uniq:
        if r["hint"] and (r["group"] not in group_split
                          or SPLIT_RANK[r["hint"]] > SPLIT_RANK[group_split[r["group"]]]):
            group_split[r["group"]] = r["hint"]
    splits = {"train": [], "valid": [], "test": []}
    for r in uniq:
        s = group_split.get(r["group"]) or hash_split(r["group"], seed, d["kda_split"])
        r.pop("hint")
        splits[s].append(r)

    # 4) identity 비율 제한
    rng = random.Random(seed)
    splits = {s: cap_identity(rows, d["identity_max_frac"], rng) for s, rows in splits.items()}

    # 4.5) 어휘 사전 기반 합성 쌍 (욕설 커버리지 확장). split 별로 그 split 의 정상 문장만 쓴다
    aug = d.get("augment") or {}
    if aug.get("enabled"):
        lex = load_lexicon(ROOT / aug["lexicon"])
        for s in splits:
            neutrals = sorted({r["group"] for r in splits[s]})
            existing = {(r["input"], r["target"]) for r in splits[s]}
            syn = make_synthetic(neutrals, lex, aug["n"][s], random.Random(f"{seed}:{s}"), existing)
            own = set(neutrals)
            # 정상 문장에서 만든 쌍은 그 split, 고정 틀 쌍은 group 해시가 가리키는 split 에만 둔다
            syn = [r for r in syn if r["group"] in own or hash_split(r["group"], seed, d["kda_split"]) == s]
            splits[s] += syn
            drops[f"synthetic_added:{s}"] = len(syn)
        seen_g = {}
        for s, rows in splits.items():          # 같은 group 이 여러 split 에 생기면 먼저 만든 split 만 남긴다
            splits[s] = [r for r in rows if seen_g.setdefault(r["group"], s) == s]
        for rows in splits.values():
            rng.shuffle(rows)

    # 4.6) 로컬 LLM 으로 만든 순화 쌍 (src/build_llm_pairs.py). 없으면 건너뛴다
    extra = d.get("extra_pairs")
    if extra and (ROOT / extra).exists():
        # 이미 어떤 split 에 있는 group 은 그 split 으로, 아니면 group 해시로 정한다 (누수 방지)
        taken = {r["group"]: s for s, rows in splits.items() for r in rows}
        existing = {(r["input"], r["target"]) for rows in splits.values() for r in rows}
        added = Counter()
        for r in read_jsonl(ROOT / extra):
            if (r["input"], r["target"]) in existing:
                drops["llm_duplicate"] += 1
                continue
            existing.add((r["input"], r["target"]))
            s = taken.setdefault(r["group"], hash_split(r["group"], seed, d["kda_split"]))
            splits[s].append(dict(input=r["input"], target=r["target"], source="llm", type=r.get("type", "llm"),
                                 group=r["group"]))
            added[s] += 1
        for s in splits:
            drops[f"llm_added:{s}"] = added[s]
            rng.shuffle(splits[s])
    elif extra:
        print(f"[extra_pairs] {extra} 없음, 건너뜀")

    # 4.65) 규칙 변형(난독화) 쌍: 욕 문장의 입력만 변형하고 정답은 그대로. train 에만 넣는다 (KOTOX 변형 test 는 그대로 평가 기준)
    obf = d.get("obfuscate") or {}
    if obf.get("enabled"):
        terms = profanity_terms(ROOT / aug.get("lexicon", "configs/lexicon.tsv"))
        orng = random.Random(f"{seed}:obfuscate")
        pool = [r for r in splits["train"] if r["type"] in set(obf.get("from_types", ["toxic", "synthetic", "multi", "llm"]))]
        existing = {(r["input"], r["target"]) for r in splits["train"]}
        new, tries = [], 0
        while pool and len(new) < obf["n"] and tries < obf["n"] * 5:
            tries += 1
            r = orng.choice(pool)
            inp = obfuscate(r["input"], orng, terms)
            if not inp or len(inp) > d["max_len"] or (inp, r["target"]) in existing:
                continue
            existing.add((inp, r["target"]))
            new.append(dict(input=inp, target=r["target"], source="obfuscate", type="obf_aug", group=r["group"]))
        splits["train"] += new
        rng.shuffle(splits["train"])
        drops["obfuscate_added:train"] = len(new)

    # 4.7) identity(정상->정상) 늘리기: 모델이 깨끗한 문장을 고치거나 뒷부분을 지우는 걸 줄인다
    ide = d.get("identity_extra") or {}
    if ide.get("from_llm_targets"):
        # LLM 쌍의 정답(새 정상 문장)을 그대로 identity 로. train 에만 넣어서 valid/test 는 이전 결과와 비교 가능하게 둔다
        rows = splits["train"]
        existing = {(r["input"], r["target"]) for r in rows}
        new = []
        for r in rows:
            k = (r["target"], r["target"])
            if r["source"] == "llm" and k not in existing:
                existing.add(k)
                new.append(dict(input=r["target"], target=r["target"], source="llm", type="identity",
                                group=r["group"]))
        rows += new
        drops["llm_identity_added:train"] = len(new)
    repeat = int(ide.get("train_repeat", 1))
    if repeat > 1:
        # train 에서만 identity 를 반복해 비중을 올린다 (valid/test 는 평가 기준이라 그대로)
        ident = [r for r in splits["train"] if r["type"] == "identity"]
        splits["train"] += [dict(r) for r in ident for _ in range(repeat - 1)]
        drops["identity_repeated:train"] = len(ident) * (repeat - 1)
    if ide:
        for rows in splits.values():
            rng.shuffle(rows)

    # 5) 누수 검사 + 저장
    groups = {s: {r["group"] for r in rows} for s, rows in splits.items()}
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        assert not (groups[a] & groups[b]), f"split 누수: {a} & {b} 에 같은 정상 문장 있음"
    for s, rows in splits.items():
        write_jsonl(out_dir / f"{s}.jsonl", rows)

    stats = {
        "drops": dict(drops),
        "splits": {s: {"n": len(rows), "by_type": dict(Counter(r["type"] for r in rows)),
                       "by_source": dict(Counter(r["source"] for r in rows))}
                   for s, rows in splits.items()},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
