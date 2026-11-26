"""로컬 LLM 워커가 만든 순화 쌍(data/llm_pairs/raw/*.txt)을 걸러서 학습용 파일로 만든다.

python src/build_llm_pairs.py [--exclude 직장_D,게임_C ...]
입력: data/llm_pairs/raw/{주제}_{유형}.txt  (한 줄에 {"input","target"} JSON)
      유형이 IDENTITY_TYPES(G)면 '욕처럼 보이지만 깨끗한 문장'이라 input == target 이어야 한다
출력: data/llm_pairs/llm_pairs.jsonl     {"input","target","source":"llm","type":"llm"|"llm_identity","group":target}
      data/llm_pairs/review_pool.jsonl   위와 같고 combo(주제_유형) 필드 추가 (검수용)
      data/llm_pairs/build_report.json   사유별 제거 개수, 유형별 남은 개수
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from common import ROOT
from preprocess import clean_input, clean_target
from profanity import find_profanity, profanity_terms

LLM_DIR = ROOT / "data" / "llm_pairs"
MIN_IN, MAX_IN = 8, 80
RATIO = (0.5, 1.5)
JAMO_MAX = 0.3
SAME_TARGET_MAX = 5

IDENTITY_TYPES = {"G"}
# G(깨끗한 문장)에 있으면 안 되는 말. 일반 뜻으로도 쓰이는 말(시발점, 졸라서, 쓰레기, 죽어)은 일부러 뺐다
IDENTITY_REJECT = ["씨발", "ㅅㅂ", "ㅆㅂ", "병신", "개새끼", "좆", "존나", "지랄", "썅", "미친놈", "미친년",
                   "개꿀", "빡치", "빡쳐", "똥겜", "깽판", "노답", "ㅈ됨"]

FIXED = {"그만좀해", "그런말은하지마", "왜그래", "야너왜그래", "조용히해줘", "저리가줘", "왜그러니",
         "그만해", "하지마", "그러지마", "말조심해"}
META = re.compile(r"죄송|언어 ?모델|인공지능|순화|input|target|프롬프트|\bAI\b", re.I)
JAMO = re.compile("[ㄱ-ㅎㅏ-ㅣ]")
NUM = re.compile(r"\d+")
LATIN = re.compile(r"[A-Za-z]+")
FOREIGN = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")  # 가나·한자 (워커가 섞어 쓰는 경우)


def known_inputs():
    """data/processed 의 기존 입력 (이 스크립트가 만든 llm 쌍은 제외)."""
    seen = set()
    for p in (ROOT / "data" / "processed").glob("*.jsonl"):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("source") != "llm":
                seen.add(r["input"])
    return seen


def check_identity(raw_line):
    """G 유형: 욕처럼 보이지만 깨끗한 문장. (사유 또는 None, input, target)"""
    try:
        o = json.loads(raw_line)
    except Exception:
        return "json_error", None, None
    if not isinstance(o, dict) or not isinstance(o.get("input"), str) or not isinstance(o.get("target"), str):
        return "missing_field", None, None
    inp = clean_input(o["input"])
    if not inp or inp != clean_input(o["target"]):
        return "identity_changed", None, None
    if not (MIN_IN <= len(inp) <= MAX_IN):
        return "input_length", None, None
    if FOREIGN.search(inp) or META.search(inp):
        return "foreign_or_meta", None, None
    if any(w in inp for w in IDENTITY_REJECT):
        return "identity_has_profanity", None, None
    return None, inp, inp


def check(raw_line, terms):
    """(사유 또는 None, input, target)"""
    try:
        o = json.loads(raw_line)
    except Exception:
        return "json_error", None, None
    if not isinstance(o, dict) or not isinstance(o.get("input"), str) or not isinstance(o.get("target"), str):
        return "missing_field", None, None
    if o["input"] == o["target"]:
        return "input_equals_target", None, None
    inp, tgt = clean_input(o["input"]), clean_target(o["target"])
    if not inp or not tgt:
        return "empty", None, None
    if inp == tgt:
        return "equal_after_clean", None, None
    if not (MIN_IN <= len(inp) <= MAX_IN):
        return "input_length", None, None
    if not (RATIO[0] <= len(tgt) / len(inp) <= RATIO[1]):
        return "length_ratio", None, None
    if FOREIGN.search(inp) or FOREIGN.search(tgt):
        return "foreign_script", None, None
    if META.search(inp) or META.search(tgt):
        return "meta_phrase", None, None
    if find_profanity(tgt, terms):
        return "profanity_in_target", None, None
    nonspace = [c for c in tgt if not c.isspace()]
    if len(JAMO.findall(tgt)) / max(1, len(nonspace)) >= JAMO_MAX:
        return "jamo_ratio", None, None
    if re.sub(r"[\s.,!?~…]", "", tgt) in FIXED:
        return "fixed_phrase", None, None
    if set(NUM.findall(tgt)) - set(NUM.findall(inp)) or \
            {w.lower() for w in LATIN.findall(tgt)} - {w.lower() for w in LATIN.findall(inp)}:
        return "new_number_or_name", None, None
    return None, inp, tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default="", help="버릴 조합 (쉼표). 예: 직장_D,게임_C")
    args = ap.parse_args()
    exclude = {c for c in args.exclude.split(",") if c}

    terms = profanity_terms(ROOT / "configs" / "lexicon.tsv")
    existing = known_inputs()
    drops, kept_by_type, target_count = Counter(), Counter(), Counter()
    seen_inputs, out = set(), []
    files = sorted((LLM_DIR / "raw").glob("*.txt"))
    for path in files:
        combo = path.stem
        typ = combo.rsplit("_", 1)[1]
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if combo in exclude:
                drops["excluded_combo"] += 1
                continue
            reason, inp, tgt = check_identity(line) if typ in IDENTITY_TYPES else check(line, terms)
            if reason is None and inp in existing:
                reason = "dup_processed"
            elif reason is None and inp in seen_inputs:
                reason = "dup_in_file"
            elif reason is None and target_count[tgt] >= SAME_TARGET_MAX:
                reason = "same_target_over_5"
            if reason:
                drops[reason] += 1
                continue
            seen_inputs.add(inp)
            target_count[tgt] += 1
            kept_by_type[typ] += 1
            out.append(dict(input=inp, target=tgt, source="llm",
                            type="llm_identity" if typ in IDENTITY_TYPES else "llm", group=tgt, combo=combo))

    LLM_DIR.mkdir(parents=True, exist_ok=True)
    with open(LLM_DIR / "llm_pairs.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps({k: v for k, v in r.items() if k != "combo"}, ensure_ascii=False) + "\n")
    with open(LLM_DIR / "review_pool.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"files": len(files), "kept": len(out), "kept_by_type": dict(sorted(kept_by_type.items())),
              "drops": dict(drops.most_common()), "excluded": sorted(exclude)}
    (LLM_DIR / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
