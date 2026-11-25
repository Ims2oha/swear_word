"""어휘 사전(configs/lexicon.tsv)으로 합성 순화 쌍을 만든다.

정상 문장에 욕설을 끼워 넣거나(정답 = 원래 정상 문장), 고정 틀에 욕설을 넣어서(정답 = 순화 틀) 만든다.
학습 데이터에 없는 욕설을 모델이 지워 버리지 않고 순화하도록 어휘 커버리지를 늘리는 용도다.
"""
import random
import re
from collections import defaultdict

from preprocess import clean_input, clean_target

INTENS_SRC = re.compile(r"(너무|진짜|정말|완전|엄청)\s")

# (입력 틀, 정답 틀). {S}=욕설, {R}=순화어. 정답 틀에 {R}이 있으면 순화어가 있는 항목에만 쓴다.
NOUN_FRAMES = [
    ("{S}들 또 시작이네", "{R}들 또 시작이네"),
    ("{S} 때문에 짜증나", "{R} 때문에 짜증나"),
    ("저 {S} 좀 봐", "저 {R} 좀 봐"),
    ("{S}들은 다 똑같네", "{R}들은 다 똑같네"),
    ("이 {S} 또 왔네", "이 {R} 또 왔네"),
    ("오늘도 {S}들 때문에 망했다", "오늘도 {R}들 때문에 망했다"),
    ("야 {S}, 뭐하냐", "야, 뭐하냐"),
    ("{S} 같은 소리 하네", "그런 소리 하네"),
    ("너 진짜 {S} 아냐?", "너 진짜 왜 그래?"),
    ("{S}야 제발 좀 조용히 해", "제발 좀 조용히 해"),
]
FAMILY_FRAMES = ["네 엄마 {S}", "너희 엄마 {S}", "니 엄마 {S}"]
FAMILY_TARGET = "그런 말은 하지 마"
INTERJ_FRAMES = [("{S} {N}", "{N}"), ("{N} {S}", "{N}"), ("{S}, {N}", "{N}")]
INSULT_PREFIX = ["이 {S}야, ", "야 {S}, ", "{S}야 ", "이 {S}아 "]
INTENS_TO = ["너무", "진짜", "정말", "완전", "엄청"]


def load_lexicon(path):
    lex = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.rstrip("\n").split("\t") + ["", "", "", ""]
        cat, word, repl, var = cols[:4]
        lex[cat].append((word, repl, [v for v in var.split(",") if v]))
    return lex


def surface(entry, rng, p_var=0.35):
    """표기 변형을 확률적으로 고른다 (등록된 변형, 없으면 음절 사이에 숫자 끼우기)."""
    word, _, variants = entry
    if rng.random() < p_var:
        if variants and rng.random() < 0.7:
            return rng.choice(variants)
        syl = [c for c in word if c != " "]
        if len(syl) >= 2 and all("가" <= c <= "힣" for c in syl):
            i = rng.randrange(1, len(syl))
            return "".join(syl[:i]) + rng.choice(["1", "2", "0"]) + "".join(syl[i:])
    return word


def _rec(inp, tgt, group):
    return dict(input=clean_input(inp), target=clean_target(tgt), source="synthetic",
                type="synthetic", group=clean_target(group))


def gen_one(kind, lex, neutrals, rng):
    """kind 별로 합성 쌍 하나. 만들 수 없으면 None."""
    if kind == "interj_neutral":
        n = rng.choice(neutrals)
        s = surface(rng.choice(lex["interj"]), rng, 0.4)
        src, tgt = rng.choice(INTERJ_FRAMES)
        return _rec(src.replace("{S}", s).replace("{N}", n), tgt.replace("{N}", n), n)
    if kind == "insult_neutral":
        n = rng.choice(neutrals)
        e = rng.choice(lex["insult"] + lex["group"])
        pre = rng.choice(INSULT_PREFIX)
        if e in lex["group"]:              # 집단 표현에는 호격 조사(야/아)를 붙이지 않는다
            pre = "야 {S}, "
        return _rec(pre.replace("{S}", surface(e, rng)) + n, n, n)
    if kind == "intens_swap":
        n = rng.choice(neutrals)
        m = INTENS_SRC.search(n)
        if not m:
            return None
        e = rng.choice(lex["intens"])
        s = surface(e, rng, 0.3)
        glue = "" if s == "개" and rng.random() < 0.6 else " "
        return _rec(n[:m.start()] + s + glue + n[m.end():], n, n)
    if kind == "noun_frame":
        e = rng.choice(lex["insult"] + lex["group"])
        src, tgt = rng.choice(NOUN_FRAMES)
        if ("{R}" in tgt and not e[1]) or (e in lex["group"] and "{S}야" in src):
            return None
        t = tgt.replace("{R}", e[1])
        return _rec(src.replace("{S}", surface(e, rng)), t, t)
    if kind == "family":
        e = rng.choice(lex["family"])
        return _rec(rng.choice(FAMILY_FRAMES).replace("{S}", surface(e, rng)),
                    FAMILY_TARGET, FAMILY_TARGET)
    if kind == "phrase":
        e = rng.choice(lex["phrase"])
        s, r = surface(e, rng, 0.3), e[1]
        form = rng.random()
        if form < 0.5:
            return _rec(s, r, r)
        if form < 0.75:
            return _rec(f"야 {s}", f"야 {r}", f"야 {r}")
        n = rng.choice(neutrals)
        return _rec(f"{s}. {n}", f"{r}. {n}", f"{r}. {n}")
    return None


# 종류별 비율 (합 1.0)
MIX = [("interj_neutral", 0.22), ("insult_neutral", 0.22), ("intens_swap", 0.14),
       ("noun_frame", 0.22), ("family", 0.04), ("phrase", 0.16)]


def make_synthetic(neutrals, lex, n, rng, existing):
    """한 split 분량. neutrals = 그 split 의 정상 문장들. existing = 이미 있는 (input,target) 집합."""
    pool = [x for x in neutrals if 6 <= len(x) <= 100]
    out, seen = [], set(existing)
    for kind, share in MIX:
        want, got, tries = round(n * share), 0, 0
        while got < want and tries < want * 30 and (pool or kind in ("noun_frame", "family")):
            tries += 1
            r = gen_one(kind, lex, pool, rng)
            if not r or not r["input"] or not r["target"] or r["input"] == r["target"]:
                continue
            k = (r["input"], r["target"])
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
            got += 1
    rng.shuffle(out)
    return out
