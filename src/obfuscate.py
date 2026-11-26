"""규칙으로 문장을 변형(난독화)한다. KOTOX 변형 욕설 방식을 흉내 낸다.

욕 문장의 입력만 변형하고 정답(순화문)은 그대로 둬서, 정답이 틀릴 일 없이 변형 욕설 쌍을 얼마든지 만든다.
prepare_data.py 의 data.obfuscate 가 train 에만 넣는다 (valid/test 의 KOTOX 변형 문장은 그대로 평가 기준).

음절 단위 변형 (문장마다 몇 가지를 골라 섞는다. 욕설 부분은 더 자주)
  final      받침 추가          먹어야지 -> 먹어얒지, 됐 -> 됞
  tense      된소리·거센소리    생기면 -> 쌩기면, 같아 -> 캍아
  yvowel     y 모음            다 -> 댜, 도 -> 됴, 리 -> 뤼
  evowel     ㅐ/ㅔ 혼동         때문에 -> 떼문애
  enclosed   원문자             다 -> ㈐, 하 -> ㉻
  split      자모 분리          나 -> ㄴㅏ
  lookalike  비슷한 글자        고 -> ㄱ丄, 이 -> oı
  hanja      같은 음 한자       오 -> 五, 대 -> 大
  roman      로마자             지 -> ji
  digit      숫자 끼우기        씨발 -> 씨1발
  mask       X 가리기           존나 -> 존X (욕설 안에서만)
문장 단위: 연음(같아서 -> 가타서), 기호 감싸기·장식(『』 ≥ㅅ≤), 띄어쓰기 없애기
"""
import random
import re

S_BASE, N_V, N_T = 0xAC00, 21, 28
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",
        "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

TENSE = {0: 1, 3: 4, 7: 8, 9: 10, 12: 13}            # ㄱ->ㄲ ㄷ->ㄸ ㅂ->ㅃ ㅅ->ㅆ ㅈ->ㅉ
ASPIRATE = {0: 15, 3: 16, 7: 17, 12: 14}             # ㄱ->ㅋ ㄷ->ㅌ ㅂ->ㅍ ㅈ->ㅊ
YVOWEL = {0: 2, 4: 6, 8: 12, 13: 17, 18: 19, 20: 16, 1: 3, 5: 7}  # ㅏ->ㅑ ㅓ->ㅕ ㅗ->ㅛ ㅜ->ㅠ ㅡ->ㅢ ㅣ->ㅟ ㅐ->ㅒ ㅔ->ㅖ
EVOWEL = {1: 5, 5: 1, 3: 7, 7: 3, 10: 11, 11: 15, 15: 10}         # ㅐ<->ㅔ ㅒ<->ㅖ ㅙ->ㅚ->ㅞ->ㅙ
ADD_FINAL = [1, 4, 8, 19, 25, 27, 21]                              # ㄱ ㄴ ㄹ ㅅ ㅌ ㅎ ㅇ
# 연음: 받침 -> 다음 음절 초성 (다음 음절 초성이 ㅇ 일 때)
LIAISON = {1: 0, 4: 2, 7: 3, 8: 5, 16: 6, 17: 7, 19: 9, 20: 10, 22: 12, 23: 14, 24: 15, 25: 16, 26: 17}
ENCLOSED_BASE = "가나다라마바사아자차카타파하"                       # ㈎~㈛, ㉮~㉻ 순서
LOOKALIKE = {"ㅏ": "ŀ", "ㅣ": "ı", "ㅗ": "丄", "ㅡ": "一", "ㄹ": "己", "ㅁ": "口", "ㅅ": "人", "ㅇ": "o",
             "ㄴ": "L", "ㄱ": "7", "ㅐ": "H", "ㅂ": "ㅂ", "ㅋ": "ㅋ", "ㅈ": "ス", "ㄷ": "ㄷ"}
HANJA = {"오": "五", "일": "一", "이": "二", "삼": "三", "사": "四", "구": "九", "십": "十", "대": "大", "소": "小",
         "인": "人", "산": "山", "수": "水", "화": "火", "목": "木", "금": "金", "토": "土", "천": "天", "백": "百",
         "만": "万", "중": "中", "국": "國", "가": "家", "하": "下", "자": "子", "녀": "女", "남": "男", "모": "母",
         "은": "恩", "문": "門", "망": "亡", "다": "多", "부": "父", "왕": "王", "생": "生", "시": "市", "도": "道"}
R_CHO = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
R_JUNG = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi", "yu",
          "eu", "ui", "i"]
R_JONG = ["", "k", "k", "k", "n", "n", "n", "t", "l", "l", "l", "l", "l", "l", "l", "l", "m", "p", "p", "t", "t",
          "ng", "t", "t", "k", "t", "p", "t"]
WRAPS = [("『", "』"), ("【", "】"), ("「", "」"), ("《", "》"), ("◈", "◈"), ("◎", "◎"), ("∥", "∥"), ("〃", "〃"), ("○", "○")]
DECOR = ["≥ㅅ≤", "≥ㅇ≤", "≥ㅂ≤", "εз", "T^T", "˚", "‥", "…", "◊◊", "●●", "【】", "「」", "¸", "´"]

SYL_METHODS = ["final", "tense", "yvowel", "evowel", "enclosed", "split", "lookalike", "hanja", "roman", "digit", "mask"]
_HANGUL = re.compile("[가-힣]")


def _dec(c):
    o = ord(c) - S_BASE
    if not (0 <= o < 11172):
        return None
    return o // (N_V * N_T), (o % (N_V * N_T)) // N_T, o % N_T


def _comp(l, v, t):
    return chr(S_BASE + (l * N_V + v) * N_T + t)


def _syllable(c, method, rng, in_profanity):
    """음절 하나를 method 로 변형. 못 하면 None."""
    d = _dec(c)
    if d is None:
        return None
    l, v, t = d
    if method == "final" and t == 0:
        return _comp(l, v, rng.choice(ADD_FINAL))
    if method == "tense":
        table = TENSE if rng.random() < 0.6 else ASPIRATE
        if l in table:
            return _comp(table[l], v, t)
    if method == "yvowel" and v in YVOWEL:
        return _comp(l, YVOWEL[v], t)
    if method == "evowel" and v in EVOWEL:
        return _comp(l, EVOWEL[v], t)
    if method == "enclosed" and t == 0 and c in ENCLOSED_BASE:
        k = ENCLOSED_BASE.index(c)
        return chr((0x320E if rng.random() < 0.5 else 0x326E) + k)
    if method == "split":
        return CHO[l] + JUNG[v] + JONG[t]
    if method == "lookalike":
        parts = [CHO[l], JUNG[v]] + ([JONG[t]] if t else [])
        out = "".join(LOOKALIKE.get(p, p) if rng.random() < 0.7 else p for p in parts)
        return out if out != CHO[l] + JUNG[v] + JONG[t] else None
    if method == "hanja" and c in HANJA:
        return HANJA[c]
    if method == "roman":
        return R_CHO[l] + R_JUNG[v] + R_JONG[t]
    if method == "digit":
        return c + rng.choice("1102")
    if method == "mask" and in_profanity:
        return rng.choice(["X", "X", "*", "○"])
    return None


def _liaison(text, rng, p):
    chars = list(text)
    for i in range(len(chars) - 1):
        a, b = _dec(chars[i]), _dec(chars[i + 1])
        if a and b and a[2] in LIAISON and b[0] == 11 and rng.random() < p:
            chars[i] = _comp(a[0], a[1], 0)
            chars[i + 1] = _comp(LIAISON[a[2]], b[1], b[2])
    return "".join(chars)


def _profanity_mask(text, terms):
    """text 의 각 글자가 욕설 안에 있으면 True."""
    flags = [False] * len(text)
    for w in terms or []:
        start = text.find(w)
        while start != -1:
            for k in range(start, start + len(w)):
                flags[k] = True
            start = text.find(w, start + 1)
    return flags


def obfuscate(text, rng: random.Random, terms=None):
    """문장 하나를 변형. 욕설(terms) 부분은 더 자주 변형한다. 변형이 안 되면 None."""
    out = _liaison(text, rng, rng.uniform(0.0, 0.6))
    methods = rng.sample(SYL_METHODS, rng.randint(2, 4))
    p_normal, p_prof = rng.uniform(0.1, 0.35), rng.uniform(0.5, 0.9)
    flags = _profanity_mask(out, terms)
    pieces = []
    for c, prof in zip(out, flags):
        new = None
        if _HANGUL.match(c) and rng.random() < (p_prof if prof else p_normal):
            for m in rng.sample(methods, len(methods)):
                new = _syllable(c, m, rng, prof)
                if new:
                    break
        pieces.append(new or c)
    out = "".join(pieces)

    words = out.split(" ")
    if len(words) > 1 and rng.random() < 0.35:                       # 기호로 감싸기·장식
        for _ in range(rng.randint(1, 3)):
            k = rng.randrange(len(words))
            if rng.random() < 0.5:
                a, b = rng.choice(WRAPS)
                words[k] = a + words[k] + b
            else:
                words.insert(k, rng.choice(DECOR))
        out = " ".join(words)
    if rng.random() < 0.12:                                          # 띄어쓰기 없애기
        out = out.replace(" ", "")
    return out if out != text else None
