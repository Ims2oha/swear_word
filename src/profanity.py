"""욕설 목록과 문자열 매칭 (torch 없이 쓸 수 있게 따로 둔다).

PROFANITY       : 평가 지표(prof_in/prof_out)용 고정 목록. 바꾸면 이전 평가 결과와 비교가 안 된다.
profanity_terms : lexicon.tsv + PROFANITY + EXTRA_PROFANITY. 데이터 필터와 봇의 1차 필터·잔존 검사용.
"""
from augment import load_lexicon

PROFANITY = ["씨발", "시발", "ㅅㅂ", "ㅆㅂ", "병신", "ㅂㅅ", "개새끼", "좆", "존나", "졸라", "지랄", "썅", "미친놈", "미친년"]

# 사전·PROFANITY 에 없는 욕
EXTRA_PROFANITY = ["염병", "빌어먹을", "엿같", "개소리", "아가리", "처먹", "븅신", "개빡", "개짜증", "씹"]
# 욕설 문자열을 품은 일반 단어. 검사 전에 지운다 (게이머의 '게이', 불이 꺼져의 '꺼져' 등)
BENIGN = ["게이머", "게이밍", "게이트", "게이지", "새끼손가락", "걸레질", "영향을 미친", "미친다",
          "불이 꺼져", "불 꺼져", "화면이 꺼져", "전원이 꺼져", "꺼져 있", "꺼져서", "꺼져버", "다 뒤져", "뒤져 봤", "뒤져봤"]


def profanity_terms(lexicon_path):
    lex = load_lexicon(lexicon_path)
    terms = set(PROFANITY) | set(EXTRA_PROFANITY)
    for entries in lex.values():
        for word, _, variants in entries:
            terms.add(word)
            terms.update(variants)
    # 한 글자('개')는 일반 단어 대부분에 들어 있어서 뺀다. '개-' 강조는 EXTRA_PROFANITY 로 일부 잡는다
    # 긴 것부터: strip_profanity 가 '아 씨발'을 '씨발'보다 먼저 지우게
    return sorted((t for t in terms if len(t.replace(" ", "")) >= 2), key=lambda t: (-len(t), t))


def _mask_benign(text):
    for b in BENIGN:
        text = text.replace(b, " ")
    return text


def find_profanity(text, terms):
    """처음 찾은 욕설 (없으면 None)."""
    t = _mask_benign(text)
    return next((w for w in terms if w in t), None)


def strip_profanity(text, terms):
    """욕설 문자열을 지운 문장 (순화 결과와 비교할 기준)."""
    for w in terms:
        if w in text and w in _mask_benign(text):
            text = text.replace(w, " ")
    return " ".join(text.split())
