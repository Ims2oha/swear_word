"""욕설 목록과 문자열 매칭 (torch 없이 쓸 수 있게 따로 둔다).

PROFANITY       : 평가 지표(prof_in/prof_out)용 고정 목록. 바꾸면 이전 평가 결과와 비교가 안 된다.
profanity_terms : lexicon.tsv + PROFANITY + EXTRA_PROFANITY. 데이터 필터와 봇의 1차 필터·잔존 검사용.
"""
import re

from augment import load_lexicon

PROFANITY = ["씨발", "시발", "ㅅㅂ", "ㅆㅂ", "병신", "ㅂㅅ", "개새끼", "좆", "존나", "졸라", "지랄", "썅", "미친놈", "미친년"]

# 사전·PROFANITY 에 없는 욕
EXTRA_PROFANITY = ["염병", "빌어먹을", "엿같", "개소리", "아가리", "처먹", "븅신", "개빡", "개짜증", "씹"]
# 한 글자지만 일반 단어에 거의 안 나와서 그대로 검사하는 것
SINGLE_OK = {"좆", "썅"}
# 일반 뜻으로도 쓰이는 말 ('죽어버린', '쓰레기 양'). 봇의 잔존 검사에서는 빼고 모델 판단에 맡긴다
AMBIGUOUS = {"죽어", "쓰레기"}
# 단 AMBIGUOUS 단어가 사람을 욕하는 형태로 남으면 잔존으로 본다 ('쓰레기 양', '쓰레기장', '죽어버린'은 안 걸림)
AMBIGUOUS_INSULT = ["쓰레기들", "쓰레기 같은", "쓰레기같은", "쓰레기 같네", "쓰레기같네", "쓰레기야", "쓰레기 새끼",
                    "쓰레기새끼", "죽어라", "죽어버려", "죽어 버려", "죽어버리라고"]
# 욕설 문자열을 품은 일반 단어. 검사 전에 지운다 (게이머의 '게이', 불이 꺼져의 '꺼져' 등)
BENIGN = ["게이머", "게이밍", "게이트", "게이지", "새끼손가락", "걸레질", "영향을 미친", "미친다",
          "시발점", "졸라맨", "졸라서", "졸라대", "졸라 대",
          "불이 꺼져", "불 꺼져", "화면이 꺼져", "전원이 꺼져", "꺼져 있", "꺼져서", "꺼져버", "다 뒤져", "뒤져 봤", "뒤져봤"]


def profanity_terms(lexicon_path):
    lex = load_lexicon(lexicon_path)
    terms = set(PROFANITY) | set(EXTRA_PROFANITY)
    for entries in lex.values():
        for word, _, variants in entries:
            terms.add(word)
            terms.update(variants)
    # 한 글자는 일반 단어에 많이 들어 있어서 뺀다 ('개', '씹'(씹다)). '개-' 강조는 EXTRA_PROFANITY 로 일부 잡는다
    # 단 SINGLE_OK 는 일반 단어에 거의 안 나와서 남긴다
    # 긴 것부터: strip_profanity 가 '아 씨발'을 '씨발'보다 먼저 지우게
    return sorted((t for t in terms if len(t.replace(" ", "")) >= 2 or t in SINGLE_OK), key=lambda t: (-len(t), t))


def _mask_benign(text):
    for b in BENIGN:
        text = text.replace(b, " ")
    return text


def find_profanity(text, terms):
    """처음 찾은 욕설 (없으면 None)."""
    t = _mask_benign(text)
    return next((w for w in terms if w in t), None)


_JAMO = re.compile("[ㄱ-ㅎㅏ-ㅣ]")
_LAUGH = re.compile("[ㅋㅎㅠㅜ]")
_SYM = re.compile(r"[^\w\s.,!?~…'\"()\-:;%/+&@#*^ㄱ-ㅎㅏ-ㅣ가-힣]")       # 문장부호가 아닌 특수 기호
_LATIN_HANGUL = re.compile(r"[가-힣][A-Za-z]|[A-Za-z][가-힣]")             # 한 단어 안에 한글·영어가 붙음
_HAN = re.compile(r"[一-鿿㐀-䶿]")


def looks_obfuscated(text):
    """변형 욕설 흔적 (사전 매칭으로 못 잡는 걸 모델로 보내기 위한 것).
    웃음이 아닌 낱자모 2개 이상, 특수 기호 2개 이상, 한글·영어 붙여 쓰기, 한자 섞기 중 하나."""
    return (sum(1 for c in _JAMO.findall(text) if not _LAUGH.match(c)) >= 2
            or len(_SYM.findall(text)) >= 2
            or bool(_LATIN_HANGUL.search(text))
            or bool(_HAN.search(text)))


def strip_profanity(text, terms):
    """욕설 문자열을 지운 문장 (순화 결과와 비교할 기준)."""
    for w in terms:
        if w in text and w in _mask_benign(text):
            text = text.replace(w, " ")
    return " ".join(text.split())
