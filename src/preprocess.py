"""학습·추론 공통 전처리.

입력: 한글 사이에 낀 특수문자 삭제 (씨@발 -> 씨발, 개.새.끼 -> 개새끼), 이모지 삭제, 공백 정리.
  - ? ! ~ 는 유지. 숫자·영문은 건드리지 않음 (2시3분 보호, 시1발은 모델이 학습).
  - 유니코드 정규화(NFKC)는 하지 않음 (변형 표기를 망가뜨림).
타깃: 이모지 삭제, 공백 정리만.
"""
import re
import unicodedata

_HANGUL = "가-힣ㄱ-ㅎㅏ-ㅣ"
# KoBART 토크나이저는 단독 자모(ㅋ, ㅅ, ㅏ)를 조합용 자모(ᄏ, ᄉ, ᅡ)로 바꿔서 내보낸다.
# 출력에서 호환 자모로 되돌리기 위한 표. 초성 19개 + 중성 21개.
_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JAMO_TO_COMPAT = {0x1100 + i: ord(c) for i, c in enumerate(_CHO)}
_JAMO_TO_COMPAT.update({0x1161 + i: 0x314F + i for i in range(21)})
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⭐⭕️‍]")
_INNER_SPECIAL = re.compile(rf"(?<=[{_HANGUL}])[^\w\s?!~]+(?=[{_HANGUL}])")
_WS = re.compile(r"\s+")


def _s(text) -> str:
    return text if isinstance(text, str) else ""


def clean_target(text) -> str:
    return _WS.sub(" ", _EMOJI.sub("", _s(text))).strip()


def clean_input(text) -> str:
    text = _EMOJI.sub("", _s(text))
    text = _INNER_SPECIAL.sub("", text)
    return _WS.sub(" ", text).strip()


def restore_jamo(text: str) -> str:
    """모델 출력의 조합용 자모를 호환 자모로 복원 (ᄏᄏ -> ㅋㅋ). 조합 가능한 건 먼저 음절로 합친다(NFC)."""
    return unicodedata.normalize("NFC", text).translate(_JAMO_TO_COMPAT)


if __name__ == "__main__":
    for t in ["씨@발 존나 덥네", "개.새.끼야", "시1발 뭐하냐?!", "2시3분에 보자~", "ㅅㅂ😂 진짜", "안녕,  잘가"]:
        print(repr(t), "->", repr(clean_input(t)))
