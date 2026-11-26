"""봇에서 쓸 메시지 판단: (1차 필터) → 순화 → 자기 재검사 → 안전장치.

결과(action)
  pass    : 그대로 둔다 (1차 필터에 안 걸림, 또는 모델이 바꿀 게 없다고 판단)
  rewrite : 원문을 지우고 output 을 다시 올려도 된다
  block   : 다시 올리지 않는다 (삭제·경고)

자기 재검사(self_check): 순화 결과를 모델에 다시 넣어서 또 바뀌면 아직 나쁜 표현이 남았다고 보고 그 결과로 다시 검사한다.
모델이 더 바꾸지 않는 문장(고정점)이 나오면 멈춘다. max_passes 안에 안 멈추면 block(unstable).
단어 목록이 아니라 모델 판단이라서, 목록에 없는 욕도 모델이 알아보면 잡힌다.
깨끗한 메시지는 첫 순화에서 그대로 나오므로 모델을 한 번만 쓴다.

마지막 안전장치(단어 목록): 고정점에도 확실한 욕이 남아 있으면 그 부분만 **로 가려서 올린다 (on_residual: block 이면 삭제).
원문과 너무 달라서 지어낸 문장이 의심되면 block.

    mod = Moderator.from_config(cfg, model, tok, device)
    for d in mod.decide(["씨발 존나 덥네", "오늘 날씨 좋다"]):
        print(d.action, d.output, d.reason)
"""
import re
from dataclasses import dataclass

import sacrebleu

from common import ROOT
from llm_common import rewrite
from profanity import (AMBIGUOUS, AMBIGUOUS_INSULT, find_profanity, looks_obfuscated, mask_profanity, profanity_terms,
                       strip_profanity)

_NORM = re.compile(r"[\s.,!?~…]+")


def _same(a, b):
    return _NORM.sub("", a) == _NORM.sub("", b)


@dataclass
class Decision:
    action: str        # pass | rewrite | block
    output: str        # 마지막 순화 결과 (pass 면 원문)
    reason: str        # prefilter_clean | unchanged | rewritten | masked | unstable | residual_profanity | too_different
    passes: int        # 모델을 몇 번 돌렸나
    similarity: float  # chrF(순화 결과, 욕을 뺀 원문). 100 에 가까울수록 원문을 지킴


class Moderator:
    def __init__(self, model, tok, model_cfg, device, terms, prefilter=True, obfuscation_check=True,
                 self_check=True, max_passes=3, min_similarity=30.0, on_residual="mask", batch_size=16):
        self.model, self.tok, self.model_cfg, self.device = model, tok, model_cfg, device
        self.terms, self.prefilter, self.obfuscation_check = terms, prefilter, obfuscation_check
        # 순화 결과에 욕이 남았는지 볼 때는 일반 뜻으로도 쓰이는 말(AMBIGUOUS)을 빼고, 욕하는 형태만 넣는다
        self.residual_terms = [t for t in terms if t not in AMBIGUOUS] + AMBIGUOUS_INSULT
        self.self_check, self.max_passes = self_check, max_passes
        self.min_similarity, self.on_residual, self.batch_size = min_similarity, on_residual, batch_size

    @classmethod
    def from_config(cls, cfg, model, tok, device, **override):
        b = {**(cfg.get("bot") or {}), **override}
        return cls(model, tok, cfg["model"], device, profanity_terms(ROOT / b.get("lexicon", "configs/lexicon.tsv")),
                   prefilter=b.get("prefilter", True), obfuscation_check=b.get("obfuscation_check", True),
                   self_check=b.get("self_check", True), max_passes=b.get("max_passes", 3),
                   min_similarity=b.get("min_similarity", 30.0), on_residual=b.get("on_residual", "mask"),
                   batch_size=b.get("batch_size", 16))

    def decide(self, texts):
        res = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            if self.prefilter and not self._suspicious(t):
                res[i] = Decision("pass", t, "prefilter_clean", 0, 100.0)
            else:
                todo.append(i)

        # 모델을 반복해서 돌린다. 한 번에 여러 문장을 배치로 돌린다
        #   self_check : 결과가 더 안 바뀔 때(고정점)까지
        #   아니면     : 결과에 목록 욕설이 남아 있을 때만 다시
        cur, passes, stable, pending = {i: texts[i] for i in todo}, {i: 0 for i in todo}, {}, list(todo)
        while pending:
            outs = rewrite(self.model, self.tok, self.model_cfg, [cur[i] for i in pending], self.device,
                           self.batch_size)
            nxt = []
            for i, o in zip(pending, outs):
                passes[i] += 1
                if self.self_check:
                    if _same(o, cur[i]):           # 모델이 더 바꿀 게 없다고 봄
                        stable[i] = True
                        continue
                    cur[i] = o
                else:
                    cur[i] = o
                    if not find_profanity(o, self.residual_terms):
                        stable[i] = True
                        continue
                if passes[i] < self.max_passes:
                    nxt.append(i)
                else:
                    stable[i] = not self.self_check  # self_check 에서 끝까지 안 멈추면 불안정
            pending = nxt

        for i in todo:
            res[i] = self._judge(texts[i], cur[i], passes[i], stable[i])
        return res

    def _suspicious(self, text):
        """1차 필터: 사전 욕설이 있거나 변형 흔적이 있으면 모델로 보낸다."""
        return bool(find_profanity(text, self.terms)) or (self.obfuscation_check and looks_obfuscated(text))

    def _judge(self, text, out, passes, stable):
        # 확실한 욕이 남았으면 지우지 않고 그 부분만 가려서 올린다 (on_residual: block 이면 삭제)
        residual = bool(find_profanity(out, self.residual_terms))
        if residual:
            if self.on_residual == "block":
                return Decision("block", out, "residual_profanity", passes, 0.0)
            out = mask_profanity(out, self.residual_terms)
        sim = sacrebleu.sentence_chrf(out, [strip_profanity(text, self.terms) or text]).score
        if _same(out, text):
            return Decision("pass", text, "unchanged", passes, sim)
        if sim < self.min_similarity:
            return Decision("block", out, "too_different", passes, sim)
        reason = "masked" if residual else ("unstable" if not stable else "rewritten")
        return Decision("rewrite", out, reason, passes, sim)
