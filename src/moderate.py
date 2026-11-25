"""봇에서 쓸 메시지 판단: 1차 필터 → 순화 → 남은 욕 재순화 → 안전장치.

결과(action)
  pass    : 그대로 둔다 (1차 필터에 안 걸림, 또는 모델이 바꿀 게 없다고 판단)
  rewrite : 원문을 지우고 output 을 다시 올려도 된다
  block   : 다시 올리지 않는다 (삭제·경고). 순화 후에도 욕이 남았거나, 원문과 너무 달라서 지어낸 문장이 의심될 때

    mod = Moderator.from_config(cfg, model, tok, device)
    for d in mod.decide(["씨발 존나 덥네", "오늘 날씨 좋다"]):
        print(d.action, d.output, d.reason)
"""
import re
from dataclasses import dataclass

import sacrebleu

from common import ROOT
from llm_common import rewrite
from profanity import find_profanity, profanity_terms, strip_profanity

_NORM = re.compile(r"[\s.,!?~…]+")


@dataclass
class Decision:
    action: str        # pass | rewrite | block
    output: str        # 마지막 순화 결과 (pass 면 원문)
    reason: str        # prefilter_clean | unchanged | rewritten | residual_profanity | too_different
    passes: int        # 모델을 몇 번 돌렸나
    similarity: float  # chrF(순화 결과, 욕을 뺀 원문). 100 에 가까울수록 원문을 지킴


class Moderator:
    def __init__(self, model, tok, model_cfg, device, terms, prefilter=True, max_passes=2,
                 min_similarity=30.0, batch_size=16):
        self.model, self.tok, self.model_cfg, self.device = model, tok, model_cfg, device
        self.terms, self.prefilter, self.max_passes = terms, prefilter, max_passes
        self.min_similarity, self.batch_size = min_similarity, batch_size

    @classmethod
    def from_config(cls, cfg, model, tok, device, **override):
        b = {**(cfg.get("bot") or {}), **override}
        return cls(model, tok, cfg["model"], device, profanity_terms(ROOT / b.get("lexicon", "configs/lexicon.tsv")),
                   prefilter=b.get("prefilter", True), max_passes=b.get("max_passes", 2),
                   min_similarity=b.get("min_similarity", 30.0), batch_size=b.get("batch_size", 16))

    def decide(self, texts):
        res = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            if self.prefilter and not find_profanity(t, self.terms):
                res[i] = Decision("pass", t, "prefilter_clean", 0, 100.0)
            else:
                todo.append(i)

        # 순화 결과에 욕이 남으면 그 결과를 다시 순화 (max_passes 까지). 한 번에 여러 문장을 배치로 돌린다
        cur, passes, pending = {i: texts[i] for i in todo}, {i: 0 for i in todo}, list(todo)
        while pending:
            outs = rewrite(self.model, self.tok, self.model_cfg, [cur[i] for i in pending], self.device,
                           self.batch_size)
            nxt = []
            for i, o in zip(pending, outs):
                cur[i], passes[i] = o, passes[i] + 1
                if find_profanity(o, self.terms) and passes[i] < self.max_passes:
                    nxt.append(i)
            pending = nxt

        for i in todo:
            res[i] = self._judge(texts[i], cur[i], passes[i])
        return res

    def _judge(self, text, out, passes):
        sim = sacrebleu.sentence_chrf(out, [strip_profanity(text, self.terms) or text]).score
        if find_profanity(out, self.terms):
            return Decision("block", out, "residual_profanity", passes, sim)
        if _NORM.sub("", out) == _NORM.sub("", text):
            return Decision("pass", text, "unchanged", passes, sim)
        if sim < self.min_similarity:
            return Decision("block", out, "too_different", passes, sim)
        return Decision("rewrite", out, "rewritten", passes, sim)
