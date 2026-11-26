"""봇에서 쓸 메시지 판단: (1차 필터) → 순화 → 자기 재검사 → 실패하면 다른 순화 찾기 → 안전장치.

결과(action)
  pass    : 그대로 둔다 (1차 필터에 안 걸림, 또는 모델이 바꿀 게 없다고 판단)
  rewrite : 원문을 지우고 output(순화문)을 다시 올린다
  block   : 다시 올리지 않는다 (삭제·경고). 다른 순화까지 다 찾아봐도 쓸 만한 게 없을 때만

1) 순화 + 자기 재검사(self_check): 순화 결과를 모델에 다시 넣어서 또 바뀌면 아직 나쁜 표현이 남았다고 보고 다시 순화한다.
   모델이 더 바꾸지 않는 문장(고정점)이 나오거나 max_passes 에 닿으면 멈춘다.
   단어 목록이 아니라 모델 판단이라서 목록에 없는 욕도 모델이 알아보면 잡힌다.
   깨끗한 메시지는 첫 순화에서 그대로 나오므로 모델을 한 번만 쓴다.
2) 검사: 확실한 욕이 남았거나(단어 목록) 원문과 너무 다르면(지어낸 문장 의심) 실패.
3) 다른 순화 찾기(retry): 실패한 문장은 원문으로 샘플링해서 순화 후보를 retry_samples 개씩 만들고,
   검사를 통과한 후보 중 원문에 가장 가까운 것을 쓴다. retry_rounds 번까지. 그래도 없으면 block.

    mod = Moderator.from_config(cfg, model, tok, device)
    for d in mod.decide(["씨발 존나 덥네", "오늘 날씨 좋다"]):
        print(d.action, d.output, d.reason)
"""
import re
from dataclasses import dataclass

import sacrebleu

from common import ROOT
from llm_common import rewrite, rewrite_samples
from profanity import AMBIGUOUS, AMBIGUOUS_INSULT, find_profanity, looks_obfuscated, profanity_terms, strip_profanity

_NORM = re.compile(r"[\s.,!?~…]+")


def _same(a, b):
    return _NORM.sub("", a) == _NORM.sub("", b)


@dataclass
class Decision:
    action: str        # pass | rewrite | block
    output: str        # 최종 순화문 (pass 면 원문, block 이면 마지막 시도)
    reason: str        # prefilter_clean | unchanged | rewritten | unstable | retried | residual_profanity | too_different
    passes: int        # 모델이 만든 문장 수 (샘플링 후보 포함)
    similarity: float  # chrF(순화 결과, 욕을 뺀 원문). 100 에 가까울수록 원문을 지킴


class Moderator:
    def __init__(self, model, tok, model_cfg, device, terms, prefilter=True, obfuscation_check=True,
                 self_check=True, max_passes=3, min_similarity=30.0, retry_rounds=2, retry_samples=4,
                 retry_temperature=0.8, retry_top_p=0.95, batch_size=16):
        self.model, self.tok, self.model_cfg, self.device = model, tok, model_cfg, device
        self.terms, self.prefilter, self.obfuscation_check = terms, prefilter, obfuscation_check
        # 순화 결과에 욕이 남았는지 볼 때는 일반 뜻으로도 쓰이는 말(AMBIGUOUS)을 빼고, 욕하는 형태만 넣는다
        self.residual_terms = [t for t in terms if t not in AMBIGUOUS] + AMBIGUOUS_INSULT
        self.self_check, self.max_passes, self.min_similarity = self_check, max_passes, min_similarity
        self.retry_rounds, self.retry_samples = retry_rounds, retry_samples
        self.retry_temperature, self.retry_top_p, self.batch_size = retry_temperature, retry_top_p, batch_size

    @classmethod
    def from_config(cls, cfg, model, tok, device, **override):
        b = {**(cfg.get("bot") or {}), **override}
        return cls(model, tok, cfg["model"], device, profanity_terms(ROOT / b.get("lexicon", "configs/lexicon.tsv")),
                   prefilter=b.get("prefilter", True), obfuscation_check=b.get("obfuscation_check", True),
                   self_check=b.get("self_check", True), max_passes=b.get("max_passes", 3),
                   min_similarity=b.get("min_similarity", 30.0), retry_rounds=b.get("retry_rounds", 2),
                   retry_samples=b.get("retry_samples", 4), retry_temperature=b.get("retry_temperature", 0.8),
                   retry_top_p=b.get("retry_top_p", 0.95), batch_size=b.get("batch_size", 16))

    def decide(self, texts):
        res = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            if self.prefilter and not self._suspicious(t):
                res[i] = Decision("pass", t, "prefilter_clean", 0, 100.0)
            else:
                todo.append(i)

        # 1) greedy 순화를 반복한다. 한 번에 여러 문장을 배치로 돌린다
        #   self_check : 결과가 더 안 바뀔 때(고정점)까지
        #   아니면     : 결과에 목록 욕설이 남아 있을 때만 다시
        cur, passes, stable, pending = {i: texts[i] for i in todo}, {i: 0 for i in todo}, {}, list(todo)
        while pending:
            outs = rewrite(self.model, self.tok, self.model_cfg, [cur[i] for i in pending], self.device,
                           self.batch_size)
            nxt = []
            for i, o in zip(pending, outs):
                passes[i] += 1
                done = _same(o, cur[i]) if self.self_check else not find_profanity(o, self.residual_terms)
                if not (self.self_check and done):
                    cur[i] = o
                if done:
                    stable[i] = True
                elif passes[i] < self.max_passes:
                    nxt.append(i)
                else:
                    stable[i] = False
            pending = nxt

        # 2) 검사
        status = {i: self._check(texts[i], cur[i]) for i in todo}
        reason = {i: "rewritten" if stable[i] else "unstable" for i in todo}

        # 3) 실패한 문장은 샘플링으로 다른 순화 후보를 만들어 통과하는 것 중 원문에 가장 가까운 것을 고른다
        retry = [i for i in todo if status[i][0] in ("residual", "too_different")]
        for _ in range(self.retry_rounds):
            if not retry:
                break
            cands = rewrite_samples(self.model, self.tok, self.model_cfg, [texts[i] for i in retry], self.device,
                                    n=self.retry_samples, temperature=self.retry_temperature,
                                    top_p=self.retry_top_p, batch_size=self.batch_size)
            still = []
            for i, cs in zip(retry, cands):
                passes[i] += len(cs)
                best = None
                for c in cs:
                    st = self._check(texts[i], c)
                    if st[0] in ("ok", "unchanged") and (best is None or st[1] > best[1][1]):
                        best = (c, st)
                if best:
                    cur[i], status[i], reason[i] = best[0], best[1], "retried"
                else:
                    still.append(i)
            retry = still

        for i in todo:
            st, sim = status[i]
            if st == "unchanged":
                res[i] = Decision("pass", texts[i], "unchanged", passes[i], sim)
            elif st == "ok":
                res[i] = Decision("rewrite", cur[i], reason[i], passes[i], sim)
            else:
                res[i] = Decision("block", cur[i], "residual_profanity" if st == "residual" else "too_different",
                                  passes[i], sim)
        return res

    def _suspicious(self, text):
        """1차 필터: 사전 욕설이 있거나 변형 흔적이 있으면 모델로 보낸다."""
        return bool(find_profanity(text, self.terms)) or (self.obfuscation_check and looks_obfuscated(text))

    def _check(self, text, out):
        """(상태, 유사도). 상태: residual(확실한 욕 남음) | unchanged | too_different | ok"""
        sim = sacrebleu.sentence_chrf(out, [strip_profanity(text, self.terms) or text]).score
        if find_profanity(out, self.residual_terms):
            return "residual", sim
        if _same(out, text):
            return "unchanged", sim
        if sim < self.min_similarity:
            return "too_different", sim
        return "ok", sim
