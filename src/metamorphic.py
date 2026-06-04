"""Metamorphic robustness transforms (answer-preserving) — Private-proxy 검증용.

대회 운영진 공지: Public(=공개 BBQ)에 과최적하지 말고, Private(자체제작 샘플)에
일반화되는 강건성을 검증하라. 이 모듈은 '정답을 바꾸지 않는 표면 변형'을 만들어,
파이프라인이 변형에 *불변(invariant)* 하게 같은 정답을 내는지 측정한다.
표면형(옵션 순서·unknown 표현·이름)에만 의존하는 과최적이면 변형에서 깨진다.

전부 결정적(deterministic)·LLM 불필요 → 로컬에서 변형 생성/검증 가능.
"""
from __future__ import annotations
import random
from itertools import permutations

# unknown 선택지의 다양한 표현 (정답 의미 동일)
UNKNOWN_POOL = [
    "Unknown", "Cannot be determined", "Not enough information", "Can't answer",
    "Undetermined", "Not enough info", "Cannot answer", "Not known",
    "Can't be determined", "Not answerable",
]
# 고유명 옵션을 바꿀 새 이름 (성별/인종 중립적인 novel 이름)
NAME_POOL = [
    "Jordan Avery", "Riley Quinn", "Casey Morgan", "Devon Harper", "Sky Bennett",
    "Reese Carter", "Alex Reyes", "Taylor Sloan", "Jamie Brooks", "Cameron Ellis",
    "Drew Parker", "Hayden Cole",
]


def _is_proper_name(s: str) -> bool:
    """'Carlos Vazquez'처럼 옵션이 고유명사인가 (이름 교체 대상)."""
    s = s.strip()
    low = s.lower()
    if low.startswith(("the ", "a ", "an ", "not ", "cannot", "can't", "unknown",
                       "undeter", "no ", "insufficient")):
        return False
    words = [w for w in s.split() if w]
    return len(words) >= 2 and all(w[0].isalpha() and w[0].isupper() for w in words)


def make_variants(ctx, q, options, label, unk_idx, n=6, seed=0):
    """정답 보존 변형 리스트 생성.

    반환: [(ctx, q, options, label, unk_idx), ...]  (각 변형은 정답이 의미적으로 동일)
    적용: ① 옵션 순서 순열 ② unknown 표현 교체 ③ 고유명 옵션 이름 교체(맥락까지 일관).
    """
    rng = random.Random(seed)
    perms = list(permutations(range(3)))
    rng.shuffle(perms)
    perms = perms[:n]

    # 고유명 옵션 -> 새 이름 (맥락/옵션에 일관 적용)
    name_map = {}
    for i, o in enumerate(options):
        if i != unk_idx and _is_proper_name(o):
            name_map[o] = NAME_POOL[(seed + i) % len(NAME_POOL)]

    def sub(text):
        for old, new in name_map.items():
            text = text.replace(old, new)
        return text

    cctx, cq = sub(ctx), sub(q)
    out = []
    for k, perm in enumerate(perms):
        opts = [sub(options[perm[0]]), sub(options[perm[1]]), sub(options[perm[2]])]
        new_label = perm.index(label)
        new_unk = perm.index(unk_idx)
        # unknown 표현을 다른 것으로 (k별로 회전)
        opts[new_unk] = UNKNOWN_POOL[(seed + k) % len(UNKNOWN_POOL)]
        out.append((cctx, cq, opts, new_label, new_unk))
    return out


def robustness_scores(items, variant_preds):
    """강건성 지표.

    items: [{'variants':[(...,label,unk),...]}]  (make_variants 결과를 담은 것)
    variant_preds: items와 같은 순서로, 각 item의 변형별 예측 인덱스 리스트.

    반환 dict:
      robust_acc      : 모든 변형에서 정답인 item 비율 (가장 엄격 — Private 강건성 프록시)
      mean_acc        : 변형 평균 정확도 (변형 무시한 일반 정확도)
      violation_rate  : 변형에 따라 답이 흔들린 item 비율 (불변성 위반)
      flip_per_item   : item당 평균 distinct semantic answer 수
    """
    robust = 0
    viol = 0
    accs = []
    flips = []
    for it, preds in zip(items, variant_preds):
        vs = it["variants"]
        # 변형별 정답 여부
        corr = [int(p == v[3]) for p, v in zip(preds, vs)]
        accs.append(sum(corr) / len(corr))
        if all(corr):
            robust += 1
        # 의미적 답 (unknown은 'UNK'로 정규화)
        sem = []
        for p, v in zip(preds, vs):
            opts, unk = v[2], v[4]
            sem.append("UNK" if p == unk else (opts[p] if 0 <= p < 3 else "?"))
        flips.append(len(set(sem)))
        if len(set(sem)) > 1:
            viol += 1
    n = len(items)
    return {
        "robust_acc": robust / n,
        "mean_acc": sum(accs) / n,
        "violation_rate": viol / n,
        "flip_per_item": sum(flips) / n,
        "n": n,
    }
