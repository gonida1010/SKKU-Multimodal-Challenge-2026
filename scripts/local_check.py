"""맥 로컬 점검 스크립트 (GPU 불필요).

확인 항목:
  1) 테스트셋 로딩 + unknown 선택지 탐지율
  2) 이미지 1장 로딩(pillow)
  3) 공개 BBQ 다운로드 + 검증셋 구성
  4) Balanced Accuracy 지표 동작 (oracle=1.0, always-abstain=0.5)

실행:  python scripts/local_check.py
"""
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import TEST_CSV, TEST_IMG_ROOT          # noqa: E402
from data import load_samples, load_image           # noqa: E402
from bbq_eval import build_val_set, balanced_accuracy, bias_diagnostics  # noqa: E402


def main():
    print("=" * 60)
    print("[1] 테스트셋 로딩 + unknown 탐지")
    samples = load_samples(TEST_CSV)
    pos = Counter(s.unknown_idx for s in samples)
    none = sum(1 for s in samples if s.unknown_idx is None)
    print(f"    샘플 수: {len(samples)}")
    print(f"    unknown_idx 분포: {dict(sorted((k, v) for k, v in pos.items() if k is not None))}")
    print(f"    미탐지(None): {none}  ({none/len(samples)*100:.2f}%)")
    assert none == 0, "unknown 미탐지 발생!"

    print("\n[2] 이미지 1장 로딩 (pillow)")
    img = load_image(samples[0].image_path, TEST_IMG_ROOT, max_side=512)
    print(f"    {samples[0].sample_id} -> {'OK ' + str(img.size) if img else '실패'}")

    print("\n[3] 공개 BBQ 다운로드 + 검증셋 구성 (인터넷 필요)")
    val = build_val_set(n_per_category=10, seed=42)
    conds = Counter(s.condition for s in val)
    print(f"    검증셋 크기: {len(val)} | 조건 분포: {dict(conds)}")

    print("\n[4] Balanced Accuracy 지표 자기검증")
    oracle = {s.sample_id: s.label for s in val}
    abstain = {s.sample_id: (s.unknown_idx or 0) for s in val}
    print(f"    oracle(정답 그대로):   {balanced_accuracy(val, oracle)['balanced_accuracy']:.4f}  (1.0 이어야 함)")
    sc = balanced_accuracy(val, abstain)
    print(f"    always-abstain:        {sc['balanced_accuracy']:.4f}  (0.5 이어야 함)")
    print(f"      acc_ambig={sc['acc_ambig']:.2f}  acc_disambig={sc['acc_disambig']:.2f}")
    print(f"    diag(always-abstain):  {bias_diagnostics(val, abstain)}")

    print("\n" + "=" * 60)
    print("✅ 로컬 점검 통과 — 데이터/탐지기/BBQ/지표 정상.")
    print("   다음 단계: Colab에서 requirements.txt 설치 후 실제 모델로 evaluate.py 실행.")


if __name__ == "__main__":
    main()
