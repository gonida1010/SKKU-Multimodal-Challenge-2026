"""v27 노트북 -> v31 (grounding OFF) ablation 노트북 생성.

원칙: v27 원본 함수를 새로 짜지 않는다(제약 #5). 최소 수정만:
  1) 셀8 recovery_permsc 의 5차 descriptor_grounding 게이트를 GROUNDING_ON 플래그로 OFF
  2) 셀9 제출 파일명 V_NAME = 'v31_grounding_off'
  3) VER(base/witness 캐시 키)는 v27 그대로 유지 -> 캐시 재사용 (재추론 없음)
"""
import json

SRC = "colab_v27_resumable.ipynb"
DST = "colab_v31_grounding_off.ipynb"

nb = json.load(open(SRC, encoding="utf-8"))
cells = nb["cells"]

# --- 셀8: 5차 게이트 OFF ---
c8 = "".join(cells[8]["source"])
old_anchor = ("    pre_final = [(j,p) for j,p in surv if j not in confirm_fail]\n"
              "    grounded_fail = set()")
new_anchor = ("    pre_final = [(j,p) for j,p in surv if j not in confirm_fail]\n"
              "    GROUNDING_ON = False  # [v31] 5차 descriptor_grounding 게이트 비활성화\n"
              "    grounded_fail = set()")
assert old_anchor in c8, "셀8 앵커1 불일치"
c8 = c8.replace(old_anchor, new_anchor)
assert "    if desc_items:\n" in c8, "셀8 앵커2 불일치"
c8 = c8.replace("    if desc_items:\n", "    if desc_items and GROUNDING_ON:\n")
cells[8]["source"] = c8.splitlines(keepends=True)

# --- 셀9: 제출 파일명 ---
c9 = "".join(cells[9]["source"])
assert "V_NAME = 'v27_descriptor_grounding'" in c9, "셀9 V_NAME 불일치"
c9 = c9.replace("V_NAME = 'v27_descriptor_grounding'", "V_NAME = 'v31_grounding_off'")
cells[9]["source"] = c9.splitlines(keepends=True)

# --- 모든 코드셀 출력 비우기 (노트북 경량화) ---
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

json.dump(nb, open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("생성:", DST)
print("- 셀8: GROUNDING_ON=False 삽입 OK")
print("- 셀9: V_NAME=v31_grounding_off OK")
print("- VER(캐시키)=v27_descriptor_grounding 유지 -> base/witness 재사용")
