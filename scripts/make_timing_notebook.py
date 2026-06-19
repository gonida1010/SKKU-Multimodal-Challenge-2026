# v31 제출 파이프라인을 '캐시 무시 전체 재계산 + 단계별 타이머'로 바꾼 A6000 70분 측정 노트북 생성.
# 원본 함수/셀은 그대로 두고 (1) FORCE 플래그만 True로, (2) mark() 타이머만 주입, (3) 제출과 무관한 셀(4,5,6,10,11) 제외.
import json

SRC = r"C:\Users\pak10\Downloads\colab_v31_grounding_off.ipynb"
OUT = r"c:\Pyg\Projects\dacon\SKKU-Multimodal-Challenge-2026\colab_v31_TIMING_a6000.ipynb"

nb = json.load(open(SRC, encoding="utf-8"))
src = nb["cells"]
def body(i): return "".join(src[i]["source"])

def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.splitlines(keepends=True)}
def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}

INTRO = """# ⏱️ v31 전체 실행 A6000 70분 타이밍 측정

**목적:** 캐시 없이 *처음부터*(데이터로드 → 모델로드 → base추론 → witness+recovery추론 → 제출 CSV) 전부 돌려, 대회 환경(RTX A6000 48GB, Test 8500건)에서 **70분 안에 끝나는지** 측정한다.

**실행 방법**
1. 런타임 = **RTX A6000 48GB**(대회와 동일 GPU)에서 실행하세요. A100에서 재면 A6000보다 빠르므로 통과해도 여유를 두고 판단.
2. 셀 0(설치) 실행 → **런타임 > 세션 다시 시작**.
3. 셀 1부터 **순서대로 끝까지** 실행. 타이머는 셀 1(모델 로드)에서 시작합니다(pip install은 대회 시간제한에 보통 미포함이라 제외).
4. 마지막 셀이 **단계별 시간 + 총 소요 + 70분 PASS/FAIL**을 출력합니다.

**이 노트북과 v31 제출본의 차이:** 로직/함수/프롬프트 100% 동일. `FORCE_BASE`/`FORCE_WITNESS`만 True(캐시 무시)로 두고 타이머를 넣었을 뿐. 제출과 무관한 COREVQA(4·5·6)·분석(10·11) 셀은 시간 절약을 위해 제외했습니다. 산출되는 `submission_v31_grounding_off.csv`는 제출본과 동일해야 합니다(추론 비결정성 범위 내).
"""

TIMER_INIT = """# ===== ⏱️ 타이머 시작 (셀 0 설치/재시작 이후, 모델 로드부터 계측) =====
import time as _t
_T0 = _t.time(); TIMING = {}
def mark(name):
    TIMING[name] = _t.time() - _T0
    print(f"[TIMER] {name}: 누적 {TIMING[name]/60:.1f}분")
print("=== 타이밍 측정 시작 (pip install 제외, 모델 로드부터) ===")

"""

SUMMARY = """# ===== ⏱️ 타이밍 요약 + 70분 PASS/FAIL =====
print("="*56)
print("  A6000 70분 타이밍 측정 결과 (Test 8500, 전체 스크래치)")
print("="*56)
_steps = [("model_loaded","모델 로드"), ("drive_ready","Drive/데이터 준비"),
          ("base_done","base permSC 추론"), ("recovery_done","witness+recovery 추론"),
          ("csv_written","제출 CSV 작성")]
_prev = 0.0
for _k, _lab in _steps:
    if _k in TIMING:
        _seg = TIMING[_k] - _prev
        print(f"  {_lab:22s}: 구간 {_seg/60:5.1f}분   (누적 {TIMING[_k]/60:5.1f}분)")
        _prev = TIMING[_k]
_total = TIMING.get("csv_written", _prev)
print("-"*56)
print(f"  총 소요 (모델로드 → 제출 CSV): {_total/60:.1f}분")
_ok = _total <= 70*60
print(f"  70분 한도: {'✅ PASS' if _ok else '❌ FAIL'}  (여유 {70 - _total/60:+.1f}분)")
print("-"*56)
print("  ※ pip install 시간은 제외(대회 시간제한 보통 미포함).")
print("  ※ A100에서 측정했다면 실제 A6000은 더 느리므로 마진 확보 필요.")
print("  ※ base≈한 번에 8500×3순열, witness≈unknown집합, recovery≈unknown×3순열.")
"""

cells = [md(INTRO)]
cells.append(code(body(0)))                                   # 셀 0: 설치 (그대로)
cells.append(code(TIMER_INIT + body(1) + "\nmark('model_loaded')\n"))  # 셀 1: 타이머+모델로드
cells.append(code(body(2)))                                   # 셀 2: 추론 헬퍼 (그대로)
cells.append(code(body(3) + "\nmark('drive_ready')\n"))       # 셀 3: Drive

c7 = body(7).replace("FORCE_BASE = False", "FORCE_BASE = True ")
assert "FORCE_BASE = True " in c7, "FORCE_BASE 치환 실패"
cells.append(code(c7 + "\nmark('base_done')\n"))              # 셀 7: base (강제 재추론)

c8 = body(8).replace("FORCE_WITNESS = False", "FORCE_WITNESS = True ")
assert "FORCE_WITNESS = True " in c8, "FORCE_WITNESS 치환 실패"
cells.append(code(c8 + "\nmark('recovery_done')\n"))          # 셀 8: witness+recovery (강제 재추론)

cells.append(code(body(9) + "\nmark('csv_written')\n"))       # 셀 9: 제출 CSV
cells.append(code(SUMMARY))                                   # 요약

nb["cells"] = cells
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("생성:", OUT, "| 셀", len(cells), "개")
