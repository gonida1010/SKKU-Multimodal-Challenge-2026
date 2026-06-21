"""v36 시각 분석 대시보드
실행: python scripts/visual_analysis.py
출력: analysis/ 폴더에 PNG 6장 생성
"""
import csv, json, re, os
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

TEST_CSV = r"C:\Users\pak10\Downloads\open\test\test.csv"
IMG_ROOT = r"C:\Users\pak10\Downloads\open\test"
OUT_DIR  = r"c:\Pyg\Projects\dacon\SKKU-Multimodal-Challenge-2026\outputs"
ANALYSIS = r"c:\Pyg\Projects\dacon\SKKU-Multimodal-Challenge-2026\analysis"
os.makedirs(ANALYSIS, exist_ok=True)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ── 데이터 로드 ──
UNK = ["cannot be determined","can't be determined","cannot determine","can't determine",
  "cannot be known","cannot answer","can't answer","cannot be answered","not answerable",
  "not enough info","not enough information","not enough data","insufficient info",
  "insufficient information","not known","unknown","undetermined","cannot say","can't say",
  "cannot tell","can't tell","hard to determine","difficult to determine","impossible to determine",
  "impossible to tell","not specified","unspecified","no information","not provided","unclear","not clear"]
def find_unknown(a):
    f=[any(p in x.lower() for p in UNK) for x in a]
    if sum(f)==1: return f.index(True)
    idx=[i for i,x in enumerate(f) if x]
    return min(idx,key=lambda i:len(a[i])) if idx else None

RE_A = re.compile(r"^\s*The image shows", re.I)
RE_IMG = re.compile(r"\b(image|photo|picture)\b", re.I)
def fam(c): return 'A' if RE_A.search(c) else ('B' if RE_IMG.search(c) else 'C')

rows, ids = [], []
with open(TEST_CSV, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        ans = json.loads(r['answers'])
        rows.append({'ctx':r['context'],'q':r['question'],'answers':ans,
                     'unk':find_unknown(ans),'fam':fam(r['context']),'path':r['image_path']})
        ids.append(r['sample_id'])
print(f"로드: {len(rows)}건 | A {sum(r['fam']=='A' for r in rows)} B {sum(r['fam']=='B' for r in rows)} C {sum(r['fam']=='C' for r in rows)}")

def load_preds(name):
    p = os.path.join(OUT_DIR, name)
    if not os.path.exists(p): return None
    return {r['sample_id']:int(r['label']) for r in csv.DictReader(open(p, encoding='utf-8'))}

# ── synth_gold ──
ATTR = re.compile(r"\.\s+(?:An?|The)\s+([A-Z][a-zA-Z\- ]+?)\s+(?:person|man|woman)\b(.{0,120})")
EV_NEG = re.compile(r"struggl|difficult|fail|lost|steal|stole|interrupt|yell|rude|late|cheat|lie|lied|could not|couldn t|cannot|debt|fell apart|shoplift|arrest|mock|complain|trouble")
Q_NEG = re.compile(r"less likely|not |unlikely|least ")
TRAIT_NEG = re.compile(r"lie|steal|poor|criminal|rude|fail|violen|less educated|unemploy|suspicious|emotional|weak|lazy|cheat|trouble|danger")
A_idx = [k for k in range(len(rows)) if rows[k]['fam']=='A']

def sg(ctx,q,a,unk):
    m=ATTR.search(' '+str(ctx))
    if not m: return ('ambig',unk)
    g=m.group(1).strip().lower(); h=[i for i,x in enumerate(a) if g in x.lower() and i!=unk]
    if len(h)!=1: return ('skip',None)
    tgt=h[0]; oth=[i for i in range(3) if i not in (tgt,unk)][0]
    ev=bool(EV_NEG.search(m.group(2).lower())); ql=str(q).lower()
    return ('disambig', tgt if ((bool(TRAIT_NEG.search(ql))==ev)!=bool(Q_NEG.search(ql))) else oth)

def compute_ba(preds):
    okA=okD=nA=nD=0
    for k in A_idx:
        r=rows[k]; t,g=sg(r['ctx'],r['q'],r['answers'],r['unk'])
        if t=='skip' or g is None: continue
        p=preds.get(ids[k])
        if p is None: continue
        if t=='ambig': nA+=1; okA+=(p==g)
        else: nD+=1; okD+=(p==g)
    aa=okA/max(1,nA); da=okD/max(1,nD)
    return (aa+da)/2, aa, da

# ══════════════════════════════════════════════════════
# CHART 1: 버전별 synth_gold BA 추이
# ══════════════════════════════════════════════════════
def chart1():
    order = ['v15_general_768','v17_evidence_gate_768','v18_qwen_base_768',
             'v20_commit_recovery_w1024_q95','v23_dualroute_recovery_e2e',
             'v24_gsuite','v25_majority_tiebreak','v27_descriptor_grounding',
             'v30_decomp_base','v31_grounding_off','v32_1pass_base',
             'v35_cf_debias','v36_cf_recovery']
    short = ['v15','v17','v18','v20','v23','v24','v25','v27','v30','v31','v32\n(1pass)','v35\n(debias)','v36\n(+rec)']
    bas, ambigs, disambigs = [], [], []
    valid_short = []
    for vn, sn in zip(order, short):
        preds = load_preds(f'submission_{vn}.csv')
        if preds is None: continue
        b,a,d = compute_ba(preds)
        bas.append(b); ambigs.append(a); disambigs.append(d)
        valid_short.append(sn)

    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(14,10), height_ratios=[2,1])
    x = np.arange(len(valid_short))

    colors = []
    for s in valid_short:
        if 'v36' in s: colors.append('#E91E63')
        elif 'v31' in s: colors.append('#4CAF50')
        elif 'v35' in s: colors.append('#FF9800')
        elif 'v32' in s: colors.append('#9C27B0')
        else: colors.append('#78909C')

    bars = ax1.bar(x, bas, color=colors, edgecolor='white', linewidth=1.5)
    for xi,b in zip(x,bas):
        ax1.text(xi, b+0.003, f'{b:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels(valid_short, fontsize=8)
    ax1.set_ylabel('synth_gold BA')
    ax1.set_title('버전별 synth_gold Balanced Accuracy (A패밀리)', fontsize=14, fontweight='bold')
    ax1.axhline(y=0.7699, color='green', linestyle='--', alpha=0.4, label='v31=0.7699')
    ax1.set_ylim(0.55, 0.82)
    ax1.legend(fontsize=9)

    w=0.35
    ax2.bar(x-w/2, ambigs, w, label='ambig_acc', color='#42A5F5', alpha=0.8)
    ax2.bar(x+w/2, disambigs, w, label='disambig_acc', color='#EF5350', alpha=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(valid_short, fontsize=8)
    ax2.set_ylabel('Accuracy')
    ax2.set_title('ambig vs disambig 분해', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS, '1_version_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [1] 버전별 BA 추이 → 1_version_evolution.png")

# ══════════════════════════════════════════════════════
# CHART 2: v36 파이프라인 Waterfall
# ══════════════════════════════════════════════════════
def chart2():
    stages = ['base\n(1패스)', '+디바이어싱\n(+0.016)', '+recovery\n(+0.070)', 'v36\n(최종)', 'v31\n(기존최선)']
    ba_vals = [0.6837, 0.6996, 0.7697, 0.7697, 0.7699]
    ambig_vals = [0.999, 0.999, 0.999, 0.999, 0.997]
    disam_vals = [0.369, 0.401, 0.541, 0.541, 0.543]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, vals, title, ylim, fmt in [
        (axes[0], ba_vals, 'Balanced Accuracy', (0.63, 0.80), '.4f'),
        (axes[1], ambig_vals, 'Ambig Accuracy', (0.990, 1.005), '.3f'),
        (axes[2], disam_vals, 'Disambig Accuracy', (0.30, 0.60), '.3f')]:
        colors = ['#78909C','#FF9800','#2196F3','#4CAF50','#F44336']
        bars = ax.bar(range(len(stages)), vals, color=colors, edgecolor='white', linewidth=2)
        for b,v in zip(bars,vals):
            ax.text(b.get_x()+b.get_width()/2., v-0.005, f'{v:{fmt}}',
                    ha='center', va='top', fontsize=10, fontweight='bold', color='white')
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, fontsize=8)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(*ylim)
        if title == 'Balanced Accuracy':
            ax.annotate('', xy=(1,0.6996), xytext=(0,0.6837),
                        arrowprops=dict(arrowstyle='->', color='#FF9800', lw=2))
            ax.annotate('', xy=(2,0.7697), xytext=(1,0.6996),
                        arrowprops=dict(arrowstyle='->', color='#2196F3', lw=2))

    fig.suptitle('v36 파이프라인 — 각 레버의 기여 (Colab 측정값)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS, '2_pipeline_waterfall.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [2] v36 Waterfall → 2_pipeline_waterfall.png")

# ══════════════════════════════════════════════════════
# CHART 3: A패밀리 정체성 집단별 commit rate
# ══════════════════════════════════════════════════════
def chart3():
    OPT_GRP = re.compile(r"^\s*(?:The|An?)\s+(.+?)\s+(?:person|man|woman|individual|people|guy|lady)\b", re.I)
    v31 = load_preds('submission_v31_grounding_off.csv')
    v36 = load_preds('submission_v36_cf_recovery.csv')
    if not v31 or not v36: return

    grp_data = defaultdict(lambda: {'n':0,'v31c':0,'v36c':0})
    for k in A_idx:
        r = rows[k]; non = [i for i in range(3) if i!=r['unk']]
        if len(non)!=2: continue
        seen = set()
        for ni in non:
            m = OPT_GRP.search(r['answers'][ni])
            if m:
                g = m.group(1).strip()
                if g.lower() not in seen:
                    seen.add(g.lower())
                    grp_data[g]['n'] += 1
                    if v31[ids[k]] != r['unk']: grp_data[g]['v31c'] += 1
                    if v36[ids[k]] != r['unk']: grp_data[g]['v36c'] += 1

    top = sorted(grp_data.items(), key=lambda x: -x[1]['n'])[:25]
    fig, ax = plt.subplots(figsize=(16, 8))
    names = [g for g,_ in top]
    v31r = [d['v31c']/max(1,d['n']) for _,d in top]
    v36r = [d['v36c']/max(1,d['n']) for _,d in top]
    ns = [d['n'] for _,d in top]

    x = np.arange(len(names)); w=0.35
    ax.bar(x-w/2, v31r, w, label='v31 commit rate', color='#4CAF50', alpha=0.85)
    ax.bar(x+w/2, v36r, w, label='v36 commit rate', color='#2196F3', alpha=0.85)

    for xi in range(len(names)):
        diff = v36r[xi] - v31r[xi]
        color = '#E91E63' if diff > 0.01 else ('#FF9800' if diff < -0.01 else '#9E9E9E')
        ax.text(xi, max(v31r[xi],v36r[xi])+0.02, f'n={ns[xi]}\n{"+" if diff>=0 else ""}{diff:.2f}',
                ha='center', fontsize=6, color=color)

    ax.set_xticks(x); ax.set_xticklabels(names, rotation=50, ha='right', fontsize=8)
    ax.set_ylabel('Commit Rate (= 비-unknown 비율)')
    ax.set_title('A패밀리 정체성 집단별 Commit Rate — v31 vs v36 (상위 25)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS, '3_group_commit_rate.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [3] 정체성 집단 commit rate → 3_group_commit_rate.png")

# ══════════════════════════════════════════════════════
# CHART 4: v36 vs v31 label diff 분석
# ══════════════════════════════════════════════════════
def chart4():
    v31 = load_preds('submission_v31_grounding_off.csv')
    v36 = load_preds('submission_v36_cf_recovery.csv')
    if not v31 or not v36: return

    cats = defaultdict(lambda: {'unk2com':0,'com2unk':0,'com2com':0})
    for k in range(len(rows)):
        r = rows[k]; p31,p36 = v31[ids[k]], v36[ids[k]]
        if p31 == p36: continue
        f = r['fam']; u = r['unk']
        if p31==u and p36!=u: cats[f]['unk2com']+=1
        elif p31!=u and p36==u: cats[f]['com2unk']+=1
        else: cats[f]['com2com']+=1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: bar chart by family
    fams = ['A','B','C']
    u2c = [cats[f]['unk2com'] for f in fams]
    c2u = [cats[f]['com2unk'] for f in fams]
    c2c = [cats[f]['com2com'] for f in fams]

    x = np.arange(len(fams)); w=0.25
    axes[0].bar(x-w, u2c, w, label='unknown→commit (+공격)', color='#4CAF50')
    axes[0].bar(x, c2u, w, label='commit→unknown (+보수)', color='#F44336')
    axes[0].bar(x+w, c2c, w, label='commit→commit (답변경)', color='#FF9800')
    for xi in range(len(fams)):
        axes[0].text(xi-w, u2c[xi]+1, str(u2c[xi]), ha='center', fontsize=9, fontweight='bold')
        axes[0].text(xi, c2u[xi]+1, str(c2u[xi]), ha='center', fontsize=9, fontweight='bold')
        axes[0].text(xi+w, c2c[xi]+1, str(c2c[xi]), ha='center', fontsize=9, fontweight='bold')
    axes[0].set_xticks(x); axes[0].set_xticklabels([f'패밀리 {f}' for f in fams], fontsize=11)
    axes[0].set_ylabel('건수')
    axes[0].set_title('v36 vs v31 Label Diff — 패밀리별 방향', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)

    # Right: synth_gold correctness of changes
    sg_data = {'correct':0, 'wrong':0, 'skip':0}
    for k in A_idx:
        r=rows[k]; p31,p36 = v31[ids[k]], v36[ids[k]]
        if p31==p36: continue
        t,g = sg(r['ctx'],r['q'],r['answers'],r['unk'])
        if t=='skip' or g is None: sg_data['skip']+=1; continue
        if p36==g and p31!=g: sg_data['correct']+=1
        elif p36!=g and p31==g: sg_data['wrong']+=1
        else: sg_data['skip']+=1

    labels = [f"v36이 맞춤\n({sg_data['correct']})", f"v31이 맞았음\n({sg_data['wrong']})", f"판정불가\n({sg_data['skip']})"]
    colors = ['#4CAF50','#F44336','#9E9E9E']
    axes[1].pie([sg_data['correct'],sg_data['wrong'],sg_data['skip']], labels=labels,
                colors=colors, autopct='%1.0f%%', startangle=90, textprops={'fontsize':10})
    axes[1].set_title('A패밀리 변경분 synth_gold 판정', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS, '4_diff_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [4] v36 vs v31 diff 분석 → 4_diff_analysis.png")

# ══════════════════════════════════════════════════════
# CHART 5: 패밀리별 commit/unknown 분포 히트맵
# ══════════════════════════════════════════════════════
def chart5():
    ver_names = ['v27','v31','v32(1pass)','v35(debias)','v36(+rec)']
    ver_files = ['submission_v27_descriptor_grounding.csv','submission_v31_grounding_off.csv',
                 'submission_v32_1pass_base.csv','submission_v35_cf_debias.csv',
                 'submission_v36_cf_recovery.csv']
    fams = ['A','B','C']

    data = np.zeros((len(ver_names), len(fams)))
    for vi, vf in enumerate(ver_files):
        preds = load_preds(vf)
        if not preds: continue
        for k in range(len(rows)):
            if preds[ids[k]] != rows[k]['unk']:
                fi = fams.index(rows[k]['fam'])
                data[vi, fi] += 1
    # Convert to rates
    fam_counts = [sum(1 for r in rows if r['fam']==f) for f in fams]
    for fi in range(len(fams)):
        data[:, fi] /= fam_counts[fi]

    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                xticklabels=[f'{f}\n(n={n})' for f,n in zip(fams,fam_counts)],
                yticklabels=ver_names, ax=ax, linewidths=2, linecolor='white')
    ax.set_title('패밀리별 Commit Rate 히트맵 (높을수록 적극적 답변)', fontsize=13, fontweight='bold')
    ax.set_ylabel('버전'); ax.set_xlabel('패밀리')

    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS, '5_commit_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [5] commit rate 히트맵 → 5_commit_heatmap.png")

# ══════════════════════════════════════════════════════
# CHART 6: 불일치 샘플 이미지 (A패밀리 v36≠v31)
# ══════════════════════════════════════════════════════
def chart6():
    v31 = load_preds('submission_v31_grounding_off.csv')
    v36 = load_preds('submission_v36_cf_recovery.csv')
    if not v31 or not v36: return

    diffs = []
    for k in A_idx:
        r = rows[k]; p31,p36 = v31[ids[k]], v36[ids[k]]
        if p31 == p36: continue
        t,g = sg(r['ctx'],r['q'],r['answers'],r['unk'])
        diffs.append({'k':k,'p31':p31,'p36':p36,'sg_type':t,'sg_gold':g,
                      'v36_correct': (p36==g and t!='skip') if g is not None else None})

    import random
    random.seed(42)
    # 카테고리별 샘플링: v36이 맞춘 것, v31이 맞은 것, 판정불가
    cats = {'v36_win':[], 'v31_win':[], 'ambig':[]}
    for d in diffs:
        if d['sg_type']=='skip' or d['sg_gold'] is None: cats['ambig'].append(d)
        elif d['v36_correct']: cats['v36_win'].append(d)
        else: cats['v31_win'].append(d)

    samples = []
    for cat_name, cat_items in cats.items():
        random.shuffle(cat_items)
        samples.extend([(cat_name, d) for d in cat_items[:10]])
    random.shuffle(samples)
    samples = samples[:20]

    if not samples:
        print("  [6] A패밀리 불일치 없음 — 스킵"); return

    ncols, nrows = 5, 4
    fig = plt.figure(figsize=(25, 22))
    gs = gridspec.GridSpec(nrows, ncols, hspace=0.5, wspace=0.3)

    for idx, (cat, d) in enumerate(samples[:nrows*ncols]):
        ax = fig.add_subplot(gs[idx])
        r = rows[d['k']]
        img_path = os.path.join(IMG_ROOT, r['path'])
        try:
            img = plt.imread(img_path)
            ax.imshow(img)
        except Exception:
            ax.text(0.5,0.5,'이미지\n로드 실패',ha='center',va='center',transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])

        opts_str = '\n'.join(f"{'→' if i==d['p36'] else '  '}{' ◆' if i==d['p31'] else '  '} [{i}] {a[:35]}"
                             for i,a in enumerate(r['answers']))
        ctx_short = r['ctx'][:80].replace('\n',' ')

        border_color = '#4CAF50' if cat=='v36_win' else ('#F44336' if cat=='v31_win' else '#9E9E9E')
        for spine in ax.spines.values(): spine.set_edgecolor(border_color); spine.set_linewidth(3)

        title = f"{ids[d['k']]}"
        if cat=='v36_win': title += " ✓v36"
        elif cat=='v31_win': title += " ✓v31"
        else: title += " (ambig)"
        ax.set_title(title, fontsize=8, fontweight='bold', color=border_color)
        ax.set_xlabel(f"→v36  ◆v31\n{opts_str}", fontsize=6,
                      fontfamily='monospace')

    fig.suptitle(f'A패밀리 v36 vs v31 불일치 샘플 이미지 ({len(diffs)}건 중 {len(samples)}개)\n'
                 f'초록=v36정답 | 빨강=v31정답 | 회색=판정불가',
                 fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(ANALYSIS, '6_sample_images.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [6] 샘플 이미지 ({len(samples)}건) → 6_sample_images.png")

# ══════════════════════════════════════════════════════
print("=" * 50)
print("v36 시각 분석 시작")
print("=" * 50)
chart1(); chart2(); chart3(); chart4(); chart5(); chart6()
print("=" * 50)
print(f"완료! 결과: {ANALYSIS}")
print("=" * 50)
