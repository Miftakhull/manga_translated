"""Diagnostik satu halaman penuh, SEKALI jalan lalu di-pickle.

detect + CTD di CPU makan ~15 menit, jadi state region disimpan dan tiap
kalibrasi ulang membacanya dari cache. Yang diukur:

  A. struktur komponen fill_mask  -> penjaga kebocoran isian (cacat #3)
  B. ink_ratio vs min_ink_ratio   -> gerbang UNREADABLE (cacat #2)
  C. block_slack == (0,0)         -> "seolah terpusat" (cacat #4)
"""
import sys, os, glob, pickle, numpy as np, cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MANGATL_WORK", ROOT)
os.environ.setdefault("MANGATL_ROOT", os.path.join(ROOT, ".stage"))
os.makedirs(os.path.join(ROOT, '.stage'), exist_ok=True)
for p in glob.glob(os.path.join(ROOT, '_nbsrc', '*.py')):
    src = open(p, encoding='utf-8').read().split('\n')
    if src and src[0].startswith('%%writefile'):
        src = src[1:]
    open(os.path.join(ROOT, '.stage', os.path.basename(p)),
         'w', encoding='utf-8').write('\n'.join(src))
sys.path.insert(0, os.path.join(ROOT, '.stage'))

import config, detect, textmask as tm, erase, typeset
from config import SETTINGS

PAGE = os.path.join(ROOT, 'jepang_002.webp')
CACHE = os.path.join(ROOT, '.pagediag.pkl')

img = cv2.cvtColor(cv2.imread(PAGE), cv2.COLOR_BGR2RGB)
print('page', img.shape, flush=True)

if os.path.exists(CACHE):
    with open(CACHE, 'rb') as fh:
        st = pickle.load(fh)
    print('dari cache', CACHE, flush=True)
else:
    regs, bubs = detect.detect(img)
    soft = tm.ctd_soft_mask(img)
    for r in regs:
        tm.build_region_mask(img, r, soft)
    tm.partition_shared_interiors(img, regs)
    tm.disjoin_overlapping_interiors(img, regs)
    st = {'bubs': [tuple(int(v) for v in b) for b in bubs], 'regs': [
        {'idx': r.idx, 'bbox': r.bbox, 'bubble_bbox': r.bubble_bbox,
         'shared': r.shared_bubble_bbox, 'det_class': r.det_class,
         'det_conf': r.det_conf, 'ink_ratio': r.ink_ratio,
         'est_font_size': r.est_font_size, 'fill_bbox': r.fill_bbox,
         'fill_mask': r.fill_mask, 'ink_mask': r.ink_mask,
         'bubble_mask': r.bubble_mask}
        for r in regs]}
    with open(CACHE, 'wb') as fh:
        pickle.dump(st, fh)
    print('cached ->', CACHE, flush=True)

R = st['regs']
print(f"regions={len(R)} bubbles={len(st['bubs'])}", flush=True)

# ---------------------------------------------------------------- A. komponen
print('== A. komponen fill_mask ==', flush=True)
for d in R:
    fm, ink = d['fill_mask'], d['ink_mask']
    if fm is None or d['fill_bbox'] is None:
        print(f"  r{d['idx']} fill=None", flush=True)
        continue
    n, lab, stats, _ = cv2.connectedComponentsWithStats((fm > 0).astype(np.uint8), 8)
    # label komponen yang MEMUAT tinta region ini
    bx1, by1, bx2, by2 = d['fill_bbox']
    x1, y1, x2, y2 = d['bbox']
    inkpage = np.zeros(img.shape[:2], np.uint8)
    mh, mw = ink.shape[:2]
    yy2, xx2 = min(y2, y1 + mh), min(x2, x1 + mw)
    inkpage[y1:yy2, x1:xx2] = ink[:yy2 - y1, :xx2 - x1]
    sub = inkpage[by1:by1 + fm.shape[0], bx1:bx1 + fm.shape[1]]
    hit = set(np.unique(lab[(sub > 0) & (fm > 0)])) - {0}
    keep = np.isin(lab, list(hit)) if hit else np.zeros_like(lab, bool)
    tot = int((fm > 0).sum())
    print(f"  r{d['idx']} cc={n-1} inkcc={len(hit)} cover={tot/fm.size:.3f} "
          f"keepfrac={int(keep.sum())/max(tot,1):.4f} "
          f"covkeep={int(keep.sum())/fm.size:.3f}", flush=True)

# ---------------------------------------------------------------- B. ink gate
print('== B. gerbang ink_ratio ==', flush=True)
print(f"  min_ink_ratio={SETTINGS.min_ink_ratio}", flush=True)
for d in R:
    flag = 'GATED' if d['ink_ratio'] < SETTINGS.min_ink_ratio else 'ok'
    print(f"  r{d['idx']} ink_ratio={d['ink_ratio']:.4f} conf={d['det_conf']:.3f} "
          f"{d['det_class']} {flag}", flush=True)

# ---------------------------------------------------------------- C. slack
print('== C. block_slack (0,0) ==', flush=True)
typeset.setup_fonts(verbose=False)
typeset.set_page_width(img.shape[1])
FP = typeset.FONT_USED
PROBE = "SOME ENGLISH TEXT FOR THIS BALLOON"
zeros = 0
for d in R:
    box = d['bubble_bbox'] or d['bbox']
    bx1, by1, bx2, by2 = box
    m = d['bubble_mask']
    if m is None or m.shape[:2] != (by2 - by1, bx2 - bx1):
        m = np.full((by2 - by1, bx2 - bx1), 255, np.uint8)
    cap = typeset.region_font_cap(m)
    size, lines, sy, over = typeset.fit(PROBE, m, cap, FP)
    if not lines:
        print(f"  r{d['idx']} tanpa baris", flush=True)
        continue
    font = typeset._font(FP, size)
    lh = typeset._line_height(font)
    it, ib = typeset._ink_band(FP, size)
    pad = int(min(m.shape[:2]) * SETTINGS.pad_ratio)
    ax = typeset.line_axis(m, lines, sy, size, FP)
    up, dn = typeset.block_slack(
        m, ax, pad, typeset._measure(lines[0], font),
        typeset._measure(lines[-1], font),
        sy + it, sy + (len(lines) - 1) * lh + ib)
    # apakah free_run gagal di salah satu pita?
    t = typeset.free_run(typeset._free_flags(m, ax, typeset._measure(lines[0], font)),
                         sy + it, sy + it)
    b = typeset.free_run(typeset._free_flags(m, ax, typeset._measure(lines[-1], font)),
                         sy + (len(lines) - 1) * lh + ib,
                         sy + (len(lines) - 1) * lh + ib)
    bad = t is None or b is None
    zeros += int(bad)
    print(f"  r{d['idx']} size={size} n={len(lines)} up={up} dn={dn} "
          f"bal={abs(up-dn)} freerun_gagal={bad} over={over}", flush=True)
print(f"slack_unmeasurable={zeros}/{len(R)}", flush=True)
