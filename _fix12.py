"""Dua pengukuran terakhir sebelum menulis perbaikan.

BAGIAN 1 — cacat #1 (titik hitam sisa).
Jalur tanpa fill_mask memakai ink_mask apa adanya, dan docstringnya sendiri
mencatat itu menyisakan coretan. Calon: ink_mask DIDILATASI lalu DIKURUNG ke
interior balon (bubble_mask). Kurungan itulah yang membuat dilatasi aman —
garis balon berada di luar interior, jadi ia mustahil termakan.
Diukur: piksel gelap (tinta Jepang) yang MASIH ada di dalam interior setelah
tiap kandidat mask, plus piksel garis balon yang tersentuh (harus 0).

BAGIAN 2 — cacat #4, sapuan ambang cakupan yang lebih rapat, plus penilai
yang SADAR LOBUS: kalau interior terpotong dinding partisi, teks memang tidak
bisa dipusatkan melintasi dinding, jadi ketimpangan harus dinilai di dalam
lobus yang benar-benar bisa dipakai — bukan terhadap seluruh kotak balon.
"""
import sys, os, pickle, numpy as np, cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MANGATL_WORK", ROOT)
os.environ.setdefault("MANGATL_ROOT", os.path.join(ROOT, ".stage"))
sys.path.insert(0, os.path.join(ROOT, '.stage'))
import typeset
from config import SETTINGS

img = cv2.cvtColor(cv2.imread(os.path.join(ROOT, 'jepang_002.webp')), cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
with open(os.path.join(ROOT, '.pagediag.pkl'), 'rb') as fh:
    st = pickle.load(fh)


def el(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(int(k) | 1, 3),) * 2)


print("== BAGIAN 1: sisa tinta di jalur ink_mask ==", flush=True)
print("  ink=piksel tinta lolos mask lama | grow=lolos mask baru | "
      "garis=piksel garis balon tersentuh", flush=True)
tot_old = tot_new = tot_line = 0
for d in st['regs']:
    ink0, bm = d['ink_mask'], d['bubble_mask']
    if ink0 is None:
        continue
    bx1, by1, bx2, by2 = d['bubble_bbox'] or d['bbox']
    x1, y1, x2, y2 = d['bbox']
    if bm is None or bm.shape[:2] != (by2 - by1, bx2 - bx1):
        bm = np.full((by2 - by1, bx2 - bx1), 255, np.uint8)

    # semua di koordinat halaman
    inkp = np.zeros(gray.shape, np.uint8)
    mh, mw = ink0.shape[:2]
    yy2, xx2 = min(y2, y1 + mh), min(x2, x1 + mw)
    inkp[y1:yy2, x1:xx2] = ink0[:yy2 - y1, :xx2 - x1]
    itr = np.zeros(gray.shape, np.uint8)
    itr[by1:by1 + bm.shape[0], bx1:bx1 + bm.shape[1]] = bm

    inside = itr > 0
    dark = (gray < 128) & inside            # tinta Jepang di dalam interior
    if not dark.any():
        continue
    old = int((dark & (inkp == 0)).sum())
    grown = cv2.bitwise_and(cv2.dilate(inkp, el(3)), itr)
    new = int((dark & (grown == 0)).sum())
    # garis balon = piksel gelap di kotak balon TAPI di luar interior
    outline = (gray[by1:by2, bx1:bx2] < 128) & (bm == 0)
    touch = int((outline & (grown[by1:by2, bx1:bx2] > 0)).sum())
    tot_old += old; tot_new += new; tot_line += touch
    print(f"  r{d['idx']:<2d} dark={int(dark.sum()):6d} ink={old:5d} "
          f"grow={new:5d} garis={touch}", flush=True)
print(f"  TOTAL ink={tot_old} grow={tot_new} garis_tersentuh={tot_line}", flush=True)
print(f"  ambang find_residue sekarang: max(30, 0.002*w*h)", flush=True)
for d in st['regs'][:3]:
    w = d['bbox'][2] - d['bbox'][0]; h = d['bbox'][3] - d['bbox'][1]
    print(f"    r{d['idx']} {w}x{h} -> ambang {max(30, int(0.002*w*h))}", flush=True)

print("\n== BAGIAN 2: sapuan ambang cakupan, penilai sadar-lobus ==", flush=True)
typeset.setup_fonts(verbose=False)
typeset.set_page_width(1134)
FP = typeset.FONT_USED
ORIG = typeset.block_slack


def make(cov):
    def f(mask, cx, pad, w_first, w_last, ink_a, ink_b):
        mh = mask.shape[0]
        w_ref = max(float(w_first), float(w_last), 1.0)
        a = int(max(cx - w_ref / 2, 0)); b = int(min(cx + w_ref / 2, mask.shape[1]))
        if b <= a:
            return ORIG(mask, cx, pad, w_first, w_last, ink_a, ink_b)
        flags = (mask[:, a:b] > 0).mean(1) >= cov
        run = typeset.free_run(flags, ink_a, ink_b)
        if run is None:
            return ORIG(mask, cx, pad, w_first, w_last, ink_a, ink_b)
        return ink_a - max(run[0], pad), min(run[1], mh - pad) - ink_b
    return f


def judge(mask, lines, y, size, lobe=True):
    """Ketimpangan visual. lobe=True: dinilai di dalam lobus tempat tinta hidup."""
    font = typeset._font(FP, size)
    lh = typeset._line_height(font)
    it, ib = typeset._ink_band(FP, size)
    ax = typeset.line_axis(mask, lines, y, size, FP)
    wmax = max(typeset._measure(ln, font) for ln in lines)
    a = int(max(ax - wmax / 2, 0)); b = int(min(ax + wmax / 2, mask.shape[1]))
    if b <= a:
        return None
    rows = np.flatnonzero((mask[:, a:b] > 0).any(1))
    if rows.size == 0:
        return None
    ia, ibb = y + it, y + (len(lines) - 1) * lh + ib
    if lobe:
        cuts = np.flatnonzero(np.diff(rows) > 1)
        starts = np.concatenate(([0], cuts + 1)); ends = np.concatenate((cuts, [rows.size - 1]))
        best = None
        for s, e in zip(starts, ends):
            lo, hi = int(rows[s]), int(rows[e])
            ov = min(hi, ibb) - max(lo, ia)
            if best is None or ov > best[0]:
                best = (ov, lo, hi)
        lo, hi = best[1], best[2]
    else:
        lo, hi = int(rows[0]), int(rows[-1])
    return abs((ia - lo) - (hi - ibb))


def oval(w, h, tail=0, cut_top=0, notch=0):
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (w // 2 - 3, h // 2 - 3), 0, 0, 360, 255, -1)
    if tail:
        cv2.rectangle(m, (w // 2 - tail // 2, h - 3), (w // 2 + tail // 2, h - 1), 255, -1)
    if cut_top:
        m[:cut_top] = 0
    if notch:
        m[h // 2 + notch:h // 2 + notch + 3] = 0
    return m


VARIANTS = [("V1", ORIG)] + [(f"c{int(c*100):02d}", make(c))
                             for c in (0.10, 0.20, 0.30, 0.40, 0.50)]
TEXTS = ["YOU MAY NOT HAVE HAD MANY EJACULATIONS.",
         "BUT THE VOLUME AND DISTANCE OF EACH ONE WAS INCREDIBLE.",
         "SO, AFTER THIS...WAIT...", "HUH?"]
CASES = [("oval", oval(110, 200)), ("oval+ekor", oval(110, 200, tail=14)),
         ("ekor panjang", oval(110, 230, tail=10)),
         ("terpotong atas", oval(110, 200, cut_top=40)),
         ("dinding partisi", oval(110, 220, notch=60)),
         ("sempit tinggi", oval(70, 240, tail=10))]

syn = {n: [] for n, _ in VARIANTS}
pag = {n: [] for n, _ in VARIANTS}
ovr = {n: 0 for n, _ in VARIANTS}
sizes = {n: [] for n, _ in VARIANTS}

masks = [(f"{s}|{t[:18]}", m, t) for s, m in CASES for t in TEXTS]
for label, m, t in masks:
    cells = []
    for name, fn in VARIANTS:
        typeset.block_slack = fn
        try:
            size, lines, y, over = typeset.fit(t, m, typeset.region_font_cap(m), FP)
        finally:
            typeset.block_slack = ORIG
        j = judge(m, lines, y, size) if lines else None
        if j is not None:
            syn[name].append(j); sizes[name].append(size)
        ovr[name] += int(bool(over))
        cells.append(f"{name}={'-' if j is None else j:>3}")
    print(f"  {label:34s} " + " ".join(cells), flush=True)

PROBE = "SOME ENGLISH TEXT FOR THIS BALLOON"
for d in st['regs']:
    bx1, by1, bx2, by2 = d['bubble_bbox'] or d['bbox']
    m = d['bubble_mask']
    if m is None or m.shape[:2] != (by2 - by1, bx2 - bx1):
        m = np.full((by2 - by1, bx2 - bx1), 255, np.uint8)
    cells = []
    for name, fn in VARIANTS:
        typeset.block_slack = fn
        try:
            size, lines, y, over = typeset.fit(PROBE, m, typeset.region_font_cap(m), FP)
        finally:
            typeset.block_slack = ORIG
        j = judge(m, lines, y, size) if lines else None
        if j is not None:
            pag[name].append(j); sizes[name].append(size)
        ovr[name] += int(bool(over))
        cells.append(f"{name}={'-' if j is None else j:>3}")
    print(f"  r{d['idx']:<2d}{'':30s} " + " ".join(cells), flush=True)

print("\n== RINGKASAN (sadar-lobus; kecil lebih baik) ==", flush=True)
for name, _ in VARIANTS:
    a = np.array(syn[name] or [0]); b = np.array(pag[name] or [0])
    z = np.array(sizes[name] or [0])
    print(f"  {name:5s} sintetis mean={a.mean():5.1f} max={a.max():3d} | "
          f"halaman mean={b.mean():5.1f} max={b.max():3d} | "
          f"font mean={z.mean():4.1f} | over={ovr[name]}", flush=True)
