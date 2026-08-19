"""Cacat #4, varian ketiga: ukur sisa ruang dari BATAS INTERIOR di kolom blok.

Hasil _center5.py: memakai satu pita acuan selebar baris terlebar dengan
_free_flags (all(1), mundur ke any(1)) memperbaiki 16 bentuk sintetis tapi
MEMPERBURUK r5 (23->43), r7 (8->48) dan r12 (59->77) di halaman asli. Sebabnya
all(1) menuntut SELURUH lebar pita bebas di satu baris; di balon oval baris
teratas/terbawah rongganya lebih sempit dari baris terlebar, jadi pita acuan
menghapus baris yang sebenarnya masih di dalam balon dan batas atas/bawah
bergeser tidak simetris.

Varian yang diuji di sini memakai CAKUPAN, bukan semua-atau-tidak: satu baris
dihitung "masih interior" kalau sebagian pita blok di baris itu interior. Itu
persis apa yang dilihat mata — batas atas dan bawah balon di kolom tempat teks
berada — dan sengaja diukur dengan aturan yang sama pada kedua ujung, tidak
lagi dua lebar berbeda.

  V1 = produksi sekarang (dua lebar berbeda, all(1))
  V2 = pita terlebar, _free_flags apa adanya      (sudah terbukti mixed)
  V3 = pita terlebar, cakupan >= COV, run tersambung
  V4 = seperti V3 dengan COV lebih longgar

Angka penilai `nyata` dihitung DI LUAR block_slack supaya tidak sirkular.
"""
import sys, os, pickle, numpy as np, cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MANGATL_WORK", ROOT)
os.environ.setdefault("MANGATL_ROOT", os.path.join(ROOT, ".stage"))
sys.path.insert(0, os.path.join(ROOT, '.stage'))
import typeset
from config import SETTINGS

typeset.setup_fonts(verbose=False)
typeset.set_page_width(1134)
FP = typeset.FONT_USED
ORIG = typeset.block_slack


def _rows_cov(mask, cx, width, cov):
    x1 = int(max(cx - width / 2, 0))
    x2 = int(min(cx + width / 2, mask.shape[1]))
    if x2 <= x1:
        return np.zeros(mask.shape[0], bool)
    return (mask[:, x1:x2] > 0).mean(1) >= cov


def make(cov):
    def f(mask, cx, pad, w_first, w_last, ink_a, ink_b):
        mh = mask.shape[0]
        w_ref = max(float(w_first), float(w_last), 1.0)
        flags = _rows_cov(mask, cx, w_ref, cov)
        run = typeset.free_run(flags, ink_a, ink_b)
        if run is None:
            return ORIG(mask, cx, pad, w_first, w_last, ink_a, ink_b)
        return ink_a - max(run[0], pad), min(run[1], mh - pad) - ink_b
    return f


def v2(mask, cx, pad, w_first, w_last, ink_a, ink_b):
    mh = mask.shape[0]
    w_ref = max(float(w_first), float(w_last), 1.0)
    run = typeset.free_run(typeset._free_flags(mask, cx, w_ref), ink_a, ink_b)
    if run is None:
        return ORIG(mask, cx, pad, w_first, w_last, ink_a, ink_b)
    return ink_a - max(run[0], pad), min(run[1], mh - pad) - ink_b


VARIANTS = [("V1", ORIG), ("V2", v2), ("V3 cov.50", make(0.50)),
            ("V4 cov.20", make(0.20)), ("V5 cov.80", make(0.80))]


def nyata(mask, lines, y, size):
    """Ketimpangan yang DILIHAT: jarak tinta ke batas interior di kolom blok."""
    font = typeset._font(FP, size)
    lh = typeset._line_height(font)
    it, ib = typeset._ink_band(FP, size)
    ax = typeset.line_axis(mask, lines, y, size, FP)
    wmax = max(typeset._measure(ln, font) for ln in lines)
    x1 = int(max(ax - wmax / 2, 0)); x2 = int(min(ax + wmax / 2, mask.shape[1]))
    if x2 <= x1:
        return None
    rows = np.flatnonzero((mask[:, x1:x2] > 0).any(1))
    if rows.size == 0:
        return None
    return abs(int((y + it) - rows[0]) - int(rows[-1] - (y + (len(lines) - 1) * lh + ib)))


def run_one(mask, text):
    out = []
    for name, fn in VARIANTS:
        typeset.block_slack = fn
        try:
            size, lines, y, over = typeset.fit(text, mask, typeset.region_font_cap(mask), FP)
        finally:
            typeset.block_slack = ORIG
        out.append((name, None if not lines else nyata(mask, lines, y, size),
                    size, len(lines), int(bool(over))))
    return out


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


TEXTS = ["YOU MAY NOT HAVE HAD MANY EJACULATIONS.",
         "BUT THE VOLUME AND DISTANCE OF EACH ONE WAS INCREDIBLE.",
         "SO, AFTER THIS...WAIT...", "HUH?"]
CASES = [("oval", oval(110, 200)), ("oval+ekor", oval(110, 200, tail=14)),
         ("ekor panjang", oval(110, 230, tail=10)),
         ("terpotong atas", oval(110, 200, cut_top=40)),
         ("dinding partisi", oval(110, 220, notch=60)),
         ("sempit tinggi", oval(70, 240, tail=10))]

tot = {n: [] for n, _ in VARIANTS}
ovr = {n: 0 for n, _ in VARIANTS}
print("== SINTETIS (angka = ketimpangan nyata, kecil lebih baik) ==", flush=True)
for sname, m in CASES:
    for t in TEXTS:
        res = run_one(m, t)
        cells = []
        for name, ny, size, n, over in res:
            if ny is not None:
                tot[name].append(ny)
            ovr[name] += over
            cells.append(f"{name}={'-' if ny is None else ny:>3}/{size}/{n}")
        print(f"  {sname:16s}|{t[:22]:22s} " + " ".join(cells), flush=True)

print("== HALAMAN ASLI ==", flush=True)
with open(os.path.join(ROOT, '.pagediag.pkl'), 'rb') as fh:
    st = pickle.load(fh)
ptot = {n: [] for n, _ in VARIANTS}
PROBE = "SOME ENGLISH TEXT FOR THIS BALLOON"
for d in st['regs']:
    bx1, by1, bx2, by2 = d['bubble_bbox'] or d['bbox']
    m = d['bubble_mask']
    if m is None or m.shape[:2] != (by2 - by1, bx2 - bx1):
        m = np.full((by2 - by1, bx2 - bx1), 255, np.uint8)
    res = run_one(m, PROBE)
    cells = []
    for name, ny, size, n, over in res:
        if ny is not None:
            ptot[name].append(ny)
        ovr[name] += over
        cells.append(f"{name}={'-' if ny is None else ny:>3}/{size}/{n}")
    print(f"  r{d['idx']:<2d} " + " ".join(cells), flush=True)

print("\n== RINGKASAN (ketimpangan nyata) ==", flush=True)
for name, _ in VARIANTS:
    a, b = np.array(tot[name] or [0]), np.array(ptot[name] or [0])
    print(f"  {name:10s} sintetis mean={a.mean():5.1f} max={a.max():3d} | "
          f"halaman mean={b.mean():5.1f} max={b.max():3d} | over={ovr[name]}", flush=True)
