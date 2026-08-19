"""Cacat #4 diisolasi tanpa halaman Colab: mask sintetis berbentuk balon nyata.

Halaman lokal jepang_002 melaporkan slack_unmeasurable=0/13 dan bal 0-8 px,
jadi jalur `block_slack -> (0,0)` ADA di kode tapi tidak terpicu di sana.
Bentuk yang terlihat di cacat/nocenter.JPG berbeda: balon tinggi dengan EKOR
(protrusi sempit) dan balon yang interiornya terpotong tetangga. Dua bentuk
itulah yang membuat `free_run` melaporkan pita bebas yang salah, karena ia
mengambil run TERSAMBUNG di pita kolom selebar baris — dan ekor/dinding
partisi memutus run itu tepat di bawah teks.

Yang diukur, per bentuk:
  bal_lapor  = |up-dn| dari block_slack (yang dipercaya layout)
  bal_nyata  = selisih ruang kosong interior di ATAS vs di BAWAH tinta,
               dihitung dari kotak tinta terhadap kotak interior — inilah
               yang dilihat mata.
Kalau bal_lapor kecil tapi bal_nyata besar, layout diam-diam menerima blok
yang menempel ke satu sisi. Itu cacat #4.
"""
import sys, os, numpy as np, cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MANGATL_WORK", ROOT)
os.environ.setdefault("MANGATL_ROOT", os.path.join(ROOT, ".stage"))
sys.path.insert(0, os.path.join(ROOT, '.stage'))
import typeset
from config import SETTINGS

typeset.setup_fonts(verbose=False)
typeset.set_page_width(1134)
FP = typeset.FONT_USED


def oval(w, h, tail=0, cut_top=0, notch=0):
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (w // 2 - 3, h // 2 - 3), 0, 0, 360, 255, -1)
    if tail:
        # Ekor sempit di bawah: menyambung, tapi jauh lebih sempit dari baris.
        cv2.rectangle(m, (w // 2 - tail // 2, h - 3), (w // 2 + tail // 2, h - 1), 255, -1)
    if cut_top:
        m[:cut_top] = 0          # interior terpotong tetangga di atas
    if notch:
        # Dinding partisi horizontal tipis: interior jadi DUA komponen.
        m[h // 2 + notch:h // 2 + notch + 3] = 0
    return m


def real_balance(mask, lines, y, size):
    """Ruang interior kosong di atas vs di bawah tinta, di kolom blok."""
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
    ia, ib_ = y + it, y + (len(lines) - 1) * lh + ib
    return int(ia - rows[0]), int(rows[-1] - ib_), int(x2 - x1)


TEXTS = [
    "YOU MAY NOT HAVE HAD MANY EJACULATIONS.",
    "BUT THE VOLUME AND DISTANCE OF EACH ONE WAS INCREDIBLE.",
    "SO, AFTER THIS...WAIT...",
    "HUH?",
]
SHAPES = [
    ("oval", oval(110, 200)),
    ("oval+ekor", oval(110, 200, tail=14)),
    ("oval+ekor panjang", oval(110, 230, tail=10)),
    ("terpotong atas", oval(110, 200, cut_top=40)),
    ("dinding partisi", oval(110, 220, notch=60)),
    ("sempit tinggi", oval(70, 240, tail=10)),
]

bad = 0
for sname, m in SHAPES:
    cap = typeset.region_font_cap(m)
    for t in TEXTS:
        size, lines, y, over = typeset.fit(t, m, cap, FP)
        if not lines:
            print(f"  {sname:18s} | {t[:26]:26s} tanpa baris", flush=True)
            continue
        font = typeset._font(FP, size)
        lh = typeset._line_height(font)
        it, ib = typeset._ink_band(FP, size)
        pad = int(min(m.shape[:2]) * SETTINGS.pad_ratio)
        ax = typeset.line_axis(m, lines, y, size, FP)
        up, dn = typeset.block_slack(
            m, ax, pad, typeset._measure(lines[0], font),
            typeset._measure(lines[-1], font),
            y + it, y + (len(lines) - 1) * lh + ib)
        ft = typeset.free_run(
            typeset._free_flags(m, ax, typeset._measure(lines[0], font)),
            y + it, y + it)
        fb = typeset.free_run(
            typeset._free_flags(m, ax, typeset._measure(lines[-1], font)),
            y + (len(lines) - 1) * lh + ib, y + (len(lines) - 1) * lh + ib)
        rb = real_balance(m, lines, y, size)
        lap = abs(up - dn)
        if rb is None:
            print(f"  {sname:18s} | {t[:26]:26s} n={len(lines)} lapor={lap} nyata=?", flush=True)
            continue
        ru, rd, wpx = rb
        gap = abs(abs(ru - rd) - lap)
        flag = ''
        if lap <= max(2, lh // 2) and abs(ru - rd) > lh:
            flag = '  <== LAPOR SEIMBANG, NYATA TIMPANG'
            bad += 1
        print(f"  {sname:18s} | {t[:26]:26s} sz={size} n={len(lines)} over={int(over)} "
              f"lapor={lap:3d} nyata={abs(ru-rd):3d} (atas={ru} bawah={rd}) "
              f"gagal_run={int(ft is None)}{int(fb is None)}{flag}", flush=True)

print(f"\nkasus 'lapor seimbang tapi nyata timpang' = {bad}", flush=True)
