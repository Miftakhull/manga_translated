"""Uji calon perbaikan cacat #4 SEBELUM menyentuh produksi (monkeypatch).

Diagnosis terukur (_center4.py): `free_run` TIDAK pernah gagal (gagal_run=00 di
24 kasus), jadi jalur `return 0, 0` bukan penyebabnya. Penyebabnya: block_slack
mengukur ruang ATAS pada pita selebar baris PERTAMA dan ruang BAWAH pada pita
selebar baris TERAKHIR. Di balon oval kedua pita menyempit pada laju berbeda,
jadi blok yang melenceng 14-52 px tetap dilaporkan bal=0 dan MENANG di
pemindaian n (`if bal <= tol: break`), mengalahkan kandidat centroid.

Calon perbaikan: satu pita acuan selebar baris TERLEBAR untuk kedua ujung, dan
run yang menaungi SELURUH rentang tinta. Konsekuensinya up-dn menjadi
"pusatkan rentang tinta di dalam rongga tempat blok memang bisa hidup" —
simetris secara konstruksi, dan ekor balon yang lebih sempit dari blok
otomatis tidak ikut terhitung.

Yang dijaga: `nyata` (ketimpangan visual, diukur bebas dari block_slack) harus
TURUN di bentuk sintetis dan TIDAK memburuk di 13 region halaman asli, dengan
over tetap False.
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


def patched(mask, cx, pad, w_first, w_last, ink_a, ink_b):
    mh = mask.shape[0]
    w_ref = max(float(w_first), float(w_last), 1.0)
    flags = typeset._free_flags(mask, cx, w_ref)
    run = typeset.free_run(flags, ink_a, ink_b)
    if run is None:
        rows = np.flatnonzero(flags)
        if rows.size == 0:
            rows = np.flatnonzero((mask > 0).any(1))
        if rows.size == 0:
            return 0, 0
        run = (int(rows[0]), int(rows[-1]))
    return ink_a - max(run[0], pad), min(run[1], mh - pad) - ink_b


def real_balance(mask, lines, y, size):
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
    ia, ibb = y + it, y + (len(lines) - 1) * lh + ib
    return int(ia - rows[0]), int(rows[-1] - ibb)


def measure(mask, text):
    """(nyata, size, n, over, verify_ok) untuk block_slack yang aktif."""
    cap = typeset.region_font_cap(mask)
    size, lines, y, over = typeset.fit(text, mask, cap, FP)
    if not lines:
        return None
    rb = real_balance(mask, lines, y, size)
    ok, _l, _y = typeset.layout(text, mask, size, FP)
    return (None if rb is None else abs(rb[0] - rb[1])), size, len(lines), over, rb


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
SHAPES = [("oval", oval(110, 200)), ("oval+ekor", oval(110, 200, tail=14)),
          ("ekor panjang", oval(110, 230, tail=10)),
          ("terpotong atas", oval(110, 200, cut_top=40)),
          ("dinding partisi", oval(110, 220, notch=60)),
          ("sempit tinggi", oval(70, 240, tail=10))]

print("== SINTETIS ==", flush=True)
worse = better = 0
for sname, m in SHAPES:
    for t in TEXTS:
        typeset.block_slack = ORIG
        a = measure(m, t)
        typeset.block_slack = patched
        b = measure(m, t)
        typeset.block_slack = ORIG
        if a is None or b is None or a[0] is None or b[0] is None:
            continue
        d = b[0] - a[0]
        tag = ''
        if d < -2:
            tag = ' BAIK'; better += 1
        elif d > 2:
            tag = ' BURUK'; worse += 1
        print(f"  {sname:16s}|{t[:24]:24s} nyata {a[0]:3d}->{b[0]:3d} "
              f"sz {a[1]}->{b[1]} n {a[2]}->{b[2]} over {int(a[3])}->{int(b[3])}{tag}",
              flush=True)
print(f"  sintetis: membaik={better} memburuk={worse}", flush=True)

print("== HALAMAN ASLI (13 region) ==", flush=True)
with open(os.path.join(ROOT, '.pagediag.pkl'), 'rb') as fh:
    st = pickle.load(fh)
typeset.set_page_width(1134)
PROBE = "SOME ENGLISH TEXT FOR THIS BALLOON"
w2 = b2 = 0
for d in st['regs']:
    bx1, by1, bx2, by2 = d['bubble_bbox'] or d['bbox']
    m = d['bubble_mask']
    if m is None or m.shape[:2] != (by2 - by1, bx2 - bx1):
        m = np.full((by2 - by1, bx2 - bx1), 255, np.uint8)
    typeset.block_slack = ORIG
    a = measure(m, PROBE)
    typeset.block_slack = patched
    b = measure(m, PROBE)
    typeset.block_slack = ORIG
    if a is None or b is None or a[0] is None or b[0] is None:
        continue
    dd = b[0] - a[0]
    tag = ' BAIK' if dd < -2 else (' BURUK' if dd > 2 else '')
    if dd < -2:
        b2 += 1
    elif dd > 2:
        w2 += 1
    print(f"  r{d['idx']:<2d} nyata {a[0]:3d}->{b[0]:3d} sz {a[1]}->{b[1]} "
          f"n {a[2]}->{b[2]} over {int(a[3])}->{int(b[3])}{tag}", flush=True)
print(f"  halaman: membaik={b2} memburuk={w2}", flush=True)
