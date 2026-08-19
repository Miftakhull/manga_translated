"""Cacat #1: apa yang DILIHAT verify vs apa yang tertinggal di halaman.

Temuan yang dikejar di sini: `pixel_residue` membatasi pemeriksaannya ke
`inside = region.ink_mask > 0`. Itu benar ketika erase hanya mengecat stroke
glyph. Sejak `erase_flat` mengecat SELURUH interior balon (fill_mask), area
yang dicat jauh lebih luas daripada ink_mask — jadi titik sisa yang duduk di
dalam interior TAPI di luar bekas stroke huruf Jepang tidak pernah masuk
hitungan sama sekali. Nilainya nol, gerbangnya lolos, tidak ada eskalasi.

Halaman ini punya satu contohnya: r9 menyimpan 198 px menyimpang di sudut
kanan-bawah interior (halaman y1280-1303 x608-625, gray 218-234). Diukur
dengan definisi lama: 0. Itulah "titik kotor" yang lolos.

Yang diukur di sini, per region, dengan erase disimulasikan persis seperti
produksi (flat + fill_mask + protected_guard):
  lama   = pixel_residue apa adanya (dibatasi ink_mask)
  luas   = piksel menyimpang di SELURUH area yang benar-benar dicat + interior
  blob   = komponen tersambung terbesar dari `luas` — ini yang terlihat mata
  ambang = max(30, 0.002*w*h)
"""
import sys, os, pickle, numpy as np, cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MANGATL_WORK", ROOT)
os.environ.setdefault("MANGATL_ROOT", os.path.join(ROOT, ".stage"))
sys.path.insert(0, os.path.join(ROOT, '.stage'))
from config import SETTINGS

DEV = SETTINGS.residue_deviation
page = cv2.cvtColor(cv2.imread(os.path.join(ROOT, 'jepang_002.webp')),
                    cv2.COLOR_BGR2RGB)
st = pickle.load(open(os.path.join(ROOT, '.pagediag.pkl'), 'rb'))
H, W = page.shape[:2]


def to_page(m, x1, y1):
    o = np.zeros((H, W), np.uint8)
    if m is None:
        return o
    mh, mw = m.shape[:2]
    y2, x2 = min(y1 + mh, H), min(x1 + mw, W)
    o[y1:y2, x1:x2] = m[:y2 - y1, :x2 - x1]
    return o


def biggest(m):
    if not m.any():
        return 0
    n, _l, s, _c = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return 0 if n <= 1 else int(s[1:, cv2.CC_STAT_AREA].max())


# --- simulasi erase produksi: flat + fill_mask untuk setiap region ---
clean = page.copy()
info = []
for d in st['regs']:
    fx1, fy1 = (d['fill_bbox'] or d['bbox'])[:2]
    fill = to_page(d['fill_mask'], fx1, fy1)
    bx1, by1 = (d['bubble_bbox'] or d['bbox'])[:2]
    itr = to_page(d['bubble_mask'], bx1, by1)
    ink = to_page(d['ink_mask'], d['bbox'][0], d['bbox'][1])
    if not itr.any():
        itr = fill
    sel = fill > 0
    if sel.any():
        # warna latar seperti fill_color: median interior di luar tinta
        pick = (itr > 0) & (ink == 0)
        bg = np.median(page[pick], 0) if pick.any() else np.array([255, 255, 255])
        clean[sel] = bg.astype(np.uint8)
    info.append((d, fill, itr, ink, sel))

print(f"residue_deviation={DEV}", flush=True)
print("  region   lama  blob_lama   luas  blob_luas ambang  verdict_lama "
      "verdict_luas", flush=True)
rows = []
for d, fill, itr, ink, sel in info:
    x1, y1, x2, y2 = d['bbox']
    g = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.int16)
    sub = ink[y1:y2, x1:x2] > 0
    crop = g[y1:y2, x1:x2]
    bg = float(np.median(crop[~sub])) if (~sub).any() else 255.0
    dev = np.abs(crop - bg) > DEV
    lama = dev & sub                                  # definisi pixel_residue
    # definisi luas: menyimpang di dalam area yang dicat ATAU bekas tinta,
    # yaitu seluruh permukaan yang seharusnya rata setelah erase
    scope = (sel[y1:y2, x1:x2] | sub)
    luas = dev & scope
    thr = max(30, int(0.002 * (x2 - x1) * (y2 - y1)))
    bl, bg2 = biggest(lama), biggest(luas)
    rows.append((int(lama.sum()), bl, int(luas.sum()), bg2, thr))
    print(f"  r{d['idx']:<2d} {int(lama.sum()):7d} {bl:9d} {int(luas.sum()):7d} "
          f"{bg2:9d} {thr:6d}  {'GAGAL' if lama.sum() > thr else 'lolos':6s} "
          f"      {'GAGAL' if luas.sum() > thr else 'lolos'}", flush=True)

a = np.array(rows)
print(f"\n  TOTAL lama={a[:,0].sum()} luas={a[:,2].sum()}", flush=True)
print(f"  blob terbesar: lama={a[:,1].max()} luas={a[:,3].max()}", flush=True)
tersembunyi = (a[:, 0] <= a[:, 4]) & (a[:, 2] > a[:, 4])
print(f"  region yang LOLOS definisi lama tapi GAGAL definisi luas: "
      f"{int(tersembunyi.sum())}/{len(a)}  -> idx "
      f"{[st['regs'][i]['idx'] for i in np.flatnonzero(tersembunyi)]}", flush=True)
