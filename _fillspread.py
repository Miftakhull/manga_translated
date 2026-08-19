"""Kalibrasi penjaga isian dari cache .pagediag.pkl — tanpa detect/CTD ulang.

Yang dicari: satu besaran yang membedakan "interior balon sungguhan" dari
"flood fill yang bocor lalu menutupi art". Dua kandidat yang sudah gugur:

  * cover (fraksi kotak yang terisi): 0.598-0.900 di halaman BERSIH. Band
    _DISCOVER_FILL (0.15, 0.85) akan menolak r9 (0.900) yang sekarang benar.
  * edge (fraksi tepi crop yang terisi): 0.000-0.695 di halaman bersih.
    Tidak memisahkan apa pun.

Kandidat yang diukur di sini: SEBARAN warna area isian dikurangi tinta
sendiri, memakai MAD x 1.4826 seperti erase._bg_stats. Interior balon rata
secara definisi; art tidak. Kalau halaman bersih semuanya jauh di bawah
SETTINGS.flat_std_thresh, ambang itu bisa dipakai apa adanya.
"""
import sys, os, glob, pickle, numpy as np, cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MANGATL_WORK", ROOT)
os.environ.setdefault("MANGATL_ROOT", os.path.join(ROOT, ".stage"))
sys.path.insert(0, os.path.join(ROOT, '.stage'))
import config
from config import SETTINGS

img = cv2.cvtColor(cv2.imread(os.path.join(ROOT, 'jepang_002.webp')), cv2.COLOR_BGR2RGB)
with open(os.path.join(ROOT, '.pagediag.pkl'), 'rb') as fh:
    st = pickle.load(fh)

print(f"flat_std_thresh={SETTINGS.flat_std_thresh} "
      f"noisy={SETTINGS.flat_std_thresh_noisy}", flush=True)

EL5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
rows = []
for d in st['regs']:
    fm, ink = d['fill_mask'], d['ink_mask']
    if fm is None or d['fill_bbox'] is None or ink is None:
        print(f"  r{d['idx']} fill=None", flush=True)
        continue
    bx1, by1, bx2, by2 = d['fill_bbox']
    x1, y1, x2, y2 = d['bbox']
    inkpage = np.zeros(img.shape[:2], np.uint8)
    mh, mw = ink.shape[:2]
    yy2, xx2 = min(y2, y1 + mh), min(x2, x1 + mw)
    inkpage[y1:yy2, x1:xx2] = ink[:yy2 - y1, :xx2 - x1]
    inkpage = cv2.dilate(inkpage, EL5)

    page = np.zeros(img.shape[:2], np.uint8)
    fh_, fw_ = fm.shape[:2]
    page[by1:by1 + fh_, bx1:bx1 + fw_] = fm
    sel = (page > 0) & (inkpage == 0)
    if sel.sum() < 30:
        print(f"  r{d['idx']} area bersih terlalu kecil ({int(sel.sum())})", flush=True)
        continue
    px = img[sel].reshape(-1, 3).astype(np.float32)
    med = np.median(px, axis=0)
    spread = np.median(np.abs(px - med), axis=0) * 1.4826
    mx = float(spread.max())
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    darkfrac = float((gray[sel] < 110).mean())
    rows.append((d['idx'], mx, darkfrac))
    print(f"  r{d['idx']} med={med.astype(int).tolist()} spread_max={mx:.2f} "
          f"dark={darkfrac:.4f} px={int(sel.sum())}", flush=True)

if rows:
    s = np.array([r[1] for r in rows]); dk = np.array([r[2] for r in rows])
    print(f"SPREAD min={s.min():.2f} max={s.max():.2f} p90={np.percentile(s,90):.2f}", flush=True)
    print(f"DARK   min={dk.min():.4f} max={dk.max():.4f} p90={np.percentile(dk,90):.4f}", flush=True)
