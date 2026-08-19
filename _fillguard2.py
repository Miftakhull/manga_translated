"""Tahap dua: apakah PEMANGKASAN KOMPONEN cukup jadi penjaga kebocoran?

Aturan yang diuji: dari `interior`, simpan HANYA komponen tersambung yang
memuat tinta region ini sendiri. Alasannya struktural, bukan ambang:

  * `_interior_from_crop` punya jalur mundur `interior = binv` kalau flood
    fill hasilnya < 5% crop. `binv` = SELURUH piksel terang crop, termasuk
    art di luar balon. Art terpotong-potong garis gambar jadi ia pecah jadi
    banyak komponen, dan hanya satu yang memuat tinta balon -> terpangkas.
  * r9 halaman bersih: cc=11, keepfrac=0.9073. Jadi kebocoran ini SUDAH ada
    di halaman lokal, bukan cuma di Colab.

Yang diukur:
  1. sesudah pangkas, cover tiap region bersih (jangan sampai ada yang
     kehilangan interiornya).
  2. sesudah pangkas, sebaran warna pita-luar (jauh dari tinta) — kalau r9
     turun ke level 12 region lain, pangkas memang mengangkat art-nya.
  3. jalur mundur binv disimulasikan: berapa keepfrac-nya.
"""
import sys, os, pickle, numpy as np, cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MANGATL_WORK", ROOT)
os.environ.setdefault("MANGATL_ROOT", os.path.join(ROOT, ".stage"))
sys.path.insert(0, os.path.join(ROOT, '.stage'))
from config import SETTINGS

img = cv2.cvtColor(cv2.imread(os.path.join(ROOT, 'jepang_002.webp')), cv2.COLOR_BGR2RGB)
with open(os.path.join(ROOT, '.pagediag.pkl'), 'rb') as fh:
    st = pickle.load(fh)

EL5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def el(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(int(k) | 1, 3),) * 2)


def keep_ink_components(interior: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Buang komponen interior yang tidak memuat tinta region ini."""
    n, lab = cv2.connectedComponents((interior > 0).astype(np.uint8), 8)
    if n <= 2:
        return interior
    hit = set(np.unique(lab[(ink > 0) & (interior > 0)])) - {0}
    if not hit:
        return interior
    return np.where(np.isin(lab, list(hit)), 255, 0).astype(np.uint8)


def band_stats(crop, fill, ink, est):
    inkd = cv2.dilate(ink, EL5)
    near = cv2.dilate(ink, el(2 * max(est, 6) + 1))
    inner = (fill > 0) & (near > 0) & (inkd == 0)
    outer = (fill > 0) & (near == 0)
    if int(inner.sum()) < 30 or int(outer.sum()) < 30:
        return None
    def sp(px):
        med = np.median(px, axis=0)
        return med, float((np.median(np.abs(px - med), axis=0) * 1.4826).max())
    mi, si = sp(crop[inner].reshape(-1, 3).astype(np.float32))
    mo, so = sp(crop[outer].reshape(-1, 3).astype(np.float32))
    return si, so, float(np.abs(mo - mi).max())


rows = []
for d in st['regs']:
    fm, ink0 = d['fill_mask'], d['ink_mask']
    if fm is None or d['fill_bbox'] is None or ink0 is None:
        print(f"  r{d['idx']} fill=None", flush=True)
        continue
    bx1, by1, bx2, by2 = d['fill_bbox']
    x1, y1, _, _ = d['bbox']
    crop = img[by1:by1 + fm.shape[0], bx1:bx1 + fm.shape[1]]
    ink = np.zeros(fm.shape[:2], np.uint8)
    mh, mw = ink0.shape[:2]
    oy, ox = y1 - by1, x1 - bx1
    hh, ww = min(mh, ink.shape[0] - oy), min(mw, ink.shape[1] - ox)
    if hh > 0 and ww > 0:
        ink[oy:oy + hh, ox:ox + ww] = ink0[:hh, :ww]
    est = d['est_font_size']

    pruned = keep_ink_components(fm, ink)
    c0, c1 = float((fm > 0).mean()), float((pruned > 0).mean())
    kf = int((pruned > 0).sum()) / max(int((fm > 0).sum()), 1)
    a = band_stats(crop, fm, ink, est)
    b = band_stats(crop, pruned, ink, est)
    fmt = lambda t: '  -' if t is None else f"in={t[0]:.2f} out={t[1]:.2f} dmed={t[2]:.1f}"
    print(f"  r{d['idx']} cover {c0:.3f}->{c1:.3f} keep={kf:.4f} | "
          f"sebelum[{fmt(a)}] sesudah[{fmt(b)}]", flush=True)
    if b:
        rows.append((d['idx'], c1, b[1], b[2]))

    # jalur mundur binv: seluruh piksel terang crop
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    vals = gray[binv > 0]
    if vals.size < binv.size - vals.size:
        vals = gray[binv == 0]
    if np.median(vals) < 128:
        binv = cv2.bitwise_not(binv)
    bp = keep_ink_components(binv, ink)
    print(f"      binv cover={float((binv>0).mean()):.3f} -> pangkas "
          f"{float((bp>0).mean()):.3f} "
          f"keep={int((bp>0).sum())/max(int((binv>0).sum()),1):.4f}", flush=True)

if rows:
    a = np.array([[r[1], r[2], r[3]] for r in rows], np.float64)
    print(f"\nSESUDAH PANGKAS n={len(rows)} cover max={a[:,0].max():.3f} "
          f"out_spread max={a[:,1].max():.2f} dmed max={a[:,2].max():.1f}", flush=True)
print(f"flat_std_thresh={SETTINGS.flat_std_thresh}", flush=True)
