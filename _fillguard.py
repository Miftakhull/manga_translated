"""Uji penjaga kebocoran isian: DUA arah, dari cache .pagediag.pkl.

Kesadaran penting soal bentuk kebocoran: `_interior_from_crop` bekerja pada
crop `bubble_bbox`, jadi flood fill TIDAK BISA keluar dari kotak itu. Yang
bisa ia rebut hanyalah bagian crop yang di luar garis balon tapi masih di
dalam kotak — yaitu POJOK-POJOK, dan pojok balon berisi ART. Itu persis
keluhan user: "box mask color like bubble is out from the bubble".

Penjaga `mean > 0.97` tidak menangkap ini karena isian sah sudah sampai
cover 0.900 (r9), sementara kebocoran pojok berhenti di ~0.90-0.95.

Kandidat yang diuji: bandingkan warna CINCIN DEKAT TINTA (pasti di dalam
balon) dengan PINGGIRAN isian. Relatif, jadi balon screentone/abu-abu tidak
ikut tertolak — beda dengan ambang mutlak flat_std_thresh.

Arah 1 (false positive): 13 region halaman bersih harus LULUS.
Arah 2 (true positive) : isian + pojok art sintetis harus GAGAL.
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


def el(k: int) -> np.ndarray:
    k = max(int(k) | 1, 3)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def spread(px: np.ndarray) -> tuple[np.ndarray, float]:
    med = np.median(px, axis=0)
    return med, float((np.median(np.abs(px - med), axis=0) * 1.4826).max())


def probe(name, crop, fill, ink, est):
    """crop/fill/ink dalam sistem koordinat fill_bbox yang sama."""
    inkd = cv2.dilate(ink, EL5)
    near = cv2.dilate(ink, el(2 * max(est, 6) + 1))
    inner = (fill > 0) & (near > 0) & (inkd == 0)
    outer = (fill > 0) & (near == 0)
    ni, no = int(inner.sum()), int(outer.sum())
    if ni < 30 or no < 30:
        print(f"  {name} inner={ni} outer={no} -> tak diuji", flush=True)
        return None
    mi, si = spread(crop[inner].reshape(-1, 3).astype(np.float32))
    mo, so = spread(crop[outer].reshape(-1, 3).astype(np.float32))
    dmed = float(np.abs(mo - mi).max())
    print(f"  {name} inner(med={mi.astype(int).tolist()} sp={si:.2f}) "
          f"outer(med={mo.astype(int).tolist()} sp={so:.2f}) "
          f"dmed={dmed:.1f} px={ni}/{no}", flush=True)
    return si, so, dmed


def art_corners(interior: np.ndarray, keep: int = 2) -> np.ndarray:
    """Pojok crop di luar garis balon: komponen latar yang menyentuh tepi."""
    comp = (interior == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(comp, 8)
    h, w = interior.shape[:2]
    edge = set(np.unique(lab[0, :])) | set(np.unique(lab[-1, :])) \
        | set(np.unique(lab[:, 0])) | set(np.unique(lab[:, -1]))
    cand = sorted(((stats[i, cv2.CC_STAT_AREA], i) for i in edge if i != 0),
                  reverse=True)[:keep]
    out = np.zeros_like(interior)
    for _, i in cand:
        out[lab == i] = 255
    return out


clean, leaked = [], []
for d in st['regs']:
    fm, ink0 = d['fill_mask'], d['ink_mask']
    if fm is None or d['fill_bbox'] is None or ink0 is None:
        continue
    bx1, by1, bx2, by2 = d['fill_bbox']
    x1, y1, x2, y2 = d['bbox']
    crop = img[by1:by1 + fm.shape[0], bx1:bx1 + fm.shape[1]]
    ink = np.zeros(fm.shape[:2], np.uint8)
    mh, mw = ink0.shape[:2]
    oy, ox = y1 - by1, x1 - bx1
    hh = min(mh, ink.shape[0] - oy); ww = min(mw, ink.shape[1] - ox)
    if hh > 0 and ww > 0:
        ink[oy:oy + hh, ox:ox + ww] = ink0[:hh, :ww]
    est = d['est_font_size']

    r = probe(f"r{d['idx']} BERSIH", crop, fm, ink, est)
    if r:
        clean.append(r)

    lk = np.maximum(fm, art_corners(fm))
    cov = float((lk > 0).mean())
    r2 = probe(f"r{d['idx']} BOCOR cover={cov:.3f}", crop, lk, ink, est)
    if r2:
        leaked.append(r2 + (cov,))

print(flush=True)
if clean:
    a = np.array(clean)
    print(f"BERSIH n={len(a)} sp_inner<={a[:,0].max():.2f} "
          f"sp_outer<={a[:,1].max():.2f} dmed<={a[:,2].max():.1f}", flush=True)
if leaked:
    b = np.array(leaked)
    print(f"BOCOR  n={len(b)} sp_outer min={b[:,1].min():.2f} "
          f"p10={np.percentile(b[:,1],10):.2f} dmed min={b[:,2].min():.1f} "
          f"cover max={b[:,3].max():.3f}", flush=True)
    print(f"  cover>0.97 (tertangkap penjaga lama) = "
          f"{int((b[:,3] > 0.97).sum())}/{len(b)}", flush=True)
print(f"flat_std_thresh={SETTINGS.flat_std_thresh} "
      f"noisy={SETTINGS.flat_std_thresh_noisy}", flush=True)
