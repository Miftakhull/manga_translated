"""Seberapa jauh dari interior balon `| ink_mask` boleh dicat?

_erasefix.py: gabungan fill|ink menghapus sisa r7/r8 (2 region ditandai -> 0)
tanpa menyentuh guard sama sekali (0 px dari 12494).
_erasecost.py: TAPI seluruh 842 px baru itu berada di LUAR bubble_mask, dan di
r11 warna catnya menyimpang 21 level dari tetangganya sampai 7.4 px keluar —
di atas residue_deviation (20), artinya TERLIHAT. Itu persis cacat #3 yang
user laporkan ("box mask color like bubble is out from the bubble"), muncul
lewat pintu belakang, di region yang bahkan tidak punya sisa untuk diperbaiki.

Jadi gabungannya harus DIKURUNG: hanya tinta yang masih menempel interior yang
dicat. Yang disapu di sini, untuk N = 0,2,3,4,6,8,12 px pelebaran interior:

  ditandai  = region yang masih lolos gerbang find_residue (target 0)
  baru      = px yang ikut tercat di luar jalur lama
  luar      = dari `baru`, yang di luar bubble_mask
  d_max     = simpangan warna cat terbesar di antara px luar itu, dibanding
              median tetangganya di halaman asli. HARUS <= 20, kalau tidak
              catnya terlihat dan kita menukar cacat #1 dengan cacat #3.
  garis     = px guard yang tercat (WAJIB 0)

N terkecil yang memberi ditandai=0 dengan d_max <= 20 adalah yang dipilih.
"""
import sys, os, pickle, numpy as np, cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MANGATL_WORK", ROOT)
os.environ.setdefault("MANGATL_ROOT", os.path.join(ROOT, ".stage"))
sys.path.insert(0, os.path.join(ROOT, '.stage'))
import erase, verify, textmask as tm
from config import SETTINGS, Region

page = cv2.cvtColor(cv2.imread(os.path.join(ROOT, 'jepang_002.webp')),
                    cv2.COLOR_BGR2RGB)
gray0 = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)
st = pickle.load(open(os.path.join(ROOT, '.pagediag.pkl'), 'rb'))
ORIG = erase.erase_flat
NEAR = 0
LOG = []


def _interior(region, shape):
    itr = np.zeros(shape, np.uint8)
    if region.bubble_mask is None or region.bubble_bbox is None:
        return None
    bx1, by1 = region.bubble_bbox[0], region.bubble_bbox[1]
    bh, bw = region.bubble_mask.shape[:2]
    yy, xx = min(by1 + bh, shape[0]), min(bx1 + bw, shape[1])
    itr[by1:yy, bx1:xx] = region.bubble_mask[:yy - by1, :xx - bx1]
    return itr


def patched(img, region, guard=None):
    if region.bg_color is None:
        return img
    fill = erase._fill_on_page(region, img.shape[:2]) if SETTINGS.bubble_fill else None
    if fill is None:
        return ORIG(img, region, guard)
    sel = fill > 0
    lama = sel.copy()
    if region.ink_mask is not None:
        ink = np.zeros(img.shape[:2], np.uint8)
        x1, y1, x2, y2 = region.bbox
        mh, mw = region.ink_mask.shape[:2]
        y2, x2 = min(y2, y1 + mh, img.shape[0]), min(x2, x1 + mw, img.shape[1])
        if y2 > y1 and x2 > x1:
            ink[y1:y2, x1:x2] = region.ink_mask[:y2 - y1, :x2 - x1]
        itr = _interior(region, img.shape[:2])
        if itr is not None and NEAR >= 0:
            if NEAR:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * NEAR + 1,) * 2)
                itr = cv2.dilate(itr, k)
            ink = cv2.bitwise_and(ink, itr)
        sel |= ink > 0
    if guard is not None:
        sel &= guard == 0
        lama &= guard == 0
    LOG.append((region, sel & ~lama))
    img[sel] = region.bg_color
    return img


def build():
    regs = []
    for d in st['regs']:
        r = Region(idx=d['idx'], bbox=d['bbox'], det_class=d['det_class'],
                   det_conf=d['det_conf'])
        r.bubble_bbox, r.bubble_mask = d['bubble_bbox'], d['bubble_mask']
        r.ink_mask = d['ink_mask'].copy()
        r.est_font_size = d['est_font_size']
        r.fill_bbox, r.fill_mask = d['fill_bbox'], d['fill_mask']
        regs.append(r)
    return regs


print("  N ditandai   baru   luar  d_max garis  idx", flush=True)
for NEAR in (-1, 0, 2, 3, 4, 6, 8, 12):
    regs = build()
    tm.protect_bubble_outline(page, regs)
    guard = erase.protected_guard(page, regs)
    LOG.clear()
    erase.erase_flat = patched
    try:
        clean = erase.erase_page(page.copy(), regs, device="cpu")
    finally:
        erase.erase_flat = ORIG
    bad = [r.idx for r in regs
           if verify.pixel_residue(clean, r) > max(30, int(0.002 * r.width * r.height))
           or verify.residue_blob(clean, r) > SETTINGS.residue_blob_max]
    baru = luar = 0
    d_max = 0.0
    for r, nb in LOG:
        if not nb.any():
            continue
        baru += int(nb.sum())
        itr = _interior(r, page.shape[:2])
        out = nb & (itr == 0) if itr is not None else nb
        if not out.any():
            continue
        luar += int(out.sum())
        ring = cv2.dilate(out.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
        ring &= ~nb
        med = float(np.median(gray0[ring])) if ring.any() else 255.0
        bgg = float(cv2.cvtColor(np.uint8([[r.bg_color]]), cv2.COLOR_RGB2GRAY)[0, 0])
        d_max = max(d_max, abs(bgg - med))
    chg = (clean != page).any(2)
    tag = "tanpa kurung" if NEAR < 0 else f"{NEAR:2d}"
    print(f" {tag:>12s} {len(bad):3d} {baru:6d} {luar:6d} {d_max:6.1f} "
          f"{int((chg & (guard > 0)).sum()):5d}  {bad}", flush=True)
