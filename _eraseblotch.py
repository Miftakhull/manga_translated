"""Apakah `| ink_mask` meninggalkan BERCAK putih di luar balon — per gumpalan.

Dua probe sebelumnya berselisih soal ini: _erasecost.py melaporkan simpangan
warna 21 level di r11, _erasenear.py melaporkan 2. Bedanya cuma definisi
"tetangga": yang pertama membuang SELURUH ink_mask region dari cincin, yang
kedua hanya px yang baru dicat. Perselisihan itu artinya kedua angka tidak
layak dipakai memutuskan.

Yang benar diukur per GUMPALAN TERSAMBUNG, karena yang dilihat mata adalah satu
bercak, bukan rata-rata region:

  untuk tiap komponen tersambung px-baru-di-luar-interior:
    med  = median halaman ASLI di cincin 2 px di sekitar komponen, dengan
           SELURUH tinta halaman (gabungan ink_mask semua region) dibuang —
           jadi yang tersisa benar-benar latar/art di situ
    d    = |bg_color - med|; d > residue_deviation (20) = bercak TERLIHAT
    area = ukuran komponen

Yang dilaporkan: komponen dengan d terbesar, dan berapa komponen yang melewati
ambang 20 sekaligus cukup besar untuk terlihat (>= residue_blob_max = 12 px).
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
NEAR = -1
LOG = []


def _big(mask, bbox, shape):
    out = np.zeros(shape, np.uint8)
    if mask is None:
        return out
    x1, y1 = bbox[0], bbox[1]
    h, w = mask.shape[:2]
    yy, xx = min(y1 + h, shape[0]), min(x1 + w, shape[1])
    out[y1:yy, x1:xx] = mask[:yy - y1, :xx - x1]
    return out


def patched(img, region, guard=None):
    if region.bg_color is None:
        return img
    fill = erase._fill_on_page(region, img.shape[:2]) if SETTINGS.bubble_fill else None
    if fill is None:
        return ORIG(img, region, guard)
    sel = fill > 0
    lama = sel.copy()
    ink = _big(region.ink_mask, region.bbox, img.shape[:2])
    if NEAR >= 0:
        itr = _big(region.bubble_mask, region.bubble_bbox or (0, 0, 0, 0), img.shape[:2])
        if itr.any():
            if NEAR:
                itr = cv2.dilate(itr, cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * NEAR + 1,) * 2))
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


for NEAR in (-1, 8, 4, 2):
    regs = build()
    tm.protect_bubble_outline(page, regs)
    guard = erase.protected_guard(page, regs)
    all_ink = np.zeros(page.shape[:2], bool)
    for r in regs:
        all_ink |= _big(r.ink_mask, r.bbox, page.shape[:2]) > 0
    LOG.clear()
    erase.erase_flat = patched
    try:
        clean = erase.erase_page(page.copy(), regs, device="cpu")
    finally:
        erase.erase_flat = ORIG
    bad = [r.idx for r in regs
           if verify.pixel_residue(clean, r) > max(30, int(0.002 * r.width * r.height))
           or verify.residue_blob(clean, r) > SETTINGS.residue_blob_max]
    rows = []
    for r, nb in LOG:
        itr = _big(r.bubble_mask, r.bubble_bbox or (0, 0, 0, 0), page.shape[:2])
        out = nb & (itr == 0)
        if not out.any():
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), 8)
        bgg = float(cv2.cvtColor(np.uint8([[r.bg_color]]), cv2.COLOR_RGB2GRAY)[0, 0])
        for i in range(1, n):
            comp = lab == i
            ring = cv2.dilate(comp.astype(np.uint8), cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (5, 5))) > 0
            ring &= ~all_ink
            if not ring.any():
                continue
            med = float(np.median(gray0[ring]))
            rows.append((abs(bgg - med), int(stats[i, cv2.CC_STAT_AREA]), r.idx, med))
    rows.sort(reverse=True)
    terlihat = [x for x in rows
                if x[0] > SETTINGS.residue_deviation and x[1] >= SETTINGS.residue_blob_max]
    tag = "tanpa kurung" if NEAR < 0 else f"kurung {NEAR}px"
    print(f"\n== {tag} == ditandai={bad} komponen_luar={len(rows)} "
          f"BERCAK_TERLIHAT={len(terlihat)}", flush=True)
    for d, a, idx, med in rows[:6]:
        print(f"   r{idx:<2d} area={a:4d} d={d:5.1f} (bg=255 vs latar {med:.0f})"
              f"{'  <-- TERLIHAT' if d > 20 and a >= 12 else ''}", flush=True)
