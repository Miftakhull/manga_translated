"""Kalibrasi penjaga kebocoran isian: seberapa besar interior yang SAH
menyentuh tepi crop bubble_bbox, dan seberapa besar ia memuat tintanya sendiri.

Angka dari sini yang menentukan ambang di textmask.build_fill_mask —
bukan taksiran. Hasil dipickle supaya kalibrasi ulang tidak perlu
mengulang detect+CTD (7 menit di CPU).
"""
import sys, os, glob, pickle, numpy as np, cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MANGATL_WORK", ROOT)
os.environ.setdefault("MANGATL_ROOT", os.path.join(ROOT, ".stage"))
os.makedirs(os.path.join(ROOT, '.stage'), exist_ok=True)
for p in glob.glob(os.path.join(ROOT, '_nbsrc', '*.py')):
    src = open(p, encoding='utf-8').read().split('\n')
    if src and src[0].startswith('%%writefile'):
        src = src[1:]
    open(os.path.join(ROOT, '.stage', os.path.basename(p)),
         'w', encoding='utf-8').write('\n'.join(src))
sys.path.insert(0, os.path.join(ROOT, '.stage'))

import config, detect, textmask as tm, erase

PAGE = os.path.join(ROOT, 'jepang_002.webp')
CACHE = os.path.join(ROOT, '.fillcal.pkl')

img = cv2.cvtColor(cv2.imread(PAGE), cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
print('page', img.shape, flush=True)

regs, bubs = detect.detect(img)
soft = tm.ctd_soft_mask(img)
for r in regs:
    tm.build_region_mask(img, r, soft)
tm.partition_shared_interiors(img, regs)
tm.disjoin_overlapping_interiors(img, regs)
print(f'regions={len(regs)} bubbles={len(bubs)}', flush=True)


def edge_touch(m: np.ndarray) -> float:
    """Fraksi piksel TEPI crop yang terisi interior.

    Flood fill yang bocor keluar balon harus menyeberangi tepi crop di sisi
    tempat ia bocor; interior yang sah sudah dikikis stroke jadi menjauh dari
    tepi. Ini pembeda geometris, bukan pembeda ukuran."""
    if m.size == 0:
        return 0.0
    border = np.concatenate([m[0, :], m[-1, :], m[1:-1, 0], m[1:-1, -1]])
    return float((border > 0).mean())


rows = []
for r in regs:
    if r.fill_mask is None or r.fill_bbox is None:
        print(f'  r{r.idx} fill=None', flush=True)
        continue
    fm = r.fill_mask
    bx1, by1, bx2, by2 = r.fill_bbox
    cover = float((fm > 0).mean())
    et = edge_touch(fm)
    # tinta region sendiri di dalam isian
    ink = np.zeros(img.shape[:2], np.uint8)
    x1, y1, x2, y2 = r.bbox
    mh, mw = r.ink_mask.shape[:2]
    yy2, xx2 = min(y2, y1 + mh), min(x2, x1 + mw)
    ink[y1:yy2, x1:xx2] = r.ink_mask[:yy2 - y1, :xx2 - x1]
    page = erase._fill_on_page(r, img.shape[:2])
    sel = page > 0
    inkin = float(((ink > 0) & sel).sum()) / max(float((ink > 0).sum()), 1)
    rows.append((r.idx, cover, et, inkin))
    print(f'  r{r.idx} bub={r.bubble_bbox} cover={cover:.3f} '
          f'edge={et:.4f} inkin={inkin:.3f}', flush=True)

if rows:
    cov = np.array([x[1] for x in rows]); ed = np.array([x[2] for x in rows])
    ik = np.array([x[3] for x in rows])
    print(f'COVER min={cov.min():.3f} max={cov.max():.3f}', flush=True)
    print(f'EDGE  min={ed.min():.4f} max={ed.max():.4f} p90={np.percentile(ed,90):.4f}', flush=True)
    print(f'INKIN min={ik.min():.3f}', flush=True)

# Balon yang benar-benar tidak punya region: bukan sekadar bbox tak sama,
# tapi TIDAK ADA region yang pusatnya di dalamnya.
print('--- balon tanpa region (pusat region di luar) ---', flush=True)
n_orphan = 0
for b in bubs:
    bx1, by1, bx2, by2 = (int(v) for v in b)
    hit = False
    for r in regs:
        x1, y1, x2, y2 = r.bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if bx1 <= cx < bx2 and by1 <= cy < by2:
            hit = True
            break
    if hit:
        continue
    n_orphan += 1
    sub = gray[by1:by2, bx1:bx2]
    print(f'  ORPHAN {(bx1,by1,bx2,by2)} {bx2-bx1}x{by2-by1} '
          f'dark={float((sub<110).mean()):.4f}', flush=True)
print(f'orphan_count={n_orphan}', flush=True)

with open(CACHE, 'wb') as fh:
    pickle.dump({'rows': rows, 'bubs': [tuple(int(v) for v in b) for b in bubs],
                 'regs': [(r.idx, r.bbox, r.bubble_bbox) for r in regs]}, fh)
print('cached ->', CACHE, flush=True)
