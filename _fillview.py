"""Lihat isian di halaman BERSIH: apakah ia benar-benar tinggal di dalam balon?

Angka sudah gagal memisahkan (cover, edge, warna). Jadi ukur dengan MATA:
tandai piksel fill_mask dengan merah di atas halaman asli. Kalau ada merah
di atas art, cacat #3 sudah ada di halaman lokal — bukan cuma di Colab.

Membaca cache saja, tanpa detect/CTD.
"""
import sys, os, pickle, numpy as np, cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MANGATL_WORK", ROOT)
os.environ.setdefault("MANGATL_ROOT", os.path.join(ROOT, ".stage"))
sys.path.insert(0, os.path.join(ROOT, '.stage'))

OUT = os.path.join(ROOT, '_view')
os.makedirs(OUT, exist_ok=True)

bgr = cv2.imread(os.path.join(ROOT, 'jepang_002.webp'))
with open(os.path.join(ROOT, '.pagediag.pkl'), 'rb') as fh:
    st = pickle.load(fh)

over = bgr.copy()
tint = np.zeros(bgr.shape[:2], np.uint8)
for d in st['regs']:
    fm = d['fill_mask']
    if fm is None or d['fill_bbox'] is None:
        continue
    bx1, by1, bx2, by2 = d['fill_bbox']
    h, w = fm.shape[:2]
    sub = tint[by1:by1 + h, bx1:bx1 + w]
    np.maximum(sub, (fm > 0).astype(np.uint8) * 255, out=sub)

sel = tint > 0
over[sel] = (0.45 * over[sel] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
for d in st['regs']:
    if d['fill_bbox'] is None:
        continue
    bx1, by1, bx2, by2 = d['fill_bbox']
    cv2.rectangle(over, (bx1, by1), (bx2 - 1, by2 - 1), (0, 255, 0), 1)
    cv2.putText(over, str(d['idx']), (bx1 + 2, by1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

p = os.path.join(OUT, 'fill_overlay.png')
cv2.imwrite(p, over)
print('page ->', p, over.shape, flush=True)

# Crop per region, diperbesar 2x supaya batas isian terlihat jelas.
for d in st['regs']:
    if d['fill_bbox'] is None:
        continue
    bx1, by1, bx2, by2 = d['fill_bbox']
    m = 18
    y1, y2 = max(by1 - m, 0), min(by2 + m, over.shape[0])
    x1, x2 = max(bx1 - m, 0), min(bx2 + m, over.shape[1])
    c = over[y1:y2, x1:x2]
    c = cv2.resize(c, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(OUT, f"r{d['idx']:02d}.png"), c)
print('crops ->', OUT, flush=True)
