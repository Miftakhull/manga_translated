"""Berapa mahal `| ink_mask` di luar interior balon — cacat #3 lewat pintu belakang?

_erasefix.py membuktikan kandidatnya menghapus sisa r7/r8 tanpa menyentuh
garis balon (0 px guard tercat). Tapi piksel yang baru ikut tercat itu justru
yang berada di LUAR bubble_mask (terukur di _residue6.py: luar_itr 36/36 dan
103/103). Mengecatnya dengan `region.bg_color` — warna latar BALON — di area
luar balon adalah bentuk lain dari cacat #3 yang user laporkan: "box mask color
like bubble is out from the bubble".

Jadi yang diukur di sini, per region, hanya untuk piksel BARU (yang lama tidak
cat tapi kandidat cat):
  baru      = jumlah px baru yang tercat
  luar_itr  = berapa dari px baru itu di luar bubble_mask
  d_lokal   = |bg_color - median tetangga| di px baru yang di luar interior.
              Kalau kecil (<= residue_deviation 20), catnya tidak terlihat:
              latar di situ memang latar balon yang sama, glyph-nya cuma
              menembus garis sedikit. Kalau besar, kandidat ini menukar
              cacat #1 dengan cacat #3 dan harus dikurung.
  jarak     = jarak terjauh px baru dari interior balon (px). Glyph yang
              menempel garis harusnya 1-3 px; angka besar berarti ada tinta
              yang sebenarnya bukan milik balon ini.
"""
import sys, os, pickle, numpy as np, cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MANGATL_WORK", ROOT)
os.environ.setdefault("MANGATL_ROOT", os.path.join(ROOT, ".stage"))
sys.path.insert(0, os.path.join(ROOT, '.stage'))
import erase, textmask as tm
from config import SETTINGS, Region

page = cv2.cvtColor(cv2.imread(os.path.join(ROOT, 'jepang_002.webp')),
                    cv2.COLOR_BGR2RGB)
st = pickle.load(open(os.path.join(ROOT, '.pagediag.pkl'), 'rb'))

regs = []
for d in st['regs']:
    r = Region(idx=d['idx'], bbox=d['bbox'], det_class=d['det_class'],
               det_conf=d['det_conf'])
    r.bubble_bbox, r.bubble_mask = d['bubble_bbox'], d['bubble_mask']
    r.ink_mask = d['ink_mask'].copy()
    r.est_font_size = d['est_font_size']
    r.fill_bbox, r.fill_mask = d['fill_bbox'], d['fill_mask']
    regs.append(r)
tm.protect_bubble_outline(page, regs)
guard = erase.protected_guard(page, regs)
gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)

print(" region  baru luar_itr  d_lokal jarak  bg_color", flush=True)
tot_baru = tot_luar = 0
for r in regs:
    erase.route_region(page.copy(), r)
    if r.route != "flat" or r.bg_color is None:
        continue
    fill = erase._fill_on_page(r, page.shape[:2])
    if fill is None:
        continue
    lama = (fill > 0) & (guard == 0)
    ink = np.zeros(page.shape[:2], np.uint8)
    x1, y1, x2, y2 = r.bbox
    mh, mw = r.ink_mask.shape[:2]
    y2b, x2b = min(y2, y1 + mh, page.shape[0]), min(x2, x1 + mw, page.shape[1])
    ink[y1:y2b, x1:x2b] = r.ink_mask[:y2b - y1, :x2b - x1]
    baru = (ink > 0) & (guard == 0) & ~lama
    if not baru.any():
        continue
    itr = np.zeros(page.shape[:2], np.uint8)
    if r.bubble_mask is not None:
        bx1, by1 = r.bubble_bbox[0], r.bubble_bbox[1]
        bh, bw = r.bubble_mask.shape[:2]
        yy, xx = min(by1 + bh, page.shape[0]), min(bx1 + bw, page.shape[1])
        itr[by1:yy, bx1:xx] = r.bubble_mask[:yy - by1, :xx - bx1]
    luar = baru & (itr == 0)
    # median tetangga: cincin 5 px di sekitar px baru-luar, bukan px tinta
    if luar.any():
        ring = (cv2.dilate(luar.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0)
        ring &= ~(ink > 0)
        med = float(np.median(gray[ring])) if ring.any() else 255.0
        bgg = float(cv2.cvtColor(np.uint8([[r.bg_color]]), cv2.COLOR_RGB2GRAY)[0, 0])
        d = abs(bgg - med)
        # jarak terjauh dari interior
        if itr.any():
            dist = cv2.distanceTransform((itr == 0).astype(np.uint8), cv2.DIST_L2, 3)
            jarak = float(dist[luar].max())
        else:
            jarak = -1.0
    else:
        d, jarak = 0.0, 0.0
    tot_baru += int(baru.sum()); tot_luar += int(luar.sum())
    print(f" r{r.idx:<2d} {int(baru.sum()):6d} {int(luar.sum()):8d} "
          f"{d:8.1f} {jarak:5.1f}  {r.bg_color}", flush=True)
print(f"\n TOTAL baru={tot_baru} luar_interior={tot_luar} "
      f"(ambang tak-terlihat: residue_deviation={SETTINGS.residue_deviation})",
      flush=True)
