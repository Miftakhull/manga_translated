"""Sisa r7/r8 itu tinta yang lolos, atau ART yang salah dituduh?

_eraseblotch.py membalik pertanyaannya. Mengecat px ink_mask di luar interior
menaruh putih 255 di atas latar gray 12-18 (d=243/237) — bercak putih di atas
art, persis cacat #3. Artinya px ink_mask di luar interior balon r7/r8 BUKAN
tinta di atas latar balon; ia duduk di atas ART GELAP.

Kalau begitu, kemungkinannya: mask ink r7/r8 meluber ke art gelap di luar
balon, dan setelah erase px itu tetap gelap (jalur fill tidak mengecatnya) —
lalu `pixel_residue` menuduhnya sisa karena ia menyimpang dari median latar.
Konsekuensinya bukan "coretan tertinggal" tapi ALARM PALSU, dan `escalate()`
menjawab alarm itu dengan LaMa yang memakan 98-152 px garis balon (_esc.py).
Jadi kerusakannya nyata dan arahnya berlawanan dengan tebakan awal.

Yang diukur untuk tiap px yang ditandai `pixel_residue`:
  sama_asli = apakah px itu TIDAK BERUBAH oleh erase (art utuh, bukan sisa)
  gray      = nilai gray-nya di halaman asli
  di_itr    = di dalam bubble_mask?
  ocr_ink   = apakah px itu benar-benar gelap (< 100) DAN di dalam interior —
              satu-satunya kombinasi yang layak disebut "tinta tertinggal"
Lalu diuji satu kandidat gerbang: lingkup residu DIKURUNG ke interior balon
(plus cincin 3 px yang sudah ada), yaitu membuang tuduhan di luar balon.
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
clean = erase.erase_page(page.copy(), regs, device="cpu")
unchanged = ~(clean != page).any(2)


def interior(r):
    itr = np.zeros(page.shape[:2], np.uint8)
    if r.bubble_mask is None or r.bubble_bbox is None:
        return itr
    bx1, by1 = r.bubble_bbox[0], r.bubble_bbox[1]
    bh, bw = r.bubble_mask.shape[:2]
    yy, xx = min(by1 + bh, page.shape[0]), min(bx1 + bw, page.shape[1])
    itr[by1:yy, bx1:xx] = r.bubble_mask[:yy - by1, :xx - bx1]
    return itr


print("== anatomi px yang ditandai ==", flush=True)
print(" region  sisa  utuh  di_itr  gray_min gray_med  di_guard", flush=True)
for r in regs:
    got = verify._residue_scope(clean, r)
    if got is None:
        continue
    hit = got[0] & got[1]
    if not hit.any():
        continue
    x1, y1 = r.bbox[0], r.bbox[1]
    h, w = hit.shape
    big = np.zeros(page.shape[:2], bool)
    big[y1:y1 + h, x1:x1 + w] = hit
    itr = interior(r) > 0
    print(f" r{r.idx:<2d} {int(big.sum()):5d} {int((big & unchanged).sum()):5d} "
          f"{int((big & itr).sum()):7d} {int(gray0[big].min()):9d} "
          f"{int(np.median(gray0[big])):8d} {int((big & (guard > 0)).sum()):9d}",
          flush=True)

print("\n== kandidat: lingkup residu DIKURUNG ke interior balon ==", flush=True)
print(" region  lama_total lama_blob  kurung_total kurung_blob ambang verdict", flush=True)
for r in regs:
    got = verify._residue_scope(clean, r)
    if got is None:
        continue
    dev, scope = got
    x1, y1 = r.bbox[0], r.bbox[1]
    h, w = scope.shape
    itr = interior(r)[y1:y1 + h, x1:x1 + w] > 0
    if not itr.any():
        itr = np.ones_like(scope)          # tanpa balon: biarkan seperti dulu
    a = dev & scope
    b = dev & scope & itr
    thr = max(30, int(0.002 * r.width * r.height))

    def blob(m):
        if not m.any():
            return 0
        n, _l, s, _c = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        return 0 if n <= 1 else int(s[1:, cv2.CC_STAT_AREA].max())

    if not (a.any() or b.any()):
        continue
    bad_a = int(a.sum()) > thr or blob(a) > SETTINGS.residue_blob_max
    bad_b = int(b.sum()) > thr or blob(b) > SETTINGS.residue_blob_max
    print(f" r{r.idx:<2d} {int(a.sum()):10d} {blob(a):9d} {int(b.sum()):13d} "
          f"{blob(b):11d} {thr:6d} {'GAGAL' if bad_a else 'ok':>5s} -> "
          f"{'GAGAL' if bad_b else 'ok'}", flush=True)
