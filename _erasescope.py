"""Berapa lebar toleransi lingkup residu — sekecil mungkin butanya, tanpa alarm palsu.

_erasewho.py memutuskan arahnya: 36 px r7 dan 103 px r8 yang ditandai adalah ART
di LUAR balon yang terbawa ink_mask, utuh tak tersentuh erase (utuh=sisa), nol di
dalam bubble_mask. Mengecatnya menaruh putih di atas art gray 12-18 (7 bercak
terlihat, _eraseblotch.py); membiarkannya membuat escalate() menggerus 250 px
garis balon (_erasegate.py). Jadi yang harus dikurung adalah LINGKUP TUDUHAN,
bukan catnya.

Tapi mengurung ke interior TELANJANG membuat gerbang buta pada tinta yang sah
tepat di tepi balon — dan itu justru mekanisme cacat #1 (tinta Jepang yang
menempel garis dihapus dari ink_mask oleh protect_bubble_outline, jadi tidak
dicat DAN tidak diperiksa). Terukur berapa banyak tinta sah yang di luar
interior: r12 0.813, r8 0.945, r5/r11 0.968, r7 0.994, sisanya 1.000. Jadi
interior telanjang membuang 18.7% tinta r12 dari pengawasan.

Disapu di sini: interior dilebarkan N px sebagai lingkup.
  N besar -> pengawasan luas (baik) tapi art ikut tertuduh (buruk)
  N kecil -> sebaliknya
Kolom:
  ditandai   = region yang lolos gerbang pada halaman bersih (WAJIB kosong;
               semua yang ditandai di sini adalah alarm palsu art)
  awas%      = rata-rata fraksi ink_mask yang masih diawasi (makin tinggi makin
               kecil butanya)
  awas_min   = fraksi terburuk di antara 13 region
  suntik     = apakah sisa BUATAN di dalam interior masih tertangkap (WAJIB ya)
  tepi       = apakah sisa buatan di TEPI interior (cincin 2 px paling luar dari
               tinta yang berbatasan garis) masih tertangkap — ini yang menguji
               kebutaan pada mekanisme cacat #1 yang sebenarnya
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
st = pickle.load(open(os.path.join(ROOT, '.pagediag.pkl'), 'rb'))


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


def big(mask, bbox, shape):
    out = np.zeros(shape, np.uint8)
    if mask is None or bbox is None:
        return out
    x1, y1 = bbox[0], bbox[1]
    h, w = mask.shape[:2]
    yy, xx = min(y1 + h, shape[0]), min(x1 + w, shape[1])
    out[y1:yy, x1:xx] = mask[:yy - y1, :xx - x1]
    return out


def blob(m):
    if not m.any():
        return 0
    n, _l, s, _c = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return 0 if n <= 1 else int(s[1:, cv2.CC_STAT_AREA].max())


def scope_of(r, clean, N):
    """(dev, scope) versi kandidat: lingkup lama & interior dilebarkan N px."""
    got = verify._residue_scope(clean, r)
    if got is None:
        return None
    dev, scope = got
    if N is None:
        return dev, scope
    itr = big(r.bubble_mask, r.bubble_bbox, clean.shape[:2])
    if not itr.any():
        return dev, scope
    if N:
        itr = cv2.dilate(itr, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * N + 1,) * 2))
    x1, y1 = r.bbox[0], r.bbox[1]
    h, w = scope.shape
    return dev, scope & (itr[y1:y1 + h, x1:x1 + w] > 0)


regs = build()
tm.protect_bubble_outline(page, regs)
clean = erase.erase_page(page.copy(), regs, device="cpu")

# sisa buatan #1: komponen tinta besar di TENGAH interior
# sisa buatan #2: tinta yang BERBATASAN dengan garis balon (cincin terluar
#                 interior) — inilah mekanisme cacat #1 yang sesungguhnya
#
# Syarat GELAP (< 128 di halaman asli) wajib ada di kedua-duanya: percobaan
# pertama memilih cincin tepi r6 apa adanya dan mendapat 48 px yang ternyata
# gray 255 di halaman ASLI — ink_mask yang meluber ke latar putih, bukan tinta.
# Memulihkannya tidak mengubah apa pun, jadi gerbangnya "gagal" menangkap
# sesuatu yang memang tidak ada. Itu mengukur probe-nya, bukan lingkupnya.
gray_page = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)
inj_mid = inj_edge = None
for r in regs:
    itr = big(r.bubble_mask, r.bubble_bbox, page.shape[:2])
    ink = big(r.ink_mask, r.bbox, page.shape[:2])
    if not (itr.any() and ink.any()):
        continue
    gelap = gray_page < 128
    er = cv2.erode(itr, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    tepi = (itr > 0) & (er == 0) & (ink > 0) & gelap
    if inj_edge is None and tepi.sum() >= 20:
        n, lab, s, _c = cv2.connectedComponentsWithStats(tepi.astype(np.uint8), 8)
        i = max(range(1, n), key=lambda j: s[j, cv2.CC_STAT_AREA])
        if s[i, cv2.CC_STAT_AREA] >= 20:
            inj_edge = (r, lab == i)
    mid = (er > 0) & (ink > 0) & gelap
    if inj_mid is None and mid.sum() >= 20:
        n, lab, s, _c = cv2.connectedComponentsWithStats(mid.astype(np.uint8), 8)
        i = max(range(1, n), key=lambda j: s[j, cv2.CC_STAT_AREA])
        inj_mid = (r, lab == i)
    if inj_mid and inj_edge:
        break

print(f" sisa buatan: tengah r{inj_mid[0].idx} {int(inj_mid[1].sum())} px | "
      f"tepi r{inj_edge[0].idx} {int(inj_edge[1].sum())} px", flush=True)
print("\n   N  ditandai        awas%  awas_min  suntik_tengah  suntik_tepi", flush=True)

for N in (None, 0, 2, 3, 4, 5, 6, 8):
    bad, fr = [], []
    for r in regs:
        got = scope_of(r, clean, N)
        if got is None:
            continue
        dev, scope = got
        hit = dev & scope
        thr = max(30, int(0.002 * r.width * r.height))
        if int(hit.sum()) > thr or blob(hit) > SETTINGS.residue_blob_max:
            bad.append(r.idx)
        ink = big(r.ink_mask, r.bbox, page.shape[:2]) > 0
        x1, y1 = r.bbox[0], r.bbox[1]
        h, w = scope.shape
        sb = np.zeros(page.shape[:2], bool)
        sb[y1:y1 + h, x1:x1 + w] = scope
        fr.append(int((ink & sb).sum()) / max(int(ink.sum()), 1))
    hasil = []
    for r_inj, comp in (inj_mid, inj_edge):
        img = clean.copy()
        img[comp] = page[comp]
        got = scope_of(r_inj, img, N)
        dev, scope = got
        hit = dev & scope
        thr = max(30, int(0.002 * r_inj.width * r_inj.height))
        ok = int(hit.sum()) > thr or blob(hit) > SETTINGS.residue_blob_max
        hasil.append(("ya" if ok else "TIDAK") + f"({int(hit.sum())})")
    tag = "lama" if N is None else str(N)
    print(f" {tag:>4s}  {str(bad):<12s} {np.mean(fr) * 100:6.1f}%  "
          f"{min(fr) * 100:6.1f}%  {hasil[0]:>13s}  {hasil[1]:>12s}", flush=True)
