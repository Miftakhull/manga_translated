"""Cacat #1: lingkup GABUNGAN + gerbang komponen terbesar, diukur.

Dari _residue4.py:
  * `protect_bubble_outline` membuang 185 px dari ink_mask di halaman ini
    (r7 79, r11 35, r12 71). Yang dibuang tidak dicat oleh jalur ink_mask DAN
    tidak diperiksa `pixel_residue`, karena lingkup pemeriksaannya justru
    `ink_mask > 0`. Di r12 71 px itu MASIH menyimpang setelah erase, dengan
    komponen terbesar 30 px — persis ukuran "titik hitam" yang dilaporkan, dan
    verify melaporkannya NOL di kedua skenario (fill ada maupun dibuang).
  * Lingkup `dilate(ink,3) & interior` melihat r12 (71 px) dan TIDAK melihat art
    di sudut interior r9 (198 px) — dilatasi 3 px dari bekas tinta tidak
    menjangkau sudut kotak. Itu yang membuatnya aman dipakai.
  * TAPI lingkup itu sendirian MENGHILANGKAN deteksi yang sekarang sudah benar:
    r7 (36 px) dan r8 (239 px) jatuh ke 0 karena piksel menyimpangnya ada di
    LUAR bubble_mask. Keduanya nyata — laporan halaman lokal residue_count=0
    justru karena escalate() memperbaikinya di ronde kedua.

Jadi yang diuji di sini GABUNGAN, bukan pengganti:
    scope = (ink_mask > 0) | (dilate(ink_mask, 3) & interior)
dan gerbangnya jumlah ATAU komponen terbesar:
    gagal = total > max(30, 0.002*w*h)  or  blob > BLOB_CAP

Yang harus dibuktikan:
  1. gabungan >= lama di setiap region (tidak ada deteksi yang hilang),
  2. r12 terdeteksi di kedua skenario,
  3. art sudut r9 (198 px) TIDAK terdeteksi walau fill_mask dibuang,
  4. BLOB_CAP tidak menyalakan region yang bersih.
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
EL3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
CAPS = (12, 16, 20, 24, 30, 40)


def build(drop_fill):
    regs = []
    for d in st['regs']:
        r = Region(idx=d['idx'], bbox=d['bbox'], det_class=d['det_class'],
                   det_conf=d['det_conf'])
        r.bubble_bbox = d['bubble_bbox']
        r.bubble_mask = d['bubble_mask']
        r.ink_mask = d['ink_mask'].copy()
        if not drop_fill:
            r.fill_bbox, r.fill_mask = d['fill_bbox'], d['fill_mask']
        regs.append(r)
    return regs


def scope_union(clean, r):
    """(total, blob) memakai lingkup gabungan. Bentuknya = calon patch verify."""
    if r.ink_mask is None:
        return 0, 0
    x1, y1, x2, y2 = r.bbox
    crop = clean[y1:y2, x1:x2]
    if crop.size == 0:
        return 0, 0
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    mh, mw = r.ink_mask.shape[:2]
    sub = r.ink_mask[:min(mh, gray.shape[0]), :min(mw, gray.shape[1])]
    area = gray[:sub.shape[0], :sub.shape[1]]
    inside = sub > 0
    if not inside.any():
        return 0, 0
    scope = inside
    if r.bubble_mask is not None and r.bubble_bbox is not None:
        big = np.zeros(clean.shape[:2], np.uint8)
        bx1, by1 = r.bubble_bbox[0], r.bubble_bbox[1]
        bh, bw = r.bubble_mask.shape[:2]
        yy = min(by1 + bh, clean.shape[0]); xx = min(bx1 + bw, clean.shape[1])
        big[by1:yy, bx1:xx] = r.bubble_mask[:yy - by1, :xx - bx1]
        itr = big[y1:y1 + sub.shape[0], x1:x1 + sub.shape[1]] > 0
        near = cv2.dilate(sub, EL3) > 0
        scope = inside | (near & itr[:sub.shape[0], :sub.shape[1]])
    bg = float(np.median(area[~inside])) if (~inside).any() else 255.0
    dev = np.abs(area.astype(np.int16) - bg) > SETTINGS.residue_deviation
    hit = dev & scope
    if not hit.any():
        return 0, 0
    n, _l, s, _c = cv2.connectedComponentsWithStats(hit.astype(np.uint8), 8)
    return int(hit.sum()), (0 if n <= 1 else int(s[1:, cv2.CC_STAT_AREA].max()))


for tag, drop in (("A fill_mask ADA", False), ("B fill_mask DIBUANG", True)):
    regs = build(drop)
    tm.protect_bubble_outline(page, regs)
    clean = erase.erase_page(page.copy(), regs, device="cpu")
    print(f"\n== {tag} ==", flush=True)
    print("  region   lama gabung  blob ambang  lama? gabung?  turun?", flush=True)
    rows = []
    for r in regs:
        lama = verify.pixel_residue(clean, r)
        tot, blob = scope_union(clean, r)
        thr = max(30, int(0.002 * r.width * r.height))
        rows.append((lama, tot, blob, thr))
        print(f"  r{r.idx:<2d} {lama:6d} {tot:6d} {blob:5d} {thr:6d}  "
              f"{'GAGAL' if lama > thr else '  ok ':6s} "
              f"{'GAGAL' if tot > thr else '  ok ':7s} "
              f"{'YA <-- deteksi hilang' if tot < lama else ''}", flush=True)
    a = np.array(rows)
    print(f"  gabungan >= lama di semua region: {bool((a[:,1] >= a[:,0]).all())}",
          flush=True)
    print(f"  gagal: lama={int((a[:,0] > a[:,3]).sum())} "
          f"gabungan={int((a[:,1] > a[:,3]).sum())}", flush=True)
    for cap in CAPS:
        fail = (a[:, 1] > a[:, 3]) | (a[:, 2] > cap)
        idx = [regs[i].idx for i in np.flatnonzero(fail)]
        print(f"    blob>{cap:2d}: gagal={int(fail.sum())} -> r{idx}", flush=True)
