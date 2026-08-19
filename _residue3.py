"""Cacat #1 diukur lewat erase_page YANG ASLI, bukan simulasi tangan.

_residue2.py mengecat sendiri dan melaporkan r7=115 r8=239 menyimpang, padahal
output/jepang_002.json melaporkan residue_count=0. Yang salah simulasinya:
warna latarnya bukan hasil fill_color() dan `protected_guard` tidak dipakai.
Jadi di sini Region dibangun ulang dari .pagediag.pkl lalu erase.erase_page()
dan verify.pixel_residue() yang SESUNGGUHNYA dipanggil.

Dua definisi residue dibandingkan:
  lama = pixel_residue apa adanya — lingkupnya `region.ink_mask > 0`
  luas = lingkup diperluas ke interior balon (bubble_mask) kalau ada

Kenapa lingkupnya penting: sejak erase_flat mengecat SELURUH interior balon,
permukaan yang seharusnya rata jauh lebih luas daripada bekas stroke huruf
Jepang. Titik kotor yang duduk di interior tapi di luar bekas stroke tidak
pernah dihitung. Itu tidak terlihat selama fill_mask ada (isian menutupinya),
tapi begitu build_fill_mask menyerah — jalur yang docstring-nya sendiri
mencatat MENYISAKAN coretan di jp_6 — sisa itu ada di halaman dan verify
melaporkan nol.

Dua skenario dijalankan:
  A. fill_mask apa adanya  -> keadaan halaman lokal sekarang (harus 0 residue)
  B. fill_mask DIBUANG     -> memaksa jalur ink_mask, yaitu keadaan Colab
Yang dicari: di skenario B, apakah definisi `lama` melaporkan nol sementara
`luas` melihat sisanya, dan seberapa besar komponen terbesarnya.
"""
import sys, os, pickle, numpy as np, cv2, copy

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MANGATL_WORK", ROOT)
os.environ.setdefault("MANGATL_ROOT", os.path.join(ROOT, ".stage"))
sys.path.insert(0, os.path.join(ROOT, '.stage'))
import erase, verify
from config import SETTINGS, Region

page = cv2.cvtColor(cv2.imread(os.path.join(ROOT, 'jepang_002.webp')),
                    cv2.COLOR_BGR2RGB)
st = pickle.load(open(os.path.join(ROOT, '.pagediag.pkl'), 'rb'))


def build(drop_fill: bool) -> list[Region]:
    out = []
    for d in st['regs']:
        r = Region(idx=d['idx'], bbox=d['bbox'], det_class=d['det_class'],
                   det_conf=d['det_conf'])
        r.bubble_bbox = d['bubble_bbox']
        r.bubble_mask = d['bubble_mask']
        r.ink_mask = d['ink_mask']
        r.ink_ratio = d['ink_ratio']
        r.est_font_size = d['est_font_size']
        if not drop_fill:
            r.fill_bbox, r.fill_mask = d['fill_bbox'], d['fill_mask']
        out.append(r)
    return out


def residue_luas(clean, r):
    """pixel_residue dengan lingkup interior balon, bukan cuma bekas stroke."""
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
    scope = inside.copy()
    if r.bubble_mask is not None and r.bubble_bbox is not None:
        bx1, by1, bx2, by2 = r.bubble_bbox
        big = np.zeros(clean.shape[:2], bool)
        bh, bw = r.bubble_mask.shape[:2]
        yy, xx = min(by1 + bh, clean.shape[0]), min(bx1 + bw, clean.shape[1])
        big[by1:yy, bx1:xx] = r.bubble_mask[:yy - by1, :xx - bx1] > 0
        cut = big[y1:y1 + scope.shape[0], x1:x1 + scope.shape[1]]
        scope = scope | cut[:scope.shape[0], :scope.shape[1]]
    bg = float(np.median(area[~inside])) if (~inside).any() else 255.0
    dev = np.abs(area.astype(np.int16) - bg) > SETTINGS.residue_deviation
    hit = dev & scope
    if not hit.any():
        return 0, 0
    n, _l, s, _c = cv2.connectedComponentsWithStats(hit.astype(np.uint8), 8)
    return int(hit.sum()), (0 if n <= 1 else int(s[1:, cv2.CC_STAT_AREA].max()))


for tag, drop in (("A fill_mask ADA", False), ("B fill_mask DIBUANG", True)):
    regs = build(drop)
    clean = erase.erase_page(page.copy(), regs, device="cpu")
    print(f"\n== {tag} ==", flush=True)
    print("  region  route   lama  luas  blob_luas ambang  lama? luas?", flush=True)
    tot = np.zeros(4, int)
    for r in regs:
        lama = verify.pixel_residue(clean, r)
        luas, blob = residue_luas(clean, r)
        thr = max(30, int(0.002 * r.width * r.height))
        tot += [lama > thr, luas > thr, blob > 12, 1]
        print(f"  r{r.idx:<2d} {r.route:6s} {lama:6d} {luas:5d} {blob:9d} "
              f"{thr:6d}  {'GAGAL' if lama > thr else '  ok ':5s} "
              f"{'GAGAL' if luas > thr else '  ok '}", flush=True)
    print(f"  gagal: lama={tot[0]}/{tot[3]} luas={tot[1]}/{tot[3]} "
          f"blob>12={tot[2]}/{tot[3]}", flush=True)
