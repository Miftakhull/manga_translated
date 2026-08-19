"""Cacat #1 (titik hitam sisa) diukur dengan definisi verify, bukan `<128`.

Pengukuran pertama (_fix12.py BAGIAN 1) memakai ambang gelap `gray < 128` dan
melaporkan 0 sisa — itu ambang yang salah. Yang dianggap sisa oleh pipeline
adalah `abs(piksel - median_latar) > residue_deviation (20)`, yaitu SEMUA yang
lebih gelap dari ~235 pada latar putih: tepi glyph yang teranti-alias masuk
hitungan, dan justru tepi itulah yang terlihat sebagai coretan/titik kotor.

Diukur di sini, per region, pada jalur TANPA fill_mask (jalur yang dipakai
begitu build_fill_mask menyerah — makin relevan setelah pemangkasan komponen
pada perbaikan cacat #3 bisa membuatnya menyerah lebih sering):

  sisa_lama  = piksel menyimpang yang TERTINGGAL setelah dicat pakai ink_mask
  sisa_baru  = idem setelah ink_mask DIDILATASI k lalu DIKURUNG ke interior
  blob_lama  = komponen tersambung TERBESAR dari sisa_lama  <- ini yang terlihat
  garis      = piksel garis balon yang tersentuh mask baru (WAJIB 0)
  ambang     = max(30, 0.002*w*h), gerbang find_residue sekarang

Kalau blob_lama besar tapi sisa_lama <= ambang, titik kotor itu memang lolos
tanpa eskalasi. Itu hipotesis yang diuji.
"""
import sys, os, pickle, numpy as np, cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MANGATL_WORK", ROOT)
os.environ.setdefault("MANGATL_ROOT", os.path.join(ROOT, ".stage"))
sys.path.insert(0, os.path.join(ROOT, '.stage'))
from config import SETTINGS

DEV = SETTINGS.residue_deviation
img = cv2.imread(os.path.join(ROOT, 'jepang_002.webp'))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
with open(os.path.join(ROOT, '.pagediag.pkl'), 'rb') as fh:
    st = pickle.load(fh)


def el(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(k) | 1, int(k) | 1))


def biggest(m):
    if not m.any():
        return 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return 0 if n <= 1 else int(stats[1:, cv2.CC_STAT_AREA].max())


print(f"residue_deviation={DEV}", flush=True)
print("  region       sisa_lama blob_lama  sisa_k3 blob_k3  sisa_k5 blob_k5 "
      "garis3 garis5 ambang lolos_gerbang", flush=True)
rows = []
for d in st['regs']:
    ink0 = d['ink_mask']
    if ink0 is None:
        continue
    x1, y1, x2, y2 = d['bbox']
    bx1, by1, bx2, by2 = d['bubble_bbox'] or d['bbox']
    bm = d['bubble_mask']
    if bm is None or bm.shape[:2] != (by2 - by1, bx2 - bx1):
        bm = np.full((by2 - by1, bx2 - bx1), 255, np.uint8)

    inkp = np.zeros(gray.shape, np.uint8)
    mh, mw = ink0.shape[:2]
    yy2, xx2 = min(y2, y1 + mh), min(x2, x1 + mw)
    inkp[y1:yy2, x1:xx2] = ink0[:yy2 - y1, :xx2 - x1]
    itr = np.zeros(gray.shape, np.uint8)
    itr[by1:by1 + bm.shape[0], bx1:bx1 + bm.shape[1]] = bm

    # median latar persis seperti pixel_residue: di dalam bbox, di luar ink_mask
    crop = gray[y1:y2, x1:x2].astype(np.int16)
    sub = inkp[y1:y2, x1:x2] > 0
    bg = float(np.median(crop[~sub])) if (~sub).any() else 255.0

    def leftover(mask_page):
        """Sisa menyimpang di dalam interior setelah dicat pakai mask_page."""
        painted = mask_page[y1:y2, x1:x2] > 0
        ins = itr[y1:y2, x1:x2] > 0
        dev = np.abs(crop - bg) > DEV
        return dev & ins & ~painted

    def outline_touch(mask_page):
        oline = (gray[by1:by2, bx1:bx2].astype(np.int16) - bg)
        oline = (np.abs(oline) > DEV) & (bm == 0)
        return int((oline & (mask_page[by1:by2, bx1:bx2] > 0)).sum())

    l0 = leftover(inkp)
    g3 = cv2.bitwise_and(cv2.dilate(inkp, el(3)), itr)
    g5 = cv2.bitwise_and(cv2.dilate(inkp, el(5)), itr)
    l3, l5 = leftover(g3), leftover(g5)
    thr = max(30, int(0.002 * (x2 - x1) * (y2 - y1)))
    s0 = int(l0.sum())
    rows.append((s0, biggest(l0), int(l3.sum()), biggest(l3),
                 int(l5.sum()), biggest(l5), thr))
    print(f"  r{d['idx']:<2d} {' '*8} {s0:6d} {biggest(l0):8d} "
          f"{int(l3.sum()):8d} {biggest(l3):7d} {int(l5.sum()):8d} "
          f"{biggest(l5):7d} {outline_touch(g3):6d} {outline_touch(g5):6d} "
          f"{thr:6d} {'YA' if s0 <= thr else 'tidak'}", flush=True)

a = np.array(rows)
print(f"\n  TOTAL sisa_lama={a[:,0].sum()} sisa_k3={a[:,2].sum()} "
      f"sisa_k5={a[:,4].sum()}", flush=True)
print(f"  blob terbesar: lama={a[:,1].max()} k3={a[:,3].max()} k5={a[:,5].max()}",
      flush=True)
lolos = int((a[:, 0] <= a[:, 6]).sum())
print(f"  region yang LOLOS gerbang find_residue padahal masih ada sisa: "
      f"{int(((a[:,0] <= a[:,6]) & (a[:,0] > 0)).sum())}/{len(a)}", flush=True)
print(f"  blob terbesar di region yang lolos gerbang: "
      f"{a[a[:,0] <= a[:,6], 1].max() if lolos else 0}", flush=True)

print("\n== usulan gerbang: komponen terbesar, bukan jumlah total ==", flush=True)
for cap in (8, 12, 16, 24, 32):
    old_fail = int((a[:, 0] > a[:, 6]).sum())
    new_fail = int(((a[:, 0] > a[:, 6]) | (a[:, 1] > cap)).sum())
    k3_fail = int(((a[:, 2] > a[:, 6]) | (a[:, 3] > cap)).sum())
    print(f"  blob>{cap:2d}: gagal lama={old_fail} -> gagal_gerbang_baru="
          f"{new_fail} | setelah mask k3 gagal={k3_fail}", flush=True)
