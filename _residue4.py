"""Cacat #1: dua lubang yang membuat coretan LOLOS, diukur bukan didalilkan.

Jalur produksi yang relevan (pipeline.py:117-129, lalu 175-181):
  build_region_mask -> partition -> disjoin -> protect_bubble_outline
  -> erase_page -> find_residue -> escalate

Lubang 1 — protect_bubble_outline() MENGHAPUS piksel dari ink_mask.
Fungsinya benar: tinta Jepang yang menempel garis balon tidak boleh dicat,
kalau tidak garis balonnya ikut hilang. Tapi konsekuensinya piksel itu
(a) tidak dicat oleh jalur ink_mask, DAN (b) tidak pernah diperiksa
pixel_residue — karena lingkup pemeriksaannya `ink_mask > 0`, dan piksel itu
justru baru saja dibuang dari ink_mask. Selama fill_mask ada, isian interior
menutupinya. Begitu build_fill_mask menyerah, ia tertinggal di halaman dan
verify melaporkan NOL.

Lubang 2 — gerbangnya JUMLAH, bukan komponen terbesar. `max(30, 0.002*w*h)`
di region 96x164 = 31 px; satu titik 20-30 px lolos utuh, dan yang dilihat
mata adalah titik itu, bukan jumlahnya.

Diukur di sini, dengan ink_mask SEBELUM dan SESUDAH protect_bubble_outline:
  dicuri   = px yang dibuang protect_bubble_outline dari ink_mask
  tinggal  = dari `dicuri`, yang MASIH menyimpang setelah erase (skenario tanpa
             fill_mask, yaitu keadaan Colab)
  blob     = komponen terbesar dari `tinggal` — inilah "titik hitam"
  lihat_?  = apakah pixel_residue sekarang melihatnya (lingkup ink_mask)
  lihat_D  = apakah lingkup dilate(ink_mask,3) & interior melihatnya

Lingkup usulan sengaja DIKURUNG ke interior balon dan hanya 3 px di sekitar
bekas tinta — bukan seluruh interior. Seluruh interior salah: sudut kotak balon
berisi ART, dan setelah perbaikan cacat #3 art itu memang TIDAK dicat lagi,
jadi lingkup seluruh interior akan melaporkan art sebagai sisa lalu
mengeskalasi inpaint ke atasnya. Terukur: r9 punya 198 px art di sudut
kanan-bawah interior (halaman y1280-1303 x608-625, gray 218-234).
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


def interior_local(r, shape):
    """bubble_mask dipetakan ke bbox region, atau None."""
    if r.bubble_mask is None or r.bubble_bbox is None:
        return None
    big = np.zeros(page.shape[:2], np.uint8)
    bx1, by1 = r.bubble_bbox[0], r.bubble_bbox[1]
    bh, bw = r.bubble_mask.shape[:2]
    yy, xx = min(by1 + bh, page.shape[0]), min(bx1 + bw, page.shape[1])
    big[by1:yy, bx1:xx] = r.bubble_mask[:yy - by1, :xx - bx1]
    x1, y1 = r.bbox[0], r.bbox[1]
    return big[y1:y1 + shape[0], x1:x1 + shape[1]]


def biggest(m):
    if not m.any():
        return 0
    n, _l, s, _c = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return 0 if n <= 1 else int(s[1:, cv2.CC_STAT_AREA].max())


for tag, drop in (("A fill_mask ADA (halaman lokal apa adanya)", False),
                  ("B fill_mask DIBUANG (jalur ink_mask = keadaan Colab)", True)):
    before = {d['idx']: d['ink_mask'].copy() for d in st['regs']}
    regs = build(drop)
    freed = tm.protect_bubble_outline(page, regs)
    clean = erase.erase_page(page.copy(), regs, device="cpu")
    gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.int16)
    print(f"\n== {tag} ==", flush=True)
    print(f"  protect_bubble_outline melepas {freed} px dari ink_mask", flush=True)
    print("  region dicuri tinggal blob  lihat_lama lihat_D  ambang "
          "verdict_lama verdict_D", flush=True)
    tot = np.zeros(5, int)
    for r in regs:
        x1, y1, x2, y2 = r.bbox
        now = r.ink_mask > 0
        old = before[r.idx][:now.shape[0], :now.shape[1]] > 0
        stolen = old & ~now
        crop = gray[y1:y1 + now.shape[0], x1:x1 + now.shape[1]]
        bg = float(np.median(crop[~old])) if (~old).any() else 255.0
        dev = np.abs(crop - bg) > SETTINGS.residue_deviation
        itr = interior_local(r, now.shape)
        ins = np.ones(now.shape, bool) if itr is None else (itr > 0)
        left = dev & stolen & ins
        wide = cv2.dilate((old).astype(np.uint8) * 255, EL3) > 0
        scope_d = wide & ins
        lama = int((dev & now).sum())
        vd = int((dev & scope_d).sum())
        thr = max(30, int(0.002 * r.width * r.height))
        b = biggest(left)
        tot += [int(lama > thr), int(vd > thr), b, int(stolen.sum()), 1]
        print(f"  r{r.idx:<2d} {int(stolen.sum()):6d} {int(left.sum()):7d} "
              f"{b:5d} {lama:10d} {vd:7d} {thr:7d} "
              f"{'GAGAL' if lama > thr else '  ok ':6s} "
              f"{'GAGAL' if vd > thr else '  ok '}", flush=True)
    print(f"  dicuri total={tot[3]}  gagal lama={tot[0]}/{tot[4]} "
          f"gagal_lingkup_D={tot[1]}/{tot[4]}  blob terbesar={tot[2]}", flush=True)
