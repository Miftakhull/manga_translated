"""Uji dua perubahan sekaligus, plus sisa BUATAN supaya gerbangnya tidak jadi buta.

Temuan _erasewho.py: 36 px r7 dan 103 px r8 yang ditandai `pixel_residue` sama
sekali TIDAK diubah erase (utuh=sisa), NOL-nya di dalam bubble_mask, dan nol di
bawah guard. Itu art di luar balon yang ikut terbawa ink_mask — bukan sisa
tinta. Mengecatnya (kandidat `| ink_mask`) menaruh putih 255 di atas art gray
12-18: 7 bercak terlihat, yaitu cacat #3. Membiarkannya membuat find_residue
memanggil escalate(), dan mask eskalasinya memakan 98-152 px garis balon
(_esc.py). Dua-duanya rusak, jadi yang salah adalah LINGKUP tuduhan.

Dua perubahan yang diuji:
  P1  lingkup pixel_residue/residue_blob dikurung ke interior balon,
      yaitu (ink_mask | cincin3) & bubble_mask, jatuh ke perilaku lama kalau
      region itu tidak punya bubble_mask.
  P2  mask escalate() dikurung ke interior balon juga — supaya andai ada
      tuduhan yang lolos, LaMa tidak menggerus garis balon.

Bahaya nyata dari P1 adalah menjadi BUTA: kalau lingkupnya menyusut, sisa
yang sungguhan bisa tidak tertangkap. Jadi diuji dengan SISA BUATAN: satu
komponen tinta di DALAM interior dikembalikan ke nilai aslinya di halaman
bersih, meniru "erase melewatkan satu stroke". Gerbangnya WAJIB tetap
menandainya, dan P2 WAJIB tetap menutupinya.

Empat baris per skenario (fill ADA / fill DIBUANG):
  murni       = halaman bersih apa adanya
  +sisa       = halaman bersih + satu komponen tinta dipulihkan di interior
Kolom:
  lama/kurung = region yang ditandai gerbang lama vs gerbang P1
  tutup       = px sisa buatan yang tertutup mask eskalasi P2 (harus ~100%)
  garis       = px guard yang dimakan mask eskalasi (lama vs P2; P2 harus 0)
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


def build(drop_fill=False):
    regs = []
    for d in st['regs']:
        r = Region(idx=d['idx'], bbox=d['bbox'], det_class=d['det_class'],
                   det_conf=d['det_conf'])
        r.bubble_bbox, r.bubble_mask = d['bubble_bbox'], d['bubble_mask']
        r.ink_mask = d['ink_mask'].copy()
        r.est_font_size = d['est_font_size']
        if not drop_fill:
            r.fill_bbox, r.fill_mask = d['fill_bbox'], d['fill_mask']
        regs.append(r)
    return regs


def interior(r, shape):
    itr = np.zeros(shape, np.uint8)
    if r.bubble_mask is None or r.bubble_bbox is None:
        return itr
    bx1, by1 = r.bubble_bbox[0], r.bubble_bbox[1]
    bh, bw = r.bubble_mask.shape[:2]
    yy, xx = min(by1 + bh, shape[0]), min(bx1 + bw, shape[1])
    itr[by1:yy, bx1:xx] = r.bubble_mask[:yy - by1, :xx - bx1]
    return itr


def blob(m):
    if not m.any():
        return 0
    n, _l, s, _c = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return 0 if n <= 1 else int(s[1:, cv2.CC_STAT_AREA].max())


def gates(clean, regs, kurung):
    """Region mana yang ditandai. kurung=True -> lingkup P1."""
    out = []
    for r in regs:
        got = verify._residue_scope(clean, r)
        if got is None:
            continue
        dev, scope = got
        if kurung:
            x1, y1 = r.bbox[0], r.bbox[1]
            h, w = scope.shape
            itr = interior(r, clean.shape[:2])[y1:y1 + h, x1:x1 + w] > 0
            if itr.any():
                scope = scope & itr
        hit = dev & scope
        thr = max(30, int(0.002 * r.width * r.height))
        if int(hit.sum()) > thr or blob(hit) > SETTINGS.residue_blob_max:
            out.append(r)
    return out


def esc_mask(regs, failed, shape, kurung, k=5):
    m = np.zeros(shape, np.uint8)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    for r in failed:
        if r.ink_mask is None:
            continue
        x1, y1, x2, y2 = r.bbox
        mh, mw = r.ink_mask.shape[:2]
        y2, x2 = min(y2, y1 + mh, shape[0]), min(x2, x1 + mw, shape[1])
        g = cv2.dilate(r.ink_mask[:y2 - y1, :x2 - x1], el)
        m[y1:y2, x1:x2] = np.maximum(m[y1:y2, x1:x2], g)
        if kurung:
            itr = interior(r, shape)
            if itr.any():
                keep = np.zeros(shape, np.uint8)
                keep[y1:y2, x1:x2] = m[y1:y2, x1:x2]
                m = np.where(cv2.bitwise_and(keep, itr) > 0, m, np.where(keep > 0, 0, m))
                m = m.astype(np.uint8)
    return m


def inject(clean, regs, page):
    """Pulihkan satu komponen tinta DI DALAM interior: meniru stroke terlewat."""
    for r in regs:
        if r.ink_mask is None:
            continue
        x1, y1 = r.bbox[0], r.bbox[1]
        itr = interior(r, page.shape[:2])
        ink = np.zeros(page.shape[:2], np.uint8)
        mh, mw = r.ink_mask.shape[:2]
        yy, xx = min(y1 + mh, page.shape[0]), min(x1 + mw, page.shape[1])
        ink[y1:yy, x1:xx] = r.ink_mask[:yy - y1, :xx - x1]
        cand = cv2.bitwise_and(ink, itr)
        if not cand.any():
            continue
        n, lab, s, _c = cv2.connectedComponentsWithStats(cand, 8)
        order = sorted(range(1, n), key=lambda i: -s[i, cv2.CC_STAT_AREA])
        for i in order:
            if s[i, cv2.CC_STAT_AREA] < 20:
                break
            comp = lab == i
            out = clean.copy()
            out[comp] = page[comp]
            return out, r, comp
    return None, None, None


for tag, drop in (("fill ADA", False), ("fill DIBUANG", True)):
    regs = build(drop)
    tm.protect_bubble_outline(page, regs)
    guard = tm.bubble_outline_guard(page, regs)
    clean = erase.erase_page(page.copy(), regs, device="cpu")
    print(f"\n== {tag} ==", flush=True)
    for name, img in (("murni", clean),) :
        la, ku = gates(img, regs, False), gates(img, regs, True)
        print(f"  {name:<6s} lama={[r.idx for r in la]} kurung={[r.idx for r in ku]}",
              flush=True)
        for nm, f, kk in (("lama", la, False), ("P2", la, True)):
            m = esc_mask(regs, f, page.shape[:2], kk)
            print(f"         eskalasi({nm}) garis_termakan="
                  f"{int(((m > 0) & (guard > 0)).sum())}", flush=True)
    inj, r_inj, comp = inject(clean, regs, page)
    if inj is None:
        print("  +sisa  TIDAK BISA disuntik (tidak ada komponen >=20 px di interior)",
              flush=True)
        continue
    la, ku = gates(inj, regs, False), gates(inj, regs, True)
    print(f"  +sisa  disuntik di r{r_inj.idx} ({int(comp.sum())} px) "
          f"lama={[r.idx for r in la]} kurung={[r.idx for r in ku]}", flush=True)
    for nm, kk in (("lama", False), ("P2", True)):
        m = esc_mask(regs, ku, page.shape[:2], kk)
        print(f"         eskalasi({nm}) tutup={int(((m > 0) & comp).sum())}/"
              f"{int(comp.sum())} garis_termakan="
              f"{int(((m > 0) & (guard > 0)).sum())}", flush=True)
