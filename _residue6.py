"""Ulang pengukuran cacat #1 dengan est_font_size TERISI seperti produksi.

_esc.py mengungkap kekeliruan pengukuran sendiri: _residue4/_residue5 membangun
Region tanpa `est_font_size`, jadi `bubble_outline_guard` memakai
`_stroke_px(20)` bawaan dan pita penjaganya berbeda dari produksi
(build_region_mask selalu mengisi est_font_size). Akibatnya r12 tampak
menyimpan 70 px sisa; dengan est_font_size terisi, piksel itu masuk pita
penjaga garis dan bukan sisa.

Jadi diukur ulang di sini, tiga definisi berdampingan, est_font_size TERISI:
  lama    = lingkup `ink_mask > 0`               (produksi sebelum patch)
  gabung  = lama | (dilate(ink,3) & interior)    (produksi setelah patch)
  blob    = komponen terbesar dari `gabung`

Plus pertanyaan yang muncul dari _esc.py: sisa r7/r8 seluruhnya berada di LUAR
bubble_mask (dikurung ke interior -> tutup=0), yaitu tinta Jepang yang menempel
garis balon dan sengaja diselamatkan `guard` di erase_flat. Yang diukur:
  luar_itr = berapa px sisa yang di luar interior balon
  di_guard = berapa px sisa yang memang berada di bawah guard
Kalau hampir semuanya di_guard, maka itu bukan kebocoran mask melainkan
konsekuensi sadar dari melindungi garis — dan yang benar bukan mengecatnya,
melainkan memastikan gerbang residue tidak menjadikannya alarm palsu abadi.
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


def build(drop_fill, with_font=True):
    regs = []
    for d in st['regs']:
        r = Region(idx=d['idx'], bbox=d['bbox'], det_class=d['det_class'],
                   det_conf=d['det_conf'])
        r.bubble_bbox = d['bubble_bbox']
        r.bubble_mask = d['bubble_mask']
        r.ink_mask = d['ink_mask'].copy()
        if with_font:
            r.est_font_size = d['est_font_size']
        if not drop_fill:
            r.fill_bbox, r.fill_mask = d['fill_bbox'], d['fill_mask']
        regs.append(r)
    return regs


def old_residue(clean, r):
    """pixel_residue versi SEBELUM patch: lingkup ink_mask saja."""
    if r.ink_mask is None:
        return 0
    x1, y1, x2, y2 = r.bbox
    crop = clean[y1:y2, x1:x2]
    if crop.size == 0:
        return 0
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    mh, mw = r.ink_mask.shape[:2]
    sub = r.ink_mask[:min(mh, g.shape[0]), :min(mw, g.shape[1])]
    area = g[:sub.shape[0], :sub.shape[1]]
    ins = sub > 0
    if not ins.any():
        return 0
    bg = float(np.median(area[~ins])) if (~ins).any() else 255.0
    return int(((np.abs(area.astype(np.int16) - bg) > SETTINGS.residue_deviation) & ins).sum())


for wf in (True, False):
    for tag, drop in (("fill ADA", False), ("fill DIBUANG", True)):
        regs = build(drop, wf)
        freed = tm.protect_bubble_outline(page, regs)
        guard = tm.bubble_outline_guard(page, regs)
        clean = erase.erase_page(page.copy(), regs, device="cpu")
        print(f"\n== est_font_size={'TERISI' if wf else 'NOL (keliru)'} | {tag} "
              f"| protect melepas {freed} px ==", flush=True)
        print("  region  lama gabung  blob luar_itr di_guard ambang verdict", flush=True)
        for r in regs:
            lama = old_residue(clean, r)
            tot = verify.pixel_residue(clean, r)
            blob = verify.residue_blob(clean, r)
            if not (lama or tot):
                continue
            got = verify._residue_scope(clean, r)
            hit = got[0] & got[1]
            x1, y1 = r.bbox[0], r.bbox[1]
            itr = np.zeros(page.shape[:2], np.uint8)
            if r.bubble_mask is not None:
                bx1, by1 = r.bubble_bbox[0], r.bubble_bbox[1]
                bh, bw = r.bubble_mask.shape[:2]
                itr[by1:by1 + bh, bx1:bx1 + bw] = r.bubble_mask
            si = itr[y1:y1 + hit.shape[0], x1:x1 + hit.shape[1]] > 0
            sg = guard[y1:y1 + hit.shape[0], x1:x1 + hit.shape[1]] > 0
            thr = max(30, int(0.002 * r.width * r.height))
            bad = tot > thr or blob > SETTINGS.residue_blob_max
            print(f"  r{r.idx:<2d} {lama:5d} {tot:6d} {blob:5d} "
                  f"{int((hit & ~si).sum()):8d} {int((hit & sg).sum()):8d} "
                  f"{thr:6d} {'GAGAL' if bad else 'ok'}", flush=True)
