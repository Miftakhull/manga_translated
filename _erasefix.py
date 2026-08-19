"""Cacat #1, mekanisme yang AKHIRNYA terukur: erase_flat berhenti di fill_mask.

erase.erase_flat baris 146-152:

    fill = _fill_on_page(...)
    if fill is not None:
        sel = fill > 0
        if guard is not None: sel &= guard == 0
        img[sel] = region.bg_color
        return img          # <-- ink_mask TIDAK PERNAH dicat

fill_mask adalah INTERIOR balon. Tinta Jepang yang duduk di luar interior —
glyph yang menempel/menembus garis balon, dan bagian yang dipotong
`disjoin_overlapping_interiors` — tidak masuk fill_mask, jadi tidak dicat
sama sekali. Terukur di halaman referensi dengan est_font_size terisi seperti
produksi: r7 36 px dan r8 103 px tinta tertinggal, SELURUHNYA di luar
bubble_mask (luar_itr = 36/36 dan 103/103), nol di bawah guard.

Halaman lokal tetap melaporkan residue_count=0 karena find_residue MENANDAI
keduanya dan `escalate()` menambalnya dengan LaMa. Tapi harga tambalan itu
terukur di _esc.py: mask eskalasi kernel 5 memakan 98 px (r7) dan 152 px (r8)
garis balon — persis yang `protect_bubble_outline` ada untuk mencegah. Jadi
halaman "lulus" dengan garis balon tergerus, dan di halaman lain yang tidak
tertandai sisanya tinggal sebagai titik kotor.

Calon perbaikan: cat GABUNGAN fill_mask | ink_mask, tetap dikurangi guard.
  * `| ink_mask` tidak bisa merusak garis balon, karena `guard` (SFX + garis)
    dikurangkan setelahnya — penjaga yang sama yang sudah dipakai jalur fill.
  * ink_mask sudah dibersihkan protect_bubble_outline, jadi piksel garis sudah
    dibuang dari sana lebih dulu.

Yang diukur di sini, dengan est_font_size TERISI:
  sisa    = pixel_residue setelah erase (harus turun ke 0)
  blob    = komponen terbesar
  ditandai= apakah find_residue masih perlu mengeskalasi (harus tidak)
  garis   = px garis balon yang ikut tercat (WAJIB 0)
  sfx     = px guard SFX yang ikut tercat  (WAJIB 0)
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
ORIG = erase.erase_flat


def patched(img, region, guard=None):
    """erase_flat calon: cat fill_mask | ink_mask, dikurangi guard."""
    if region.bg_color is None:
        return img
    fill = erase._fill_on_page(region, img.shape[:2]) if SETTINGS.bubble_fill else None
    if fill is None:
        return ORIG(img, region, guard)
    sel = fill > 0
    if region.ink_mask is not None:
        x1, y1, x2, y2 = region.bbox
        mh, mw = region.ink_mask.shape[:2]
        y2, x2 = min(y2, y1 + mh, img.shape[0]), min(x2, x1 + mw, img.shape[1])
        if y2 > y1 and x2 > x1:
            sel[y1:y2, x1:x2] |= region.ink_mask[:y2 - y1, :x2 - x1] > 0
    if guard is not None:
        sel &= guard == 0
    img[sel] = region.bg_color
    return img


def build():
    regs = []
    for d in st['regs']:
        r = Region(idx=d['idx'], bbox=d['bbox'], det_class=d['det_class'],
                   det_conf=d['det_conf'])
        r.bubble_bbox = d['bubble_bbox']
        r.bubble_mask = d['bubble_mask']
        r.ink_mask = d['ink_mask'].copy()
        r.est_font_size = d['est_font_size']
        r.fill_bbox, r.fill_mask = d['fill_bbox'], d['fill_mask']
        regs.append(r)
    return regs


for tag, fn in (("LAMA (fill_mask saja)", ORIG),
                ("CALON (fill_mask | ink_mask)", patched)):
    regs = build()
    tm.protect_bubble_outline(page, regs)
    guard = tm.bubble_outline_guard(page, regs)
    before = page.copy()
    erase.erase_flat = fn
    try:
        clean = erase.erase_page(page.copy(), regs, device="cpu")
    finally:
        erase.erase_flat = ORIG
    changed = (clean != before).any(2)
    print(f"\n== {tag} ==", flush=True)
    print(f"  garis balon tercat = {int((changed & (guard > 0)).sum())}  "
          f"(guard total {int((guard > 0).sum())} px)", flush=True)
    print("  region  sisa  blob ambang  ditandai", flush=True)
    n_bad = 0
    for r in regs:
        v = verify.pixel_residue(clean, r)
        b = verify.residue_blob(clean, r)
        thr = max(30, int(0.002 * r.width * r.height))
        bad = v > thr or b > SETTINGS.residue_blob_max
        n_bad += bad
        if v or b:
            print(f"  r{r.idx:<2d} {v:5d} {b:5d} {thr:6d}  "
                  f"{'GAGAL' if bad else 'ok'}", flush=True)
    print(f"  total region ditandai find_residue = {n_bad}/13", flush=True)
