"""Apakah MENANDAI r12 benar-benar memperbaikinya — dan tanpa merusak garis balon?

Gerbang baru menandai r12 (blob 30 px). Menandai saja tidak ada gunanya kalau
`escalate()` tidak bisa menjangkau piksel itu, atau kalau jangkauannya justru
memakan garis balon yang baru saja dilindungi `protect_bubble_outline`.

escalate() memekarkan `r.ink_mask` dengan kernel 5 lalu 7, jadi dua hal diukur
di sini untuk kedua kernel:
  tutup   = berapa dari 70 px sisa r12 yang MASUK mask eskalasi (harus ~semua)
  garis   = berapa px garis balon (bubble_outline_guard) yang ikut masuk mask
            eskalasi — inilah risikonya, dan angka ini yang menentukan apakah
            eskalasi perlu dikurung ke interior balon.

Kalau `garis` > 0, jalur eskalasi mengembalikan cacat "garis balon putus" yang
protect_bubble_outline ada untuk mencegahnya, dan mask eskalasi harus dikurung.
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

for tag, drop in (("A fill ADA", False), ("B fill DIBUANG", True)):
    regs = []
    for d in st['regs']:
        r = Region(idx=d['idx'], bbox=d['bbox'], det_class=d['det_class'],
                   det_conf=d['det_conf'])
        r.bubble_bbox = d['bubble_bbox']
        r.bubble_mask = d['bubble_mask']
        r.ink_mask = d['ink_mask'].copy()
        r.est_font_size = d['est_font_size']
        if not drop:
            r.fill_bbox, r.fill_mask = d['fill_bbox'], d['fill_mask']
        regs.append(r)
    tm.protect_bubble_outline(page, regs)
    guard = tm.bubble_outline_guard(page, regs)
    clean = erase.erase_page(page.copy(), regs, device="cpu")
    failed = [r for r in regs
              if verify.pixel_residue(clean, r) > max(30, int(0.002 * r.width * r.height))
              or verify.residue_blob(clean, r) > SETTINGS.residue_blob_max]
    print(f"\n== {tag} == ditandai: {[r.idx for r in failed]}", flush=True)
    for r in failed:
        got = verify._residue_scope(clean, r)
        dev, scope = got
        hit = dev & scope
        x1, y1 = r.bbox[0], r.bbox[1]
        for attempt in (1, 2):
            k = 3 + 2 * attempt
            el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            m = np.zeros(page.shape[:2], np.uint8)
            mh, mw = r.ink_mask.shape[:2]
            y2, x2 = min(r.bbox[3], y1 + mh), min(r.bbox[2], x1 + mw)
            m[y1:y2, x1:x2] = cv2.dilate(r.ink_mask[:y2 - y1, :x2 - x1], el)
            sub = m[y1:y1 + hit.shape[0], x1:x1 + hit.shape[1]] > 0
            cov = int((hit & sub).sum())
            gg = int(((m > 0) & (guard > 0)).sum())
            # kalau dikurung ke interior balon
            itr = np.zeros(page.shape[:2], np.uint8)
            if r.bubble_mask is not None:
                bx1, by1 = r.bubble_bbox[0], r.bubble_bbox[1]
                bh, bw = r.bubble_mask.shape[:2]
                itr[by1:by1 + bh, bx1:bx1 + bw] = r.bubble_mask[:bh, :bw]
            mk = cv2.bitwise_and(m, itr) if itr.any() else m
            subk = mk[y1:y1 + hit.shape[0], x1:x1 + hit.shape[1]] > 0
            print(f"  r{r.idx} sisa={int(hit.sum()):4d} k={k}: "
                  f"tutup={cov:4d}/{int(hit.sum()):4d} garis_termakan={gg:4d} | "
                  f"dikurung: tutup={int((hit & subk).sum()):4d} "
                  f"garis={int(((mk > 0) & (guard > 0)).sum()):4d}", flush=True)
