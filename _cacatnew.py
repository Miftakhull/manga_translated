"""Ukur struktur balon-ganda pada dua crop cacat baru — read-only, tanpa OCR.

cacatbaru/jp_cacatnew1.JPG dan jp_cacatnew2.JPG dua-duanya balon MENYATU
(figura-8): satu lobus atas + satu lobus bawah yang bersambung. Hasilnya
(cacatnew1/2.JPG) menunjukkan pola yang sama di kedua halaman:

  - satu lobus TIDAK diterjemah: tinta Jepangnya masih utuh di tempatnya
  - terjemahan lobus itu MUNCUL, tapi mungil dan di LUAR balon (di atasnya):
    "OR..." di cacatnew1, "SO THAT IT.." di cacatnew2
  - lobus yang lain dicat rata melewati bentuk lobusnya sendiri

Yang diukur di sini: berapa region, berapa kotak balon, apakah dua region
berbagi SATU kotak balon (jalur _partition_shared_bubbles), dan seberapa
besar/di mana interior tiap region SETELAH partisi + disjoin. Tidak ada OCR
dan tidak ada LLM — murni geometri.
"""
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
NBSRC, STAGE = ROOT / "_nbsrc", ROOT / ".stage"
_MAGIC = re.compile(r"^%%writefile\b.*\n?")
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(STAGE))
STAGE.mkdir(exist_ok=True)
for _s in sorted(NBSRC.glob("*.py")):
    _b = _MAGIC.sub("", _s.read_text(encoding="utf-8"), count=1)
    _d = STAGE / _s.name
    if not _d.exists() or _d.read_text(encoding="utf-8") != _b:
        _d.write_text(_b, encoding="utf-8")
sys.path.insert(0, str(STAGE))

import detect      # noqa: E402
import imgio       # noqa: E402
import textmask    # noqa: E402
import typeset     # noqa: E402


def area(m):
    return 0 if m is None else int((m > 0).sum())


def probe(name):
    img = imgio.load_any(ROOT / "cacatbaru" / f"{name}.JPG")
    h, w = img.shape[:2]
    typeset.set_page_width(w)
    print(f"\n===== {name}  {w}x{h}", flush=True)

    regions, bubbles = detect.detect(img)
    print(f" region={len(regions)}  kotak_balon={len(bubbles)}", flush=True)
    for b in bubbles:
        print(f"   balon {b}  ukuran={b[2]-b[0]}x{b[3]-b[1]}", flush=True)

    soft = textmask.ctd_soft_mask(img)
    for r in regions:
        textmask.build_region_mask(img, r, soft)
    npart = textmask.partition_shared_interiors(img, regions)
    ndis = textmask.disjoin_overlapping_interiors(img, regions)
    textmask.protect_bubble_outline(img, regions)
    print(f" partition_shared={npart}  disjoin={ndis}", flush=True)

    print(" idx kelas        bbox                    bubble_bbox"
          "              shared        tinta interior", flush=True)
    for r in regions:
        bb = r.bubble_bbox
        sh = r.shared_bubble_bbox
        print(f"  r{r.idx} {r.det_class:<12s} {str(r.bbox):<22s} "
              f"{str(bb):<24s} {'YA ' + str(sh) if sh else '-':<14s} "
              f"{area(r.ink_mask):>5d} {area(r.bubble_mask):>6d}", flush=True)

    # Peta: interior tiap region diwarnai beda, tinta digambar putih.
    vis = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                       cv2.COLOR_GRAY2BGR)
    warna = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
             (255, 0, 255), (255, 255, 0)]
    for r in regions:
        if r.bubble_mask is None or r.bubble_bbox is None:
            continue
        bx1, by1, bx2, by2 = r.bubble_bbox
        m = r.bubble_mask
        mh, mw = m.shape[:2]
        yy, xx = min(by1 + mh, h), min(bx1 + mw, w)
        sub = m[: yy - by1, : xx - bx1] > 0
        tile = vis[by1:yy, bx1:xx]
        c = np.array(warna[r.idx % len(warna)], np.float32)
        tile[sub] = (tile[sub] * 0.55 + c * 0.45).astype(np.uint8)
        cv2.rectangle(vis, (bx1, by1), (bx2 - 1, by2 - 1),
                      warna[r.idx % len(warna)], 1)
    out = ROOT / f"_cn_{name}.png"
    cv2.imwrite(str(out), cv2.resize(vis, None, fx=3, fy=3,
                                     interpolation=cv2.INTER_NEAREST))
    print(f" -> {out.name}", flush=True)
    return regions


typeset.setup_fonts(verbose=False)
for _n in ("jp_cacatnew1", "jp_cacatnew2"):
    probe(_n)
