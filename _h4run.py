"""Jalankan pipeline SUNGGUHAN pada halaman asli hasilnew4, DUA kali:

  LAMA  drop_nested_duplicates() dimatikan  -> harus mereproduksi cacatnya
  BARU  drop_nested_duplicates() menyala    -> harus menghilangkannya

Yang dilewati cuma OCR dan terjemahan (butuh manga-ocr + jaringan); src_text
disuntik dari laporan hasilnew4 supaya label/SFX keluar sama seperti di Colab.
Semua yang lain — detect, ctd_soft_mask, semua mask, erase, find_residue —
adalah kode yang sama yang dijalankan Colab.

Yang dibuktikan:
  1. LAMA memang membelah balon kanan-atas (shared_bubble_bbox terisi) dan BARU
     tidak — jadi jahitan dua warna itu memang lahir di situ
  2. residue_idx LAMA sama dengan yang tercatat di laporan (probe-nya valid)
  3. residue_idx BARU
  4. crop balon kanan-atas dari kedua run, untuk dilihat mata

Probe murni: menulis _h4run_*.png, tidak menyentuh _nbsrc/ maupun notebook.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / ".stage"
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(STAGE))
STAGE.mkdir(exist_ok=True)
_MAGIC = re.compile(r"^%%writefile\b.*\n?")
for p in sorted((ROOT / "_nbsrc").glob("*.py")):
    (STAGE / p.name).write_text(
        _MAGIC.sub("", p.read_text(encoding="utf-8"), count=1), encoding="utf-8")
sys.path.insert(0, str(STAGE))

import detect                     # noqa: E402
import erase                      # noqa: E402
import textmask                   # noqa: E402
import translate as tl            # noqa: E402
import typeset                    # noqa: E402
import verify                     # noqa: E402

H4 = ROOT / "hasilnew4"
REP = json.loads((H4 / "hitomi_3740721_015.json").read_text(encoding="utf-8"))
img = cv2.imread(str(H4 / "hitomi_3740721_015.webp"), cv2.IMREAD_COLOR)
if img is None:
    sys.exit("gagal baca halaman asli")
H, W = img.shape[:2]
print(f"halaman asli {W}x{H}\n", flush=True)

_OLD = REP["regions"]


def inject_src(regions) -> None:
    """Salin src_text dari laporan ke region dengan bbox paling bertumpang."""
    for r in regions:
        ax1, ay1, ax2, ay2 = r.bbox
        best, best_iou = None, 0.0
        for o in _OLD:
            bx1, by1, bx2, by2 = o["bbox"]
            iw = min(ax2, bx2) - max(ax1, bx1)
            ih = min(ay2, by2) - max(ay1, by1)
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            union = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1)
                     - inter)
            iou = inter / max(union, 1)
            if iou > best_iou:
                best, best_iou = o, iou
        r.src_text = (best or {}).get("src_text") or ""


def run(dedupe: bool) -> dict:
    real = detect.drop_nested_duplicates
    if not dedupe:
        detect.drop_nested_duplicates = lambda regs: 0
    try:
        typeset.set_page_width(W)
        regions, bubbles = detect.detect(img)
        soft = textmask.ctd_soft_mask(img)
        for r in regions:
            textmask.build_region_mask(img, r, soft)
        nsplit = textmask.partition_shared_interiors(img, regions)
        ndis = textmask.disjoin_overlapping_interiors(img, regions)
        textmask.protect_bubble_outline(img, regions)
        inject_src(regions)
        tl._fallback_labels(regions)
        emask, pmask = textmask.compose_page_mask(img, regions)
        sfx_ok = verify.assert_sfx_intact(emask, pmask)
        cleaned = erase.erase_page(img, regions, "cpu")
        failed = verify.find_residue(cleaned, regions)
    finally:
        detect.drop_nested_duplicates = real
    return dict(regions=regions, bubbles=bubbles, cleaned=cleaned,
                failed=failed, nsplit=nsplit, ndis=ndis, sfx_ok=sfx_ok)


def topright(regions):
    """Region yang bbox-nya bersinggungan dengan balon kanan-atas."""
    return [r for r in regions
            if r.bbox[0] >= 800 and r.bbox[1] < 450 and r.bbox[3] < 450]


out = {}
for tag, ded in (("LAMA", False), ("BARU", True)):
    print(f"===== {tag} (drop_nested_duplicates {'ON' if ded else 'OFF'}) =====",
          flush=True)
    res = run(ded)
    out[tag] = res
    rs = res["regions"]
    print(f"  region={len(rs)} bubble={len(res['bubbles'])} "
          f"partisi_lobus={res['nsplit']} disjoin={res['ndis']} "
          f"sfx_utuh={res['sfx_ok']}", flush=True)
    print(f"  residue_idx={sorted(r.idx for r in res['failed'])}", flush=True)
    print("  balon kanan-atas:", flush=True)
    for r in topright(rs):
        fc = erase.fill_color(img, r)
        print(f"    r{r.idx} bbox={r.bbox} bubble={r.bubble_bbox}\n"
              f"        shared={r.shared_bubble_bbox} fill={fc} "
              f"label={r.label} src={r.src_text[:22]!r}", flush=True)
    # crop balon kanan-atas dari hasil erase
    crop = res["cleaned"][120:420, 820:1040]
    s = 2
    cv2.imwrite(str(ROOT / f"_h4run_{tag}.png"),
                cv2.resize(crop, (crop.shape[1] * s, crop.shape[0] * s),
                           interpolation=cv2.INTER_NEAREST))
    print("", flush=True)

print("===== putusan =====", flush=True)
tl_lama, tl_baru = topright(out["LAMA"]["regions"]), topright(out["BARU"]["regions"])
split_lama = [r for r in tl_lama if r.shared_bubble_bbox is not None]
split_baru = [r for r in tl_baru if r.shared_bubble_bbox is not None]
print(f"  balon kanan-atas: LAMA {len(tl_lama)} region "
      f"({len(split_lama)} bertanda balon-terbelah)  ->  "
      f"BARU {len(tl_baru)} region ({len(split_baru)} terbelah)", flush=True)
fl = sorted(r.idx for r in out["LAMA"]["failed"])
fb = sorted(r.idx for r in out["BARU"]["failed"])
print(f"  residue: LAMA {fl}  ->  BARU {fb}", flush=True)
print(f"  laporan Colab mencatat residue_idx {REP['residue_idx']} "
      f"({REP['region_count']} region)", flush=True)
print("\n -> _h4run_LAMA.png / _h4run_BARU.png (balon kanan-atas setelah erase)",
      flush=True)
