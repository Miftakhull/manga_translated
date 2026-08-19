"""Buktikan r8 'ヒ．．．ッ！？' memang DI DALAM balon menurut detector sungguhan.

Ini penting karena cabang labelnya beda total: kalau bubble_bbox None,
_label_region masuk cabang luar-balon (n<=3 -> SFX) dan perbaikan aturan K
tidak akan menyentuhnya sama sekali. Sidecar hasilnew4 TIDAK menyimpan
bubble_bbox (kuncinya tidak ada), jadi det_class saja bukan bukti — harus
dari detect.assign_bubbles yang sungguhan.

Menulis _h5bub_r8.png (crop 3x balon itu) supaya bisa dilihat mata juga.
Probe murni: tidak menyentuh _nbsrc/ maupun notebook.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import cv2

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

import detect                                                   # noqa: E402
import translate as tl                                          # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

H4 = ROOT / "hasilnew4"
REP = json.loads((H4 / "hitomi_3740721_015.json").read_text(encoding="utf-8"))
img = cv2.imread(str(H4 / "hitomi_3740721_015.webp"), cv2.IMREAD_COLOR)
if img is None:
    sys.exit("gagal baca halaman asli")
H, W = img.shape[:2]
print(f"halaman asli {W}x{H}", flush=True)

TARGET = (485, 1140, 525, 1237)          # r8 di sidecar hasilnew4
rows = REP["regions"] if isinstance(REP, dict) else REP


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if not inter:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua else 0.0


regions, bubbles = detect.detect(img)
print(f"region={len(regions)} bubble={len(bubbles)}", flush=True)

hit = max(regions, key=lambda r: iou(r.bbox, TARGET))
print(f"\nregion paling cocok dengan sidecar r8 {TARGET}:", flush=True)
print(f"  idx={hit.idx} bbox={hit.bbox} iou={iou(hit.bbox, TARGET):.3f}",
      flush=True)
print(f"  bubble_bbox = {hit.bubble_bbox}", flush=True)
print(f"  det_class   = {getattr(hit, 'det_class', None)}", flush=True)
print(f"  IN_BUBBLE   = {hit.bubble_bbox is not None}   <-- penentu cabang label",
      flush=True)

# jalankan gerbang label sungguhan dengan src_text sidecar
hit.src_text = "ヒ．．．ッ！？"
tl._label_region(hit)
core = tl._sfx_core(hit.src_text)
print(f"\ngerbang label pada teks sidecar:", flush=True)
print(f"  src_text = {hit.src_text!r}", flush=True)
print(f"  _sfx_core -> {core!r}  (n={len(core)})", flush=True)
print(f"  _has_kanji={tl._has_kanji(hit.src_text)} "
      f"_all_katakana={tl._all_katakana(core)} "
      f"ekor_kecil={core[-1] in tl._SMALL if core else None}", flush=True)
print(f"  label   = {hit.label}   protected={hit.label in tl.PROTECTED_LABELS}",
      flush=True)

x0, y0, x1, y1 = hit.bbox
pad = 60
cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
crop = img[cy0:cy1, cx0:cx1]
crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(str(ROOT / "_h5bub_r8.png"), crop)
print(f"\n -> _h5bub_r8.png (crop 3x, pad {pad}px)", flush=True)
