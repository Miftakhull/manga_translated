"""Buktikan typeset di halaman SUNGGUHAN sesudah duplikat digabung.

_h4run.py berhenti di find_residue, jadi pertanyaan yang paling penting bagi
user belum terjawab: sesudah dua region duplikat jadi SATU, apakah hurufnya
saling timpa, dan apakah ada huruf yang termakan mask region lain?

Yang diukur (bukan cuma dilihat):
  1. overflow per region  (harus 0)
  2. piksel tinta yang diklaim DUA region sekaligus  (harus 0)
  3. piksel tinta di luar interior balon region-nya sendiri  (harus 0)
  4. SFX masih identik piksel dengan aslinya
Plus crop balon kanan-atas hasil typeset, untuk dilihat mata.

Probe murni: menulis _h4type_*.png, tidak menyentuh _nbsrc/ maupun notebook.
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
print(f"halaman asli {W}x{H}", flush=True)
_OLD = REP["regions"]


def best_match(bbox):
    ax1, ay1, ax2, ay2 = bbox
    best, best_iou = None, 0.0
    for o in _OLD:
        bx1, by1, bx2, by2 = o["bbox"]
        iw = min(ax2, bx2) - max(ax1, bx1)
        ih = min(ay2, by2) - max(ay1, by1)
        if iw <= 0 or ih <= 0:
            continue
        inter = iw * ih
        union = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
        iou = inter / max(union, 1)
        if iou > best_iou:
            best, best_iou = o, iou
    return best or {}


typeset.set_page_width(W)
fp = typeset.setup_fonts(verbose=False)
print(f"font terpilih = {fp!r}  FONT_USED={typeset.FONT_USED!r}", flush=True)
if not typeset.FONT_USED:
    sys.exit("FONT_USED kosong -> render_region akan diam-diam tidak menggambar")
regions, bubbles = detect.detect(img)
soft = textmask.ctd_soft_mask(img)
for r in regions:
    textmask.build_region_mask(img, r, soft)
nsplit = textmask.partition_shared_interiors(img, regions)
ndis = textmask.disjoin_overlapping_interiors(img, regions)
textmask.protect_bubble_outline(img, regions)
for r in regions:
    o = best_match(r.bbox)
    r.src_text = o.get("src_text") or ""
    r.translation = o.get("translation") or None
tl._fallback_labels(regions)
# region simbol-saja (perbaikan C) diselesaikan tanpa jaringan
for r in regions:
    if (r.label not in tl.PROTECTED_LABELS and r.translation is None
            and tl._symbols_only(r.src_text)):
        r.translation = tl._symbols_as_text(r.src_text) or None
emask, pmask = textmask.compose_page_mask(img, regions)
sfx_ok = verify.assert_sfx_intact(emask, pmask)
cleaned = erase.erase_page(img, regions, "cpu")
residue = verify.find_residue(cleaned, regions)
final = typeset.render_page(cleaned, regions)
print(f"region={len(regions)} partisi_lobus={nsplit} disjoin={ndis} "
      f"sfx_utuh={sfx_ok} residue={sorted(r.idx for r in residue)}", flush=True)

fail = 0


def check(name, ok, detail=""):
    global fail
    print(f"  [{'OK ' if ok else 'GAGAL'}] {name}{('  ' + detail) if detail else ''}",
          flush=True)
    if not ok:
        fail += 1


print("\n1) overflow", flush=True)
ov = [(r.idx, r.overflowed) for r in regions
      if r.overflowed]
check("tidak ada overflow", not ov, f"{ov}")
noft = [r.idx for r in regions
        if r.translation and not r.final_font_size]
check("setiap balon berterjemahan dapat ukuran font > 0", not noft, f"idx={noft}")

# --- tinta hasil typeset: beda antara cleaned dan final -------------------
print("\n2) tinta hasil typeset: saling timpa & termakan mask lain", flush=True)
diff = cv2.absdiff(cv2.cvtColor(final, cv2.COLOR_BGR2GRAY),
                   cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY))
ink = (diff > 24).astype(np.uint8)
print(f"   piksel tinta baru total = {int(ink.sum())}", flush=True)
# Tanpa penjaga ini, "tidak ada tinta saling timpa" lolos secara HAMPA saat
# typeset tidak menggambar apa pun sama sekali. Itu yang terjadi di run
# pertama: FONT_USED kosong -> render_region diam-diam return img.
check("typeset benar-benar menggambar sesuatu", int(ink.sum()) > 500,
      f"ink_px={int(ink.sum())}")

# klaim per region: tinta di dalam kotak region-nya
own = {}
for r in regions:
    if not r.translation:
        continue
    m = np.zeros((H, W), np.uint8)
    x1, y1, x2, y2 = r.bbox
    m[y1:y2, x1:x2] = 1
    own[r.idx] = ink * m
acc = np.zeros((H, W), np.uint16)
for m in own.values():
    acc += m
clash = int((acc > 1).sum())
check("tidak ada piksel tinta diklaim dua region", clash == 0, f"clash_px={clash}")

# tinta di luar interior balon sendiri
out_of_bubble = []
for r in regions:
    if not r.translation or r.idx not in own:
        continue
    itr = getattr(r, "bubble_mask", None)
    if itr is None or not isinstance(itr, np.ndarray) or itr.size == 0:
        continue
    box, bm = typeset._region_box_mask(r)
    bx1, by1, bx2, by2 = box
    sub = own[r.idx][by1:by2, bx1:bx2]
    if sub.shape != bm.shape:
        out_of_bubble.append((r.idx, "bentuk mask tidak cocok"))
        continue
    outside = int((sub & (bm == 0)).sum())
    if outside:
        out_of_bubble.append((r.idx, outside))
check("tinta tidak keluar interior balonnya sendiri", not out_of_bubble,
      f"{out_of_bubble}")

print("\n3) SFX tetap identik piksel dengan halaman asli", flush=True)
bad_sfx = []
for r in regions:
    if r.label not in tl.PROTECTED_LABELS:
        continue
    x1, y1, x2, y2 = r.bbox
    if not np.array_equal(final[y1:y2, x1:x2], img[y1:y2, x1:x2]):
        d = int((cv2.absdiff(final[y1:y2, x1:x2], img[y1:y2, x1:x2]) > 0).sum())
        bad_sfx.append((r.idx, r.label, d))
check("region terlindungi identik dengan aslinya", not bad_sfx, f"{bad_sfx}")

print("\n4) balon kanan-atas (yang tadinya duplikat)", flush=True)
tr = [r for r in regions if r.bbox[0] >= 800 and r.bbox[1] < 450 and r.bbox[3] < 450]
check("tinggal SATU region di balon itu", len(tr) == 1, f"n={len(tr)}")
for r in tr:
    print(f"    r{r.idx} bbox={r.bbox} bubble={r.bubble_bbox}\n"
          f"        font={r.final_font_size} "
          f"overflow={r.overflowed}\n"
          f"        src={r.src_text[:26]!r}\n"
          f"        en={(r.translation or '')[:70]!r}", flush=True)

for tag, im in (("asli", img), ("hasil", final)):
    crop = im[70:470, 780:1060]
    s = 2
    cv2.imwrite(str(ROOT / f"_h4type_{tag}.png"),
                cv2.resize(crop, (crop.shape[1] * s, crop.shape[0] * s),
                           interpolation=cv2.INTER_NEAREST))
cv2.imwrite(str(ROOT / "_h4type_full.png"), final)
print("\n -> _h4type_asli.png / _h4type_hasil.png / _h4type_full.png", flush=True)
print(f"\n{'SEMUA LOLOS' if fail == 0 else f'{fail} GAGAL'}", flush=True)
sys.exit(1 if fail else 0)
