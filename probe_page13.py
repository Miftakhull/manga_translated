"""Diagnosa balon yang tidak diterjemahkan di hasilnew/13.JPG.

Jalankan detect -> textmask -> OCR -> label pada gambar JEPANG ASLI
(hasilnew/jp_13.JPG), TANPA jaringan dan TANPA menerjemahkan apa pun. Tujuannya
satu: memisahkan tiga kemungkinan penyebab balon 'えっ!?♥' tertinggal berbahasa
Jepang —

  (a) region-nya tidak terdeteksi sama sekali,
  (b) terdeteksi tapi OCR mengembalikan string kosong -> label UNREADABLE ->
      masuk PROTECTED_LABELS -> tidak pernah dikirim ke penyedia,
  (c) terdeteksi & terbaca & berlabel DIALOGUE -> berarti penyebabnya di sisi
      model (id-nya tidak dibalas) atau di _clean_translation.

Nol token faucet: tidak ada satu pun panggilan jaringan di sini.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(ROOT / ".stage"))
sys.path.insert(0, str(ROOT / ".stage"))

import gc  # noqa: E402

import detect  # noqa: E402
import imgio  # noqa: E402
import ocr  # noqa: E402
import textmask  # noqa: E402
import translate as tl  # noqa: E402
from config import SETTINGS  # noqa: E402

SRC = ROOT / "hasilnew" / "jp_13.JPG"

img = imgio.load_any(SRC)
print(f"gambar {SRC.name} {img.shape[1]}x{img.shape[0]}")

regions, bubbles = detect.detect(img)
print(f"region={len(regions)} bubble={len(bubbles)}")

soft = textmask.ctd_soft_mask(img)
for r in regions:
    textmask.build_region_mask(img, r, soft)
textmask.partition_shared_interiors(img, regions)
textmask.disjoin_overlapping_interiors(img, regions)
textmask.protect_bubble_outline(img, regions)
del soft
gc.collect()

ocr.read_all(img, regions)
tl._fallback_labels(regions)

print()
print(f"{'idx':>3} {'bbox':>24} {'kelas':>12} {'ink':>7} {'label':>10}  src_text")
print("-" * 96)
for r in regions:
    gate = "" if r.ink_ratio >= SETTINGS.min_ink_ratio else "  <- DI BAWAH min_ink_ratio"
    print(f"{r.idx:>3} {str(r.bbox):>24} {r.det_class:>12} {r.ink_ratio:>7.4f} "
          f"{r.label:>10}  {r.src_text!r}{gate}")

kirim = [r for r in regions if r.label not in tl.PROTECTED_LABELS and r.src_text]
tertahan = [r for r in regions if r.label in tl.PROTECTED_LABELS]
print()
print(f"akan dikirim ke penyedia: {len(kirim)} region -> idx {[r.idx for r in kirim]}")
print(f"ditahan (SFX/UNREADABLE): {[(r.idx, r.label, r.src_text) for r in tertahan]}")

# Log stdout manga-ocr mengaburkan tabel di terminal, jadi hasil yang dipakai
# untuk menyimpulkan ditulis juga sebagai JSON — itu yang dibaca, bukan stdout.
import json  # noqa: E402
(ROOT / "probe_page13.json").write_text(json.dumps({
    "image": SRC.name,
    "size": [int(img.shape[1]), int(img.shape[0])],
    "bubble_count": len(bubbles),
    "min_ink_ratio": SETTINGS.min_ink_ratio,
    "regions": [{
        "idx": r.idx, "bbox": list(r.bbox), "det_class": r.det_class,
        "det_conf": round(r.det_conf, 3), "in_bubble": r.bubble_bbox is not None,
        "ink_ratio": round(r.ink_ratio, 4), "label": r.label,
        "src_text": r.src_text,
        "sfx_core": tl._sfx_core(r.src_text or ""),
        "protected": r.label in tl.PROTECTED_LABELS,
    } for r in regions],
    "sent_idx": [r.idx for r in kirim],
    "held": [[r.idx, r.label, r.src_text] for r in tertahan],
}, ensure_ascii=False, indent=1), encoding="utf-8")
print("JSON -> probe_page13.json")
