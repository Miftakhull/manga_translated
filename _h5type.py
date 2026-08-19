"""Buktikan balon r8 hasilnew5 ter-RENDER benar sesudah labelnya jadi DIALOGUE.

Aturan V memindahkan label r8 dari SFX ke DIALOGUE. Itu membuka jalur yang
region ini BELUM PERNAH lewati: dulu PROTECTED, jadi erase dan typeset
melewatinya sama sekali. Dua risiko yang lahir dari situ tidak bisa dijawab
selftest label:

  1. Balonnya HITAM. typeset.py:1486 memilih warna huruf dari
     _bg_luminance(img, region) < 128, diukur pada halaman yang SUDAH
     dihapus. Kalau erase mencerahkan balon hitam itu, hurufnya jadi
     HITAM di atas HITAM -> balon "kosong" yang lolos semua penghitung.
  2. Interiornya cuma 62x133 px (474,1121,536,1254). Balon paling sempit
     di halaman ini. Huruf Inggris bisa luber keluar balon.

Yang diukur (bukan dilihat):
  1. label DIALOGUE + tidak PROTECTED  (aturan V benar-benar menyala di sini)
  2. tinta Jepang r8 BERUBAH sesudah erase   (dulu utuh; ini buktinya terhapus)
  3. r8 tidak muncul di find_residue
  4. _bg_luminance(cleaned, r8) < 128 -> fill PUTIH; dan tinta yang
     TERGAMBAR memang terang  (kalau typeset tidak menggambar -> GAGAL,
     bukan lolos; pelajaran run pertama _h4type.py)
  5. tinta r8 tidak keluar interior balonnya, tidak diklaim region lain
  6. overflow 0
  7. region PROTECTED lain (r15 '♥ー．．．ッ') tetap identik piksel
  8. putaran kedua dengan teks sengaja PANJANG: gagalnya harus BERTERIAK
     (overflowed=True atau font mengecil) dan tetap tidak bocor keluar balon

Probe murni: menulis _h5type_*.png, tidak menyentuh _nbsrc/ maupun notebook.
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
_OLD = REP["regions"]
# bbox sidecar r8 -- jeritan di balon hitam yang user laporkan
R8_SIDECAR = (485, 1140, 525, 1237)
R8_SRC = "ヒ．．．ッ！？"
# Terjemahan yang wajar untuk ヒ．．．ッ！？ (napas tertahan + bertanya).
# Disuntik tangan: sidecar-nya tidak punya, justru KARENA dulu dilabeli SFX.
R8_EN = "EEP!?"
R8_EN_PANJANG = "WHAT IN THE WORLD IS THAT SUPPOSED TO MEAN"
print(f"halaman asli {W}x{H}", flush=True)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / max((ax2 - ax1) * (ay2 - ay1)
                       + (bx2 - bx1) * (by2 - by1) - inter, 1)


def best_match(bbox):
    best, best_iou = None, 0.0
    for o in _OLD:
        v = iou(bbox, tuple(o["bbox"]))
        if v > best_iou:
            best, best_iou = o, v
    return best or {}


typeset.set_page_width(W)
fp = typeset.setup_fonts(verbose=False)
if not typeset.FONT_USED:
    sys.exit("FONT_USED kosong -> render_region diam-diam tidak menggambar")
print(f"font = {typeset.FONT_USED!r}", flush=True)

regions, bubbles = detect.detect(img)
soft = textmask.ctd_soft_mask(img)
for r in regions:
    textmask.build_region_mask(img, r, soft)
textmask.partition_shared_interiors(img, regions)
textmask.disjoin_overlapping_interiors(img, regions)
textmask.protect_bubble_outline(img, regions)
for r in regions:
    o = best_match(r.bbox)
    r.src_text = o.get("src_text") or ""
    r.translation = o.get("translation") or None
tl._fallback_labels(regions)
for r in regions:
    if (r.label not in tl.PROTECTED_LABELS and r.translation is None
            and tl._symbols_only(r.src_text)):
        r.translation = tl._symbols_as_text(r.src_text) or None

r8 = max(regions, key=lambda r: iou(r.bbox, R8_SIDECAR))
print(f"\nr8 -> idx={r8.idx} bbox={r8.bbox} iou={iou(r8.bbox, R8_SIDECAR):.3f}\n"
      f"     bubble={r8.bubble_bbox} src={r8.src_text!r}\n"
      f"     label={r8.label} protected={r8.label in tl.PROTECTED_LABELS}",
      flush=True)

fail = 0


def check(name, ok, detail=""):
    global fail
    print(f"  [{'OK ' if ok else 'GAGAL'}] {name}{('  ' + detail) if detail else ''}",
          flush=True)
    if not ok:
        fail += 1


print("\n1) aturan V menyala di region yang sungguhan", flush=True)
check("region yang cocok memang jeritan yang dilaporkan",
      r8.src_text == R8_SRC, f"src={r8.src_text!r}")
check("label DIALOGUE dan tidak PROTECTED",
      r8.label == "DIALOGUE" and r8.label not in tl.PROTECTED_LABELS,
      f"label={r8.label}")
check("punya bubble_bbox (kalau None, jalur balon-hitam tidak diuji)",
      r8.bubble_bbox is not None, f"{r8.bubble_bbox}")

r8.translation = R8_EN
emask, pmask = textmask.compose_page_mask(img, regions)
sfx_ok = verify.assert_sfx_intact(emask, pmask)
cleaned = erase.erase_page(img, regions, "cpu")
residue = verify.find_residue(cleaned, regions)
final = typeset.render_page(cleaned.copy(), regions)
print(f"\nregion={len(regions)} sfx_utuh={sfx_ok} "
      f"residue={sorted(r.idx for r in residue)}", flush=True)

print("\n2) tinta Jepang r8 benar-benar dihapus", flush=True)
x1, y1, x2, y2 = r8.bbox
d_erase = int((cv2.absdiff(cleaned[y1:y2, x1:x2], img[y1:y2, x1:x2]) > 16).sum())
# Dulu r8 PROTECTED: erase melewatinya, jadi angka ini NOL. > 0 = buktinya
# sekarang benar-benar dihapus, bukan cuma labelnya berubah.
check("erase menyentuh kotak r8 (dulu 0 karena PROTECTED)", d_erase > 200,
      f"piksel_berubah={d_erase}")
check("r8 tidak ditandai residu", r8.idx not in {r.idx for r in residue},
      f"residue={sorted(r.idx for r in residue)}")

print("\n3) warna huruf di balon HITAM", flush=True)
lum = typeset._bg_luminance(cleaned, r8)
check("luminans interior sesudah erase tetap GELAP (balon tidak diputihkan)",
      lum < 128, f"_bg_luminance={lum:.1f} -> fill={'PUTIH' if lum < 128 else 'HITAM'}")

diff = cv2.absdiff(cv2.cvtColor(final, cv2.COLOR_BGR2GRAY),
                   cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY))
ink = (diff > 24).astype(np.uint8)
bx1, by1, bx2, by2 = r8.bubble_bbox
ink8 = ink[by1:by2, bx1:bx2]
n8 = int(ink8.sum())
# Penjaga LEBIH DULU: tanpa ini, "hurufnya terang" lolos HAMPA saat typeset
# tidak menggambar apa pun -- persis cara run pertama _h4type.py menipu saya.
check("typeset benar-benar menggambar di dalam balon r8", n8 > 30,
      f"ink_px={n8} font={r8.final_font_size}")


def ink_stat(r):
    """(lum_latar, median, p90, maks) tinta yang TERGAMBAR di interior r."""
    if r.bubble_bbox is None:
        return None
    a1, b1, a2, b2 = r.bubble_bbox
    sl = ink[b1:b2, a1:a2]
    if not int(sl.sum()):
        return None
    g = cv2.cvtColor(final[b1:b2, a1:a2], cv2.COLOR_BGR2GRAY)[sl > 0]
    return (typeset._bg_luminance(cleaned, r), float(np.median(g)),
            float(np.percentile(g, 90)), float(g.max()))


# Ambang mutlak atas MEDIAN adalah karangan: mask diff>24 ikut menangkap
# piksel anti-alias, dan pada glyph 11 px mayoritas piksel MEMANG piksel tepi,
# jadi median duduk di tengah antara latar dan putih. Yang bisa dipercaya:
# (a) INTI glyph benar-benar putih, (b) arah kontrasnya ke terang, dan
# (c) angkanya sekelas dengan balon GELAP LAIN di halaman yang sama -- balon
# yang sudah terbukti benar di _h4type_full.png. Kalibrasi, bukan tebakan.
peers = [(r.idx, ink_stat(r)) for r in regions
         if r.translation and r is not r8 and ink_stat(r) is not None]
gelap = [(i, s) for i, s in peers if s[0] < 128]
terang = [(i, s) for i, s in peers if s[0] >= 128]
print("   perbandingan tinta per balon (lum_latar, median, p90, maks):", flush=True)
for tag, grp in (("GELAP", gelap), ("terang", terang)):
    for i, s in grp[:6]:
        print(f"     {tag} r{i}: latar={s[0]:.0f} med={s[1]:.0f} "
              f"p90={s[2]:.0f} maks={s[3]:.0f}", flush=True)
s8 = ink_stat(r8)
print(f"     ->    r8: latar={s8[0]:.0f} med={s8[1]:.0f} "
      f"p90={s8[2]:.0f} maks={s8[3]:.0f}", flush=True)
check("INTI glyph r8 benar-benar putih (bukan kelabu)", s8[3] >= 240,
      f"maks={s8[3]:.0f} p90={s8[2]:.0f}")
check("kontras r8 mengarah ke TERANG, jauh dari latarnya",
      s8[1] - s8[0] > 100, f"median-latar={s8[1] - s8[0]:.0f}")
if gelap:
    lo = min(s[1] for _, s in gelap)
    check("median r8 sekelas balon GELAP lain di halaman ini (terkalibrasi)",
          s8[1] >= lo * 0.85,
          f"r8={s8[1]:.0f} vs terendah balon gelap lain={lo:.0f}")
else:
    print("     (tidak ada balon gelap lain berterjemahan -> "
          "pembandingnya cuma penjaga mutlak di atas)", flush=True)
if terang:
    hi = max(s[1] for _, s in terang)
    check("balon TERANG di halaman ini tetap berhuruf gelap (arah tidak tertukar)",
          hi < 128, f"median tertinggi balon terang={hi:.0f}")

print("\n4) huruf tetap di dalam balonnya, tidak bertabrakan", flush=True)
ov = [(r.idx, r.src_text[:8]) for r in regions if r.overflowed]
check("tidak ada overflow di halaman", not ov, f"{ov}")
own = {}
for r in regions:
    if not r.translation:
        continue
    m = np.zeros((H, W), np.uint8)
    a1, b1, a2, b2 = r.bbox
    m[b1:b2, a1:a2] = 1
    own[r.idx] = ink * m
acc = np.zeros((H, W), np.uint16)
for m in own.values():
    acc += m
check("tidak ada piksel tinta diklaim dua region", int((acc > 1).sum()) == 0,
      f"clash_px={int((acc > 1).sum())}")
box, bm = typeset._region_box_mask(r8)
qx1, qy1, qx2, qy2 = box
sub = own[r8.idx][qy1:qy2, qx1:qx2]
outside = int((sub & (bm == 0)).sum()) if sub.shape == bm.shape else -1
check("tinta r8 tidak keluar interior balonnya sendiri", outside == 0,
      f"luar={outside}")

print("\n5) SFX di halaman yang sama tidak ikut bergerak", flush=True)
bad = []
for r in regions:
    if r.label not in tl.PROTECTED_LABELS:
        continue
    a1, b1, a2, b2 = r.bbox
    if not np.array_equal(final[b1:b2, a1:a2], img[b1:b2, a1:a2]):
        bad.append((r.idx, r.label, r.src_text[:8]))
prot = [(r.idx, r.src_text[:9]) for r in regions if r.label in tl.PROTECTED_LABELS]
check("region terlindungi identik piksel dengan aslinya", not bad, f"{bad}")
check("masih ada region terlindungi (kalau 0, uji di atas hampa)", bool(prot),
      f"{prot}")

print("\n6) teks PANJANG: gagalnya berteriak, bukan bocor", flush=True)
r8.translation = R8_EN_PANJANG
r8.overflowed = False
final2 = typeset.render_page(cleaned.copy(), regions)
d2 = cv2.absdiff(cv2.cvtColor(final2, cv2.COLOR_BGR2GRAY),
                 cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY))
ink2 = (d2 > 24).astype(np.uint8)
m8 = np.zeros((H, W), np.uint8)
m8[y1:y2, x1:x2] = 1
sub2 = (ink2 * m8)[qy1:qy2, qx1:qx2]
out2 = int((sub2 & (bm == 0)).sum()) if sub2.shape == bm.shape else -1
print(f"   font={r8.final_font_size} overflow={r8.overflowed} "
      f"ink={int((ink2 * m8).sum())}", flush=True)
check("teks panjang tetap tidak bocor keluar balon r8", out2 == 0, f"luar={out2}")
# typeset TIDAK punya MIN_FONT_SIZE; lantainya emergency_floor(). Memakai nama
# yang tidak ada akan membuat penjaga ini lolos HAMPA (hasattr -> False -> < 99).
_floor = typeset.emergency_floor()
check("teks mustahil ditandai, bukan diam-diam dianggap pas",
      bool(r8.overflowed) or (r8.final_font_size or 0) <= _floor,
      f"overflow={r8.overflowed} font={r8.final_font_size} lantai={_floor}")

s = 3
for tag, im in (("asli", img), ("hasil", final)):
    crop = im[max(0, by1 - 30):by2 + 30, max(0, bx1 - 40):bx2 + 40]
    cv2.imwrite(str(ROOT / f"_h5type_r8_{tag}.png"),
                cv2.resize(crop, (crop.shape[1] * s, crop.shape[0] * s),
                           interpolation=cv2.INTER_NEAREST))
cv2.imwrite(str(ROOT / "_h5type_full.png"), final)
print("\n -> _h5type_r8_asli.png / _h5type_r8_hasil.png / _h5type_full.png",
      flush=True)
print(f"\n{'SEMUA LOLOS' if fail == 0 else f'{fail} GAGAL'}", flush=True)
sys.exit(1 if fail else 0)
