#!/usr/bin/env python3
"""Uji end-to-end satu halaman + laporan cacat, dibanding CONTOH/2.webp.

    python run_page.py                 # jepang_002.webp, dengan DeepL
    python run_page.py --no-translate  # tanpa DeepL (cek deteksi/mask/erase saja)

Key DeepL dibaca dari `deepl.txt` ke env DEEPL_API_KEY dan TIDAK PERNAH dicetak
— baik utuh maupun sebagian. Itu syarat AGRNTS.md ("jangan simpan rahasia di
kode"): file-nya tetap satu-satunya tempat key berada.

Yang dicetak di akhir adalah enam cacat dari plan.txt sebagai angka, bukan kesan:
overflow, residu, tumpang tindih antar region, tinta keluar interior balon,
sebaran ukuran font, jumlah tanda hubung, dan sisa glyph tofu.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
NBSRC = ROOT / "_nbsrc"
STAGE = ROOT / ".stage"
_MAGIC = re.compile(r"^%%writefile\b.*\n?")

os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(STAGE))


def _stage() -> None:
    """Sama seperti verify_local.py: buang baris magic, baru bisa diimpor."""
    STAGE.mkdir(exist_ok=True)
    for src in sorted(NBSRC.glob("*.py")):
        body = _MAGIC.sub("", src.read_text(encoding="utf-8"), count=1)
        dest = STAGE / src.name
        if not dest.exists() or dest.read_text(encoding="utf-8") != body:
            dest.write_text(body, encoding="utf-8")


def _load_key() -> bool:
    """deepl.txt -> env. True kalau ada. Isinya tidak pernah dicetak."""
    if os.environ.get("DEEPL_API_KEY", "").strip():
        return True
    f = ROOT / "deepl.txt"
    if not f.exists():
        return False
    key = f.read_text(encoding="utf-8").strip()
    if not key:
        return False
    os.environ["DEEPL_API_KEY"] = key
    return True


def _page_map(typeset, r, shape: tuple[int, int]) -> np.ndarray:
    """Interior balon region di kanvas seukuran halaman."""
    (bx1, by1, _, _), mask = typeset._region_box_mask(r)
    h, w = shape
    out = np.zeros((h, w), np.uint8)
    mh, mw = mask.shape[:2]
    by2, bx2 = min(by1 + mh, h), min(bx1 + mw, w)
    if by2 > by1 and bx2 > bx1:
        out[by1:by2, bx1:bx2] = mask[: by2 - by1, : bx2 - bx1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="jepang_002.webp")
    ap.add_argument("--no-translate", action="store_true")
    args = ap.parse_args()

    _stage()
    sys.path.insert(0, str(STAGE))

    import imgio, pipeline, typeset                      # noqa: E401

    typeset.setup_fonts(verbose=True)

    key = None
    if not args.no_translate:
        if not _load_key():
            print("[key] DEEPL_API_KEY / deepl.txt tidak ada -> jalan tanpa terjemahan")
        else:
            print("[key] dimuat dari deepl.txt (tidak dicetak)")
            key = os.environ["DEEPL_API_KEY"]

    path = ROOT / args.image
    print(f"[input] {path.name} {imgio.load_any(path).shape[1::-1]}")

    # process_batch, BUKAN process_page: hanya batch yang memanggil pick_model()
    # dan mengisi argumen `model`. process_page menggatekan terjemahan di
    # `if client is not None and model:` — dipanggil tanpa model, seluruh
    # terjemahan dilewati diam-diam dan tiap angka cacat jadi nol semu.
    results, summary = pipeline.process_batch(
        [path], key, True, None, "English",
    )
    pipeline.release_all()
    res = results[0]
    print(f"[batch] {summary}")

    regions = res.regions
    out, cleaned = res.final, res.cleaned
    h, w = out.shape[:2]

    print("\n=== report.json ===")
    for k, v in (res.report or {}).items():
        print(f"  {k}: {v}")

    print("\n=== per region ===")
    hdr = f"  {'idx':>3} {'cls':<12} {'label':<9} {'bub':<4} {'est':>5} {'fin':>4} {'ovf':>4}  lines"
    print(hdr)
    dial = [r for r in regions if not r.is_protected and r.translation]
    for r in regions:
        print(f"  {r.idx:>3} {str(r.det_class):<12} {str(r.label):<9} "
              f"{'yes' if r.bubble_bbox else 'NONE':<4} "
              f"{(r.est_font_size or 0):>5.1f} {(r.final_font_size or 0):>4} "
              f"{str(bool(r.overflowed)):>4}  {r.lines or ''}")

    # --- enam cacat plan.txt sebagai angka ---
    print("\n=== kriteria lulus ===")
    fails: list[str] = []

    # DULU dan SELALU pertama: tanpa terjemahan, tidak ada tinta yang dirender,
    # jadi overlap/keluar-balon/hyphen semuanya nol SEMU dan laporan ini akan
    # tampak lulus bersih padahal belum menguji apa pun.
    n_tl = sum(1 for r in regions if r.translation)
    print(f"  region terterjemah        : {n_tl} / {len(regions)}")
    if not n_tl:
        fails.append("NOL terjemahan — semua angka di bawah nol semu, bukan lulus")

    ovf = sum(1 for r in regions if r.overflowed)
    print(f"  overflow_count            : {ovf}")
    if ovf:
        fails.append(f"{ovf} region overflow")

    maps = {r.idx: _page_map(typeset, r, (h, w)) for r in dial}
    worst = 0
    for i, a in enumerate(dial):
        for b in dial[i + 1:]:
            ov = int(((maps[a.idx] > 0) & (maps[b.idx] > 0)).sum())
            worst = max(worst, ov)
    print(f"  overlap interior maks (px): {worst}")
    if worst:
        fails.append(f"interior dua region beririsan {worst} px")

    ink = (np.abs(out.astype(np.int16) - cleaned.astype(np.int16)).sum(2) > 120)
    allowed = np.zeros((h, w), bool)
    for m in maps.values():
        allowed |= m > 0
    outside = int((ink & ~allowed).sum())
    print(f"  tinta di luar interior (px): {outside}")
    if outside:
        fails.append(f"{outside} px tinta di luar interior balon")

    fs = [r.final_font_size for r in dial if r.final_font_size]
    spread = (max(fs) / min(fs)) if fs else 0.0
    print(f"  font size                 : {sorted(fs)}  spread={spread:.2f}")
    # Sebaran mentah TIDAK dinilai. Typeset referensi CONTOH/2.webp diukur
    # (probe_refnative.py) dan justru menskalakan teks ke besar balon: sebaran
    # ukurannya sendiri 2.08x, sementara cap_height/sisi-terpendek interior
    # konstan 0.117. Kriteria lama `spread <= 1.35` menghukum halaman yang benar.
    # Yang dinilai: tiap region duduk di plafon proporsionalnya, dan tak ada yang
    # jatuh di bawah min_font_size (jalur darurat fit() -> _MIN_FONT_FLOOR).
    from config import SETTINGS  # noqa: PLC0415
    gap = []
    for r in dial:
        if not r.final_font_size:
            continue
        m = typeset._region_box_mask(r)[1]
        feas = typeset._max_feasible(str(r.translation).upper(), m, typeset.FONT_USED)
        tgt = min(typeset.region_font_cap(m), feas) if feas else 0
        if tgt:
            gap.append((r.idx, tgt - r.final_font_size))
    slack = [f"r{i}:-{g}" for i, g in gap if g > 1]
    print(f"  kurang dari plafon (>1 px): {len(slack)}  {slack}")
    if slack:
        fails.append(f"{len(slack)} region dirender lebih kecil dari yang muat")
    tiny = [r.idx for r in dial
            if r.final_font_size and r.final_font_size < SETTINGS.min_font_size]
    print(f"  di bawah min_font_size    : {tiny}")
    if tiny:
        fails.append(f"region {tiny} jatuh di bawah min_font_size")

    hy = [ln for r in dial for ln in (r.lines or []) if ln.endswith("-")]
    print(f"  baris ber-tanda hubung    : {len(hy)}  {hy}")

    tofu = [t for r in dial for t in [r.translation or ""]
            if any(c in t for c in "「」『』【】〔〕：／…　")]
    print(f"  sisa punctuation CJK      : {len(tofu)}  {tofu}")
    if tofu:
        fails.append("punctuation CJK masih lolos ke terjemahan")

    # SFX: yang dijamin adalah TINTANYA UTUH, bukan adanya region berlabel SFX.
    # Kriteria lama menuntut `sfx_idx` tidak kosong, dan itu salah sasaran —
    # halaman ini memang tidak punya region SFX karena detektor tidak
    # mengeluarkan kotak apa pun di ノノノ maupun 三 (debug/03_boxes.png: 13 kotak,
    # semuanya DIALOGUE). Tidak terdeteksi berarti tidak masuk erase_mask, jadi
    # justru AMAN; probe_sfx.py mengukurnya per kotak dan hasilnya nol piksel
    # tinta hilang di 09_cleaned maupun di hasil akhir.
    #
    # Lebih jauh, `sfx_idx` kosong membuat assert_sfx_intact() (verify.py:79)
    # langsung return True karena protected_mask.max() == 0 — jadi kontrak yang
    # sesungguhnya penting justru TIDAK diuji di jalur itu. Di sini diuji
    # langsung ke piksel: tinta gelap di kotak SFX yang menjadi terang = hilang.
    sfx = [r.idx for r in regions if r.is_protected]
    print(f"  sfx_idx                   : {sfx}")
    # Kotak SFX halaman ini, dibaca manual dari 03_boxes.png (1134x1577).
    # Manual karena mengambilnya dari detektor akan selalu kosong — ketiadaan
    # deteksi itulah yang sedang dijaga.
    _SFX_BOX = {"nonono": (566, 487, 628, 553), "san": (300, 1178, 344, 1232)}
    src_img = imgio.load_any(path)
    if args.image == "jepang_002.webp" and src_img.shape[:2] == (h, w):
        lost = {}
        for nm, (x1, y1, x2, y2) in _SFX_BOX.items():
            a = src_img[y1:y2, x1:x2].astype(np.int16)
            c = out[y1:y2, x1:x2].astype(np.int16)
            dark = a.mean(2) < 128
            lost[nm] = int((dark & (c.mean(2) >= 128)).sum())
        print(f"  tinta SFX hilang (px)     : {lost}")
        if any(lost.values()):
            fails.append(f"tinta SFX hilang: {lost}")

    print("\n=== HASIL ===")
    if fails:
        for f in fails:
            print(f"  GAGAL: {f}")
    else:
        print("  semua kriteria terukur LOLOS (sisanya perlu mata: banding ke CONTOH/2.webp)")
    for name, p in (res.paths or {}).items():
        print(f"  output[{name}]: {p}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
