#!/usr/bin/env python3
"""Uji end-to-end penuh lewat freetokenfaucet, + audit KEBERSIHAN balon.

    python run_full.py hasilnew/jp_6.JPG hasilnew/jp_13.JPG

Bedanya dengan run_page.py: penyedianya LLM (freetokenfaucet), bukan DeepL, dan
yang dinilai di akhir bukan cuma cacat typeset tapi juga satu tuntutan yang
sebelumnya tidak pernah diukur sama sekali — "balon bersih, tidak ada titik
hitam sekecil apa pun dan tidak ada garis tipis apa pun".

Kebersihan diukur pada 09_cleaned (sebelum teks Inggris ditulis), karena setelah
typeset tinta baru tidak bisa dibedakan dari sisa tinta lama. Yang dihitung:

  * piksel gelap yang menyimpang dari background lokal DI DALAM area bekas teks
    (ink_mask, dilatasi sedikit supaya tepi stroke ikut terperiksa),
  * komponen tersambung dari piksel itu — satu titik 2 px pun muncul sebagai
    komponen, jadi "sekecil apa pun" benar-benar terwakili angka,
  * bentuk tiap komponen: komponen panjang-kurus (rasio sisi >= 3 atau panjang
    >= 12 px) dilaporkan terpisah sebagai kandidat GARIS TIPIS, karena itu cacat
    yang berbeda dan lebih mengganggu daripada titik.

Key faucet dibaca lewat tl.get_api_key() — dari Colab Secrets/env/
freetokenfaucet.txt — dan TIDAK PERNAH dicetak, utuh maupun sebagian.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
NBSRC = ROOT / "_nbsrc"
STAGE = ROOT / ".stage"
_MAGIC = re.compile(r"^%%writefile\b.*\n?")

os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(STAGE))


def _stage() -> None:
    STAGE.mkdir(exist_ok=True)
    for src in sorted(NBSRC.glob("*.py")):
        body = _MAGIC.sub("", src.read_text(encoding="utf-8"), count=1)
        dest = STAGE / src.name
        if not dest.exists() or dest.read_text(encoding="utf-8") != body:
            dest.write_text(body, encoding="utf-8")


def _region_ink(r, shape: tuple[int, int], grow: int) -> np.ndarray:
    """ink_mask region di kanvas halaman, didilatasi `grow` px."""
    h, w = shape
    out = np.zeros((h, w), np.uint8)
    if r.ink_mask is None:
        return out
    x1, y1, x2, y2 = r.bbox
    mh, mw = r.ink_mask.shape[:2]
    y2, x2 = min(y2, y1 + mh, h), min(x2, x1 + mw, w)
    if y2 <= y1 or x2 <= x1:
        return out
    out[y1:y2, x1:x2] = r.ink_mask[: y2 - y1, : x2 - x1]
    if grow > 0:
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1,) * 2)
        out = cv2.dilate(out, el, iterations=1)
    return out


def _bubble_map(typeset, r, shape: tuple[int, int]) -> np.ndarray:
    (bx1, by1, _, _), mask = typeset._region_box_mask(r)
    h, w = shape
    out = np.zeros((h, w), np.uint8)
    mh, mw = mask.shape[:2]
    by2, bx2 = min(by1 + mh, h), min(bx1 + mw, w)
    if by2 > by1 and bx2 > bx1:
        out[by1:by2, bx1:bx2] = mask[: by2 - by1, : bx2 - bx1]
    return out


def audit_clean(cleaned: np.ndarray, regions, typeset, dev_thr: int = 16) -> dict:
    """Kebersihan area bekas teks di halaman yang sudah dihapus.

    dev_thr 16 lebih ketat dari SETTINGS.residue_deviation (20) dengan sengaja:
    yang diminta bukan "lolos gate pipeline" tapi "tidak ada titik sekecil apa
    pun", jadi ambangnya diturunkan sampai mendekati derau JPEG (~8-12 pada
    balon putih) tanpa menjadikan derau itu sendiri sebagai temuan.
    """
    h, w = cleaned.shape[:2]
    gray = cv2.cvtColor(cleaned, cv2.COLOR_RGB2GRAY)
    per_region, dots, lines = [], [], []
    for r in regions:
        if r.is_protected or r.ink_mask is None:
            continue
        ink = _region_ink(r, (h, w), grow=2) > 0
        if not ink.any():
            continue
        bub = _bubble_map(typeset, r, (h, w)) > 0
        # Background = interior balon di LUAR bekas teks. Kalau balonnya tidak
        # dikenali, pakai bbox region; median tetap wakil yang jujur karena
        # sebagian besar piksel di sana memang latar.
        ref = (bub & ~ink)
        if ref.sum() < 50:
            x1, y1, x2, y2 = r.bbox
            box = np.zeros((h, w), bool)
            box[y1:y2, x1:x2] = True
            ref = box & ~ink
        bg = float(np.median(gray[ref])) if ref.any() else 255.0
        bad = (np.abs(gray.astype(np.int16) - bg) > dev_thr) & ink
        # Dilatasi 2 px di atas sengaja melebar supaya ekor antialias stroke ikut
        # terperiksa — tapi pada balon kecil ia menembus GARIS TEPI balon dan
        # tinta panel sebelah, dan keduanya memang tidak boleh dihapus. Terukur
        # di jp_13 (17 Agu): 3 "komponen" yang dilaporkan semuanya berpiksel
        # input == cleaned dan berada di luar mask hapus, yakni tepi balon.
        # Karena itu temuan dibatasi ke INTERIOR balon: sisa tinta Jepang selalu
        # di dalam balon, tepi balon selalu di luar interiornya.
        if bub.any():
            bad &= bub
        n, lab, stats, _ = cv2.connectedComponentsWithStats(
            bad.astype(np.uint8), connectivity=8)
        comps = []
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            long_side, short_side = max(cw, ch), max(min(cw, ch), 1)
            kind = ("garis" if (long_side >= 12 or long_side / short_side >= 3.0)
                    else "titik")
            comps.append({"kind": kind, "area": int(area), "bbox": [int(x), int(y),
                          int(x + cw), int(y + ch)], "wh": [int(cw), int(ch)]})
            (lines if kind == "garis" else dots).append(
                {"idx": r.idx, **comps[-1]})
        per_region.append({
            "idx": r.idx, "bg": round(bg, 1), "bad_px": int(bad.sum()),
            "components": len(comps),
            "max_area": max((c["area"] for c in comps), default=0),
        })
    return {
        "dev_threshold": dev_thr,
        "dirty_px_total": sum(p["bad_px"] for p in per_region),
        "components_total": sum(p["components"] for p in per_region),
        "dots": sorted(dots, key=lambda d: -d["area"])[:20],
        "lines": sorted(lines, key=lambda d: -d["area"])[:20],
        "per_region": per_region,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--provider", default="LLM (freetokenfaucet)")
    args = ap.parse_args()

    _stage()
    sys.path.insert(0, str(STAGE))

    import imgio, pipeline, typeset                      # noqa: E401
    import translate as tl
    from config import SETTINGS

    SETTINGS.provider = args.provider
    typeset.setup_fonts(verbose=True)
    try:
        key = tl.get_api_key(provider=args.provider)
    except RuntimeError as exc:
        print(f"[key] {exc}")
        return 2
    print(f"[key] {args.provider}: dimuat (tidak dicetak)")

    paths = [ROOT / p for p in args.images]
    for p in paths:
        print(f"[input] {p.name} {imgio.load_any(p).shape[1::-1]}")

    results, summary = pipeline.process_batch(paths, key, True, None, "English")
    pipeline.release_all()
    print(f"\n[batch] model={summary['model']} provider={summary['provider']} "
          f"residu={summary['residue_total']} overflow={summary['overflow_total']}")

    fails: list[str] = []
    out_json: dict = {"provider": args.provider, "model": summary["model"], "pages": {}}

    for res in results:
        print(f"\n{'=' * 70}\n=== {res.stem} ===")
        rep = res.report or {}
        for k in ("region_count", "bubble_count", "translated_count",
                  "untranslated_count", "untranslated_idx", "residue_count",
                  "overflow_count", "sfx_idx"):
            print(f"  {k}: {rep.get(k)}")
        if rep.get("untranslated_count"):
            fails.append(f"{res.stem}: balon {rep['untranslated_idx']} tidak diterjemahkan")
        if rep.get("overflow_count"):
            fails.append(f"{res.stem}: {rep['overflow_count']} region overflow")

        print(f"\n  {'idx':>3} {'label':<9} {'bub':<4} {'fin':>4} {'ovf':<5} src -> en")
        for r in res.regions:
            print(f"  {r.idx:>3} {str(r.label):<9} {'yes' if r.bubble_bbox else 'NONE':<4} "
                  f"{(r.final_font_size or 0):>4} {str(bool(r.overflowed)):<5} "
                  f"{(r.src_text or '')!r} -> {(r.translation or '')!r}")

        aud = audit_clean(res.cleaned, res.regions, typeset)
        print(f"\n  --- kebersihan balon (09_cleaned, ambang {aud['dev_threshold']}) ---")
        print(f"  piksel kotor total : {aud['dirty_px_total']}")
        print(f"  komponen total     : {aud['components_total']}")
        print(f"  kandidat GARIS     : {len(aud['lines'])} {aud['lines'][:6]}")
        print(f"  kandidat TITIK     : {len(aud['dots'])} {aud['dots'][:6]}")
        for p in aud["per_region"]:
            print(f"    r{p['idx']:<3} bg={p['bg']:<6} kotor={p['bad_px']:<6} "
                  f"komponen={p['components']:<4} terbesar={p['max_area']}")
        if aud["components_total"]:
            fails.append(f"{res.stem}: {aud['components_total']} sisa "
                         f"({len(aud['lines'])} garis, {len(aud['dots'])} titik) "
                         f"di area bekas teks")
        out_json["pages"][res.stem] = {
            "report": {k: v for k, v in rep.items() if k != "regions"},
            "regions": [{"idx": r.idx, "label": r.label, "src": r.src_text,
                         "en": r.translation, "font": r.final_font_size,
                         "overflow": bool(r.overflowed)} for r in res.regions],
            "clean_audit": aud,
            "paths": {k: str(v) for k, v in (res.paths or {}).items()},
        }

    (ROOT / "run_full.json").write_text(
        json.dumps(out_json, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{'=' * 70}\n=== HASIL ===")
    if fails:
        for f in fails:
            print(f"  GAGAL: {f}")
    else:
        print("  semua kriteria terukur LOLOS")
    print("  detail -> run_full.json")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
