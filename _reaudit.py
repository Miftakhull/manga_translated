#!/usr/bin/env python3
"""Ulangi HANYA audit kebersihan jp_13 dengan aturan baru (temuan dibatasi ke
interior balon), tanpa satu token pun: client=None, jadi jalur terjemah dilewati
sepenuhnya dan yang dihitung tetap 09_cleaned dari kode yang sama.

Ini yang memisahkan "tepi balon kena dilatasi audit" dari "tinta Jepang benar-
benar tertinggal". Kalau setelah pembatasan komponen jadi 0, laporan GAGAL tadi
memang cacat alat ukur, bukan cacat hasil.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_full  # noqa: E402  (menyiapkan .stage + env)

run_full._stage()
sys.path.insert(0, str(run_full.STAGE))

import pipeline, typeset  # noqa: E402
import imgio  # noqa: E402


def main() -> int:
    typeset.setup_fonts(verbose=False)
    img = imgio.load_any(ROOT / "hasilnew/jp_13.JPG")
    res = pipeline.process_page(img, "audit_jp_13", None, "", debug=True)
    pipeline.release_all()

    for thr in (16, 20):
        aud = run_full.audit_clean(res.cleaned, res.regions, typeset, dev_thr=thr)
        print(f"\n=== ambang {thr} ===")
        print(f"  piksel kotor total : {aud['dirty_px_total']}")
        print(f"  komponen total     : {aud['components_total']}")
        print(f"  garis              : {len(aud['lines'])} {aud['lines'][:6]}")
        print(f"  titik              : {len(aud['dots'])} {aud['dots'][:6]}")
        for p in aud["per_region"]:
            print(f"    r{p['idx']:<3} bg={p['bg']:<6} kotor={p['bad_px']:<5} "
                  f"komponen={p['components']:<3} terbesar={p['max_area']}")
    rep = res.report or {}
    print(f"\n  region={rep.get('region_count')} bubble={rep.get('bubble_count')} "
          f"residue={rep.get('residue_count')} idx={rep.get('residue_idx')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
