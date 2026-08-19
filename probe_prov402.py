#!/usr/bin/env python3
"""Satu panggilan NYATA per penyedia LLM, lewat jalur kode yang sebenarnya.

Bukan HTTP mentah: yang diuji get_api_key() -> make_client() -> translate_page(),
supaya yang terbukti adalah rantai yang dipakai app.py, termasuk pembacaan
gorouter.txt dan header User-Agent per kelas.

Sengaja TIDAK memakai pipeline penuh: detect/OCR di CPU butuh menit-menitan dan
tidak ada hubungannya dengan pertanyaan "penyedia mana yang jawab". Region-nya
dibuat langsung dengan src_text Jepang.

SATU panggilan tiap penyedia. Token faucet terbatas, gorouter pakai kredit.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(ROOT / ".stage"))
sys.path.insert(0, str(ROOT / ".stage"))

JP = ["やめて…！", "んっ…♥", "そんなとこ…", "ドキッ", "部長、待って",
      "誰か来ちゃう", "平気だよ", "ばか…", "好き…♥"]


def main() -> int:
    import translate as tl
    from config import Region

    bad: list[str] = []
    for provider in ("LLM (freetokenfaucet)", "Router LLM (gorouter)"):
        print("=" * 70)
        print(provider)
        regions = []
        for i, t in enumerate(JP):
            r = Region(idx=i, bbox=(0, 0, 160, 120))
            r.src_text = t
            r.label = "SFX" if t == "ドキッ" else "DIALOGUE"
            regions.append(r)

        try:
            key = tl.get_api_key(None, provider)
        except RuntimeError as exc:
            bad.append(f"{provider}: key tidak terbaca ({exc})")
            print("  KEY GAGAL:", exc)
            continue
        client = tl.make_client(key, provider)
        print(f"  client={type(client).__name__} base={client.base} "
              f"model={client.model} headers={sorted(client.headers)} "
              f"key=(panjang {len(key)}, tidak dicetak)")

        t0 = time.monotonic()
        try:
            out = tl.translate_page(client, client.model, regions,
                                    "English", "Uncensored", True)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{provider}: {type(exc).__name__}: {exc}")
            print(f"  GAGAL {type(exc).__name__}: {str(exc)[:300]}")
            continue
        dt = time.monotonic() - t0

        done = [r for r in out if r.translation]
        want = [r for r in out if not r.is_protected and r.src_text]
        print(f"  {dt:.1f}s  diterjemah={len(done)} dari {len(want)} translatable "
              f"({len(out)} region)")
        for r in out:
            mark = "SFX " if r.is_protected else "    "
            print(f"    {mark}{r.src_text:<12} -> {r.translation or '<<KOSONG>>'}")
        if len(done) < len(want):
            bad.append(f"{provider}: {len(want) - len(done)} balon tanpa terjemahan")

    print("=" * 70)
    if bad:
        print("GAGAL:")
        for b in bad:
            print(" -", b)
        return 1
    print("LOLOS: kedua penyedia menjawab lewat jalur kode asli.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
