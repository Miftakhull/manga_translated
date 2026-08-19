#!/usr/bin/env python3
"""Daftar model di router — untuk memastikan nama model Opus 5 yang benar.

Ditulis sebagai file, bukan perintah inline, karena pipe inline di sesi ini dua
kali mengembalikan string tak terkait ('clean - nothing to commit') alih-alih
keluaran skrip. Hasilnya diarahkan ke _cmp/models.txt lalu dibaca dari file.

Kunci dibaca dari test.txt ke variabel lokal dan TIDAK PERNAH dicetak, bahkan
sebagian, bahkan panjangnya.
"""

from __future__ import annotations

import json
import pathlib
import re
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent


def main() -> int:
    ln = [l.strip() for l in (ROOT / "test.txt").read_text(encoding="utf-8").splitlines()]
    base, key = ln[2], ln[3]
    req = urllib.request.Request(base.rstrip("/") + "/models")
    req.add_header("Authorization", "Bearer " + key)
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            d = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
        return 1
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"{type(e).__name__}: {str(e)[:200]}")
        return 1

    ids = sorted(str(m.get("id", "")) for m in d.get("data", []))
    print(f"total model: {len(ids)}")
    print("\n== cocok /opus|claude/:")
    for i in ids:
        if re.search(r"opus|claude", i, re.I):
            print(f"   {i}")
    print("\n== prefix gorouter/ (30 pertama):")
    for i in [x for x in ids if x.startswith("gorouter/")][:30]:
        print(f"   {i}")
    print("\n== semua prefix penyedia:")
    pref: dict[str, int] = {}
    for i in ids:
        pref[i.split("/", 1)[0]] = pref.get(i.split("/", 1)[0], 0) + 1
    for k, v in sorted(pref.items(), key=lambda t: -t[1]):
        print(f"   {k:<24} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
