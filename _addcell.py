#!/usr/bin/env python3
"""Sisipkan Sel 26 (audit kebersihan balon + status terjemah) ke 2.ipynb.

Notebook-nya 596 KB, jadi disunting lewat JSON langsung, bukan dibaca utuh ke
konteks. Idempoten: kalau sel dengan penanda yang sama sudah ada, isinya DIGANTI
bukan ditambah, supaya menjalankan skrip ini dua kali tidak menghasilkan dua sel.
"""

from __future__ import annotations

import json
import pathlib
import sys

MARK = "# Sel 26 (opsional) — AUDIT KEBERSIHAN BALON"


def main() -> int:
    nb_path = pathlib.Path("2.ipynb")
    src_path = pathlib.Path("_dbg/cell26.py")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    body = src_path.read_text(encoding="utf-8").rstrip("\n")
    lines = [f"{ln}\n" for ln in body.split("\n")]
    lines[-1] = lines[-1].rstrip("\n")

    cell = {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}

    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code" and "".join(c["source"]).startswith(MARK):
            nb["cells"][i] = cell
            print(f"sel {i}: DIGANTI (sudah ada sebelumnya)")
            break
    else:
        nb["cells"].append(cell)
        print(f"sel {len(nb['cells']) - 1}: DITAMBAH di akhir")

    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")
    print(f"2.ipynb: {len(nb['cells'])} sel, {nb_path.stat().st_size} B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
