#!/usr/bin/env python3
"""Ubah jawaban LLM mentah -> bentuk yang typeset SUNGGUHAN terima.

Perlu berdiri sendiri karena dua jalur mengambil teks dari tempat berbeda:
pipeline asli memanggil translate._clean_translation() di translate_page tepat
sebelum typeset, sedangkan probe_font.py memasang teks LANGSUNG ke
r.translation. Jadi kalau file wording mentah (masih ada 〈 〉 dan ＼) diumpankan
ke probe_font.py, yang terukur adalah lebar glyph tofu yang pipeline tidak
pernah render — dan angka ukuran fontnya bohong.

  python probe_clean_texts.py probe_llm2_seekai-claude-opus-5.json
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / ".stage"
_MAGIC = re.compile(r"^%%writefile\b.*\n?")
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(STAGE))
STAGE.mkdir(exist_ok=True)
for _s in sorted((ROOT / "_nbsrc").glob("*.py")):
    _b = _MAGIC.sub("", _s.read_text(encoding="utf-8"), count=1)
    _d = STAGE / _s.name
    if not _d.exists() or _d.read_text(encoding="utf-8") != _b:
        _d.write_text(_b, encoding="utf-8")
sys.path.insert(0, str(STAGE))

import translate  # noqa: E402


def main(argv: list[str]) -> int:
    src = ROOT / (argv[0] if argv else "probe_llm2_texts.json")
    dst = ROOT / (argv[1] if len(argv) > 1 else src.stem + "_clean.json")
    raw = json.loads(src.read_text(encoding="utf-8"))
    out = {k: translate._clean_translation(v or "").upper()
           for k, v in sorted(raw.items(), key=lambda kv: int(kv[0]))}
    for k in out:
        if out[k] != raw[k]:
            print(f"  {k:>3} {raw[k]!r} -> {out[k]!r}")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {dst.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
