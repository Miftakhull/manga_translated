"""Uji jalur 'balon isinya cuma simbol' di translate.py.

Yang harus benar:
  - _symbols_only() menyala HANYA untuk teks tanpa karakter berkata
  - _symbols_as_text() TIDAK mengosongkan '．．．' (jebakan lstrip di
    _clean_translation)
  - translate_page() menyelesaikannya SEBELUM items dibangun, jadi tidak ada
    permintaan ke penyedia mana pun dan tidak ada catatan error
  - verify.report() tidak lagi menghitungnya untranslated

Probe murni, tanpa jaringan. Penyedianya diganti mata-mata yang MELEDAK kalau
dipanggil untuk region simbol.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

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

import translate as T                      # noqa: E402
from config import Region                  # noqa: E402

fail = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global fail
    print(f"  [{'OK ' if ok else 'GAGAL'}] {name}{('  ' + detail) if detail else ''}",
          flush=True)
    if not ok:
        fail += 1


print("1) _symbols_only()", flush=True)
for s in ("．．．", "。。。", "！？", "！！", "♥", "♡", "♪", "☆", "〜", "～",
          "…", "・・・", "ー．．．", "？", "．．．♥"):
    check(f"simbol {s!r} -> True", T._symbols_only(s) is True)
for s in ("ぁ", "な．．．", "そん．．．", "でも、", "俺も", "A", "1", "ヒ．．．ッ！？",
          "あ．．．ぁ", "ん", "ッ"):
    check(f"berkata {s!r} -> False", T._symbols_only(s) is False)
for s in ("", "   ", "\n"):
    check(f"kosong {s!r} -> False", T._symbols_only(s) is False)

print("\n2) _symbols_as_text() — jebakan lstrip", flush=True)
cases = {
    "．．．": "...",
    "。。。": "...",
    "！？": "!?",
    "！！": "!!",
    "♥": "♥",
    "〜": "~",
    "…": "...",
    "？": "?",
    "．．．♥": "...♥",
}
for src, want in cases.items():
    got = T._symbols_as_text(src)
    check(f"{src!r} -> {want!r}", got == want, f"got={got!r}")
# Bukti bahwa _clean_translation memang MENGOSONGKANNYA, jadi helper ini perlu
check("_clean_translation('．．．') memang kosong -> helper terpisah wajib",
      T._clean_translation("．．．") == "",
      f"got={T._clean_translation('．．．')!r}")

print("\n3) translate_page(): region simbol tidak dikirim ke penyedia", flush=True)
regs = [
    Region(idx=0, bbox=(0, 0, 40, 60), det_class="text_bubble", det_conf=0.9),
    Region(idx=1, bbox=(0, 80, 40, 140), det_class="text_bubble", det_conf=0.9),
]
regs[0].src_text = "．．．"
regs[1].src_text = "こんにちは"

seen: list[list[str]] = []


def _spy(client, texts, *a, **k):
    seen.append(list(texts))
    if any(T._symbols_only(t) for t in texts):
        raise AssertionError(f"region simbol dikirim ke penyedia: {texts!r}")
    return ["HELLO"] * len(texts)


notes: list[tuple] = []
orig_tex, orig_note = T._translate_texts, T.note
T._translate_texts = _spy
T.note = lambda lvl, tag, msg: notes.append((lvl, tag, msg))
try:
    # client bukan RouterClient -> jalur DeepL, yang memanggil _translate_texts
    T.translate_page(object(), "dummy-model", regs)
finally:
    T._translate_texts, T.note = orig_tex, orig_note

check("r0 '．．．' -> '...'", regs[0].translation == "...",
      f"got={regs[0].translation!r}")
check("r1 tetap lewat penyedia", regs[1].translation == "HELLO",
      f"got={regs[1].translation!r}")
check("penyedia dipanggil TANPA region simbol",
      all(not any(T._symbols_only(t) for t in b) for b in seen), f"{seen!r}")
check("tidak ada catatan error", not [n for n in notes if n[0] == "error"],
      f"{notes!r}")

print("\n4) verify.report(): tidak lagi untranslated", flush=True)
import verify as V                          # noqa: E402
untr = [r.idx for r in regs
        if not r.is_protected and r.src_text and not r.translation]
check("daftar untranslated kosong", untr == [], f"{untr}")
check("verify punya report()", hasattr(V, "report"))

print(f"\n{'SEMUA LOLOS' if fail == 0 else f'{fail} GAGAL'}", flush=True)
sys.exit(1 if fail else 0)
