"""Ukur lebar advance simbol emosi: font utama vs font pengganti.

Sekali pakai — jawab satu pertanyaan: apakah mengganti glyph simbol mengubah
lebar baris, yang berarti jalur fit() (memakai _measure/font.getlength) bisa
salah ukur dibanding jalur gambar (memakai _line_width).
"""
from PIL import Image, ImageDraw, ImageFont

SIZE = 20
CHARS = "♥♡♪☆★〜"
FONTS = {
    "anime_ace": "fonts/anime_ace.ttf",
    "symbols2": "fonts/NotoSansSymbols2-Regular.ttf",
    "cjk": "fonts/NotoSansCJKjp-Regular.otf",
}

for name, path in FONTS.items():
    ft = ImageFont.truetype(path, SIZE)
    row = []
    for ch in CHARS:
        try:
            row.append(f"{ch}={ft.getlength(ch):.1f}")
        except Exception as exc:  # noqa: BLE001
            row.append(f"{ch}=ERR")
    print(f"{name:10s} {'  '.join(row)}")

print()
for name, path in (("anime_ace", FONTS["anime_ace"]), ("symbols2", FONTS["symbols2"])):
    ft = ImageFont.truetype(path, SIZE)
    im = Image.new("L", (24, 24), 255)
    ImageDraw.Draw(im).text((2, 1), "♥", font=ft, fill=0)
    px = im.load()
    print(f"--- {name} U+2665 ---")
    for y in range(24):
        print("".join("#" if px[x, y] < 128 else "." for x in range(24)))
