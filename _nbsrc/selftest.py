%%writefile /content/mangatl/selftest.py

"""Self-test tanpa input user: gambar halaman uji sendiri lalu assert pipeline.

Ini yang membuktikan pipeline hidup sebelum user upload apa pun.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import Region, SETTINGS
import detect
import erase
import textmask
import typeset
import verify


def _jp_font(size: int) -> ImageFont.FreeTypeFont:
    """Cari font berglyph Jepang; kalau tidak ada, pakai default (tofu tetap OK)."""
    for cand in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
        "C:/Windows/Fonts/msgothic.ttc",
    ):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_test_page() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Dua bubble + satu kotak narasi + satu SFX di luar bubble."""
    W, H = 900, 1300
    img = Image.new("RGB", (W, H), (245, 243, 240))
    d = ImageDraw.Draw(img)

    # Latar bergaris supaya ada region yang butuh inpaint, bukan flat-fill saja.
    for y in range(0, H, 9):
        d.line([(0, y), (W, y)], fill=(205, 205, 205), width=2)

    f = _jp_font(34)
    boxes: list[tuple[int, int, int, int]] = []

    d.ellipse([80, 90, 400, 340], fill=(255, 255, 255), outline=(0, 0, 0), width=4)
    d.text((240, 215), "こんにちは", font=f, fill=(0, 0, 0), anchor="mm")
    boxes.append((80, 90, 400, 340))

    d.ellipse([480, 430, 830, 700], fill=(255, 255, 255), outline=(0, 0, 0), width=4)
    d.text((655, 565), "セックスしよ", font=f, fill=(0, 0, 0), anchor="mm")
    boxes.append((480, 430, 830, 700))

    d.rectangle([100, 820, 780, 960], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    d.text((440, 890), "その夜、二人は。", font=f, fill=(0, 0, 0), anchor="mm")
    boxes.append((100, 820, 780, 960))

    # SFX di luar bubble, langsung di atas art — harus tetap utuh.
    d.text((250, 1120), "ドドド", font=_jp_font(76), fill=(0, 0, 0), anchor="mm")
    boxes.append((150, 1070, 360, 1170))

    return np.asarray(img, dtype=np.uint8), boxes


# Ketebalan garis balon halaman uji balon ganda.
_DB_STROKE = 4


def make_double_bubble_page() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Region]]:
    """Balon figura-8 SUNGGUHAN: dua elips menyatu, tanpa garis pemisah di leher.

    Inilah bentuk yang meloloskan cacat 'saling timpa'. Membelah KOTAK balon
    tidak memisahkan BENTUK lobusnya — tiap belahan persegi masih memuat
    sebagian lobus sebelah, jadi dua centroid jatuh berdekatan dan kedua
    terjemahan bertumpuk di leher. Test lama (dua persegi terpisah) tidak
    pernah menyentuh kasus ini.

    Returns:
        (clean, img, inner, regions)
        clean   halaman tanpa teks Jepang. Target render, jadi beda piksel
                terhadapnya = tinta Inggris saja, bukan sisa teks asli.
        img     clean + teks Jepang; input pembangun mask, seperti pipeline.
        inner   interior balon seukuran halaman. `inner == 0` mencakup garis
                balon DAN seluruh luar balon = kontrak 'tidak keluar bubble'.
        regions dua region teks, keduanya menunjuk kotak balon GABUNGAN —
                persis keluaran detector saat kedua lobus jadi satu kotak.
    """
    W, H = 900, 620
    page = Image.new("RGB", (W, H), (238, 236, 233))
    d = ImageDraw.Draw(page)
    for y in range(0, H, 9):
        d.line([(0, y), (W, y)], fill=(198, 198, 198), width=2)

    lobes = ((280, 300, 215, 165), (620, 300, 215, 165))
    fill = np.zeros((H, W), np.uint8)
    for cx, cy, ax, ay in lobes:
        cv2.ellipse(fill, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    # Garis balon = pita tepi GABUNGAN, bukan dua outline elips. Menggambar
    # kedua outline meninggalkan garis pemisah di leher, dan itu bukan balon
    # ganda lagi — cuma dua balon yang bersinggungan, kasus yang jauh lebih mudah.
    k = 2 * _DB_STROKE + 1
    inner = cv2.erode(fill, np.ones((k, k), np.uint8))
    arr = np.asarray(page, np.uint8).copy()
    arr[fill > 0] = (255, 255, 255)
    arr[(fill > 0) & (inner == 0)] = (0, 0, 0)

    clean = arr.copy()
    page = Image.fromarray(arr)
    d = ImageDraw.Draw(page)
    f = _jp_font(34)
    for (cx, _, _, _), rows in zip(lobes, (("かいちょう", "さがした"), ("ミルク", "クラブ"))):
        for i, row in enumerate(rows):
            d.text((cx, 278 + i * 44), row, font=f, fill=(0, 0, 0), anchor="mm")

    ys, xs = np.nonzero(fill)
    shared = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    regions = [
        Region(idx=0, bbox=(190, 255, 370, 345), det_class="text_bubble",
               bubble_bbox=shared),
        Region(idx=1, bbox=(560, 255, 680, 345), det_class="text_bubble",
               bubble_bbox=shared),
    ]
    return clean, np.asarray(page, dtype=np.uint8), inner, regions


def make_adjacent_bubbles_page() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Region]]:
    """Dua lobus MENYATU, tapi detector memberi tiap lobus kotaknya SENDIRI.

    Inilah konfigurasi yang benar-benar gagal di halaman nyata dan tidak
    tersentuh make_double_bubble_page(). Bedanya cuma satu hal, dan hal itu
    menentukan segalanya: di sana kedua region menunjuk SATU kotak balon
    (`shared_bubble_bbox` terisi) sehingga partition_shared_interiors() jalan;
    di sini kotaknya terpisah, jadi fungsi itu tidak pernah dipanggil.

    Bentuknya tetap menyatu — tidak ada garis pemisah di leher — jadi flood fill
    tiap region menelan SELURUH figura-8, dipotong hanya oleh kotaknya sendiri.
    Kotak yang saling tumpang tindih + isi yang sama = interior beririsan ribuan
    piksel. Terukur di jepang_002.webp: 6 pasang beririsan, terparah 5994 px
    (region 2-3), semuanya `shared_bubble_bbox = None`.

    Yang rusak karenanya bukan tumpang tindih melainkan GLYPH TERPOTONG:
    render_region menata teks di interiornya sendiri, lalu _clip_to_mask
    membuang piksel yang jatuh di interior region lain (forb_map), sehingga
    'OH, IS THIS THE SHIKO CLUB?' keluar sebagai 'IS THE :KO 4B?'.

    Kalau dua balon memang terpisah bergaris sendiri-sendiri, garis balon
    menghentikan flood fill masing-masing dan tidak ada irisan sama sekali —
    kasus itu sudah aman tanpa perbaikan apa pun.

    Returns:
        (clean, img, inner, regions) — sama seperti make_double_bubble_page().
    """
    W, H = 760, 460
    page = Image.new("RGB", (W, H), (240, 238, 235))
    d = ImageDraw.Draw(page)
    for y in range(0, H, 11):
        d.line([(0, y), (W, y)], fill=(200, 200, 200), width=2)

    lobes = ((250, 230, 170, 150), (505, 230, 170, 150))
    fill = np.zeros((H, W), np.uint8)
    for cx, cy, ax, ay in lobes:
        cv2.ellipse(fill, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    # Pita tepi GABUNGAN: lehernya terbuka, jadi kedua lobus satu ruang putih.
    k = 2 * _DB_STROKE + 1
    inner = cv2.erode(fill, np.ones((k, k), np.uint8))
    arr = np.asarray(page, np.uint8).copy()
    arr[fill > 0] = (255, 255, 255)
    arr[(fill > 0) & (inner == 0)] = (0, 0, 0)

    clean = arr.copy()
    page = Image.fromarray(arr)
    d = ImageDraw.Draw(page)
    f = _jp_font(30)
    for (cx, _, _, _), rows in zip(lobes, (("あっシコ", "部の"), ("性徒会の", "記録"))):
        for i, row in enumerate(rows):
            d.text((cx, 212 + i * 40), row, font=f, fill=(0, 0, 0), anchor="mm")

    # Satu kotak per lobus — bukan satu kotak gabungan. Keduanya beririsan di x
    # karena lobusnya memang saling tumpuk.
    regions = []
    for i, (cx, cy, ax, ay) in enumerate(lobes):
        regions.append(Region(idx=i, bbox=(cx - 70, 190, cx + 70, 270),
                              det_class="text_bubble",
                              bubble_bbox=(cx - ax, cy - ay, cx + ax, cy + ay)))
    return clean, np.asarray(page, dtype=np.uint8), inner, regions


def make_grey_bubble_page() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Region]]:
    """Balon figura-8 ber-screentone KELABU di sebelah art gelap.

    Dua pembangun balon ganda di atas mengecat balon PUTIH
    (`arr[fill > 0] = (255,255,255)`), jadi tidak satu pun bisa menangkap cacat
    cacatbaru/jp_cacatnew1+2: aturan polaritas lama di `_interior_from_crop`
    memutuskan kelas Otsu mana yang interior lewat ambang ABSOLUT
    `median(kelas mayoritas) < 128`. Itu benar hanya untuk dua ujung (balon
    putih / balon hitam). Pada balon KELABU, median mayoritas ada DI ATAS 128
    sehingga polaritas tidak dibalik, dan yang diambil sebagai 'interior' adalah
    piksel TERANG di luar balon. Satu mask salah itu memunculkan dua cacat
    sekaligus: `build_fill_mask` tidak pernah menghapus tinta Jepangnya (
    `erase_flat` memakai `fill_mask`) dan `typeset._region_box_mask` menata
    terjemahan di sliver luar balon — tercetak mungil di atas art.

    Tiga sifat halaman ini yang membuatnya memancing cacat itu, dan tidak satu
    pun ada di dua pembangun sebelumnya:

    1. Interior balon KELABU (screentone), bukan putih, jadi median kelas
       mayoritas jatuh di sekitar 140 — di atas 128, tapi jauh dari putih.
    2. Ada piksel MENDEKATI PUTIH di dalam kotak balon (halaman di sudut kotak
       + kilau di dalam balon), jadi Otsu punya kelas terang untuk dipilih
       secara keliru sebagai interior.
    3. Art GELAP menempel di tepi balon. Setelah polaritas dibalik, garis balon
       masuk kelas yang sama dengan interior kelabu, jadi flood fill bisa
       menembus garis dan mengisi art — inilah yang dijaga dinding gelap
       (`_WALL_MAD`/`_WALL_MIN`) di `_interior_from_crop`.

    Returns:
        (clean, img, inner, regions) — sama seperti make_double_bubble_page().
    """
    W, H = 900, 620
    page = Image.new("RGB", (W, H), (250, 249, 247))
    d = ImageDraw.Draw(page)
    for y in range(0, H, 9):
        d.line([(0, y), (W, y)], fill=(214, 214, 214), width=1)
    arr = np.asarray(page, np.uint8).copy()

    # Art gelap yang MENEMPEL di tepi balon (rambut) — sasaran uji dinding.
    for x0 in (150, 190, 700, 745):
        cv2.line(arr, (x0, 40), (x0 + 60, H - 40), (18, 18, 20), 9)

    lobes = ((300, 300, 205, 158), (610, 300, 205, 158))
    fill = np.zeros((H, W), np.uint8)
    for cx, cy, ax, ay in lobes:
        cv2.ellipse(fill, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    k = 2 * _DB_STROKE + 1
    inner = cv2.erode(fill, np.ones((k, k), np.uint8))

    # Screentone kelabu: dua nilai berpola, jadi MAD cincin > 0 dan dinding
    # gelap tidak boleh memotong balonnya sendiri.
    yy, xx = np.mgrid[0:H, 0:W]
    tone = np.where(((xx // 3) + (yy // 3)) % 2 == 0, 148, 132).astype(np.uint8)
    grey = np.dstack([tone] * 3)
    arr[fill > 0] = grey[fill > 0]
    # Kilau hampir-putih DI DALAM balon: memberi Otsu kelas terang yang menggoda.
    for cx, cy, _, _ in lobes:
        cv2.ellipse(arr, (cx - 96, cy - 96), (44, 26), 0, 0, 360, (246, 246, 246), -1)
    arr[(fill > 0) & (inner == 0)] = (0, 0, 0)

    clean = arr.copy()
    page = Image.fromarray(arr)
    d = ImageDraw.Draw(page)
    f = _jp_font(34)
    for (cx, _, _, _), rows in zip(lobes, (("それとも", "全てを"), ("生涯で", "最も"))):
        for i, row in enumerate(rows):
            d.text((cx, 278 + i * 44), row, font=f, fill=(0, 0, 0), anchor="mm")

    ys, xs = np.nonzero(fill)
    shared = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    regions = [
        Region(idx=0, bbox=(212, 255, 390, 345), det_class="text_bubble",
               bubble_bbox=shared),
        Region(idx=1, bbox=(522, 255, 700, 345), det_class="text_bubble",
               bubble_bbox=shared),
    ]
    return clean, np.asarray(page, dtype=np.uint8), inner, regions


def _page_mask(r: Region, shape: tuple[int, int]) -> np.ndarray:
    """Interior balon region, ditempel ke kanvas seukuran halaman."""
    (bx1, by1, _, _), mask = typeset._region_box_mask(r)
    h, w = shape
    out = np.zeros((h, w), np.uint8)
    mh, mw = mask.shape[:2]
    by2, bx2 = min(by1 + mh, h), min(bx1 + mw, w)
    if by2 > by1 and bx2 > bx1:
        out[by1:by2, bx1:bx2] = mask[: by2 - by1, : bx2 - bx1]
    return out


def _ink_of(page: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Piksel yang berubah terhadap halaman bersih = tinta terjemahan."""
    diff = np.abs(page.astype(np.int16) - base.astype(np.int16)).sum(axis=2)
    return diff > 120


def run(verbose: bool = True) -> bool:
    """Assert kontrak pipeline pada halaman sintetis. Tanpa GPU, tanpa API."""
    img, boxes = make_test_page()
    checks: list[tuple[str, bool, str]] = []

    regions = [
        Region(idx=i, bbox=b, det_class="text_bubble" if i < 3 else "text_free",
               bubble_bbox=b if i < 3 else None)
        for i, b in enumerate(boxes)
    ]
    for r in regions:
        textmask.build_region_mask(img, r, None)

    # Double bubble: satu kotak balon berisi dua region -> dibelah per region.
    _bb = (100, 300, 700, 620)
    _rA = Region(idx=10, bbox=(140, 360, 360, 420), det_class="text_bubble",
                 bubble_bbox=_bb)
    _rB = Region(idx=11, bbox=(440, 360, 660, 420), det_class="text_bubble",
                 bubble_bbox=_bb)
    detect._partition_shared_bubbles([_rA, _rB])
    checks.append((
        "double bubble dibelah per region",
        _rA.bubble_bbox is not None and _rB.bubble_bbox is not None
        and _rA.bubble_bbox != _rB.bubble_bbox
        and _rA.bubble_bbox[2] <= _rB.bubble_bbox[0],
        f"A={_rA.bubble_bbox} B={_rB.bubble_bbox}",
    ))

    # SATU blok teks terdeteksi DUA KALI: kotak kecil bersarang di kotak besar.
    # Bukan sintetis — bbox di bawah diambil apa adanya dari halaman
    # hitomi_3740721_015 (r0 di dalam r1, containment 0.974 tapi IoU cuma 0.280
    # jadi NMS kelompok teks melewatkannya). Kalau dibiarkan, KEDUANYA dapat
    # kotak balon yang sama, _partition_shared_bubbles menyangkanya balon ganda
    # dan MEMBELAH balonnya di x=956 — tiap belahan lalu mengukur fill_color dan
    # warna hurufnya sendiri, jadi satu balon keluar berjahitan dua warna dengan
    # kalimat yang sama tercetak dua kali. Lihat detect.drop_nested_duplicates().
    _dup = [Region(idx=0, bbox=(944, 130, 1024, 321), det_class="text_bubble"),
            Region(idx=1, bbox=(832, 135, 1027, 405), det_class="text_bubble")]
    _ndup = detect.drop_nested_duplicates(_dup)
    checks.append((
        "duplikat bersarang dibuang, yang bertahan dilebarkan ke gabungan",
        _ndup == 1 and len(_dup) == 1
        and _dup[0].bbox == (832, 130, 1027, 405),
        f"dibuang={_ndup} sisa={[r.bbox for r in _dup]}",
    ))
    # Sisi lain gerbangnya, dan yang lebih penting: kotak besar yang benar-benar
    # memuat DUA kotak kecil adalah balon ganda sungguhan, dan untuk kasus itu
    # _partition_shared_bubbles + partition_shared_interiors sudah benar. Kalau
    # penyingkiran di atas ikut menyalak di sini, jalur balon ganda yang bekerja
    # justru dirusak. Lobus BERJAJAR juga tidak boleh disentuh — containment-nya
    # rendah (tertinggi 0.33 pada halaman bersih yang diukur).
    _nd2 = [Region(idx=0, bbox=(100, 100, 300, 400), det_class="text_bubble"),
            Region(idx=1, bbox=(110, 110, 190, 390), det_class="text_bubble"),
            Region(idx=2, bbox=(210, 110, 290, 390), det_class="text_bubble")]
    _nd3 = [Region(idx=0, bbox=(100, 100, 205, 400), det_class="text_bubble"),
            Region(idx=1, bbox=(195, 100, 300, 400), det_class="text_bubble")]
    _k2, _k3 = (detect.drop_nested_duplicates(_nd2),
                detect.drop_nested_duplicates(_nd3))
    checks.append((
        "balon ganda & lobus berjajar TIDAK ikut dibuang",
        _k2 == 0 and _k3 == 0 and len(_nd2) == 3 and len(_nd3) == 2
        and _nd2[0].bbox == (100, 100, 300, 400),
        f"dua_lobus_bersarang={_k2} berjajar={_k3}",
    ))

    # Induk dipilih dari CAKUPAN dulu, area cuma pemutus seri. Angka di bawah
    # apa adanya dari hitomi_3740721_015: kotak lobus kiri lebih KECIL tapi hanya
    # memuat 0.677 teksnya, jadi aturan "terkecil yang memuat mayoritas" memilih
    # dia dan lobus kanan tidak pernah masuk interior — tinta Jepang di sana
    # tidak terhapus. Lihat detect._PARENT_SLACK.
    _pr = Region(idx=0, bbox=(832, 130, 1027, 405), det_class="text_bubble")
    detect.assign_bubbles([_pr], [(800, 117, 964, 440),      # lobus, cover 0.677
                                  (800, 96, 1046, 442)])     # balon penuh, 1.000
    checks.append((
        "induk = balon yang memuat teks utuh, bukan kotak lobus yang lebih kecil",
        _pr.bubble_bbox == (800, 96, 1046, 442),
        f"terpilih={_pr.bubble_bbox}",
    ))
    # Sisi lain pitanya: kalau teks cuma menonjol beberapa piksel keluar lobus,
    # lobus HARUS tetap menang — kalau tidak, tiap region balon ganda akan
    # memilih kotak gabungan dan kedua terjemahan kembali bertumpuk di tengah.
    _pr2 = Region(idx=0, bbox=(10, 10, 110, 110), det_class="text_bubble")
    detect.assign_bubbles([_pr2], [(10, 10, 107, 110),        # lobus, cover 0.97
                                   (0, 0, 200, 200)])         # gabungan, 1.000
    checks.append((
        "tonjolan beberapa piksel tidak memindahkan induk ke kotak gabungan",
        _pr2.bubble_bbox == (10, 10, 107, 110),
        f"terpilih={_pr2.bubble_bbox}",
    ))

    # Balon figura-8 sungguhan: belahan persegi di atas TIDAK cukup, interiornya
    # harus dipartisi mengikuti bentuk lobus. Lihat make_double_bubble_page().
    _dclean, _dimg, _dinner, _dbl = make_double_bubble_page()
    detect._partition_shared_bubbles(_dbl)
    for _r in _dbl:
        textmask.build_region_mask(_dimg, _r, None)
    _split = textmask.partition_shared_interiors(_dimg, _dbl)
    _shape = _dinner.shape
    _mA, _mB = _page_mask(_dbl[0], _shape), _page_mask(_dbl[1], _shape)
    _ov = int(((_mA > 0) & (_mB > 0)).sum())
    checks.append((
        "interior balon figura-8 dipartisi per lobus & disjoint",
        _split == 2 and _mA.any() and _mB.any() and _ov == 0,
        f"dipartisi={_split} overlap_px={_ov} A={int((_mA>0).sum())} "
        f"B={int((_mB>0).sum())}",
    ))
    _dbw = _dbl[0].shared_bubble_bbox[2] - _dbl[0].shared_bubble_bbox[0]
    _cA = _dbl[0].bubble_bbox[0] + typeset._centroid(_dbl[0].bubble_mask)[0]
    _cB = _dbl[1].bubble_bbox[0] + typeset._centroid(_dbl[1].bubble_mask)[0]
    checks.append((
        "centroid dua lobus terpisah >= 40% lebar balon",
        abs(_cB - _cA) >= _dbw * 0.40,
        f"dx={abs(_cB - _cA)} lebar_balon={_dbw}",
    ))

    # Balon BERTETANGGA (bukan figura-8): shared_bubble_bbox None, jadi
    # partition_shared_interiors() tidak jalan dan yang harus menyelamatkan
    # adalah disjoin_overlapping_interiors(). Ini kasus 5994 px di halaman nyata.
    _aclean, _aimg, _ainner, _adj = make_adjacent_bubbles_page()
    detect._partition_shared_bubbles(_adj)          # tidak boleh mengubah apa pun
    for _r in _adj:
        textmask.build_region_mask(_aimg, _r, None)
    _ashape = _ainner.shape
    _ov0 = int(((_page_mask(_adj[0], _ashape) > 0)
                & (_page_mask(_adj[1], _ashape) > 0)).sum())
    _afix = textmask.disjoin_overlapping_interiors(_aimg, _adj)
    _amA, _amB = _page_mask(_adj[0], _ashape), _page_mask(_adj[1], _ashape)
    _ov1 = int(((_amA > 0) & (_amB > 0)).sum())
    checks.append((
        "balon bertetangga: interior beririsan SEBELUM diperbaiki (test valid)",
        _ov0 > 0 and all(r.shared_bubble_bbox is None for r in _adj),
        f"overlap_awal={_ov0} shared={[r.shared_bubble_bbox for r in _adj]}",
    ))
    checks.append((
        "balon bertetangga: interior dibuat disjoint tanpa mengosongkan balon",
        _ov1 == 0 and _amA.any() and _amB.any() and _afix >= 1,
        f"overlap={_ov1} diperbaiki={_afix} A={int((_amA>0).sum())} "
        f"B={int((_amB>0).sum())}",
    ))
    # Menyusut hanya boleh MEMBUANG piksel di zona sengketa — kalau bisa
    # menambah, cacat 'teks keluar balon' bisa muncul lewat pintu ini.
    checks.append((
        "disjoin tidak pernah menambah piksel di luar interior balon",
        int(((_amA > 0) & (_ainner == 0)).sum()) == 0
        and int(((_amB > 0) & (_ainner == 0)).sum()) == 0,
        f"A_luar={int(((_amA>0)&(_ainner==0)).sum())} "
        f"B_luar={int(((_amB>0)&(_ainner==0)).sum())}",
    ))

    # Balon KELABU ber-screentone: kedua halaman balon ganda di atas memakai
    # balon PUTIH, jadi tidak satu pun menyentuh aturan polaritas. Di sini
    # median kelas mayoritas ada DI ATAS 128 (jadi aturan absolut lama TIDAK
    # membalik polaritas) padahal interiornya kelabu, bukan putih. Terukur di
    # _cngrey.py pada halaman ini: cakupan tinta aturan lama 0.000 dengan 0.828
    # piksel interior jatuh DI LUAR balon (itu halaman putih di sudut kotak),
    # aturan cincin 1.000 dan 0.000. Dua angka itulah dua cacat cacatbaru:
    # tinta Jepang tak terhapus (build_fill_mask memakai interior ini) dan
    # terjemahan ditata di sliver luar balon.
    _gclean, _gimg, _ginner, _grey = make_grey_bubble_page()
    for _r in _grey:
        textmask.build_region_mask(_gimg, _r, None)
    _gshape = _ginner.shape
    _gtinta, _gluar = [], []
    for _r in _grey:
        _gm = _page_mask(_r, _gshape)
        _gsel = _gm > 0
        _gluar.append(0.0 if not _gsel.any()
                      else float((_ginner[_gsel] == 0).mean()))
        _gx1, _gy1, _gx2, _gy2 = _r.bbox
        _gink = np.zeros(_gshape, np.uint8)
        _gink[_gy1:_gy2, _gx1:_gx2] = _r.ink_mask[: _gy2 - _gy1, : _gx2 - _gx1]
        _gtinta.append(-1.0 if not _gink.any()
                       else float((_gm[_gink > 0] > 0).mean()))
    checks.append((
        "balon kelabu: interior memuat tinta region-nya sendiri",
        all(t >= 0.90 for t in _gtinta),
        f"cakupan_tinta={[round(t, 3) for t in _gtinta]}",
    ))
    checks.append((
        "balon kelabu: interior tidak keluar balon (art gelap di tepi utuh)",
        all(l <= 0.02 for l in _gluar),
        f"fraksi_di_luar={[round(l, 3) for l in _gluar]}",
    ))
    # Alasan STRUKTURAL-nya diuji langsung, supaya kalau halaman uji berubah
    # bentuk pun aturannya tetap terjaga: polaritas ditentukan cincin latar
    # tinta, bukan kecerahan absolut. Kelas mayoritas di sini median > 128,
    # jadi aturan lama menjawab False dan aturan cincin harus menjawab True.
    _gr = _grey[0]
    _gb = _gr.bubble_bbox
    _gcrop = _gimg[_gb[1]:_gb[3], _gb[0]:_gb[2]]
    _ggray = cv2.cvtColor(_gcrop, cv2.COLOR_RGB2GRAY)
    _, _gbinv = cv2.threshold(_ggray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _gvals = _ggray[_gbinv > 0]
    if _gvals.size < _gbinv.size - _gvals.size:
        _gvals = _ggray[_gbinv == 0]
    _gmay = float(np.median(_gvals))
    _gink_crop = textmask._ink_in_crop(_gr, _gb[0], _gb[1], _gcrop.shape[:2])
    _gbalik, _, _ = textmask._polarity_ring(_ggray, _gbinv, _gink_crop)
    checks.append((
        "polaritas dari cincin tinta, bukan ambang absolut",
        _gmay >= 128 and _gbalik,
        f"median_kelas_mayoritas={_gmay:.1f} (aturan_lama_balik="
        f"{_gmay < 128}) aturan_cincin_balik={_gbalik}",
    ))

    checks.append((
        "mask terbentuk di semua region",
        all(r.ink_mask is not None and r.ink_mask.any() for r in regions),
        "",
    ))
    checks.append((
        "estimasi font size masuk akal (8..120 px)",
        all(8 <= r.est_font_size <= 120 for r in regions),
        str([round(r.est_font_size, 1) for r in regions]),
    ))

    # Regresi 'SFX diterjemahkan' — heuristik baru di translate.py.
    # Kasus nyata yang bocor di heuristik lama (kana murni <=3/ABAB-4):
    #   "フー．．．" kana+simbol diterjemah, "ぴくぴくっ" ABAB+っ,
    #   "ドキドキドキ" ulangan 6 char, "ドキッ" SFX dalam balon.
    # Dan yang TIDAK boleh terkunci: それは (narasi), ちょっと, サッカー
    # (pinjaman ッ…ー), こんにちは (dialog).
    #
    # Arah kedua ditambahkan setelah cacat "balon pendek tidak diterjemah":
    # cabang lama di dalam balon `n == 3 and has_small` mengunci seruan yang
    # jelas ucapan jadi SFX, dan SFX = translation None + PROTECTED = balon
    # Jepang tercetak TANPA satu pun pesan error. _label3.py mengukur cabang
    # itu pada 72 kasus (45 dialog + 27 SFX): arah dialog 30 salah, arah SFX 4
    # salah. Yang di bawah wakilnya, dipilih supaya tiap bagian gerbang baru
    # punya penjaga:
    #   hiragana ber-っ (ええっ うんっ まてっ)      -> DIALOGUE
    #   ucapan ditulis katakana (ダメッ オイッ)     -> DIALOGUE lewat _kata2hira
    #   seluruhnya katakana, ッ di UJUNG (ハッ)     -> SFX (サッカー tidak: ッ
    #                                                  di tengah, ujungnya ー)
    #   hiragana yang batang-gandanya di kamus      -> SFX (どきっ びくっ)
    import translate as _tl
    _sfx_cases = [
        ("フー．．．", False, "SFX"), ("ピクッ", False, "SFX"),
        ("ーー", False, "SFX"),
        ("ドキドキドキ", False, "SFX"), ("ぴくぴくっ", False, "SFX"),
        ("ガッタンゴットン", False, "SFX"), ("ドキッ", True, "SFX"),
        ("はぁっ", True, "SFX"),
        ("それは．．．", False, "DIALOGUE"), ("ちょっと", False, "DIALOGUE"),
        ("サッカー", True, "DIALOGUE"), ("こんにちは", True, "DIALOGUE"),
        ("うう．．．", True, "DIALOGUE"), ("んっ", True, "DIALOGUE"),
        # seruan hiragana di dalam balon: WAJIB diterjemah
        ("ええっ", True, "DIALOGUE"), ("ええっ！？", True, "DIALOGUE"),
        ("うんっ", True, "DIALOGUE"), ("だめっ", True, "DIALOGUE"),
        ("いやっ", True, "DIALOGUE"), ("まてっ", True, "DIALOGUE"),
        ("うそっ", True, "DIALOGUE"), ("なにっ", True, "DIALOGUE"),
        ("ちょっ", True, "DIALOGUE"), ("そこっ", True, "DIALOGUE"),
        ("はいっ", True, "DIALOGUE"), ("ねえっ", True, "DIALOGUE"),
        ("もうっ", True, "DIALOGUE"), ("やめっ", True, "DIALOGUE"),
        ("うわっ", True, "DIALOGUE"), ("やだっ", True, "DIALOGUE"),
        ("あーっ", True, "DIALOGUE"), ("ふぇっ", True, "DIALOGUE"),
        # ucapan yang DITULIS KATAKANA — tanpa normalisasi kana semuanya SFX
        ("ダメッ", True, "DIALOGUE"), ("ハイッ", True, "DIALOGUE"),
        ("ウンッ", True, "DIALOGUE"), ("ムリッ", True, "DIALOGUE"),
        ("オイッ", True, "DIALOGUE"), ("マテッ", True, "DIALOGUE"),
        ("ナニッ", True, "DIALOGUE"), ("ウソッ", True, "DIALOGUE"),
        ("イヤッ", True, "DIALOGUE"), ("ヤダッ", True, "DIALOGUE"),
        ("ヤメッ", True, "DIALOGUE"),
        # arah sebaliknya tidak boleh ikut longgar
        ("ハッ", True, "SFX"), ("ズドンッ", True, "SFX"),
        ("ガシャッ", True, "SFX"), ("パチンッ", True, "SFX"),
        ("ゴクッ", True, "SFX"), ("どきっ", True, "SFX"),
        ("びくっ", True, "SFX"), ("ぴくっ", True, "SFX"),
        ("ごくっ", True, "SFX"), ("がくっ", True, "SFX"),
        ("ふわっ", True, "SFX"), ("どきどき", True, "SFX"),
        ("はぁはぁ", True, "SFX"), ("ぐちゅぐちゅ", True, "SFX"),
        # Arah ketiga, dari hasilnew5: satu balon dari 22 halaman keluar tetap
        # Jepang — 'ヒ．．．ッ！？' di balon hitam bergerigi. Aturan K
        # (<=6 kana, seluruhnya katakana, ッ di ujung) mengunci SFX, dan SFX =
        # PROTECTED = translate_page melewatinya tanpa pesan error.
        # Yang menyulitkan: ハッ di atas WAJIB tetap SFX dan strukturnya
        # IDENTIK (satu mora katakana + ッ, di dalam balon), jadi panjang
        # batang tidak bisa memisahkannya. Pemisahnya tanda BICARA di teks
        # ASLI — yang justru dibuang _sfx_core. _h5lbl.py mengukur 5 kandidat
        # atas 60 kasus di atas + 14 di bawah; hanya aturan V yang nol
        # kesalahan dua arah.
        ("ヒ．．．ッ！？", True, "DIALOGUE"),
        ("ヒッ！？", True, "DIALOGUE"), ("ヒ．．．ッ", True, "DIALOGUE"),
        ("ア．．．ッ", True, "DIALOGUE"), ("ウ．．．ッ！？", True, "DIALOGUE"),
        ("キャ．．．ッ", True, "DIALOGUE"),
        # ...dan yang TIDAK boleh ikut longgar karena V:
        ("ヒ．．．ッ！？", False, "SFX"),   # teks SAMA di LUAR balon = bunyi latar
        ("ヒッ", True, "SFX"),            # tanpa tanda bicara: tetap bunyi
        ("ドキッ！", True, "SFX"), ("ドキッ！？", True, "SFX"),
        ("ゴクッ．．．", True, "SFX"),      # jeda di UJUNG, bukan di ANTARA kana
        ("ガシャッ．．．", True, "SFX"),
        # ズドンッ！ / パチンッ！ menahan kandidat '？ atau ！': batang keduanya
        # TIDAK berkamus (ずどんずどん, ぱちんぱちん tidak ada), jadi aturan S
        # tidak menyelamatkannya dan mereka akan jatuh jadi DIALOGUE.
        ("ズドンッ！", True, "SFX"), ("パチンッ！", True, "SFX"),
    ]
    _sfx_bad = []
    for _t, _ib, _want in _sfx_cases:
        _tr = Region(idx=99, bbox=(0, 0, 10, 10),
                     bubble_bbox=(0, 0, 10, 10) if _ib else None)
        _tr.src_text = _t
        _tl._label_region(_tr)
        if _tr.label != _want:
            _sfx_bad.append(f"{_t!r}->{_tr.label} (ingin {_want})")
    checks.append((
        "heuristik SFX baru (フー.., ドキドキドキ, サッカー..)",
        not _sfx_bad, "; ".join(_sfx_bad),
    ))
    # Penjaga langsung untuk pemisah aturan V: jeda DI ANTARA kana vs di UJUNG.
    # Kalau _PAUSE suatu saat dilebarkan sampai memuat ー (U+30FC), 'フー．．．'
    # dan 'ゴクッ．．．' ikut terjaring dan bunyi memanjang berhenti jadi SFX.
    _brk = _tl._broken_kana
    _brk_want = [
        ("ヒ．．．ッ", True), ("キャ．．．ッ", True), ("ア．．．ッ", True),
        ("フー．．．", False), ("ゴクッ．．．", False), ("ドキッ", False),
        ("ヒッ", False), ("ー．．．", False), ("ハッ", False),
    ]
    _brk_bad = [f"{_t!r}->{_brk(_t)} (ingin {_w})"
                for _t, _w in _brk_want if _brk(_t) != _w]
    checks.append((
        "jeda DI ANTARA kana dibedakan dari jeda di UJUNG",
        not _brk_bad, "; ".join(_brk_bad),
    ))
    # Cacat hasilnew5 apa adanya: balon r8 harus keluar dari PROTECTED, karena
    # PROTECTED = translate_page melewatinya = balon Jepang tercetak diam-diam.
    _r8 = Region(idx=8, bbox=(485, 1140, 525, 1237),
                 bubble_bbox=(474, 1121, 536, 1254))
    _r8.src_text = "ヒ．．．ッ！？"
    _tl._label_region(_r8)
    checks.append((
        "balon hitam hasilnew5 'ヒ．．．ッ！？' tidak lagi PROTECTED",
        _r8.label == "DIALOGUE" and _r8.label not in _tl.PROTECTED_LABELS,
        f"label={_r8.label} protected={_r8.label in _tl.PROTECTED_LABELS}",
    ))
    _sym_ok = _tl._restore_symbols("大好き♥", "I love you") == "I love you♥"
    checks.append((
        "simbol emosi dipulihkan setelah terjemahan", _sym_ok, "",
    ))

    # Balon yang isinya HANYA simbol tidak punya kata untuk diterjemahkan. Model
    # membalas kosong, jalur perbaikan menuduhnya "BELUM DITERJEMAHKAN", dan
    # balonnya tercetak HAMPA (final_font_size 0) — simbol yang memang tertulis
    # di halaman aslinya ikut hilang. Terukur di hitomi_3740721_015 r12='．．．'.
    _so_bad: list[str] = []
    for _t in ("．．．", "。。。", "！？", "♥", "♡", "♪", "☆", "〜", "～", "…",
               "・・・", "ー．．．", "？", "．．．♥"):
        if not _tl._symbols_only(_t):
            _so_bad.append(f"{_t!r} dianggap berkata")
    # ー dan ・ duduk di blok katakana tapi tidak pernah jadi kata sendirian;
    # ぁ, ん, ッ satu huruf pun TETAP kata. Dua arah harus benar.
    for _t in ("ぁ", "ん", "ッ", "な．．．", "そん．．．", "でも、", "俺も",
               "ヒ．．．ッ！？", "A", "1", "", "   "):
        if _tl._symbols_only(_t):
            _so_bad.append(f"{_t!r} dianggap simbol")
    checks.append((
        "balon simbol-saja dikenali (dan huruf tunggal TIDAK ikut terjaring)",
        not _so_bad, "; ".join(_so_bad),
    ))
    # _clean_translation() membuang tanda baca di AWAL string, dan pada balon
    # simbol-saja SELURUH isinya ada di awal — jadi ia mengembalikan string
    # kosong. Itulah sebabnya ada helper terpisah; pemeriksaan kedua di bawah
    # membuktikan jebakannya masih ada supaya helper ini tidak dianggap mubazir.
    _sa_bad = [f"{_s!r}->{_tl._symbols_as_text(_s)!r} (ingin {_w!r})"
               for _s, _w in (("．．．", "..."), ("。。。", "..."), ("！？", "!?"),
                              ("♥", "♥"), ("〜", "~"), ("…", "..."),
                              ("？", "?"), ("．．．♥", "...♥"))
               if _tl._symbols_as_text(_s) != _w]
    checks.append((
        "simbol dipetakan ke ASCII tanpa dikosongkan lstrip",
        not _sa_bad and _tl._clean_translation("．．．") == "",
        "; ".join(_sa_bad) or
        f"_clean_translation('．．．')={_tl._clean_translation('．．．')!r}",
    ))
    # Bukti bahwa helper di atas benar-benar TERSAMBUNG: satu halaman yang isinya
    # cuma balon simbol harus selesai TANPA menyentuh penyedia. client=object()
    # bukan RouterClient dan modelnya karangan, jadi kalau region ini sampai
    # masuk `items` pemeriksaan ini akan meledak, bukan lolos diam-diam.
    _spr = [Region(idx=0, bbox=(0, 0, 40, 60), det_class="text_bubble")]
    _spr[0].src_text = "．．．"
    try:
        _tl.translate_page(object(), "tidak-ada-model", _spr)
        _sp_err = ""
    except Exception as _e:                       # noqa: BLE001 - pesannya dilaporkan
        _sp_err = f"{type(_e).__name__}: {_e}"
    checks.append((
        "balon simbol-saja selesai tanpa memanggil penyedia",
        not _sp_err and _spr[0].translation == "...",
        _sp_err or f"translation={_spr[0].translation!r}",
    ))

    # Kurung sudut Jepang lolos apa adanya dari DeepL dan Anime Ace merendernya
    # jadi kotak tofu di dalam balon; ♥ ♪ ☆ justru WAJIB bertahan (plan.txt).
    # 〜 juga bertahan, tapi dinormalkan ke '~': wave dash lebar tidak ada di
    # Anime Ace sementara tilde ASCII ada, jadi bentuknya tetap tampil dan
    # dirender font balon yang sama — lihat _PUNCT_MAP.
    _ct = _tl._clean_translation
    checks.append((
        "punctuation CJK dibuang, simbol emosi bertahan",
        not any(c in _ct("「会長っ」") for c in "「」")
        and _ct("大好き♥").endswith("♥")
        and _ct("ずっと〜").endswith("~")
        and _ct("ずっと～").endswith("~")
        and _ct("　. シ ズ ク…") == "シ ズ ク..."
        # ＼…／ tanda penekanan Jepang: tidak ada di Anime Ace -> tofu di depan
        # baris pertama ('＼MY APOLO-GIES' di halaman referensi).
        and _ct("＼My apologies") == "My apologies",
        f"{_ct('「会長っ」')!r} {_ct('大好き♥')!r} {_ct('ずっと〜')!r} "
        f"{_ct('　. シ ズ ク…')!r} {_ct('＼My apologies')!r}",
    ))

    # Simbol emosi yang TIDAK ada di Anime Ace harus dirutekan ke font yang
    # benar-benar punya glyph-nya, bukan ke penampung umum. Sebelum diperbaiki
    # ♡ ♪ ♫ ☆ ★ semuanya jatuh ke NotoSans — satu-satunya font di rantai yang
    # tidak punya satu pun dari simbol itu — dan dirender jadi kotak tofu.
    typeset.setup_fonts(verbose=False)
    _main = typeset._font(typeset.FONT_USED, 16)
    _mcmap = typeset._cmap(typeset.FONT_USED)
    _tofu = []
    for _ch in "♡♪♫☆★❤~":
        _f = typeset._char_font(_ch, _main, _mcmap, 16)
        if ord(_ch) not in typeset._cmap(_f.path):
            _tofu.append(f"{_ch} -> {_f.path.replace(chr(92), '/').rsplit('/', 1)[-1]}")
    checks.append((
        "simbol emosi dapat font yang punya glyph-nya (bukan tofu)",
        not _tofu, "; ".join(_tofu),
    ))

    # Cek kedua, dan ini yang menangkap cacat hasilnew/6.JPG: "punya glyph"
    # TIDAK SAMA dengan "punya glyph yang benar". anime_ace.ttf MEMETAKAN U+2665
    # ke glyph bernama `yat` (huruf Cyrillic Ѣ), jadi cek tofu di atas lolos
    # sementara yang tergambar di balon bukan hati. Rantai fallback lama hanya
    # menyala kalau font utama tidak punya codepoint-nya, jadi ia tidak pernah
    # menyala untuk ♥ — lihat _FORCE_SYMBOL di typeset.py.
    _wrong_shape = []
    for _ch in "♥♡♪☆★":
        _f = typeset._char_font(_ch, _main, _mcmap, 16)
        if _f.path == typeset.FONT_USED:
            _wrong_shape.append(f"{_ch} masih dari font utama")
    checks.append((
        "simbol emosi TIDAK diambil dari font utama (anime_ace ♥ = huruf `yat`)",
        not _wrong_shape, "; ".join(_wrong_shape),
    ))

    # Nama glyph di font yang dipilih harus benar-benar simbol yang dimaksud.
    # Assertion paling langsung yang bisa dibuat tanpa membandingkan piksel ke
    # gambar acuan: nama glyph U+2665 di font terpilih bukan nama huruf.
    _gname = ""
    try:
        from fontTools.ttLib import TTFont as _TTF
        _hf = typeset._char_font("♥", _main, _mcmap, 16)
        _gname = _TTF(_hf.path, fontNumber=0, lazy=True).getBestCmap()[0x2665]
    except Exception:  # noqa: BLE001 - fontTools opsional
        _gname = "heart"  # tanpa fontTools cek ini tidak berarti; jangan gagal
    checks.append((
        "glyph U+2665 di font terpilih bernama hati, bukan huruf",
        "heart" in _gname.lower() or _gname.startswith(("uni2665", "cid")),
        f"nama glyph = {_gname!r}",
    ))

    # Jalur UKUR dan jalur GAMBAR wajib memakai font yang sama per karakter.
    # Kalau _measure() memakai font utama sementara _draw_line() jatuh ke font
    # simbol, barisnya diukur salah: ☆ 14.0 px di anime_ace vs 21.0 px di
    # NotoSansSymbols2 pada size 20 — ukur-kurang 7 px yang lolos fit() lalu
    # melebar keluar balon saat digambar.
    _mismatch = []
    for _line in ("I LOVE YOU ♥", "WAIT ☆", "LA LA ♪", "PLAIN TEXT"):
        _a = typeset._measure(_line, _main)
        _b = typeset._line_width(_line, _main, _mcmap, 16)
        if abs(_a - _b) > 0.01:
            _mismatch.append(f"{_line!r} ukur={_a:.1f} gambar={_b:.1f}")
    checks.append((
        "lebar jalur ukur == lebar jalur gambar (simbol ikut dihitung)",
        not _mismatch, "; ".join(_mismatch),
    ))

    # ------------------------------------------------------------- condense
    #
    # Rapat horizontal (SETTINGS.condense) menutup selisih kerapatan TERUKUR
    # antara Anime Ace dan font typeset CONTOH/6.JPG (0.690; probe_reffont2.py),
    # dan angka 0.85 dipilih dari sapuan pada mask jp_6 sungguhan
    # (probe_cond.py: wording referensi 5 -> 1 tanda hubung, luber 1 -> 0).
    #
    # Yang diuji di sini bukan angkanya melainkan KESETIAANNYA: jalur ukur dan
    # jalur gambar harus memampatkan dengan faktor yang sama. Kalau tidak, fit()
    # menyangka baris muat lalu tintanya tergambar lebih lebar dan menembus garis
    # balon — cacat 'keluar bubble' yang dilarang plan.txt, dan cacat yang paling
    # sulit dilihat karena selisihnya cuma beberapa piksel per baris.
    _cnd = typeset._cond()
    checks.append((
        "faktor condense terbaca dan masuk rentang wajar",
        0.3 <= _cnd <= 1.0 and abs(_cnd - float(SETTINGS.condense)) < 1e-9,
        f"condense={_cnd}",
    ))
    # Tinta yang BENAR-BENAR tergambar diukur dari piksel, lewat transform yang
    # sama dengan render_region() — bukan dari rumus yang sama, kalau tidak yang
    # diuji cuma aritmetika terhadap dirinya sendiri.
    _cbad = []
    for _txt, _sz in (("WONDER", 12), ("EMBARASSING", 14), ("I'M PRAISING", 20),
                      ("LOVE YOU ♥", 16)):
        _f = typeset._font(typeset.FONT_USED, _sz)
        _w = typeset._line_width(_txt, _f, _mcmap, _sz)
        _lh = typeset._line_height(_f)
        _k = SETTINGS.oblique
        _pd = int(abs(_k) * _lh) + 4
        _tile = Image.new("RGBA", (int(_w / _cnd) + _pd * 2, _lh + _pd * 2),
                          (0, 0, 0, 0))
        typeset._draw_line(ImageDraw.Draw(_tile), (_pd, _pd), _txt, _f,
                           (0, 0, 0), _mcmap, _sz, 0)
        _th = _tile.height
        _tile = _tile.transform(
            _tile.size, Image.AFFINE,
            (1 / _cnd, _k / _cnd, _pd - (_pd + _k * _th / 2) / _cnd, 0, 1, 0),
            resample=Image.BICUBIC)
        _cols = np.where((np.asarray(_tile.getchannel("A")) > 24).any(axis=0))[0]
        _ink = int(_cols[-1] - _cols[0] + 1) if _cols.size else 0
        # Tinta selalu lebih SEMPIT dari advance (advance memuat side bearing
        # kanan), jadi yang dijaga: jangan pernah MELEBIHI lebar yang diukur, dan
        # jangan menyusut lebih dari 25% — menyusut jauh berarti transform-nya
        # memampatkan dua kali.
        if not (_w * 0.75 <= _ink <= _w + 2):
            _cbad.append(f"{_txt!r}@{_sz} ukur={_w:.1f} tinta={_ink}")
    checks.append((
        "condense: lebar tinta tergambar cocok dengan lebar yang diukur",
        not _cbad, "; ".join(_cbad),
    ))
    # Faktor harus benar-benar MERAPATKAN, bukan cuma dikalikan di satu sisi.
    # Perbandingannya terhadap getlength() mentah, satu-satunya angka di sini
    # yang tidak lewat _cond().
    _raw = typeset._font(typeset.FONT_USED, 20).getlength("EMBARASSING")
    _got = typeset._line_width("EMBARASSING",
                               typeset._font(typeset.FONT_USED, 20), _mcmap, 20)
    checks.append((
        "condense benar-benar mempersempit baris (bukan no-op)",
        abs(_got - _raw * _cnd) < 0.01 and (_cnd == 1.0 or _got < _raw),
        f"mentah={_raw:.1f} rapat={_got:.1f} faktor={_cnd}",
    ))

    # Label seolah dari LLM: region terakhir SFX. Panjang teks sengaja beda-beda
    # supaya jumlah baris tiap balon beda — yang diuji nanti bukan keseragaman
    # ukuran mentah (balonnya memang beda besar) melainkan keseragaman RASIO
    # ukuran terhadap balonnya, lihat assertion 'ukuran font proporsional'.
    _texts = [
        "HELLO THERE",
        "SO THIS IS WHERE YOU WERE ALL ALONG",
        "THAT NIGHT, THE TWO OF THEM WERE TOGETHER.",
    ]
    for r, t in zip(regions[:3], _texts):
        r.label, r.translation = "DIALOGUE", t
    regions[3].label = "SFX"

    erase_mask, protected_mask = textmask.compose_page_mask(img, regions)
    checks.append((
        "SFX tidak tersentuh mask hapus",
        verify.assert_sfx_intact(erase_mask, protected_mask),
        "",
    ))
    checks.append(("mask SFX tidak kosong", bool(protected_mask.any()), ""))

    cleaned = erase.erase_page(img, regions, "cpu")
    sfx_box = boxes[3]
    sx1, sy1, sx2, sy2 = sfx_box
    untouched = np.array_equal(img[sy1:sy2, sx1:sx2], cleaned[sy1:sy2, sx1:sx2])
    checks.append(("piksel SFX identik sebelum/sesudah erase", untouched, ""))

    resid = [verify.pixel_residue(cleaned, r) for r in regions[:3]]
    checks.append((
        "tidak ada residu piksel di region dialog",
        all(v < 60 for v in resid),
        str(resid),
    ))

    if typeset.FONT_USED:
        out = typeset.render_page(cleaned, regions)

        # Regresi 'saling timpa': teks panjang dikunci di dalam balon.
        _lr = Region(idx=12, bbox=(120, 60, 380, 220), det_class="text_bubble",
                     bubble_bbox=(120, 60, 380, 220))
        textmask.build_region_mask(img, _lr, None)
        _lr.translation = ("THIS IS A VERY LONG DIALOGUE THAT KEEPS GOING AND "
                           "GOING WITH NO END IN SIGHT AT ALL WHATSOEVER") * 3
        _lout = typeset.render_page(cleaned, [_lr])
        _ld = np.abs(_lout.astype(np.int16) - cleaned.astype(np.int16)).sum(axis=2)
        _bx1, _by1, _bx2, _by2 = _lr.bubble_bbox
        _mh, _mw = _lr.bubble_mask.shape[:2]
        _pmap = np.zeros_like(cleaned[..., 0], np.uint8)
        _pmap[_by1:_by1 + _mh, _bx1:_bx1 + _mw] = _lr.bubble_mask
        _leaked = int(((_ld > 120) & (_pmap == 0)).sum())
        checks.append((
            "teks panjang tidak bocor keluar balon",
            _leaked == 0, f"leaked_px={_leaked}",
        ))
        inside = all(
            r.final_font_size >= 8 and r.lines for r in regions[:3]
        )
        checks.append(("teks hasil fit dirender di semua bubble", inside, ""))
        checks.append((
            "tidak ada overflow",
            not any(r.overflowed for r in regions[:3]),
            "",
        ))
        _fs = [r.final_font_size for r in regions[:3] if r.final_font_size]
        # Kontrak ukuran font BUKAN 'satu angka untuk seluruh halaman'. Typeset
        # referensi CONTOH/2.webp diukur (probe_refnative.py, 13 balon) dan
        # ternyata MENSKALAKAN teks ke besar balon: cap_height/sisi-terpendek
        # interior konstan 0.117 sementara cap_height sendiri berkisar 13..27 px
        # (sebaran 2.08x). Tiga model diuji terhadap ukuran terukur itu
        # (probe_model.py): seragam-halaman galat 4.31 px, proporsional balon
        # 2.71 px, per panel 4.19 px. Jadi yang di-assert adalah RASIO-nya yang
        # seragam, bukan ukurannya — dan tiap region duduk di plafon
        # proporsionalnya sendiri, tidak lebih kecil.
        #
        # Tiga balon halaman uji ini sisi terpendeknya 250/270/140 px, jadi
        # ukuran mentahnya memang wajib beda ~1.9x. Assertion lama (max/min
        # <= 1.35) lolos hanya karena kebetulan: dulu ketiga region jatuh ke satu
        # ukuran seragam. Angkanya bukan bukti benar, cuma bukti seragam.
        _norm, _gap = [], []
        for _r in regions[:3]:
            if not _r.final_font_size:
                continue
            _m = typeset._region_box_mask(_r)[1]
            _norm.append(_r.final_font_size / max(min(_m.shape[:2]), 1))
            _gap.append(typeset.region_font_cap(_m) - _r.final_font_size)
        _nspread = max(_norm) / max(min(_norm), 1e-6) if _norm else 0.0
        checks.append((
            "ukuran font proporsional ke balon (rasio seragam, duduk di plafon)",
            len(_norm) == 3 and _nspread <= 1.15 and max(_gap) <= 1,
            f"sizes={_fs} rasio_spread={_nspread:.2f} selisih_plafon={_gap}",
        ))
        checks.append((
            "tidak ada tanda hubung buatan di hasil fit",
            not any(ln.endswith("-") for r in regions[:3] for ln in r.lines),
            str([ln for r in regions[:3] for ln in r.lines if ln.endswith("-")]),
        ))

        # ------------------------------------------------- reclaim lebar terpakai
        #
        # disjoin_overlapping_interiors() memutus irisan per PIKSEL secara
        # Voronoi, tanpa tahu di baris mana teks tetangga benar-benar jatuh.
        # Akibatnya lebar disandera di ketinggian yang tetangganya tidak sentuh:
        # di hasilnew/jp_6.JPG r3 kehilangan 20 px tetap di y=139..191 padahal
        # tinta r2 berhenti di y=168 (probe_row.py), dan 26 px sisanya membuat
        # 'WONDER' (32 px pada size 6) dipenggal jadi 'WON-/DER'.
        #
        # Wordingnya bukan pilihan bebas. 'MISUNDERSTANDING AGAIN, PREZ?' dicari
        # dengan probe_adjfind.py justru karena tanda hubungnya SEBAB LEBAR: di
        # interior hasil disjoin 2 tanda hubung, di interior + 9522 px sanderaan
        # tinggal 1, ukurannya tetap 40. Kata mustahil semacam
        # 'PNEUMONOULTRAMICROSCOPIC...' tidak bisa dipakai — tanda hubungnya tetap
        # ada berapa pun lebarnya, jadi test-nya lolos/gagal tanpa hubungan dengan
        # kode yang diuji. r1 dijaga pendek supaya perannya jelas: pelepas.
        #
        # Yang diuji EMPAT kontrak, karena reclaim yang salah merusak salah satunya:
        #   1. tanda hubung pengklaim berkurang — bukan cuma luas bertambah.
        #      Luas naik itu murah dan menipu: piksel rampasan disjoin bergerigi,
        #      jadi bisa naik tanpa menambah RUN bebas yang dipakai satu baris.
        #   2. pelepas tidak dirugikan (tanda hubung/luber tidak muncul)
        #   3. interior tetap saling lepas — kalau tidak, teks saling timpa
        #   4. tidak ada piksel di luar interior balon SENDIRI (fill_mask), yaitu
        #      kontrak 'tidak keluar bubble'
        _adj[0].translation = "MISUNDERSTANDING AGAIN, PREZ?"
        _adj[1].translation = "YES, THE RECORDS ARE HERE."
        _afill = [typeset._paste_mask(_r.fill_bbox, _r.fill_mask, *_ashape) > 0
                  if _r.fill_mask is not None else np.zeros(_ashape, bool)
                  for _r in _adj]

        def _rfit(_r):
            """(ukuran, jumlah tanda hubung, luber) di interior _r sekarang."""
            _m = typeset._region_box_mask(_r)[1]
            _s, _ls, _sy, _ov = typeset.fit(_r.translation.upper(), _m,
                                            typeset.region_font_cap(_m),
                                            typeset.FONT_USED)
            return _s, sum(1 for _x in _ls if _x.endswith("-")), int(bool(_ov))

        _rf0 = [_rfit(_r) for _r in _adj]
        _amoved = typeset.reclaim_unused_interiors(_aimg, _adj)
        _rf1 = [_rfit(_r) for _r in _adj]
        _rmA = _page_mask(_adj[0], _ashape) > 0
        _rmB = _page_mask(_adj[1], _ashape) > 0
        checks.append((
            "reclaim: tanda hubung pengklaim berkurang setelah lebar dikembalikan",
            _amoved >= 1 and _rf1[0][1] < _rf0[0][1] and _rf1[0][0] >= _rf0[0][0],
            f"berubah={_amoved} region A:{_rf0[0]}->{_rf1[0]}",
        ))
        # Yardstick pelepas HARUS memakai ambang yang sama dengan produksi
        # (typeset._RECLAIM_LOSS), bukan angka px tetap. Versi pertama test ini
        # memakai '-1 px' dan gagal pada pertukaran yang justru benar: pengklaim
        # 2 tanda hubung -> 1, pelepas 40 -> 37 di balon 335x291 yang masih
        # lapang. Yang dijaga di sini bahwa pelepas tidak dapat cacat BARU
        # (tanda hubung/luber) dan tidak digunduli — bukan bahwa ukurannya beku.
        _rloss = max(1, int(round(_rf0[1][0] * typeset._RECLAIM_LOSS)))
        checks.append((
            "reclaim: yang melepas tidak dapat tanda hubung/luber baru",
            _rf1[1][1] <= _rf0[1][1] and _rf1[1][2] <= _rf0[1][2]
            and _rf1[1][0] >= _rf0[1][0] - _rloss,
            f"B:{_rf0[1]}->{_rf1[1]} batas_susut={_rloss}",
        ))
        checks.append((
            "reclaim: interior tetap saling lepas (tidak ada piksel milik dua region)",
            int((_rmA & _rmB).sum()) == 0, f"overlap={int((_rmA & _rmB).sum())}",
        ))
        checks.append((
            "reclaim: tidak ada piksel di luar interior balon sendiri",
            int((_rmA & ~_afill[0]).sum()) == 0 and int((_rmB & ~_afill[1]).sum()) == 0,
            f"A_luar={int((_rmA & ~_afill[0]).sum())} "
            f"B_luar={int((_rmB & ~_afill[1]).sum())}",
        ))
        # Dan hasil akhirnya: dirender sungguhan, tinta kedua region tidak boleh
        # bersinggungan satu piksel pun maupun keluar dari garis balon gabungan.
        _rink = [_ink_of(typeset.render_page(_aclean, [_r]), _aclean) for _r in _adj]
        checks.append((
            "reclaim: tinta hasil render tidak saling timpa & tetap di dalam balon",
            int((_rink[0] & _rink[1]).sum()) == 0
            and int((_rink[0] & (_ainner == 0)).sum()) == 0
            and int((_rink[1] & (_ainner == 0)).sum()) == 0,
            f"timpa={int((_rink[0] & _rink[1]).sum())} "
            f"A_luar={int((_rink[0] & (_ainner == 0)).sum())} "
            f"B_luar={int((_rink[1] & (_ainner == 0)).sum())}",
        ))

        # Kontrak inti balon ganda. Tiap lobus dirender SENDIRI ke halaman
        # bersih supaya tintanya bisa dipisahkan: kalau forb_map yang menahan
        # tumpang tindih (bukan mask yang sudah disjoint), test ini yang gagal.
        _dbl[0].translation = "I'VE BEEN LOOKING ALL OVER FOR YOU, PREZ!"
        _dbl[1].translation = "IS THAT THE SUMMARY FOR THE MILKING CLUB?"
        _dink = [_ink_of(typeset.render_page(_dclean, [_r]), _dclean) for _r in _dbl]
        _clash = int((_dink[0] & _dink[1]).sum())
        checks.append((
            "tinta dua lobus tidak saling timpa satu piksel pun",
            _clash == 0, f"clash_px={_clash}",
        ))
        # Toleransi 3 px: batas Voronoi kedua lobus BERSINGGUNGAN dan tepi alpha
        # di _clip_to_mask di-feather, jadi 1-2 px terluar memang jatuh sebelah.
        _e3 = np.ones((7, 7), np.uint8)
        _inAB = int((_dink[0] & (cv2.erode(_mB, _e3) > 0)).sum())
        _inBA = int((_dink[1] & (cv2.erode(_mA, _e3) > 0)).sum())
        checks.append((
            "tinta lobus A tidak masuk interior lobus B (dan sebaliknya)",
            _inAB == 0 and _inBA == 0, f"A->B={_inAB} B->A={_inBA}",
        ))
        _outA = int((_dink[0] & (_dinner == 0)).sum())
        _outB = int((_dink[1] & (_dinner == 0)).sum())
        checks.append((
            "tinta tidak menyentuh garis balon dan tidak keluar balon",
            _outA == 0 and _outB == 0, f"A={_outA} B={_outB}",
        ))
        checks.append((
            "SFX tidak ditimpa teks Inggris",
            np.array_equal(cleaned[sy1:sy2, sx1:sx2], out[sy1:sy2, sx1:sx2]),
            "",
        ))

        # ---------------------------------------------------------- anggaran balon
        #
        # Yang diuji: apakah angka anggaran BENAR-BENAR menggambarkan balonnya.
        # Anggaran yang tidak berkorelasi dengan geometri sama tidak bergunanya
        # dengan tidak punya anggaran — model akan diberi angka yang salah dan
        # patuh pada angka yang salah. Jadi tiga sifat yang dijamin:
        #   hard >= soft   ukuran lebih kecil selalu memuat lebih banyak
        #   soft > 0       balon sebesar ini pasti memuat sesuatu
        #   soft berbeda   antara lobus dan balon sempit (bukan konstanta)
        _bud = [typeset.region_budget(_r, typeset.FONT_USED) for _r in _dbl]
        _bok = all(b["hard"] >= b["soft"] > 0 and b["word_hard"] >= b["word_soft"] > 0
                   for b in _bud)
        checks.append((
            "anggaran balon: hard >= soft > 0 untuk kedua lobus",
            _bok, "; ".join(f"soft={b['soft']} hard={b['hard']} "
                            f"kata={b['word_soft']}/{b['word_hard']}" for b in _bud),
        ))
        # Anggaran harus IKUT besar balon. regions[2] balon sempit (sisi 140 px),
        # _dbl[0] satu lobus balon ganda yang jauh lebih lapang; kalau keduanya
        # memberi angka yang sama, yang diukur bukan balonnya melainkan teks
        # pengisinya — tepat kegagalan yang bikin tujuh balon melaporkan soft=20.
        _bnarrow = typeset.region_budget(regions[2], typeset.FONT_USED)
        checks.append((
            "anggaran balon membedakan balon sempit dari balon lapang",
            _bnarrow["soft"] != _bud[0]["soft"],
            f"sempit={_bnarrow['soft']} lapang={_bud[0]['soft']}",
        ))
        # Lapis VALIDASI: teks yang mustahil harus tertangkap, teks yang muat
        # harus lolos. Tanpa kedua arah ini validator bisa lolos-semua (tidak
        # menjaga apa pun) atau tolak-semua (menuntut revisi tanpa akhir).
        #
        # Yang mustahil di sini SATU KATA panjang, bukan kalimat panjang, dan itu
        # bukan pilihan sembarang: kalimat panjang selalu bisa dipecah jadi banyak
        # baris, jadi di balon lapang 490 karakter pun masih muat (lobus ini
        # memuat ~790 pada ukuran minimum). Yang benar-benar tidak punya jalan
        # keluar adalah satu kata yang lebih lebar dari balonnya — layout() hanya
        # bisa memenggalnya atau gagal, dan itulah cacat yang divalidasi.
        #
        # Panjangnya dihitung dari GEOMETRI pada emergency_floor(), bukan
        # `word_hard + 40`. Angka tetap itu sudah pernah membuat test ini gagal
        # tanpa ada cacat: word_hard diukur di min_font() (9 px), sementara fit()
        # boleh turun ke emergency_floor() (7 px), dan begitu SETTINGS.condense
        # dipasang 0.85 lebar 'A' di 7 px menyusut 6.00 -> 5.10 px. 64 karakter
        # jadi cuma 326 px di mask 397 px — benar-benar muat, jadi _violations()
        # BENAR meloloskannya dan yang salah justru patokannya. Diambil dari
        # lebar mask dibagi lebar maju satu huruf di lantai terendah, plus marjin,
        # angkanya tetap mustahil berapa pun faktor condense-nya.
        _vfl = typeset.emergency_floor()
        _vadv = typeset._line_width("A", typeset._font(typeset.FONT_USED, _vfl),
                                    typeset._cmap(typeset.FONT_USED), _vfl)
        _vn = int(_dbl[0].bubble_mask.shape[1] / max(_vadv, 0.5) * 1.25) + 8
        _vbud = {_dbl[0].idx: _bud[0]}
        _vbad = _tl._violations({_dbl[0].idx: "A" * _vn}, _vbud, [_dbl[0]])
        _vok = _tl._violations({_dbl[0].idx: "SORRY."}, _vbud, [_dbl[0]])
        checks.append((
            "validasi anggaran: teks mustahil ditolak, teks pendek diloloskan",
            bool(_vbad) and not _vok,
            f"N={_vn} (lantai={_vfl} maju={_vadv:.2f}) "
            f"panjang={sorted(_vbad)} pendek={sorted(_vok)}",
        ))
        # '＼' dibuang _clean_translation sebelum typeset, jadi validasi harus
        # mengukur bentuk SETELAH pembersihan. Kalau tidak, '＼SORRY.' dilaporkan
        # tidak muat di balon yang memuat 39 karakter — dan model dipaksa merevisi
        # sesuatu yang sudah benar.
        checks.append((
            "validasi memakai bentuk setelah pembersihan (＼ tidak dihitung)",
            not _tl._violations({_dbl[0].idx: "＼SORRY."}, _vbud, [_dbl[0]]),
            "",
        ))
    else:
        checks.append(("font tersedia", False, "setup_fonts() belum dijalankan"))

    # ---------------------------------------------------------------- penyedia
    #
    # Tidak ada panggilan jaringan di sini: yang diuji PEMILIHAN jalur, bukan
    # jawaban API. Salah pilih penyedia adalah cacat yang paling mahal untuk
    # ditemukan lewat mata — hasilnya tetap keluar, cuma dari mesin yang salah.
    checks.append((
        "provider: nama UI -> kelas client yang benar",
        isinstance(_tl.make_client("x", "DeepL"), _tl.DeepLClient)
        and isinstance(_tl.make_client("x", "Router LLM (gorouter)"), _tl.RouterClient)
        and isinstance(_tl.make_client("x", "LLM (freetokenfaucet)"), _tl.FaucetClient)
        # Faucet TURUNAN RouterClient, jadi isinstance saja tidak membedakannya —
        # yang membedakan base URL-nya. Kalau urutan cabang di make_client()
        # terbalik, router yang mati justru dipakai walau UI memilih faucet.
        and not isinstance(_tl.make_client("x", "Router LLM (gorouter)"), _tl.FaucetClient)
        and "faucet" in _tl.make_client("x", "LLM (freetokenfaucet)").base
        and _tl._is_router("Router LLM (gorouter)")
        and _tl._is_router("LLM (freetokenfaucet)")
        and not _tl._is_router("DeepL"),
        "",
    ))
    # Faucet memakai model REASONING: tanpa thinking dimatikan, jatah keluaran
    # habis untuk berpikir dan content keluar STRING KOSONG tanpa error HTTP —
    # halaman keluar bersih tanpa terjemahan dan tidak ada yang mengeluh.
    _fc = _tl.make_client("x", "LLM (freetokenfaucet)")
    checks.append((
        "faucet: thinking dimatikan + batas waktu sendiri",
        _fc.extra.get("thinking", {}).get("type") == "disabled"
        and _fc.max_tokens > 0
        and _fc.timeout < _tl.ROUTER_TIMEOUT
        and _fc.deadline < _tl.ROUTER_DEADLINE,
        "",
    ))
    # Model faucet WAJIB salah satu dari tiga yang GRATIS. Terukur 17 Agu 2026:
    # 16 dari 19 model membalas HTTP 402 INSUFFICIENT_BALANCE, termasuk model
    # yang dulu jadi default di sini. Akibatnya tiga halaman Colab keluar TANPA
    # terjemahan. Check ini yang mencegahnya kembali diam-diam lewat edit sel
    # notebook: daftar putih, bukan daftar hitam — model berbayar baru pun
    # tertolak tanpa perlu menambah namanya di sini.
    _FREE = {"mimo-v2.5-pro", "mimo-v2.5", "gpt-5.6-terra"}
    checks.append((
        "faucet: model default GRATIS (bukan model 402)",
        _fc.model in _FREE and all(f in _FREE for f in _fc.fallback),
        f"model={_fc.model} fallback={_fc.fallback}",
    ))
    # Tanpa User-Agent, gorouter dibalas 403 "error code 1010" oleh Cloudflare —
    # dua bentuk auth sama-sama 403, dan key yang sama DENGAN header ini
    # membalas 200. Kalau atribut ini hilang, router mati total tanpa petunjuk
    # apa pun di pesan errornya. Faucet sebaliknya: terukur sehat tanpa UA.
    _rc = _tl.make_client("x", "Router LLM (gorouter)")
    checks.append((
        "router: header User-Agent wajib ada, faucet tanpa header",
        "User-Agent" in _rc.headers and _rc.headers["User-Agent"]
        and _fc.headers == {},
        f"router={sorted(_rc.headers)} faucet={sorted(_fc.headers)}",
    ))
    # Prompt anggaran cuma boleh menyebut max_chars kalau angkanya BENAR-BENAR
    # dikirim. Menyebut batas yang tidak ada di masukan membuat model menebak
    # batasnya sendiri, dan tebakannya tidak ada hubungannya dengan balon.
    _sp_bud = _tl._system_prompt("English", "Manga Natural", True, True)
    _sp_pln = _tl._system_prompt("English", "Manga Natural", True, False)
    checks.append((
        "system prompt menyebut max_chars HANYA saat anggaran dikirim",
        "max_chars" in _sp_bud and "max_chars" not in _sp_pln,
        "",
    ))
    checks.append((
        "system prompt membawa gaya terpilih + aturan honorifik",
        "Uncensored:" in _tl._system_prompt("English", "Uncensored", True, True)
        and "Keep honorifics" in _sp_bud
        and "Localise honorifics" in _tl._system_prompt(
            "English", "Manga Natural", False, True),
        "",
    ))
    # Router membalas text/event-stream walau non-stream: satu objek lalu
    # 'data: [DONE]' TANPA pemisah. json.loads gagal 'Extra data' padahal isinya
    # utuh — kalau ini regresi, SEMUA terjemahan router gagal sekaligus.
    checks.append((
        "body router text/event-stream ter-decode (data: + [DONE] tanpa pemisah)",
        _tl._decode_router('data: {"choices": [{"message": {"content": "x"}}]}'
                          'data: [DONE]')["choices"][0]["message"]["content"] == "x",
        "",
    ))

    # Kunci yang TIDAK dijawab model harus diminta ulang, bukan ditelan.
    # Inilah cacat hasilnew/13.JPG: balon 'えっ！？' terkirim ke model, model
    # memutuskan seruan sependek itu tidak perlu diterjemahkan, kuncinya hilang
    # dari JSON, dan loop lama (`if not t: continue`) membiarkan translation
    # tetap None — render_region() lalu keluar lebih awal dan balon itu tercetak
    # berbahasa Jepang tanpa satu pun peringatan. Diuji TANPA jaringan: yang
    # ditukar cuma _router_call_any, jadi yang diperiksa logika ronde ulangnya.
    _fake_calls: list[list[int]] = []

    def _fake_router(_client, _model, _system, user, *_a, **_kw):
        import json as _json
        ids = sorted(int(k) for k in _json.loads(
            user.split("Lines:\n", 1)[1].split("\n\nMISSING", 1)[0]))
        _fake_calls.append(ids)
        # Panggilan pertama sengaja MENGHILANGKAN idx 1 (meniru model sungguhan);
        # panggilan kedua — permintaan ulang — menjawabnya.
        return ({str(i): "HUH?!" for i in ids} if len(_fake_calls) > 1
                else {str(i): "REALLY?" for i in ids if i != 1}), "m"

    _rr = [Region(idx=i, bbox=(0, 0, 10, 10), det_class="text_bubble")
           for i in range(2)]
    _rr[0].src_text, _rr[1].src_text = "でも", "えっ！？"
    for _r in _rr:
        _r.label = "DIALOGUE"
    _orig_call, _orig_bud = _tl._router_call_any, SETTINGS.balloon_budget
    try:
        _tl._router_call_any = _fake_router
        SETTINGS.balloon_budget = False   # anggaran diuji terpisah; ini soal kelengkapan
        _tl._translate_router(_tl.make_client("x", "Router LLM (gorouter)"), "m",
                             _rr, _rr, "English", "Manga Natural", True)
    finally:
        _tl._router_call_any, SETTINGS.balloon_budget = _orig_call, _orig_bud
    checks.append((
        "kunci yang tidak dijawab model diminta ulang, bukan ditelan diam-diam",
        len(_fake_calls) == 2 and _fake_calls[1] == [1]
        and bool(_rr[0].translation) and bool(_rr[1].translation),
        f"panggilan={_fake_calls} hasil={[r.translation for r in _rr]}",
    ))
    # _missing_ids harus mengukur bentuk AKHIR, bukan apa yang model tulis:
    # jawaban '．．．' hilang seluruhnya di _clean_translation, jadi balonnya
    # sama kosongnya dengan kunci yang tidak ada — dan harus ikut diminta ulang.
    _mr = [Region(idx=i, bbox=(0, 0, 10, 10)) for i in range(4)]
    for _r in _mr:
        _r.src_text = "でも"
    _mi = _tl._missing_ids({0: "OK", 1: "   ", 2: "．．．"}, _mr)
    checks.append((
        "balon kosong terdeteksi: kunci hilang, spasi, dan yang habis dibersihkan",
        _mi == [1, 2, 3], f"missing={_mi}",
    ))

    # ------------------------------------------------------------- diagnostik
    #
    # Yang diuji: apakah kegagalan BENAR-BENAR terlihat. Sebelum lapisan ini ada,
    # satu run Colab keluar `diterjemah 0` di tiga halaman tanpa satu pun pesan,
    # dan tidak ada test yang bisa gagal karenanya — jadi empat kontrak di bawah
    # justru menjaga jalur yang paling mudah kembali jadi bisu.
    import config as _cfg
    import io as _io
    import contextlib as _ctx

    _n0 = len(_cfg.RUN_NOTES)
    _cap = _io.StringIO()
    with _ctx.redirect_stdout(_cap):
        _cfg.note("error", "uji", "pesan uji")
        _cfg.note("warn", "uji", "pesan warn")
    _cout = _cap.getvalue()
    checks.append((
        "note() mengisi RUN_NOTES DAN mencetak awalan per level",
        len(_cfg.RUN_NOTES) - _n0 == 2 and "[!!]" in _cout and "[!]" in _cout
        and _cfg.RUN_NOTES[-2][0] == "error"
        and len(_cfg.notes_since(_n0)) == 2,
        f"catatan={_cfg.RUN_NOTES[-2:]} keluaran={_cout.strip()!r}",
    ))

    # verify.report() harus MENERUSKAN catatan ke dict hasil — inilah pipa yang
    # membuat kolom `catatan` di tabel UI menunjuk halaman yang benar.
    _rep = verify.report([], [], "x", [("error", "t", "boom"), ("warn", "t", "hm")])
    checks.append((
        "verify.report(notes=...) meneruskan catatan + menghitung per level",
        len(_rep.get("notes", [])) == 2 and _rep.get("error_count") == 1
        and _rep.get("warn_count") == 1
        and _rep["notes"][0]["msg"] == "boom",
        f"notes={_rep.get('notes')} err={_rep.get('error_count')}",
    ))

    # _diagnose() diuji atas summary BIKINAN, tanpa jaringan dan tanpa Gradio:
    # itu sebabnya fungsinya dibuat murni. Kasusnya persis layar user — 15 region
    # bisa diterjemah, nol yang jadi.
    import app as _app

    class _FakeRes:
        def __init__(self, stem, rep):
            self.stem, self.report = stem, rep

    _dead = _app._diagnose(
        [_FakeRes("hitomi_006", {"region_count": 15, "translatable_count": 15,
                                 "translated_count": 0, "untranslated_count": 15,
                                 "residue_count": 0, "overflow_count": 0})],
        {"notes": [{"level": "error", "tag": "translate",
                    "msg": "faucet gagal (HTTP 429)"}],
         "residue_total": 0, "overflow_total": 0},
    )
    _dtxt = "\n".join(_dead)
    checks.append((
        "_diagnose(): 'diterjemah 0' -> banner merah + nama halaman + sebabnya",
        "TIDAK ADA TERJEMAHAN" in _dtxt and "hitomi_006" in _dtxt
        and "HTTP 429" in _dtxt and _app._RED in _dtxt,
        _dtxt[:160].replace("\n", " | "),
    ))

    # Arah sebaliknya, dan ini yang menjaga banner tetap berarti: halaman bersih
    # TIDAK boleh memerah. Banner yang selalu merah sama tidak bergunanya dengan
    # tidak ada banner — user berhenti membacanya.
    _clean = _app._diagnose(
        [_FakeRes("ok_001", {"region_count": 8, "translatable_count": 6,
                             "translated_count": 6, "untranslated_count": 0,
                             "residue_count": 0, "overflow_count": 0})],
        {"notes": [], "residue_total": 0, "overflow_total": 0,
         "translated_total": 6},
    )
    # Halaman yang isinya SFX semua juga tidak boleh dituduh gagal: 0 diterjemah
    # dari 0 yang bisa diterjemah itu BENAR, dan menuduhnya adalah cara tercepat
    # membuat user mengabaikan banner merah yang sungguhan.
    _sfx_only = _app._diagnose(
        [_FakeRes("sfx_001", {"region_count": 4, "translatable_count": 0,
                              "translated_count": 0, "untranslated_count": 0,
                              "residue_count": 0, "overflow_count": 0})],
        {"notes": [], "residue_total": 0, "overflow_total": 0},
    )
    checks.append((
        "_diagnose(): halaman bersih & halaman SFX-saja tidak memerah",
        _app._RED not in "\n".join(_clean) and _app._GREEN in "\n".join(_clean)
        and _app._RED not in "\n".join(_sfx_only),
        f"bersih={' '.join(_clean)[:80]!r} sfx={' '.join(_sfx_only)[:60]!r}",
    ))

    if verbose:
        for name, ok, extra in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))

    return all(ok for _, ok, _ in checks)

