%%writefile /content/mangatl/textmask.py

"""Pembangun mask teks — bagian paling menentukan kualitas hasil akhir.

Urutan operasinya bukan sembarangan. Langkah dual-polarity Otsu dan halo pass
adalah yang biasanya dilewatkan orang, dan itu penyebab ghost outline.
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # jalur Otsu tetap jalan tanpa ORT

from config import SETTINGS, WEIGHTS, Region, ort_session

_CTD = None
_CTD_FAILED = False


def get_ctd():
    """comic-text-detector opsional: kalau gagal muat, jalur Otsu tetap jalan."""
    global _CTD, _CTD_FAILED
    if _CTD is not None or _CTD_FAILED:
        return _CTD
    path = WEIGHTS / "comictextdetector.pt.onnx"
    if ort is None or not path.exists():
        _CTD_FAILED = True
        return None
    try:
        _CTD = ort_session(path, "ctd")
    except Exception:  # noqa: BLE001 - ORT melempar tipe khusus per build
        _CTD_FAILED = True
        _CTD = None
    return _CTD


def _letterbox(img: np.ndarray, size: int = 1024) -> tuple[np.ndarray, float, int, int, int, int]:
    """Resize menjaga rasio lalu pad ke size x size. RGB dulu, baru /255."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ph, pw = size - nh, size - nw
    top, left = ph // 2, pw // 2
    out = cv2.copyMakeBorder(
        resized, top, ph - top, left, pw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return out, r, left, top, nw, nh


def ctd_soft_mask(img: np.ndarray) -> np.ndarray | None:
    """Mask teks lembut skala halaman penuh, nilai float 0..1. None kalau CTD mati."""
    sess = get_ctd()
    if sess is None:
        return None
    h, w = img.shape[:2]
    lb, _r, dx, dy, nw, nh = _letterbox(img, 1024)
    inp = (lb.transpose(2, 0, 1).astype(np.float32) / 255.0)[None]
    try:
        outs = sess.run(None, {sess.get_inputs()[0].name: inp})
    except (RuntimeError, ValueError):
        return None

    # Address by shape, bukan index — urutan output CTD berbeda antar build,
    # dan cv2.dnn punya bug yang menukar mask dengan lines_map.
    mask = None
    for o in outs:
        a = np.squeeze(o)
        if a.ndim == 2 and min(a.shape) >= 128:
            mask = a
            break
        if a.ndim == 3 and a.shape[0] == 1 and min(a.shape[1:]) >= 128:
            mask = a[0]
            break
    if mask is None:
        return None

    mask = mask.astype(np.float32)
    if mask.max() > 1.5:
        mask /= 255.0
    mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    crop = mask[dy : dy + nh, dx : dx + nw]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR).clip(0, 1)


def _dual_polarity_otsu(gray: np.ndarray) -> np.ndarray:
    """OR-kan biner teks-gelap dengan teks-terang.

    Satu polaritas saja melewatkan glyph putih ber-outline hitam, yang umum
    di manga untuk teks di atas art gelap.
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, light = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Pilih polaritas minoritas: teks selalu lebih sedikit dari background.
    return dark if dark.mean() <= light.mean() else light


def _glyph_height(binary: np.ndarray, vertical: bool = False) -> float:
    """Ukuran glyph dari connected components — dasar estimasi ukuran font.

    Diukur MELINTANG terhadap arah tulisan, bukan selalu tinggi. Di teks Jepang
    vertikal, glyph yang bertumpuk menyatu jadi satu komponen menjulur: terukur
    di halaman uji, kolom selebar 104 px menghasilkan komponen setinggi 163 px,
    jadi "tinggi" melaporkan panjang rangkaian, bukan ukuran huruf. Karena glyph
    CJK persegi, lebar kolom itulah ukuran font sebenarnya. Salah di sini merusak
    dua hal sekaligus: est_font_size dan kernel dilasi yang diturunkan darinya.

    Median mentah juga salah: dakuten, handakuten, dan tanda baca kecil menarik
    median ke bawah (ドドド 76 px terbaca 12 px), jadi komponen kecil dibuang
    dulu relatif terhadap yang terbesar.
    """
    stat = cv2.CC_STAT_WIDTH if vertical else cv2.CC_STAT_HEIGHT
    n, _, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    hs = np.array([
        stats[i, stat]
        for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] >= SETTINGS.min_cc_area
    ], dtype=np.float32)
    if hs.size == 0:
        fallback = binary.shape[1] if vertical else binary.shape[0]
        return float(max(fallback * 0.25, 8))
    body = hs[hs >= hs.max() * 0.45]  # buang diakritik & titik
    return float(np.median(body if body.size else hs))


def _adaptive_dilate(mask: np.ndarray, glyph_h: float) -> np.ndarray:
    """Kernel per-glyph, bukan global.

    Kernel global salah dua arah sekaligus: memakan art di sekitar teks kecil,
    dan kurang menutup teks besar.
    """
    k = max((int((glyph_h + 30) * SETTINGS.dilate_ratio) // 2) * 2 + 1, 3)
    k = min(k, 31)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, el, iterations=1)


def _halo_pass(mask: np.ndarray, gray: np.ndarray, bg: float) -> np.ndarray:
    """Tangkap tepi anti-alias di sekeliling glyph.

    Melewatkan langkah ini adalah penyebab nomor satu ghost outline: stroke
    utamanya terhapus tapi bayangan abu-abunya tertinggal.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grown = cv2.dilate(mask, k, iterations=1)
    ring = cv2.subtract(grown, mask)
    deviating = (np.abs(gray.astype(np.int16) - bg) > SETTINGS.halo_deviation).astype(np.uint8) * 255
    return cv2.bitwise_or(mask, cv2.bitwise_and(ring, deviating))


def build_region_mask(img: np.ndarray, region: Region, soft: np.ndarray | None) -> None:
    """Isi region.ink_mask, region.bubble_mask, region.est_font_size, ink_ratio.

    Semua mask disimpan pada koordinat lokal bbox region.
    """
    x1, y1, x2, y2 = region.bbox
    pad = 6
    h, w = img.shape[:2]
    px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
    px2, py2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = img[py1:py2, px1:px2]
    if crop.size == 0:
        region.ink_mask = np.zeros((1, 1), np.uint8)
        return

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    # 1-2. seed + grow dari CTD kalau tersedia
    ink = None
    if soft is not None:
        sub = soft[py1:py2, px1:px2]
        seed = (sub >= SETTINGS.seed_thresh).astype(np.uint8) * 255
        grow = (sub >= SETTINGS.grow_thresh).astype(np.uint8) * 255
        if seed.any():
            # rekonstruksi: hanya komponen grow yang menyentuh seed
            n, lab = cv2.connectedComponents(grow // 255)
            keep = np.unique(lab[seed > 0])
            ink = np.isin(lab, keep[keep > 0]).astype(np.uint8) * 255

    # 3-4. dual-polarity Otsu — jalur utama kalau CTD tidak ada, penguat kalau ada.
    #
    # Saat CTD ada, Otsu TIDAK di-OR mentah. Otsu memilih polaritas minoritas, dan
    # untuk teks di atas screentone padat yang minoritas itu justru ART-nya: pada
    # kolom narasi halaman uji, CTD sendiri menutup 27% kotak sedangkan hasil OR
    # mentah menutup 57% — seluruh kolom, art dan semua. Jadi Otsu dibatasi ke
    # pita tipis di sekitar tinta CTD: cukup untuk menambal tepi anti-alias dan
    # stroke tipis yang CTD lewatkan, tanpa mengimpor art di sekelilingnya.
    otsu = _dual_polarity_otsu(gray)
    if ink is None or ink.sum() == 0:
        ink = otsu
    else:
        band = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        ink = cv2.bitwise_or(ink, cv2.bitwise_and(otsu, band))

    # Buang komponen yang jelas bukan glyph: noise, garis panel, dan bingkai.
    # Bingkai kotak narasi paling menipu — luasnya kecil (cuma stroke) tapi
    # bounding box-nya sebesar crop, jadi harus disaring lewat fill ratio.
    n, lab, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    clean = np.zeros_like(ink)
    ih, iw = ink.shape[:2]
    box_area = ih * iw
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        if area < 8 or area > box_area * 0.55:
            continue
        bb = ch * cw
        if bb > box_area * 0.30 and area < bb * 0.25:
            continue  # hollow: bingkai / outline balon, bukan huruf
        if ch > ih * 0.90 and cw > iw * 0.90:
            continue
        clean[lab == i] = 255
    if clean.sum() > 0:
        ink = clean

    # Arah tulisan harus diketahui SEBELUM ukuran glyph diukur — pengukurannya
    # melintang terhadap arah itu.
    region.is_vertical = (py2 - py1) > (px2 - px1) * 1.6
    glyph_h = _glyph_height(ink, vertical=region.is_vertical)
    region.est_font_size = glyph_h

    bg = float(np.median(gray[ink == 0])) if (ink == 0).any() else 255.0

    # 5. dilasi adaptif  6. halo pass
    ink = _adaptive_dilate(ink, glyph_h)
    ink = _halo_pass(ink, gray, bg)

    region.ink_mask = ink
    region.ink_ratio = float((ink > 0).mean())
    # bbox ikut padding dulu, baru bubble_mask dibuat — supaya ukuran mask
    # cocok dengan kotak yang dipakai typeset.
    region.bbox = (px1, py1, px2, py2)
    region.bubble_mask = _bubble_interior(img, region)
    # ...lalu mask ISIAN direkam, SEBELUM disjoin_overlapping_interiors()
    # memangkas bubble_mask. Lihat build_fill_mask().
    build_fill_mask(img, region)


def build_fill_mask(img: np.ndarray, region: Region) -> None:
    """Isi region.fill_mask/fill_bbox: SELURUH interior balon untuk erase.

    Kenapa interior penuh dan bukan ink_mask: ink_mask dibangun dari ambang +
    dilasi adaptif dan sistematis melewatkan glyph tipis/renggang. Terukur di
    hasilnew/jp_6.JPG setelah hapusan — tanda '——' selamat utuh sebagai garis
    tipis, 'うう…' menyisakan dua coretan. Menaikkan dilasi cuma memindahkan
    cacatnya ke garis balon yang ikut termakan; interior balon tidak punya
    masalah itu karena batasnya garis balon sungguhan, bukan taksiran.

    Kenapa direkam di sini, bukan dibaca dari bubble_mask saat erase:
    disjoin_overlapping_interiors() MEMANGKAS bubble_mask supaya tata letak dua
    balon bertetangga tidak beririsan. Mengisi pakai mask yang sudah dipangkas
    membuat sliver yang dipangkas itu tidak pernah tersentuh — dan justru di
    sliver itu tinta Jepang paling mungkin tertinggal, karena letaknya di tepi.

    Kenapa hanya kalau ada balon induk: bubble_bbox None berarti teks duduk di
    ART, bukan di balon. Memutihkan persegi di atas art adalah cacat yang jauh
    lebih parah daripada satu titik sisa, jadi region itu tetap jalur ink_mask.
    """
    region.fill_bbox = None
    region.fill_mask = None
    if not SETTINGS.bubble_fill or region.bubble_bbox is None:
        return
    bx1, by1, bx2, by2 = region.bubble_bbox
    crop = img[by1:by2, bx1:bx2]
    if crop.size == 0:
        return
    # Kikis lebih sedikit daripada interior untuk tata letak: isian harus mepet
    # garis supaya tidak ada pita tinta lama tertinggal di tepi, sementara tata
    # letak perlu jarak aman supaya glyph tidak menempel di garis.
    stroke = max(int(round(_stroke_px(region.est_font_size)
                           * SETTINGS.fill_erode_stroke)), 1)
    interior = _interior_from_crop(crop, stroke, _ink_center(region, bx1, by1),
                                   _ink_in_crop(region, bx1, by1, crop.shape[:2]))
    if not interior.any():
        return
    # Pangkas ke komponen yang MEMUAT tinta region ini — lihat _keep_ink_lobes().
    interior = _keep_ink_lobes(interior, region, bx1, by1)
    if interior is None:
        return
    # Interior yang praktis seluruh kotak = flood fill bocor keluar balon dan
    # mengisi art. Lebih baik jatuh ke ink_mask daripada memutihkan panel.
    # Diuji SETELAH pangkas: sebelum pangkas ambang ini tidak bisa dipakai sama
    # sekali, karena interior yang SAH pun sampai 0.900 di halaman uji (r9).
    if float((interior > 0).mean()) > 0.97:
        return
    region.fill_bbox = region.bubble_bbox
    region.fill_mask = interior


def _keep_ink_lobes(interior: np.ndarray, region: Region,
                    ox: int, oy: int) -> np.ndarray | None:
    """Buang komponen interior yang tidak memuat tinta region ini. None = batal.

    Ini penjaga kebocoran isian yang sesungguhnya. Alasannya STRUKTURAL, bukan
    ambang: isian hanya boleh mengisi rongga tempat tinta Jepangnya berada.
    Piksel terang yang tersambung ke tinta itu memang interior balon; gumpalan
    terang lain di dalam kotak yang sama adalah ART di luar garis balon, dan
    mengecatnya dengan warna balon persis cacat 'isian keluar dari balon'.

    Kenapa ambang ukuran tidak bisa dipakai (terukur di jepang_002.webp,
    _fillcal.py + _fillguard2.py):

      * cover (fraksi kotak yang terisi) di 13 region BERSIH = 0.598-0.900.
        Band _DISCOVER_FILL (0.15, 0.85) akan menolak r9 di 0.900 yang isinya
        benar, jadi angka itu tidak bisa dipindah ke sini.
      * fraksi TEPI kotak yang terisi = 0.0000-0.6946 di halaman bersih. Tidak
        memisahkan apa pun.
      * sebaran warna isian juga tidak: interior balon dan kertas kosong di
        luar balon dua-duanya putih rata (spread <= 1.48 di 13 region).

    Yang dipangkas nyata, bukan hipotetis: r9 halaman uji punya cc=11 dan
    turun cover 0.900 -> 0.817; sebaran warna pita-luar isiannya jatuh dari
    19.27 ke 0.00 dan selisih median dari 13 ke 0 — bukti gumpalan yang
    dibuang itu memang art, bukan interior. 12 region lain tidak berubah
    (keepfrac >= 0.9996), jadi pangkas ini bukan pertukaran untung-rugi.

    Ia juga menutup jalur terburuk: `_interior_from_crop` punya jalur mundur
    `interior = binv` (SELURUH piksel terang crop, art dan semua) ketika flood
    fill gagal. Di halaman uji jalur itu bercover 0.82-0.94 — di bawah ambang
    0.97 sehingga LOLOS — dan pangkas menariknya kembali ke 0.66-0.83.

    Returns:
        Interior terpangkas, atau None kalau tidak ada komponen yang memuat
        tinta region ini sama sekali (isian yang tidak memuat teksnya sendiri
        tidak pernah benar; lebih baik jatuh ke jalur ink_mask).
    """
    if region.ink_mask is None:
        return interior
    ih, iw = interior.shape[:2]
    ink = np.zeros((ih, iw), np.uint8)
    x1, y1 = region.bbox[0], region.bbox[1]
    mh, mw = region.ink_mask.shape[:2]
    sy, sx = y1 - oy, x1 - ox
    dy, dx = max(sy, 0), max(sx, 0)
    hh, ww = min(mh + sy, ih) - dy, min(mw + sx, iw) - dx
    if hh <= 0 or ww <= 0:
        return interior
    ink[dy:dy + hh, dx:dx + ww] = region.ink_mask[dy - sy:dy - sy + hh,
                                                  dx - sx:dx - sx + ww]
    n, lab = cv2.connectedComponents((interior > 0).astype(np.uint8), 8)
    if n <= 2:
        return interior          # satu komponen: tidak ada yang bisa dipangkas
    hit = set(int(v) for v in np.unique(lab[(ink > 0) & (interior > 0)])) - {0}
    if not hit:
        return None
    return np.where(np.isin(lab, sorted(hit)), 255, 0).astype(np.uint8)


def _white_seed(binv: np.ndarray, near: tuple[int, int] | None = None) -> tuple[int, int]:
    """Piksel interior (255) terdekat ke `near` — titik awal flood fill.

    Default `near` = pusat crop. Untuk balon ganda pusat crop bisa jatuh di
    lobus sebelah, jadi pemanggil memberi titik tengah tinta region.

    Setelah inversi polaritas di _interior_from_crop, 'interior' ini bisa balon
    terang maupun gelap - nilai 255 selalu menandai interior."""
    hh, ww = binv.shape[:2]
    cx, cy = near if near is not None else (ww // 2, hh // 2)
    cx, cy = int(np.clip(cx, 0, ww - 1)), int(np.clip(cy, 0, hh - 1))
    ys, xs = np.nonzero(binv)
    if ys.size == 0:
        return cx, cy
    i = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
    return int(xs[i]), int(ys[i])


def _ink_center(region: Region, ox: int, oy: int) -> tuple[int, int]:
    """Titik tengah tinta region, dalam koordinat crop yang mulai di (ox, oy)."""
    x1, y1, x2, y2 = region.bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    ink = region.ink_mask
    if ink is not None and ink.any():
        ys, xs = np.nonzero(ink)
        cx, cy = x1 + int(xs.mean()), y1 + int(ys.mean())
    return cx - ox, cy - oy


def _stroke_px(glyph_h: float) -> int:
    """Perkiraan ketebalan garis balon dari tinggi glyph, dibatasi 1..4 px."""
    return int(np.clip(round(glyph_h * 0.06), 1, 4))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Tutup lubang TERTUTUP di dalam mask tanpa memekarkan tepi luarnya.

    Versi lama memakai findContours(RETR_EXTERNAL) + drawContours(FILLED).
    Kontur terluar daerah terisi ADALAH garis balon, jadi mengisinya penuh
    membuat interior menelan stroke hitam balon — probe baris lalu menganggap
    piksel di atas garis masih 'di dalam' dan teks dirender menembus balon.
    Flood fill dari luar hanya memulihkan lubang yang benar-benar tertutup.
    """
    # Pagar 1 px bernilai latar menjamin benih (0,0) ada di luar, walau daerah
    # terisi menyentuh tepi crop.
    inv = cv2.copyMakeBorder(
        cv2.bitwise_not(mask), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255
    )
    ff = np.zeros((inv.shape[0] + 2, inv.shape[1] + 2), np.uint8)
    cv2.floodFill(inv, ff, (0, 0), 0)
    return cv2.bitwise_or(mask, inv[1:-1, 1:-1])


def _ink_in_crop(region: Region, ox: int, oy: int,
                 shape: tuple[int, int]) -> np.ndarray:
    """ink_mask region dipetakan ke koordinat crop yang mulai di (ox, oy)."""
    hh, ww = shape[0], shape[1]
    out = np.zeros((hh, ww), np.uint8)
    ink = region.ink_mask
    if ink is None:
        return out
    x1, y1 = region.bbox[0] - ox, region.bbox[1] - oy
    mh, mw = ink.shape[:2]
    sy1, sx1 = max(y1, 0), max(x1, 0)
    sy2, sx2 = min(y1 + mh, hh), min(x1 + mw, ww)
    if sy2 > sy1 and sx2 > sx1:
        out[sy1:sy2, sx1:sx2] = ink[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1]
    return out


# Cincin latar tinta = tinta didilatasi sebesar ini, minus tinta itu sendiri.
# Piksel di cincin inilah LATAR tempat teks region berdiri, jadi kelas Otsu
# yang memuatnya adalah interior balon — apa pun kecerahan absolutnya.
_RING_K = 9
# Saat polaritas dibalik, tinta dan GARIS balon masuk kelas yang sama dengan
# interior kelabu, jadi flood fill bisa menembus garis. Piksel yang lebih gelap
# dari latar sebanyak ini dijadikan dinding. MAD dipakai supaya balon
# ber-screentone kasar tidak ikut terpotong; lantai 20 supaya balon rata tetap
# punya dinding walau MAD-nya 0.
_WALL_MAD, _WALL_MIN = 3.0, 20.0


def _polarity_ring(gray: np.ndarray, binv: np.ndarray,
                   ink: np.ndarray | None) -> tuple[bool, float, float]:
    """Apakah polaritas harus dibalik, plus (bg, mad) latar tinta region.

    Aturan lama absolut: `median(kelas mayoritas) < 128` -> balik. Itu benar
    hanya untuk dua ujung (balon putih / balon hitam) dan SALAH untuk balon
    ber-screentone kelabu, yang di manga ini justru dipakai untuk balon dalam
    panel gelap. Terukur di cacatbaru/jp_cacatnew1+2 (_cnpol.py): median kelas
    mayoritas 140/124/145 — dua di atas 128 jadi tidak dibalik — sedangkan
    latar tinta region berada di kelas TERANG hanya 0.000/0.014/0.025. Artinya
    yang diambil sebagai 'interior' adalah HALAMAN PUTIH DI LUAR balon, bukan
    rongga balonnya. Satu mask salah itu memunculkan dua cacat sekaligus:
    build_fill_mask mengisi lobus yang salah (tinta Jepang tidak pernah
    terhapus) dan typeset menata teks di sliver luar balon (terjemahan tercetak
    mungil di atas art).

    Penggantinya STRUKTURAL, bukan ambang: interior balon adalah kelas yang
    memuat CINCIN LATAR di sekeliling tinta region — alasan yang sama dengan
    _keep_ink_lobes ("isian hanya boleh mengisi rongga tempat tinta Jepangnya
    berada"). Tanpa tinta (pemanggil lama / ink_mask kosong) aturan absolut
    lama tetap dipakai sebagai jalur mundur.

    Terukur pada 12 region balon PUTIH bersih di hasilnew/jp_6 + jp_13
    (_cnband.py): keputusan aturan cincin sama dengan aturan lama di
    SEMUA-nya, dan cover interiornya identik. Jadi ini bukan pertukaran.
    """
    if ink is not None and ink.any():
        ring = (cv2.dilate(ink, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_RING_K, _RING_K))) > 0) & (ink == 0)
        if ring.any():
            vals = gray[ring].astype(np.int16)
            bg = float(np.median(vals))
            mad = float(np.median(np.abs(vals - bg)))
            return float((binv[ring] > 0).mean()) < 0.5, bg, mad
    vals = gray[binv > 0]
    if vals.size < binv.size - vals.size:
        vals = gray[binv == 0]
    return float(np.median(vals)) < 128, -1.0, -1.0


def _interior_from_crop(crop: np.ndarray, stroke: int,
                        seed: tuple[int, int] | None = None,
                        ink: np.ndarray | None = None) -> np.ndarray:
    """Interior balon dari satu crop: Otsu -> flood fill -> tambal -> kikis."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Balon gelap (teks putih di atas hitam) DAN balon kelabu ber-screentone:
    # Otsu mengasumsikan interior TERANG, jadi hasilnya kebalik - interior jadi
    # 0 dan teks jadi 255, lalu flood fill malah mengisi glyph. Polaritas
    # ditentukan dari cincin latar tinta region; lihat _polarity_ring().
    balik, bg, mad = _polarity_ring(gray, binv, ink)
    if balik:
        binv = cv2.bitwise_not(binv)
        # Setelah dibalik, GARIS balon ikut masuk kelas interior dan flood fill
        # bisa menembusnya lalu mengisi art gelap di luar balon. Terukur di
        # jp_cacatnew1 (_cnwall3.py): tanpa dinding 16.0% piksel interior jatuh
        # di art gelap (taji keluar ke garis rambut), dengan dinding 5.5%.
        if bg >= 0:
            lantai = bg - max(_WALL_MAD * mad, _WALL_MIN)
            binv = cv2.bitwise_and(binv, (gray > lantai).astype(np.uint8) * 255)
    hh, ww = binv.shape

    # Flood fill dari benih menangkap interior, bukan art di luar bubble — tapi
    # HANYA kalau benihnya jatuh di putih. Titik tengah balon justru sering kena
    # goresan huruf (pada halaman uji, piksel tengah bernilai 3), dan flood fill
    # dari sana mengisi goresan itu: interior terbaca 1.2% lalu balon dianggap
    # penuh, sehingga tiap baris gagal probe dan teks dipaksa ke ukuran minimum.
    ff = binv.copy()
    m = np.zeros((hh + 2, ww + 2), np.uint8)
    cv2.floodFill(ff, m, _white_seed(binv, seed), 128)
    interior = (ff == 128).astype(np.uint8) * 255

    # Hitung PIKSEL, bukan jumlah nilai: mask ini 0/255, jadi .sum() 255x lebih
    # besar dari cacah piksel dan ambang 5% ini diam-diam jadi 0.02% — jaring
    # pengaman yang tidak pernah menangkap apa pun.
    if int((interior > 0).sum()) < hh * ww * 0.05:
        interior = binv

    # Dinding gelap di atas MEMBUANG tinta region dari kelas interior, jadi
    # tiap glyph jadi lubang. _fill_holes hanya menambal lubang TERTUTUP, dan
    # glyph yang menempel di garis balon terbuka ke tepi — lubangnya bertahan.
    # Itu tidak boleh dibiarkan: build_fill_mask memakai interior ini untuk
    # MENGHAPUS tinta Jepang (erase_flat memakai fill_mask, bukan ink_mask),
    # jadi interior berlubang sebentuk glyph meninggalkan sisa tinta — persis
    # cacat yang sedang diperbaiki. Tinta region ada di DALAM balon menurut
    # definisi, jadi dipulihkan di sini. Terukur di _cnwall3.py: cakupan tinta
    # 0.92/0.89/0.71 -> 0.99/0.99/1.00, sementara kebocoran ke art gelap tetap
    # 5.5%/4.7%/6.6% (tanpa dinding: 16.0%/11.4%/8.2%).
    if ink is not None and ink.shape[:2] == interior.shape[:2]:
        interior = cv2.bitwise_or(interior, ink)
        binv = cv2.bitwise_or(binv, ink)

    # Tambal lubang bekas glyph. Mask ini dibangun dari gambar ASLI yang teksnya
    # masih ada, jadi flood fill mengalir MENGELILINGI tiap huruf dan menyisakan
    # stroke-nya sebagai lubang. Padahal teks itu dihapus sebelum typeset, jadi
    # lubangnya semu — tapi _row_free merata-rata dan ikut menghitungnya, membuat
    # balon terbaca cuma 56-71% bebas lalu baris gagal probe padahal ruangnya ada.
    interior = _fill_holes(interior)
    # Closing menyambung takik glyph yang MENEMPEL di garis balon — takik begitu
    # terbuka ke tepi, jadi _fill_holes tidak bisa menutupnya. Tapi closing juga
    # mengisi cekungan bentuk: di leher balon ganda kernel 7 px membuat interior
    # menonjol ke ATAS garis balon, dan teks lalu dirender menyentuh garis.
    # Jadi hasilnya dikurung ke piksel yang bukan garis: `binv` sudah membuang
    # semua piksel gelap, dan lubang glyph yang ikut terbuang sudah ditambal
    # `interior` di baris atas.
    closed = cv2.morphologyEx(
        interior, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    interior = cv2.bitwise_and(closed, cv2.bitwise_or(interior, binv))
    # Flood fill berhenti di piksel anti-aliased garis balon, jadi tepi interior
    # masih menyentuh garis. Kikis setebal stroke supaya baris terluar tidak
    # pernah menempel di garis balon.
    k = 2 * max(stroke, 1) + 1
    return cv2.erode(interior, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


# Jendela pencarian balon = bbox region dilebarkan sebanyak ini ke tiap sisi.
_DISCOVER_DILATE = 0.40
# Interior temuan harus mengisi sebagian jendela yang wajar: terlalu kecil =
# flood fill terjebak antar-huruf, terlalu besar = ia bocor keluar balon dan
# mengisi art. Dua-duanya lebih buruk dari persegi mentah.
_DISCOVER_FILL = (0.15, 0.85)


def _discover_bubble(
    img: np.ndarray, region: Region, stroke: int
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """Cari balon induk langsung dari gambar saat detector kehilangan kotaknya.

    Tanpa ini, `bubble_bbox is None` memberi mask persegi 255 penuh dan fit()
    bebas membesar sampai teks memotong garis balon. Hanya untuk text_bubble:
    text_free memang tidak punya balon, persegi bbox-nya sudah benar.
    """
    if region.det_class != "text_bubble":
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = region.bbox
    mx = int((x2 - x1) * _DISCOVER_DILATE)
    my = int((y2 - y1) * _DISCOVER_DILATE)
    wx1, wy1 = max(x1 - mx, 0), max(y1 - my, 0)
    wx2, wy2 = min(x2 + mx, w), min(y2 + my, h)
    crop = img[wy1:wy2, wx1:wx2]
    if crop.size == 0:
        return None

    interior = _interior_from_crop(crop, stroke, _ink_center(region, wx1, wy1),
                                   _ink_in_crop(region, wx1, wy1, crop.shape[:2]))
    lo, hi = _DISCOVER_FILL
    if not lo <= float((interior > 0).mean()) <= hi:
        return None
    ys, xs = np.nonzero(interior)
    bx1, by1 = wx1 + int(xs.min()), wy1 + int(ys.min())
    bx2, by2 = wx1 + int(xs.max()) + 1, wy1 + int(ys.max()) + 1
    # Balon yang benar memuat teksnya sendiri. Toleransi 10% karena bbox region
    # sudah ikut padding dan bisa menonjol beberapa piksel keluar interior.
    tx, ty = max((x2 - x1) // 10, 2), max((y2 - y1) // 10, 2)
    if bx1 > x1 + tx or by1 > y1 + ty or bx2 < x2 - tx or by2 < y2 - ty:
        return None
    return (bx1, by1, bx2, by2), interior[by1 - wy1:by2 - wy1, bx1 - wx1:bx2 - wx1]


def _bubble_interior(img: np.ndarray, region: Region) -> np.ndarray:
    """Mask area putih bubble untuk memandu layout baris teks."""
    stroke = _stroke_px(region.est_font_size)
    x1, y1, x2, y2 = region.bbox

    if region.bubble_bbox is None:
        found = _discover_bubble(img, region, stroke)
        if found is None:
            return np.full((y2 - y1, x2 - x1), 255, np.uint8)
        region.bubble_bbox, interior = found
        return interior

    bx1, by1, bx2, by2 = region.bubble_bbox
    crop = img[by1:by2, bx1:bx2]
    if crop.size == 0:
        return np.full((y2 - y1, x2 - x1), 255, np.uint8)
    return _interior_from_crop(crop, stroke, _ink_center(region, bx1, by1),
                               _ink_in_crop(region, bx1, by1, crop.shape[:2]))


def partition_shared_interiors(img: np.ndarray, regions: list[Region]) -> int:
    """Balon ganda: partisi interior GABUNGAN ke lobus milik tiap region.

    Membelah KOTAK balon (detect._partition_shared_bubbles) tidak memisahkan
    BENTUK lobusnya — tiap belahan persegi masih memuat sebagian lobus sebelah,
    jadi dua centroid jatuh berdekatan dan teksnya bertumpuk. Di sini interior
    dihitung untuk kotak balon ASLI, lalu tiap pikselnya diberikan ke region
    dengan tinta terdekat (Voronoi berbobot tinta): interior tiap region pasti
    disjoint dan bentuknya mengikuti lobus sungguhan, bukan potongan persegi.

    Harus dipanggil SETELAH build_region_mask semua region — butuh ink_mask.

    Returns:
        Jumlah region yang interiornya diganti.
    """
    from collections import defaultdict

    groups: dict[tuple[int, int, int, int], list[Region]] = defaultdict(list)
    for r in regions:
        if r.shared_bubble_bbox is not None and r.ink_mask is not None:
            groups[r.shared_bubble_bbox].append(r)
    return sum(_split_interior(img, bbox, grp)
               for bbox, grp in groups.items() if len(grp) >= 2)


def _eff_box_mask(r: Region) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """Kotak + mask interior EFEKTIF region — harus sama dengan yang dipakai
    typeset._region_box_mask(), termasuk jatuh ke persegi 255 penuh saat mask
    tidak cocok. Kalau keduanya berbeda, disjoin di sini memotong peta yang
    berbeda dari peta yang dipakai render dan irisannya tetap ada.
    """
    box = r.bubble_bbox or r.bbox
    bx1, by1, bx2, by2 = box
    bw, bh = max(bx2 - bx1, 0), max(by2 - by1, 0)
    m = r.bubble_mask
    if m is None or m.shape[:2] != (bh, bw):
        m = np.full((bh, bw), 255, np.uint8)
    return box, m


def disjoin_overlapping_interiors(img: np.ndarray, regions: list[Region]) -> int:
    """Interior balon BERTETANGGA yang beririsan -> tiap piksel jadi milik satu.

    partition_shared_interiors() hanya menangani kasus detector menyatukan dua
    lobus ke SATU kotak balon (`shared_bubble_bbox`). Kasus yang jauh lebih
    sering di halaman nyata: balon-balon berdekatan masing-masing dapat kotaknya
    sendiri — jadi fungsi itu tidak pernah jalan — tapi interiornya tetap
    beririsan ribuan piksel karena kotak persegi di sekitar balon bulat saling
    menabrak di sudut.

    Akibatnya bukan tumpang tindih, tapi GLYPH TERPOTONG: render_region menata
    teks memakai interior sendiri, lalu _clip_to_mask membuang piksel yang masuk
    interior region lain (forb_map). Terukur di halaman referensi: irisan 5994 px
    membuat 'OH, IS THIS THE SHIKO CLUB?' dirender jadi 'IS THE :KO 4B?' dan
    'I WAS COMPILING' jadi 'OMPILING'.

    Di sini irisan diselesaikan SEBELUM tata letak: piksel yang diklaim lebih
    dari satu region diberikan ke region dengan tinta terdekat (aturan Voronoi
    yang sama dengan _split_interior). Piksel yang cuma diklaim satu region tidak
    pernah disentuh — jadi fungsi ini hanya MEMBUANG piksel, tidak pernah
    menambah, dan tidak mungkin memunculkan cacat 'teks keluar balon' yang baru.

    Region terlindungi (SFX) ikut berlomba: tinta SFX menarik piksel di
    sekitarnya, jadi dialog tetap tidak ditulis menimpa SFX.

    Returns:
        Jumlah region yang interiornya menyusut.
    """
    items = [(r, *_eff_box_mask(r)) for r in regions]
    items = [(r, b, m) for r, b, m in items if m.size]
    if len(items) < 2:
        return 0

    return sum(_disjoin_group(img, [items[i] for i in grp])
               for grp in _overlap_groups([b for _, b, _ in items]))


def _overlap_groups(boxes: list[tuple[int, int, int, int]]) -> list[list[int]]:
    """Kelompok indeks yang kotaknya saling bersinggungan (union-find).

    Kotak beririsan adalah syarat PERLU bagi mask beririsan, bukan syarat cukup
    — cukup untuk menyaring kandidat murah; piksel yang ternyata tidak
    diperebutkan tetap tidak diubah di _disjoin_group().
    """
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


def _disjoin_group(img: np.ndarray, grp: list[tuple]) -> int:
    """Satu kelompok balon bersinggungan -> interiornya dibuat saling lepas.

    Returns:
        Jumlah region yang benar-benar kehilangan piksel.
    """
    h, w = img.shape[:2]
    wx1 = max(min(b[0] for _, b, _ in grp), 0)
    wy1 = max(min(b[1] for _, b, _ in grp), 0)
    wx2 = min(max(b[2] for _, b, _ in grp), w)
    wy2 = min(max(b[3] for _, b, _ in grp), h)
    ww, wh = wx2 - wx1, wy2 - wy1
    if ww <= 0 or wh <= 0:
        return 0

    # Mask tiap region dipindah ke jendela bersama supaya bisa dibandingkan
    # piksel-per-piksel.
    local: list[np.ndarray] = []
    for _, (bx1, by1, bx2, by2), m in grp:
        canvas = np.zeros((wh, ww), np.uint8)
        ty1, tx1 = by1 - wy1, bx1 - wx1
        sy1, sx1 = max(ty1, 0), max(tx1, 0)
        sy2, sx2 = min(ty1 + m.shape[0], wh), min(tx1 + m.shape[1], ww)
        if sy2 > sy1 and sx2 > sx1:
            canvas[sy1:sy2, sx1:sx2] = m[sy1 - ty1:sy2 - ty1, sx1 - tx1:sx2 - tx1]
        local.append(canvas)

    claims = np.zeros((wh, ww), np.uint16)
    for m in local:
        claims += (m > 0).astype(np.uint16)
    contested = claims >= 2
    if not contested.any():
        return 0

    owner = np.argmin(
        np.stack([_ink_distance(r, (wh, ww), wx1, wy1) for r, _, _ in grp]), axis=0
    )
    shrunk = 0
    for i, (r, _, _) in enumerate(grp):
        lost = contested & (owner != i) & (local[i] > 0)
        if not lost.any():
            continue
        keep = local[i].copy()
        keep[lost] = 0
        ys, xs = np.nonzero(keep)
        if ys.size == 0:
            continue                       # jangan pernah membuat region tanpa balon
        lx1, ly1 = wx1 + int(xs.min()), wy1 + int(ys.min())
        lx2, ly2 = wx1 + int(xs.max()) + 1, wy1 + int(ys.max()) + 1
        r.bubble_bbox = (lx1, ly1, lx2, ly2)
        r.bubble_mask = keep[ly1 - wy1:ly2 - wy1, lx1 - wx1:lx2 - wx1]
        shrunk += 1
    return shrunk


def _split_interior(img: np.ndarray, bbox: tuple[int, int, int, int],
                    grp: list[Region]) -> int:
    """Satu balon bersama -> satu interior per region, dijamin tidak beririsan."""
    bx1, by1, bx2, by2 = bbox
    crop = img[by1:by2, bx1:bx2]
    if crop.size == 0:
        return 0

    # Satu fill per region lalu digabung: kalau lobusnya menyatu semua benih
    # memberi hasil sama, kalau ada garis pemisah tiap lobus tetap terjaring.
    stroke = _stroke_px(max(r.est_font_size for r in grp))
    merged = np.zeros(crop.shape[:2], np.uint8)
    for r in grp:
        merged = np.maximum(
            merged, _interior_from_crop(crop, stroke, _ink_center(r, bx1, by1),
                                        _ink_in_crop(r, bx1, by1, crop.shape[:2]))
        )
    if not merged.any():
        return 0

    owner = np.argmin(
        np.stack([_ink_distance(r, merged.shape, bx1, by1) for r in grp]), axis=0
    )
    fixed = 0
    for i, r in enumerate(grp):
        lobe = np.where((owner == i) & (merged > 0), 255, 0).astype(np.uint8)
        ys, xs = np.nonzero(lobe)
        if ys.size == 0:
            continue                     # tanpa lobus: belahan persegi tetap dipakai
        lx1, ly1 = bx1 + int(xs.min()), by1 + int(ys.min())
        lx2, ly2 = bx1 + int(xs.max()) + 1, by1 + int(ys.max()) + 1
        r.bubble_bbox = (lx1, ly1, lx2, ly2)
        r.bubble_mask = lobe[ly1 - by1:ly2 - by1, lx1 - bx1:lx2 - bx1]
        fixed += 1
    return fixed


def _ink_distance(region: Region, shape: tuple[int, ...],
                  ox: int, oy: int) -> np.ndarray:
    """Jarak tiap piksel crop ke tinta region — dasar partisi Voronoi.

    distanceTransform mengukur jarak ke piksel BERNILAI 0, jadi tinta dipetakan
    ke 0 dan sisanya 255.
    """
    hh, ww = shape[0], shape[1]
    seed = np.full((hh, ww), 255, np.uint8)
    ink = region.ink_mask
    if ink is not None and ink.any():
        x1, y1 = region.bbox[0] - ox, region.bbox[1] - oy
        mh, mw = ink.shape[:2]
        sy1, sx1 = max(y1, 0), max(x1, 0)
        sy2, sx2 = min(y1 + mh, hh), min(x1 + mw, ww)
        if sy2 > sy1 and sx2 > sx1:
            sub = ink[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1]
            seed[sy1:sy2, sx1:sx2] = np.where(sub > 0, 0, 255)
    if seed.min() > 0:                   # tinta di luar crop: pakai pusat bbox
        cx, cy = _ink_center(region, ox, oy)
        seed[int(np.clip(cy, 0, hh - 1)), int(np.clip(cx, 0, ww - 1))] = 0
    return cv2.distanceTransform(seed, cv2.DIST_L2, 3)


def _paste(page: np.ndarray, mask: np.ndarray,
           box: tuple[int, int, int, int]) -> None:
    """Tempel mask lokal ke kanvas halaman. `box` = kotak asal mask itu sendiri.

    Dipisah karena salah kotak di sini tidak pernah kelihatan sebagai error:
    bubble_mask hidup di koordinat bubble_bbox, bukan bbox, dan untuk r8 halaman
    referensi kedua kotak itu beda 29 px. Mask yang bergeser menghasilkan angka
    yang rapi tapi salah tempat.
    """
    h, w = page.shape[:2]
    x1, y1 = box[0], box[1]
    mh, mw = mask.shape[:2]
    sy1, sx1 = max(y1, 0), max(x1, 0)
    sy2, sx2 = min(y1 + mh, h), min(x1 + mw, w)
    if sy2 <= sy1 or sx2 <= sx1:
        return
    sub = mask[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1]
    page[sy1:sy2, sx1:sx2] = np.maximum(page[sy1:sy2, sx1:sx2], sub)


# Ambang gelap untuk "ini garis balon". Sengaja jauh di bawah 128 (ambang
# interior di _bubble_interior): yang dicari cuma tinta pekat, bukan raster abu.
_LINE_DARK = 110
# Pita pencarian garis: dari tepi interior sampai stroke + segini px keluar.
# Harus stroke-aware — _interior_from_crop mengikis interior sebesar stroke,
# jadi garisnya duduk stroke..2*stroke px di luar. Pita tetap 4 px melaporkan
# NOL piksel garis di balon berstroke 4 px: pitanya sendiri belum sampai.
_LINE_BAND = 4
# Komponen gelap di pita dihitung garis kalau bentang terpanjangnya (lebar atau
# tinggi kotak pembatasnya) minimal sekian kali stroke. Garis balon membentang
# jauh menyusuri tepi; coretan glyph yang kebetulan menyeberang pita ringkas.
# Dipakai bentang, BUKAN luas relatif: pada balon sempit garisnya terputus jadi
# beberapa potong, dan ambang "seperempat komponen terbesar" ikut membuang
# potongan yang sah.
_LINE_MIN_SPAN = 8


def bubble_outline_guard(img: np.ndarray, regions: list[Region]) -> np.ndarray:
    """Piksel GARIS balon yang tidak boleh ikut terhapus. Biner 0/255.

    Kenapa perlu: _adaptive_dilate memekarkan ink_mask dengan kernel sampai
    31 px, dan teks Jepang vertikal di balon sempit duduk cuma beberapa piksel
    dari garisnya. Dilasi itu menyeberang garis, erase menghapus yang tersentuh,
    dan garis balon jadi putus-putus. Terukur 267 px garis termakan di halaman
    referensi (r7 79, r8 136, r11 36, r12 16).

    Kenapa bukan "kurung erase ke dalam interior" — obat yang lebih sederhana
    itu sudah diuji dan salah: interior bukan selubung yang bisa dipercaya.
    Glyph yang MENEMPEL di garis membuat takik yang terbuka ke tepi, jadi flood
    fill tak bisa mengelilinginya dan _fill_holes tak bisa menambalnya (hanya
    lubang tertutup). Hasilnya 79/148/36/187 px tinta Jepang di r7/r8/r11/r12
    ikut selamat — bukan halo, tapi glyph utuh ('すか' masih terbaca).

    Yang dilindungi di sini justru GARISNYA saja, jadi _halo_pass tetap bekerja
    penuh di dalam balon dan alasannya (penyebab nomor satu ghost outline) tidak
    dilanggar.
    """
    h, w = img.shape[:2]
    dark = img.mean(2) < _LINE_DARK
    guard = np.zeros((h, w), np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for r in regions:
        if r.bubble_mask is None:
            continue
        box, bm = _eff_box_mask(r)
        if bm.size == 0 or bm.min() == 255:
            continue          # persegi 255 penuh: bukan balon, tak ada garis
        inner = np.zeros((h, w), np.uint8)
        _paste(inner, (bm > 0).astype(np.uint8), box)
        if not inner.any():
            continue
        stroke = _stroke_px(r.est_font_size or 20)
        band = cv2.dilate(inner, k3, iterations=stroke + _LINE_BAND) - inner
        sel = (dark & (band > 0)).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(sel, 8)
        if n <= 1:
            continue
        span = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT])
        keep = 1 + np.flatnonzero(span >= _LINE_MIN_SPAN * max(stroke, 1))
        if keep.size:
            guard[np.isin(lab, keep)] = 255
    return guard


def protect_bubble_outline(img: np.ndarray, regions: list[Region]) -> int:
    """Kurangkan garis balon dari SETIAP ink_mask. Return piksel yang dilepas.

    Harus di ink_mask, bukan cuma di erase_mask hasil compose_page_mask().
    erase_page() menghapus dari `r.ink_mask` per region (erase.py:86 dan 113);
    erase_mask halaman hanya dipakai untuk assert SFX dan dump debug. Versi
    pertama perbaikan ini mengurangkan penjaga di compose_page_mask saja, dan
    hasilnya tepat seperti yang diukur: penjaga bersih dari erase_mask (irisan
    0 px) sementara garis yang hilang di plat bersih tidak bergerak satu piksel
    pun — 326 px tetap termakan karena erase tidak pernah membaca mask itu.

    Dipanggil setelah partition_shared_interiors + disjoin_overlapping_interiors,
    karena penjaganya dihitung dari bubble_mask yang sudah final.
    """
    guard = bubble_outline_guard(img, regions)
    if not guard.any():
        return 0
    h, w = img.shape[:2]
    freed = 0
    for r in regions:
        if r.ink_mask is None:
            continue
        x1, y1 = r.bbox[0], r.bbox[1]
        mh, mw = r.ink_mask.shape[:2]
        sy1, sx1 = max(y1, 0), max(x1, 0)
        sy2, sx2 = min(y1 + mh, h), min(x1 + mw, w)
        if sy2 <= sy1 or sx2 <= sx1:
            continue
        sub = r.ink_mask[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1]
        hit = (sub > 0) & (guard[sy1:sy2, sx1:sx2] > 0)
        if hit.any():
            freed += int(hit.sum())
            sub[hit] = 0
    return freed


def compose_page_mask(
    img: np.ndarray, regions: list[Region]
) -> tuple[np.ndarray, np.ndarray]:
    """Gabung semua ink mask jadi satu mask halaman, LALU kecualikan SFX.

    Returns:
        (erase_mask, protected_mask) — dua-duanya biner 0/255 skala halaman.

    Ini titik paling kritis di seluruh pipeline. Mask teks dan penyelamat
    furigana AKAN ikut menandai SFX. Kalau exclusion tidak jalan sebelum
    erase, pipeline menghapus persis apa yang diminta untuk dijaga.
    """
    h, w = img.shape[:2]
    page = np.zeros((h, w), np.uint8)
    protected = np.zeros((h, w), np.uint8)

    for r in regions:
        if r.ink_mask is None:
            continue
        x1, y1, x2, y2 = r.bbox
        mh, mw = r.ink_mask.shape[:2]
        x2, y2 = min(x2, x1 + mw), min(y2, y1 + mh)
        sub = r.ink_mask[: y2 - y1, : x2 - x1]
        target = protected if r.is_protected else page
        target[y1:y2, x1:x2] = np.maximum(target[y1:y2, x1:x2], sub)

    # SFX menang mutlak: lebarkan sedikit lalu kurangkan dari mask hapus.
    if protected.any():
        guard = cv2.dilate(
            protected, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1
        )
        page = cv2.bitwise_and(page, cv2.bitwise_not(guard))

    # Garis balon juga menang: lihat bubble_outline_guard(). Di sini hanya
    # menjaga konsistensi dump/assert — pengurangan yang benar-benar berpengaruh
    # ke hasil dilakukan protect_bubble_outline() pada ink_mask, karena
    # erase_page() membaca ink_mask per region, bukan mask halaman ini.
    outline = bubble_outline_guard(img, regions)
    if outline.any():
        page = cv2.bitwise_and(page, cv2.bitwise_not(outline))

    return ((page > 0).astype(np.uint8)) * 255, ((protected > 0).astype(np.uint8)) * 255


def release() -> None:
    global _CTD
    _CTD = None

