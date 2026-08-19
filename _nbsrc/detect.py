%%writefile /content/mangatl/detect.py

"""Deteksi region: RT-DETR-v2 (bubble/text_bubble/text_free) via ONNX Runtime."""

from __future__ import annotations

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # ORT gagal terpasang: beri pesan jelas saat dipakai,
    ort = None       # jangan matikan seluruh notebook saat import.

from config import ID2LABEL, SETTINGS, WEIGHTS, Region, ort_session

_SESSION = None

# Dua lobus balon ganda saling tumpang tindih banyak tapi tidak ada yang MEMUAT
# yang lain; dua deteksi atas balon yang sama containment-nya ~1.0. 0.80 memisah
# keduanya. Lihat _nms().
_BUBBLE_CONTAIN = 0.80

# Kotak teks kerap menonjol keluar kotak balon, dan pada balon ganda tonjolannya
# besar. 0.80 menolak pasangan yang jelas benar -> region kehilangan induk ->
# mask interior jadi persegi mentah dan teks keluar garis. Lihat assign_bubbles().
_PARENT_OVERLAP = 0.65

# SATU blok teks yang terdeteksi DUA KALI: kotak kecil hampir seluruhnya termuat
# di kotak besar. NMS kelompok teks memakai IoU murni (contain_thresh=0.0), dan
# IoU pasangan bersarang bisa RENDAH walau containment-nya ~1.0 — pada halaman
# hitomi_3740721_015: containment 0.974 tapi IoU cuma 0.280, di bawah det_iou
# 0.45, jadi keduanya lolos. Lihat drop_nested_duplicates().
#
# 0.80 dipilih karena ada jurang lebar yang terukur di antara dua populasi:
# containment teks-vs-teks pada halaman bersih (debug/jp_6 8 region, jepang_002
# 13 region, jp_13 4 region) tertinggi cuma 0.33, sedangkan duplikat sejati
# 0.974. Angka yang sama dengan _BUBBLE_CONTAIN, dengan alasan yang sama.
_DUP_CONTAIN = 0.80


# Toleransi cakupan saat memilih induk. Aturan "balon TERKECIL yang memuat
# mayoritas teks" salah kalau balonnya BERLOBUS: detector mengeluarkan satu kotak
# untuk seluruh balon plus satu per lobus, dan kotak lobus selalu lebih kecil.
# Terukur di hitomi_3740721_015 — teks (832,130,1027,405):
#   lobus kiri  (800,117,964,440)  cakupan 0.677  area 52972  <- lama menang
#   balon penuh (800,96,1046,442)  cakupan 1.000  area 85116  <- yang benar
# Yang menang dulu memuat cuma 2/3 teks, jadi lobus kanan tidak masuk interior:
# `そうならないように．．．` tidak pernah terhapus dan `find_residue` menandainya.
#
# Jadi cakupan dipakai LEBIH DULU, area cuma pemutus seri. Serinya diberi pita
# 0.05 supaya kotak lobus yang benar tidak kalah hanya karena teksnya menonjol
# beberapa piksel keluar lobus: 0.97 lawan 1.00 tetap seri (lobus menang, dan
# itu yang diinginkan pada balon ganda), sedangkan 0.677 lawan 1.00 tidak.
_PARENT_SLACK = 0.05


def get_session():
    """Muat detector sekali, pakai ulang. GPU kalau ada, CPU kalau tidak."""
    global _SESSION
    if _SESSION is None:
        if ort is None:
            raise RuntimeError(
                "onnxruntime tidak terpasang — jalankan ulang sel install (sel 3)."
            )
        path = WEIGHTS / "detector.onnx"
        if not path.exists():
            raise FileNotFoundError(f"Weight detector tidak ada: {path}")
        _SESSION = ort_session(path, "detector")
    return _SESSION


def _preprocess(img: np.ndarray, size: int) -> np.ndarray:
    """Resize langsung ke size x size — RT-DETR tidak pakai letterbox."""
    import cv2

    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    chw = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw[None]


def _input_names(sess) -> list[str]:
    return [i.name for i in sess.get_inputs()]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float,
         contain_thresh: float = 0.0) -> list[int]:
    """NMS greedy. Dipakai untuk menggabung hasil full-page + tile.

    `contain_thresh > 0` menambah syarat kedua sebelum suppress: kotak yang
    kalah harus BENAR-BENAR termuat di kotak pemenang (inter/area_kecil).
    Tanpa itu, satu lobus balon ganda ditekan hanya karena IoU-nya tinggi, dua
    region teks jatuh ke satu kotak balon, dan terjemahannya bertumpuk.
    Default 0.0 = perilaku IoU murni (dipakai kelompok teks).
    """
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        contain = inter / (np.minimum(areas[i], areas[rest]) + 1e-9)
        order = rest[(iou <= iou_thresh) | (contain < contain_thresh)]
    return keep


def _decode(outputs: list[np.ndarray], w: int, h: int, conf: float) -> tuple[np.ndarray, ...]:
    """Decode keluaran detector.

    Export resmi ogkalu sudah memuat postprocessor RT-DETR: keluarannya
    (labels int64, boxes xyxy ABSOLUT, scores) dan boxes sudah diskalakan oleh
    `orig_target_sizes`. Terverifikasi langsung terhadap detector.onnx —
    3 keluaran, 300 query, skor sudah urut menurun.

    Cabang kedua menangani export mentah (logits, boxes cxcywh ternormalisasi)
    yang dipakai sebagian mirror. RT-DETR memakai sigmoid per-kelas, bukan
    softmax, jadi tidak ada slot background yang harus dibuang.
    """
    empty = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int))

    if len(outputs) >= 3:
        labels, boxes, scores = outputs[0], outputs[1], outputs[2]
        labels, boxes, scores = np.squeeze(labels, 0), boxes[0], np.squeeze(scores, 0)
        keep = scores >= conf
        if not keep.any():
            return empty
        return (
            boxes[keep].astype(np.float32),
            scores[keep].astype(np.float32),
            labels[keep].astype(int),
        )

    logits, boxes = outputs[0], outputs[1]
    if logits.ndim == 3:
        logits, boxes = logits[0], boxes[0]
    probs = 1.0 / (1.0 + np.exp(-logits))          # sigmoid, bukan softmax
    cls_ids = probs.argmax(axis=1)
    scores = probs.max(axis=1)

    keep = scores >= conf
    if not keep.any():
        return empty

    boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xyxy = np.stack(
        [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
        axis=1,
    ).astype(np.float32)
    return xyxy, scores.astype(np.float32), cls_ids.astype(int)


def _run_once(img: np.ndarray, conf: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sess = get_session()
    h, w = img.shape[:2]
    feed = {"images": _preprocess(img, SETTINGS.det_size)}
    # (w, h), BUKAN (h, w): terverifikasi — urutan terbalik menghasilkan box
    # yang meluber melewati tepi kanan halaman.
    if "orig_target_sizes" in _input_names(sess):
        feed["orig_target_sizes"] = np.array([[w, h]], dtype=np.int64)
    outs = sess.run(None, feed)
    return _decode(outs, w, h, conf)


def _tiled(img: np.ndarray, conf: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pass 2x2 dengan tumpang tindih — resize 640 melewatkan teks kecil."""
    h, w = img.shape[:2]
    ov = 0.15
    th, tw = int(h * (0.5 + ov)), int(w * (0.5 + ov))
    all_b, all_s, all_c = [], [], []
    for oy in (0, h - th):
        for ox in (0, w - tw):
            tile = img[oy : oy + th, ox : ox + tw]
            b, s, c = _run_once(tile, conf)
            if not len(b):
                continue
            b, s, c = _drop_tile_edge(b, s, c, ox, oy, tw, th, w, h)
            if len(b):
                all_b.append(b + np.array([ox, oy, ox, oy], dtype=np.float32))
                all_s.append(s)
                all_c.append(c)
    if not all_b:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    return np.concatenate(all_b), np.concatenate(all_s), np.concatenate(all_c)


def _drop_tile_edge(
    b: np.ndarray, s: np.ndarray, c: np.ndarray,
    ox: int, oy: int, tw: int, th: int, w: int, h: int, margin: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Buang box yang menempel di garis potong tile.

    Box seperti itu terpotong, dan skornya sering LEBIH TINGGI daripada versi
    utuh dari full-page pass — jadi NMS justru memilih yang cacat. Terukur di
    halaman referensi: balon 415..560 px jadi 448..562 px karena tile mulai di
    x=448. Tepi yang berimpit dengan tepi halaman tidak dihitung: di sana box
    memang boleh mentok.
    """
    edge = np.zeros(len(b), dtype=bool)
    if ox > 0:
        edge |= b[:, 0] <= margin
    if oy > 0:
        edge |= b[:, 1] <= margin
    if ox + tw < w:
        edge |= b[:, 2] >= tw - margin
    if oy + th < h:
        edge |= b[:, 3] >= th - margin
    return b[~edge], s[~edge], c[~edge]


def detect(img: np.ndarray, conf: float | None = None) -> tuple[list[Region], list[tuple]]:
    """Deteksi region teks + bubble.

    Returns:
        (regions, bubbles) — regions hanya text_bubble/text_free; bubbles
        adalah bbox bubble kosong untuk mencari induk tiap region.
    """
    conf = SETTINGS.det_conf if conf is None else conf
    h, w = img.shape[:2]

    b1, s1, c1 = _run_once(img, conf)
    if SETTINGS.tiled_pass and max(h, w) > 900:
        b2, s2, c2 = _tiled(img, conf)
        boxes = np.concatenate([b1, b2]) if len(b2) else b1
        scores = np.concatenate([s1, s2]) if len(s2) else s1
        cls_ids = np.concatenate([c1, c2]) if len(c2) else c1
    else:
        boxes, scores, cls_ids = b1, s1, c1

    if len(boxes) == 0:
        return [], []

    boxes[:, 0::2] = boxes[:, 0::2].clip(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, h)

    regions: list[Region] = []
    bubbles: list[tuple[int, int, int, int]] = []
    idx = 0
    # NMS dua kelompok, bukan per-kelas. `bubble` memang tumpang tindih dengan
    # teks di dalamnya jadi harus dipisah, tapi text_bubble dan text_free adalah
    # teks fisik yang sama — NMS per-kelas membiarkan keduanya lolos dan region
    # yang sama masuk pipeline dua kali (terukur di halaman referensi: balon '!'
    # muncul sebagai #3 dan #4).
    is_bubble = cls_ids == 0
    for group, iou_t, contain_t in (
        (is_bubble, SETTINGS.det_iou_bubble, _BUBBLE_CONTAIN),
        (~is_bubble, SETTINGS.det_iou, 0.0),
    ):
        if not group.any():
            continue
        keep = _nms(boxes[group], scores[group], iou_t, contain_thresh=contain_t)
        kb, ks, kc = boxes[group][keep], scores[group][keep], cls_ids[group][keep]
        for box, sc, cid in zip(kb, ks, kc):
            xyxy = tuple(int(v) for v in box)
            if xyxy[2] - xyxy[0] < 4 or xyxy[3] - xyxy[1] < 4:
                continue
            if int(cid) == 0:
                bubbles.append(xyxy)
            else:
                regions.append(
                    Region(
                        idx=idx,
                        bbox=xyxy,
                        det_class=ID2LABEL.get(int(cid), "text_bubble"),
                        det_conf=float(sc),
                    )
                )
                idx += 1

    drop_nested_duplicates(regions)
    regions = assign_bubbles(regions, bubbles)
    return sort_reading_order(regions), bubbles


def drop_nested_duplicates(regions: list[Region]) -> int:
    """Buang kotak teks yang hampir seluruhnya termuat di kotak teks LAIN.

    Detector kadang mengeluarkan dua kotak untuk satu blok teks: satu menutup
    seluruh blok, satu lagi hanya sebagian kolomnya. NMS kelompok teks tidak
    menangkapnya karena dipanggil dengan `contain_thresh=0.0` — IoU pasangan
    bersarang rendah (0.280 pada halaman hitomi_3740721_015) walau containment-nya
    0.974. Keduanya lolos, dan akibatnya BERTUMPUK tiga kali:

    1. Balon berlobus (satu balon, garis luarnya berpinggang) membuat detector
       mengeluarkan TIGA kotak balon: satu untuk seluruh balon plus satu per
       lobus. Terukur di halaman itu: (800,96,1046,442) untuk balonnya, lalu
       (800,117,964,440) dan (934,89,1045,353) untuk kedua lobusnya. Kedua kotak
       teks duplikat lalu memilih induk yang BERBEDA — yang kecil dapat lobus
       kanan, yang besar dapat lobus kiri — jadi masing-masing mengukur
       `fill_color`-nya sendiri: kelabu 152 di kanan (lobus itu duduk di atas
       halaman putih) lawan 120 di kiri (di atas art gelap). Interior keduanya
       beririsan, `disjoin_overlapping_interiors` memotongnya jadi saling lepas,
       dan potongan itulah jahitan bergerigi vertikal yang terlihat di tengah
       balon — dua kelabu berbeda bersebelahan, dengan teks hitam di sebelah
       teks putih karena warna huruf pun dihitung per region.
    2. OCR membaca blok yang sama dua kali; teks yang kecil jadi AWALAN teks yang
       besar (`そうならないように．．．` di dalam `そうならないように．．．生涯に…`).
    3. Keduanya diterjemahkan dan DITATA, jadi kalimat yang sama tercetak dua kali
       di dalam satu balon — melanggar kontrak "tidak boleh saling timpa".

    Yang bertahan adalah kotak yang LEBIH BESAR, dan kotaknya dilebarkan ke gabungan
    keduanya. Melebarkan itu bukan hiasan: kotak kecil bisa menonjol beberapa piksel
    keluar kotak besar (di halaman itu y1 130 lawan 135), dan tinta di sliver itu
    tidak akan tercakup mask siapa pun kalau kotaknya tidak dilebarkan — `find_residue`
    akan menandainya sebagai sisa tinta.

    Bersarang BUKAN balon ganda: lobus balon ganda BERJAJAR, jadi containment-nya
    rendah (tertinggi 0.33 pada halaman bersih yang diukur). Tapi satu kotak besar
    yang benar-benar memuat DUA lobus juga membuat dua kotak kecil bersarang di
    dalamnya — dan untuk kasus itu `_partition_shared_bubbles` +
    `partition_shared_interiors` justru sudah benar. Jadi penyingkiran di sini
    hanya jalan kalau kotak besar itu memuat TEPAT SATU kotak bersarang.

    Harus dipanggil SEBELUM assign_bubbles(): di sanalah kedua duplikat diberi
    induk yang berbeda, dan seluruh mask dibangun sesudahnya. Dedupe setelah OCR
    — walau di sanalah bukti teksnya ada, karena teks kecil persis awalan teks
    besar — tiba terlambat untuk mencegah jahitan dua warna itu.

    Returns:
        Jumlah region yang dibuang.
    """
    n = len(regions)
    if n < 2:
        return 0
    areas = [max(r.width, 0) * max(r.height, 0) for r in regions]

    # nested[j] = daftar i yang bersarang di j. Dikumpulkan lengkap dulu, supaya
    # syarat "tepat satu" bisa diuji sebelum ada yang dibuang.
    nested: dict[int, list[int]] = {}
    for i in range(n):
        for j in range(n):
            if i == j or areas[i] <= 0 or areas[j] <= 0 or areas[i] > areas[j]:
                continue
            ax1, ay1, ax2, ay2 = regions[i].bbox
            bx1, by1, bx2, by2 = regions[j].bbox
            iw = min(ax2, bx2) - max(ax1, bx1)
            ih = min(ay2, by2) - max(ay1, by1)
            if iw <= 0 or ih <= 0:
                continue
            if (iw * ih) / areas[i] >= _DUP_CONTAIN:
                nested.setdefault(j, []).append(i)

    drop: set[int] = set()
    for j, inner in nested.items():
        if len(inner) != 1:
            continue          # dua lobus dalam satu kotak = balon ganda, jangan disentuh
        i = inner[0]
        if i in drop or j in drop:
            continue
        drop.add(i)
        ax1, ay1, ax2, ay2 = regions[i].bbox
        bx1, by1, bx2, by2 = regions[j].bbox
        regions[j].bbox = (min(ax1, bx1), min(ay1, by1),
                           max(ax2, bx2), max(ay2, by2))

    if not drop:
        return 0
    kept = [r for k, r in enumerate(regions) if k not in drop]
    regions[:] = kept
    return len(drop)


def assign_bubbles(regions: list[Region], bubbles: list[tuple]) -> list[Region]:
    """Cari bubble induk tiap region: yang memuat teksnya paling utuh, lalu terkecil.

    Pakai overlap, bukan containment ketat. Kotak teks kerap menonjol beberapa
    piksel keluar dari kotak balon — terukur di halaman referensi: teks mulai di
    x=1041 sedangkan balonnya di x=1046, jadi containment ketat menolak pasangan
    yang jelas benar. Akibatnya typeset kehilangan mask interior balon, menata
    teks di kotak mentah, dan memberi stroke putih yang tidak seharusnya ada.

    Urutan pemilihannya CAKUPAN dulu, baru area — lihat _PARENT_SLACK. "Terkecil
    yang memuat mayoritas" saja memilih kotak LOBUS di atas kotak balon penuh
    pada balon berlobus, dan lobus yang salah pilih itu meninggalkan tinta Jepang
    di lobus sebelahnya tanpa terhapus.
    """
    for r in regions:
        rx1, ry1, rx2, ry2 = r.bbox
        r_area = max((rx2 - rx1) * (ry2 - ry1), 1)
        cands: list[tuple[float, int, tuple]] = []
        for b in bubbles:
            bx1, by1, bx2, by2 = b
            iw = min(rx2, bx2) - max(rx1, bx1)
            ih = min(ry2, by2) - max(ry1, by1)
            if iw <= 0 or ih <= 0:
                continue
            cover = (iw * ih) / r_area
            if cover < _PARENT_OVERLAP:  # mayoritas teks di dalam balon
                continue
            cands.append((cover, (bx2 - bx1) * (by2 - by1), b))
        best = None
        if cands:
            top = max(c[0] for c in cands)
            best = min((c for c in cands if c[0] >= top - _PARENT_SLACK),
                       key=lambda c: c[1])[2]
        r.bubble_bbox = best
        if best is not None and r.det_class == "text_free":
            r.det_class = "text_bubble"  # ternyata di dalam bubble
    _partition_shared_bubbles(regions)
    return regions


def sort_reading_order(regions: list[Region]) -> list[Region]:
    """Urutan baca manga: kanan->kiri, atas->bawah, dikelompokkan per baris."""
    if not regions:
        return regions
    heights = [r.height for r in regions] or [20]
    band = max(int(np.median(heights) * 1.3), 20)
    rows: list[list[Region]] = []
    for r in sorted(regions, key=lambda x: x.bbox[1]):
        placed = False
        for row in rows:
            if abs(row[0].bbox[1] - r.bbox[1]) < band:
                row.append(r)
                placed = True
                break
        if not placed:
            rows.append([r])
    ordered: list[Region] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda x: -x.bbox[0]))
    for i, r in enumerate(ordered):
        r.idx = i
    return ordered


def release() -> None:
    """Bebaskan sesi ONNX — RAM Colab cuma ~12.7 GB."""
    global _SESSION
    _SESSION = None


def _partition_shared_bubbles(regions: list[Region]) -> None:
    """Double bubble: SATU kotak balon berisi >= 2 region -> bagi balon per region.

    RT-DETR kadang mengeluarkan satu kotak bubble untuk dua balon yang menyatu
    (double bubble). Tanpa pemecahan, kedua region memakai mask gabungan
    (figura-8) dan dua terjemahan ditumpuk di tengahnya — saling timpa.

    Belahan di sini hanya KASAR: potongan persegi tidak sejajar bentuk lobus,
    jadi tiap belahan masih memuat sebagian lobus sebelah. Kotak asli disimpan
    di `shared_bubble_bbox` supaya textmask.partition_shared_interiors() bisa
    memartisi piksel interiornya mengikuti bentuk lobus sungguhan; belahan
    persegi ini tinggal jadi fallback kalau partisi itu gagal.
    """
    from collections import defaultdict

    groups: dict[tuple[int, int, int, int], list[Region]] = defaultdict(list)
    for r in regions:
        if r.bubble_bbox is not None:
            groups[r.bubble_bbox].append(r)

    def _cx(r: Region) -> float:
        return (r.bbox[0] + r.bbox[2]) / 2

    def _cy(r: Region) -> float:
        return (r.bbox[1] + r.bbox[3]) / 2

    for bbox, grp in groups.items():
        if len(grp) < 2:
            continue
        bx1, by1, bx2, by2 = bbox
        for r in grp:
            r.shared_bubble_bbox = bbox
        by_x = sorted(grp, key=_cx)
        by_y = sorted(grp, key=_cy)
        # Belah sepanjang sumbu dengan sebaran centroid terbesar.
        horizontal = _cx(by_x[-1]) - _cx(by_x[0]) >= _cy(by_y[-1]) - _cy(by_y[0])

        order = by_x if horizontal else by_y
        prev = bx1 if horizontal else by1
        for i, r in enumerate(order):
            if i == len(order) - 1:
                r.bubble_bbox = ((prev, by1, bx2, by2) if horizontal
                                 else (bx1, prev, bx2, by2))
                continue
            nxt = order[i + 1]
            if horizontal:
                gap = ((r.bbox[2] + nxt.bbox[0]) // 2 if r.bbox[2] < nxt.bbox[0]
                       else int((_cx(r) + _cx(nxt)) / 2))
                cut = max(gap, prev + 4)
                r.bubble_bbox = (prev, by1, cut, by2)
            else:
                gap = ((r.bbox[3] + nxt.bbox[1]) // 2 if r.bbox[3] < nxt.bbox[1]
                       else int((_cy(r) + _cy(nxt)) / 2))
                cut = max(gap, prev + 4)
                r.bubble_bbox = (bx1, prev, bx2, cut)
            prev = cut

