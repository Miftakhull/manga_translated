%%writefile /content/mangatl/inpaint.py

"""LaMa inpainting — generator FFC di-vendor karena checkpoint-nya state_dict mentah.

`lama_large_512px.ckpt` menyimpan bobot di bawah key `gen_state_dict`, BUKAN
TorchScript. Jadi `simple-lama-inpainting` (yang pakai `torch.jit.load`) tidak
kompatibel dan generator harus dibangun sendiri.

Kalau bobot gagal dimuat, modul turun ke `cv2.inpaint` tanpa crash — jalur
flat-fill sudah menangani 70-85% region jadi degradasinya kecil.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn

from config import SETTINGS, WEIGHTS, note

# ---------------------------------------------------------------- FFC blocks


class FourierUnit(nn.Module):
    """Konvolusi 1x1 di domain frekuensi — inti receptive field global LaMa."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 1):
        super().__init__()
        self.groups = groups
        self.conv_layer = nn.Conv2d(
            in_channels * 2, out_channels * 2, 1, 1, 0, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x.float(), dim=fft_dim, norm="ortho")
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1) + ffted.size()[3:])

        ffted = self.relu(self.bn(self.conv_layer(ffted)))

        ffted = (
            ffted.view((batch, -1, 2) + ffted.size()[2:])
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        return torch.fft.irfftn(ffted, s=x.shape[-2:], dim=fft_dim, norm="ortho")


class SpectralTransform(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1,
        groups: int = 1, enable_lfu: bool = False,
    ):
        super().__init__()
        self.enable_lfu = enable_lfu
        self.downsample = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        if enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, 1, groups=groups, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self.downsample(x))
        return self.conv2(x + self.fu(x))


class FFC(nn.Module):
    """Bagi channel jadi cabang lokal (spasial) dan global (spektral)."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int,
        ratio_gin: float, ratio_gout: float, stride: int = 1, padding: int = 0,
        dilation: int = 1, groups: int = 1, bias: bool = False,
        enable_lfu: bool = False, padding_type: str = "reflect",
    ):
        super().__init__()
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        self.ratio_gin, self.ratio_gout = ratio_gin, ratio_gout
        self.global_in_num = in_cg

        def conv(ci: int, co: int) -> nn.Module:
            if ci == 0 or co == 0:
                return nn.Identity()
            return nn.Conv2d(
                ci, co, kernel_size, stride, padding, dilation, groups, bias,
                padding_mode=padding_type,
            )

        self.convl2l = conv(in_cl, out_cl)
        self.convl2g = conv(in_cl, out_cg)
        self.convg2l = conv(in_cg, out_cl)
        self.convg2g = (
            nn.Identity()
            if in_cg == 0 or out_cg == 0
            else SpectralTransform(
                in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu
            )
        )

    def forward(self, x):
        x_l, x_g = x if isinstance(x, tuple) else (x, 0)
        out_xl, out_xg = 0, 0
        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)
        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int,
        ratio_gin: float, ratio_gout: float, stride: int = 1, padding: int = 0,
        dilation: int = 1, groups: int = 1, bias: bool = False,
        norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
        padding_type: str = "reflect", enable_lfu: bool = False,
    ):
        super().__init__()
        self.ffc = FFC(
            in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride,
            padding, dilation, groups, bias, enable_lfu, padding_type,
        )
        gc = int(out_channels * ratio_gout)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l = lnorm(out_channels - gc)
        self.bn_g = gnorm(gc)
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact()
        self.act_g = gact()

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        return self.act_l(self.bn_l(x_l)), self.act_g(self.bn_g(x_g))


class FFCResnetBlock(nn.Module):
    def __init__(self, dim: int, padding_type: str, norm_layer, activation_layer,
                 dilation: int = 1, ratio_gin: float = 0.75, ratio_gout: float = 0.75,
                 enable_lfu: bool = False):
        super().__init__()
        kw = dict(
            kernel_size=3, padding=dilation, dilation=dilation,
            ratio_gin=ratio_gin, ratio_gout=ratio_gout, norm_layer=norm_layer,
            activation_layer=activation_layer, padding_type=padding_type,
            enable_lfu=enable_lfu,
        )
        self.conv1 = FFC_BN_ACT(dim, dim, **kw)
        self.conv2 = FFC_BN_ACT(dim, dim, **kw)

    def forward(self, x):
        x_l, x_g = x if isinstance(x, tuple) else (x, 0)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv2(self.conv1((x_l, x_g)))
        return id_l + x_l, id_g + x_g


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        x_l, x_g = x
        return x_l if not torch.is_tensor(x_g) else torch.cat(x, dim=1)


class FFCResNetGenerator(nn.Module):
    """Arsitektur generator LaMa. n_blocks 9 (base) atau 18 (large)."""

    def __init__(
        self, input_nc: int = 4, output_nc: int = 3, ngf: int = 64,
        n_downsampling: int = 3, n_blocks: int = 18, max_features: int = 1024,
    ):
        super().__init__()
        norm, act = nn.BatchNorm2d, nn.ReLU
        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            FFC_BN_ACT(
                input_nc, ngf, kernel_size=7, padding=0, ratio_gin=0, ratio_gout=0,
                norm_layer=norm, activation_layer=act,
            ),
        ]
        for i in range(n_downsampling):
            mult = 2 ** i
            model.append(
                FFC_BN_ACT(
                    min(max_features, ngf * mult),
                    min(max_features, ngf * mult * 2),
                    kernel_size=3, stride=2, padding=1,
                    ratio_gin=0, ratio_gout=0.75 if i == n_downsampling - 1 else 0,
                    norm_layer=norm, activation_layer=act,
                )
            )
        feats = min(max_features, ngf * 2 ** n_downsampling)
        for _ in range(n_blocks):
            model.append(FFCResnetBlock(feats, "reflect", norm, act))
        model.append(ConcatTupleLayer())
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    min(max_features, ngf * mult),
                    min(max_features, int(ngf * mult / 2)),
                    kernel_size=3, stride=2, padding=1, output_padding=1,
                ),
                norm(min(max_features, int(ngf * mult / 2))),
                nn.ReLU(True),
            ]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------- runtime

_MODEL: FFCResNetGenerator | None = None
_LOAD_FAILED = False

# Konteks di sekeliling mask yang ikut dikirim ke generator, dan sisi minimum
# tile. LaMa butuh tekstur tetangga untuk ditiru; kotak yang mepet glyph cuma
# berisi lubang. 64 px cukup untuk beberapa periode screentone.
_TILE_PAD = 64
_TILE_MIN = 192


def _infer_n_blocks(sd: dict) -> int:
    """Baca jumlah resnet block dari nama key — jangan tebak base vs large."""
    idxs = [
        int(k.split(".")[1])
        for k in sd
        if k.startswith("model.") and ".conv1.ffc." in k and k.split(".")[1].isdigit()
    ]
    return (max(idxs) - 4) if idxs else 18  # blok resnet mulai di index 5


def get_model(device: str = "cuda") -> FFCResNetGenerator | None:
    """Muat generator sekali. None kalau bobot tidak cocok — pemanggil fallback."""
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None or _LOAD_FAILED:
        return _MODEL

    path = WEIGHTS / "lama_large_512px.ckpt"
    if not path.exists():
        _LOAD_FAILED = True
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("gen_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        sd = {k.replace("generator.", "", 1): v for k, v in sd.items()}

        model = FFCResNetGenerator(n_blocks=_infer_n_blocks(sd))
        missing, unexpected = model.load_state_dict(sd, strict=False)

        # Self-check: kalau lebih dari 5% parameter tidak terisi, arsitekturnya
        # beda dan hasilnya akan jadi bubur. Lebih baik jatuh ke cv2.
        total = len(model.state_dict())
        if len(missing) > total * 0.05:
            note("warn", "inpaint",
                 f"arsitektur tidak cocok ({len(missing)}/{total} kosong) -> cv2.inpaint")
            _LOAD_FAILED = True
            return None
        if unexpected:
            print(f"[inpaint] {len(unexpected)} key ekstra diabaikan")

        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        _MODEL = model
    except (RuntimeError, KeyError, OSError, ValueError) as exc:
        note("warn", "inpaint", f"gagal muat LaMa ({exc}) -> pakai cv2.inpaint")
        _LOAD_FAILED = True
        _MODEL = None
    return _MODEL


def _pad8(arr: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = arr.shape[:2]
    ph, pw = (-h) % 8, (-w) % 8
    if ph or pw:
        arr = cv2.copyMakeBorder(arr, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    return arr, ph, pw


def _cv2_fallback(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Telea cukup baik untuk screentone manga saat LaMa tidak tersedia."""
    return cv2.inpaint(img, (mask > 0).astype(np.uint8), 5, cv2.INPAINT_TELEA)


def _grow(box: list[int], w: int, h: int) -> tuple[int, int, int, int]:
    """Perbesar kotak sampai minimal _TILE_MIN, lalu jepit ke tepi halaman."""
    x1, y1, x2, y2 = box
    for _ in range(2):  # dua lintasan: setelah dijepit di tepi, sisi lain digeser
        dw, dh = _TILE_MIN - (x2 - x1), _TILE_MIN - (y2 - y1)
        if dw > 0:
            x1, x2 = x1 - dw // 2, x2 + dw - dw // 2
        if dh > 0:
            y1, y2 = y1 - dh // 2, y2 + dh - dh // 2
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return x1, y1, x2, y2


def _mask_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Kotak berisi mask + konteks sekelilingnya; yang bertumpuk digabung."""
    h, w = mask.shape[:2]
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    boxes = [
        [
            int(stats[i, 0]) - _TILE_PAD, int(stats[i, 1]) - _TILE_PAD,
            int(stats[i, 0] + stats[i, 2]) + _TILE_PAD,
            int(stats[i, 1] + stats[i, 3]) + _TILE_PAD,
        ]
        for i in range(1, n)
    ]

    # Digabung sampai tidak ada yang bertumpuk: satu blok teks tidak boleh pecah
    # jadi satu tile per glyph — tiap tile kehilangan konteks tetangganya, dan
    # tile yang bertumpuk akan menimpa hasil tetangganya.
    changed = True
    while changed:
        changed, merged = False, []
        for b in boxes:
            for o in merged:
                if b[0] < o[2] and o[0] < b[2] and b[1] < o[3] and o[1] < b[3]:
                    o[0], o[1] = min(o[0], b[0]), min(o[1], b[1])
                    o[2], o[3] = max(o[2], b[2]), max(o[3], b[3])
                    changed = True
                    break
            else:
                merged.append(b)
        boxes = merged

    return [_grow(b, w, h) for b in boxes]


def _run(
    model: FFCResNetGenerator, crop: np.ndarray, mask: np.ndarray, device: str
) -> np.ndarray | None:
    """Satu forward pass. None kalau OOM — pemanggil jatuh ke cv2."""
    h, w = crop.shape[:2]
    scale = min(1.0, SETTINGS.lama_size / max(h, w))
    if scale < 1.0:
        sm = (max(1, int(w * scale)), max(1, int(h * scale)))
        crop_s = cv2.resize(crop, sm, interpolation=cv2.INTER_AREA)
        mask_s = cv2.resize(mask, sm, interpolation=cv2.INTER_NEAREST)
    else:
        crop_s, mask_s = crop, mask

    pimg, ph, pw = _pad8(crop_s)
    pmask, _, _ = _pad8(mask_s)

    t_img = torch.from_numpy(pimg).permute(2, 0, 1).float().div_(255.0)[None]
    t_msk = torch.from_numpy((pmask > 0).astype(np.float32))[None, None]
    t_img, t_msk = t_img.to(device), t_msk.to(device)

    try:
        with torch.inference_mode():
            # fp32 saja: torch.fft ada di cast-policy fp32, AMP nihil manfaat.
            pred = model(torch.cat([t_img * (1 - t_msk), t_msk], dim=1))
            out = pred * t_msk + t_img * (1 - t_msk)
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    arr = (out[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    arr = arr[: arr.shape[0] - ph if ph else None, : arr.shape[1] - pw if pw else None]
    if arr.shape[:2] != (h, w):
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_CUBIC)
    return arr


def inpaint(img: np.ndarray, mask: np.ndarray, device: str | None = None) -> np.ndarray:
    """Inpaint area mask. img RGB uint8, mask uint8 0/255. Return RGB uint8.

    Kontrak forward LaMa: input 4-channel cat([img*(1-m), m]), img float [0,1],
    mask float {0,1} — bukan 255. Output dikomposit pred*m + (1-m)*img.

    Dijalankan per TILE di resolusi asli, bukan sekali untuk seluruh halaman.
    Halaman manga tingginya ~2000 px; menyusutkannya ke 512 lalu membesarkannya
    lagi meratakan screentone jadi bercak kelabu buram — persis artefak yang
    terlihat di kolom narasi. Tile di sekeliling mask hampir selalu di bawah
    512 px, jadi crosshatch direkonstruksi 1:1 tanpa resample sama sekali.
    """
    if mask.max() == 0:
        return img
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    if model is None:
        return _cv2_fallback(img, mask)

    out = img.copy()
    for x1, y1, x2, y2 in _mask_boxes(mask):
        sub_mask = mask[y1:y2, x1:x2]
        if sub_mask.max() == 0:
            continue
        arr = _run(model, img[y1:y2, x1:x2], sub_mask, device)
        if arr is None:
            return _cv2_fallback(img, mask)
        # Piksel di luar mask harus persis asli, jadi komposit di sini juga.
        m3 = (sub_mask > 0)[:, :, None]
        out[y1:y2, x1:x2] = np.where(m3, arr, out[y1:y2, x1:x2])
    return out


def release() -> None:
    """Bebaskan ~205 MB + VRAM."""
    global _MODEL
    _MODEL = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

