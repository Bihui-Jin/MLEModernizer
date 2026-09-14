# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.13

# 2. Installed packages

No external packages required in the script and installed.

# 3. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 4. Code solution

## === cell 0
import os
import glob
import re
import math
import csv
from typing import List, Tuple, Dict

import pandas as pd




## === cell 1
class Config:
    train_dir = "/kaggle/working/train"
    test_dir = "/kaggle/working/test"


cfg = Config()




## === cell 2
def extract_id_from_path(p: str) -> int:
    base = os.path.basename(p)
    m = re.match(r"^(\d+)\.", base)
    if m is None:
        m2 = re.search(r"(\d+)", base)
        if m2 is None:
            raise ValueError(f"Could not extract numeric id from filename: {base}")
        return int(m2.group(1))
    return int(m.group(1))




## === cell 3
candidate_test_dirs = [cfg.test_dir]
for base in ("/kaggle/working", "/kaggle/input", "/kaggle/data"):
    candidate_test_dirs.extend(
        [
            os.path.join(base, "test"),
            os.path.join(base, "dogs-vs-cats-redux-kernels-edition", "test"),
            os.path.join(base, "dogs-vs-cats-redux-kernels-edition", "test", "test"),
            os.path.join(base, "dogs-vs-cats-redux-kernels-edition", "test", "unknown"),
            os.path.join(base, "test", "test"),
            os.path.join(base, "test", "unknown"),
        ]
    )

test_paths: List[str] = []
for td in candidate_test_dirs:
    if not td or not os.path.isdir(td):
        continue
    test_paths = glob.glob(os.path.join(td, "**", "*.jpg"), recursive=True)
    if len(test_paths) == 0:
        test_paths = glob.glob(os.path.join(td, "**", "*.png"), recursive=True)
    if len(test_paths) > 0:
        cfg.test_dir = td  # preserve consistent directory reference for later cells
        break

if len(test_paths) == 0:
    raise FileNotFoundError(
        f"No test images found under {cfg.test_dir}. "
        "Ensure the test zip has been unzipped to /kaggle/working/test."
    )

image_ids = [extract_id_from_path(p) for p in test_paths]



## === cell 4


def _autodiscover_train_dir(cfg: Config) -> str:
    candidates = [cfg.train_dir]
    for base in ("/kaggle/working", "/kaggle/input", "/kaggle/data"):
        candidates.extend(
            [
                os.path.join(base, "train"),
                os.path.join(base, "dogs-vs-cats-redux-kernels-edition", "train"),
                os.path.join(
                    base, "dogs-vs-cats-redux-kernels-edition", "train", "train"
                ),
            ]
        )
    for td in candidates:
        if not td or not os.path.isdir(td):
            continue
        if os.path.isdir(os.path.join(td, "cat")) and os.path.isdir(
            os.path.join(td, "dog")
        ):
            return td
    return cfg.train_dir


cfg.train_dir = _autodiscover_train_dir(cfg)
cat_dir = os.path.join(cfg.train_dir, "cat")
dog_dir = os.path.join(cfg.train_dir, "dog")

if not (os.path.isdir(cat_dir) and os.path.isdir(dog_dir)):
    raise FileNotFoundError(
        f"Could not find train/cat and train/dog under cfg.train_dir={cfg.train_dir}."
    )

cat_paths = glob.glob(os.path.join(cat_dir, "*.jpg")) + glob.glob(
    os.path.join(cat_dir, "*.png")
)
dog_paths = glob.glob(os.path.join(dog_dir, "*.jpg")) + glob.glob(
    os.path.join(dog_dir, "*.png")
)

if len(cat_paths) == 0 or len(dog_paths) == 0:
    raise FileNotFoundError(
        f"Empty training data: cat={len(cat_paths)} dog={len(dog_paths)} under {cfg.train_dir}."
    )

print(f"Train dir: {cfg.train_dir} | cats: {len(cat_paths)} dogs: {len(dog_paths)}")
print(f"Test dir:  {cfg.test_dir} | test images: {len(test_paths)}")



## === cell 5

import subprocess
import tempfile


def _which(cmd: str) -> bool:
    from shutil import which

    return which(cmd) is not None


_HAS_CONVERT = _which("convert") or _which("magick")


def _run_convert_to_pgm(src: str, dst_pgm: str, size: int = 48) -> None:
    if _which("magick"):
        cmd = [
            "magick",
            "convert",
            src,
            "-colorspace",
            "Gray",
            "-resize",
            f"{size}x{size}!",
            "pgm:" + dst_pgm,
        ]
    else:
        cmd = [
            "convert",
            src,
            "-colorspace",
            "Gray",
            "-resize",
            f"{size}x{size}!",
            "pgm:" + dst_pgm,
        ]
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def _read_pgm(pgm_path: str) -> Tuple[int, int, List[int]]:
    with open(pgm_path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError(f"Unsupported PGM magic: {magic}")

        def _read_non_comment_line():
            line = f.readline()
            while line.startswith(b"#"):
                line = f.readline()
            return line

        wh = _read_non_comment_line()
        while wh.strip() == b"":
            wh = _read_non_comment_line()
        w_str, h_str = wh.split()
        w, h = int(w_str), int(h_str)

        maxv_line = _read_non_comment_line()
        while maxv_line.strip() == b"":
            maxv_line = _read_non_comment_line()
        maxv = int(maxv_line)
        if maxv > 255:
            raise ValueError("Only 8-bit PGM supported.")

        data = f.read(w * h)
        if len(data) != w * h:
            raise ValueError("Truncated PGM data.")
        pixels = list(data)
    return w, h, pixels


def _extract_features_from_pixels(
    w: int, h: int, pixels: List[int], bins: int = 16
) -> List[int]:
    hist_g = [0] * bins
    for v in pixels:
        b = (v * bins) // 256
        if b >= bins:
            b = bins - 1
        hist_g[b] += 1

    hist_e = [0] * bins
    for y in range(h - 1):
        row = y * w
        row_next = (y + 1) * w
        for x in range(w - 1):
            v = pixels[row + x]
            dx = abs(v - pixels[row + x + 1])
            dy = abs(v - pixels[row_next + x])
            e = dx + dy  # 0..510
            b = (e * bins) // 512
            if b >= bins:
                b = bins - 1
            hist_e[b] += 1

    return hist_g + hist_e


def extract_features(path: str, size: int = 48) -> List[int]:
    if not _HAS_CONVERT:
        raise RuntimeError(
            "No ImageMagick 'convert' or 'magick' found. Cannot decode JPG/PNG without external python packages."
        )
    with tempfile.TemporaryDirectory() as td:
        pgm_path = os.path.join(td, "img.pgm")
        _run_convert_to_pgm(path, pgm_path, size=size)
        w, h, pixels = _read_pgm(pgm_path)
    return _extract_features_from_pixels(w, h, pixels, bins=16)




## === cell 6


def train_multinomial_nb(
    X: List[List[int]], y: List[int], alpha: float = 1.0
) -> Dict[str, object]:
    n = len(X)
    d = len(X[0])
    n1 = sum(y)
    n0 = n - n1
    sum0 = [0.0] * d
    sum1 = [0.0] * d
    tot0 = 0.0
    tot1 = 0.0
    for xi, yi in zip(X, y):
        if yi == 1:
            for j, v in enumerate(xi):
                sum1[j] += v
                tot1 += v
        else:
            for j, v in enumerate(xi):
                sum0[j] += v
                tot0 += v

    denom0 = tot0 + alpha * d
    denom1 = tot1 + alpha * d
    logp0 = [math.log((sum0[j] + alpha) / denom0) for j in range(d)]
    logp1 = [math.log((sum1[j] + alpha) / denom1) for j in range(d)]
    prior1 = (n1 + 1.0) / (n + 2.0)
    prior0 = 1.0 - prior1
    return {
        "logp0": logp0,
        "logp1": logp1,
        "logprior0": math.log(prior0),
        "logprior1": math.log(prior1),
    }


def predict_proba_nb(model: Dict[str, object], X: List[List[int]]) -> List[float]:
    logp0 = model["logp0"]
    logp1 = model["logp1"]
    lp0_prior = model["logprior0"]
    lp1_prior = model["logprior1"]
    out = []
    for xi in X:
        s0 = lp0_prior
        s1 = lp1_prior
        for v, a0, a1 in zip(xi, logp0, logp1):
            if v:
                s0 += v * a0
                s1 += v * a1
        m = max(s0, s1)
        e0 = math.exp(s0 - m)
        e1 = math.exp(s1 - m)
        p1 = e1 / (e0 + e1)
        out.append(p1)
    return out




## === cell 7
import struct
import zlib


def _resize_nearest_gray(
    pix: List[int], w: int, h: int, size: int
) -> Tuple[int, int, List[int]]:
    if w == size and h == size:
        return w, h, pix
    out = [0] * (size * size)
    for y in range(size):
        sy = (y * h) // size
        row_src = sy * w
        row_dst = y * size
        for x in range(size):
            sx = (x * w) // size
            out[row_dst + x] = pix[row_src + sx]
    return size, size, out


def _png_unfilter(scan: bytearray, prev: bytes, bpp: int, ftype: int) -> bytearray:
    out = bytearray(scan)  # will modify in-place
    if ftype == 0:  # None
        return out
    if ftype == 1:  # Sub
        for i in range(len(out)):
            left = out[i - bpp] if i >= bpp else 0
            out[i] = (out[i] + left) & 0xFF
        return out
    if ftype == 2:  # Up
        for i in range(len(out)):
            up = prev[i] if prev else 0
            out[i] = (out[i] + up) & 0xFF
        return out
    if ftype == 3:  # Average
        for i in range(len(out)):
            left = out[i - bpp] if i >= bpp else 0
            up = prev[i] if prev else 0
            out[i] = (out[i] + ((left + up) // 2)) & 0xFF
        return out
    if ftype == 4:  # Paeth

        def paeth(a: int, b: int, c: int) -> int:
            p = a + b - c
            pa = abs(p - a)
            pb = abs(p - b)
            pc = abs(p - c)
            if pa <= pb and pa <= pc:
                return a
            if pb <= pc:
                return b
            return c

        for i in range(len(out)):
            a = out[i - bpp] if i >= bpp else 0
            b = prev[i] if prev else 0
            c = prev[i - bpp] if (prev and i >= bpp) else 0
            out[i] = (out[i] + paeth(a, b, c)) & 0xFF
        return out
    raise ValueError(f"Unsupported PNG filter type: {ftype}")


def _read_png_grayscale(path: str) -> Tuple[int, int, List[int]]:
    with open(path, "rb") as f:
        sig = f.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Not a PNG")

        width = height = None
        bit_depth = color_type = None
        interlace = None
        idat = bytearray()

        while True:
            chunk_len_b = f.read(4)
            if not chunk_len_b:
                break
            (chunk_len,) = struct.unpack(">I", chunk_len_b)
            ctype = f.read(4)
            data = f.read(chunk_len)
            f.read(4)  # crc
            if ctype == b"IHDR":
                width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                    ">IIBBBBB", data
                )
                if bit_depth != 8:
                    raise ValueError("Only 8-bit PNG supported")
                if interlace != 0:
                    raise ValueError("Interlaced PNG not supported")
                if color_type not in (0, 2):  # grayscale or truecolor
                    raise ValueError(f"Unsupported PNG color type: {color_type}")
            elif ctype == b"IDAT":
                idat.extend(data)
            elif ctype == b"IEND":
                break

        if width is None or height is None:
            raise ValueError("Invalid PNG (missing IHDR)")

        raw = zlib.decompress(bytes(idat))
        if color_type == 0:
            bpp = 1
        else:  # truecolor RGB
            bpp = 3

        stride = width * bpp
        pos = 0
        prev = b""
        out_pix: List[int] = []
        for _ in range(height):
            ftype = raw[pos]
            pos += 1
            scan = bytearray(raw[pos : pos + stride])
            pos += stride
            recon = _png_unfilter(scan, prev, bpp=bpp, ftype=ftype)
            if color_type == 0:
                out_pix.extend(recon)
            else:
                for i in range(0, len(recon), 3):
                    r, g, b = recon[i], recon[i + 1], recon[i + 2]
                    y = (299 * r + 587 * g + 114 * b) // 1000
                    out_pix.append(y)
            prev = bytes(recon)
        return width, height, out_pix


_ZZ = [
    0,
    1,
    8,
    16,
    9,
    2,
    3,
    10,
    17,
    24,
    32,
    25,
    18,
    11,
    4,
    5,
    12,
    19,
    26,
    33,
    40,
    48,
    41,
    34,
    27,
    20,
    13,
    6,
    7,
    14,
    21,
    28,
    35,
    42,
    49,
    56,
    57,
    50,
    43,
    36,
    29,
    22,
    15,
    23,
    30,
    37,
    44,
    51,
    58,
    59,
    52,
    45,
    38,
    31,
    39,
    46,
    53,
    60,
    61,
    54,
    47,
    55,
    62,
    63,
]


def _build_huff_table(lengths: List[int], symbols: List[int]):
    table = {}
    code = 0
    k = 0
    maxlen = 0
    for l in range(1, 17):
        cnt = lengths[l - 1]
        maxlen = max(maxlen, l if cnt else maxlen)
        for _ in range(cnt):
            table[(code, l)] = symbols[k]
            k += 1
            code += 1
        code <<= 1
    return table, maxlen


class _BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.i = 0
        self.buf = 0
        self.nbits = 0

    def _fill(self):
        while self.nbits <= 16 and self.i < len(self.data):
            b = self.data[self.i]
            self.i += 1
            if b == 0xFF:
                if self.i < len(self.data) and self.data[self.i] == 0x00:
                    self.i += 1
                else:
                    pass
            self.buf = (self.buf << 8) | b
            self.nbits += 8

    def get_bits(self, n: int) -> int:
        if n == 0:
            return 0
        while self.nbits < n:
            self._fill()
            if self.nbits < n and self.i >= len(self.data):
                raise ValueError("Unexpected end of JPEG bitstream")
        shift = self.nbits - n
        val = (self.buf >> shift) & ((1 << n) - 1)
        self.nbits -= n
        self.buf &= (1 << self.nbits) - 1 if self.nbits else 0
        return val

    def get_bit(self) -> int:
        return self.get_bits(1)


def _huff_decode(br: _BitReader, table, maxlen: int) -> int:
    code = 0
    for l in range(1, maxlen + 1):
        code = (code << 1) | br.get_bit()
        sym = table.get((code, l))
        if sym is not None:
            return sym
    raise ValueError("Invalid Huffman code")


def _extend_sign(v: int, t: int) -> int:
    if t == 0:
        return 0
    vt = 1 << (t - 1)
    if v < vt:
        return v - ((1 << t) - 1)
    return v


def _idct8(block: List[float]) -> List[int]:
    out = [0] * 64
    for y in range(8):
        for x in range(8):
            s = 0.0
            for v in range(8):
                for u in range(8):
                    cu = 0.7071067811865476 if u == 0 else 1.0
                    cv = 0.7071067811865476 if v == 0 else 1.0
                    s += (
                        cu
                        * cv
                        * block[v * 8 + u]
                        * math.cos(((2 * x + 1) * u * math.pi) / 16.0)
                        * math.cos(((2 * y + 1) * v * math.pi) / 16.0)
                    )
            val = int(round(s / 4.0 + 128.0))
            if val < 0:
                val = 0
            elif val > 255:
                val = 255
            out[y * 8 + x] = val
    return out


def _read_jpeg_grayscale(path: str) -> Tuple[int, int, List[int]]:
    with open(path, "rb") as f:
        data = f.read()

    if not (len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8):
        raise ValueError("Not a JPEG")

    i = 2
    qt = {}  # id -> [64]
    huff_dc = {}
    huff_ac = {}
    width = height = None
    comp = None
    scan_data = b""

    def _read_marker():
        nonlocal i
        while i < len(data) and data[i] != 0xFF:
            i += 1
        while i < len(data) and data[i] == 0xFF:
            i += 1
        if i >= len(data):
            return None
        m = data[i]
        i += 1
        return m

    while i < len(data):
        m = _read_marker()
        if m is None:
            break
        if m == 0xD9:  # EOI
            break
        if m == 0xDA:  # SOS
            (seglen,) = struct.unpack(">H", data[i : i + 2])
            seg = data[i + 2 : i + seglen]
            i += seglen
            ns = seg[0]
            off = 1
            for _ in range(ns):
                off += 2
            scan_data = data[i:]
            break
        if 0xD0 <= m <= 0xD7:  # RSTn
            continue
        if m == 0x01 or m == 0x00:  # TEM or invalid
            continue
        if i + 2 > len(data):
            break
        (seglen,) = struct.unpack(">H", data[i : i + 2])
        seg = data[i + 2 : i + seglen]
        i += seglen

        if m == 0xDB:  # DQT
            off = 0
            while off < len(seg):
                pq_tq = seg[off]
                off += 1
                pq = pq_tq >> 4
                tq = pq_tq & 0x0F
                if pq != 0:
                    raise ValueError("Only 8-bit quant tables supported")
                q = list(seg[off : off + 64])
                off += 64
                q2 = [0] * 64
                for idx, zz in enumerate(_ZZ):
                    q2[zz] = q[idx]
                qt[tq] = q2
        elif m == 0xC0:  # SOF0 baseline
            if seg[0] != 8:
                raise ValueError("Only 8-bit JPEG supported")
            height = (seg[1] << 8) | seg[2]
            width = (seg[3] << 8) | seg[4]
            ncomp = seg[5]
            comps = {}
            off = 6
            for _ in range(ncomp):
                cid = seg[off]
                hv = seg[off + 1]
                tq = seg[off + 2]
                off += 3
                h = hv >> 4
                v = hv & 0x0F
                comps[cid] = (h, v, tq)
            comp = comps

            if ncomp == 1:
                if 1 not in comp:
                    raise ValueError("Only grayscale JPEG supported")
                if comp[1][0] != 1 or comp[1][1] != 1:
                    raise ValueError("Subsampled grayscale not supported")
            elif ncomp == 3:
                if 1 not in comp:
                    raise ValueError("Unsupported JPEG (missing Y component id=1)")
                if comp[1][0] != 1 or comp[1][1] != 1:
                    raise ValueError("Subsampled JPEG not supported")
            else:
                raise ValueError("Unsupported JPEG component count")
        elif m == 0xC4:  # DHT
            off = 0
            while off < len(seg):
                tc_th = seg[off]
                off += 1
                tc = tc_th >> 4  # 0 DC, 1 AC
                th = tc_th & 0x0F
                lengths = list(seg[off : off + 16])
                off += 16
                total = sum(lengths)
                symbols = list(seg[off : off + total])
                off += total
                table, maxlen = _build_huff_table(lengths, symbols)
                if tc == 0:
                    huff_dc[th] = (table, maxlen)
                else:
                    huff_ac[th] = (table, maxlen)
        else:
            pass

    if width is None or height is None or not scan_data or comp is None:
        raise ValueError("Incomplete JPEG")

    y_tq = comp[1][2]
    if y_tq not in qt:
        raise ValueError("Missing quant table")
    qtbl = qt[y_tq]

    if 0 not in huff_dc or 0 not in huff_ac:
        raise ValueError("Missing Huffman tables")
    dc_table, dc_max = huff_dc[0]
    ac_table, ac_max = huff_ac[0]

    br = _BitReader(scan_data)

    mcu_w = (width + 7) // 8
    mcu_h = (height + 7) // 8
    out = [0] * (width * height)

    dc_pred = 0
    for by in range(mcu_h):
        for bx in range(mcu_w):
            coeff = [0] * 64

            t = _huff_decode(br, dc_table, dc_max)
            diff = _extend_sign(br.get_bits(t), t)
            dc_pred += diff
            coeff[0] = dc_pred

            k = 1
            while k < 64:
                rs = _huff_decode(br, ac_table, ac_max)
                if rs == 0:  # EOB
                    break
                r = rs >> 4
                s = rs & 0x0F
                if s == 0 and r == 15:  # ZRL
                    k += 16
                    continue
                k += r
                if k >= 64:
                    break
                acv = _extend_sign(br.get_bits(s), s)
                coeff[_ZZ[k]] = acv
                k += 1

            block = [float(coeff[ii] * qtbl[ii]) for ii in range(64)]
            pix8 = _idct8(block)

            x0 = bx * 8
            y0 = by * 8
            for yy in range(8):
                y = y0 + yy
                if y >= height:
                    continue
                row_out = y * width
                row8 = yy * 8
                for xx in range(8):
                    x = x0 + xx
                    if x >= width:
                        continue
                    out[row_out + x] = pix8[row8 + xx]

    return width, height, out


def _read_image_grayscale(path: str) -> Tuple[int, int, List[int]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        return _read_png_grayscale(path)

    try:
        return _read_jpeg_grayscale(path)
    except Exception:
        if "_HAS_CONVERT" in globals() and globals().get("_HAS_CONVERT"):
            with tempfile.TemporaryDirectory() as td:
                pgm_path = os.path.join(td, "img.pgm")
                _run_convert_to_pgm(
                    path, pgm_path, size=48
                )  # size here is irrelevant; caller resizes anyway
                w, h, pixels = _read_pgm(pgm_path)
            return w, h, pixels
        raise


def extract_features(path: str, size: int = 48) -> List[int]:
    try:
        w, h, pixels = _read_image_grayscale(path)
        w, h, pixels = _resize_nearest_gray(pixels, w, h, size)
        return _extract_features_from_pixels(w, h, pixels, bins=16)
    except Exception:
        if "_HAS_CONVERT" in globals() and globals().get("_HAS_CONVERT"):
            with tempfile.TemporaryDirectory() as td:
                pgm_path = os.path.join(td, "img.pgm")
                _run_convert_to_pgm(path, pgm_path, size=size)
                w, h, pixels = _read_pgm(pgm_path)
            return _extract_features_from_pixels(w, h, pixels, bins=16)
        raise


MAX_PER_CLASS = 2500  # total 5000 images; should run under 600s with 48x48 conversion.

cat_paths = sorted(cat_paths)[:MAX_PER_CLASS]
dog_paths = sorted(dog_paths)[:MAX_PER_CLASS]

X_train: List[List[int]] = []
y_train: List[int] = []

for p in cat_paths:
    X_train.append(extract_features(p, size=48))
    y_train.append(0)

for p in dog_paths:
    X_train.append(extract_features(p, size=48))
    y_train.append(1)

model = train_multinomial_nb(X_train, y_train, alpha=1.0)
print(f"Trained NB on {len(X_train)} images with {len(X_train[0])} features.")


## --- ERROR in cell 7, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3792733163.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    518[0m [0;34m[0m[0m
[1;32m    519[0m [0;32mfor[0m [0mp[0m [0;32min[0m [0mcat_paths[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 520[0;31m     [0mX_train[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mextract_features[0m[0;34m([0m[0mp[0m[0;34m,[0m [0msize[0m[0;34m=[0m[0;36m48[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    521[0m     [0my_train[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0;36m0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    522[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3792733163.py[0m in [0;36mextract_features[0;34m(path, size)[0m
[1;32m    496[0m [0;32mdef[0m [0mextract_features[0m[0;34m([0m[0mpath[0m[0;34m:[0m [0mstr[0m[0;34m,[0m [0msize[0m[0;34m:[0m [0mint[0m [0;34m=[0m [0;36m48[0m[0;34m)[0m [0;34m->[0m [0mList[0m[0;34m[[0m[0mint[0m[0;34m][0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    497[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 498[0;31m         [0mw[0m[0;34m,[0m [0mh[0m[0;34m,[0m [0mpixels[0m [0;34m=[0m [0m_read_image_grayscale[0m[0;34m([0m[0mpath[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    499[0m         [0mw[0m[0;34m,[0m [0mh[0m[0;34m,[0m [0mpixels[0m [0;34m=[0m [0m_resize_nearest_gray[0m[0;34m([0m[0mpixels[0m[0;34m,[0m [0mw[0m[0;34m,[0m [0mh[0m[0;34m,[0m [0msize[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    500[0m         [0;32mreturn[0m [0m_extract_features_from_pixels[0m[0;34m([0m[0mw[0m[0;34m,[0m [0mh[0m[0;34m,[0m [0mpixels[0m[0;34m,[0m [0mbins[0m[0;34m=[0m[0;36m16[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3792733163.py[0m in [0;36m_read_image_grayscale[0;34m(path)[0m
[1;32m    481[0m     [0;31m# If ImageMagick is available, transparently fall back to convert->PGM so training/inference do not crash.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    482[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 483[0;31m         [0;32mreturn[0m [0m_read_jpeg_grayscale[0m[0;34m([0m[0mpath[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    484[0m     [0;32mexcept[0m [0mException[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    485[0m         [0;32mif[0m [0;34m"_HAS_CONVERT"[0m [0;32min[0m [0mglobals[0m[0;34m([0m[0;34m)[0m [0;32mand[0m [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mget[0m[0;34m([0m[0;34m"_HAS_CONVERT"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3792733163.py[0m in [0;36m_read_jpeg_grayscale[0;34m(path)[0m
[1;32m    384[0m                     [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"Unsupported JPEG (missing Y component id=1)"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    385[0m                 [0;32mif[0m [0mcomp[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m[[0m[0;36m0[0m[0;34m][0m [0;34m!=[0m [0;36m1[0m [0;32mor[0m [0mcomp[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m[[0m[0;36m1[0m[0;34m][0m [0;34m!=[0m [0;36m1[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 386[0;31m                     [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"Subsampled JPEG not supported"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    387[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    388[0m                 [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"Unsupported JPEG component count"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Subsampled JPEG not supported

## === cell 8
test_paths_sorted = sorted(test_paths, key=extract_id_from_path)
image_ids = [extract_id_from_path(p) for p in test_paths_sorted]

X_test: List[List[int]] = []
for p in test_paths_sorted:
    X_test.append(extract_features(p, size=48))

outputs = predict_proba_nb(model, X_test)
