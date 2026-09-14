# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

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

# 5. Target score

0.0339717376907435

# 6. Current score

0.69315

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.69315) has done: 'Your notebook isn’t yielding a Kaggle score because it currently won’t run in a clean environment: it imports packages that are not installed (timm, albumentations, cv2, pytorch_lightning, sklearn), and it also points to an external checkpoint directory that likely doesn’t exist. To get a valid submission while preserving your inference semantics (probability of dog for each test image id), I’m making the smallest possible change: generate a properly formatted `submission.csv` directly from `sample_submission.csv` with a constant, safe probability (0.5) so the pipeline always produces a valid file. This does not try to maximize score; it just unblocks submission generation end-to-end under the “no external packages installed” constraint. Once you confirm a valid score is produced, we can re-enable model inference if/when those dependencies/checkpoints are available.'
- What this solution (achieved 0.69315) has done: 'I fix the crash by replacing the PNG-only decoder with a minimal JPEG/PNG reader that works using only the Python standard library (via `tkinter.PhotoImage` for PNG/GIF and a small baseline-JPEG decoder for typical Kaggle JPGs). I also make sure `model` is always defined (even when training fails) so submission generation never hits a `NameError`. To keep the core logic intact, I preserve the exact histogram Naive Bayes approach and only change image loading/decoding so it can actually read the provided `.jpg` files. Finally, I ensure the submission is written to `/kaggle/working/submission.csv` with `id,label` and sorted IDs.'

# 9. Code solution

## === cell 0
import os
import re
import math
import struct
import zlib
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")




## === cell 1
class Config:
    input_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"
    working_dir = "/kaggle/working"
    sample_sub_path = os.path.join(input_dir, "sample_submission.csv")
    out_path = os.path.join(working_dir, "submission.csv")

    train_cat_dir = os.path.join(input_dir, "train", "cat")
    train_dog_dir = os.path.join(input_dir, "train", "dog")
    test_dir = os.path.join(input_dir, "test", "unknown")

    max_train_per_class = 4000  # keep runtime under 600s in pure Python
    bins = 32  # pixel intensity bins
    laplace = 1.0  # smoothing for calibrated probabilities


cfg = Config()
os.makedirs(cfg.working_dir, exist_ok=True)



## === cell 2


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def _paeth_predictor(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _png_gray_mean_from_bytes(data, max_pixels=256 * 256):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a PNG file")

    pos = 8
    width = height = None
    bit_depth = color_type = None
    idat = bytearray()

    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        ctype = data[pos + 4 : pos + 8]
        pos += 8
        chunk = data[pos : pos + length]
        pos += length + 4  # skip CRC

        if ctype == b"IHDR":
            width, height, bit_depth, color_type, comp, filt, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
            if bit_depth != 8:
                raise ValueError(f"Unsupported bit depth {bit_depth}")
            if color_type not in (0, 2, 4, 6):
                raise ValueError(f"Unsupported color type {color_type}")
            if comp != 0 or filt != 0 or interlace != 0:
                raise ValueError("Unsupported PNG compression/filter/interlace")
        elif ctype == b"IDAT":
            idat.extend(chunk)
        elif ctype == b"IEND":
            break

    if width is None or height is None:
        raise ValueError("Malformed PNG (no IHDR)")

    raw = zlib.decompress(bytes(idat))

    if color_type == 0:
        channels = 1
    elif color_type == 2:
        channels = 3
    elif color_type == 4:
        channels = 2
    elif color_type == 6:
        channels = 4
    bpp = channels  # 8-bit per channel

    stride = width * bpp
    expected = height * (1 + stride)
    if len(raw) < expected:
        raise ValueError("Truncated PNG data")

    prev = bytearray(stride)
    inpos = 0

    total_pixels = width * height
    step = 1
    if total_pixels > max_pixels:
        step = int(math.sqrt(total_pixels / max_pixels)) + 1

    s = 0
    cnt = 0

    for y in range(height):
        ftype = raw[inpos]
        inpos += 1
        scan = raw[inpos : inpos + stride]
        inpos += stride

        recon = bytearray(stride)
        if ftype == 0:  # None
            recon[:] = scan
        elif ftype == 1:  # Sub
            for i in range(stride):
                left = recon[i - bpp] if i >= bpp else 0
                recon[i] = (scan[i] + left) & 0xFF
        elif ftype == 2:  # Up
            for i in range(stride):
                recon[i] = (scan[i] + prev[i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(stride):
                left = recon[i - bpp] if i >= bpp else 0
                up = prev[i]
                recon[i] = (scan[i] + ((left + up) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                left = recon[i - bpp] if i >= bpp else 0
                up = prev[i]
                up_left = prev[i - bpp] if i >= bpp else 0
                recon[i] = (scan[i] + _paeth_predictor(left, up, up_left)) & 0xFF
        else:
            raise ValueError(f"Unsupported PNG filter {ftype}")

        if y % step == 0:
            for x in range(0, width, step):
                base = x * bpp
                if channels == 1 or channels == 2:
                    g = recon[base]
                else:
                    r = recon[base]
                    gch = recon[base + 1]
                    b = recon[base + 2]
                    g = (299 * r + 587 * gch + 114 * b) // 1000
                s += g
                cnt += 1

        prev = recon

    return s / max(1, cnt)




class _BitReader:
    def __init__(self, data, pos):
        self.data = data
        self.pos = pos
        self.buf = 0
        self.nbits = 0

    def _fill(self):
        while self.nbits <= 24 and self.pos < len(self.data):
            b = self.data[self.pos]
            self.pos += 1
            if b == 0xFF:
                while self.pos < len(self.data) and self.data[self.pos] == 0xFF:
                    self.pos += 1
                if self.pos < len(self.data) and self.data[self.pos] == 0x00:
                    self.pos += 1
                    b = 0xFF
                else:
                    self.pos -= 2
                    break
            self.buf = (self.buf << 8) | b
            self.nbits += 8

    def getbits(self, n):
        while self.nbits < n:
            self._fill()
            if self.nbits < n:
                raise ValueError("Unexpected end of JPEG entropy data")
        self.nbits -= n
        return (self.buf >> self.nbits) & ((1 << n) - 1)

    def getbit(self):
        return self.getbits(1)


def _extend_sign(v, t):
    if t == 0:
        return 0
    vt = 1 << (t - 1)
    if v < vt:
        v -= (1 << t) - 1
    return v


def _build_huff(table_counts, symbols):
    code = 0
    k = 0
    lut = {}
    for i in range(1, 17):
        for _ in range(table_counts[i - 1]):
            lut[(code, i)] = symbols[k]
            k += 1
            code += 1
        code <<= 1
    return lut


def _huff_decode(br, lut):
    code = 0
    for length in range(1, 17):
        code = (code << 1) | br.getbit()
        sym = lut.get((code, length))
        if sym is not None:
            return sym
    raise ValueError("Invalid Huffman code")


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


def _idct8(block):
    out = [0.0] * 64
    for y in range(8):
        for x in range(8):
            s = 0.0
            for v in range(8):
                for u in range(8):
                    cu = 1.0 / math.sqrt(2.0) if u == 0 else 1.0
                    cv = 1.0 / math.sqrt(2.0) if v == 0 else 1.0
                    coeff = block[v * 8 + u]
                    s += (
                        cu
                        * cv
                        * coeff
                        * math.cos(((2 * x + 1) * u * math.pi) / 16.0)
                        * math.cos(((2 * y + 1) * v * math.pi) / 16.0)
                    )
            out[y * 8 + x] = 0.25 * s
    return out


def _jpeg_gray_mean_from_bytes(data, max_pixels=256 * 256):
    if not (len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8):
        raise ValueError("Not a JPEG file")
    pos = 2

    qtables = {}
    huff_dc = {}
    huff_ac = {}
    frame = None
    scan = None

    def read_marker():
        nonlocal pos
        while pos < len(data) and data[pos] != 0xFF:
            pos += 1
        while pos < len(data) and data[pos] == 0xFF:
            pos += 1
        if pos >= len(data):
            return None
        m = data[pos]
        pos += 1
        return m

    while True:
        m = read_marker()
        if m is None:
            break
        if m == 0xD9:  # EOI
            break
        if m == 0xDA:  # SOS
            Ls = struct.unpack(">H", data[pos : pos + 2])[0]
            hdr = data[pos : pos + Ls]
            pos += Ls
            ns = hdr[2]
            comps = []
            idx = 3
            for _ in range(ns):
                cs = hdr[idx]
                tdta = hdr[idx + 1]
                td = (tdta >> 4) & 0xF
                ta = tdta & 0xF
                comps.append((cs, td, ta))
                idx += 2
            scan = {"comps": comps, "entropy_pos": pos}
            break
        if m in (0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7):
            continue
        if m == 0x01:
            continue
        if pos + 2 > len(data):
            break
        L = struct.unpack(">H", data[pos : pos + 2])[0]
        seg = data[pos + 2 : pos + L]
        pos += L

        if m == 0xDB:  # DQT
            i = 0
            while i < len(seg):
                pq_tq = seg[i]
                i += 1
                pq = (pq_tq >> 4) & 0xF
                tq = pq_tq & 0xF
                if pq != 0:
                    raise ValueError("16-bit quant tables not supported")
                qt = list(seg[i : i + 64])
                i += 64
                qtables[tq] = qt
        elif m == 0xC0:  # SOF0
            p = seg[0]
            if p != 8:
                raise ValueError("Only 8-bit JPEG supported")
            Y = struct.unpack(">H", seg[1:3])[0]
            X = struct.unpack(">H", seg[3:5])[0]
            nf = seg[5]
            comps = {}
            i = 6
            for _ in range(nf):
                cid = seg[i]
                hv = seg[i + 1]
                tq = seg[i + 2]
                h = (hv >> 4) & 0xF
                v = hv & 0xF
                comps[cid] = {"h": h, "v": v, "tq": tq}
                i += 3
            frame = {"w": X, "h": Y, "nf": nf, "comps": comps}
        elif m == 0xC4:  # DHT
            i = 0
            while i < len(seg):
                tc_th = seg[i]
                i += 1
                tc = (tc_th >> 4) & 0xF
                th = tc_th & 0xF
                counts = list(seg[i : i + 16])
                i += 16
                total = sum(counts)
                symbols = list(seg[i : i + total])
                i += total
                lut = _build_huff(counts, symbols)
                if tc == 0:
                    huff_dc[th] = lut
                else:
                    huff_ac[th] = lut

    if frame is None or scan is None:
        raise ValueError("Unsupported or malformed JPEG")

    w, h = frame["w"], frame["h"]
    total_pixels = w * h
    step = 1
    if total_pixels > max_pixels:
        step = int(math.sqrt(total_pixels / max_pixels)) + 1

    max_h = max(c["h"] for c in frame["comps"].values())
    max_v = max(c["v"] for c in frame["comps"].values())
    mcu_w = 8 * max_h
    mcu_h = 8 * max_v
    mcus_x = (w + mcu_w - 1) // mcu_w
    mcus_y = (h + mcu_h - 1) // mcu_h

    y_cid = 1 if 1 in frame["comps"] else list(frame["comps"].keys())[0]

    scan_map = {}
    for cid, td, ta in scan["comps"]:
        scan_map[cid] = {"td": td, "ta": ta}

    if y_cid not in scan_map:
        return 127.5

    y_h = frame["comps"][y_cid]["h"]
    y_v = frame["comps"][y_cid]["v"]
    y_q = qtables.get(frame["comps"][y_cid]["tq"])
    if y_q is None:
        raise ValueError("Missing quant table")

    y_dc_lut = huff_dc.get(scan_map[y_cid]["td"])
    y_ac_lut = huff_ac.get(scan_map[y_cid]["ta"])
    if y_dc_lut is None or y_ac_lut is None:
        raise ValueError("Missing Huffman tables")

    br = _BitReader(data, scan["entropy_pos"])
    prev_dc = 0

    s = 0.0
    cnt = 0

    for my in range(mcus_y):
        for mx in range(mcus_x):
            for vy in range(y_v):
                for hx in range(y_h):
                    block = [0] * 64

                    t = _huff_decode(br, y_dc_lut)
                    diff = _extend_sign(br.getbits(t) if t else 0, t)
                    dc = prev_dc + diff
                    prev_dc = dc
                    block[0] = dc

                    k = 1
                    while k < 64:
                        rs = _huff_decode(br, y_ac_lut)
                        r = (rs >> 4) & 0xF
                        ss = rs & 0xF
                        if rs == 0:
                            break
                        if rs == 0xF0:
                            k += 16
                            continue
                        k += r
                        if k >= 64:
                            break
                        acv = _extend_sign(br.getbits(ss) if ss else 0, ss)
                        block[_ZZ[k]] = acv
                        k += 1

                    for i in range(64):
                        block[i] = block[i] * y_q[i]

                    pix = _idct8(block)

                    bx0 = mx * mcu_w + hx * 8
                    by0 = my * mcu_h + vy * 8

                    for yy in range(8):
                        y_img = by0 + yy
                        if y_img >= h or (y_img % step) != 0:
                            continue
                        row_off = yy * 8
                        for xx in range(8):
                            x_img = bx0 + xx
                            if x_img >= w or (x_img % step) != 0:
                                continue
                            val = pix[row_off + xx] + 128.0
                            if val < 0.0:
                                val = 0.0
                            if val > 255.0:
                                val = 255.0
                            s += val
                            cnt += 1

    if cnt == 0:
        return 127.5
    return s / cnt


def image_to_gray_mean(path, max_pixels=256 * 256):
    data = _read_bytes(path)
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return _png_gray_mean_from_bytes(data, max_pixels=max_pixels)
    if len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8:
        return _jpeg_gray_mean_from_bytes(data, max_pixels=max_pixels)

    try:
        import tkinter as tk  # stdlib
        from tkinter import PhotoImage  # PNG/GIF support (not JPEG)

        root = tk.Tk()
        root.withdraw()
        img = PhotoImage(file=path)
        w, h = img.width(), img.height()
        total = w * h
        step = 1
        if total > max_pixels:
            step = int(math.sqrt(total / max_pixels)) + 1
        s = 0
        cnt = 0
        for y in range(0, h, step):
            for x in range(0, w, step):
                rgb = img.get(x, y)
                if isinstance(rgb, tuple):
                    if len(rgb) >= 3:
                        r, g, b = rgb[0], rgb[1], rgb[2]
                    else:
                        r = g = b = int(rgb[0])
                else:
                    r = int(rgb[1:3], 16)
                    g = int(rgb[3:5], 16)
                    b = int(rgb[5:7], 16)
                gray = (299 * r + 587 * g + 114 * b) // 1000
                s += gray
                cnt += 1
        root.destroy()
        return s / max(1, cnt)
    except Exception:
        return 127.5




## === cell 3
def list_pngs(folder):
    if not os.path.isdir(folder):
        return []
    files = []
    for name in os.listdir(folder):
        if (
            name.lower().endswith(".jpg")
            or name.lower().endswith(".jpeg")
            or name.lower().endswith(".png")
        ):
            files.append(os.path.join(folder, name))
    files.sort()
    return files


def mean_to_bin(m, bins):
    b = int(m * bins / 256.0)
    if b < 0:
        b = 0
    if b >= bins:
        b = bins - 1
    return b


def fit_hist_nb(cat_files, dog_files, bins=32, laplace=1.0):
    cat_hist = [0.0] * bins
    dog_hist = [0.0] * bins

    n_cat = len(cat_files)
    n_dog = len(dog_files)
    total = n_cat + n_dog
    prior_cat = n_cat / total if total else 0.5
    prior_dog = n_dog / total if total else 0.5

    for p in cat_files:
        m = image_to_gray_mean(p)
        cat_hist[mean_to_bin(m, bins)] += 1.0
    for p in dog_files:
        m = image_to_gray_mean(p)
        dog_hist[mean_to_bin(m, bins)] += 1.0

    cat_den = sum(cat_hist) + laplace * bins
    dog_den = sum(dog_hist) + laplace * bins
    cat_prob = [(c + laplace) / cat_den for c in cat_hist]
    dog_prob = [(d + laplace) / dog_den for d in dog_hist]

    return {
        "bins": bins,
        "prior_cat": prior_cat,
        "prior_dog": prior_dog,
        "cat_prob": cat_prob,
        "dog_prob": dog_prob,
    }


def predict_proba_dog(model, img_path):
    m = image_to_gray_mean(img_path)
    b = mean_to_bin(m, model["bins"])

    log_cat = math.log(max(model["prior_cat"], 1e-12)) + math.log(
        max(model["cat_prob"][b], 1e-12)
    )
    log_dog = math.log(max(model["prior_dog"], 1e-12)) + math.log(
        max(model["dog_prob"][b], 1e-12)
    )

    mx = max(log_cat, log_dog)
    p_cat = math.exp(log_cat - mx)
    p_dog = math.exp(log_dog - mx)
    prob_dog = p_dog / (p_cat + p_dog)

    if prob_dog < 1e-6:
        prob_dog = 1e-6
    if prob_dog > 1 - 1e-6:
        prob_dog = 1 - 1e-6
    return prob_dog




## === cell 4
model = None

cat_files_all = list_pngs(cfg.train_cat_dir)
dog_files_all = list_pngs(cfg.train_dog_dir)

n = min(cfg.max_train_per_class, len(cat_files_all), len(dog_files_all))
cat_files = cat_files_all[:n]
dog_files = dog_files_all[:n]

if n == 0:
    model = None
else:
    model = fit_hist_nb(cat_files, dog_files, bins=cfg.bins, laplace=cfg.laplace)

print("Train files used per class:", n)
print("Model ready:", model is not None)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2856508845.py in <cell line: 0>()
     13     model = None
     14 else:
---> 15     model = fit_hist_nb(cat_files, dog_files, bins=cfg.bins, laplace=cfg.laplace)
     16 
     17 print("Train files used per class:", n)

/tmp/ipykernel_55/205190415.py in fit_hist_nb(cat_files, dog_files, bins, laplace)
     34 
     35     for p in cat_files:
---> 36         m = image_to_gray_mean(p)
     37         cat_hist[mean_to_bin(m, bins)] += 1.0
     38     for p in dog_files:

/tmp/ipykernel_55/652132741.py in image_to_gray_mean(path, max_pixels)
    525     # JPEG
    526     if len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8:
--> 527         return _jpeg_gray_mean_from_bytes(data, max_pixels=max_pixels)
    528 
    529     # Fallback: try tkinter for PNG/GIF (some environments have it); otherwise return mid-gray

/tmp/ipykernel_55/652132741.py in _jpeg_gray_mean_from_bytes(data, max_pixels)
    461                     block = [0] * 64
    462 
--> 463                     t = _huff_decode(br, y_dc_lut)
    464                     diff = _extend_sign(br.getbits(t) if t else 0, t)
    465                     dc = prev_dc + diff

/tmp/ipykernel_55/652132741.py in _huff_decode(br, lut)
    205         if sym is not None:
    206             return sym
--> 207     raise ValueError("Invalid Huffman code")
    208 
    209 

ValueError: Invalid Huffman code

## === cell 5
sample = pd.read_csv(cfg.sample_sub_path)
sample["id"] = pd.to_numeric(sample["id"])
sample = sample.sort_values("id").reset_index(drop=True)

test_dir = cfg.test_dir
if not os.path.isdir(test_dir):
    alt = os.path.join(cfg.input_dir, "test", "test", "unknown")
    if os.path.isdir(alt):
        test_dir = alt

id_to_path = {}
for fn in os.listdir(test_dir) if os.path.isdir(test_dir) else []:
    m = re.match(r"^(\d+)\.(jpg|jpeg|png)$", fn.lower())
    if m:
        id_to_path[int(m.group(1))] = os.path.join(test_dir, fn)

labels = []
missing = 0
for _id in sample["id"].tolist():
    p = id_to_path.get(int(_id))
    if p is None or (not os.path.exists(p)) or model is None:
        labels.append(0.5)
        missing += 1
    else:
        try:
            labels.append(predict_proba_dog(model, p))
        except Exception:
            labels.append(0.5)
            missing += 1

sample["label"] = labels

os.makedirs(cfg.working_dir, exist_ok=True)
sample.to_csv(cfg.out_path, index=False)

print("Wrote:", cfg.out_path)
print("Missing test images or model fallback count:", missing)
print(sample.head())



## === cell 6
submission = pd.read_csv(cfg.out_path)
print(submission.shape)
print(submission.head())
