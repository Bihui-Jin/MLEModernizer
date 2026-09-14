# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.69027

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.69315) has done: 'Your notebook isn’t yielding a Kaggle score because it currently won’t run in a clean environment: it imports packages that are not installed (timm, albumentations, cv2, pytorch_lightning, sklearn), and it also points to an external checkpoint directory that likely doesn’t exist. To get a valid submission while preserving your inference semantics (probability of dog for each test image id), I’m making the smallest possible change: generate a properly formatted `submission.csv` directly from `sample_submission.csv` with a constant, safe probability (0.5) so the pipeline always produces a valid file. This does not try to maximize score; it just unblocks submission generation end-to-end under the “no external packages installed” constraint. Once you confirm a valid score is produced, we can re-enable model inference if/when those dependencies/checkpoints are available.'
- What this solution (achieved 0.69315) has done: 'I fix the crash by replacing the PNG-only decoder with a minimal JPEG/PNG reader that works using only the Python standard library (via `tkinter.PhotoImage` for PNG/GIF and a small baseline-JPEG decoder for typical Kaggle JPGs). I also make sure `model` is always defined (even when training fails) so submission generation never hits a `NameError`. To keep the core logic intact, I preserve the exact histogram Naive Bayes approach and only change image loading/decoding so it can actually read the provided `.jpg` files. Finally, I ensure the submission is written to `/kaggle/working/submission.csv` with `id,label` and sorted IDs.'
- What this solution (achieved 0.69027) has done: 'The timeout is dominated by per-image pure-Python decoding and per-scanline filter reconstruction (PNG) / entropy parsing (JPEG), repeated thousands of times. I keep the same exact feature (grayscale mean) and the same Naive Bayes histogram model, but make the extraction much faster by (1) using Pillow if available (common on Kaggle) to compute the same grayscale mean directly from decoded pixels, and (2) parallelizing feature extraction across CPU cores while keeping deterministic ordering. I also precompute gray means for all train+test images once and reuse them (instead of re-reading/decoding during both fit and predict), which is provably equivalent because the feature is a pure function of the file bytes. Finally, I speed up the histogram counting and prediction loops by avoiding repeated dictionary lookups and Python overhead.'

# 9. Code solution

## === cell 0
import os
import re
import math
import struct
import zlib
import pandas as pd
import multiprocessing as mp

os.environ.setdefault("PYTHONHASHSEED", "0")


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



## === cell 1
_GRAY_MEAN_CACHE = {}

try:
    from PIL import Image  # commonly available on Kaggle

    _HAVE_PIL = True
except Exception:
    Image = None
    _HAVE_PIL = False


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
        data = self.data
        pos = self.pos
        buf = self.buf
        nbits = self.nbits

        while nbits <= 24 and pos < len(data):
            b = data[pos]
            pos += 1

            if b == 0xFF:
                while pos < len(data) and data[pos] == 0xFF:
                    pos += 1
                if pos >= len(data):
                    break

                nxt = data[pos]
                if nxt == 0x00:
                    pos += 1
                    b = 0xFF
                elif 0xD0 <= nxt <= 0xD7:
                    pos += 1
                    continue
                else:
                    pos -= 1
                    break

            buf = (buf << 8) | b
            nbits += 8

        self.pos = pos
        self.buf = buf
        self.nbits = nbits

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


def _build_huff_fast(table_counts, symbols):
    mincode = [-1] * 17
    maxcode = [-1] * 17
    valptr = [0] * 17

    code = 0
    k = 0
    for l in range(1, 17):
        c = table_counts[l - 1]
        if c:
            mincode[l] = code
            valptr[l] = k
            code += c
            maxcode[l] = code - 1
            k += c
        code <<= 1
    return (mincode, maxcode, valptr, symbols)


def _huff_decode_fast(br, tbl):
    mincode, maxcode, valptr, symbols = tbl
    code = 0
    for l in range(1, 17):
        code = (code << 1) | br.getbit()
        if mincode[l] != -1 and code <= maxcode[l]:
            return symbols[valptr[l] + (code - mincode[l])]
    raise ValueError("Invalid Huffman code")


def _jpeg_gray_mean_from_bytes_dc_only(data, max_pixels=256 * 256):
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
        d = data
        n = len(d)
        while pos < n and d[pos] != 0xFF:
            pos += 1
        while pos < n and d[pos] == 0xFF:
            pos += 1
        if pos >= n:
            return None
        m = d[pos]
        pos += 1
        return m

    while True:
        m = read_marker()
        if m is None or m == 0xD9:
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

        if m in (0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0x01):
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
                qtables[tq] = list(seg[i : i + 64])
                i += 64

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
                h_s = (hv >> 4) & 0xF
                v_s = hv & 0xF
                comps[cid] = {"h": h_s, "v": v_s, "tq": tq}
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
                syms = list(seg[i : i + total])
                i += total
                tbl = _build_huff_fast(counts, syms)
                if tc == 0:
                    huff_dc[th] = tbl
                else:
                    huff_ac[th] = tbl

    if frame is None or scan is None:
        raise ValueError("Unsupported or malformed JPEG")

    w, h = frame["w"], frame["h"]
    total_pixels = w * h
    step = 1
    if total_pixels > max_pixels:
        step = int(math.sqrt(total_pixels / max_pixels)) + 1

    y_cid = 1 if 1 in frame["comps"] else next(iter(frame["comps"].keys()))
    y_td = y_ta = None
    for cid, td, ta in scan["comps"]:
        if cid == y_cid:
            y_td, y_ta = td, ta
            break
    if y_td is None:
        return 127.5

    y_comp = frame["comps"][y_cid]
    y_h = y_comp["h"]
    y_v = y_comp["v"]
    y_q = qtables.get(y_comp["tq"])
    if y_q is None:
        raise ValueError("Missing quant table")

    y_dc_tbl = huff_dc.get(y_td)
    y_ac_tbl = huff_ac.get(y_ta)
    if y_dc_tbl is None or y_ac_tbl is None:
        raise ValueError("Missing Huffman tables")

    q00 = y_q[0]

    max_h = 1
    max_v = 1
    for c in frame["comps"].values():
        if c["h"] > max_h:
            max_h = c["h"]
        if c["v"] > max_v:
            max_v = c["v"]
    mcu_w = 8 * max_h
    mcu_h = 8 * max_v
    mcus_x = (w + mcu_w - 1) // mcu_w
    mcus_y = (h + mcu_h - 1) // mcu_h

    br = _BitReader(data, scan["entropy_pos"])
    prev_dc = 0

    s = 0.0
    cnt = 0

    huff_dec = _huff_decode_fast
    getbits = br.getbits
    ext = _extend_sign

    for my in range(mcus_y):
        by_mcu = my * mcu_h
        for mx in range(mcus_x):
            bx_mcu = mx * mcu_w
            for vy in range(y_v):
                by0 = by_mcu + vy * 8
                for hx in range(y_h):
                    bx0 = bx_mcu + hx * 8

                    t = huff_dec(br, y_dc_tbl)
                    diff = ext(getbits(t) if t else 0, t)
                    dc = prev_dc + diff
                    prev_dc = dc

                    k = 1
                    while k < 64:
                        rs = huff_dec(br, y_ac_tbl)
                        if rs == 0:
                            break
                        if rs == 0xF0:
                            k += 16
                            continue
                        r = (rs >> 4) & 0xF
                        ss = rs & 0xF
                        k += r
                        if k >= 64:
                            break
                        if ss:
                            getbits(ss)
                        k += 1

                    if by0 >= h or bx0 >= w:
                        continue
                    if (by0 % step) != 0 or (bx0 % step) != 0:
                        continue

                    mean_block = (dc * q00) / 8.0 + 128.0
                    if mean_block < 0.0:
                        mean_block = 0.0
                    elif mean_block > 255.0:
                        mean_block = 255.0
                    s += mean_block
                    cnt += 1

    if cnt == 0:
        return 127.5
    return s / cnt


def _pil_gray_mean(path, max_pixels=256 * 256):
    with Image.open(path) as im:
        im = im.convert("L")
        w, h = im.size
        total = w * h
        step = 1
        if total > max_pixels:
            step = int(math.sqrt(total / max_pixels)) + 1

        pix = im.tobytes()  # bytes length w*h
        s = 0
        cnt = 0
        for y in range(0, h, step):
            row_off = y * w
            for x in range(0, w, step):
                s += pix[row_off + x]
                cnt += 1
        return s / max(1, cnt)


def image_to_gray_mean(path, max_pixels=256 * 256):
    v = _GRAY_MEAN_CACHE.get(path)
    if v is not None:
        return v

    if _HAVE_PIL:
        try:
            v = _pil_gray_mean(path, max_pixels=max_pixels)
            _GRAY_MEAN_CACHE[path] = v
            return v
        except Exception:
            pass

    data = _read_bytes(path)
    try:
        if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
            v = _png_gray_mean_from_bytes(data, max_pixels=max_pixels)
        elif len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8:
            try:
                v = _jpeg_gray_mean_from_bytes_dc_only(data, max_pixels=max_pixels)
            except Exception:
                v = 127.5
        else:
            v = 127.5
    finally:
        _GRAY_MEAN_CACHE[path] = v
    return v




## === cell 2
_IMG_EXTS = (".jpg", ".jpeg", ".png")


def list_pngs(folder):
    if not os.path.isdir(folder):
        return []
    files = []
    with os.scandir(folder) as it:
        for e in it:
            if not e.is_file():
                continue
            low = e.name.lower()
            if low.endswith(_IMG_EXTS):
                files.append(e.path)
    files.sort()
    return files


def mean_to_bin(m, bins):
    b = int(m * bins / 256.0)
    if b < 0:
        b = 0
    if b >= bins:
        b = bins - 1
    return b


def fit_hist_nb(cat_files, dog_files, bins=32, laplace=1.0, precomputed_means=None):
    cat_hist = [0.0] * bins
    dog_hist = [0.0] * bins

    n_cat = len(cat_files)
    n_dog = len(dog_files)
    total = n_cat + n_dog
    prior_cat = n_cat / total if total else 0.5
    prior_dog = n_dog / total if total else 0.5

    get_mean = precomputed_means.get if precomputed_means is not None else None
    for p in cat_files:
        m = get_mean(p) if get_mean is not None else image_to_gray_mean(p)
        cat_hist[mean_to_bin(m, bins)] += 1.0
    for p in dog_files:
        m = get_mean(p) if get_mean is not None else image_to_gray_mean(p)
        dog_hist[mean_to_bin(m, bins)] += 1.0

    cat_den = sum(cat_hist) + laplace * bins
    dog_den = sum(dog_hist) + laplace * bins
    cat_prob = [(c + laplace) / cat_den for c in cat_hist]
    dog_prob = [(d + laplace) / dog_den for d in dog_hist]

    eps = 1e-12
    log_prior_cat = math.log(max(prior_cat, eps))
    log_prior_dog = math.log(max(prior_dog, eps))
    log_cat_prob = [math.log(max(p, eps)) for p in cat_prob]
    log_dog_prob = [math.log(max(p, eps)) for p in dog_prob]

    return {
        "bins": bins,
        "prior_cat": prior_cat,
        "prior_dog": prior_dog,
        "cat_prob": cat_prob,
        "dog_prob": dog_prob,
        "log_prior_cat": log_prior_cat,
        "log_prior_dog": log_prior_dog,
        "log_cat_prob": log_cat_prob,
        "log_dog_prob": log_dog_prob,
    }


def predict_proba_dog(model, img_path, precomputed_means=None):
    if precomputed_means is not None:
        m = precomputed_means[img_path]
    else:
        m = image_to_gray_mean(img_path)
    b = mean_to_bin(m, model["bins"])

    log_cat = model["log_prior_cat"] + model["log_cat_prob"][b]
    log_dog = model["log_prior_dog"] + model["log_dog_prob"][b]

    mx = max(log_cat, log_dog)
    p_cat = math.exp(log_cat - mx)
    p_dog = math.exp(log_dog - mx)
    prob_dog = p_dog / (p_cat + p_dog)

    if prob_dog < 1e-6:
        prob_dog = 1e-6
    if prob_dog > 1 - 1e-6:
        prob_dog = 1 - 1e-6
    return prob_dog


def _gray_mean_worker(args):
    p, max_pixels = args
    return (p, image_to_gray_mean(p, max_pixels=max_pixels))


def precompute_gray_means(paths, max_pixels=256 * 256):
    paths = list(paths)
    if not paths:
        return {}

    nprocs = min(max(1, (os.cpu_count() or 1)), 8)
    if len(paths) < 256 or nprocs == 1:
        out = {}
        for p in paths:
            out[p] = image_to_gray_mean(p, max_pixels=max_pixels)
        return out

    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
    with ctx.Pool(processes=nprocs, maxtasksperchild=200) as pool:
        out = {}
        for p, m in pool.imap_unordered(
            _gray_mean_worker, ((p, max_pixels) for p in paths), chunksize=64
        ):
            out[p] = m
        return out




## === cell 3
model = None

cat_files_all = list_pngs(cfg.train_cat_dir)
dog_files_all = list_pngs(cfg.train_dog_dir)

n = min(cfg.max_train_per_class, len(cat_files_all), len(dog_files_all))
cat_files = cat_files_all[:n]
dog_files = dog_files_all[:n]

train_paths = cat_files + dog_files
train_means = precompute_gray_means(train_paths, max_pixels=256 * 256)

if n == 0:
    model = None
else:
    model = fit_hist_nb(
        cat_files,
        dog_files,
        bins=cfg.bins,
        laplace=cfg.laplace,
        precomputed_means=train_means,
    )

print("Train files used per class:", n)
print("Model ready:", model is not None)



## === cell 4
sample = pd.read_csv(cfg.sample_sub_path)
sample["id"] = pd.to_numeric(sample["id"])
sample = sample.sort_values("id").reset_index(drop=True)

test_dir = cfg.test_dir
if not os.path.isdir(test_dir):
    alt = os.path.join(cfg.input_dir, "test", "test", "unknown")
    if os.path.isdir(alt):
        test_dir = alt

id_to_path = {}
_pat = re.compile(r"^(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
if os.path.isdir(test_dir):
    with os.scandir(test_dir) as it:
        for e in it:
            if not e.is_file():
                continue
            m = _pat.match(e.name)
            if m:
                id_to_path[int(m.group(1))] = e.path

test_paths = [id_to_path.get(int(_id)) for _id in sample["id"].tolist()]
test_paths_existing = [p for p in test_paths if p is not None and os.path.exists(p)]
test_means = precompute_gray_means(test_paths_existing, max_pixels=256 * 256)

labels = []
missing = 0

_get = id_to_path.get
_exists = os.path.exists
_pred = predict_proba_dog

for _id in sample["id"].tolist():
    p = _get(int(_id))
    if p is None or (not _exists(p)) or model is None:
        labels.append(0.5)
        missing += 1
    else:
        try:
            labels.append(_pred(model, p, precomputed_means=test_means))
        except Exception:
            labels.append(0.5)
            missing += 1

sample["label"] = labels

os.makedirs(cfg.working_dir, exist_ok=True)
sample.to_csv(cfg.out_path, index=False)

print("Wrote:", cfg.out_path)
print("Missing test images or model fallback count:", missing)
print(sample.head())



## === cell 5
submission = pd.read_csv(cfg.out_path)
print(submission.shape)
print(submission.head())
