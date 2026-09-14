# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os, glob, zipfile, random, io, struct, math, hashlib
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("✅ Imports OK (no TensorFlow)")
print("Python OK")



## === cell 1
train_zip_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
test_zip_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"


def _list_jpg_members(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [
            n
            for n in zf.namelist()
            if n.lower().endswith(".jpg") and not n.endswith("/")
        ]
    return names


train_members = _list_jpg_members(train_zip_path)
test_members = _list_jpg_members(test_zip_path)

print(f"Resolved train from ZIP: {train_zip_path} ({len(train_members)} jpg members)")
print(f"Resolved test from ZIP:  {test_zip_path} ({len(test_members)} jpg members)")

if len(train_members) == 0:
    raise FileNotFoundError(f"No training images found inside zip: {train_zip_path}")
if len(test_members) == 0:
    raise FileNotFoundError(f"No test images found inside zip: {test_zip_path}")

print("✅ Paths OK")




## === cell 2
def infer_labels_vectorized(paths):
    paths = np.asarray(paths, dtype=object)
    bases = np.char.lower(np.array([os.path.basename(p) for p in paths], dtype=str))
    parents = np.char.lower(
        np.array([os.path.basename(os.path.dirname(p)) for p in paths], dtype=str)
    )
    is_dog = np.char.find(bases, "dog") >= 0
    is_dog |= parents == "dog"
    is_cat = np.char.find(bases, "cat") >= 0
    is_cat |= parents == "cat"
    labels = np.where(is_dog, 1, np.where(is_cat, 0, 0)).astype(np.int32)
    return labels.tolist()


labels = infer_labels_vectorized(train_members)

idx = np.arange(len(train_members))
idx_dog = idx[np.array(labels) == 1]
idx_cat = idx[np.array(labels) == 0]
rng = np.random.RandomState(SEED)
rng.shuffle(idx_dog)
rng.shuffle(idx_cat)

val_frac = 0.15
n_val_dog = int(round(len(idx_dog) * val_frac))
n_val_cat = int(round(len(idx_cat) * val_frac))

val_idx = np.concatenate([idx_dog[:n_val_dog], idx_cat[:n_val_cat]])
train_idx = np.concatenate([idx_dog[n_val_dog:], idx_cat[n_val_cat:]])
rng.shuffle(val_idx)
rng.shuffle(train_idx)

train_paths = [train_members[i] for i in train_idx]
val_paths = [train_members[i] for i in val_idx]
train_labels = [labels[i] for i in train_idx]
val_labels = [labels[i] for i in val_idx]

print("✅ Split OK")
print(
    "Train:",
    len(train_paths),
    "Val:",
    len(val_paths),
    "Pos rate train:",
    float(np.mean(train_labels)),
)



## === cell 3

ZIGZAG = np.array(
    [
        0,
        1,
        5,
        6,
        14,
        15,
        27,
        28,
        2,
        4,
        7,
        13,
        16,
        26,
        29,
        42,
        3,
        8,
        12,
        17,
        25,
        30,
        41,
        43,
        9,
        11,
        18,
        24,
        31,
        40,
        44,
        53,
        10,
        19,
        23,
        32,
        39,
        45,
        52,
        54,
        20,
        22,
        33,
        38,
        46,
        51,
        55,
        60,
        21,
        34,
        37,
        47,
        50,
        56,
        59,
        61,
        35,
        36,
        48,
        49,
        57,
        58,
        62,
        63,
    ],
    dtype=np.int32,
)
_ZIGZAG_INV = np.empty(64, dtype=np.int32)
_ZIGZAG_INV[ZIGZAG] = np.arange(64, dtype=np.int32)


def _read_u16be(b, p):
    return (b[p] << 8) | b[p + 1]


_DCT_N = 8
_DCT_x = np.arange(_DCT_N, dtype=np.float32)
_DCT_u = np.arange(_DCT_N, dtype=np.float32)
_DCT_C = np.cos(
    ((2.0 * _DCT_x[:, None] + 1.0) * _DCT_u[None, :] * np.pi) / 16.0
).astype(np.float32)
_DCT_alpha = np.ones((_DCT_N,), dtype=np.float32)
_DCT_alpha[0] = 1.0 / np.sqrt(2.0)
_IDCT_M = (_DCT_C * _DCT_alpha[None, :]).astype(np.float32)


def _idct8(block):
    return (0.25 * (_IDCT_M @ block @ _IDCT_M.T)).astype(np.float32)


class _BitReader:
    __slots__ = ("data", "i", "bitbuf", "bitcnt")

    def __init__(self, data):
        self.data = data
        self.i = 0
        self.bitbuf = 0
        self.bitcnt = 0

    def _fill(self):
        if self.i >= len(self.data):
            raise EOFError("Unexpected end of stream")
        b = self.data[self.i]
        self.i += 1
        if b == 0xFF:
            if self.i >= len(self.data):
                raise EOFError("Unexpected end after 0xFF")
            nxt = self.data[self.i]
            self.i += 1
            if nxt != 0x00:
                self.i -= 2
                raise ValueError("Marker encountered in entropy data")
        self.bitbuf = (self.bitbuf << 8) | b
        self.bitcnt += 8

    def get_bits(self, n):
        while self.bitcnt < n:
            self._fill()
        self.bitcnt -= n
        return (self.bitbuf >> self.bitcnt) & ((1 << n) - 1)

    def get_bit(self):
        return self.get_bits(1)


def _extend_sign(v, t):
    if t == 0:
        return 0
    vt = 1 << (t - 1)
    if v < vt:
        return v - (1 << t) + 1
    return v


def _build_huff_table(lengths, symbols):
    codes = {}
    code = 0
    k = 0
    for i in range(16):
        for _ in range(lengths[i]):
            codes[(code, i + 1)] = symbols[k]
            code += 1
            k += 1
        code <<= 1
    by_len = [None] * 17
    for (c, l), sym in codes.items():
        if by_len[l] is None:
            by_len[l] = {}
        by_len[l][c] = sym
    return by_len


def _huff_decode(br, table_by_len):
    code = 0
    for l in range(1, 17):
        code = (code << 1) | br.get_bit()
        m = table_by_len[l]
        if m is not None and code in m:
            return m[code]
    raise ValueError("Invalid Huffman code")


def _downsample_mean_boxes(Y, h, w, out_hw):
    oh, ow = out_hw
    ys = np.linspace(0, h, oh + 1).astype(np.int32)
    xs = np.linspace(0, w, ow + 1).astype(np.int32)
    dy = ys[1:] - ys[:-1]
    dx = xs[1:] - xs[:-1]
    if np.any(dy <= 0):
        ys = ys.copy()
        for i in range(oh):
            if ys[i + 1] <= ys[i]:
                ys[i + 1] = min(h, ys[i] + 1)
    if np.any(dx <= 0):
        xs = xs.copy()
        for j in range(ow):
            if xs[j + 1] <= xs[j]:
                xs[j + 1] = min(w, xs[j] + 1)

    ii = np.empty((h + 1, w + 1), dtype=np.float32)
    ii[0, :] = 0.0
    ii[:, 0] = 0.0
    np.cumsum(Y, axis=0, dtype=np.float32, out=ii[1:, 1:])
    np.cumsum(ii[1:, 1:], axis=1, dtype=np.float32, out=ii[1:, 1:])

    y0 = ys[:-1][:, None]
    y1 = ys[1:][:, None]
    x0 = xs[:-1][None, :]
    x1 = xs[1:][None, :]

    sums = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    areas = (y1 - y0) * (x1 - x0)
    return (sums / areas.astype(np.float32)).astype(np.float32)


def _parse_jpeg_to_gray_thumb(jpeg_bytes, out_hw=(32, 32)):
    b = jpeg_bytes
    if b[0:2] != b"\xFF\xD8":
        raise ValueError("Not a JPEG")
    p = 2

    qtables = {}
    htables_dc = {}
    htables_ac = {}
    sof = None
    sos = None
    scan_data = None
    comps = {}
    restart_interval = 0

    while p < len(b):
        if b[p] != 0xFF:
            p += 1
            continue
        while p < len(b) and b[p] == 0xFF:
            p += 1
        if p >= len(b):
            break
        marker = b[p]
        p += 1
        if marker == 0xD9:
            break
        if marker == 0xDA:
            Ls = _read_u16be(b, p)
            p += 2
            ns = b[p]
            p += 1
            scan_comps = []
            for _ in range(ns):
                cid = b[p]
                tdta = b[p + 1]
                p += 2
                td = (tdta >> 4) & 0x0F
                ta = tdta & 0x0F
                scan_comps.append((cid, td, ta))
            ss = b[p]
            se = b[p + 1]
            ah_al = b[p + 2]
            p += 3
            sos = (scan_comps, ss, se, ah_al)
            scan_data = b[p:]
            break
        if marker == 0xDD:
            L = _read_u16be(b, p)
            p += 2
            restart_interval = _read_u16be(b, p)
            p += 2
            continue
        if marker in [0x01] or (0xD0 <= marker <= 0xD7):
            continue
        L = _read_u16be(b, p)
        p += 2
        seg = b[p : p + L - 2]
        p += L - 2

        if marker == 0xDB:
            sp = 0
            while sp < len(seg):
                pq_tq = seg[sp]
                sp += 1
                pq = (pq_tq >> 4) & 0x0F
                tq = pq_tq & 0x0F
                if pq != 0:
                    raise ValueError("16-bit quant not supported")
                qt = np.frombuffer(seg[sp : sp + 64], dtype=np.uint8).astype(np.int32)
                sp += 64
                qmat = np.empty(64, dtype=np.int32)
                qmat[ZIGZAG] = qt
                qtables[tq] = qmat.reshape(8, 8)
        elif marker == 0xC0:
            precision = seg[0]
            h = _read_u16be(seg, 1)
            w = _read_u16be(seg, 3)
            nf = seg[5]
            sp = 6
            for _ in range(nf):
                cid = seg[sp]
                hv = seg[sp + 1]
                tq = seg[sp + 2]
                sp += 3
                comps[cid] = {"h": (hv >> 4) & 0x0F, "v": hv & 0x0F, "tq": tq}
            sof = (w, h, precision, nf)
        elif marker == 0xC4:
            sp = 0
            while sp < len(seg):
                tc_th = seg[sp]
                sp += 1
                tc = (tc_th >> 4) & 0x0F
                th = tc_th & 0x0F
                lengths = list(seg[sp : sp + 16])
                sp += 16
                total = sum(lengths)
                symbols = list(seg[sp : sp + total])
                sp += total
                table = _build_huff_table(lengths, symbols)
                if tc == 0:
                    htables_dc[th] = table
                else:
                    htables_ac[th] = table

    if sof is None or sos is None or scan_data is None:
        raise ValueError("Missing SOF/SOS")
    w, h, _, nf = sof
    scan_comps, ss, se, ah_al = sos
    if ss != 0 or se != 63:
        raise ValueError("Progressive/unsupported scan")
    if nf < 1:
        raise ValueError("No components")

    y_cid = scan_comps[0][0]
    if y_cid not in comps:
        y_cid = next(iter(comps.keys()))

    max_h = max(c["h"] for c in comps.values())
    max_v = max(c["v"] for c in comps.values())
    mcu_w = 8 * max_h
    mcu_h = 8 * max_v
    mcus_x = (w + mcu_w - 1) // mcu_w
    mcus_y = (h + mcu_h - 1) // mcu_h

    br = _BitReader(scan_data)
    dc_pred = {cid: 0 for cid in comps.keys()}

    scan_info = {cid: (td, ta) for cid, td, ta in scan_comps}

    comp_items = []
    for cid, cinfo in comps.items():
        td, ta = scan_info.get(cid, (0, 0))
        q = qtables.get(cinfo["tq"])
        dc_table = htables_dc.get(td)
        ac_table = htables_ac.get(ta)
        if q is None or dc_table is None or ac_table is None:
            raise ValueError("Missing table(s)")
        comp_items.append(
            (cid, cinfo["h"], cinfo["v"], q.astype(np.float32), dc_table, ac_table)
        )

    Y = np.zeros((h, w), dtype=np.float32)

    coef = np.zeros(64, dtype=np.int32)
    tmp = np.empty(64, dtype=np.float32)

    try:
        for my in range(mcus_y):
            base_y_mcu = my * mcu_h
            for mx in range(mcus_x):
                base_x_mcu = mx * mcu_w
                for cid, hfac, vfac, qf, dc_table, ac_table in comp_items:
                    for vy in range(vfac):
                        for hx in range(hfac):
                            t = _huff_decode(br, dc_table)
                            diff = _extend_sign(br.get_bits(t), t)
                            dc = dc_pred[cid] + diff
                            dc_pred[cid] = dc

                            coef.fill(0)
                            coef[0] = dc
                            k = 1
                            while k < 64:
                                rs = _huff_decode(br, ac_table)
                                if rs == 0x00:
                                    break
                                if rs == 0xF0:
                                    k += 16
                                    continue
                                r = (rs >> 4) & 0x0F
                                s = rs & 0x0F
                                k += r
                                if k >= 64:
                                    break
                                aval = _extend_sign(br.get_bits(s), s)
                                coef[k] = aval
                                k += 1

                            if cid != y_cid:
                                continue

                            tmp[_ZIGZAG_INV] = coef.astype(np.float32, copy=False)
                            block = tmp.reshape(8, 8) * qf

                            pix = _idct8(block) + 128.0
                            pix = np.clip(pix, 0.0, 255.0)

                            x0 = base_x_mcu + hx * 8
                            y0 = base_y_mcu + vy * 8
                            x1 = min(w, x0 + 8)
                            y1 = min(h, y0 + 8)
                            Y[y0:y1, x0:x1] = pix[: (y1 - y0), : (x1 - x0)]
    except (EOFError, ValueError):
        return np.full(out_hw, 127.0, dtype=np.float32) / 255.0

    thumb = _downsample_mean_boxes(Y, h, w, out_hw)
    return thumb / 255.0


IMG_SIZE = 32

_ZIP_HANDLES = {}


def _get_zip_handle(zip_path):
    zf = _ZIP_HANDLES.get(zip_path)
    if zf is None:
        zf = zipfile.ZipFile(zip_path, "r")
        _ZIP_HANDLES[zip_path] = zf
    return zf


def _read_zip_member(zip_path, member_path):
    zf = _get_zip_handle(zip_path)
    return zf.read(member_path)


def featurize_member(zip_path, member_path):
    jb = _read_zip_member(zip_path, member_path)
    thumb = _parse_jpeg_to_gray_thumb(jb, out_hw=(IMG_SIZE, IMG_SIZE))
    return thumb.reshape(-1).astype(np.float32)


print("✅ JPEG decoder + featurizer ready")




## === cell 4
def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def train_logreg(X, y, lr=0.5, epochs=200, l2=1e-4, batch_size=256, seed=SEED):
    rng = np.random.RandomState(seed)
    n, d = X.shape
    w = np.zeros(d, dtype=np.float32)
    b = np.float32(0.0)

    y = y.astype(np.float32, copy=False)
    idx = np.arange(n, dtype=np.int32)
    for ep in range(epochs):
        rng.shuffle(idx)
        for s in range(0, n, batch_size):
            bi = idx[s : s + batch_size]
            xb = X[bi]
            yb = y[bi]
            p = sigmoid(xb @ w + b)
            err = p - yb
            gw = (xb.T @ err) / len(bi) + l2 * w
            gb = np.mean(err)
            w -= lr * gw.astype(np.float32, copy=False)
            b -= lr * np.float32(gb)
    return w, b


def logloss(y_true, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _cache_key(zip_path, members, img_size):
    h = hashlib.sha1()
    h.update(zip_path.encode("utf-8"))
    h.update(str(img_size).encode("utf-8"))
    for m in members:
        h.update(m.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


import multiprocessing as mp


def _featurize_worker(args):
    zip_path, member_path = args
    return featurize_member(zip_path, member_path)


def build_features(zip_path, members):
    cache_dir = "/kaggle/working/feat_cache"
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(zip_path, members, IMG_SIZE)
    cache_path = os.path.join(cache_dir, f"gray{IMG_SIZE}_{key}.npz")

    if os.path.exists(cache_path):
        with np.load(cache_path) as z:
            X = z["X"]
        X = np.asarray(X, dtype=np.float32)
        if X.shape != (len(members), IMG_SIZE * IMG_SIZE):
            raise ValueError(
                "Cached features have unexpected shape; refusing to use cache."
            )
        print(f"Loaded cached features: {cache_path}  shape={X.shape}")
        return X

    n = len(members)
    X = np.zeros((n, IMG_SIZE * IMG_SIZE), dtype=np.float32)

    cpu = os.cpu_count() or 2
    workers = max(1, min(cpu, 8))
    chunksize = 32

    tasks = [(zip_path, m) for m in members]
    if workers == 1:
        for i, t in enumerate(tasks):
            X[i] = _featurize_worker(t)
            if (i + 1) % 2000 == 0:
                print(f"Featurized {i+1}/{n}")
    else:
        ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context("spawn")
        with ctx.Pool(processes=workers, maxtasksperchild=200) as pool:
            for i, feat in enumerate(
                pool.imap(_featurize_worker, tasks, chunksize=chunksize)
            ):
                X[i] = feat
                if (i + 1) % 2000 == 0:
                    print(f"Featurized {i+1}/{n}")

    np.savez_compressed(cache_path, X=X)
    print(f"Saved cached features: {cache_path}  shape={X.shape}")
    return X


print("Building train/val features...")
X_train = build_features(train_zip_path, train_paths)
X_val = build_features(train_zip_path, val_paths)
y_train = np.array(train_labels, dtype=np.float32)
y_val = np.array(val_labels, dtype=np.float32)

mean = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
std = X_train.std(axis=0, dtype=np.float64).astype(np.float32)
std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

X_train_s = (X_train - mean) / std
X_val_s = (X_val - mean) / std

print("Training logistic regression...")
w, b = train_logreg(
    X_train_s, y_train, lr=0.3, epochs=220, l2=2e-4, batch_size=256, seed=SEED
)

p_val = sigmoid(X_val_s @ w + b)
print("✅ Train OK")
print(
    "Val logloss:",
    logloss(y_val, p_val),
    "Val acc:",
    float(np.mean((p_val >= 0.5) == (y_val >= 0.5))),
)



## === cell 5
np.savez("/kaggle/working/logreg_gray32.npz", w=w, b=b, mean=mean, std=std)
print("✅ Saved model params")



## === cell 6
print("✅ Skipped plots to save time")



## === cell 7
test_basenames = np.array([os.path.basename(p) for p in test_members], dtype=str)
test_ids = np.char.partition(test_basenames, ".")[:, 0].astype(np.int64)
order = np.argsort(test_ids)

test_paths = [test_members[i] for i in order]
test_ids_sorted = test_ids[order].tolist()

print("Building test features...")
X_test = build_features(test_zip_path, test_paths)
X_test_s = (X_test - mean) / std

preds = sigmoid(X_test_s @ w + b).astype(np.float64)
preds = np.clip(preds, 0.005, 0.995)

print("✅ Predict OK")



## === cell 8
print(f"预测最大值：{preds.max():.4f}")
print(f"预测最小值：{preds.min():.4f}")
print(f"预测均值：{preds.mean():.4f}")
print("✅ Pred stats OK")



## === cell 9
submission = pd.DataFrame({"id": test_ids_sorted, "label": preds})
submission = submission.sort_values("id").reset_index(drop=True)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print(f"✅ Wrote submission: {out_path}  shape={submission.shape}")
print(submission.head())



## === cell 10
import shutil


def safe_rmtree(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            print(f"Removed: {path}")
        except Exception as e:
            print(f"Skip removing {path}, reason: {e}")
    else:
        print(f"Not found, skip: {path}")


safe_rmtree("/kaggle/working/train")
safe_rmtree("/kaggle/working/test")

for _p, _zf in list(_ZIP_HANDLES.items()):
    try:
        _zf.close()
    except Exception:
        pass
_ZIP_HANDLES.clear()

print("✅ Cleanup OK")
