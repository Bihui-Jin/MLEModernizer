# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Identify hotels from images.

## Metric
Mean Average Precision @ 5 (MAP@5)

## Submission Format
For each image in the test set, you must predict a space-delimited list of hotel IDs that could match that image. The first ID should be the most relevant one and the last the least relevant one. The file should contain a header and have the following format:

```
image,hotel_id
99e91ad5f2870678.jpg,36363 53586 18807 64314 60181
b5cc62ab665591a9.jpg,36363 53586 18807 64314 60181
d5664a972d5a644b.jpg,36363 53586 18807 64314 60181
```

## Dataset
**train.csv** - The training set metadata.

- `image` - The image ID.

- `chain` - An ID code for the hotel chain. A `chain` of zero (0) indicates that the hotel is either not part of a chain or the chain is not known. This field is not available for the test set. The number of hotels per chain varies widely.

- `hotel_id` - The hotel ID. The target class.

- `timestamp` - When the image was taken. Provided for the training set only.

**sample_submission.csv** - A sample submission file in the correct format.

- `image` The image ID

- `hotel_id` The hotel ID. The target class.

**train_images** - The training set contains 97000+ images from around 7700 hotels from across the globe. All of the images for each hotel chain are in a dedicated subfolder for that chain.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 13,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
            train/
                train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        input/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
                    test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
            train/
                train/
                    train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        working/
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
```

-> data/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> data/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> input/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> input/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import pandas as pd

BASE_CANDIDATES = [
    "/kaggle/input/hotel-id-2021-fgvc8",
    "/kaggle/data/hotel-id-2021-fgvc8",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(relpath: str) -> str:
    for base in BASE_CANDIDATES:
        p = os.path.join(base, relpath)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {relpath} under any of: {BASE_CANDIDATES}")


def find_dir(relpath: str) -> str:
    for base in BASE_CANDIDATES:
        p = os.path.join(base, relpath)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        f"Could not find dir {relpath} under any of: {BASE_CANDIDATES}"
    )


train_csv = find_file("train.csv")
sample_csv = find_file("sample_submission.csv")

train = pd.read_csv(train_csv)
sample = pd.read_csv(sample_csv)

assert {"image", "hotel_id"}.issubset(
    train.columns
), f"train columns: {train.columns.tolist()}"
assert {"image", "hotel_id"}.issubset(
    sample.columns
), f"sample columns: {sample.columns.tolist()}"

print("train:", train.shape, "sample:", sample.shape)
print("train head:\n", train.head())
print("sample head:\n", sample.head())



## === cell 1
import numpy as np
from PIL import Image, ImageOps
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

Image.MAX_IMAGE_PIXELS = None
try:
    Image.draft
except Exception:
    pass

TRAIN_IMG_CANDIDATES = [
    "train_images",
    "hotel-id-2021-fgvc8/train_images",
    "hotel-id-2021-fgvc8/train_images/train_images",
]
TEST_IMG_CANDIDATES = [
    "test_images",
    "hotel-id-2021-fgvc8/test_images",
    "hotel-id-2021-fgvcvc8/test_images/test_images",  # harmless if missing
    "hotel-id-2021-fgvc8/test_images/test_images",
]

train_img_dir = None
for rp in TRAIN_IMG_CANDIDATES:
    try:
        train_img_dir = find_dir(rp)
        break
    except FileNotFoundError:
        pass
if train_img_dir is None:
    raise FileNotFoundError("Could not locate train_images directory.")

test_img_dir = None
for rp in TEST_IMG_CANDIDATES:
    try:
        test_img_dir = find_dir(rp)
        break
    except FileNotFoundError:
        pass
if test_img_dir is None:
    raise FileNotFoundError("Could not locate test_images directory.")

print("train_img_dir:", train_img_dir)
print("test_img_dir :", test_img_dir)

top5_global = train["hotel_id"].value_counts().head(5).index.astype(str).tolist()
if len(top5_global) < 5:
    top5_global = top5_global + ["0"] * (5 - len(top5_global))
top5_global_str = " ".join(top5_global[:5])


def image_path_train(chain: int, image_name: str) -> str:
    return os.path.join(train_img_dir, str(chain), image_name)


def image_path_test(image_name: str) -> str:
    return os.path.join(test_img_dir, image_name)


def normalize_for_hash(im: Image.Image) -> Image.Image:
    im = ImageOps.exif_transpose(im)  # apply EXIF orientation consistently
    if im.mode != "RGB":
        im = im.convert("RGB")  # drop alpha/palette to stabilize conversion
    return im


def _dct_matrix(n: int = 32) -> np.ndarray:
    x = np.arange(n, dtype=np.float32)
    u = x[:, None]
    alpha = np.sqrt(2.0 / n) * np.ones(n, dtype=np.float32)
    alpha[0] = np.sqrt(1.0 / n)
    return alpha[:, None] * np.cos((np.pi / (2.0 * n)) * (2.0 * x[None, :] + 1.0) * u)


_DCT32 = _dct_matrix(32).astype(np.float32, copy=False)


def phash256(img: Image.Image) -> np.ndarray:
    g = ImageOps.grayscale(img)
    g = g.resize((32, 32), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    d = _DCT32 @ a @ _DCT32.T  # (32,32)
    low = d[:16, :16].reshape(-1)  # 256
    med = np.median(low[1:])  # exclude DC
    bits = (low > med).astype(np.uint8)
    packed = np.packbits(bits, bitorder="little")
    return np.frombuffer(packed.tobytes(), dtype="<u8", count=4).astype(
        np.uint64, copy=False
    )


_POPCNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_all_u256(query_u256: np.ndarray, train_u256: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(train_u256, query_u256)  # (N,4) uint64
    xb = x.view(np.uint8).reshape(-1, 32)  # 32 bytes per row
    return _POPCNT8[xb].sum(axis=1).astype(np.uint16)


MAX_TRAIN_HASH = 80000  # was 35000

train_small = train.copy()
if len(train_small) > MAX_TRAIN_HASH:
    vc = train_small["hotel_id"].value_counts()
    keep_hotels = set(vc.head(5000).index)  # was 2000; more coverage improves recall
    train_small = train_small[train_small["hotel_id"].isin(keep_hotels)]
    if len(train_small) > MAX_TRAIN_HASH:
        train_small = train_small.iloc[:MAX_TRAIN_HASH].copy()

print("Using train images for hashing:", len(train_small))


def _hash_train_row(row):
    img_name = row.image
    chain = row.chain
    hid = str(row.hotel_id)
    p = image_path_train(chain, img_name)
    try:
        with Image.open(p) as im:
            im = normalize_for_hash(im)
            try:
                im.draft("L", (32, 32))
            except Exception:
                pass
            h = phash256(im)
        return (h, hid)
    except Exception:
        return None


train_hashes_list = []
train_hotels_list = []

hash_to_hotel_votes = defaultdict(Counter)

failed = 0

rows_iter = train_small.itertuples(index=False)

max_workers = min(32, (os.cpu_count() or 8) * 2)
chunksize = 64
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for out in ex.map(_hash_train_row, rows_iter, chunksize=chunksize):
        if out is None:
            failed += 1
            continue
        h, hid = out
        train_hashes_list.append(h)
        train_hotels_list.append(hid)
        hash_to_hotel_votes[(int(h[0]), int(h[1]))][hid] += 1

if train_hashes_list:
    train_hashes = np.vstack(train_hashes_list).astype(np.uint64, copy=False)  # (N,4)
    train_hotels = np.array(train_hotels_list, dtype=object)
else:
    train_hashes = np.empty((0, 4), dtype=np.uint64)
    train_hotels = np.empty((0,), dtype=object)

print("Computed train hashes:", len(train_hashes), "failed:", failed)

_BUCKET_BITS = 16  # was 14
if len(train_hashes) > 0:
    k0 = (train_hashes[:, 0] >> np.uint64(64 - _BUCKET_BITS)).astype(
        np.uint32, copy=False
    )
    k1 = (train_hashes[:, 1] >> np.uint64(64 - _BUCKET_BITS)).astype(
        np.uint32, copy=False
    )
    _bucket_key = (k0 ^ k1).astype(np.uint32, copy=False)

    order = np.argsort(_bucket_key, kind="mergesort")
    bucket_sorted = _bucket_key[order]
    hashes_sorted = train_hashes[order]
    hotels_sorted = train_hotels[order]
else:
    bucket_sorted = np.empty((0,), dtype=np.uint32)
    hashes_sorted = train_hashes
    hotels_sorted = train_hotels

nb = 1 << _BUCKET_BITS
counts = np.bincount(bucket_sorted.astype(np.int64), minlength=nb).astype(
    np.int32, copy=False
)
bucket_starts = np.concatenate(([0], np.cumsum(counts[:-1]))).astype(
    np.int32, copy=False
)
bucket_ends = (bucket_starts + counts).astype(np.int32, copy=False)


def _candidate_slices_for_query(th_u256: np.ndarray, target_n: int):
    k0 = int(th_u256[0] >> np.uint64(64 - _BUCKET_BITS))
    k1 = int(th_u256[1] >> np.uint64(64 - _BUCKET_BITS))
    bkey = (k0 ^ k1) & ((1 << _BUCKET_BITS) - 1)

    total = 0
    slices = []
    step = 0
    while total < target_n and (bkey - step >= 0 or bkey + step < nb):
        if step == 0:
            s = int(bucket_starts[bkey])
            e = int(bucket_ends[bkey])
            if e > s:
                slices.append((s, e))
                total += e - s
            step = 1
            continue

        left = bkey - step
        right = bkey + step

        if left >= 0:
            s = int(bucket_starts[left])
            e = int(bucket_ends[left])
            if e > s:
                slices.append((s, e))
                total += e - s

        if total >= target_n:
            break

        if right < nb:
            s = int(bucket_starts[right])
            e = int(bucket_ends[right])
            if e > s:
                slices.append((s, e))
                total += e - s

        step += 1
        if step > nb:
            break

    return slices


if len(train_hashes) < 1000:
    print("Too few train hashes computed; falling back to global top5 baseline.")
    sub = sample[["image"]].copy()
    sub["hotel_id"] = top5_global_str
    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv:", sub.shape)
else:
    KNN = 200  # was 120
    BEST = 5

    sub = sample[["image"]].copy()
    preds = []

    def _hash_test_image(img_name: str):
        p = image_path_test(img_name)
        try:
            with Image.open(p) as im:
                im = normalize_for_hash(im)
                try:
                    im.draft("L", (32, 32))
                except Exception:
                    pass
                th = phash256(im)
            return (img_name, th, None)
        except Exception as e:
            return (img_name, None, e)

    test_names = sub["image"].tolist()
    test_hash_by_name = {}
    test_failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for img_name, th, err in ex.map(
            _hash_test_image, test_names, chunksize=chunksize
        ):
            if err is not None or th is None:
                test_failed += 1
                test_hash_by_name[img_name] = None
            else:
                test_hash_by_name[img_name] = th

    pred_cache = {}

    for img_name in test_names:
        th = test_hash_by_name.get(img_name)
        if th is None:
            preds.append(top5_global_str)
            continue

        cache_key = (int(th[0]), int(th[1]), int(th[2]), int(th[3]))
        cached = pred_cache.get(cache_key)
        if cached is not None:
            preds.append(cached)
            continue

        votes = Counter()

        exact = hash_to_hotel_votes.get((int(th[0]), int(th[1])))
        if exact:
            votes.update(exact)

        if len(votes) < 20:
            target = max(KNN, 4096)
            cand_slices = _candidate_slices_for_query(th, target)

            if not cand_slices:
                best = top5_global[:BEST]
            else:
                if len(cand_slices) == 1:
                    s, e = cand_slices[0]
                    cand_hashes = hashes_sorted[s:e]
                    dists = hamming_all_u256(th, cand_hashes)
                    k = min(KNN, max(1, len(dists)))
                    idx_local = np.argpartition(dists, k - 1)[:k]
                    cand_hotels = hotels_sorted[s:e]

                    sel_hotels = cand_hotels[idx_local].astype(str, copy=False)
                    sel_w = 257 - dists[idx_local].astype(np.int32, copy=False)
                    sel_w[sel_w < 1] = 1
                    uh, inv = np.unique(sel_hotels, return_inverse=True)
                    sums = np.bincount(inv, weights=sel_w, minlength=uh.size).astype(
                        np.int64
                    )
                    for h, ssum in zip(uh.tolist(), sums.tolist()):
                        votes[h] += int(ssum)
                else:
                    best_d = None
                    best_h = None
                    for s, e in cand_slices:
                        cand_hashes = hashes_sorted[s:e]
                        d = hamming_all_u256(th, cand_hashes)
                        if d.size == 0:
                            continue
                        kk = min(KNN, d.size)
                        loc = np.argpartition(d, kk - 1)[:kk]
                        d_k = d[loc]
                        h_k = hotels_sorted[s:e][loc].astype(str, copy=False)

                        if best_d is None:
                            best_d = d_k
                            best_h = h_k
                        else:
                            best_d = np.concatenate([best_d, d_k])
                            best_h = np.concatenate([best_h, h_k])

                    if best_d is None or best_d.size == 0:
                        best = top5_global[:BEST]
                    else:
                        k = min(KNN, best_d.size)
                        top = np.argpartition(best_d, k - 1)[:k]
                        sel_hotels = best_h[top]
                        sel_w = 257 - best_d[top].astype(np.int32, copy=False)
                        sel_w[sel_w < 1] = 1
                        uh, inv = np.unique(sel_hotels, return_inverse=True)
                        sums = np.bincount(
                            inv, weights=sel_w, minlength=uh.size
                        ).astype(np.int64)
                        for h, ssum in zip(uh.tolist(), sums.tolist()):
                            votes[h] += int(ssum)

        best = [hid for hid, _ in votes.most_common(BEST)]
        for hid in top5_global:
            if len(best) >= BEST:
                break
            if hid not in best:
                best.append(hid)
        if len(best) < BEST:
            best += ["0"] * (BEST - len(best))
        pred_str = " ".join(best[:BEST])
        pred_cache[cache_key] = pred_str
        preds.append(pred_str)

    sub["hotel_id"] = preds

    assert sub.shape[0] == sample.shape[0]
    assert list(sub.columns) == ["image", "hotel_id"]
    assert sub["hotel_id"].str.split().str.len().eq(5).all()

    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv:", sub.shape, "test_failed:", test_failed)
    print(sub.head())



## === cell 2
import subprocess

subprocess.run(
    ["bash", "-lc", "ls -lha submission.csv && head -n 5 submission.csv"], check=False
)
