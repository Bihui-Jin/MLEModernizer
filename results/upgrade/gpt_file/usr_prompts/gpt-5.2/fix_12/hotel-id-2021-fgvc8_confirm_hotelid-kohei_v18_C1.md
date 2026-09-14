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

DATA_CANDIDATES = [
    "/kaggle/input/hotel-id-2021-fgvc8",
    "/kaggle/data/hotel-id-2021-fgvc8",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename: str):
    for base in DATA_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    for base in DATA_CANDIDATES:
        path = os.path.join(base, "hotel-id-2021-fgvc8", filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {filename} under any of: {DATA_CANDIDATES}"
    )


train_path = find_file("train.csv")
sub_path = find_file("sample_submission.csv")

print("Using train:", train_path)
print("Using sample_submission:", sub_path)

train = pd.read_csv(train_path)
sample_sub = pd.read_csv(sub_path)

train.head(), sample_sub.head(), train.shape, sample_sub.shape



## === cell 1
import os

global_top5 = train["hotel_id"].value_counts().head(5).index.astype(str).tolist()
global_pred_str = " ".join(global_top5)
print("Global top-5 fallback:", global_pred_str)

_hotel_cooccur = (
    train.groupby(["hotel_id"], sort=False).size().rename("cnt").reset_index()
)

TRAIN_IMAGES_CANDIDATES = [
    "/kaggle/input/hotel-id-2021-fgvc8/train_images",
    "/kaggle/data/hotel-id-2021-fgvc8/train_images",
    "/kaggle/input/train_images",
    "/kaggle/data/train_images",
    "/kaggle/input/hotel-id-2021-fgvc8/train_images/train_images",
    "/kaggle/data/hotel-id-2021-fgvc8/train_images/train_images",
]
TEST_IMAGES_CANDIDATES = [
    "/kaggle/input/hotel-id-2021-fgvc8/test_images",
    "/kaggle/data/hotel-id-2021-fgvc8/test_images",
    "/kaggle/input/test_images",
    "/kaggle/data/test_images",
    "/kaggle/input/hotel-id-2021-fgvc8/test_images/test_images",
    "/kaggle/data/hotel-id-2021-fgvcvc8/test_images/test_images",
    "/kaggle/data/hotel-id-2021-fgvc8/test_images/test_images",
    "/kaggle/input/test_images/test_images",
    "/kaggle/data/test_images/test_images",
]

train_images_root = None
for p in TRAIN_IMAGES_CANDIDATES:
    if os.path.isdir(p):
        train_images_root = p
        break

test_images_root = None
for p in TEST_IMAGES_CANDIDATES:
    if os.path.isdir(p):
        test_images_root = p
        break

print("Using train_images root:", train_images_root)
print("Using test_images root:", test_images_root)

from PIL import Image
import numpy as np


def _ahash_64(image_path: str):
    """64-bit average hash (8x8) -> Python int; returns None if unreadable."""
    try:
        with Image.open(image_path) as im:
            im = im.convert("L").resize((8, 8), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32).reshape(-1)
        mean = float(arr.mean())
        bits = (arr > mean).astype(np.uint8)
        return int(np.packbits(bits, bitorder="big").view(np.uint64)[0])
    except Exception:
        return None


HASH_TRAIN_CAP = 60000
PER_CHAIN_CAP = 800

_hash_to_hotel_counts = {}
_hash_ready = False

_hash_keys_u64 = None
_hash_keys_py = None
_popcount_lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_u64_vectorized(x_u64: np.ndarray) -> np.ndarray:
    """Exact popcount for uint64 array using byte LUT; deterministic and fast."""
    xb = x_u64.view(np.uint8).reshape(-1, 8)
    return _popcount_lut[xb].sum(axis=1).astype(np.int16)


def _build_existing_train_files_map(train_root: str):
    chain_to_files = {}
    try:
        for ch in os.listdir(train_root):
            ch_dir = os.path.join(train_root, ch)
            if not os.path.isdir(ch_dir):
                continue
            try:
                chain_to_files[ch] = set(os.listdir(ch_dir))
            except Exception:
                continue
    except Exception:
        return {}
    return chain_to_files


from concurrent.futures import ThreadPoolExecutor

if train_images_root is not None and test_images_root is not None:
    try:
        train2 = train.loc[:, ["image", "chain", "hotel_id"]].copy()
        train2["image"] = train2["image"].astype(str)
        train2["chain"] = train2["chain"].astype(int)
        train2["hotel_id"] = train2["hotel_id"].astype(int)

        train2 = (
            train2.sort_values(["chain", "image"], kind="mergesort")
            .groupby("chain", sort=False)
            .head(PER_CHAIN_CAP)
            .reset_index(drop=True)
        )

        if len(train2) > HASH_TRAIN_CAP:
            train2 = train2.iloc[:HASH_TRAIN_CAP].copy()

        train2["chain_str"] = train2["chain"].astype(str)

        existing_map = _build_existing_train_files_map(train_images_root)

        if existing_map:
            exists_mask = [
                (cs in existing_map) and (img in existing_map[cs])
                for cs, img in zip(
                    train2["chain_str"].tolist(), train2["image"].tolist()
                )
            ]
            train2 = train2.loc[exists_mask].reset_index(drop=True)
        else:
            train2["path"] = (
                train_images_root.rstrip("/")
                + "/"
                + train2["chain_str"]
                + "/"
                + train2["image"]
            )
            exists_mask = train2["path"].map(os.path.exists)
            train2 = train2[exists_mask].reset_index(drop=True)

        if "path" not in train2.columns:
            train2["path"] = (
                train_images_root.rstrip("/")
                + "/"
                + train2["chain_str"]
                + "/"
                + train2["image"]
            )

        def _hash_one(path_and_hotel):
            path, hotel_val = path_and_hotel
            h = _ahash_64(path)
            return h, int(hotel_val)

        paths = train2["path"].tolist()
        hotels = train2["hotel_id"].tolist()

        max_workers = min(32, (os.cpu_count() or 4) * 2)

        n_hashed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for h, hid in ex.map(_hash_one, zip(paths, hotels), chunksize=64):
                if h is None:
                    continue
                d = _hash_to_hotel_counts.get(h)
                if d is None:
                    d = {}
                    _hash_to_hotel_counts[h] = d
                d[hid] = d.get(hid, 0) + 1
                n_hashed += 1

        _hash_ready = n_hashed > 0
        if _hash_ready:
            _hash_keys_py = list(_hash_to_hotel_counts.keys())
            _hash_keys_u64 = np.array(_hash_keys_py, dtype=np.uint64)

        print(
            f"Built train hash index: hashed {n_hashed} train images (cap={HASH_TRAIN_CAP}, per_chain={PER_CHAIN_CAP})."
        )
    except Exception as e:
        print(
            "WARNING: hash index build failed; falling back to global-only predictions. Error:",
            repr(e),
        )
        _hash_ready = False
else:
    print(
        "WARNING: train_images/test_images roots not found; falling back to global-only predictions."
    )

_test_hash_cache = {}

TOPK_NEIGHBORS = 25
HAMMING_MAX_DIST = 14


def infer_hotels_for_test_image(image_name: str):
    """Infer ranked hotel_id candidates via hash NN voting; returns list[int] (best-first)."""
    if not _hash_ready or test_images_root is None:
        return []

    hq = _test_hash_cache.get(image_name, "__MISS__")
    if hq == "__MISS__":
        test_path = os.path.join(test_images_root, image_name)
        if not os.path.exists(test_path):
            _test_hash_cache[image_name] = None
            return []
        hq = _ahash_64(test_path)
        _test_hash_cache[image_name] = hq

    if hq is None:
        return []

    if hq in _hash_to_hotel_counts:
        counts = _hash_to_hotel_counts[hq]
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [hid for hid, _ in ranked]

    if _hash_keys_u64 is None or _hash_keys_u64.size == 0:
        return []

    q = np.uint64(hq)
    dists = _popcount_u64_vectorized(_hash_keys_u64 ^ q)

    K = min(TOPK_NEIGHBORS, int(_hash_keys_u64.size))
    nn_idx = np.argpartition(dists, K - 1)[:K]
    nn_idx = nn_idx[np.argsort(dists[nn_idx], kind="mergesort")]

    hotel_votes = {}
    for j in nn_idx:
        dj = int(dists[j])
        if dj > HAMMING_MAX_DIST:
            break
        best_h = int(_hash_keys_u64[int(j)])
        counts = _hash_to_hotel_counts.get(best_h, {})
        w = (HAMMING_MAX_DIST - dj) + 1
        for hid, c in counts.items():
            hotel_votes[hid] = hotel_votes.get(hid, 0) + int(c) * w

    if not hotel_votes:
        return []

    ranked = sorted(hotel_votes.items(), key=lambda kv: (-kv[1], kv[0]))
    return [hid for hid, _ in ranked]


def _safe_hotel_id_str(x):
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return None
            x = int(float(x))
        else:
            if isinstance(x, (float, np.floating)) and not np.isfinite(x):
                return None
            x = int(x)
        return str(x)
    except Exception:
        return None


global_top5_str = [str(x) for x in global_top5]

preds = []
n_any_found = 0
images = sample_sub["image"].astype(str).tolist()

for img in images:
    cands = infer_hotels_for_test_image(img)

    out = []
    seen = set()

    if cands:
        for hid in cands:
            s = _safe_hotel_id_str(hid)
            if s is None:
                continue
            if s not in seen:
                out.append(s)
                seen.add(s)
            if len(out) == 5:
                break
        if len(out) > 0:
            n_any_found += 1

    for s in global_top5_str:
        if s not in seen:
            out.append(s)
            seen.add(s)
        if len(out) == 5:
            break

    if len(out) != 5:
        out = (out + global_top5_str)[:5]
        if len(out) < 5:
            out = (out + [global_top5_str[0]] * 5)[:5]

    preds.append(" ".join(out))

print(
    f"Produced non-empty candidate list for {n_any_found}/{len(sample_sub)} test images."
)

submission = sample_sub.copy()
submission["hotel_id"] = preds

assert list(submission.columns) == ["image", "hotel_id"]
assert len(submission) == len(sample_sub)
assert submission["hotel_id"].astype(str).str.split().map(len).eq(5).all()

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())



## === cell 2
import subprocess, shlex, os

if os.path.exists("submission.csv"):
    print(
        subprocess.check_output(shlex.split("ls -lha submission.csv")).decode("utf-8")
    )
    print(
        subprocess.check_output(shlex.split("head -n 5 submission.csv")).decode("utf-8")
    )
else:
    raise FileNotFoundError("submission.csv was not created.")
