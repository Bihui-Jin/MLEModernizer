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
import os, glob, time, hashlib
import pandas as pd

BASE_CANDIDATES = [
    "/kaggle/input/hotel-id-2021-fgvc8",
    "/kaggle/input",
    "/kaggle/data/hotel-id-2021-fgvc8",
    "/kaggle/data",
]
base = None
for b in BASE_CANDIDATES:
    if os.path.exists(b):
        base = b
        break
if base is None:
    raise FileNotFoundError("Could not find Kaggle input/data base directory.")


def pick_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


train_csv = pick_existing(
    os.path.join(base, "train.csv"),
    "/kaggle/input/train.csv",
    "/kaggle/data/train.csv",
)
sample_csv = pick_existing(
    os.path.join(base, "sample_submission.csv"),
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
)

test_images_dir = pick_existing(
    os.path.join(base, "test_images"),
    "/kaggle/input/test_images",
    "/kaggle/data/test_images",
)
if test_images_dir is None:
    cand = glob.glob(os.path.join(base, "**", "test_images"), recursive=True)
    test_images_dir = cand[0] if cand else None

train_images_dir = pick_existing(
    os.path.join(base, "train_images"),
    "/kaggle/input/train_images",
    "/kaggle/data/train_images",
)
if train_images_dir is None:
    cand = glob.glob(os.path.join(base, "**", "train_images"), recursive=True)
    train_images_dir = cand[0] if cand else None

print("Using:")
print(" base:", base)
print(" train_csv:", train_csv)
print(" sample_csv:", sample_csv)
print(" test_images_dir:", test_images_dir)
print(" train_images_dir:", train_images_dir)

if train_csv is None or sample_csv is None:
    raise FileNotFoundError(
        "Missing required CSV files (train.csv / sample_submission.csv)."
    )
if test_images_dir is None:
    raise FileNotFoundError("Could not locate test_images directory.")
if train_images_dir is None:
    raise FileNotFoundError("Could not locate train_images directory.")




## === cell 1
from PIL import Image
import numpy as np

_DHASH_CACHE = {}


def dhash_image(path, hash_size=8):
    cached = _DHASH_CACHE.get(path, None)
    if cached is not None or path in _DHASH_CACHE:
        return cached
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
            a = np.asarray(im, dtype=np.uint8)  # shape (hash_size, hash_size+1)
        diff = a[:, :-1] > a[:, 1:]
        bits = np.packbits(diff.reshape(-1).astype(np.uint8), bitorder="little")
        out = int.from_bytes(bits.tobytes(), byteorder="little", signed=False)
        _DHASH_CACHE[path] = out
        return out
    except Exception:
        _DHASH_CACHE[path] = None
        return None


def hamming64(a, b):
    return (a ^ b).bit_count()


train = pd.read_csv(train_csv, usecols=["image", "chain", "hotel_id"])
train["image"] = train["image"].astype(str)
train["chain"] = train["chain"].astype(str)
train["hotel_id"] = train["hotel_id"].astype(str)

vc_global = train["hotel_id"].value_counts()
global_top5 = vc_global.sort_values(ascending=False).head(5).index.tolist()
global_top5 = list(dict.fromkeys(global_top5))
while len(global_top5) < 5:
    global_top5.append(global_top5[-1])
global_top5 = global_top5[:5]
global_pred_str = " ".join(global_top5)
print("Global top-5 hotel_id:", global_pred_str)

chain_top5 = {}
gb = train.groupby("chain", sort=False)["hotel_id"].value_counts()
for ch, s in gb.groupby(level=0, sort=False):
    top = s.droplevel(0).sort_values(ascending=False).head(5).index.astype(str).tolist()
    top = list(dict.fromkeys(top + global_top5))[:5]
    while len(top) < 5:
        top.append(top[-1])
    chain_top5[str(ch)] = top
print("Precomputed chains:", len(chain_top5))


def _iter_jpg_files(root_dir):
    for d, _, files in os.walk(root_dir):
        for fn in files:
            if fn.endswith(".jpg"):
                yield os.path.join(d, fn)


all_test_files = list(_iter_jpg_files(test_images_dir))
if len(all_test_files) == 0:
    raise FileNotFoundError(f"No .jpg found under {test_images_dir}")
print("Found test jpg files:", len(all_test_files))

test_path_by_image = {os.path.basename(p): p for p in all_test_files}


def infer_chain_from_path(p: str):
    d = os.path.dirname(p)
    while True:
        name = os.path.basename(d)
        if name.isdigit():
            return name
        parent = os.path.dirname(d)
        if parent == d or parent == "":
            return None
        d = parent


def infer_parent_folder_key(p: str):
    parent = os.path.basename(os.path.dirname(p))
    return parent if parent else None


train_path_by_image = {}
constructed_missing = 0
for img, ch in zip(train["image"].tolist(), train["chain"].tolist()):
    p = os.path.join(train_images_dir, str(ch), img)
    if os.path.exists(p):
        train_path_by_image[img] = p
    else:
        constructed_missing += 1

if constructed_missing:
    missing_imgs = [
        img for img in train["image"].tolist() if img not in train_path_by_image
    ]
    t_walk = time.time()
    fallback_map = {}
    missing_set = set(missing_imgs)
    for p in _iter_jpg_files(train_images_dir):
        bn = os.path.basename(p)
        if bn in missing_set and bn not in fallback_map:
            fallback_map[bn] = p
            if len(fallback_map) == len(missing_set):
                break
    for bn, p in fallback_map.items():
        train_path_by_image[bn] = p
    print(
        f"Fallback train path search: recovered {len(fallback_map)}/{len(missing_set)} in {time.time()-t_walk:.1f}s"
    )

print(
    "Train images with on-disk path lookup:", len(train_path_by_image), "/", len(train)
)


train_folder_key_by_image = {}
missing_train_paths = 0
for img in train["image"].tolist():
    p = train_path_by_image.get(img)
    if p is None:
        missing_train_paths += 1
        continue
    k = infer_parent_folder_key(p)
    if k is not None:
        train_folder_key_by_image[img] = k

print("Train images missing on-disk path lookup:", missing_train_paths, "/", len(train))
print("Train images with inferred folder key:", len(train_folder_key_by_image))

tmp = train[["image", "hotel_id"]].copy()
tmp["folder_key"] = tmp["image"].map(train_folder_key_by_image)
tmp = tmp.dropna(subset=["folder_key"])
folder_key_top5 = {}
gb_fk = tmp.groupby("folder_key", sort=False)["hotel_id"].value_counts()
for fk, s in gb_fk.groupby(level=0, sort=False):
    top = s.droplevel(0).sort_values(ascending=False).head(5).index.astype(str).tolist()
    top = list(dict.fromkeys(top + global_top5))[:5]
    while len(top) < 5:
        top.append(top[-1])
    folder_key_top5[str(fk)] = top
print("Precomputed folder_key groups:", len(folder_key_top5))

test_chain_by_image = {}
test_folder_key_by_image = {}
unknown_chain_paths = 0
unknown_folder_keys = 0

for p in all_test_files:
    img = os.path.basename(p)

    ch = infer_chain_from_path(p)
    if ch is None:
        unknown_chain_paths += 1
    else:
        test_chain_by_image[img] = ch

    fk = infer_parent_folder_key(p)
    if fk is None:
        unknown_folder_keys += 1
    else:
        test_folder_key_by_image[img] = fk

print("Test images with chain not inferred from path:", unknown_chain_paths)
print("Test images with inferred chain:", len(test_chain_by_image))
print("Test images with missing parent-folder key:", unknown_folder_keys)
print("Test images with inferred parent-folder key:", len(test_folder_key_by_image))

MAX_TRAIN_HASH = 30000  # keep unchanged
NN_K = 25
NN_TOP5 = 5


def keep_train_image(img, modulo=3):
    h = int(hashlib.md5(img.encode("utf-8")).hexdigest()[:8], 16)
    return (h % modulo) == 0


t0 = time.time()
train_hash_db = []  # list of (hash:int, hotel_id:str)
hashed = 0
seen_paths = 0
modulo = 3

for row in train[["image", "hotel_id"]].itertuples(index=False):
    img, hid = row
    p = train_path_by_image.get(img)
    if p is None:
        continue
    seen_paths += 1
    if not keep_train_image(img, modulo=modulo):
        continue
    h = dhash_image(p)
    if h is None:
        continue
    train_hash_db.append((h, hid))
    hashed += 1
    if hashed >= MAX_TRAIN_HASH:
        break

print(
    f"Train dHash DB size: {len(train_hash_db)} (cap {MAX_TRAIN_HASH}), built in {time.time()-t0:.1f}s"
)

if len(train_hash_db) > 0:
    _train_hash_arr = np.array([h for h, _ in train_hash_db], dtype=np.uint64)
    _train_hid_arr = np.array([hid for _, hid in train_hash_db], dtype=object)
else:
    _train_hash_arr = np.empty((0,), dtype=np.uint64)
    _train_hid_arr = np.empty((0,), dtype=object)

_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_vec_u64(query_u64, db_u64_arr):
    x = np.bitwise_xor(db_u64_arr, np.uint64(query_u64)).view(np.uint8).reshape(-1, 8)
    return _POPCOUNT8[x].sum(axis=1, dtype=np.uint16)


sample = pd.read_csv(sample_csv)
if "image" not in sample.columns:
    raise ValueError("sample_submission.csv must contain 'image' column.")

sample_images = sample["image"].astype(str).tolist()
priors = []
for img in sample_images:
    ch = test_chain_by_image.get(img)
    if ch is not None and ch in chain_top5:
        priors.append(chain_top5[ch])
    else:
        fk = test_folder_key_by_image.get(img)
        if fk is not None and fk in folder_key_top5:
            priors.append(folder_key_top5[fk])
        else:
            priors.append(global_top5)

if _train_hid_arr.size:
    uniq_hids, inv_codes = np.unique(_train_hid_arr.astype(str), return_inverse=True)
else:
    uniq_hids = np.empty((0,), dtype=str)
    inv_codes = np.empty((0,), dtype=np.int32)

preds = []
missing_file = 0
used_nn = 0
used_chain = 0
used_folder_key = 0
used_global = 0

t1 = time.time()
for img, prior in zip(sample_images, priors):
    p = test_path_by_image.get(img)
    if p is None:
        missing_file += 1

    top = None
    if p is not None and _train_hash_arr.size > 0:
        qh = dhash_image(p)
        if qh is not None:
            d = hamming_vec_u64(qh, _train_hash_arr)

            k = NN_K if NN_K < d.size else d.size
            idx = np.argpartition(d, k - 1)[:k]
            idx = idx[np.argsort(d[idx], kind="mergesort")]

            codes = inv_codes[idx]
            counts = np.bincount(codes, minlength=uniq_hids.size)
            nz = np.flatnonzero(counts)
            if nz.size:
                hids_nz = uniq_hids[nz]
                cnts_nz = counts[nz]
                order = np.lexsort((hids_nz, -cnts_nz))
                nn_top = hids_nz[order][:NN_TOP5].tolist()
            else:
                nn_top = []

            top = list(dict.fromkeys(nn_top + prior + global_top5))[:5]
            used_nn += 1

    if top is None:
        ch = test_chain_by_image.get(img)
        if ch is not None and ch in chain_top5:
            used_chain += 1
        else:
            fk = test_folder_key_by_image.get(img)
            if fk is not None and fk in folder_key_top5:
                used_folder_key += 1
            else:
                used_global += 1
        top = list(dict.fromkeys(prior + global_top5))[:5]

    while len(top) < 5:
        top.append(top[-1])
    preds.append(" ".join(top))

print(f"Prediction loop time: {time.time()-t1:.1f}s for {len(sample)} rows")
print("Rows using NN (dHash) top-5:", used_nn, "/", len(sample))
print("Rows using chain-conditioned top-5 (no NN):", used_chain, "/", len(sample))
print(
    "Rows using folder-key-conditioned top-5 (no NN):",
    used_folder_key,
    "/",
    len(sample),
)
print("Rows using global fallback top-5 (no NN):", used_global, "/", len(sample))
print(
    "Rows whose image filename not found under test_images (still predicted):",
    missing_file,
    "/",
    len(sample),
)

sub = pd.DataFrame({"image": sample["image"].astype(str), "hotel_id": preds})
assert len(sub) == len(sample), (len(sub), len(sample))

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", sub.shape)




## === cell 2
check = pd.read_csv("submission.csv")
print(check.head())
print("Columns:", list(check.columns))
print("n_rows:", len(check))
assert list(check.columns) == ["image", "hotel_id"]
assert (
    check["hotel_id"].astype(str).str.split().map(len).eq(5).all()
), "Each prediction must have 5 space-delimited ids"
