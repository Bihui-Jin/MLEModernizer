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
from PIL import Image

import numpy as np

CANDIDATE_SAMPLE_PATHS = [
    "/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv",
    "/kaggle/data/hotel-id-2021-fgvc8/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]

CANDIDATE_TRAIN_PATHS = [
    "/kaggle/input/hotel-id-2021-fgvc8/train.csv",
    "/kaggle/data/hotel-id-2021-fgvc8/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/train.csv",
]

CANDIDATE_TRAIN_IMAGE_DIRS = [
    "/kaggle/input/hotel-id-2021-fgvc8/train_images",
    "/kaggle/data/hotel-id-2021-fgvc8/train_images",
    "/kaggle/input/train_images",
    "/kaggle/data/train_images",
]

CANDIDATE_TEST_IMAGE_DIRS = [
    "/kaggle/input/hotel-id-2021-fgvc8/test_images",
    "/kaggle/data/hotel-id-2021-fgvc8/test_images",
    "/kaggle/input/test_images",
    "/kaggle/data/test_images",
]


def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


sample_path = find_first_existing(CANDIDATE_SAMPLE_PATHS)
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations: "
        + ", ".join(CANDIDATE_SAMPLE_PATHS)
    )

train_path = find_first_existing(CANDIDATE_TRAIN_PATHS)
if train_path is None:
    raise FileNotFoundError(
        "Could not find train.csv in expected locations: "
        + ", ".join(CANDIDATE_TRAIN_PATHS)
    )

train_img_root = find_first_existing(CANDIDATE_TRAIN_IMAGE_DIRS)
if train_img_root is None:
    raise FileNotFoundError(
        "Could not find train_images directory in expected locations: "
        + ", ".join(CANDIDATE_TRAIN_IMAGE_DIRS)
    )

test_img_root = find_first_existing(CANDIDATE_TEST_IMAGE_DIRS)
if test_img_root is None:
    raise FileNotFoundError(
        "Could not find test_images directory in expected locations: "
        + ", ".join(CANDIDATE_TEST_IMAGE_DIRS)
    )

sample = pd.read_csv(sample_path)
required_cols = ["image", "hotel_id"]
missing = [c for c in required_cols if c not in sample.columns]
if missing:
    raise ValueError(f"sample_submission.csv missing required columns: {missing}")
sample = sample[required_cols].copy()
sample["image"] = sample["image"].astype(str)

train_meta = pd.read_csv(train_path, usecols=["image", "chain", "hotel_id"])
train_meta["chain"] = train_meta["chain"].fillna(0).astype(int)
train_meta["hotel_id"] = train_meta["hotel_id"].astype(str)
train_meta["image"] = train_meta["image"].astype(str)

global_counts = train_meta["hotel_id"].value_counts()
global_top = global_counts.head(300).index.tolist()
if len(global_top) == 0:
    raise ValueError("train.csv seems empty or hotel_id could not be read.")

chain_counts_map = {}
chain_top_map = {}
for chain_id, grp in train_meta.groupby("chain", sort=False):
    vc = grp["hotel_id"].value_counts()
    chain_counts_map[int(chain_id)] = vc
    chain_top_map[int(chain_id)] = vc.head(300).index.tolist()

K = 5

_chain_pred_cache = {}


def _precompute_chain_pred_cache():
    chain_ids = set(chain_counts_map.keys())
    chain_ids.add(0)
    for chain_id in chain_ids:
        cvc = chain_counts_map.get(chain_id, None)
        cand = []
        if cvc is not None:
            cand.extend(chain_top_map.get(chain_id, [])[:300])
        cand.extend(global_top[:300])

        seen = set()
        uniq = []
        for x in cand:
            if x not in seen:
                seen.add(x)
                uniq.append(x)

        if cvc is None:
            out = uniq[:K]
        else:
            scored = []
            for hid in uniq:
                chain_cnt = float(cvc.get(hid, 0.0))
                glob_cnt = float(global_counts.get(hid, 0.0))
                score = 1_000_000.0 * chain_cnt + glob_cnt
                scored.append((score, hid))
            scored.sort(key=lambda t: (-t[0], t[1]))
            out = [hid for _, hid in scored[:K]]

        if len(out) < K:
            for hid in global_top:
                if hid not in out:
                    out.append(hid)
                if len(out) == K:
                    break
        if len(out) < K:
            out = (out * K)[:K]

        _chain_pred_cache[int(chain_id)] = out


_precompute_chain_pred_cache()


def predict_topk_for_chain(chain_id: int):
    return _chain_pred_cache.get(int(chain_id), _chain_pred_cache[0])


def build_test_indexes_from_sample(root_dir: str, sample_images):
    img2path = {}
    img2chain = (
        {}
    )  # chain not available for test; keep identical semantics by always using 0
    join = os.path.join
    exists = os.path.exists
    for fn in sample_images:
        p = join(root_dir, fn)
        if exists(p):
            img2path[fn] = p
        img2chain[fn] = 0
    return img2path, img2chain


_RESAMPLE_BILINEAR = getattr(
    getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR
)


def dhash64_from_path(path: str):
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((9, 8), _RESAMPLE_BILINEAR)
            a = np.asarray(im, dtype=np.uint8)  # shape (8, 9)
            bits = (a[:, :-1] > a[:, 1:]).reshape(-1)
            h = int(
                np.packbits(bits.astype(np.uint8), bitorder="little").view(np.uint64)[0]
            )
            return h
    except Exception:
        return None


PC_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

sample_imgs = sample["image"].to_numpy(dtype=object)
test_img_paths, img2chain_fs = build_test_indexes_from_sample(
    test_img_root, sample_imgs
)

print(f"Using sample_submission.csv from: {sample_path}")
print(f"Using train.csv from: {train_path}")
print(f"Using train_images root from: {train_img_root}")
print(f"Using test_images root from: {test_img_root}")
print(f"Indexed test image paths (from sample list): {len(test_img_paths)}")
print(f"Built img2chain_fs size: {len(img2chain_fs)}")

TRAIN_HASH_PER_HOTEL = 2
MAX_TRAIN_HASHES = 20000  # time cap
train_meta_small = train_meta.copy()

train_meta_small["_rn"] = train_meta_small.groupby("hotel_id").cumcount()
train_meta_small = train_meta_small[
    train_meta_small["_rn"] < TRAIN_HASH_PER_HOTEL
].drop(columns=["_rn"])
if len(train_meta_small) > MAX_TRAIN_HASHES:
    train_meta_small = train_meta_small.iloc[:MAX_TRAIN_HASHES].copy()

hash_bank = []  # list of (hash_int, hotel_id)
hashed_train = 0
missing_train_files = 0

imgs_arr = train_meta_small["image"].to_numpy(dtype=object)
chains_arr = train_meta_small["chain"].to_numpy(dtype=np.int32)
hids_arr = train_meta_small["hotel_id"].to_numpy(dtype=object)

join = os.path.join
exists = os.path.exists
for img, chain, hid in zip(imgs_arr, chains_arr, hids_arr):
    p = join(train_img_root, str(int(chain)), str(img))
    if not exists(p):
        missing_train_files += 1
        continue
    h = dhash64_from_path(p)
    if h is None:
        continue
    hash_bank.append((h, hid))
    hashed_train += 1

print(
    f"Train hash bank size: {len(hash_bank)} (hashed_train={hashed_train}, missing_train_files={missing_train_files})"
)

NN = 25  # number of nearest train hashes to vote from (kept modest for speed)

if len(hash_bank) > 0:
    train_hashes = np.array([h for h, _ in hash_bank], dtype=np.uint64)
    train_hids = np.array([hid for _, hid in hash_bank], dtype=object)
else:
    train_hashes = np.empty((0,), dtype=np.uint64)
    train_hids = np.empty((0,), dtype=object)

if train_hashes.size:
    train_bytes = train_hashes.view(np.uint8).reshape(-1, 8)
else:
    train_bytes = None

global_counts_f = global_counts.astype(float).to_dict()
n_test = sample_imgs.shape[0]

test_hashes = np.empty((n_test,), dtype=np.uint64)
test_hash_valid = np.zeros((n_test,), dtype=bool)
missing_test_file = 0
hash_fail = 0
used_fs_chain = 0
missing_chain = 0

img2chain_get = img2chain_fs.get
test_paths_get = test_img_paths.get

for i, img in enumerate(sample_imgs):
    c = int(img2chain_get(img, 0))
    if c != 0:
        used_fs_chain += 1
    else:
        missing_chain += 1

    tp = test_paths_get(img)
    if tp is None:
        missing_test_file += 1
        continue
    th = dhash64_from_path(tp)
    if th is None:
        hash_fail += 1
        continue
    test_hashes[i] = np.uint64(th)
    test_hash_valid[i] = True


def _build_byte_inverted_index(train_b: np.ndarray):
    inv = [None] * 8
    for j in range(8):
        buckets = [[] for _ in range(256)]
        col = train_b[:, j]
        for i, v in enumerate(col.tolist()):
            buckets[v].append(i)
        inv[j] = buckets
    return inv


def _hamm64_from_bytes(xb: np.ndarray, yb: np.ndarray, pc_lut: np.ndarray) -> int:
    return int(pc_lut[np.bitwise_xor(xb, yb)].sum())


def _top_nn_indices_for_test_bytewise(
    test_b: np.ndarray,
    train_b: np.ndarray,
    inv,
    nn_eff: int,
    pc_lut: np.ndarray,
    max_radius_bits: int = 64,
):
    n_train = train_b.shape[0]
    candidates = None

    def add_bucket(dst_set, idx_list):
        for idx in idx_list:
            dst_set.add(idx)

    for r in range(0, 9):  # byte mismatches: 0..8
        need = 8 - r
        cand_set = set()
        for j in range(8):
            add_bucket(cand_set, inv[j][int(test_b[j])])
            if len(cand_set) >= nn_eff and need <= 7:
                break

        if len(cand_set) >= nn_eff:
            candidates = np.fromiter(cand_set, dtype=np.int32, count=len(cand_set))
            break

    if candidates is None:
        candidates = np.arange(n_train, dtype=np.int32)

    tb = test_b
    cb = train_b[candidates]  # (m,8)
    dists = pc_lut[np.bitwise_xor(cb, tb)].sum(axis=1).astype(np.int16, copy=False)
    if dists.shape[0] > nn_eff:
        part = np.argpartition(dists, nn_eff - 1)[:nn_eff]
        candidates = candidates[part]
        dists = dists[part]
    return candidates.astype(np.int32, copy=False), dists.astype(np.int16, copy=False)


nn_idx = None
nn_dist = None
valid_pos = np.flatnonzero(test_hash_valid)
n_valid = int(valid_pos.size)

if train_hashes.size and n_valid:
    nn_eff = min(int(NN), int(train_hashes.size))
    nn_idx = np.empty((n_valid, nn_eff), dtype=np.int32)
    nn_dist = np.empty((n_valid, nn_eff), dtype=np.int16)

    inv = _build_byte_inverted_index(train_bytes)
    pc_lut = PC_LUT
    for j, pos in enumerate(valid_pos.tolist()):
        tb = test_hashes[pos].view(np.uint8)
        idx_j, dist_j = _top_nn_indices_for_test_bytewise(
            tb, train_bytes, inv, nn_eff, pc_lut
        )
        if idx_j.shape[0] < nn_eff:
            rep = (nn_eff + idx_j.shape[0] - 1) // idx_j.shape[0]
            idx_j = np.tile(idx_j, rep)[:nn_eff]
            dist_j = np.tile(dist_j, rep)[:nn_eff]
        nn_idx[j] = idx_j
        nn_dist[j] = dist_j

preds = [""] * n_test
used_hash = 0

predict_chain = predict_topk_for_chain
gc_f = global_counts_f
gc_top = global_top
K_local = K

valid_row_index = np.full((n_test,), -1, dtype=np.int32)
if n_valid:
    valid_row_index[valid_pos] = np.arange(n_valid, dtype=np.int32)

for i, img in enumerate(sample_imgs):
    c = int(img2chain_get(img, 0))
    base = predict_chain(c)

    out = []
    seen = set()

    if test_hash_valid[i] and train_hashes.size:
        bi = int(valid_row_index[i])
        if bi != -1:
            idx = nn_idx[bi]
            dist = nn_dist[bi]

            hids = train_hids[idx].astype(object, copy=False)
            order = np.argsort(hids, kind="mergesort")
            hids_s = hids[order]
            dist_s = dist[order]
            order2 = np.argsort(dist_s, kind="mergesort")
            hids_s = hids_s[order2]
            dist_s = dist_s[order2]

            cnt = {}
            bestd = {}
            for d, hid in zip(dist_s.tolist(), hids_s.tolist()):
                cnt[hid] = cnt.get(hid, 0) + 1
                if hid not in bestd or d < bestd[hid]:
                    bestd[hid] = d

            voted = list(cnt.keys())
            voted.sort(
                key=lambda hid: (
                    -cnt[hid],
                    bestd.get(hid, 10**9),
                    -float(gc_f.get(hid, 0.0)),
                    hid,
                )
            )
            for hid in voted:
                if hid not in seen:
                    seen.add(hid)
                    out.append(hid)
                if len(out) == K_local:
                    break
            if len(out) > 0:
                used_hash += 1

    for hid in base:
        if hid not in seen:
            seen.add(hid)
            out.append(hid)
        if len(out) == K_local:
            break

    if len(out) < K_local:
        for hid in gc_top:
            if hid not in seen:
                seen.add(hid)
                out.append(hid)
            if len(out) == K_local:
                break
    if len(out) < K_local:
        out = (out * K_local)[:K_local]

    preds[i] = " ".join(out[:K_local])

sub = pd.DataFrame({"image": sample["image"], "hotel_id": preds})
out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print(
    f"Signals usage: hash for {used_hash}/{len(sample)} rows; "
    f"fs-chain for {used_fs_chain}/{len(sample)} rows; "
    f"missing/0-chain for {missing_chain}/{len(sample)} rows"
)
print(
    f"Missing test files in index: {missing_test_file}, hash read failures: {hash_fail}"
)
print(f"Global fallback top-5: {gc_top[:5]}")
print(f"Wrote {out_path} with shape={sub.shape}")
print(sub.head())



## === cell 1
import pandas as pd

sub = pd.read_csv("submission.csv")
assert list(sub.columns) == [
    "image",
    "hotel_id",
], "Submission columns must be exactly: image, hotel_id"
assert sub["image"].notna().all(), "All image values must be present"
assert sub["hotel_id"].notna().all(), "All hotel_id predictions must be present"
assert (
    sub["hotel_id"].astype(str).str.len().gt(0).all()
), "hotel_id strings must be non-empty"

counts = sub["hotel_id"].astype(str).str.split().map(len)
assert counts.min() >= 1, "Each prediction must contain at least 1 hotel_id"
assert (
    counts.min() == 5 and counts.max() == 5
), "Each prediction must contain exactly 5 hotel_ids"
print("submission.csv looks valid.")
print(sub.head())
print(
    "Predictions per row (min/mean/max):",
    int(counts.min()),
    float(counts.mean()),
    int(counts.max()),
)
