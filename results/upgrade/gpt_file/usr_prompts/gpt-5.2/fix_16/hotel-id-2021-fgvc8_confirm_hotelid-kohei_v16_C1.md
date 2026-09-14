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

# 5. Target score

0.8164877494316718

# 6. Current score

0.00169

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently can’t yield a score because it depends on private wheel files and a `peko` module entrypoint that won’t exist in this environment, so it never reliably produces a valid `submission.csv`. To fix this with minimal, metric-aligned logic, I’m replacing that dependency with a simple, always-runnable baseline that creates a correct submission by predicting the global top-5 most frequent `hotel_id` values from `train.csv` for every test image (a standard fallback for MAP@5). This preserves evaluation semantics (space-delimited 5 IDs per image) and guarantees end-to-end execution within the time limit. Once you can submit and get a real score, we can make further minimal adjustments specifically to close the remaining gap to the target.'
- What this solution (achieved 0.00209) has done: 'Your current submission is far below the target, so we should improve it with the smallest metric-aligned change that preserves the “no-model” baseline core logic. Instead of predicting the same global top-5 hotels for every image, we use the provided `chain` metadata to make chain-conditioned top-5 predictions: for each test image, infer its chain from the folder name under `test_images/`, then predict the most frequent hotels within that chain from `train.csv`. For unknown chains (or chains with too few samples), we safely fall back to the global top-5 list to keep every row valid. This keeps the approach simple and fast, but should substantially raise MAP@5 compared to the unconditional global frequency baseline.'
- What this solution (achieved 0.00209) has done: 'Your current chain-conditioned frequency baseline likely maps most test images to the wrong “chain” because the test folder structure is not the chain id (it’s usually nested like `test_images/test_images/*.jpg`), causing lots of fallbacks and a very low MAP@5. I make the smallest fix: derive each test image’s chain by joining with `train.csv` on `image` (test images in this competition are drawn from known hotels/chains), and only fall back to the global top-5 when an image is truly unseen. This keeps the exact same core logic (top-5 frequency within chain), but fixes the key alignment bug so predictions are actually chain-conditioned. I also make the chain/top-5 padding deterministic and guarantee 5 unique ids per row.'
- What this solution (achieved 0.00209) has done: 'Your current score is extremely low because the “image→chain” lookup uses `train.csv` image names, but test images are different files, so almost every test row falls back to the same global top‑5 list. With minimal change (keeping the same chain-conditioned frequency baseline), we instead infer each test image’s chain from the test image file path (the parent folder name under `test_images/`), which matches the dataset structure. We also ensure the `sample_submission` image list is aligned to actual found test files (so no silent mismatch), while keeping a safe global fallback for any image whose chain cannot be inferred. This should substantially increase MAP@5 toward your target without changing the overall approach.'
- What this solution (achieved 0.0026) has done: 'Your current score is far below target because almost every test image is falling back to the same global top‑5 list (the test images are not stored in numeric chain folders), making MAP@5 near random. With minimal changes and the same “frequency top‑5” core logic, we infer a better per-image grouping signal directly from the test image filename: use its first character (hex-like) to bucket images, then use training data to build top‑5 hotels per bucket and predict accordingly (fallback to global top‑5 when needed). This keeps the approach fast, deterministic, and submission-format correct, but should move the score substantially toward the target versus a near-constant prediction. We also remove the expensive per-row `any(...)` scan over all test files and replace it with an O(1) set lookup (no semantic change, just avoids timeouts).'
- What this solution (achieved 0.00209) has done: 'Your current baseline is low because bucketing by the first character of the filename has almost no relationship to the true hotel, so MAP@5 stays near-random. With minimal changes and the same “top-5 frequency list” core logic, we instead build a more informative per-image prior using the provided `train.csv` metadata: compute top-5 hotels per `chain`, and for each test image, infer its `chain` from the test folder structure (numeric chain subfolders exist in this dataset layout) and use the corresponding chain top-5 (fallback to global top-5 if chain can’t be inferred). This keeps everything deterministic, fast, and submission-format correct, but should move the score substantially upward toward your target compared to filename-bucketing. We also keep the O(1) test filename set check and guarantee exactly 5 unique ids per row.'
- What this solution (achieved 0.00209) has done: 'Your current baseline is near-random because almost all test images have a non-numeric parent folder (so `chain` can’t be inferred), causing a fallback to the same global top‑5 for nearly every row. With minimal changes and the same “frequency top‑5” core logic, we infer chain from the test image *path* by scanning for the nearest numeric directory anywhere in its parents (not just the immediate parent), which matches common dataset layouts like `.../test_images/test_images/<chain>/<img>.jpg`. Then we build a chain-conditioned top‑5 and (critically) backfill chains that have <5 unique hotels using the global list so every row stays valid. This should substantially increase MAP@5 toward your target while keeping runtime low and producing the same submission format.'
- What this solution (achieved 0.00209) has done: 'Your current score is low because most test images can’t have a `chain` inferred from the path, so they fall back to the same global top‑5 for nearly all rows. With minimal change and the same core “frequency top‑5 list” logic, we add a second lightweight conditioning signal that is actually available for test: the test image’s immediate parent directory name (e.g., `test_images/test_images/<folder>/<img>.jpg`). We precompute top‑5 hotels for each train-time folder (train images are stored under `train_images/<chain>/...`, which matches `chain`) and then, for test images, use either an inferred numeric chain (as you already do) or else use the parent folder name as a key to a precomputed top‑5; we still safely fall back to global top‑5. This keeps runtime fast, preserves submission semantics, and should improve MAP@5 vs near-constant predictions by reducing the number of global fallbacks.'
- What this solution (achieved 0.00169) has done: 'The timeout is dominated by computing ~30k train image dHashes with PIL, plus per-test image dHash computation; the nearest-neighbor step itself is relatively cheap. I keep the exact same hashing/NN/prior logic, but make it faster by (1) avoiding repeated pandas-to-list conversions and repeated MD5 computations, (2) parallelizing dHash extraction for train and test images using a deterministic multiprocessing pool, and (3) removing the per-path dHash cache (which doesn’t help when each path is hashed once) to reduce Python dict overhead. I also precompute the “keep” mask in one vectorized pass so we don’t hash/MD5 inside a Python loop, while preserving the exact same modulo selection semantics. All file paths and prediction semantics remain unchanged.'
- What this solution (achieved 0.00169) has done: 'Your current MAP@5 is near-random because the dHash nearest-neighbor stage is effectively broken: you compute nearest indices in the hashed DB, but then use those indices to index `inv_codes` that was built for the *unique-hotel list*, not for the hashed DB rows, so the “NN votes” are wrong. With minimal changes (no new model, same hashing/NN approach), I fix this by storing hotel-id codes per hashed row and voting on those codes directly. I also make the dHash type consistent (force uint64 end-to-end) to avoid accidental negative/overflow behavior in XOR/popcount. This should substantially improve score toward your target while preserving the exact overall logic and keeping runtime within limits.'
- What this solution (achieved 0.00169) has done: 'Your current score is far below target, so we should improve it with the smallest change that makes your existing dHash+NN logic actually “see” the hidden test images and reduces the chance we’re silently missing files. I (1) fix test image discovery to prefer the true leaf directory that contains the `.jpg` files (avoiding nested `.../test_images/test_images/...` mismatches), (2) build `test_path_by_image` in a collision-safe way (some competitions can have duplicate basenames across folders), and (3) use the sample submission’s image list to validate/resolve paths deterministically before hashing/predicting. This preserves your exact core approach (priors + dHash nearest neighbors + vote) but should materially increase MAP@5 by ensuring we hash the correct test images instead of falling back due to path lookup issues.'

# 9. Code solution

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
import multiprocessing as mp


def dhash_image_nocache(path, hash_size=8):
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
            a = np.asarray(im, dtype=np.uint8)
        diff = a[:, :-1] > a[:, 1:]
        bits = np.packbits(diff.reshape(-1).astype(np.uint8), bitorder="little")
        return int.from_bytes(bits.tobytes(), byteorder="little", signed=False) & (
            (1 << 64) - 1
        )
    except Exception:
        return None


def _dhash_worker(args):
    path, payload = args
    return (payload, dhash_image_nocache(path))


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

t_list = time.time()
cands = glob.glob(os.path.join(test_images_dir, "**", "*.jpg"), recursive=True)
if len(cands) == 0:
    raise FileNotFoundError(f"No .jpg found under {test_images_dir}")

best_root = test_images_dir
best_count = len(cands)
nested = os.path.join(test_images_dir, "test_images")
if os.path.isdir(nested):
    nested_jpgs = glob.glob(os.path.join(nested, "**", "*.jpg"), recursive=True)
    if len(nested_jpgs) > best_count:
        best_root, best_count, cands = nested, len(nested_jpgs), nested_jpgs

all_test_files = cands
print(
    "Found test jpg files:",
    len(all_test_files),
    f"(listed in {time.time()-t_list:.1f}s)",
)
print("Using best test jpg root:", best_root)

test_paths_by_image = {}
for p in all_test_files:
    bn = os.path.basename(p)
    if bn not in test_paths_by_image:
        test_paths_by_image[bn] = []
    test_paths_by_image[bn].append(p)


def resolve_test_path(img_name: str):
    ps = test_paths_by_image.get(img_name)
    if not ps:
        return None
    if len(ps) == 1:
        return ps[0]
    ps_sorted = sorted(ps, key=lambda x: (len(x), x))
    return ps_sorted[0]


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


images_arr = train["image"].to_numpy(dtype=str)
chains_arr = train["chain"].to_numpy(dtype=str)
hids_arr = train["hotel_id"].to_numpy(dtype=str)

train_path_by_image = {
    img: os.path.join(train_images_dir, ch, img)
    for img, ch in zip(images_arr, chains_arr)
}
print(
    "Train images with constructed path mapping:",
    len(train_path_by_image),
    "/",
    len(train),
)

train_folder_key_by_image = {img: ch for img, ch in zip(images_arr, chains_arr)}
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

for bn, ps in test_paths_by_image.items():
    p = resolve_test_path(bn)
    if p is None:
        continue

    ch = infer_chain_from_path(p)
    if ch is None:
        unknown_chain_paths += 1
    else:
        test_chain_by_image[bn] = ch

    fk = infer_parent_folder_key(p)
    if fk is None:
        unknown_folder_keys += 1
    else:
        test_folder_key_by_image[bn] = fk

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
modulo = 3

keep_mask = np.fromiter(
    (keep_train_image(img, modulo=modulo) for img in images_arr),
    dtype=np.bool_,
    count=len(images_arr),
)
kept_indices = np.flatnonzero(keep_mask)

train_hash_db = []
if kept_indices.size:
    cpu = os.cpu_count() or 2
    workers = min(8, cpu)
    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp

    cand = []
    for idx in kept_indices:
        img = images_arr[idx]
        hid = hids_arr[idx]
        p = train_path_by_image.get(img)
        if p is not None:
            cand.append((p, hid))

    hashed = 0
    if cand:
        chunksize = 32
        with ctx.Pool(processes=workers) as pool:
            for hid, h in pool.imap(
                _dhash_worker, ((p, hid) for p, hid in cand), chunksize=chunksize
            ):
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

test_path_by_image = {img: resolve_test_path(img) for img in sample_images}

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
    uniq_hids, inv_codes_db = np.unique(_train_hid_arr.astype(str), return_inverse=True)
else:
    uniq_hids = np.empty((0,), dtype=str)
    inv_codes_db = np.empty((0,), dtype=np.int32)

t_hash_test = time.time()
test_dhash_by_image = {img: None for img in sample_images}
if sample_images:
    cpu = os.cpu_count() or 2
    workers = min(8, cpu)
    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
    tasks = []
    for img in sample_images:
        p = test_path_by_image.get(img)
        if p is not None:
            tasks.append((p, img))
    if tasks:
        chunksize = 64
        with ctx.Pool(processes=workers) as pool:
            for img, h in pool.imap(_dhash_worker, tasks, chunksize=chunksize):
                test_dhash_by_image[img] = h

print(
    f"Precomputed test dHashes for {len(sample_images)} images in {time.time()-t_hash_test:.1f}s"
)


def nn_top5_from_codes(codes_k, uniq_hids_arr, topn=5):
    if codes_k.size == 0:
        return []
    u, c = np.unique(codes_k, return_counts=True)
    h = uniq_hids_arr[u]
    order = np.lexsort((h, -c))
    return h[order][:topn].tolist()


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
        qh = test_dhash_by_image.get(img)
        if qh is not None:
            d = hamming_vec_u64(qh, _train_hash_arr)

            k = NN_K if NN_K < d.size else d.size
            idx = np.argpartition(d, k - 1)[:k]
            idx = idx[np.argsort(d[idx], kind="mergesort")]

            codes_k = inv_codes_db[idx]
            nn_top = nn_top5_from_codes(codes_k, uniq_hids, topn=NN_TOP5)

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
