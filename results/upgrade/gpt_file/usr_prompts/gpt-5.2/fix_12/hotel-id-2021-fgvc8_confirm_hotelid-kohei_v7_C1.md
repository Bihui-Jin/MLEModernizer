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

0.7201545002946846

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00232) has done: 'Your notebook currently doesn’t yield a score because it relies on private `/kaggle/input/pekolib*` wheels and a `peko.subs.hotelid.v7` module that are not present in the provided environment, so it never produces a valid `submission.csv` from your actual competition data. The smallest safe fix is to remove that dependency and instead generate a valid submission directly from the provided `sample_submission.csv`, which run end-to-end and create a properly formatted `submission.csv`. This not reach your target MAP@5 (since it’s a baseline), but it unblock scoring and let you iterate further with real model code later while preserving evaluation semantics and submission format.'
- What this solution (achieved 0.00209) has done: 'Your current score is extremely far below the target, and the main issue is that you’re submitting the untouched `sample_submission.csv` (all blank/constant predictions), which yields near-random MAP@5. To move toward the target with minimal core-logic changes, I keep the “no external model” approach but replace the dummy predictions with a simple, legitimate frequency baseline: predict the 5 most common `hotel_id`s from `train.csv` for every test image. This preserves evaluation semantics and produces a valid `submission.csv`, and it should substantially increase MAP@5 versus the current file. I also keep your robust path search and add a similarly robust search for `train.csv`.'
- What this solution (achieved 0.0014) has done: 'Your current baseline predicts the same top-5 hotels for every test image, which severely limits MAP@5. To move toward the target while keeping the same “no model, frequency-based” core logic, I make the predictions chain-aware: infer each test image’s chain from its folder name (as the dataset is organized by chain), then predict the top hotels within that chain (falling back to global top hotels if needed). This is still a simple frequency baseline (no architecture/training changes) but uses legitimate metadata structure to produce more relevant candidate lists. I also ensure we always output exactly 5 unique hotel IDs per row and keep robust path discovery intact.'
- What this solution (achieved 0.0014) has done: 'Your score is far below the target, so we should increase MAP@5 with the smallest change that keeps your “frequency baseline + chain awareness” core logic intact. The biggest weakness is the very expensive and unreliable chain inference loop that searches up to 100 folders per image; instead, we build a one-time lookup from `test_images/**/image.jpg -> chain_id` by scanning the filesystem once, then use O(1) lookup per test row. We also improve the chain-aware ranking slightly (still purely frequency-based) by using a “chain weight + global weight” combined score to choose the top-5, which tends to be more robust when chain information is noisy or sparse. The submission format checks remain, and we still write a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.0014) has done: 'Your current score is far below the target, so we should increase MAP@5 with the smallest change that keeps the same “chain-aware frequency baseline” core logic. The main weakness is that using only `chain` leaves many hotels within a chain and gives weak per-image ranking; we can legitimately add a second metadata signal from the folder structure: infer `hotel_id` directly when test images are stored under `test_images/<hotel_id>/...` (and similarly allow nested layouts). We build a one-time `image -> hotel_id` map by scanning `test_images` recursively (fast enough for ~10k files), and when available we place that inferred hotel first, then fill remaining slots with your existing chain/global frequency ranker. This preserves the overall approach (no model, still frequency-based) while improving relevance for many rows and should move the score substantially toward your target.'
- What this solution (achieved 0.00171) has done: 'Your current approach is fundamentally limited because the test folder structure does not reliably expose `chain` or `hotel_id`, so the “chain-aware + inferred hotel from folder name” signals are mostly missing and you fall back to near-constant global predictions (hence ~0.0014). To move the score substantially toward your target while keeping the same lightweight “no model, metadata/frequency-based candidate retrieval” core logic, I add one more legitimate signal that’s available at inference: text similarity between the test image filename and training filenames (same-camera/series often shares prefixes). Concretely, we build a mapping from a filename prefix (e.g., first 6–10 hex chars) to the most frequent `hotel_id` in train for that prefix, then for each test image we place that prefix-based hotel first when available and fill the remaining slots with your existing chain/global ranker. This keeps the submission semantics identical (top-5 IDs per image), is fast (single pass over train.csv), and should increase MAP@5 relative to your current mostly-global fallback.'
- What this solution (achieved 0.00171) has done: 'Your current score is far below the target, so we should increase MAP@5 with the smallest change that preserves your existing “metadata/frequency-based candidate retrieval” logic. The biggest issue is that `img2chain`/`img2hotel` built from `test_images` folder names won’t generalize to the hidden test set (which is flat), so the code mostly falls back to near-constant global predictions. To add a legitimate, test-available signal without changing the overall approach, we derive a per-test-image “chain prior” using the *most frequent chain among the nearest training filename-prefix matches* (computed only from `train.csv`), then feed that chain into your existing chain-aware scorer. We keep your prefix-to-hotel hint as the top-1 candidate, but also make chain inference prefix-based (no filesystem dependency), which should move the score upward toward your target while staying lightweight and deterministic.'
- What this solution (achieved 0.0014) has done: 'Your current score is far below the target, so we should improve MAP@5 while keeping the same “metadata/frequency-based candidate retrieval” core logic. The smallest high-impact fix is to stop relying on filename-prefix heuristics (which are essentially random because filenames are hex and not correlated to hotel) and instead use the one strong signal available in your metadata: `chain`. We predict using a chain-specific top-5 list for each test image by inferring a plausible chain from the test image’s parent folder when present, and otherwise fall back to the global top-5; this removes noisy prefix-based candidates that were hurting rank-1/2 positions. We also ensure determinism, enforce exactly 5 unique IDs, and keep the same input paths and submission schema.'

# 9. Code solution

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


def build_test_indexes(root_dir: str):
    img2path = {}
    img2chain = {}

    def _walk(dirpath, chain_id_hint=0):
        try:
            with os.scandir(dirpath) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        name = entry.name
                        cid = chain_id_hint
                        if name.isdigit():
                            cid = int(name)
                        _walk(entry.path, cid)
                    elif entry.is_file(follow_symlinks=False):
                        fn = entry.name
                        if fn.lower().endswith(".jpg"):
                            if fn not in img2path:
                                img2path[fn] = entry.path
                            if fn not in img2chain:
                                img2chain[fn] = chain_id_hint
        except FileNotFoundError:
            return

    _walk(root_dir, 0)
    return img2path, img2chain


def dhash64_from_path(path: str):
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
            a = np.asarray(im, dtype=np.uint8)  # shape (8, 9)
            bits = (a[:, :-1] > a[:, 1:]).reshape(-1)
            h = int(
                np.packbits(bits.astype(np.uint8), bitorder="little").view(np.uint64)[0]
            )
            return h
    except Exception:
        return None


PC_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

test_img_paths, img2chain_fs = build_test_indexes(test_img_root)

print(f"Using sample_submission.csv from: {sample_path}")
print(f"Using train.csv from: {train_path}")
print(f"Using train_images root from: {train_img_root}")
print(f"Using test_images root from: {test_img_root}")
print(f"Indexed test image paths: {len(test_img_paths)}")
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

for img, chain, hid in zip(imgs_arr, chains_arr, hids_arr):
    p = os.path.join(train_img_root, str(int(chain)), str(img))
    if not os.path.exists(p):
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

preds = []
used_hash = 0
used_fs_chain = 0
missing_chain = 0
missing_test_file = 0
hash_fail = 0

global_counts_f = global_counts.astype(float).to_dict()

img2chain_get = img2chain_fs.get
test_paths_get = test_img_paths.get
predict_chain = predict_topk_for_chain
pc_lut = PC_LUT
gc_f = global_counts_f
gc_top = global_top
K_local = K
NN_local = NN
train_hids_local = train_hids

for img in sample["image"].to_numpy(dtype=object):
    c = int(img2chain_get(img, 0))
    if c != 0:
        used_fs_chain += 1
    else:
        missing_chain += 1

    base = predict_chain(c)

    out = []
    tp = test_paths_get(img)
    if tp is None:
        missing_test_file += 1
    else:
        th = dhash64_from_path(tp)
        if th is None:
            hash_fail += 1
        elif train_hashes.size > 0:
            th_u = np.uint64(th)
            th_bytes = th_u.view(np.uint8)  # shape (8,)

            xb = np.bitwise_xor(train_bytes, th_bytes)
            dists = pc_lut[xb].sum(axis=1).astype(np.int16, copy=False)

            if dists.size > NN_local:
                idx = np.argpartition(dists, NN_local - 1)[:NN_local]
            else:
                idx = np.arange(dists.size)

            top_pairs = list(zip(dists[idx].tolist(), train_hids_local[idx].tolist()))
            top_pairs.sort(key=lambda t: (t[0], t[1]))

            cnt = {}
            bestd = {}
            for d, hid in top_pairs:
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
                if hid not in out:
                    out.append(hid)
                if len(out) == K_local:
                    break
            if len(out) > 0:
                used_hash += 1

    for hid in base:
        if hid not in out:
            out.append(hid)
        if len(out) == K_local:
            break

    if len(out) < K_local:
        for hid in gc_top:
            if hid not in out:
                out.append(hid)
            if len(out) == K_local:
                break
    if len(out) < K_local:
        out = (out * K_local)[:K_local]

    preds.append(" ".join(out[:K_local]))

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


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/169956309.py in <cell line: 0>()
    300         elif train_hashes.size > 0:
    301             th_u = np.uint64(th)
--> 302             th_bytes = th_u.view(np.uint8)  # shape (8,)
    303 
    304             # XOR in byte space then LUT popcount; identical to prior xor->view->lut sum

ValueError: Changing the dtype of a 0d array is only supported if the itemsize is unchanged

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

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1239055264.py in <cell line: 0>()
      1 import pandas as pd
      2 
----> 3 sub = pd.read_csv("submission.csv")
      4 assert list(sub.columns) == [
      5     "image",

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'
