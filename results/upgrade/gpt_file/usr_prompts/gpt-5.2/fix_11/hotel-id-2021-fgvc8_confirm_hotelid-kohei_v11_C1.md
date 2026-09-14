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

0.7607266144649304

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your current notebook doesn’t yield a score because it never reliably produces a valid `submission.csv`: it depends on external wheels (`pekolib`) that aren’t present in the provided environment, so the pipeline fails before writing the file. I replace those external-package calls with a minimal, self-contained baseline that (1) reads the provided `sample_submission.csv`, (2) builds a simple, legitimate prediction list from the most frequent `hotel_id` values in `train.csv`, and (3) writes a valid `submission.csv` with the exact required columns and row alignment. This not be a top solution, but it run end-to-end within the time limit and produce a valid submission so you can obtain a real Kaggle score and iterate toward the target afterward.'
- What this solution (achieved 0.00209) has done: 'Your current score (0.00209) is far below the target (0.7607), so we need a legitimate improvement while keeping changes minimal and preserving the “simple baseline” core logic. The main issue is that predicting the same global top-5 hotels for every test image is extremely weak for MAP@5; a small, safe step up is to use available metadata (`chain`) to make per-image predictions more relevant. We can compute per-chain top-5 hotel IDs from `train.csv`, then for each test image infer its chain from the folder structure in `test_images/<chain>/...` and output the corresponding chain-specific top-5; if chain can’t be inferred, we fall back to the global top-5. This keeps the approach simple (frequency-based), runs fast, and should move the score substantially toward the target without changing the evaluation semantics.'
- What this solution (achieved 0.00209) has done: 'Your current score is far below the target, so we should improve the weakest link while keeping your frequency-based “chain-aware top-5” core logic unchanged. The main issue is that `infer_chain_from_fs()` scans *all* chain folders for *every* test image, which is slow and can fail to find images reliably within Kaggle time limits/IO constraints, causing many fallbacks to the global top-5 (very low MAP@5). I replace that per-image directory scan with a one-time precomputed `image -> chain` lookup by scanning the test directory once (same semantics, just correct/efficient). I also make the test_images path robust to the common nested `test_images/test_images` layout so chain inference doesn’t silently miss everything.'
- What this solution (achieved 0.00209) has done: 'Your current approach is already “chain-aware top-5,” but it’s still scoring extremely low because the test images are not organized into chain subfolders, so your chain inference fails for almost all rows and you end up predicting the same global top-5 for nearly every test image. To move the score meaningfully toward the target while keeping the same frequency-based core logic, I instead infer a per-image “pseudo-chain” by matching the test image filename against `train.csv` (many public/private test images overlap filenames with train in this competition), and then use that chain’s top-5; if not found, we fall back to the global top-5 exactly as before. I keep the one-pass mapping idea (but from `train.csv` rather than filesystem) to make inference reliable and fast within the time limit. The output schema/row order remains identical and it still write a valid `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'Your current score is far below the target, so we need a legitimate (but still minimal) improvement over the frequency-only baseline. The biggest issue is that `infer_chain()` almost always fails because test images don’t share filenames with train, so you fall back to the same global top-5 for nearly every row (terrible for MAP@5). Keeping the same “simple frequency-based retrieval” core logic, I instead build per-chain top-5 lists and infer a test image’s chain from the test image filesystem by doing a single one-time scan to map `image -> chain` (fast and reliable). This should substantially increase MAP@5 while preserving your overall approach and still writing a valid `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'Your current score is far below the target, so we need a legitimate improvement, but your current “chain-aware top-5” can’t work because the test set does not provide chain folders, so `img2chain` is almost always empty and you fall back to the same global top-5 for nearly all images. Keeping the same simple frequency-based core logic (no model, no image features), the minimal meaningful step is to condition predictions on *time* instead of chain: compute per-month top-5 hotel IDs from `train.csv` and, for each test image, assign it a month using the average month-of-year observed for that hotel’s training photos (a light, non-leaky prior); if the month can’t be inferred, fall back to the global top-5. This uses only provided metadata, keeps the “top-5 by frequency” semantics, and should move MAP@5 upward versus a constant prediction while still running fast and writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

BASE = "/kaggle/input/hotel-id-2021-fgvc8"
train_csv_path = os.path.join(BASE, "train.csv")
sample_path = os.path.join(BASE, "sample_submission.csv")
test_images_dir = os.path.join(BASE, "test_images")
train_images_dir = os.path.join(BASE, "train_images")

assert os.path.exists(train_csv_path), f"Missing: {train_csv_path}"
assert os.path.exists(sample_path), f"Missing: {sample_path}"
assert os.path.exists(test_images_dir), f"Missing: {test_images_dir}"
assert os.path.exists(train_images_dir), f"Missing: {train_images_dir}"

print("train.csv:", train_csv_path)
print("sample_submission.csv:", sample_path)
print("test_images_dir:", test_images_dir)
print("train_images_dir:", train_images_dir)

nested = os.path.join(test_images_dir, "test_images")
if os.path.isdir(nested):
    test_images_dir = nested
    print("Adjusted test_images_dir to nested:", test_images_dir)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)




## === cell 1
train = pd.read_csv(train_csv_path)
sample = pd.read_csv(sample_path)

global_top5 = train["hotel_id"].value_counts().head(5).index.astype(str).tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 + ["0"] * 5)[:5]
global_pred_str = " ".join(global_top5)

ts = pd.to_datetime(train["timestamp"], errors="coerce", utc=True)
train = train.assign(_month=ts.dt.month)

month_top5 = (
    train.dropna(subset=["_month"])
    .groupby("_month")["hotel_id"]
    .apply(lambda s: s.value_counts().head(5).index.astype(str).tolist())
    .to_dict()
)

for m in range(1, 13):
    ids = month_top5.get(m, [])
    out = []
    seen = set()
    for x in [str(v) for v in ids]:
        if x not in seen:
            out.append(x)
            seen.add(x)
        if len(out) == 5:
            break
    if len(out) < 5:
        for x in global_top5:
            if x not in seen:
                out.append(x)
                seen.add(x)
            if len(out) == 5:
                break
    if len(out) < 5:
        out = (out + ["0"] * 5)[:5]
    month_top5[m] = out

hotel_mean_month = train.dropna(subset=["_month"]).groupby("hotel_id")["_month"].mean()
hotel_mean_month = hotel_mean_month.round().clip(1, 12).astype("int16").to_dict()

print("Global top5:", global_top5)
print("Example month_top5[1]:", month_top5[1])
print("hotel_mean_month entries:", len(hotel_mean_month))




## === cell 2
def build_test_image_to_chain_map_from_fs(test_dir: str):
    img2chain = {}
    try:
        entries = os.listdir(test_dir)
    except FileNotFoundError:
        return img2chain

    has_subdirs = False
    for e in entries:
        p = os.path.join(test_dir, e)
        if os.path.isdir(p):
            has_subdirs = True
            break
    if not has_subdirs:
        return img2chain

    for sub in entries:
        subdir = os.path.join(test_dir, sub)
        if not os.path.isdir(subdir):
            continue
        try:
            ch = int(sub)
        except ValueError:
            continue
        for fn in os.listdir(subdir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                img2chain.setdefault(fn, ch)
    return img2chain


img2chain = build_test_image_to_chain_map_from_fs(test_images_dir)
print("Precomputed image->chain mappings from test_images filesystem:", len(img2chain))


def infer_chain(image_name: str):
    return img2chain.get(image_name)


train_image_to_hotel = train.set_index("image")["hotel_id"].astype(str).to_dict()


def infer_month_from_train_overlap(image_name: str):
    hid = train_image_to_hotel.get(image_name)
    if hid is None:
        return None
    m = hotel_mean_month.get(hid)
    if m is None:
        try:
            m = hotel_mean_month.get(int(hid))
        except Exception:
            m = None
    return int(m) if m is not None else None


def safe_avg_rgb(image_path: str, size=(32, 32)):
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im = im.resize(size)
            arr = np.asarray(im, dtype=np.float32)
        arr *= 1.0 / 255.0
        return arr.mean(axis=(0, 1))  # (3,)
    except Exception:
        return None


_HAS_NESTED_TEST = os.path.isdir(os.path.join(test_images_dir, "test_images"))
_test_base_dir = (
    os.path.join(test_images_dir, "test_images")
    if _HAS_NESTED_TEST
    else test_images_dir
)


def resolve_test_path(img_name: str) -> str:
    return os.path.join(_test_base_dir, img_name)


def resolve_train_path(chain: int, img_name: str) -> str:
    return os.path.join(train_images_dir, str(chain), img_name)


PER_HOTEL = 3
MAX_INDEX_IMAGES = 20000  # keep constant; core logic unchanged.

train_small = train[["image", "chain", "hotel_id"]].copy()
train_small["chain"] = train_small["chain"].astype(int)


def _hash_u32_series(img_series: pd.Series, salt_tuple):
    salt = "|".join(map(str, salt_tuple))
    s = (salt + "|" + img_series.astype(str)).astype("string")
    h64 = pd.util.hash_pandas_object(s, index=False).astype("uint64")
    return (h64 & np.uint64(0xFFFFFFFF)).astype("uint32")


k1 = _hash_u32_series(train_small["image"], (int(SEED),))
train_small = train_small.assign(_k=k1)
selected = (
    train_small.sort_values(["hotel_id", "_k"], kind="mergesort")
    .groupby("hotel_id", sort=False, as_index=False, group_keys=False)
    .head(PER_HOTEL)
    .drop(columns=["_k"])
    .reset_index(drop=True)
)

if len(selected) > MAX_INDEX_IMAGES:
    k2 = _hash_u32_series(selected["image"], (12345, int(SEED)))
    selected = (
        selected.assign(_k=k2)
        .sort_values("_k", kind="mergesort")
        .head(MAX_INDEX_IMAGES)
        .drop(columns=["_k"])
        .reset_index(drop=True)
    )

print("Indexing train images (selected rows):", len(selected))

from concurrent.futures import ThreadPoolExecutor


def _feat_from_selected_row(tup):
    image, chain, hotel_id = tup
    p = resolve_train_path(int(chain), str(image))
    feat = safe_avg_rgb(p)
    return feat, str(hotel_id)


selected_tuples = list(selected.itertuples(index=False, name=None))
max_workers = min(32, (os.cpu_count() or 4) * 2)

index_feats_list = []
index_hotels_list = []
fail_count = 0

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for feat, hid in ex.map(_feat_from_selected_row, selected_tuples, chunksize=128):
        if feat is None:
            fail_count += 1
        else:
            index_feats_list.append(feat)
            index_hotels_list.append(hid)

index_feats = np.asarray(index_feats_list, dtype=np.float32)
index_hotels = np.asarray(index_hotels_list, dtype=object)
print("Built visual index size:", len(index_hotels), "failed:", fail_count)

pop_rank = train["hotel_id"].value_counts()
pop_rank = pop_rank.reindex(pop_rank.index.astype(str))
pop = pop_rank.to_dict()

_unique_hotels, _inv = np.unique(index_hotels, return_inverse=True)
_unique_pop = np.asarray([pop.get(h, 0) for h in _unique_hotels], dtype=np.int64)

_index_norm = (index_feats * index_feats).sum(axis=1).astype(np.float32)


def visual_vote_top5(test_feat, k=50):
    if test_feat is None or index_feats.shape[0] == 0:
        return None

    test_norm = float((test_feat * test_feat).sum())
    dist = _index_norm + test_norm - 2.0 * (index_feats @ test_feat.astype(np.float32))

    kk = min(k, dist.shape[0])
    nn_idx = np.argpartition(dist, kk - 1)[:kk]

    nn_dist = dist[nn_idx]
    nn_h_inv = _inv[nn_idx]  # integer ids into _unique_hotels

    order = np.argsort(nn_h_inv, kind="mergesort")
    h_sorted = nn_h_inv[order]
    d_sorted = nn_dist[order]

    grp_start = np.r_[0, np.flatnonzero(h_sorted[1:] != h_sorted[:-1]) + 1]
    grp_ids = h_sorted[grp_start]
    counts = np.diff(np.r_[grp_start, h_sorted.size])
    best_dist = np.minimum.reduceat(d_sorted, grp_start)

    pop_u = _unique_pop[grp_ids]
    sort_idx = np.lexsort((best_dist, -pop_u, -counts))
    ranked_u = grp_ids[sort_idx]

    top = []
    seen = set()
    for uid in ranked_u:
        hs = str(_unique_hotels[uid])
        if hs not in seen:
            top.append(hs)
            seen.add(hs)
        if len(top) == 5:
            break
    if len(top) < 5:
        for h in global_top5:
            if h not in seen:
                top.append(h)
                seen.add(h)
            if len(top) == 5:
                break
    if len(top) < 5:
        top = (top + ["0"] * 5)[:5]
    return top


test_feat_cache = {}


def get_test_feat(img_name: str):
    feat = test_feat_cache.get(img_name)
    if feat is not None or img_name in test_feat_cache:
        return feat
    p = resolve_test_path(img_name)
    feat = safe_avg_rgb(p)
    test_feat_cache[img_name] = feat
    return feat


test_images = sample["image"].astype(str).values
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    feats = list(ex.map(get_test_feat, test_images, chunksize=128))




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/224008088.py in <cell line: 0>()
    157 
    158 _unique_hotels, _inv = np.unique(index_hotels, return_inverse=True)
--> 159 _unique_pop = np.asarray([pop.get(h, 0) for h in _unique_hotels], dtype=np.int64)
    160 
    161 # --- Speed: precompute norms for distance expansion to avoid per-query (N,3) temp arrays.

ValueError: cannot convert float NaN to integer

## === cell 3
preds = []
missing_chain = 0
used_month_fallback = 0
used_global_fallback = 0
used_visual = 0

for img in sample["image"].values.astype(str, copy=False):
    ch = infer_chain(img)
    if ch is not None:
        preds.append(global_pred_str)
        continue

    missing_chain += 1

    top5 = visual_vote_top5(get_test_feat(img), k=50)
    if top5 is not None:
        used_visual += 1
        preds.append(" ".join(top5))
        continue

    m = infer_month_from_train_overlap(img)
    if m is not None:
        used_month_fallback += 1
        preds.append(" ".join(month_top5.get(m, global_top5)))
    else:
        used_global_fallback += 1
        preds.append(global_pred_str)

submission = sample.copy()
submission["hotel_id"] = preds

print("Rows with inferred chain missing:", missing_chain, "/", len(sample))
print("Used visual kNN:", used_visual)
print("Used month-conditioned fallback:", used_month_fallback)
print("Used global fallback:", used_global_fallback)

assert submission.shape[0] == sample.shape[0]
assert list(submission.columns) == ["image", "hotel_id"]

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1106866979.py in <cell line: 0>()
     13     missing_chain += 1
     14 
---> 15     top5 = visual_vote_top5(get_test_feat(img), k=50)
     16     if top5 is not None:
     17         used_visual += 1

NameError: name 'visual_vote_top5' is not defined

## === cell 4
sub = pd.read_csv("submission.csv")
print("submission.csv shape:", sub.shape)
print("columns:", sub.columns.tolist())
print("example row:", sub.iloc[0].to_dict())
print("unique prediction strings:", sub["hotel_id"].nunique())
lens = sub["hotel_id"].astype(str).str.split().map(len)
print("min/max #ids per row:", int(lens.min()), int(lens.max()))
assert lens.min() == 5 and lens.max() == 5

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1119450327.py in <cell line: 0>()
----> 1 sub = pd.read_csv("submission.csv")
      2 print("submission.csv shape:", sub.shape)
      3 print("columns:", sub.columns.tolist())
      4 print("example row:", sub.iloc[0].to_dict())
      5 print("unique prediction strings:", sub["hotel_id"].nunique())

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
