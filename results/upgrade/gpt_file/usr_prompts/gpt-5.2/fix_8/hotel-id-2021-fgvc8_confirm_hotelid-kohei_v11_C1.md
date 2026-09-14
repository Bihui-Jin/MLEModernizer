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
    for root, _, files in os.walk(test_dir):
        base = os.path.basename(root)
        chain_candidate = base
        for fn in files:
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            if fn in img2chain:
                continue
            try:
                ch = int(chain_candidate)
            except ValueError:
                ch = None
            if ch is not None:
                img2chain[fn] = ch
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
            arr = np.asarray(im, dtype=np.float32) / 255.0
            return arr.mean(axis=(0, 1))  # (3,)
    except Exception:
        return None


def resolve_test_path(img_name: str) -> str:
    p = os.path.join(test_images_dir, img_name)
    if os.path.exists(p):
        return p
    p2 = os.path.join(test_images_dir, "test_images", img_name)
    if os.path.exists(p2):
        return p2
    return p  # keep a sensible default


def resolve_train_path(chain: int, img_name: str) -> str:
    return os.path.join(train_images_dir, str(chain), img_name)


PER_HOTEL = 3
MAX_INDEX_IMAGES = 20000  # ensures runtime stays reasonable (<600s) while improving over constant predictions.

train_small = train[["image", "chain", "hotel_id"]].copy()
train_small["chain"] = train_small["chain"].astype(int)

selected_rows = []
for hid, g in train_small.groupby("hotel_id", sort=False):
    if len(g) <= PER_HOTEL:
        selected_rows.append(g)
    else:
        gg = g.copy()
        gg["_k"] = gg["image"].apply(lambda x: hash((str(x), int(SEED))) & 0xFFFFFFFF)
        gg = gg.sort_values("_k").head(PER_HOTEL).drop(columns=["_k"])
        selected_rows.append(gg)
selected = pd.concat(selected_rows, ignore_index=True)

if len(selected) > MAX_INDEX_IMAGES:
    selected["_k"] = selected["image"].apply(
        lambda x: hash((str(x), 12345, int(SEED))) & 0xFFFFFFFF
    )
    selected = selected.sort_values("_k").head(MAX_INDEX_IMAGES).drop(columns=["_k"])

print("Indexing train images (selected rows):", len(selected))

index_feats = []
index_hotels = []
fail_count = 0

for r in selected.itertuples(index=False):
    p = resolve_train_path(int(r.chain), str(r.image))
    feat = safe_avg_rgb(p)
    if feat is None:
        fail_count += 1
        continue
    index_feats.append(feat)
    index_hotels.append(str(r.hotel_id))

index_feats = np.asarray(index_feats, dtype=np.float32)
index_hotels = np.asarray(index_hotels, dtype=object)
print("Built visual index size:", len(index_hotels), "failed:", fail_count)

pop_rank = train["hotel_id"].value_counts()
pop_rank = pop_rank.reindex(pop_rank.index.astype(str))
pop = pop_rank.to_dict()


def visual_vote_top5(test_feat, k=50):
    if test_feat is None or index_feats.shape[0] == 0:
        return None
    d = index_feats - test_feat[None, :]
    dist = (d * d).sum(axis=1)
    kk = min(k, dist.shape[0])
    nn_idx = np.argpartition(dist, kk - 1)[:kk]
    nn_hotels = index_hotels[nn_idx]

    counts = {}
    best_dist = {}
    for h, di in zip(nn_hotels, dist[nn_idx]):
        counts[h] = counts.get(h, 0) + 1
        if h not in best_dist or di < best_dist[h]:
            best_dist[h] = float(di)

    cand = list(counts.keys())
    cand.sort(key=lambda h: (-counts[h], -(pop.get(h, 0)), best_dist.get(h, 1e9)))
    top = []
    seen = set()
    for h in cand:
        hs = str(h)
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
    if img_name in test_feat_cache:
        return test_feat_cache[img_name]
    p = resolve_test_path(img_name)
    feat = safe_avg_rgb(p)
    test_feat_cache[img_name] = feat
    return feat




## === cell 3
preds = []
missing_chain = 0
used_month_fallback = 0
used_global_fallback = 0
used_visual = 0

for img in sample["image"].astype(str).tolist():
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



## === cell 4
sub = pd.read_csv("submission.csv")
print("submission.csv shape:", sub.shape)
print("columns:", sub.columns.tolist())
print("example row:", sub.iloc[0].to_dict())
print("unique prediction strings:", sub["hotel_id"].nunique())
lens = sub["hotel_id"].astype(str).str.split().map(len)
print("min/max #ids per row:", int(lens.min()), int(lens.max()))
assert lens.min() == 5 and lens.max() == 5
