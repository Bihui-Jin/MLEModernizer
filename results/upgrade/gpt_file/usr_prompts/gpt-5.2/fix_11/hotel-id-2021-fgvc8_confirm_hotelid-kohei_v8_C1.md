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

BASE_CANDIDATES = [
    "/kaggle/input/hotel-id-2021-fgvc8",
    "/kaggle/data/hotel-id-2021-fgvc8",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename: str) -> str:
    for base in BASE_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
        path2 = os.path.join(base, "hotel-id-2021-fgvcvc8", filename)
        if os.path.exists(path2):
            return path2
        path3 = os.path.join(base, "hotel-id-2021-fgvc8", filename)
        if os.path.exists(path3):
            return path3
    raise FileNotFoundError(
        f"Could not find {filename} under candidates: {BASE_CANDIDATES}"
    )


def find_dir(dirname: str) -> str:
    for base in BASE_CANDIDATES:
        p1 = os.path.join(base, dirname)
        if os.path.isdir(p1):
            return p1
        p2 = os.path.join(base, "hotel-id-2021-fgvc8", dirname)
        if os.path.isdir(p2):
            return p2
    raise FileNotFoundError(
        f"Could not find directory {dirname} under candidates: {BASE_CANDIDATES}"
    )


def top5_strict_from_counts(series: pd.Series) -> list:
    top = series.value_counts().head(5).index.astype(str).tolist()
    if len(top) == 0:
        return []
    while len(top) < 5:
        top.append(top[-1])
    return top[:5]


def merge_top5(primary: list, fallback: list) -> list:
    out = []
    for x in primary:
        xs = str(x)
        if xs not in out:
            out.append(xs)
        if len(out) == 5:
            return out
    for x in fallback:
        xs = str(x)
        if xs not in out:
            out.append(xs)
        if len(out) == 5:
            return out
    while len(out) < 5 and len(out) > 0:
        out.append(out[-1])
    return out[:5]


def dhash64_from_image(im: Image.Image, hash_size: int = 8):
    im = im.resize((hash_size + 1, hash_size), Image.BILINEAR)
    pixels = list(im.getdata())
    rows = [
        pixels[i * (hash_size + 1) : (i + 1) * (hash_size + 1)]
        for i in range(hash_size)
    ]
    h = 0
    bit = 1
    for r in range(hash_size):
        row = rows[r]
        for c in range(hash_size):
            if row[c] > row[c + 1]:
                h |= bit
            bit <<= 1
    return h


def dhash128_from_path(img_path: str, hash_size: int = 8):
    try:
        with Image.open(img_path) as im0:
            im_rgb = im0.convert("RGB")
            im_gray = im_rgb.convert("L")
            h1 = dhash64_from_image(im_gray, hash_size=hash_size)

            im_hsv = im_rgb.convert("HSV")
            v = im_hsv.split()[2]
            h2 = dhash64_from_image(v, hash_size=hash_size)

            return (h1, h2)
    except Exception:
        return None


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def hamming128(a, b) -> int:
    return hamming64(a[0], b[0]) + hamming64(a[1], b[1])


train_csv = find_file("train.csv")
sample_sub_csv = find_file("sample_submission.csv")

print("Using train.csv:", train_csv)
print("Using sample_submission.csv:", sample_sub_csv)

train = pd.read_csv(train_csv)
sub = pd.read_csv(sample_sub_csv)

train["hotel_id"] = train["hotel_id"].astype(str)
train["chain"] = train["chain"].astype(int)

global_top5 = top5_strict_from_counts(train["hotel_id"])
if len(global_top5) == 0:
    raise RuntimeError("No hotel_id found in training data.")

chain_top5 = {}
vc = train.groupby("chain")["hotel_id"].value_counts()
for chain_id in vc.index.get_level_values(0).unique().tolist():
    top = vc.loc[chain_id].head(5).index.astype(str).tolist()
    if len(top) == 0:
        continue
    while len(top) < 5:
        top.append(top[-1])
    chain_top5[int(chain_id)] = top[:5]

chain_hotelid_to_top5_other = {}
chain_hotel_counts = (
    train.groupby(["chain", "hotel_id"]).size().rename("cnt").reset_index()
)
for ch, g in chain_hotel_counts.groupby("chain", sort=False):
    ordered = g.sort_values("cnt", ascending=False)["hotel_id"].astype(str).tolist()
    for hid in ordered:
        others = [x for x in ordered if x != hid][:5]
        if len(others) > 0:
            while len(others) < 5:
                others.append(others[-1])
            chain_hotelid_to_top5_other[(int(ch), str(hid))] = others[:5]

test_images_dir = find_dir("test_images")
train_images_dir = find_dir("train_images")
print("Using test_images dir:", test_images_dir)
print("Using train_images dir:", train_images_dir)

sub_images = sub["image"].astype(str).tolist()
sub_image_set = set(sub_images)
sub_basename_set = set(os.path.basename(x) for x in sub_images)


def infer_chain_from_path(p: str):
    d = os.path.dirname(p)
    while True:
        base = os.path.basename(d)
        if base.isdigit():
            return int(base)
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


img2chain_by_exact = {}
img2chain_by_basename = {}

hit_exact = 0
hit_base = 0
walk_count = 0

test_basename_to_fullpath = {}

for root, _, files in os.walk(test_images_dir):
    for fn in files:
        if not fn.lower().endswith(".jpg"):
            continue
        walk_count += 1

        if fn in sub_basename_set and fn not in test_basename_to_fullpath:
            test_basename_to_fullpath[fn] = os.path.join(root, fn)

        if fn not in sub_basename_set:
            continue

        full_path = os.path.join(root, fn)
        ch = infer_chain_from_path(full_path)
        if ch is None:
            continue

        if fn not in img2chain_by_basename:
            img2chain_by_basename[fn] = ch
            hit_base += 1

        rel = os.path.relpath(full_path, test_images_dir).replace("\\", "/")
        candidates = [
            rel,
            f"test_images/{rel}",
        ]
        for key in candidates:
            if key in sub_image_set and key not in img2chain_by_exact:
                img2chain_by_exact[key] = ch
                hit_exact += 1

print("os.walk scanned jpg files:", walk_count)
print(
    "Inferred chain for",
    hit_base,
    "unique test basenames out of",
    len(sub_basename_set),
)
print(
    "Inferred chain for",
    hit_exact,
    "exact submission image strings out of",
    len(sub_image_set),
)
print(
    "Built basename->fullpath for",
    len(test_basename_to_fullpath),
    "test images out of",
    len(sub_basename_set),
)

MAX_TRAIN_PER_CHAIN = 1400
NN_K = 25  # unchanged (keeps the same voting approach/semantics)

chains_to_index = set(img2chain_by_exact.values()) | set(img2chain_by_basename.values())
chains_to_index.add(0)

train_hash_index = {}  # chain -> list of (hash:(int,int), hotel_id:str)
total_hashed = 0

for ch in sorted(chains_to_index):
    dfc = train[train["chain"] == int(ch)][["image", "hotel_id"]].copy()
    if dfc.empty:
        continue

    dfc["__cnt"] = dfc.groupby("hotel_id")["hotel_id"].transform("size")
    dfc = dfc.sort_values("__cnt", ascending=False).drop(columns="__cnt")

    entries = []
    seen = 0
    for _, row in dfc.iterrows():
        if seen >= MAX_TRAIN_PER_CHAIN:
            break
        img = str(row["image"])
        hid = str(row["hotel_id"])
        img_path = os.path.join(train_images_dir, str(ch), img)
        if not os.path.exists(img_path):
            continue
        h = dhash128_from_path(img_path)
        if h is None:
            continue
        entries.append((h, hid))
        seen += 1

    if entries:
        train_hash_index[int(ch)] = entries
        total_hashed += len(entries)

print(
    "Built train hash entries:",
    total_hashed,
    "across chains:",
    sorted(train_hash_index.keys())[:10],
    "...",
)


def resolve_test_path(img_key: str) -> str:
    base = os.path.basename(img_key)
    p = test_basename_to_fullpath.get(base)
    if p is not None and os.path.exists(p):
        return p

    candidates = [
        os.path.join(test_images_dir, img_key),
        os.path.join(test_images_dir, base),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    ch = img2chain_by_exact.get(img_key, None)
    if ch is None:
        ch = img2chain_by_basename.get(base, None)
    if ch is not None:
        p = os.path.join(test_images_dir, str(ch), base)
        if os.path.exists(p):
            return p
    return None


def predict_top5_for_image(img_key: str) -> list:
    ch = img2chain_by_exact.get(img_key, None)
    if ch is None:
        ch = img2chain_by_basename.get(os.path.basename(img_key), None)

    if ch is not None and ch in chain_top5:
        anchors = chain_top5[ch]
        mixed = merge_top5(anchors, [])
        for a in anchors:
            others = chain_hotelid_to_top5_other.get((int(ch), str(a)), None)
            if others is not None:
                mixed = merge_top5(mixed, others)
        prior_top5 = merge_top5(mixed, global_top5)
    else:
        prior_top5 = global_top5

    idx = None
    if ch is not None and int(ch) in train_hash_index:
        idx = train_hash_index[int(ch)]
    elif 0 in train_hash_index:
        idx = train_hash_index[0]

    if not idx:
        return prior_top5

    tp = resolve_test_path(img_key)
    if tp is None:
        return prior_top5

    th = dhash128_from_path(tp)
    if th is None:
        return prior_top5

    dists = []
    for hh, hid in idx:
        dists.append((hamming128(th, hh), hid))
    dists.sort(key=lambda x: x[0])
    nn = dists[:NN_K]

    vote = {}
    for dist, hid in nn:
        w = 1.0 / (1.0 + float(dist))
        vote[hid] = vote.get(hid, 0.0) + w

    nn_ranked = sorted(vote.items(), key=lambda x: (-x[1], x[0]))
    nn_top = [hid for hid, _ in nn_ranked][:5]

    return merge_top5(nn_top, prior_top5)


sub = sub[["image"]].copy()

preds = []
for img in sub["image"].astype(str).tolist():
    top5 = predict_top5_for_image(img)
    preds.append(" ".join(top5))

sub["hotel_id"] = preds

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print("Rows:", len(sub), "Cols:", list(sub.columns))



## === cell 1
import pandas as pd

chk = pd.read_csv("submission.csv")
assert list(chk.columns) == [
    "image",
    "hotel_id",
], "Submission columns must be: image, hotel_id"
assert chk["image"].notnull().all(), "All images must be present"
assert chk["hotel_id"].notnull().all(), "All predictions must be present"
assert (
    chk["hotel_id"].astype(str).str.split().str.len() == 5
).all(), "Each prediction must contain 5 space-delimited hotel_ids"
print("submission.csv looks valid.")
print(chk.head())
