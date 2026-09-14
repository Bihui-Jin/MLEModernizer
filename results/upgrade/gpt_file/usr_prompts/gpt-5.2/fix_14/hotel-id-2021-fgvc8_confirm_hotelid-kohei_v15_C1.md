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
import hashlib
import zlib
import pandas as pd

BASE = "/kaggle/input/hotel-id-2021-fgvc8"
train_path = os.path.join(BASE, "train.csv")
sample_sub_path = os.path.join(BASE, "sample_submission.csv")
test_images_dir = os.path.join(BASE, "test_images")

assert os.path.exists(train_path), f"Missing train.csv at {train_path}"
assert os.path.exists(
    sample_sub_path
), f"Missing sample_submission.csv at {sample_sub_path}"
assert os.path.exists(test_images_dir), f"Missing test_images dir at {test_images_dir}"

train = pd.read_csv(train_path)
sample_sub = pd.read_csv(sample_sub_path)

assert "hotel_id" in train.columns, "train.csv must contain hotel_id"
assert "image" in train.columns, "train.csv must contain image"
assert (
    "image" in sample_sub.columns and "hotel_id" in sample_sub.columns
), "sample_submission.csv schema mismatch"



## === cell 1
train["hotel_id"] = train["hotel_id"].astype(str)
train["image"] = train["image"].astype(str)

train["stem"] = train["image"].str.replace(".jpg", "", regex=False)
train["prefix2"] = train["image"].str[:2]
train["prefix4"] = train["image"].str[:4]

global_top5 = train["hotel_id"].value_counts().head(5).index.astype(str).tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 + [global_top5[-1]] * 5)[:5]


def _pad_to_5(lst):
    lst = [str(x) for x in lst]
    if len(lst) >= 5:
        return lst[:5]
    need = 5 - len(lst)
    pad = [x for x in global_top5 if x not in lst][:need]
    lst = (lst + pad)[:5]
    if len(lst) < 5:
        lst = (lst + [lst[-1]] * 5)[:5]
    return lst


def _build_top5_map(key_col: str):
    mp = {}
    vc = train.groupby(key_col)["hotel_id"].value_counts().groupby(level=0).head(5)
    for (k, hid), _cnt in vc.items():
        mp.setdefault(str(k), []).append(str(hid))
    for k in list(mp.keys()):
        mp[k] = _pad_to_5(mp[k])
    return mp


stem_top5 = _build_top5_map("stem")
prefix2_top5 = _build_top5_map("prefix2")
prefix4_top5 = _build_top5_map("prefix4")

train_images_dir = os.path.join(BASE, "train_images")
assert os.path.exists(
    train_images_dir
), f"Missing train_images dir at {train_images_dir}"

folder_ids = [
    d
    for d in os.listdir(train_images_dir)
    if d.isdigit() and os.path.isdir(os.path.join(train_images_dir, d))
]

train_img_to_folder = {}
for d in folder_ids:
    p = os.path.join(train_images_dir, d)
    try:
        for fn in os.listdir(p):
            if fn.endswith(".jpg"):
                train_img_to_folder[fn] = d
    except PermissionError:
        continue

train["folder_dir"] = train["image"].map(train_img_to_folder).fillna("-1").astype(str)
folder_top5 = _build_top5_map("folder_dir")


def _build_prefix_folder_rank(prefix_col: str, topk: int):
    rank = {}
    vc = (
        train[train["folder_dir"].ne("-1")]
        .groupby(prefix_col)["folder_dir"]
        .value_counts()
        .groupby(level=0)
        .head(topk)
    )
    for (p, fd), _cnt in vc.items():
        rank.setdefault(str(p), []).append(str(fd))
    return rank


prefix4_folder_rank = _build_prefix_folder_rank("prefix4", topk=10)
prefix2_folder_rank = _build_prefix_folder_rank("prefix2", topk=10)




## === cell 2
def infer_prefix2(img_name: str) -> str:
    img_name = str(img_name)
    return img_name[:2] if len(img_name) >= 2 else ""


def infer_prefix4(img_name: str) -> str:
    img_name = str(img_name)
    return img_name[:4] if len(img_name) >= 4 else infer_prefix2(img_name)


def infer_stem(img_name: str) -> str:
    img_name = str(img_name)
    return img_name[:-4] if img_name.endswith(".jpg") else img_name


def infer_candidate_folders(img_name: str):
    p4 = infer_prefix4(img_name)
    p2 = infer_prefix2(img_name)
    cands = []
    for fd in prefix4_folder_rank.get(p4, []):
        if fd != "-1":
            cands.append(fd)
    for fd in prefix2_folder_rank.get(p2, []):
        if fd != "-1" and fd not in cands:
            cands.append(fd)
    return cands


need_imgs = set(sample_sub["image"].astype(str).tolist())

test_img_to_folder = {}
test_dir_entries = os.listdir(test_images_dir)
digit_subdirs = [
    d
    for d in test_dir_entries
    if d.isdigit() and os.path.isdir(os.path.join(test_images_dir, d))
]

if digit_subdirs:
    for d in digit_subdirs:
        p = os.path.join(test_images_dir, d)
        try:
            for fn in os.listdir(p):
                if fn in need_imgs:
                    test_img_to_folder[fn] = d
        except PermissionError:
            continue
else:
    test_img_to_folder = {}



## === cell 3
from PIL import Image
from collections import defaultdict, Counter
import time


def _safe_open_rgb(path):
    try:
        with Image.open(path) as im:
            return im.convert("L")
    except Exception:
        return None


def dhash64(image_gray, hash_size=8):
    im = image_gray.resize((hash_size + 1, hash_size), resample=Image.BILINEAR)
    pixels = list(im.getdata())
    rows = [
        pixels[i * (hash_size + 1) : (i + 1) * (hash_size + 1)]
        for i in range(hash_size)
    ]
    bits = 0
    bitpos = 0
    for r in rows:
        for c in range(hash_size):
            if r[c] > r[c + 1]:
                bits |= 1 << bitpos
            bitpos += 1
    return bits  # 64-bit int


def hamming64(a, b):
    return (a ^ b).bit_count()


def _img_path_train(folder_dir, fn):
    return os.path.join(train_images_dir, str(folder_dir), fn)


def _img_path_test(fn):
    fd = test_img_to_folder.get(fn, None)
    if fd is None:
        return os.path.join(test_images_dir, fn)
    return os.path.join(test_images_dir, str(fd), fn)


PER_FOLDER_TRAIN = 350
MAX_TRAIN_TOTAL = 25000
HASH_NEAR_DIST = 6  # allow slight variation; still minimal and deterministic

t0 = time.time()
train_hash_to_hotels = defaultdict(list)

total_added = 0
for d in folder_ids:
    p = os.path.join(train_images_dir, d)
    try:
        fns = [fn for fn in os.listdir(p) if fn.endswith(".jpg")]
    except PermissionError:
        continue
    fns = sorted(fns)[:PER_FOLDER_TRAIN]
    for fn in fns:
        if total_added >= MAX_TRAIN_TOTAL:
            break
        img = _safe_open_rgb(os.path.join(p, fn))
        if img is None:
            continue
        h = dhash64(img)
        hid = train.loc[train["image"].eq(fn), "hotel_id"]
        if len(hid) == 0:
            continue
        train_hash_to_hotels[h].append(str(hid.iloc[0]))
        total_added += 1
    if total_added >= MAX_TRAIN_TOTAL:
        break

train_hash_top5 = {}
for h, hids in train_hash_to_hotels.items():
    cnt = Counter(hids)
    top5 = [k for k, _v in cnt.most_common(5)]
    train_hash_top5[h] = _pad_to_5(top5)

hash_bucket = defaultdict(list)
for h in train_hash_top5.keys():
    hash_bucket[(h >> 48) & 0xFFFF].append(h)

print(
    f"Built train hash index: {len(train_hash_top5)} unique hashes from {total_added} images in {time.time()-t0:.1f}s"
)


def predict_from_image_hash(test_fn: str):
    path = _img_path_test(test_fn)
    img = _safe_open_rgb(path)
    if img is None:
        return None
    th = dhash64(img)
    if th in train_hash_top5:
        return train_hash_top5[th]
    b = (th >> 48) & 0xFFFF
    cands = hash_bucket.get(b, [])
    best = None
    best_d = 10**9
    for ch in cands:
        d = hamming64(th, ch)
        if d < best_d:
            best_d = d
            best = ch
            if best_d == 0:
                break
    if best is not None and best_d <= HASH_NEAR_DIST:
        return train_hash_top5[best]
    return None




## === cell 4
preds = []
for img in sample_sub["image"].astype(str).tolist():
    top5 = predict_from_image_hash(img)

    if top5 is None:
        fd = test_img_to_folder.get(img, None)

        if fd is not None and fd in folder_top5 and fd != "-1":
            top5 = folder_top5[fd]
        else:
            for cfd in infer_candidate_folders(img):
                if cfd in folder_top5 and cfd != "-1":
                    top5 = folder_top5[cfd]
                    break

            if top5 is None:
                st = infer_stem(img)
                top5 = stem_top5.get(st, None)

            if top5 is None:
                p4 = infer_prefix4(img)
                p2 = infer_prefix2(img)
                top5 = prefix4_top5.get(p4, None)
                if top5 is None:
                    top5 = prefix2_top5.get(p2, global_top5)

    preds.append(" ".join(_pad_to_5(top5)))

sub = sample_sub[["image"]].copy()
sub["hotel_id"] = preds

assert len(sub) == len(sample_sub), "Submission row count mismatch"
assert (
    sub["hotel_id"].str.split().map(len).eq(5).all()
), "Each prediction must contain exactly 5 IDs"
sub.to_csv("submission.csv", index=False)
print(sub.head())



## === cell 5
import subprocess

subprocess.run(["ls", "-lha", "submission.csv"], check=True)
subprocess.run(["head", "submission.csv"], check=True)
