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
import numpy as np
import pandas as pd
from PIL import Image

BASE = "/kaggle/input/hotel-id-2021-fgvc8"
train_csv = os.path.join(BASE, "train.csv")
sample_csv = os.path.join(BASE, "sample_submission.csv")
train_img_dir = os.path.join(BASE, "train_images")
test_img_dir = os.path.join(BASE, "test_images")

assert os.path.exists(train_csv), f"Missing train.csv at {train_csv}"
assert os.path.exists(sample_csv), f"Missing sample_submission.csv at {sample_csv}"
assert os.path.isdir(train_img_dir), f"Missing train_images dir at {train_img_dir}"
assert os.path.isdir(test_img_dir), f"Missing test_images dir at {test_img_dir}"

train_df = pd.read_csv(train_csv, usecols=["image", "chain", "hotel_id"])
sample = pd.read_csv(sample_csv)

top5_global = train_df["hotel_id"].value_counts().head(5).index.astype(str).tolist()
pred_str_global = " ".join(top5_global)

print("Global Top-5 prior hotel_ids:", top5_global)
print("Train rows:", len(train_df), "Test rows:", len(sample))




## === cell 1
import numpy as np
from collections import Counter, defaultdict

_POPCOUNT8 = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def dhash64(image_path, hash_size=8):
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8)
        diff = arr[:, 1:] > arr[:, :-1]  # shape (8,8) of booleans
        bits_u8 = diff.reshape(-1).astype(np.uint8, copy=False)  # row-major flatten

        packed = np.packbits(bits_u8, bitorder="big")  # 8 bytes
        return np.frombuffer(packed.tobytes(), dtype=">u8")[0].astype(
            np.uint64, copy=False
        )
    except Exception:
        return None


def _u64_bytes_view(u64_arr: np.ndarray) -> np.ndarray:
    return u64_arr.view(np.uint8).reshape(-1, 8)


def hamming_u64_vec_uint64_from_bytes(
    train_hash_bytes_u8: np.ndarray, th_uint64: np.uint64
) -> np.ndarray:
    th_bytes = np.uint64(th_uint64).view(np.uint8)  # shape (8,)
    xored = np.bitwise_xor(train_hash_bytes_u8, th_bytes)  # (N,8) uint8
    return _POPCOUNT8[xored].sum(axis=1).astype(np.int16, copy=False)


def train_image_path(chain, image_name):
    return os.path.join(train_img_dir, str(chain), image_name)


def test_image_path(image_name):
    return os.path.join(test_img_dir, image_name)




## === cell 2
MAX_TRAIN_INDEX = 50000  # cap for speed/memory; adjust upward if your runtime allows
K_NEIGHBORS = 50  # retrieve this many nearest hashes to vote hotel_ids
TOPK_SUBMIT = 5

train_subset = train_df.iloc[: min(len(train_df), MAX_TRAIN_INDEX)].copy()

train_hashes = []
train_hotels = []
train_missing = 0

for r in train_subset.itertuples(index=False):
    p = train_image_path(r.chain, r.image)
    h = dhash64(p)
    if h is None:
        train_missing += 1
        continue
    train_hashes.append(h)
    train_hotels.append(str(r.hotel_id))

train_hashes = np.array(train_hashes, dtype=np.uint64)
train_hotels = np.array(train_hotels, dtype=object)

assert len(train_hashes) == len(train_hotels)
print("Indexed train images:", len(train_hashes), "Missing/unreadable:", train_missing)

pop_rank = [k for k, _ in Counter(train_hotels.tolist()).most_common(50)]
if len(pop_rank) < 5:
    pop_rank = top5_global[:]  # ultimate fallback

train_hash_bytes = _u64_bytes_view(train_hashes)




## === cell 3
from concurrent.futures import ThreadPoolExecutor

preds = []
test_missing = 0


def predict_from_neighbors(nei_hotels, nei_dists, topk=5):
    if len(nei_hotels) == 0:
        return pred_str_global

    cnt = defaultdict(int)
    bestdist = {}
    for hid, d in zip(nei_hotels, nei_dists):
        cnt[hid] += 1
        if hid not in bestdist or d < bestdist[hid]:
            bestdist[hid] = d

    ranked = sorted(cnt.keys(), key=lambda h: (-cnt[h], bestdist[h], h))
    ranked = ranked[:topk]

    if len(ranked) < topk:
        for h in pop_rank:
            if h not in ranked:
                ranked.append(h)
            if len(ranked) == topk:
                break

    return " ".join(ranked[:topk])


test_images_list = sample["image"].astype(str).tolist()


def _hash_test_image(img_name: str):
    tp = test_image_path(img_name)
    return img_name, dhash64(tp)


max_workers = min(8, (os.cpu_count() or 2))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    hashed_iter = ex.map(_hash_test_image, test_images_list, chunksize=64)

    for img_name, th in hashed_iter:
        if th is None or len(train_hashes) == 0:
            test_missing += 1
            preds.append(pred_str_global)
            continue

        dists = hamming_u64_vec_uint64_from_bytes(train_hash_bytes, np.uint64(th))

        if len(dists) > K_NEIGHBORS:
            idx = np.argpartition(dists, K_NEIGHBORS)[:K_NEIGHBORS]
        else:
            idx = np.arange(len(dists))

        nei_hotels = train_hotels[idx]
        nei_dists = dists[idx]
        preds.append(predict_from_neighbors(nei_hotels, nei_dists, topk=TOPK_SUBMIT))

print("Test images unreadable:", test_missing)
print("Example prediction:", sample["image"].iloc[0], "->", preds[0])




## === cell 4
sub = sample.copy()
sub["hotel_id"] = preds

assert list(sub.columns) == [
    "image",
    "hotel_id",
], "Submission must have columns: image, hotel_id"
assert len(sub) == len(sample), "Submission row count must match sample_submission"
assert (
    sub["hotel_id"].astype(str).str.split().map(len).min() >= 1
), "Each prediction must be non-empty"

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())




## === cell 5
import os

assert os.path.exists("submission.csv"), "submission.csv was not created"
with open("submission.csv", "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))
