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
from collections import Counter
from PIL import Image
import pandas as pd

print("Listing /kaggle/input:")
print("\n".join(os.listdir("/kaggle/input")))

candidate_sample_paths = [
    "/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
]
sample_path = next((p for p in candidate_sample_paths if os.path.exists(p)), None)
if sample_path is None:
    raise FileNotFoundError(
        f"Could not find sample_submission.csv in any of: {candidate_sample_paths}"
    )

candidate_train_paths = [
    "/kaggle/input/hotel-id-2021-fgvc8/train.csv",
    "/kaggle/input/train.csv",
]
train_path = next((p for p in candidate_train_paths if os.path.exists(p)), None)
if train_path is None:
    raise FileNotFoundError(
        f"Could not find train.csv in any of: {candidate_train_paths}"
    )

candidate_train_img_dirs = [
    "/kaggle/input/hotel-id-2021-fgvc8/train_images",
    "/kaggle/input/train_images",
]
train_img_dir = next((p for p in candidate_train_img_dirs if os.path.isdir(p)), None)
if train_img_dir is None:
    raise FileNotFoundError(
        f"Could not find train_images directory in any of: {candidate_train_img_dirs}"
    )

candidate_test_img_dirs = [
    "/kaggle/input/hotel-id-2021-fgvc8/test_images",
    "/kaggle/input/test_images",
]
test_img_dir = next((p for p in candidate_test_img_dirs if os.path.isdir(p)), None)
if test_img_dir is None:
    raise FileNotFoundError(
        f"Could not find test_images directory in any of: {candidate_test_img_dirs}"
    )

sub = pd.read_csv(sample_path)
if list(sub.columns) != ["image", "hotel_id"]:
    raise ValueError(f"Unexpected columns in sample submission: {sub.columns.tolist()}")

train = pd.read_csv(train_path, usecols=["image", "chain", "hotel_id"])

print(f"Using train_images dir: {train_img_dir}")
print(f"Using test_images dir: {test_img_dir}")
print(f"Train shape: {train.shape}, Test(sub) shape: {sub.shape}")



## === cell 1
import numpy as np

try:
    import cv2
except Exception as e:
    cv2 = None
    print("WARNING: cv2 not available; will fall back to PIL (slower). Error:", repr(e))


def dhash64_from_gray8(gray8_2d: np.ndarray, hash_size: int = 8) -> int:
    if cv2 is None:
        im = Image.fromarray(gray8_2d, mode="L").resize(
            (hash_size + 1, hash_size), Image.Resampling.BILINEAR
        )
        a = np.asarray(im, dtype=np.uint8)
    else:
        a = cv2.resize(
            gray8_2d,
            (hash_size + 1, hash_size),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8, copy=False)
    diff = (a[:, :-1] > a[:, 1:]).reshape(-1).astype(np.uint8, copy=False)
    bits = np.packbits(diff, bitorder="little")
    return int(np.frombuffer(bits.tobytes(), dtype="<u8")[0])


def safe_open_gray8(path: str):
    if cv2 is not None:
        try:
            g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            return g
        except Exception:
            return None
    try:
        with Image.open(path) as im:
            im.load()
            return np.asarray(im.convert("L"), dtype=np.uint8)
    except Exception:
        return None


global_top5 = train["hotel_id"].value_counts().head(5).index.tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 * 5)[:5]
global_default_pred = " ".join(map(str, global_top5))
print(f"Global top5 hotel_ids: {global_top5}")

SEED = 123
random.seed(SEED)
np.random.seed(SEED)

max_index_images = 88000  # near full train; improves score vs 12k, while still feasible
per_hotel_cap = (
    1000  # effectively "no cap" for most hotels, but avoids extreme domination
)

train_sorted = train.copy()
hotel_counts = train_sorted["hotel_id"].value_counts()
train_sorted["hotel_freq"] = train_sorted["hotel_id"].map(hotel_counts).astype(int)
train_sorted = train_sorted.sort_values(["hotel_freq"], ascending=False)

selected_rows = []
hotel_used = Counter()
for row in train_sorted.itertuples(index=False):
    img_id = row.image
    ch = int(row.chain)
    hid = int(row.hotel_id)
    if hotel_used[hid] >= per_hotel_cap:
        continue
    img_path = os.path.join(train_img_dir, str(ch), img_id)
    if not os.path.exists(img_path):
        continue
    selected_rows.append((img_id, ch, hid, img_path))
    hotel_used[hid] += 1
    if len(selected_rows) >= max_index_images:
        break

print(
    f"Indexed training images: {len(selected_rows)} (max_index_images={max_index_images}, per_hotel_cap={per_hotel_cap})"
)

train_hashes = []
bad_train = 0
for _, _, hid, path in selected_rows:
    g = safe_open_gray8(path)
    if g is None:
        bad_train += 1
        continue
    train_hashes.append((dhash64_from_gray8(g), int(hid)))
print(f"Train hashes computed: {len(train_hashes)} (failed: {bad_train})")

if len(train_hashes) < 500:
    print(
        "WARNING: Too few train hashes; falling back to global prior for all predictions."
    )
    sub["hotel_id"] = global_default_pred
    out_path = "submission.csv"
    sub.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape={sub.shape}")
else:
    k_nn = 50  # Change: slightly larger neighborhood for stability with larger index.
    preds = []
    bad_test = 0

    th_arr = np.fromiter(
        (h for h, _ in train_hashes), dtype=np.uint64, count=len(train_hashes)
    )
    tid_arr = np.fromiter(
        (hid for _, hid in train_hashes), dtype=np.int32, count=len(train_hashes)
    )

    _pc_table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def hamming_all(qh_uint64: np.uint64, th_uint64_arr: np.ndarray) -> np.ndarray:
        xor = np.bitwise_xor(th_uint64_arr, qh_uint64)
        xb = xor.view(np.uint8).reshape(-1, 8)
        return _pc_table[xb].sum(axis=1, dtype=np.uint16)

    PREFIX_BITS = 12
    _shift = np.uint64(64 - PREFIX_BITS)
    prefixes = (th_arr >> _shift).astype(np.uint16, copy=False)
    sort_idx = np.argsort(prefixes, kind="mergesort")
    prefixes_sorted = prefixes[sort_idx]
    th_sorted = th_arr[sort_idx]
    tid_sorted = tid_arr[sort_idx]

    uniq_pfx, start_idx = np.unique(prefixes_sorted, return_index=True)
    end_idx = np.empty_like(start_idx)
    end_idx[:-1] = start_idx[1:]
    end_idx[-1] = len(prefixes_sorted)
    pfx_to_range = {
        int(p): (int(s), int(e)) for p, s, e in zip(uniq_pfx, start_idx, end_idx)
    }

    test_path_map = {}
    try:
        for fn in os.listdir(test_img_dir):
            if fn.endswith(".jpg"):
                test_path_map[fn] = os.path.join(test_img_dir, fn)
    except Exception:
        test_path_map = {}

    if len(test_path_map) == 0:
        for root, _, files in os.walk(test_img_dir):
            for fn in files:
                if fn.endswith(".jpg"):
                    test_path_map[fn] = os.path.join(root, fn)

    for img in sub["image"].tolist():
        test_path = test_path_map.get(img, "")
        g = safe_open_gray8(test_path) if test_path else None
        if g is None:
            bad_test += 1
            preds.append(global_default_pred)
            continue

        qh = np.uint64(dhash64_from_gray8(g))
        qprefix = int((qh >> _shift) & np.uint64((1 << PREFIX_BITS) - 1))

        r = pfx_to_range.get(qprefix, None)
        if r is None:
            cand_th = th_arr
            cand_tid = tid_arr
        else:
            s, e = r
            if (e - s) < 200:
                cand_th = th_arr
                cand_tid = tid_arr
            else:
                cand_th = th_sorted[s:e]
                cand_tid = tid_sorted[s:e]

        dists = hamming_all(qh, cand_th)

        if len(dists) > k_nn:
            idx = np.argpartition(dists, k_nn - 1)[:k_nn]
            idx = idx[np.argsort(dists[idx], kind="mergesort")]
        else:
            idx = np.argsort(dists, kind="mergesort")

        nn_hids = cand_tid[idx].astype(np.int64, copy=False)
        nn_d = dists[idx].astype(np.float64, copy=False)
        w = 1.0 / (1.0 + nn_d)

        order = np.argsort(nn_hids, kind="mergesort")
        nn_hids_s = nn_hids[order]
        w_s = w[order]
        uniq, start = np.unique(nn_hids_s, return_index=True)
        sums = np.add.reduceat(w_s, start)

        ranked = uniq[np.argsort(-sums, kind="mergesort")].tolist()
        out5 = (ranked + global_top5)[:5]
        preds.append(" ".join(map(str, out5)))

    sub["hotel_id"] = preds
    out_path = "submission.csv"
    sub.to_csv(out_path, index=False)

    print(f"Bad/unreadable test images (fallback to global): {bad_test} / {len(sub)}")
    print(f"Wrote {out_path} with shape={sub.shape} and columns={sub.columns.tolist()}")
    print(sub.head())



## === cell 2
with open("submission.csv", "r", encoding="utf-8") as f:
    for _ in range(6):
        print(f.readline().rstrip("\n"))
