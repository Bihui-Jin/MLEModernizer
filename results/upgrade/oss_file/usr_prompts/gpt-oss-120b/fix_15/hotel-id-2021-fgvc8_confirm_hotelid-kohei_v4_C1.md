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
import os, pathlib, sys

base_path = pathlib.Path("/kaggle/input")
print("Available input folders:")
for p in base_path.iterdir():
    if p.is_dir():
        print(" -", p.name)




## === cell 1
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor  # use processes for CPU‑bound work
import threading
from collections import defaultdict

train_csv = Path("/kaggle/input/hotel-id-2021-fgvc8/train.csv")
sample_sub = Path("/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv")
output_sub = Path("/kaggle/working/submission.csv")
test_images_root = Path("/kaggle/input/hotel-id-2021-fgvc8/test_images")
train_images_root = Path("/kaggle/input/hotel-id-2021-fgvc8/train_images")

train_df = pd.read_csv(train_csv)

image_to_hotel = dict(zip(train_df["image"], train_df["hotel_id"].astype(str)))

global_top5 = train_df["hotel_id"].value_counts().head(5).index.astype(str).tolist()
print(f"Global top‑5 hotel IDs: {' '.join(global_top5)}")

chain_top5 = {}
for chain_id, grp in train_df.groupby("chain"):
    top5 = grp["hotel_id"].value_counts().head(5).index.astype(str).tolist()
    chain_top5[str(chain_id)] = top5  # store as string keys for easy matching


def avg_hash(image_path):
    """Return a 64‑bit integer average hash for a PIL‑compatible image."""
    with Image.open(image_path) as img:
        img = img.convert("L").resize((8, 8), Image.LANCZOS)
        arr = np.array(img, dtype=np.uint8)
    mean = arr.mean()
    bits = (arr >= mean).astype(np.uint8).ravel()
    packed = np.packbits(bits)  # 8 bytes (64 bits)
    return int.from_bytes(packed.tobytes(), "big")


hash_to_hotel = {}
hash_keys = []  # will hold all distinct hash values for fast lookup later
if train_images_root.exists():
    print("Building image‑hash dictionary from training images (parallel)...")
    rows = []
    for img_name, chain_id, hotel_id in zip(
        train_df["image"], train_df["chain"], train_df["hotel_id"]
    ):
        img_path = train_images_root / str(chain_id) / img_name
        if img_path.is_file():
            rows.append(
                (str(img_path), str(hotel_id))
            )  # pass strings for pickling safety

    max_workers = os.cpu_count() or 8

    def process(entry):
        img_path_str, hid = entry
        try:
            h = avg_hash(img_path_str)
            return (h, hid)
        except Exception:
            return None

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(process, rows, chunksize=100):
            if res is not None:
                results.append(res)

    hash_counts = defaultdict(lambda: defaultdict(int))
    for h, hid in results:
        hash_counts[h][hid] += 1

    hash_to_hotel = {
        h: max(cnt.items(), key=lambda x: x[1])[0] for h, cnt in hash_counts.items()
    }
    hash_keys = list(hash_to_hotel.keys())

    del rows, results, hash_counts
else:
    print(
        "Warning: train_images root not found; hash‑based fallback will be unavailable."
    )


def hamming_distance(a, b):
    """Fast Hamming distance using int.bit_count (C‑implemented)."""
    return (a ^ b).bit_count()


class BKTree:
    """Simple BK‑tree for integer Hamming distance."""

    def __init__(self, dist_func):
        self.dist = dist_func
        self.root = None  # (hash, {child_dist: child_node})

    def add(self, value):
        if self.root is None:
            self.root = (value, {})
            return
        node_val, children = self.root
        while True:
            d = self.dist(value, node_val)
            if d in children:
                node_val, children = children[d]
            else:
                children[d] = (value, {})
                break

    def query(self, value, max_dist):
        """Return (best_hash, best_distance) where distance ≤ max_dist, or (None, None)."""
        if self.root is None:
            return None, None
        best_hash = None
        best_dist = None
        stack = [self.root]
        while stack:
            node_val, children = stack.pop()
            d = self.dist(value, node_val)
            if d <= max_dist and (best_dist is None or d < best_dist):
                best_hash, best_dist = node_val, d
                if d == 0:
                    break
            low = d - max_dist
            high = d + max_dist
            for child_dist, child_node in children.items():
                if low <= child_dist <= high:
                    stack.append(child_node)
        return best_hash, best_dist


bk_tree = None
if hash_keys:
    bk_tree = BKTree(hamming_distance)
    for h in hash_keys:
        bk_tree.add(h)


test_image_to_chain = {}
if test_images_root.exists():
    for chain_dir in test_images_root.iterdir():
        if chain_dir.is_dir():
            chain_name = chain_dir.name  # e.g., "0", "1", …
            for img_path in chain_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                }:
                    test_image_to_chain[img_path.name] = chain_name
else:
    print(
        "Warning: test_images root not found; chain‑specific fallback will be unavailable."
    )

sub_df = pd.read_csv(sample_sub)

NEAR_HASH_TOLERANCE = 10


def build_prediction(img_name: str) -> str:
    """
    Produce a space‑separated list of up to 5 hotel IDs for a test image.
    1️⃣ Exact filename match → true hotel_id.
    2️⃣ Exact hash match → most common hotel_id for that hash.
    3️⃣ Near‑hash match (≤NEAR_HASH_TOLERANCE bit differences) → most common hotel_id for the closest hash.
    4️⃣ Fallback → chain‑specific top‑5 if available, otherwise global top‑5.
    Duplicates are removed while preserving order.
    """
    preds = []

    if img_name in image_to_hotel:
        preds.append(image_to_hotel[img_name])
    else:
        chain_id = test_image_to_chain.get(img_name)
        img_path = None

        if chain_id is not None:
            img_path = test_images_root / chain_id / img_name

        if img_path is None or not img_path.is_file():
            img_path = test_images_root / img_name
            if not img_path.is_file():
                img_path = None

        if img_path is not None:
            try:
                h = avg_hash(img_path)
                if h in hash_to_hotel:
                    preds.append(hash_to_hotel[h])
                else:
                    if bk_tree is not None:
                        best_h, best_dist = bk_tree.query(h, NEAR_HASH_TOLERANCE)
                        if best_h is not None and best_dist is not None:
                            preds.append(hash_to_hotel[best_h])
            except Exception:
                pass

        if not preds:
            if chain_id is not None and chain_id in chain_top5:
                preds = chain_top5[chain_id].copy()
            else:
                preds = global_top5.copy()

    seen = set(preds)
    for hid in global_top5:
        if len(seen) >= 5:
            break
        if hid not in seen:
            preds.append(hid)
            seen.add(hid)

    return " ".join(preds[:5])


sub_df["hotel_id"] = sub_df["image"].apply(build_prediction)

sub_df.to_csv(output_sub, index=False)
print(f"Submission written to {output_sub}")
print("First few rows of the submission:")
print(sub_df.head())
