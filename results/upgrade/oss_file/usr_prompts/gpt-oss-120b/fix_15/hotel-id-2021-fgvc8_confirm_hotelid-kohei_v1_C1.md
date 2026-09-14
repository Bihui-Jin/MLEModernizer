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

numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88

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
import csv
import cv2
from collections import Counter, defaultdict
import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


def locate_path(possible_paths):
    for p in possible_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the expected paths exist: {possible_paths}")


train_csv_candidates = [
    "hotel-id-2021-fgvc8/train.csv",
    "input/hotel-id-2021-fgvc8/train.csv",
    "data/hotel-id-2021-fgvc8/train.csv",
    "train.csv",
]
train_csv_path = locate_path(train_csv_candidates)

sample_sub_candidates = [
    "hotel-id-2021-fgvc8/sample_submission.csv",
    "input/hotel-id-2021-fgvc8/sample_submission.csv",
    "data/hotel-id-2021-fgvc8/sample_submission.csv",
    "sample_submission.csv",
]
sample_sub_path = locate_path(sample_sub_candidates)

global_counter = Counter()
chain_counters = defaultdict(Counter)
image_to_hotel = {}

with open(train_csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        hid = row["hotel_id"]
        cid = row["chain"]
        img = row["image"]
        global_counter[hid] += 1
        chain_counters[cid][hid] += 1
        image_to_hotel[img] = hid

global_top5 = [hid for hid, _ in global_counter.most_common(5)]
while len(global_top5) < 5:  # safety net if <5 distinct IDs
    global_top5.append(global_top5[0] if global_top5 else "0")

chain_top5 = {}
for cid, counter in chain_counters.items():
    top5 = [hid for hid, _ in counter.most_common(5)]
    while len(top5) < 5:
        top5.append(global_top5[len(top5) % len(global_top5)])
    chain_top5[cid] = top5

test_image_patterns = [
    "**/test_images/**/*.jpg",
    "**/test_images/*.jpg",
]
test_image_files = []
for pattern in test_image_patterns:
    test_image_files.extend(glob.glob(pattern, recursive=True))

image_to_chain = {}
image_path_lookup = {}  # filename -> full path
if test_image_files:
    for img_path in test_image_files:
        image_name = os.path.basename(img_path)
        image_path_lookup[image_name] = img_path
        parts = os.path.normpath(img_path).split(os.sep)
        chain_id = None
        if "test_images" in parts:
            idx = parts.index("test_images")
            if idx + 1 < len(parts):
                possible = parts[idx + 1]
                if possible.isdigit():
                    chain_id = possible
        image_to_chain[image_name] = chain_id

test_images = []
with open(sample_sub_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        test_images.append(row["image"])

if not test_images:
    raise RuntimeError("No test image IDs found in sample submission.")

selected_hotels = set(global_counter.keys())

train_image_patterns = [
    "**/train_images/**/*.jpg",
    "**/train_images/*.jpg",
]
train_image_files = []
for pattern in train_image_patterns:
    train_image_files.extend(glob.glob(pattern, recursive=True))


def compute_histogram(img_path):
    """Read an image, resize, convert to HSV and return a normalized 4096‑dim histogram."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 16], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


partial_sums = defaultdict(lambda: np.zeros(4096, dtype=np.float32))
partial_counts = Counter()


def process_path(img_path):
    img_name = os.path.basename(img_path)
    hid = image_to_hotel.get(img_name)
    if not hid:  # skip images without a known hotel id
        return None
    hist = compute_histogram(img_path)
    if hist is None:
        return None
    return (hid, hist)


max_workers = min(32, os.cpu_count() + 4)
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_path, p): p for p in train_image_files}
    for fut in as_completed(futures):
        result = fut.result()
        if result is None:
            continue
        hid, hist = result
        partial_sums[hid] += hist
        partial_counts[hid] += 1

hist_sum = defaultdict(lambda: np.zeros(4096, dtype=np.float32))
hist_cnt = Counter()
for hid, vec in partial_sums.items():
    hist_sum[hid] = vec
for hid, cnt in partial_counts.items():
    hist_cnt[hid] = cnt

hotel_mean_hist = {}
for hid, cnt in hist_cnt.items():
    if cnt > 0:
        hotel_mean_hist[hid] = hist_sum[hid] / cnt  # 4096‑dim mean vector

mean_ids = np.array(list(hotel_mean_hist.keys()))
mean_hists = np.stack([hotel_mean_hist[hid] for hid in mean_ids])  # shape (N,4096)


def top5_by_histogram(test_img_path, fallback_ids):
    """Return top‑5 hotel ids for a test image based on Euclidean distance to mean histograms."""
    hist = compute_histogram(test_img_path)
    if hist is None or mean_hists.size == 0:
        return fallback_ids
    diffs = mean_hists - hist  # (N,4096) broadcast
    dists = np.linalg.norm(diffs, axis=1)  # (N,)
    top_idx = np.argpartition(dists, 5)[:5]
    top_idx = top_idx[np.argsort(dists[top_idx])]
    top_ids = [mean_ids[i] for i in top_idx]
    for fid in fallback_ids:
        if len(top_ids) >= 5:
            break
        if fid not in top_ids:
            top_ids.append(fid)
    return top_ids[:5]


def merge_top_ids(primary, secondary, fallback, limit=5):
    """
    Combine three lists of candidate hotel ids.
    Order: primary → secondary → fallback, keeping uniqueness, up to `limit`.
    """
    result = []
    for src in (primary, secondary, fallback):
        for hid in src:
            if hid not in result:
                result.append(hid)
            if len(result) >= limit:
                return result[:limit]
    return result[:limit]


submission_path = "submission.csv"
with open(submission_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "hotel_id"])
    for image_name in test_images:
        if image_name in image_to_hotel:
            exact_hid = image_to_hotel[image_name]
            remaining = [hid for hid in global_top5 if hid != exact_hid][:4]
            top_ids = [exact_hid] + remaining
        else:
            chain_id = image_to_chain.get(image_name)
            chain_ids = chain_top5.get(chain_id, []) if chain_id else []
            hist_ids = []
            test_path = image_path_lookup.get(image_name, "")
            if test_path:
                hist_ids = top5_by_histogram(test_path, global_top5)
            top_ids = merge_top_ids(hist_ids, chain_ids, global_top5, limit=5)
        writer.writerow([image_name, " ".join(top_ids)])

print(f"Submission written to {submission_path} with {len(test_images)} rows.")
print(f"Global top‑5 IDs: {' '.join(global_top5)}")
print(f"Prepared per‑chain top‑5 for {len(chain_top5)} chains.")
print(f"Histogram model built for {len(hotel_mean_hist)} hotels.")




## === cell 1
import subprocess, sys

with open("submission.csv", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        sys.stdout.write(line)

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers have different ids
