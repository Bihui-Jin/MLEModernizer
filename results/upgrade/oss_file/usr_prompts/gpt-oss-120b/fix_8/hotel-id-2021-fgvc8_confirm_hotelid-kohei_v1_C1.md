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

TOP_HOTEL_LIMIT = 2000
selected_hotels = {hid for hid, _ in global_counter.most_common(TOP_HOTEL_LIMIT)}

train_image_patterns = [
    "**/train_images/**/*.jpg",
    "**/train_images/*.jpg",
]
train_image_files = []
for pattern in train_image_patterns:
    train_image_files.extend(glob.glob(pattern, recursive=True))

hist_sum = defaultdict(lambda: np.zeros(512, dtype=np.float32))
hist_cnt = Counter()


def compute_histogram(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()  # shape (512,)
    return hist


for img_path in train_image_files:
    img_name = os.path.basename(img_path)
    hid = image_to_hotel.get(img_name)
    if not hid or hid not in selected_hotels:
        continue
    hist = compute_histogram(img_path)
    if hist is None:
        continue
    hist_sum[hid] += hist
    hist_cnt[hid] += 1

hotel_mean_hist = {}
for hid, cnt in hist_cnt.items():
    if cnt > 0:
        hotel_mean_hist[hid] = hist_sum[hid] / cnt  # already 512‑dim vector


def top5_by_histogram(test_img_path, fallback_ids):
    """Return top‑5 hotel ids for a test image based on histogram distance."""
    hist = compute_histogram(test_img_path)
    if hist is None or not hotel_mean_hist:
        return fallback_ids
    distances = []
    for hid, mean_hist in hotel_mean_hist.items():
        dist = np.linalg.norm(hist - mean_hist)
        distances.append((dist, hid))
    distances.sort()
    top_ids = [hid for _, hid in distances[:5]]
    for fid in fallback_ids:
        if len(top_ids) >= 5:
            break
        if fid not in top_ids:
            top_ids.append(fid)
    return top_ids[:5]


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
            test_path = image_path_lookup.get(image_name)
            if test_path:
                top_ids = top5_by_histogram(test_path, global_top5)
            else:
                chain_id = image_to_chain.get(image_name)
                if chain_id and chain_id in chain_top5:
                    primary = chain_top5[chain_id][0]
                    filler = [hid for hid in global_top5 if hid != primary][:4]
                    top_ids = [primary] + filler
                else:
                    top_ids = global_top5
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
