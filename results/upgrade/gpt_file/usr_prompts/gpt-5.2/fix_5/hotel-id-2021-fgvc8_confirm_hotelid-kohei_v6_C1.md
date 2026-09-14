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
from collections import Counter, defaultdict
from PIL import Image
import pandas as pd

print("Listing /kaggle/input:")
print("\n".join(os.listdir("/kaggle/input")))



## === cell 1
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



## === cell 2


def dhash64(img: Image.Image, hash_size: int = 8) -> int:
    img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = list(img.getdata())
    rows = [
        pixels[i * (hash_size + 1) : (i + 1) * (hash_size + 1)]
        for i in range(hash_size)
    ]
    h = 0
    bit = 0
    for r in rows:
        for c in range(hash_size):
            if r[c] > r[c + 1]:
                h |= 1 << bit
            bit += 1
    return h


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def safe_open_image(path: str):
    try:
        with Image.open(path) as im:
            im.load()
            return im.copy()
    except Exception:
        return None


def find_image_path(root_dir: str, image_id: str) -> str:
    p = os.path.join(root_dir, image_id)
    if os.path.exists(p):
        return p
    for c in range(0, 100):
        pc = os.path.join(root_dir, str(c), image_id)
        if os.path.exists(pc):
            return pc
    return ""


global_top5 = train["hotel_id"].value_counts().head(5).index.tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 * 5)[:5]
global_default_pred = " ".join(map(str, global_top5))
print(f"Global top5 hotel_ids: {global_top5}")

SEED = 123
random.seed(SEED)

max_index_images = 12000  # keeps runtime under control vs hashing all ~88k images
per_hotel_cap = 3  # avoid over-indexing single frequent hotels

train_sorted = train.copy()
hotel_counts = train_sorted["hotel_id"].value_counts()
train_sorted["hotel_freq"] = train_sorted["hotel_id"].map(hotel_counts).astype(int)
train_sorted = train_sorted.sort_values(["hotel_freq"], ascending=False)

selected_rows = []
hotel_used = Counter()
for _, row in train_sorted.iterrows():
    hid = int(row["hotel_id"])
    if hotel_used[hid] >= per_hotel_cap:
        continue
    img_path = os.path.join(train_img_dir, str(int(row["chain"])), row["image"])
    if not os.path.exists(img_path):
        continue
    selected_rows.append((row["image"], int(row["chain"]), hid, img_path))
    hotel_used[hid] += 1
    if len(selected_rows) >= max_index_images:
        break

if len(selected_rows) < 2000:
    extra_needed = 2000 - len(selected_rows)
    extra = []
    for _, row in train.sample(
        min(len(train), extra_needed * 5), random_state=SEED
    ).iterrows():
        img_path = os.path.join(train_img_dir, str(int(row["chain"])), row["image"])
        if os.path.exists(img_path):
            extra.append(
                (row["image"], int(row["chain"]), int(row["hotel_id"]), img_path)
            )
            if len(extra) >= extra_needed:
                break
    selected_rows.extend(extra)

print(
    f"Indexed training images: {len(selected_rows)} (max_index_images={max_index_images}, per_hotel_cap={per_hotel_cap})"
)

train_hashes = []
bad_train = 0
for img_id, ch, hid, path in selected_rows:
    im = safe_open_image(path)
    if im is None:
        bad_train += 1
        continue
    train_hashes.append((dhash64(im), hid))
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
    k_nn = 40  # evaluate a modest neighborhood to form a more reliable top-5 vote
    preds = []
    bad_test = 0

    th = [h for h, _ in train_hashes]
    tid = [hid for _, hid in train_hashes]

    for img in sub["image"].tolist():
        test_path = find_image_path(test_img_dir, img)
        im = safe_open_image(test_path) if test_path else None
        if im is None:
            bad_test += 1
            preds.append(global_default_pred)
            continue

        qh = dhash64(im)

        best = []
        for i in range(len(th)):
            d = hamming64(qh, th[i])
            best.append((d, tid[i]))
        best.sort(key=lambda x: x[0])
        best = best[:k_nn]

        vote = defaultdict(float)
        for d, hid in best:
            vote[int(hid)] += 1.0 / (1.0 + d)

        ranked = [
            hid for hid, _ in sorted(vote.items(), key=lambda x: x[1], reverse=True)
        ]
        out5 = (ranked + global_top5)[:5]
        preds.append(" ".join(map(str, out5)))

    sub["hotel_id"] = preds
    out_path = "submission.csv"
    sub.to_csv(out_path, index=False)

    print(f"Bad/unreadable test images (fallback to global): {bad_test} / {len(sub)}")
    print(f"Wrote {out_path} with shape={sub.shape} and columns={sub.columns.tolist()}")
    print(sub.head())



## === cell 3
with open("submission.csv", "r", encoding="utf-8") as f:
    for _ in range(6):
        print(f.readline().rstrip("\n"))
