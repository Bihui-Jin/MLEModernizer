# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.7433042855940036

# 6. Current score

0.00302

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I remove the non‑code text and the faulty external‑library calls, and replace them with a self‑contained script that loads the training CSV, extracts the five most frequent hotel_id values, lists all test‑image filenames, and writes a proper `submission.csv` where each image is assigned that same space‑separated list. This ensures a valid CSV is always produced, fixing the syntax errors and guaranteeing the notebook finishes successfully.'
- What this solution (achieved 0.00209) has done: 'I keep the original data‑loading and folder‑checking logic, but replace the single constant “top‑5” string with a per‑image list: if a test image name appears in the training CSV we use its true `hotel_id` as the first prediction and fill the remaining slots with the most common hotels (skipping duplicates). For all other images we fall back to the original most‑common list. This minimal change adds useful label information where it exists and keeps the fallback behavior, moving the MAP@5 score toward the target without altering the overall pipeline.'
- What this solution (achieved 0.00179) has done: 'I keep the overall workflow the same but add a cheap “nearest‑file‑size” heuristic: for each test image we look at its file size, find the training image with the closest size, and use that image’s hotel_id as the first prediction (skipping it from the global most‑common list). This keeps the core logic intact, still writes a correct CSV, and is expected to raise the MAP@5 a bit toward the target without altering the model architecture or training process. The rest of the code remains unchanged.'
- What this solution (achieved 0.00112) has done: 'I enhance the heuristic by selecting up to five nearest‑size training images for each test image and using their hotel IDs (in order of size closeness) as the primary predictions, filling any remaining slots with the overall most‑common hotels. This keeps the original workflow intact while providing more varied and potentially more accurate top‑5 lists, moving the MAP@5 score much closer to the target. The script still writes a correct `submission.csv`.'
- What this solution (achieved 0.00112) has done: 'I increase the neighbor pool used for the size‑based heuristic from 5 to 50 so that each test image can draw from a larger set of nearby training images before filling the remaining slots with the global most‑common hotels. This keeps the original workflow intact while giving a higher chance of including the correct hotel_id in the top‑5 list, moving the MAP@5 score nearer to the target.'
- What this solution (achieved 0.00302) has done: 'I increase the pool of size‑based neighbours and select the most frequent hotels among them, which gives a better chance of covering the correct hotel ID while keeping the same overall workflow. The `nearest_hotels` helper now returns the raw neighbour list (allowing duplicates) and the prediction loop picks the most common IDs from that list before filling any remaining slots with the global top‑5 hotels.'

# 9. Code solution

## === cell 0
import os, glob, pandas as pd
from collections import Counter
import bisect

print("Input directories:")
for root, dirs, files in os.walk("/kaggle/input", topdown=False):
    if files:
        print(os.path.relpath(root, "/kaggle/input"))




## === cell 1
submission_path = "submission.csv"

train_csv_candidates = [
    "/kaggle/input/hotel-id-2021-fgvc8/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/input/hotel-id-2021-fgvc8/train.csv.zip",
    "/kaggle/input/train.csv.zip",
]
train_path = next((p for p in train_csv_candidates if os.path.exists(p)), None)
if train_path is None:
    raise FileNotFoundError("Training CSV not found in expected locations.")

train_df = pd.read_csv(train_path)

top5_hotels = [str(h) for h, _ in Counter(train_df["hotel_id"]).most_common(5)]

image_to_hotel = dict(
    zip(train_df["image"].astype(str), train_df["hotel_id"].astype(str))
)

train_image_dirs = [
    "/kaggle/input/hotel-id-2021-fgvc8/train_images",
    "/kaggle/input/train_images",
    "/kaggle/input/hotel-id-2021-fgvc8/train_images/",
    "/kaggle/input/train_images/",
]
train_img_dir = next((d for d in train_image_dirs if os.path.isdir(d)), None)
if train_img_dir is None:
    raise FileNotFoundError("Train images directory not found.")

size_hotel_pairs = []
for root, _, files in os.walk(train_img_dir):
    for f in files:
        if f.lower().endswith(".jpg"):
            img_path = os.path.join(root, f)
            try:
                sz = os.path.getsize(img_path)
            except OSError:
                continue
            hid = image_to_hotel.get(f)
            if hid is not None:
                size_hotel_pairs.append((sz, hid))

size_hotel_pairs.sort(key=lambda x: x[0])
sizes_sorted = [p[0] for p in size_hotel_pairs]
hotels_sorted = [p[1] for p in size_hotel_pairs]

test_image_dirs = [
    "/kaggle/input/hotel-id-2021-fgvc8/test_images",
    "/kaggle/input/test_images",
    "/kaggle/input/hotel-id-2021-fgvc8/test_images/",
    "/kaggle/input/test_images/",
]
test_dir = next((d for d in test_image_dirs if os.path.isdir(d)), None)
if test_dir is None:
    raise FileNotFoundError("Test images directory not found.")

test_files = sorted(
    [os.path.basename(p) for p in glob.glob(os.path.join(test_dir, "*.jpg"))]
)


def nearest_hotels(test_sz, k=200):
    """
    Return up to k hotel_ids whose training‑image sizes are closest to ``test_sz``.
    Duplicates are kept so that frequency information can be used later.
    """
    if not sizes_sorted or test_sz is None:
        return []
    pos = bisect.bisect_left(sizes_sorted, test_sz)
    left, right = pos - 1, pos
    results = []
    while (left >= 0 or right < len(sizes_sorted)) and len(results) < k:
        left_diff = abs(sizes_sorted[left] - test_sz) if left >= 0 else None
        right_diff = (
            abs(sizes_sorted[right] - test_sz) if right < len(sizes_sorted) else None
        )
        if left_diff is not None and (right_diff is None or left_diff <= right_diff):
            hid = hotels_sorted[left]
            left -= 1
        else:
            hid = hotels_sorted[right]
            right += 1
        results.append(hid)
    return results


predictions = []
for img in test_files:
    if img in image_to_hotel:
        first_candidates = [image_to_hotel[img]]
    else:
        test_path = os.path.join(test_dir, img)
        try:
            test_sz = os.path.getsize(test_path)
        except OSError:
            test_sz = None

        neighbour_ids = nearest_hotels(test_sz, k=200) if test_sz is not None else []
        freq = Counter(neighbour_ids)
        first_candidates = [hid for hid, _ in freq.most_common()]

    pred_list = []
    for hid in first_candidates:
        if len(pred_list) >= 5:
            break
        pred_list.append(str(hid))
    for hid in top5_hotels:
        if len(pred_list) >= 5:
            break
        if hid not in pred_list:
            pred_list.append(hid)
    pred_list = pred_list[:5]
    predictions.append(" ".join(pred_list))

submission_df = pd.DataFrame({"image": test_files, "hotel_id": predictions})
submission_df.to_csv(submission_path, index=False)
print(f"Submission created with {len(submission_df)} rows at '{submission_path}'.")




## === cell 2
print("\nFirst 10 rows of submission:")
print(pd.read_csv(submission_path).head(10))
