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

0.00209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I remove the non‑code text and the faulty external‑library calls, and replace them with a self‑contained script that loads the training CSV, extracts the five most frequent hotel_id values, lists all test‑image filenames, and writes a proper `submission.csv` where each image is assigned that same space‑separated list. This ensures a valid CSV is always produced, fixing the syntax errors and guaranteeing the notebook finishes successfully.'
- What this solution (achieved 0.00209) has done: 'I keep the original data‑loading and folder‑checking logic, but replace the single constant “top‑5” string with a per‑image list: if a test image name appears in the training CSV we use its true `hotel_id` as the first prediction and fill the remaining slots with the most common hotels (skipping duplicates). For all other images we fall back to the original most‑common list. This minimal change adds useful label information where it exists and keeps the fallback behavior, moving the MAP@5 score toward the target without altering the overall pipeline.'
- What this solution (achieved 0.00179) has done: 'I keep the overall workflow the same but add a cheap “nearest‑file‑size” heuristic: for each test image we look at its file size, find the training image with the closest size, and use that image’s hotel_id as the first prediction (skipping it from the global most‑common list). This keeps the core logic intact, still writes a correct CSV, and is expected to raise the MAP@5 a bit toward the target without altering the model architecture or training process. The rest of the code remains unchanged.'
- What this solution (achieved 0.00112) has done: 'I enhance the heuristic by selecting up to five nearest‑size training images for each test image and using their hotel IDs (in order of size closeness) as the primary predictions, filling any remaining slots with the overall most‑common hotels. This keeps the original workflow intact while providing more varied and potentially more accurate top‑5 lists, moving the MAP@5 score much closer to the target. The script still writes a correct `submission.csv`.'
- What this solution (achieved 0.00112) has done: 'I increase the neighbor pool used for the size‑based heuristic from 5 to 50 so that each test image can draw from a larger set of nearby training images before filling the remaining slots with the global most‑common hotels. This keeps the original workflow intact while giving a higher chance of including the correct hotel_id in the top‑5 list, moving the MAP@5 score nearer to the target.'
- What this solution (achieved 0.00302) has done: 'I increase the pool of size‑based neighbours and select the most frequent hotels among them, which gives a better chance of covering the correct hotel ID while keeping the same overall workflow. The `nearest_hotels` helper now returns the raw neighbour list (allowing duplicates) and the prediction loop picks the most common IDs from that list before filling any remaining slots with the global top‑5 hotels.'
- What this solution (achieved 0.00467) has done: 'I added a lightweight “chain‑aware” heuristic and expanded the size‑based neighbour pool. For each test image we now look at the folder it resides in (the hotel chain id) and, if available, prepend the most frequent hotels for that chain. This keeps the original workflow intact while giving a more informed first‑guess list. The neighbour search also uses a larger pool (k=500) to improve the chance of covering the correct hotel ID. No core modeling logic was changed – only the prediction preprocessing was enhanced to move the MAP@5 score toward the target.'
- What this solution (achieved 0.00467) has done: 'I tighten the heuristic so that when a test image belongs to a known chain we rely exclusively on the most frequent hotels for that chain (up to five), discarding the noisy size‑based neighbours in that case. Only when the chain is unknown do we fall back to the size‑nearest heuristic. This keeps the core workflow unchanged while giving a much stronger, chain‑aware ranking, which should raise MAP@5 toward the target.'
- What this solution (achieved 0.00532) has done: 'I enhance the prediction logic by merging the chain‑aware frequent hotels with the size‑based neighbour list instead of using one of them exclusively. This combined candidate pool is then ranked by frequency, which should increase the chance that the true hotel_id appears in the top‑5 and move the MAP@5 score closer to the target while keeping the overall workflow unchanged.'
- What this solution (achieved 0.001) has done: 'I add a lightweight, size‑based heuristic that works at the hotel‑level instead of per‑image: for every hotel we compute the median file size of its training images, sort these medians, and then for each test image we select the hotels whose median sizes are closest to the test image size. These candidates are merged with any chain‑aware frequent hotels and the global most‑common list, then ranked by frequency to produce the final top‑5 prediction list. This change keeps the overall workflow unchanged while providing a more informative size‑based signal, which should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.00135) has done: 'I add a lightweight “nearest‑size” heuristic that looks at individual training image sizes (instead of only hotel‑median sizes) to pick a richer candidate set. This keeps the overall pipeline unchanged while providing more informative neighbours, which should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.00315) has done: 'I increased the neighbourhood size used for the size‑based heuristic to a much larger pool (k=200) and also added the median‑size heuristic to the candidate list. For each chain I now pull up to the 10 most frequent hotels (instead of 5). The combined candidate list is then ranked by frequency, and any remaining slots are filled with the global top‑5 hotels. These changes keep the original pipeline but give a richer, more diverse set of possible hotel IDs per image, which should raise the MAP@5 score toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'The change focuses on a clearer, chain‑aware ranking: for each test image we now directly use the five most frequent hotels of its chain (if the chain is known), otherwise we fall back to the global five most common hotels. This removes the noisy large size‑based neighbour pools, keeping the core pipeline intact while providing a stronger, more relevant candidate list, which should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.00209) has done: 'I keep the overall pipeline unchanged but enrich the candidate list used for each test image. After the chain‑aware or global top‑5 fallback, I add size‑based and median‑size‑based nearest‑hotel heuristics (using modest k=20) and then fill any remaining slots with the global most‑common hotels. This adds useful signals while preserving the core logic and should raise MAP@5 toward the target without over‑hauling the solution.'

# 9. Code solution

## === cell 0
import os, glob, pandas as pd
from collections import Counter, defaultdict
import bisect, statistics

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

chain_counter = defaultdict(Counter)  # chain_id -> Counter of hotel_ids
size_hotel_pairs = []  # (size, hotel_id) per image
hotel_sizes = defaultdict(list)  # hotel_id -> list of image sizes

for root, _, files in os.walk(train_img_dir):
    for f in files:
        if not f.lower().endswith(".jpg"):
            continue
        img_path = os.path.join(root, f)
        try:
            sz = os.path.getsize(img_path)
        except OSError:
            continue
        hid = image_to_hotel.get(f)
        if hid is None:
            continue
        size_hotel_pairs.append((sz, hid))
        hotel_sizes[hid].append(sz)

        rel = os.path.relpath(root, train_img_dir)
        chain_id = rel.split(os.sep)[0] if rel != "." else None
        if chain_id is not None:
            chain_counter[chain_id][hid] += 1

chain_top5 = {
    cid: [hid for hid, _ in cnt.most_common(5)] for cid, cnt in chain_counter.items()
}

size_hotel_pairs.sort(key=lambda x: x[0])
sizes_sorted = [p[0] for p in size_hotel_pairs]
hotels_sorted = [p[1] for p in size_hotel_pairs]

hotel_medians = []
for hid, sz_list in hotel_sizes.items():
    if sz_list:
        median_sz = statistics.median(sz_list)
        hotel_medians.append((median_sz, hid))
hotel_medians.sort(key=lambda x: x[0])
median_sizes_sorted = [p[0] for p in hotel_medians]
median_hotels_sorted = [p[1] for p in hotel_medians]


def nearest_hotels_by_median(test_sz, k=200):
    """Return up to k hotel_ids whose *median* training‑image size is closest to ``test_sz``."""
    if not median_sizes_sorted:
        return []
    pos = bisect.bisect_left(median_sizes_sorted, test_sz)
    left, right = pos - 1, pos
    results = []
    while (left >= 0 or right < len(median_sizes_sorted)) and len(results) < k:
        left_diff = abs(median_sizes_sorted[left] - test_sz) if left >= 0 else None
        right_diff = (
            abs(median_sizes_sorted[right] - test_sz)
            if right < len(median_sizes_sorted)
            else None
        )
        if left_diff is not None and (right_diff is None or left_diff <= right_diff):
            hid = median_hotels_sorted[left]
            left -= 1
        else:
            hid = median_hotels_sorted[right]
            right += 1
        results.append(hid)
    return results


def nearest_hotels_by_size(test_sz, k=200):
    """Return up to k hotel_ids whose *individual* training‑image size is closest to ``test_sz``."""
    if not sizes_sorted:
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


test_image_dirs = [
    "/kaggle/input/hotel-id-2021-fgvc8/test_images",
    "/kaggle/input/test_images",
    "/kaggle/input/hotel-id-2021-fgvc8/test_images/",
    "/kaggle/input/test_images/",
]
test_dir = next((d for d in test_image_dirs if os.path.isdir(d)), None)
if test_dir is None:
    raise FileNotFoundError("Test images directory not found.")

test_file_paths = sorted(glob.glob(os.path.join(test_dir, "**/*.jpg"), recursive=True))
test_files = [os.path.basename(p) for p in test_file_paths]
filename_to_path = {os.path.basename(p): p for p in test_file_paths}

predictions = []
for img in test_files:
    if img in image_to_hotel:
        first_candidates = [image_to_hotel[img]]
    else:
        test_path = filename_to_path[img]

        rel_test = os.path.relpath(os.path.dirname(test_path), test_dir)
        chain_id = rel_test.split(os.sep)[0] if rel_test != "." else None
        if chain_id and chain_id in chain_top5:
            first_candidates = chain_top5[chain_id].copy()
        else:
            first_candidates = top5_hotels.copy()

    pred_list = []
    for hid in first_candidates:
        if len(pred_list) >= 5:
            break
        hid_str = str(hid)
        if hid_str not in pred_list:
            pred_list.append(hid_str)

    if len(pred_list) < 5:
        try:
            test_sz = os.path.getsize(filename_to_path[img])
        except OSError:
            test_sz = None
        if test_sz is not None:
            size_neighbors = nearest_hotels_by_size(test_sz, k=20)
            median_neighbors = nearest_hotels_by_median(test_sz, k=20)
            for hid in size_neighbors + median_neighbors:
                if len(pred_list) >= 5:
                    break
                hid_str = str(hid)
                if hid_str not in pred_list:
                    pred_list.append(hid_str)

    for hid in top5_hotels:
        if len(pred_list) >= 5:
            break
        hid_str = str(hid)
        if hid_str not in pred_list:
            pred_list.append(hid_str)

    predictions.append(" ".join(pred_list[:5]))

submission_path = "submission.csv"
submission_df = pd.DataFrame({"image": test_files, "hotel_id": predictions})
submission_df.to_csv(submission_path, index=False)
print(f"Submission created with {len(submission_df)} rows at '{submission_path}'.")


## === cell 1
print("\nFirst 10 rows of submission:")
print(pd.read_csv(submission_path).head(10))
