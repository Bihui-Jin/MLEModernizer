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

0.7116317251831245

# 6. Current score

0.0017

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I remove the stray plain‑text and markdown fences that cause syntax errors, comment out the unnecessary pip‑install steps, and simplify the workflow to always generate a valid baseline submission (the five most frequent hotels) directly. I also make the image search recursive so it finds all test JPGs, and replace the notebook‑only `display` call with a simple `print` of the head rows.'
- What this solution (achieved 0.00209) has done: 'I replace the simple “always predict the global top‑5 hotels” baseline with a chain‑aware heuristic: for each test image I infer its chain from the first folder in its path and use the five most frequent hotel IDs that appear for that chain in the training data (falling back to the global top‑5 when the chain is missing). This small change respects the original workflow, adds no external packages, and is expected to raise MAP@5 toward the target score.'
- What this solution (achieved 0.00209) has done: 'I adjust the submission generation to output only the image filename (as required by the competition) instead of the full relative path, and sort the rows alphabetically before saving. This fixes the column mismatch that caused the near‑zero MAP@5 and should move the score much closer to the target while keeping the original heuristic unchanged.'
- What this solution (achieved 0.00173) has done: 'The changes introduce a faster RGB‑mean calculation using Pillow’s ImageStat (avoiding the costly NumPy conversion) and parallelise the per‑image processing for both training and test sets with a thread pool. The default top‑5 popular hotels are pre‑computed once instead of on every failure, and all other logic—including centroid aggregation and distance ranking—remains unchanged, preserving exact predictions while cutting the total runtime well below the 600‑second limit.'
- What this solution (achieved 0.00209) has done: 'I added a cheap chain‑aware heuristic: for each test image we infer its hotel‑chain from the parent folder name and predict the five most frequent hotels seen for that chain in the training data (falling back to the global top‑5 when the chain is unseen). This change keeps the original workflow and avoids heavy modelling while moving the MAP@5 score much closer to the target. The rest of the script – loading data, handling missing files, and writing the submission CSV – remains unchanged.'
- What this solution (achieved 0.00176) has done: 'I replace the simple chain‑based top‑5 heuristic with a cheap image‑similarity approach: compute the average RGB colour of every training image, aggregate them per hotel to obtain a hotel‑level colour centroid, then for each test image pick the five hotels whose centroids are closest in Euclidean RGB space. This keeps the overall workflow unchanged, adds only a lightweight nearest‑neighbour step, and is expected to raise MAP@5 toward the target while still outputting a valid `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'I replace the costly RGB‑centroid nearest‑neighbour logic with the inexpensive chain‑aware heuristic that predicts, for each test image, the five most frequent hotels seen for the same chain in the training data (falling back to the global top‑5). This change removes the heavy image‑averaging step, speeds up execution, and is expected to raise MAP@5 substantially toward the target while keeping the overall workflow unchanged.'
- What this solution (achieved 0.00209) has done: 'I keep the original chain‑aware heuristic but make the predictions more robust: when a chain’s list of popular hotels has fewer than five entries (or the chain is missing), I fill the remaining slots with the global most‑frequent hotels that aren’t already in the list. This small change preserves the core logic while improving coverage, which should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.0017) has done: 'The fix replaces the slow NumPy‑based RGB averaging with Pillow’s lightweight `ImageStat` (which avoids constructing a full image array) and limits the thread pool size to a sensible default, cutting I/O and computation overhead while preserving the exact centroid logic and prediction flow. All other steps—including centroid aggregation, distance computation, and fallback handling—remain unchanged, so the results are identical apart from negligible floating‑point round‑off.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageStat
import concurrent.futures


def _avg_rgb(image_path):
    """Return the mean R,G,B values of an image as a length‑3 np.array.
    Uses Pillow's ImageStat to avoid creating a full NumPy array, which is faster.
    Returns None if the image cannot be opened."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            stat = ImageStat.Stat(img)
            return np.array(stat.mean)
    except Exception:
        return None


def _build_hotel_centroids(train_df, train_images_root, max_workers=None):
    """Compute colour centroids for each hotel in parallel."""
    tasks = []
    for _, row in train_df.iterrows():
        img_path = train_images_root / str(row["chain"]) / row["image"]
        if img_path.is_file():
            tasks.append((str(row["hotel_id"]), img_path))

    centroids = {}
    counts = {}
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda t: _avg_rgb(t[1]), tasks))

    for (hotel_id, _), mean_rgb in zip(tasks, results):
        if mean_rgb is None:
            continue
        if hotel_id in centroids:
            centroids[hotel_id] += mean_rgb
            counts[hotel_id] += 1
        else:
            centroids[hotel_id] = mean_rgb.copy()
            counts[hotel_id] = 1

    for hid in centroids:
        centroids[hid] /= counts[hid]
    return centroids


def generate_submission():
    """Create submission using colour‑centroid NN with parallel processing."""
    possible_train_paths = [
        Path("/kaggle/input/hotel-id-2021-fgvc8/train.csv"),
        Path("/kaggle/input/train.csv"),
    ]
    train_path = next((p for p in possible_train_paths if p.is_file()), None)
    if train_path is None:
        raise FileNotFoundError("train.csv not found in expected locations.")
    train_df = pd.read_csv(train_path)

    global_top5 = (
        train_df["hotel_id"].value_counts().nlargest(5).index.astype(str).tolist()
    )

    chain_top5 = {}
    for chain_val, group in train_df.groupby("chain"):
        top5 = group["hotel_id"].value_counts().nlargest(5).index.astype(str).tolist()
        chain_top5[str(chain_val)] = top5

    possible_train_images_roots = [
        Path("/kaggle/input/hotel-id-2021-fgvc8/train_images"),
        Path("/kaggle/input/train_images"),
    ]
    train_images_root = next(
        (p for p in possible_train_images_roots if p.is_dir()), None
    )
    if train_images_root is None:
        raise FileNotFoundError("train_images directory not found.")
    hotel_centroids = _build_hotel_centroids(train_df, train_images_root)

    hotel_ids = list(hotel_centroids.keys())
    centroid_matrix = np.stack([hotel_centroids[hid] for hid in hotel_ids])

    possible_test_dirs = [
        Path("/kaggle/input/hotel-id-2021-fgvc8/test_images"),
        Path("/kaggle/input/test_images"),
    ]
    test_dir = next((d for d in possible_test_dirs if d.is_dir()), None)
    if test_dir is None:
        raise FileNotFoundError(
            "test_images directory not found in expected locations."
        )
    test_image_paths = sorted(list(test_dir.rglob("*.jpg")), key=lambda p: p.name)
    if not test_image_paths:
        raise FileNotFoundError("No test images found in the test_images directory.")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        test_rgbs = list(executor.map(_avg_rgb, test_image_paths))

    rows = []
    for path, test_rgb in zip(test_image_paths, test_rgbs):
        if test_rgb is not None:
            dists = np.linalg.norm(centroid_matrix - test_rgb, axis=1)
            nearest_idxs = np.argpartition(dists, 5)[:5]
            nearest_idxs = nearest_idxs[np.argsort(dists[nearest_idxs])]
            pred_ids = [hotel_ids[i] for i in nearest_idxs]
        else:
            pred_ids = None  # trigger fallback

        if not pred_ids or len(pred_ids) < 5:
            chain_name = path.parent.name
            pred_ids = chain_top5.get(chain_name, None)
            if pred_ids is None:
                pred_ids = global_top5.copy()
            elif len(pred_ids) < 5:
                for gid in global_top5:
                    if gid not in pred_ids:
                        pred_ids.append(gid)
                    if len(pred_ids) == 5:
                        break
            pred_ids = pred_ids[:5]

        rows.append({"image": path.name, "hotel_id": " ".join(pred_ids)})

    rows.sort(key=lambda x: x["image"])
    sub_df = pd.DataFrame(rows)
    sub_path = Path("/kaggle/working/submission.csv")
    sub_df.to_csv(sub_path, index=False)
    return sub_path




## === cell 1
submission_path = generate_submission()
print(f"Submission file created at: {submission_path}")
print(pd.read_csv(submission_path).head())
