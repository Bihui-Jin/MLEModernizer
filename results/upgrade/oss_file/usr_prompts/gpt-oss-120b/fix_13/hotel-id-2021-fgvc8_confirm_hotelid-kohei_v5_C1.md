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
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import concurrent.futures


def _avg_rgb(image_path):
    """Return the mean R,G,B values of an image as a length‑3 np.array.
    Returns None if the image cannot be opened."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            arr = np.array(img)
            return arr.mean(axis=(0, 1))
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
