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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.976958525345622

# 6. Current score

0.79285

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.17487) has done: 'I replace the failing ensemble code with a reliable fallback that always produces a valid submission CSV. The new script reads the training labels, picks the most frequent class, lists all test image filenames, assigns that class to every test sample, and writes the result to `model_submission_v10.csv`. This removes the missing‑file errors and guarantees the required `.csv` output while keeping the core logic minimal and deterministic.'
- What this solution (achieved 0.19485) has done: 'I replace the constant‑label fallback with a tiny, data‑driven model that uses only the file‑size of each image (readily available without extra libraries) to predict the disease class. By training a very shallow DecisionTree on the training image sizes we can capture any simple size‑based patterns, which should modestly raise accuracy toward the target while keeping the original pipeline and output format intact. The script still writes `model_submission_v10.csv` and respects the sample‑submission ordering.'
- What this solution (achieved 0.4731) has done: 'I enhance the tiny model by extracting simple visual features (image dimensions, file size, and average RGB values) using Pillow, which are cheap to compute but far more informative than file size alone. These six numeric features train a modest RandomForest classifier, improving prediction accuracy while preserving the overall pipeline and output format.'
- What this solution (achieved 0.79285) has done: 'I added a few inexpensive visual descriptors (aspect ratio, per‑channel standard deviations) to the feature vector and let the RandomForest use more trees and no depth limit, which usually raises classification accuracy without changing the overall pipeline. The feature extraction is updated in both the training and test loops, keeping the same model class and output format.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
from collections import Counter



## === cell 1
train_path = os.path.abspath(
    os.path.join("..", "input", "paddy-disease-classification", "train.csv")
)
train_df = pd.read_csv(train_path)

most_common_label = Counter(train_df["label"]).most_common(1)[0][0]
print(f"Most common label in training set: {most_common_label}")



## === cell 2
train_images_dir = os.path.abspath(
    os.path.join("..", "input", "paddy-disease-classification", "train_images")
)

train_features = []
train_labels = []

try:
    from PIL import Image
    import numpy as np

    pillow_available = True
except Exception:
    pillow_available = False
    print("Pillow not available; will fall back to file‑size only model.")

for label_name in os.listdir(train_images_dir):
    label_dir = os.path.join(train_images_dir, label_name)
    if not os.path.isdir(label_dir):
        continue
    for fname in os.listdir(label_dir):
        if not (
            fname.lower().endswith(".jpg")
            or fname.lower().endswith(".jpeg")
            or fname.lower().endswith(".png")
        ):
            continue
        fpath = os.path.join(label_dir, fname)
        try:
            filesize = os.path.getsize(fpath)
        except OSError:
            continue

        if pillow_available:
            try:
                with Image.open(fpath) as img:
                    img = img.convert("RGB")
                    width, height = img.size
                    aspect = width / height if height != 0 else 0
                    img_arr = np.array(img).astype(np.float32)
                    mean_r = img_arr[:, :, 0].mean()
                    mean_g = img_arr[:, :, 1].mean()
                    mean_b = img_arr[:, :, 2].mean()
                    std_r = img_arr[:, :, 0].std()
                    std_g = img_arr[:, :, 1].std()
                    std_b = img_arr[:, :, 2].std()
                feature_vec = [
                    filesize,
                    width,
                    height,
                    aspect,
                    mean_r,
                    mean_g,
                    mean_b,
                    std_r,
                    std_g,
                    std_b,
                ]
            except Exception:
                feature_vec = [filesize, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            feature_vec = [filesize]

        train_features.append(feature_vec)
        train_labels.append(label_name)

if not train_features:
    print(
        "No training features collected; using most‑common label for all predictions."
    )
    model = None
else:
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=1
    )
    clf.fit(train_features, train_labels)
    model = clf
    print(
        f"Trained RandomForest model on {len(train_features)} images with {len(set(train_labels))} classes."
    )



## === cell 3
test_images_dir = os.path.abspath(
    os.path.join("..", "input", "paddy-disease-classification", "test_images")
)
test_image_files = [
    f
    for f in os.listdir(test_images_dir)
    if f.lower().endswith(".jpg")
    or f.lower().endswith(".jpeg")
    or f.lower().endswith(".png")
]
test_image_ids = [os.path.basename(f) for f in test_image_files]

if model is not None:
    test_features = []
    for f in test_image_files:
        fpath = os.path.join(test_images_dir, f)
        try:
            filesize = os.path.getsize(fpath)
        except OSError:
            filesize = 0

        if pillow_available:
            try:
                with Image.open(fpath) as img:
                    img = img.convert("RGB")
                    width, height = img.size
                    aspect = width / height if height != 0 else 0
                    img_arr = np.array(img).astype(np.float32)
                    mean_r = img_arr[:, :, 0].mean()
                    mean_g = img_arr[:, :, 1].mean()
                    mean_b = img_arr[:, :, 2].mean()
                    std_r = img_arr[:, :, 0].std()
                    std_g = img_arr[:, :, 1].std()
                    std_b = img_arr[:, :, 2].std()
                feature_vec = [
                    filesize,
                    width,
                    height,
                    aspect,
                    mean_r,
                    mean_g,
                    mean_b,
                    std_r,
                    std_g,
                    std_b,
                ]
            except Exception:
                feature_vec = [filesize, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            feature_vec = [filesize]

        test_features.append(feature_vec)

    predicted_labels = model.predict(test_features)
else:
    predicted_labels = [most_common_label] * len(test_image_files)

submission_df = pd.DataFrame({"image_id": test_image_ids, "label": predicted_labels})

sample_sub_path = os.path.abspath(
    os.path.join("..", "input", "paddy-disease-classification", "sample_submission.csv")
)
if os.path.exists(sample_sub_path):
    sample_sub = pd.read_csv(sample_sub_path)
    submission_df = (
        submission_df.set_index("image_id")
        .reindex(sample_sub["image_id"])
        .fillna(most_common_label)
        .reset_index()
    )



## === cell 4
output_path = "model_submission_v10.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path} with {len(submission_df)} rows.")
