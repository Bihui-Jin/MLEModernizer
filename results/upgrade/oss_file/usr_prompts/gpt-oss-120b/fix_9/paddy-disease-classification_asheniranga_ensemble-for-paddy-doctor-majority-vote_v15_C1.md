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

0.9827188940092166

# 6. Current score

0.58109

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.16679) has done: 'The fix removes the broken external model loading and undefined objects, replaces them with a lightweight transfer‑learning pipeline using EfficientNetB0, and ensures the script writes a proper `model_submission_v23.csv` containing the required `image_id` and `label` columns. This resolves all import and name errors, creates functional data generators, trains a quick model, generates predictions for the test set, and produces a valid submission file.'
- What this solution (achieved 0.06341) has done: 'The changes increase data‑loading parallelism and use a larger batch size, which cuts the number of training steps and speeds up each epoch without altering the model architecture, loss, or training schedule. The same ImageDataGenerator logic is kept, and the only added arguments (`workers`, `use_multiprocessing`) only affect how data is read, preserving exact model semantics.'
- What this solution (achieved 0.16833) has done: 'The changes increase data‑loading parallelism and configure TensorFlow’s thread pools, which removes the main CPU bottleneck while keeping the model architecture, training schedule, and augmentation identical. We add explicit `workers` and `use_multiprocessing` arguments to the `flow_from_dataframe` calls and to `model.fit`, set a reproducible seed for the generators, and limit TensorFlow thread usage to the available CPU cores. These adjustments speed up image preprocessing and feeding without altering any core learning logic, preserving the exact model and results.'
- What this solution (achieved 0.59032) has done: 'I remove the TensorFlow imports that cause the protobuf error and replace the deep‑learning pipeline with a lightweight scikit‑learn model that loads and resizes the images using Pillow, encodes the labels, trains a RandomForest classifier, and finally writes a correctly formatted CSV submission. This fixes the runtime failures and, by using a reasonable classifier on resized pixel data, should improve accuracy toward the target while preserving the overall data‑handling flow.'
- What this solution (achieved 0.61414) has done: 'I enhance the feature extraction by adding color histograms, channel means/stds, and increase the RandomForest size to capture more information from the images. These changes keep the overall pipeline (train/val split, RF classifier, CSV output) intact while providing richer inputs, which should raise validation accuracy toward the target score.'
- What this solution (achieved 0.58109) has done: 'The changes add HSV color histograms to the feature vector (giving the model more discriminative color information) and increase the RandomForest capacity (more trees, balanced class weights, and a different max_features setting). These adjustments keep the overall pipeline unchanged while aiming to raise validation accuracy toward the target score.'

# 9. Code solution

## === cell 0
import os
import multiprocessing

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image

num_threads = multiprocessing.cpu_count()




## === cell 1
base_path = "../input/paddy-disease-classification"
train_csv_path = os.path.join(base_path, "train.csv")
test_csv_path = os.path.join(base_path, "sample_submission.csv")
train_images_dir = os.path.join(base_path, "train_images")
test_images_dir = os.path.join(base_path, "test_images")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)




## === cell 2
train_df["filepath"] = train_df.apply(
    lambda row: os.path.join(train_images_dir, row["label"], row["image_id"]), axis=1
)
test_df["filepath"] = test_df["image_id"].apply(
    lambda x: os.path.join(test_images_dir, x)
)

train_df = train_df[train_df["filepath"].apply(os.path.exists)].reset_index(drop=True)
test_df = test_df[test_df["filepath"].apply(os.path.exists)].reset_index(drop=True)

label_list = sorted(train_df["label"].unique())
label_to_idx = {label: idx for idx, label in enumerate(label_list)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}




## === cell 3
train_df_split, val_df_split = train_test_split(
    train_df, test_size=0.1, stratify=train_df["label"], random_state=42
)

IMG_SIZE = (96, 96)  # width, height


def extract_image_features(df):
    """
    Returns a 2‑D numpy array where each row contains:
    - flattened RGB pixels (normalized)
    - 16‑bin histogram for each RGB channel (48 values)
    - 16‑bin histogram for each HSV channel (48 values)  <-- new
    - mean and std for each RGB channel (6 values)
    """
    features = []
    for path in df["filepath"]:
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize to [0,1]

        flat = arr.flatten()

        hist_r, _ = np.histogram(arr[..., 0], bins=16, range=(0.0, 1.0), density=True)
        hist_g, _ = np.histogram(arr[..., 1], bins=16, range=(0.0, 1.0), density=True)
        hist_b, _ = np.histogram(arr[..., 2], bins=16, range=(0.0, 1.0), density=True)

        hsv_img = img.convert("HSV")
        hsv_arr = np.asarray(hsv_img, dtype=np.float32) / 255.0
        hist_h, _ = np.histogram(
            hsv_arr[..., 0], bins=16, range=(0.0, 1.0), density=True
        )
        hist_s, _ = np.histogram(
            hsv_arr[..., 1], bins=16, range=(0.0, 1.0), density=True
        )
        hist_v, _ = np.histogram(
            hsv_arr[..., 2], bins=16, range=(0.0, 1.0), density=True
        )

        mean_r = arr[..., 0].mean()
        std_r = arr[..., 0].std()
        mean_g = arr[..., 1].mean()
        std_g = arr[..., 1].std()
        mean_b = arr[..., 2].mean()
        std_b = arr[..., 2].std()
        stats = np.array(
            [mean_r, std_r, mean_g, std_g, mean_b, std_b], dtype=np.float32
        )

        feat_vec = np.concatenate(
            [flat, hist_r, hist_g, hist_b, hist_h, hist_s, hist_v, stats]
        )
        features.append(feat_vec)
    return np.stack(features)


X_train = extract_image_features(train_df_split)
y_train = train_df_split["label"].map(label_to_idx).values

X_val = extract_image_features(val_df_split)
y_val = val_df_split["label"].map(label_to_idx).values




## === cell 4
rf_clf = RandomForestClassifier(
    n_estimators=1000,  # increased number of trees
    max_depth=None,
    n_jobs=num_threads,
    random_state=42,
    max_features="log2",  # alternative feature subset size
    class_weight="balanced",  # help with any class imbalance
)

rf_clf.fit(X_train, y_train)

val_pred = rf_clf.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation accuracy: {val_acc:.5f}")




## === cell 5
X_test = extract_image_features(test_df)
test_pred_idx = rf_clf.predict(X_test)
test_pred_labels = [idx_to_label[idx] for idx in test_pred_idx]

submission = pd.DataFrame({"image_id": test_df["image_id"], "label": test_pred_labels})
submission_path = "model_submission_v23.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path}")
