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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.8928

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix corrects the channel statistics calculation (removing mismatched dimensions), ensures features are built correctly, and creates predictions that align exactly with the expected test IDs by loading only existing images and filling missing ones with a neutral probability. This resolves the earlier shape errors and the submission‑row mismatch, allowing the script to run end‑to‑end and produce a valid `submission.csv` that should achieve a validation AUC close to the target.'
- What this solution (achieved 0.5) has done: 'I enhance the feature extraction by adding per‑channel min/max and overall min/max statistics, giving the model richer information while keeping the overall pipeline unchanged. I also modestly increase the number of trees in the RandomForest to improve its stability. These small, targeted tweaks should raise the validation AUC from the current ~0.5 toward the target 0.8928 without altering the core modeling approach.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

BASE_INPUT = "/kaggle/input/aerial-cactus-identification"
WORK_DIR = "/kaggle/working"
os.makedirs(WORK_DIR, exist_ok=True)

TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test")
nested_test = os.path.join(TEST_IMG_DIR, "test")
if os.path.isdir(nested_test):
    TEST_IMG_DIR = nested_test

TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
SAMPLE_SUB = os.path.join(BASE_INPUT, "sample_submission.csv")

print("Data directories:", TRAIN_IMG_DIR, TEST_IMG_DIR)
print("TensorFlow is not used in this script.")




## === cell 1
df = pd.read_csv(TRAIN_CSV)
print("Dataset size:", df.shape)
df.sample(3)

df.has_cactus.value_counts().plot.bar()
plt.title("Label distribution")
plt.show()




## === cell 2
def load_images(img_dir, ids):
    """Load images given a directory and a list of filenames."""
    if len(ids) == 0:
        return np.empty((0, 32, 32, 3), dtype=np.float32)
    images = []
    for fname in ids:
        path = os.path.join(img_dir, fname)
        with Image.open(path) as img:
            img = img.convert("RGB").resize((32, 32))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            images.append(arr)
    return np.stack(images, axis=0)


train_ids = df["id"].values
X = load_images(TRAIN_IMG_DIR, train_ids)
y = df["has_cactus"].values.astype(np.float32)




## === cell 3
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train_flat = X_train.reshape(len(X_train), -1)
X_valid_flat = X_valid.reshape(len(X_valid), -1)


def channel_stats(arr):
    """
    Compute per‑channel mean, std, min, max (12 values) plus overall
    mean, std, min, max (4 values) for each image in `arr`
    (shape N × 32 × 32 × 3). Returns an (N, 16) array.
    """
    mean_chan = arr.mean(axis=(1, 2))  # (N, 3)
    std_chan = arr.std(axis=(1, 2))  # (N, 3)
    min_chan = arr.min(axis=(1, 2))  # (N, 3)
    max_chan = arr.max(axis=(1, 2))  # (N, 3)
    mean_all = arr.mean(axis=(1, 2, 3)).reshape(-1, 1)  # (N, 1)
    std_all = arr.std(axis=(1, 2, 3)).reshape(-1, 1)  # (N, 1)
    min_all = arr.min(axis=(1, 2, 3)).reshape(-1, 1)  # (N, 1)
    max_all = arr.max(axis=(1, 2, 3)).reshape(-1, 1)  # (N, 1)
    return np.concatenate(
        [mean_chan, std_chan, min_chan, max_chan, mean_all, std_all, min_all, max_all],
        axis=1,
    )  # (N, 16)


train_stats = channel_stats(X_train)
valid_stats = channel_stats(X_valid)

X_train_feat = np.concatenate([X_train_flat, train_stats], axis=1)
X_valid_feat = np.concatenate([X_valid_flat, valid_stats], axis=1)




## === cell 4
rf = RandomForestClassifier(
    n_estimators=2000,  # increased from 1200
    max_depth=None,
    max_features="sqrt",
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)

rf.fit(X_train_feat, y_train)

valid_pred = rf.predict_proba(X_valid_feat)[:, 1]
auc = roc_auc_score(y_valid, valid_pred)
print(f"Validation AUC: {auc:.5f}")




## === cell 5
sample_sub = pd.read_csv(SAMPLE_SUB)
expected_ids = sample_sub["id"].tolist()

existing_ids = [
    i for i in expected_ids if os.path.isfile(os.path.join(TEST_IMG_DIR, i))
]

X_test = load_images(TEST_IMG_DIR, existing_ids)

if X_test.shape[0] > 0:
    X_test_flat = X_test.reshape(len(X_test), -1)
    test_stats = channel_stats(X_test)
    X_test_feat = np.concatenate([X_test_flat, test_stats], axis=1)
    preds_existing = rf.predict_proba(X_test_feat)[:, 1]
else:
    preds_existing = np.array([])

pred_dict = dict(zip(existing_ids, preds_existing))
default_prob = 0.5
test_pred = np.array(
    [pred_dict.get(i, default_prob) for i in expected_ids], dtype=np.float32
)

test_df = pd.DataFrame({"id": expected_ids, "has_cactus": test_pred})




## === cell 6
submission_path = os.path.join(WORK_DIR, "submission.csv")
test_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
