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

0.8144

# 6. Current score

0.94534

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.97438) has done: 'I keep the overall workflow unchanged but fix two issues that prevented a proper Kaggle submission and nudge the validation AUC a bit closer to the target. First, I increase the number of trees in the RandomForest (from 500 to 1000) which usually yields a modest AUC boost without altering the model type. Second, I write the submission file directly to the Kaggle working directory (`/kaggle/working/submission.csv`) so the platform can detect it. Minor safety checks are added for locating test images, and the cell numbering is renumbered to start at 1 as required.'
- What this solution (achieved 0.96567) has done: 'The current validation AUC (0.974) is well above the target, so we purposely make the model a bit less powerful to bring the score down into the acceptable range. I reduced the RandomForest size and limited its depth (200 trees, max_depth 10, higher min_samples_split) which typically lowers AUC without changing any core logic. No other parts of the pipeline are altered, and the script still creates a proper `submission.csv` in the Kaggle working directory.'
- What this solution (achieved 0.94534) has done: 'I keep the overall workflow unchanged and only make the RandomForest less powerful so its validation AUC drops toward the target (≈0.814). By lowering the number of trees, reducing max depth, and requiring more samples per split/leaf, the model’s discriminative ability diminishes, moving the score into the acceptable range while still using the same architecture and producing a proper `submission.csv`.'

# 9. Code solution

## === cell 0
import os, shutil, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

print("Python version OK")



## === cell 1
INPUT_BASE = "/kaggle/input/aerial-cactus-identification"
WORK_DIR = "/kaggle/working/aerial-cactus-identification"

os.makedirs(WORK_DIR, exist_ok=True)


def safe_copy(src, dst_dir):
    """Copy src to dst_dir unless they refer to the same file."""
    dst = os.path.join(dst_dir, os.path.basename(src))
    try:
        if os.path.samefile(src, dst):
            return
    except FileNotFoundError:
        pass
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy(src, dst_dir)


safe_copy(os.path.join(INPUT_BASE, "train.csv"), WORK_DIR)
safe_copy(os.path.join(INPUT_BASE, "sample_submission.csv"), WORK_DIR)

subprocess.run(
    ["unzip", "-q", "-o", os.path.join(INPUT_BASE, "train.zip"), "-d", WORK_DIR],
    check=True,
)
subprocess.run(
    ["unzip", "-q", "-o", os.path.join(INPUT_BASE, "test.zip"), "-d", WORK_DIR],
    check=True,
)

TRAIN_DIR = os.path.join(WORK_DIR, "train")
TEST_DIR = os.path.join(WORK_DIR, "test")

if not any(fname.lower().endswith(".jpg") for fname in os.listdir(TEST_DIR)):
    possible_subdirs = [
        os.path.join(TEST_DIR, d)
        for d in os.listdir(TEST_DIR)
        if os.path.isdir(os.path.join(TEST_DIR, d))
    ]
    if possible_subdirs:
        TEST_DIR = possible_subdirs[0]



## === cell 2
df = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
print("Training samples:", df.shape[0])
df.has_cactus.value_counts().plot.bar()
plt.close()  # avoid hanging in non‑interactive environments



## === cell 3
train_df, val_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df.has_cactus
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)


def load_images(ids, base_dir):
    """Load images given a list/Series of filenames, return (N, 32*32*3) array."""
    images = []
    for img_id in ids:
        path = os.path.join(base_dir, img_id)
        with Image.open(path) as im:
            im = im.convert("RGB")
            im = im.resize((32, 32))
            img_arr = np.asarray(im, dtype=np.float32) / 255.0  # normalize
            images.append(img_arr.reshape(-1))
    return np.stack(images)


X_train = load_images(train_df["id"], TRAIN_DIR)
y_train = train_df["has_cactus"].values.astype(np.float32)

X_val = load_images(val_df["id"], TRAIN_DIR)
y_val = val_df["has_cactus"].values.astype(np.float32)

print("Loaded images – train:", X_train.shape, "val:", X_val.shape)



## === cell 4
model = RandomForestClassifier(
    n_estimators=100,  # fewer trees
    max_depth=5,  # shallower depth
    min_samples_split=15,  # need more samples to split
    min_samples_leaf=4,  # more samples per leaf
    max_features="sqrt",  # limit features considered at each split
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

model.fit(X_train, y_train)
val_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUC: {auc:.5f}")



## === cell 5
test_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")]
test_files.sort()
test_df = pd.DataFrame({"id": test_files})

X_test = load_images(test_df["id"], TEST_DIR)
test_pred = model.predict_proba(X_test)[:, 1]
test_df["has_cactus"] = test_pred

submission_path = os.path.join("/kaggle/working", "submission.csv")
test_df.to_csv(submission_path, index=False)
print("Submission saved to:", submission_path)



## === cell 6
submission = pd.read_csv(submission_path)
print(submission.head())
print("Columns:", submission.columns.tolist())
