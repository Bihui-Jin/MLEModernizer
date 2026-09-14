# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.12

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from PIL import Image
import concurrent.futures  # used for parallel image loading

np.random.seed(42)

DATA_ROOT = Path("/kaggle/input/plant-seedlings-classification")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"

IMG_SIZE = (128, 128)  # larger size for better feature representation
RANDOM_STATE = 42
BATCH_SIZE = 32  # kept for compatibility; not used directly




## === cell 1
def load_images_from_folder(folder_path: Path, label_encoder: LabelEncoder = None):
    """
    Efficiently load, resize, flatten images and optionally encode labels.
    Returns:
        X: np.ndarray shape (n_samples, IMG_SIZE[0]*IMG_SIZE[1]*3), dtype float32 in [0,1]
        y: np.ndarray of integer encoded labels (or None)
        raw_labels: list of original class names
    """
    img_paths = []
    class_names = []
    for class_dir in sorted(folder_path.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for img_path in sorted(class_dir.glob("*.png")):
            img_paths.append(img_path)
            class_names.append(class_name)

    n_images = len(img_paths)
    feature_len = IMG_SIZE[0] * IMG_SIZE[1] * 3
    X = np.empty((n_images, feature_len), dtype=np.float32)

    max_workers = min(8, os.cpu_count() or 1)  # avoid oversubscription

    def load_one(idx_path):
        idx, path = idx_path
        img = Image.open(path).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.asarray(img, dtype=np.float32).flatten() / 255.0
        return idx, arr

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, arr in executor.map(load_one, enumerate(img_paths)):
            X[idx] = arr

    raw_labels = class_names  # keep original order

    if label_encoder is not None:
        y = label_encoder.transform(np.array(class_names))
    else:
        y = None
    return X, y, raw_labels


class_names = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
le = LabelEncoder()
le.fit(class_names)

X_train_full, y_train_full, _ = load_images_from_folder(TRAIN_DIR, label_encoder=le)

print(
    f"Loaded {X_train_full.shape[0]} training images with shape {X_train_full.shape[1:]}"
)




## === cell 2
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)

scaler = StandardScaler(copy=False)
X_tr = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)

model = LogisticRegression(
    multi_class="multinomial",
    solver="saga",  # fast on high‑dimensional dense data
    max_iter=1000,  # unchanged convergence goal
    C=10.0,
    n_jobs=-1,  # use all cores to speed up training while preserving algorithm
    random_state=RANDOM_STATE,
    class_weight="balanced",
)

print("Training logistic regression model...")
model.fit(X_tr, y_tr)

val_pred = model.predict(X_val)
val_f1 = f1_score(y_val, val_pred, average="micro")
print(f"Local validation micro‑F1: {val_f1:.4f}")




## === cell 3
def load_test_images(test_dir: Path):
    img_paths = sorted(test_dir.rglob("*.png"))
    n_images = len(img_paths)
    feature_len = IMG_SIZE[0] * IMG_SIZE[1] * 3
    X_test = np.empty((n_images, feature_len), dtype=np.float32)
    filenames = [p.name for p in img_paths]

    max_workers = min(8, os.cpu_count() or 1)

    def load_one(idx_path):
        idx, path = idx_path
        img = Image.open(path).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.asarray(img, dtype=np.float32).flatten() / 255.0
        return idx, arr

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, arr in executor.map(load_one, enumerate(img_paths)):
            X_test[idx] = arr

    return X_test, filenames


X_test, test_filenames = load_test_images(TEST_DIR)
print(f"Loaded {X_test.shape[0]} test images.")

X_test = scaler.transform(X_test)

test_probs = model.predict_proba(X_test)
test_pred_idx = np.argmax(test_probs, axis=1)
test_pred_species = le.inverse_transform(test_pred_idx)

submission = pd.DataFrame({"file": test_filenames, "species": test_pred_species})

sample_sub = pd.read_csv(SAMPLE_SUB)
submission = submission.set_index("file").reindex(sample_sub["file"]).reset_index()

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")




## === cell 4
print("All done.")
