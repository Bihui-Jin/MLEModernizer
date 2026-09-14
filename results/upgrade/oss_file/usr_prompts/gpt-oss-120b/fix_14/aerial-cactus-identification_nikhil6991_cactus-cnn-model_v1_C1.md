# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.11

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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from concurrent.futures import ThreadPoolExecutor
import multiprocessing  # used to set optimal worker count

try:
    import tensorflow as tf
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input

    USE_TF = True
except Exception as e:
    print("TensorFlow import failed, falling back to sklearn MLP:", e)
    USE_TF = False

np.random.seed(42)
if USE_TF:
    tf.random.set_seed(42)




## === cell 1
label_path = "/kaggle/input/aerial-cactus-identification/train.csv"
label = pd.read_csv(label_path)
label["has_cactus"] = label["has_cactus"].astype(int)

train_dir = "/kaggle/input/aerial-cactus-identification/train/"
test_dir = "/kaggle/input/aerial-cactus-identification/test/"




## === cell 2
train_files = [f for f in os.listdir(train_dir) if f.lower().endswith(".jpg")]
test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
print(f"train images found: {len(train_files)}")
print(f"test images found : {len(test_files)}")




## === cell 3
def load_images(file_list, directory, target_size):
    """Load a list of images into a pre‑allocated uint8 array."""
    n = len(file_list)
    h, w = target_size
    imgs = np.empty((n, h, w, 3), dtype=np.uint8)

    def _load(idx_fname):
        idx, fname = idx_fname
        img_path = os.path.join(directory, fname)
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            im = im.resize(target_size, Image.BILINEAR)
            imgs[idx] = np.array(im, dtype=np.uint8)

    max_workers = multiprocessing.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        exe.map(_load, enumerate(file_list))

    return imgs  # (N, H, W, 3)




## === cell 4
if USE_TF:
    target_sz = (224, 224)
else:
    target_sz = (32, 32)

y = label["has_cactus"].values

if USE_TF:
    vgg = VGG16(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(target_sz[0], target_sz[1], 3),
    )

    def extract_features(file_list, directory, batch_size=512):
        """Stream images through VGG16, returning the stacked feature matrix.
        Uses a single ThreadPoolExecutor for all batches to reduce overhead."""
        feats = []
        h, w = target_sz
        max_workers = multiprocessing.cpu_count()
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            for i in range(0, len(file_list), batch_size):
                batch_files = file_list[i : i + batch_size]
                imgs = np.empty((len(batch_files), h, w, 3), dtype=np.uint8)

                def _load(idx_fname):
                    idx, fname = idx_fname
                    img_path = os.path.join(directory, fname)
                    with Image.open(img_path) as im:
                        im = im.convert("RGB")
                        im = im.resize(target_sz, Image.BILINEAR)
                        imgs[idx] = np.array(im, dtype=np.uint8)

                exe.map(_load, enumerate(batch_files))

                batch_pre = preprocess_input(imgs)
                batch_feat = vgg.predict(
                    batch_pre, batch_size=len(batch_files), verbose=0
                )
                feats.append(batch_feat)
        return np.concatenate(feats, axis=0)

    X_feat = extract_features(train_files, train_dir, batch_size=512)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_feat, y, test_size=0.1, stratify=y, random_state=42
    )
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)

    val_pred = clf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    print(f"Validation AUC (VGG16 + LR): {val_auc:.5f}")
else:
    scaler = StandardScaler()
    X_raw = load_images(train_files, train_dir, target_sz)  # (N, H, W, 3)
    X_flat = X_raw.reshape(len(X_raw), -1)  # (N, 3072)
    X_scaled = scaler.fit_transform(X_flat)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_scaled, y, test_size=0.1, stratify=y, random_state=42
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
        batch_size=256,
    )
    mlp.fit(X_tr, y_tr)

    val_pred = mlp.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    print(f"Validation AUC (MLP on raw pixels): {val_auc:.5f}")




## === cell 5
if USE_TF:
    test_feat = extract_features(test_files, test_dir, batch_size=512)
    test_prob = clf.predict_proba(test_feat)[:, 1]
else:
    test_raw = load_images(test_files, test_dir, target_sz)
    test_flat = test_raw.reshape(len(test_raw), -1)
    test_scaled = scaler.transform(test_flat)
    test_prob = mlp.predict_proba(test_scaled)[:, 1]

submission = pd.DataFrame({"id": test_files, "has_cactus": test_prob})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file saved to {submission_path}")
