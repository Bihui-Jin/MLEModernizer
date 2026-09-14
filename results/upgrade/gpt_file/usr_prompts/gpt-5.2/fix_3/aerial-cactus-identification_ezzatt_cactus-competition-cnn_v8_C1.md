# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.13

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

0.9798333333333332

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
from zipfile import ZipFile

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

SEED = 42
rng = np.random.default_rng(SEED)
np.random.seed(SEED)



## === cell 1
path = "/kaggle/input/aerial-cactus-identification/"
train_labels = pd.read_csv(os.path.join(path, "train.csv"))
sample_sub = pd.read_csv(os.path.join(path, "sample_submission.csv"))

train_labels.head()



## === cell 2
class_names = ["Has cactus", "Hasn't cactus"]
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
nb_classes = len(class_names)

class_names_label, nb_classes



## === cell 3
train_labels.info()



## === cell 4
train_labels.id.shape



## === cell 5
train_labels.size



## === cell 6
extract_root = "/kaggle/working/aerial_cactus_data"
os.makedirs(extract_root, exist_ok=True)

train_zip = os.path.join(path, "train.zip")
test_zip = os.path.join(path, "test.zip")

with ZipFile(train_zip) as zipper:
    zipper.extractall(path=extract_root)

with ZipFile(test_zip) as zipper:
    zipper.extractall(path=extract_root)


def find_image_dir(root: str, split: str) -> str:
    candidates = [
        os.path.join(root, split),
        os.path.join(root, split, split),
    ]
    for c in candidates:
        if os.path.isdir(c) and len(glob.glob(os.path.join(c, "*.jpg"))) > 0:
            return c
    for c in glob.glob(os.path.join(root, "**", split), recursive=True):
        if os.path.isdir(c) and len(glob.glob(os.path.join(c, "*.jpg"))) > 0:
            return c
    return candidates[0]


train_path = find_image_dir(extract_root, "train")
test_path = find_image_dir(extract_root, "test")

print(
    "train_path:",
    train_path,
    "exists:",
    os.path.isdir(train_path),
    "n_files:",
    len(glob.glob(os.path.join(train_path, "*.jpg"))),
)
print(
    "test_path :",
    test_path,
    "exists:",
    os.path.isdir(test_path),
    "n_files:",
    len(glob.glob(os.path.join(test_path, "*.jpg"))),
)




## === cell 7
def load_data(train_df: pd.DataFrame, train_dir: str):
    x = np.empty((len(train_df), 32, 32, 3), dtype=np.float32)
    y = train_df["has_cactus"].to_numpy(dtype=np.int32)

    missing = 0
    for idx, img_id in enumerate(train_df["id"].values):
        img_path = os.path.join(train_dir, img_id)
        if not os.path.isfile(img_path):
            missing += 1
            continue
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.float32)
        x[idx] = arr

    if missing:
        print(f"WARNING: {missing} train images were missing under {train_dir}")
    return x, y




## === cell 8
x_all, y_all = load_data(train_labels, train_path)
print("Loaded:", x_all.shape, y_all.shape, "dtype:", x_all.dtype, y_all.dtype)




## === cell 9
def display_examples(class_names, images, labels):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("plots of a sample of the data", fontsize=10)
    n = min(20, len(images))
    for i in range(n):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i].astype(np.uint8))
        plt.xlabel(class_names[int(labels[i])])
    plt.show()


display_examples(class_names, x_all, y_all)



## === cell 10
unique_labels, train_counts = np.unique(y_all, return_counts=True)
print(f"{unique_labels[0]}: {train_counts[0]}\n{unique_labels[1]}: {train_counts[1]}")



## === cell 11
plt.figure(figsize=(4, 4))
plt.bar(unique_labels, train_counts, color="violet", edgecolor="black")
plt.xlabel("Class Labels")
plt.ylabel("Number of Samples")
plt.title("Training Set Class Distribution")
plt.xticks(unique_labels)
plt.grid(axis="y", linestyle="", alpha=0.4)
plt.tight_layout()
plt.show()



## === cell 12
x_all = x_all / 255.0


def stratified_split(x, y, test_size=0.25, seed=SEED):
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng_local = np.random.default_rng(seed)
    rng_local.shuffle(idx0)
    rng_local.shuffle(idx1)

    n0_val = int(round(len(idx0) * test_size))
    n1_val = int(round(len(idx1) * test_size))

    val_idx = np.concatenate([idx0[:n0_val], idx1[:n1_val]])
    train_idx = np.concatenate([idx0[n0_val:], idx1[n1_val:]])

    rng_local.shuffle(train_idx)
    rng_local.shuffle(val_idx)

    return x[train_idx], x[val_idx], y[train_idx], y[val_idx]


x_train, x_val, y_train, y_val = stratified_split(
    x_all, y_all, test_size=0.25, seed=SEED
)
print("train:", x_train.shape, y_train.shape)
print("val  :", x_val.shape, y_val.shape)



## === cell 13


def make_features(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    return x.reshape(n, -1).astype(np.float32)


Xtr = make_features(x_train)
Xva = make_features(x_val)

mu = Xtr.mean(axis=0, keepdims=True)
sigma = Xtr.std(axis=0, keepdims=True) + 1e-6
Xtr_s = (Xtr - mu) / sigma
Xva_s = (Xva - mu) / sigma

Xtr_b = np.concatenate([Xtr_s, np.ones((Xtr_s.shape[0], 1), dtype=np.float32)], axis=1)
Xva_b = np.concatenate([Xva_s, np.ones((Xva_s.shape[0], 1), dtype=np.float32)], axis=1)

ytr = y_train.astype(np.float32)
yva = y_val.astype(np.float32)


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def train_logreg(X, y, lr=0.05, epochs=200, l2=1e-3, seed=SEED):
    rng_local = np.random.default_rng(seed)
    w = rng_local.normal(0, 0.01, size=(X.shape[1],)).astype(np.float32)

    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    w_pos = (n_pos + n_neg) / (2.0 * n_pos + 1e-12)
    w_neg = (n_pos + n_neg) / (2.0 * n_neg + 1e-12)

    history = {"loss": [], "val_loss": []}

    for _ in range(epochs):
        p = sigmoid(X @ w)
        weights = np.where(y > 0.5, w_pos, w_neg).astype(np.float32)
        eps = 1e-7
        loss = -np.mean(
            weights * (y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
        )
        loss += 0.5 * l2 * float(np.sum(w[:-1] * w[:-1]))

        grad = (X.T @ (weights * (p - y))) / X.shape[0]
        grad[:-1] += l2 * w[:-1]
        w -= lr * grad.astype(np.float32)

        history["loss"].append(float(loss))
    return w, history


w, history = train_logreg(Xtr_b, ytr, lr=0.05, epochs=250, l2=1e-3, seed=SEED)

p_val = sigmoid(Xva_b @ w)
val_acc = float(((p_val > 0.5).astype(np.int32) == y_val).mean())
print("Validation accuracy:", val_acc)



## === cell 14
plt.plot(history["loss"], label="Training logloss")
plt.xlabel("Epoch")
plt.ylabel("Logloss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.grid()
plt.show()




## === cell 15
def classification_report_simple(y_true, y_pred):
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
    }


preds_labels = (p_val > 0.5).astype(int).flatten()
rep = classification_report_simple(y_val, preds_labels)
print(rep)

conf_matrix = np.array([[rep["tn"], rep["fp"]], [rep["fn"], rep["tp"]]], dtype=int)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



## === cell 16
test_images = glob.glob(os.path.join(test_path, "*.jpg"))
print("n_test_images:", len(test_images))
assert len(test_images) > 0, f"No test images found in {test_path}"



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/1166024870.py in <cell line: 0>()
      1 test_images = glob.glob(os.path.join(test_path, "*.jpg"))
      2 print("n_test_images:", len(test_images))
----> 3 assert len(test_images) > 0, f"No test images found in {test_path}"
      4 

AssertionError: No test images found in /kaggle/working/aerial_cactus_data/test

## === cell 17
x_test = np.empty((len(test_images), 32, 32, 3), dtype=np.float32)
image_names = []

for i, p in enumerate(test_images):
    image_names.append(os.path.basename(p))
    with Image.open(p) as im:
        im = im.convert("RGB")
        x_test[i] = np.asarray(im, dtype=np.float32)

x_test = x_test / 255.0
print("x_test:", x_test.shape, x_test.dtype)



## === cell 18
Xt = make_features(x_test)
Xt_s = (Xt - mu) / sigma
Xt_b = np.concatenate([Xt_s, np.ones((Xt_s.shape[0], 1), dtype=np.float32)], axis=1)

pred_probs = sigmoid(Xt_b @ w).reshape(-1).astype(np.float32)

pred_map = dict(zip(image_names, pred_probs))
sub = sample_sub.copy()
sub["has_cactus"] = sub["id"].map(pred_map).astype(np.float32)

sub["has_cactus"] = sub["has_cactus"].fillna(0.5).astype(np.float32)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print(
    "has_cactus range:", float(sub["has_cactus"].min()), float(sub["has_cactus"].max())
)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4075875692.py in <cell line: 0>()
      1 # Feature transform using training standardization
----> 2 Xt = make_features(x_test)
      3 Xt_s = (Xt - mu) / sigma
      4 Xt_b = np.concatenate([Xt_s, np.ones((Xt_s.shape[0], 1), dtype=np.float32)], axis=1)
      5 

/tmp/ipykernel_11/1963902062.py in make_features(x)
      7     # (Same "spirit" as CNN input pipeline, but avoids TF dependency in this runtime.)
      8     n = x.shape[0]
----> 9     return x.reshape(n, -1).astype(np.float32)
     10 
     11 

ValueError: cannot reshape array of size 0 into shape (0,newaxis)
