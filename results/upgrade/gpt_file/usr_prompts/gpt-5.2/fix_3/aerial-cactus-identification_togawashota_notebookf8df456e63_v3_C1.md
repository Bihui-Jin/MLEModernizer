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

0.5

# 6. Current score

0.79321

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.79321) has done: 'I fix the TensorFlow import crash by removing the TF/Keras dependency entirely (it’s currently failing before any training happens), while keeping the overall pipeline intent: load images, train a classifier, and output probabilities for `has_cactus`. To keep changes minimal and stable in this environment, I replace the broken TF model with a lightweight, deterministic NumPy logistic-regression classifier trained on raw 32×32 RGB pixels (same data/labels, still outputs probabilities suitable for ROC AUC). I also fix the `flow_from_dataframe` label type issue implicitly by not using that API, and ensure the submission is written as `/kaggle/working/submission.csv` with correct row order matching `sample_submission.csv`. This run end-to-end within the time limit and produce a valid submission CSV.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
import random

os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)



## === cell 2
import zipfile

extract_dir = "/kaggle/working"

train_zip = "/kaggle/input/aerial-cactus-identification/train.zip"
test_zip = "/kaggle/input/aerial-cactus-identification/test.zip"

os.makedirs(os.path.join(extract_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(extract_dir, "test"), exist_ok=True)

with zipfile.ZipFile(train_zip, "r") as zip_ref:
    zip_ref.extractall(os.path.join(extract_dir, "train"))

with zipfile.ZipFile(test_zip, "r") as zip_ref:
    zip_ref.extractall(os.path.join(extract_dir, "test"))

print("Extracted train/test zips.")



## === cell 3
for dirname, subdirs, _ in os.walk("/kaggle/working"):
    if dirname.count(os.sep) <= ("/kaggle/working".count(os.sep) + 2):
        print(dirname, "subdirs:", subdirs)



## === cell 4
pass




## === cell 5
def resolve_image_dir(base_dir: str) -> str:
    nested = os.path.join(base_dir, os.path.basename(base_dir))
    if os.path.isdir(nested):
        return nested
    for cand in ["train", "test"]:
        c = os.path.join(base_dir, cand)
        if os.path.isdir(c):
            return c
    return base_dir


train_dir = resolve_image_dir("/kaggle/working/train")
test_dir = resolve_image_dir("/kaggle/working/test")

train_df = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
print("train_dir:", train_dir)
print("test_dir :", test_dir)
train_df.head(5)




## === cell 6
def count_files(directory):
    if not os.path.isdir(directory):
        return 0
    return len(
        [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    )


train_count = count_files(train_dir)
test_count = count_files(test_dir)

print(f"Train images: {train_count}")
print(f"Test images: {test_count}")

if train_count == 0 or test_count == 0:
    raise RuntimeError(
        f"Image directories appear empty. train_dir={train_dir} ({train_count}), test_dir={test_dir} ({test_count})."
    )



## === cell 7
class_ratio = train_df["has_cactus"].value_counts(normalize=True) * 100
print(class_ratio)



## === cell 8
import matplotlib.pyplot as plt

counts = train_df["has_cactus"].value_counts()
labels = ["Has Cactus (1)", "No Cactus (0)"]
colors = ["lightgreen", "lightcoral"]

plt.figure(figsize=(6, 6))
plt.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
plt.title("Distribution of Cactus Presence (has_cactus)")
plt.axis("equal")
plt.show()



## === cell 9
import cv2

idxs = [0, 1, 2, 8, 9, 12, 6, 7, 11, 14, 16, 17]
imgs = []
shown_labels = []

for i in idxs:
    img_path = os.path.join(train_dir, train_df.loc[i, "id"])
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs.append(img)
    shown_labels.append(f"id={train_df.loc[i,'id']} y={train_df.loc[i,'has_cactus']}")

if len(imgs) > 0:
    plt.figure(figsize=[10, 10])
    for x in range(len(imgs)):
        plt.subplot(4, 3, x + 1)
        plt.imshow(imgs[x])
        plt.title(shown_labels[x], fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("No images could be read for preview (non-fatal).")



## === cell 10
train_df["has_cactus"] = train_df["has_cactus"].astype(np.int32)



## === cell 11
import sys

print("Python:", sys.version)




## === cell 12
def custom_preprocessing(image):
    k = random.randint(0, 3)
    image = np.rot90(image, k)

    if random.random() > 0.5:
        image = np.fliplr(image)

    if random.random() > 0.5:
        image = np.flipud(image)

    factor = random.uniform(0.8, 1.2)
    image = np.clip(image * factor, 0, 255).astype(np.float32) / 255.0
    return image


cactus = []
for i in range(12):
    img_path = os.path.join(train_dir, train_df.loc[i, "id"])
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cactus.append(img)

if len(cactus) > 0:
    cactus_augmented = [custom_preprocessing(img) for img in cactus]

    plt.figure(figsize=(10, 10))
    for i in range(min(12, len(cactus_augmented))):
        plt.subplot(4, 3, i + 1)
        plt.imshow(cactus_augmented[i])
        plt.title(f"Aug {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
else:
    print("Skipped augmentation preview (no images read).")



## === cell 13
sample_sub = pd.read_csv(
    "/kaggle/input/aerial-cactus-identification/sample_submission.csv"
)
test_df = sample_sub[["id"]].copy()
print("test_df shape:", test_df.shape)
test_df.head()




## === cell 14
def load_images(ids, directory, augment=False):
    X = np.empty((len(ids), 32, 32, 3), dtype=np.float32)
    bad = 0
    for i, fname in enumerate(ids):
        path = os.path.join(directory, fname)
        img = cv2.imread(path)
        if img is None:
            bad += 1
            X[i] = 0.0
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (32, 32):
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        if augment:
            img = custom_preprocessing(img)  # already scales to [0,1]
        else:
            img = img.astype(np.float32) / 255.0
        X[i] = img
    if bad:
        print(
            f"Warning: {bad} images could not be read from {directory}. They were filled with zeros."
        )
    return X


y = train_df["has_cactus"].values.astype(np.float32)
X_train = load_images(train_df["id"].values, train_dir, augment=False)
print("X_train:", X_train.shape, X_train.dtype, "y:", y.shape, y.dtype)

X_test = load_images(test_df["id"].values, test_dir, augment=False)
print("X_test :", X_test.shape, X_test.dtype)




## === cell 15
def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


Xtr = X_train.reshape(len(X_train), -1)
Xte = X_test.reshape(len(X_test), -1)

mu = Xtr.mean(axis=0, keepdims=True)
std = Xtr.std(axis=0, keepdims=True) + 1e-6
Xtr_s = (Xtr - mu) / std
Xte_s = (Xte - mu) / std

Xtr_b = np.concatenate([Xtr_s, np.ones((Xtr_s.shape[0], 1), dtype=np.float32)], axis=1)
Xte_b = np.concatenate([Xte_s, np.ones((Xte_s.shape[0], 1), dtype=np.float32)], axis=1)

print("Design matrix:", Xtr_b.shape, Xtr_b.dtype)



## === cell 16
w = np.zeros((Xtr_b.shape[1],), dtype=np.float32)

lr = 0.1
epochs = 200
l2 = 1e-3  # mild regularization
n = Xtr_b.shape[0]

for ep in range(epochs):
    p = sigmoid(Xtr_b @ w)
    grad = (Xtr_b.T @ (p - y)) / n
    grad[:-1] += l2 * w[:-1]  # no regularization on bias
    w -= lr * grad

    if ep in {0, 1, 2, 4, 9, 19, 49, 99, 199}:
        eps = 1e-7
        loss = -np.mean(
            y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
        ) + 0.5 * l2 * np.sum(w[:-1] ** 2)
        acc = np.mean((p >= 0.5).astype(np.float32) == y)
        print(f"epoch={ep:3d} loss={loss:.5f} acc={acc:.4f}")



## === cell 17
preds = sigmoid(Xte_b @ w).astype(np.float64)
print(
    "preds:",
    preds.shape,
    preds.dtype,
    "min/max:",
    float(preds.min()),
    float(preds.max()),
)



## === cell 18
plt.figure(figsize=(6, 4))
plt.hist(preds, bins=30)
plt.title("Test prediction distribution")
plt.xlabel("P(has_cactus)")
plt.ylabel("count")
plt.show()



## === cell 19
print(
    "Skipping Keras model loading; using NumPy-trained weights. preds ready:",
    preds.shape,
)



## === cell 20
predictions = preds.reshape(-1)
predictions = predictions[: len(test_df)]  # safety

submission = pd.DataFrame({"id": test_df["id"].values, "has_cactus": predictions})
print(submission.head())
print("submission shape:", submission.shape)
assert (
    submission.shape[0] == sample_sub.shape[0]
), "Submission row count must match sample_submission."
assert list(submission.columns) == [
    "id",
    "has_cactus",
], "Submission columns must be exactly: id, has_cactus"



## === cell 21
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)



## === cell 22
print(os.listdir("/kaggle/working")[:20])
print("Exists submission.csv:", os.path.exists("/kaggle/working/submission.csv"))
print(pd.read_csv("/kaggle/working/submission.csv").head())
