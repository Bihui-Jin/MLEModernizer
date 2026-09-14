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

3.7

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tqdm==4.67.1

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

0.8015

# 6. Current score

0.50292

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50272) has done: 'The fix loads and preprocesses all images consistently, replaces the failing TensorFlow model with a simple scikit‑learn logistic regression (which avoids the protobuf issue), correctly splits data, evaluates ROC‑AUC, and writes a proper `sample_submission.csv` containing the required `id,has_cactus` columns.'
- What this solution (achieved 0.50377) has done: 'I replace the sharpening‑based features with the original raw pixel values, add standard‑scaler normalization, and give the logistic regression a balanced class weight to improve discrimination. These minimal adjustments keep the overall pipeline (logistic regression on flattened images) unchanged while addressing scaling and class imbalance, which should raise the ROC‑AUC toward the target.'
- What this solution (achieved 0.50497) has done: 'I add the missing use of the sharpening feature by computing sharpened versions of both train and test images, flattening them, and concatenating them with the raw pixel features before scaling and fitting the logistic regression. This small feature‑engineering tweak keeps the same logistic‑regression pipeline while providing richer information, which should raise the ROC‑AUC toward the target.'
- What this solution (achieved 0.50292) has done: 'I fixed the missing `glob` import, corrected the fallback directory logic, and added a safety check so the image‑loading step always succeeds. With these fixes the whole pipeline runs, computes the validation ROC‑AUC, and writes a proper `sample_submission.csv` containing the required `id,has_cactus` columns.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter




## === cell 1
def load_imgs(path):
    """
    Load all JPEG images from a directory, resize to 32x32, convert to RGB,
    and scale pixel values to [0,1]. Returns a dict {filename: image_array}.
    """
    imgs = {}
    for f in [os.path.basename(p) for p in glob.glob(os.path.join(path, "*.jpg"))]:
        fp = os.path.join(path, f)
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            continue  # skip unreadable files
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (32, 32))
        imgs[f] = img.astype(np.float32) / 255.0
    return imgs


train_dir = "../input/aerial-cactus-identification/train"
if not os.path.isdir(train_dir):
    train_dir = "../input/train"
test_dir = "../input/aerial-cactus-identification/test"
if not os.path.isdir(test_dir):
    test_dir = "../input/test"

img_train = load_imgs(train_dir)
img_test = load_imgs(test_dir)

assert len(img_train) > 0, "No training images were loaded."
assert len(img_test) > 0, "No test images were loaded."




## === cell 2
train_csv = pd.read_csv("../input/aerial-cactus-identification/train.csv")
if train_csv.empty:
    train_csv = pd.read_csv("../input/train.csv")

X_train = []
Y_train = []

for _, row in train_csv.iterrows():
    fname = row["id"]
    if fname in img_train:
        X_train.append(img_train[fname])
        Y_train.append(int(row["has_cactus"]))

X_train = np.array(X_train)  # (n_samples, 32, 32, 3)
Y_train = np.array(Y_train)

test_filenames = sorted(img_test.keys())
X_test = np.array([img_test[f] for f in test_filenames])

print("Training data shape:", X_train.shape, "=>", Y_train.shape)
print("Test data shape:", X_test.shape)




## === cell 3
plt.rcParams["axes.grid"] = False




## === cell 4
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for i, ax in enumerate(axes):
    idx = i * 200  # spread out samples
    if idx < len(X_train):
        ax.imshow(X_train[idx])
        ax.set_title("Has cactus: " + str(Y_train[idx]))
plt.show()




## === cell 5
def img_sharpen(img):
    blurred_f = gaussian_filter(img, 2)
    filter_blurred_f = gaussian_filter(blurred_f, 2)
    alpha = 15
    sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
    return np.clip(sharpened, 0, 1)


sharp_img_xtrain = np.array([img_sharpen(im) for im in X_train])
sharp_img_test = np.array([img_sharpen(im) for im in X_test])




## === cell 6
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for i, ax in enumerate(axes):
    idx = i * 200
    if idx < len(sharp_img_xtrain):
        ax.imshow(sharp_img_xtrain[idx])
        ax.set_title("Has cactus: " + str(Y_train[idx]))
plt.show()




## === cell 7
X_flat_raw = X_train.reshape(len(X_train), -1)  # (n_samples, 3072)
X_flat_sharp = sharp_img_xtrain.reshape(len(sharp_img_xtrain), -1)  # (n_samples, 3072)


def channel_stats(arr):
    """Mean and std per channel, concatenated -> (n_samples, 6)"""
    means = arr.mean(axis=(1, 2))
    stds = arr.std(axis=(1, 2))
    return np.concatenate([means, stds], axis=1)


stats_raw = channel_stats(X_train)  # (n_samples, 6)
stats_sharp = channel_stats(sharp_img_xtrain)  # (n_samples, 6)

X_combined = np.concatenate(
    [X_flat_raw, X_flat_sharp, stats_raw, stats_sharp], axis=1
)  # (n_samples, 6156)

x_tr, x_val, y_tr, y_val = train_test_split(
    X_combined, Y_train, test_size=0.2, random_state=42, stratify=Y_train
)




## === cell 8
scaler = StandardScaler()
x_tr_scaled = scaler.fit_transform(x_tr)
x_val_scaled = scaler.transform(x_val)

clf = LogisticRegression(
    max_iter=1000,
    solver="liblinear",
    C=4.0,
    class_weight="balanced",
    random_state=42,
)
clf.fit(x_tr_scaled, y_tr)

val_probs = clf.predict_proba(x_val_scaled)[:, 1]
val_auc = roc_auc_score(y_val, val_probs)
print(f"Validation ROC‑AUC: {val_auc:.4f}")




## === cell 9
fpr, tpr, _ = roc_curve(y_val, val_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=1.5,
    label="ROC curve (area = %0.3f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Validation ROC Curve")
plt.legend(loc="lower right")
plt.show()




## === cell 10
X_test_flat_raw = X_test.reshape(len(X_test), -1)
X_test_flat_sharp = sharp_img_test.reshape(len(sharp_img_test), -1)

stats_test_raw = channel_stats(X_test)
stats_test_sharp = channel_stats(sharp_img_test)

X_test_combined = np.concatenate(
    [X_test_flat_raw, X_test_flat_sharp, stats_test_raw, stats_test_sharp], axis=1
)

X_test_scaled = scaler.transform(X_test_combined)




## === cell 11
test_probs = clf.predict_proba(X_test_scaled)[:, 1]




## === cell 12
submission = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")
if submission.empty:
    submission = pd.read_csv("../input/sample_submission.csv")

submission["has_cactus"] = test_probs
submission.to_csv("sample_submission.csv", index=False)
print("Submission file saved as sample_submission.csv")
print(submission.head())
