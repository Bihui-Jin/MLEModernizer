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

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-image==0.25.2
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
tf_keras==2.18.0

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

0.8573

# 6. Current score

0.94893

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.94893) has done: 'I replace the TensorFlow‐based CNN with a lightweight scikit‑learn logistic regression that uses the same pre‑processed images. This removes the protobuf import error, restores the training/inference flow, and still outputs a correctly formatted `submission.csv`. The core data handling and preprocessing remain unchanged, preserving the original logic while fixing the runtime failures.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = (
    "python"  # fix protobuf import issue
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import skimage.exposure as exposure
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE_PATH = "/kaggle/input/aerial-cactus-identification/"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")

files_dataframe = pd.read_csv(os.path.join(BASE_PATH, "train.csv"), dtype=str)
files_dataframe.head()




## === cell 1
class_counts = files_dataframe["has_cactus"].value_counts()
ax = class_counts.plot.bar()
plt.title("Class distribution")
plt.show()

total = class_counts.sum()
has_cactus_w = total / (2 * class_counts.get("1", 1))
no_cactus_w = total / (2 * class_counts.get("0", 1))
class_weights = {0: no_cactus_w, 1: has_cactus_w}
print("Class weights:", class_weights)




## === cell 2
plt.figure(figsize=(12, 6))
for i, idx in enumerate(np.random.choice(len(files_dataframe), size=12, replace=False)):
    img_path = os.path.join(TRAIN_DIR, files_dataframe.iloc[idx]["id"])
    if not os.path.exists(img_path):
        continue
    plt.subplot(3, 4, i + 1)
    plt.imshow(imread(img_path))
    plt.title(f"Label: {files_dataframe.iloc[idx]['has_cactus']}")
plt.tight_layout()
plt.show()




## === cell 3
def preprocess(img):
    p2, p98 = np.percentile(img, (2, 98))
    return exposure.rescale_intensity(img, in_range=(p2, p98))


def load_images(df, directory):
    imgs = []
    lbls = []
    for _, row in df.iterrows():
        fp = os.path.join(directory, row["id"])
        if not os.path.exists(fp):
            continue
        img = imread(fp).astype(np.float32)
        img = preprocess(img)
        imgs.append(img)
        lbls.append(int(row["has_cactus"]))
    return np.stack(imgs), np.array(lbls)


X, y = load_images(files_dataframe, TRAIN_DIR)

X_flat = X.reshape((X.shape[0], -1))

X_train, X_val, y_train, y_val = train_test_split(
    X_flat, y, test_size=0.1, stratify=y, random_state=42
)




## === cell 4
logreg = LogisticRegression(
    max_iter=500,
    class_weight=class_weights,
    solver="lbfgs",
    multi_class="auto",
    n_jobs=-1,
)

logreg.fit(X_train, y_train)




## === cell 5
val_probs = logreg.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_probs)
print(f"Validation AUC: {val_auc:.5f}")




## === cell 6
test_files = sorted(os.listdir(TEST_DIR))
test_imgs = []
valid_filenames = []  # keep only files that are actual images
for fn in test_files:
    fp = os.path.join(TEST_DIR, fn)
    if not os.path.isfile(fp):
        continue
    img = imread(fp).astype(np.float32)
    img = preprocess(img)
    test_imgs.append(img)
    valid_filenames.append(fn)

test_imgs = np.stack(test_imgs)
test_flat = test_imgs.reshape((test_imgs.shape[0], -1))

test_probs = logreg.predict_proba(test_flat)[:, 1]

submission = pd.DataFrame({"id": valid_filenames, "has_cactus": test_probs})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv (", len(submission), "rows)")




## === cell 7
import shutil

for folder in ["test", "training"]:
    try:
        shutil.rmtree(folder)
    except OSError:
        pass
