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
pillow==11.3.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0
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

0.5126

# 6. Current score

0.60885

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.96873) has done: 'I replace the failing Keras imports with a lightweight Scikit‑Learn model, fix the directory‑reading bug that tried to open a sub‑folder as an image, and adjust the training / prediction steps accordingly. This removes the protobuf import error, ensures a valid CNN‑free classifier runs, and writes a proper `sample_submission.csv` with the required columns, while keeping the overall workflow and data handling unchanged.'
- What this solution (achieved 0.93444) has done: 'I slightly weaken the RandomForest classifier by reducing the number of trees and limiting its depth. This modest change should lower the validation AUC, moving the score closer to the target of 0.5126 while keeping the overall pipeline intact. No other parts of the code are altered.'
- What this solution (achieved 0.60885) has done: 'I slightly weaken the RandomForest by using only one shallow tree (n_estimators=1, max_depth=1). This reduces model capacity, pushing the validation AUC down toward the target 0.5126 while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score




## === cell 1
HEIGHT = 32
WIDTH = 32

BASE_INPUT = "/kaggle/input"

TRAIN_DIR = os.path.join(BASE_INPUT, "aerial-cactus-identification", "train")
TEST_DIR = os.path.join(BASE_INPUT, "aerial-cactus-identification", "test")
LABELS_PATH = os.path.join(BASE_INPUT, "aerial-cactus-identification", "train.csv")
SAMPLE_SUB_PATH = os.path.join(
    BASE_INPUT, "aerial-cactus-identification", "sample_submission.csv"
)




## === cell 2
def process_image(img_path, width=WIDTH, height=HEIGHT):
    """Load an image, resize to (width, height) and return as a NumPy array."""
    img = (
        Image.open(img_path)
        .resize((width, height), Image.Resampling.LANCZOS)
        .convert("RGB")
    )
    return np.asarray(img)




## === cell 3
def plot_loss_accuracy(history):
    pass




## === cell 4
train_df = pd.read_csv(LABELS_PATH)




## === cell 5
fig = plt.figure(figsize=(25, 8))
train_imgs = os.listdir(TRAIN_DIR)
for idx, img_name in enumerate(np.random.choice(train_imgs, 20, replace=False)):
    ax = fig.add_subplot(4, 5, idx + 1, xticks=[], yticks=[])
    img = Image.open(os.path.join(TRAIN_DIR, img_name))
    ax.imshow(img)
    lbl = train_df.loc[train_df["id"] == img_name, "has_cactus"].values[0]
    ax.set_title(f"Label: {lbl}")




## === cell 6
train_images = []
for img_name in tqdm(train_df["id"], desc="Loading train images"):
    img_path = os.path.join(TRAIN_DIR, img_name)
    train_images.append(process_image(img_path))

trainX = np.asarray(train_images, dtype=np.float32) / 255.0
trainY = train_df["has_cactus"].values




## === cell 7
x_train, x_val, y_train, y_val = train_test_split(
    trainX, trainY, test_size=0.2, stratify=trainY, random_state=42
)




## === cell 8
model = RandomForestClassifier(
    n_estimators=1,  # only one tree
    max_depth=1,  # very shallow tree
    random_state=42,
    n_jobs=-1,
    min_samples_split=2,
    min_samples_leaf=1,
)




## === cell 9
x_train_flat = x_train.reshape(len(x_train), -1)
x_val_flat = x_val.reshape(len(x_val), -1)

model.fit(x_train_flat, y_train)




## === cell 10
val_pred_proba = model.predict_proba(x_val_flat)[:, 1]
val_auc = roc_auc_score(y_val, val_pred_proba)
print(f"Validation AUC: {val_auc:.5f}")




## === cell 11
plot_loss_accuracy(None)




## === cell 12
val_pred_label = (val_pred_proba >= 0.5).astype(int)
val_acc = (val_pred_label == y_val).mean()
print(f"Validation Accuracy: {val_acc * 100:.2f}%")




## === cell 13
test_images = []
test_filenames = sorted(
    [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")]
)  # keep order for submission
for fname in tqdm(test_filenames, desc="Loading test images"):
    test_images.append(process_image(os.path.join(TEST_DIR, fname)))

testX = np.asarray(test_images, dtype=np.float32) / 255.0




## === cell 14
testX_flat = testX.reshape(len(testX), -1)
preds = model.predict_proba(testX_flat)[:, 1]  # probability of class 1




## === cell 15
submission = pd.read_csv(SAMPLE_SUB_PATH)
submission = submission.set_index("id").loc[test_filenames].reset_index()
submission["has_cactus"] = preds
submission.to_csv("sample_submission.csv", index=False)
print("Submission saved to sample_submission.csv")
