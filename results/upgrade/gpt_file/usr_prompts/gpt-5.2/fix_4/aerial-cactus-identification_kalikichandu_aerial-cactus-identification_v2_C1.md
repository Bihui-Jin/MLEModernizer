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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8216

# 6. Current score

0.99967

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the environment/runtime issues caused by Keras 3 API changes by switching to `tf_keras` (TensorFlow Keras) for stable `keras.preprocessing.image`, callbacks, and weight saving/loading. I update the `ModelCheckpoint` to monitor `val_accuracy` and save weights to a valid `.weights.h5` filename, then load those same weights before inference. I replace the deprecated `predict_proba` with `predict`, ensure AUC is computed on proper 1D arrays, and fix the test folder traversal so directories (like the nested `test/` folder) are skipped. Finally, I create the submission by merging predictions into `sample_submission.csv` to guarantee row count/order correctness and write a valid `submission.csv`.'
- What this solution (achieved 0.99934) has done: 'I fix the initial import crash by forcing the TensorFlow/Keras stack to use the pure-Python protobuf implementation, which avoids the `MessageFactory.GetPrototype` error in this environment. Then I correct the dataset path resolution so `TRAIN_DIR` and `TEST_DIR` point to the actual image folders (the current nested `train/train` guess is wrong here), and add a safety fallback that auto-detects the correct directory if Kaggle’s directory structure differs. These changes unblock image loading/training so the model can actually learn (your current 0.5 score is consistent with a broken training/prediction pipeline), while preserving the same CNN architecture/training loop and producing a valid `submission.csv` in the required format.'
- What this solution (achieved 0.99967) has done: 'I fix the runtime crash happening before your code runs by pinning protobuf to the pure-Python backend *before any TF/Keras-related import*, and by clearing any conflicting `PROTOCOL_BUFFERS_*` env vars that can override it in Kaggle. I keep your CNN/training loop and prediction logic unchanged so score behavior stays essentially the same (your current score is already well above the target). I also add a small, score-neutral safety check that ensures `sub` has the correct row count/order and no missing predictions after the merge, so the submission is always valid. Finally, I keep the output as `submission.csv` in the working directory.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import tf_keras as keras
from tf_keras.preprocessing import image
from tf_keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tf_keras.models import Sequential
from tf_keras.callbacks import ModelCheckpoint

from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

print(os.listdir("../input"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE = "../input/aerial-cactus-identification"
if not os.path.exists(BASE):
    BASE = "../input"

TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")


def resolve_image_dir(base, split_name):
    candidates = [
        os.path.join(base, split_name),
        os.path.join(base, split_name, split_name),
    ]
    for c in candidates:
        if os.path.isdir(c):
            try:
                files = os.listdir(c)
                if any(f.lower().endswith(".jpg") for f in files):
                    return c
            except Exception:
                pass

    try:
        for root, dirs, files in os.walk(base):
            rel = os.path.relpath(root, base)
            if rel.count(os.sep) > 2:
                continue
            if os.path.basename(root) == split_name and any(
                f.lower().endswith(".jpg") for f in files
            ):
                return root
    except Exception:
        pass

    return os.path.join(base, split_name)


TRAIN_DIR = resolve_image_dir(BASE, "train")
TEST_DIR = resolve_image_dir(BASE, "test")

output_dir = "../working/model_output/CNN"
os.makedirs(output_dir, exist_ok=True)

seed = 7
np.random.seed(seed)

WEIGHTS_PATH = os.path.join(output_dir, "weights.best.weights.h5")

print("Using paths:")
print("BASE:", BASE)
print("TRAIN_CSV:", TRAIN_CSV, "exists:", os.path.exists(TRAIN_CSV))
print("SAMPLE_SUB:", SAMPLE_SUB, "exists:", os.path.exists(SAMPLE_SUB))
print("TRAIN_DIR:", TRAIN_DIR, "exists:", os.path.isdir(TRAIN_DIR))
print("TEST_DIR:", TEST_DIR, "exists:", os.path.isdir(TEST_DIR))
print("WEIGHTS_PATH:", WEIGHTS_PATH)



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
train_df.head()



## === cell 3
train_image = []
missing = 0
for i in tqdm(range(len(train_df))):
    fp = os.path.join(TRAIN_DIR, train_df["id"][i])
    if not os.path.exists(fp):
        missing += 1
        continue
    img = image.load_img(fp, target_size=(32, 32))
    img = image.img_to_array(img)
    img = img / 255.0
    train_image.append(img)

if missing > 0:
    print(f"WARNING: {missing} training images were missing and were skipped.")

X = np.array(train_image, dtype=np.float32)
X.shape



## === cell 4
plt.imshow(X[1])
plt.axis("off")



## === cell 5
if len(X) != len(train_df):
    existing_mask = [
        os.path.exists(os.path.join(TRAIN_DIR, _id)) for _id in train_df["id"].values
    ]
    train_df_aligned = train_df.loc[existing_mask].reset_index(drop=True)
else:
    train_df_aligned = train_df

y = train_df_aligned["has_cactus"].values.astype(np.float32).reshape(-1, 1)
y.shape



## === cell 6
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y
)



## === cell 7
X_train.shape, X_test.shape, y_train.shape, y_test.shape



## === cell 8
model = Sequential()
model.add(
    Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 3))
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.summary()



## === cell 9
modelcheckpoint = ModelCheckpoint(
    filepath=WEIGHTS_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
    mode="max",
    verbose=1,
)



## === cell 10
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(
    X_train,
    y_train,
    epochs=80,
    validation_data=(X_test, y_test),
    batch_size=32,
    shuffle=True,
    callbacks=[modelcheckpoint],
    verbose=2,
)



## === cell 11
pred = {}


def predictions(imagepath, imagename):
    img = image.load_img(imagepath, target_size=(32, 32))
    img = image.img_to_array(img).astype(np.float32) / 255.0
    proba = model.predict(img.reshape(1, 32, 32, 3), verbose=0)[0][0]
    pred.update({imagename: float(proba)})




## === cell 12
if os.path.exists(WEIGHTS_PATH):
    model.load_weights(WEIGHTS_PATH)
else:
    print(
        "WARNING: Best weights not found at",
        WEIGHTS_PATH,
        "using last epoch weights instead.",
    )



## === cell 13
y_hat = model.predict(X_test, verbose=0).reshape(-1)
auc = roc_auc_score(y_test.reshape(-1), y_hat)
print("Validation ROC AUC:", auc)



## === cell 14
files = sorted(os.listdir(TEST_DIR))
for file in tqdm(files):
    fp = os.path.join(TEST_DIR, file)
    if (not os.path.isfile(fp)) or (not file.lower().endswith(".jpg")):
        continue
    predictions(fp, file)

print("Predicted files:", len(pred))



## === cell 15
sample_df = pd.read_csv(SAMPLE_SUB)
pred_df = pd.DataFrame(list(pred.items()), columns=["id", "has_cactus"])

sub = sample_df.merge(pred_df, on="id", how="left", suffixes=("_sample", ""))
if "has_cactus_sample" in sub.columns:
    sub["has_cactus"] = sub["has_cactus"].fillna(sub["has_cactus_sample"])
    sub = sub[["id", "has_cactus"]]

if len(sub) != len(sample_df):
    raise RuntimeError(f"Submission row count mismatch: {len(sub)} vs {len(sample_df)}")
if sub["has_cactus"].isna().any():
    sub["has_cactus"] = sub["has_cactus"].fillna(0.5)

sub.shape, sub.head()



## === cell 16
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
