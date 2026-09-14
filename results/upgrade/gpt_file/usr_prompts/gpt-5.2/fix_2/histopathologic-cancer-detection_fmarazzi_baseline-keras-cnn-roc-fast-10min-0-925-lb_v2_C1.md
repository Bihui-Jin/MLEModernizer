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
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

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
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.8008

# 6. Current score

0.89536

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.89536) has done: 'I fix the runtime break caused by a protobuf/Keras import conflict by switching to `tf_keras` (which matches the installed TensorFlow-backed stack) and set the protobuf env var early to avoid the `MessageFactory` crash. I also update the Adam optimizer argument from deprecated `lr` to `learning_rate` so `compile()` succeeds, which unblocks training and inference. To improve score toward the target AUC, I keep the same CNN/training loop but generate submission probabilities (softmax positive-class) instead of hard class labels, and compute ROC using probabilities as well. Finally, I make the dataset paths robust for Kaggle (`../input/histopathologic-cancer-detection/...`) and fix brittle ID parsing.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd

from glob import glob
from skimage.io import imread
import gc

from sklearn.model_selection import train_test_split

from tf_keras.utils import to_categorical

print("Listing ../input:", os.listdir("../input")[:10])



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "../input/histopathologic-cancer-detection"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

if not os.path.exists(TRAIN_DIR):
    TRAIN_DIR = "../input/train"
if not os.path.exists(TEST_DIR):
    TEST_DIR = "../input/test"
if not os.path.exists(LABELS_CSV):
    LABELS_CSV = "../input/train_labels.csv"
if not os.path.exists(SAMPLE_SUB_CSV):
    SAMPLE_SUB_CSV = "../input/sample_submission.csv"

print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR :", TEST_DIR)
print("LABELS   :", LABELS_CSV)

base_tile_dir = TRAIN_DIR
df = pd.DataFrame({"path": glob(os.path.join(base_tile_dir, "*.tif"))})

df["id"] = df["path"].map(lambda x: os.path.splitext(os.path.basename(x))[0])

labels = pd.read_csv(LABELS_CSV)
df = df.merge(labels, on="id")
df.head(3)



## === cell 2
SAMPLES_N = 10000
SAMPLES_P = 10000

n0 = min(SAMPLES_N, (df.label == 0).sum())
n1 = min(SAMPLES_P, (df.label == 1).sum())

df0 = df[df.label == 0].sample(n0, random_state=42)
df1 = df[df.label == 1].sample(n1, random_state=42)
df = pd.concat([df0, df1], ignore_index=True).reset_index(drop=True)

df = df[["path", "id", "label"]]
df["image"] = df["path"].map(imread)

y = df["label"].values
X = np.stack(df["image"].values)

y = to_categorical(y, num_classes=2)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y.argmax(axis=1)
)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
train_mean = X_train.mean()
train_std = X_train.std() + 1e-7
X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

gc.collect()



## === cell 3
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, Flatten
from tf_keras.layers import Conv2D, MaxPool2D
from tf_keras.optimizers import Adam

kernel_size = (3, 3)
pool_size = (2, 2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.3

model = Sequential()
model.add(
    Conv2D(first_filters, kernel_size, activation="relu", input_shape=(96, 96, 3))
)
model.add(Conv2D(first_filters, kernel_size, activation="relu"))
model.add(Conv2D(first_filters, kernel_size, activation="relu"))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, activation="relu"))
model.add(Conv2D(second_filters, kernel_size, activation="relu"))
model.add(Conv2D(second_filters, kernel_size, activation="relu"))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, activation="relu"))
model.add(Conv2D(third_filters, kernel_size, activation="relu"))
model.add(Conv2D(third_filters, kernel_size, activation="relu"))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(2, activation="softmax"))

optimizer = Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
)

model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)
model.summary()



## === cell 4
from tf_keras.callbacks import EarlyStopping

earlystopper = EarlyStopping(
    monitor="val_loss", patience=2, verbose=1, restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=64,
    callbacks=[earlystopper],
    verbose=2,
)



## === cell 5
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

y_pred_proba = model.predict(X_test, batch_size=128, verbose=0)[:, 1]
y_true = y_test.argmax(axis=1)

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_pred_proba)
auc_keras = auc(fpr_keras, tpr_keras)
auc_sklearn = roc_auc_score(y_true, y_pred_proba)

plt.figure(figsize=(6, 5))
plt.plot([0, 1], [0, 1], "k--")
plt.plot(fpr_keras, tpr_keras, label="AUC = {:.4f}".format(auc_keras))
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve (validation)")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

print("Validation AUC (auc):", float(auc_keras))
print("Validation AUC (roc_auc_score):", float(auc_sklearn))



## === cell 6
base_test_dir = TEST_DIR
test_files = glob(os.path.join(base_test_dir, "*.tif"))

test_files = sorted(test_files)

submission_parts = []
file_batch = 5000
max_idx = len(test_files)

for idx in range(0, max_idx, file_batch):
    print("Indexes: %i - %i" % (idx, min(idx + file_batch, max_idx)))
    batch_paths = test_files[idx : idx + file_batch]
    test_df = pd.DataFrame({"path": batch_paths})
    test_df["id"] = test_df["path"].map(
        lambda x: os.path.splitext(os.path.basename(x))[0]
    )
    test_df["image"] = test_df["path"].map(imread)

    K_test = np.stack(test_df["image"].values).astype("float32")
    K_test = (K_test - train_mean) / train_std

    preds = model.predict(K_test, batch_size=128, verbose=0)[:, 1]
    test_df["label"] = preds.astype("float32")

    submission_parts.append(test_df[["id", "label"]])

    del test_df, K_test, preds
    gc.collect()

submission = pd.concat(submission_parts, ignore_index=True)

if os.path.exists(SAMPLE_SUB_CSV):
    sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
    submission = sample_sub[["id"]].merge(submission, on="id", how="left")
    submission["label"] = submission["label"].fillna(0.5).astype("float32")

submission.head()



## === cell 7
submission.to_csv("submission.csv", index=False, header=True)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.describe(include="all").head())
