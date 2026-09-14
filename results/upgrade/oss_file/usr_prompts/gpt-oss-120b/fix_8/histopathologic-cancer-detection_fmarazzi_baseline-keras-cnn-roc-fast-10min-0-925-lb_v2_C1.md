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

0.94728

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.94595) has done: 'The changes focus on eliminating Python‑level loops when reading images, reusing the training mean/std for all normalizations, and pre‑allocating NumPy arrays, all of which drastically cut I/O and computation time while keeping the exact model, data splits, and training procedure unchanged.'
- What this solution (achieved 0.94072) has done: 'The fix adds a protobuf compatibility setting before importing TensorFlow to stop the `MessageFactory` attribute error, allowing the rest of the pipeline (which already achieves a high AUC) to run unchanged and produce a valid `submission.csv`. No other logic is altered, preserving the current high score.'
- What this solution (achieved 0.94546) has done: 'The fix moves the protobuf compatibility setting to the very top, before any library imports, ensuring TensorFlow loads without the `MessageFactory` error. No modeling changes are made, preserving the high AUC score while now allowing the pipeline to run end‑to‑end and create a valid `submission.csv`.'
- What this solution (achieved 0.94451) has done: 'The fix moves the protobuf compatibility setting to the very top, before **any** library imports, preventing the `MessageFactory` error while keeping all modeling logic unchanged, so the high AUC score (0.945 …) is retained and a proper `submission.csv` is generated.'
- What this solution (achieved 0.93391) has done: 'The fix moves the protobuf compatibility setting to the very first lines of the script, before any library imports, preventing the `MessageFactory` attribute error while keeping the original model and training logic unchanged. No other logic is altered, so the high AUC score is retained and a proper `submission.csv` is generated.'
- What this solution (achieved 0.94728) has done: 'I moved the protobuf compatibility environment variable to the very first lines of the script (before any imports) to prevent the `MessageFactory` attribute error, and renumbered the cells so the notebook runs from cell 1 onward while keeping all original logic unchanged. This fix restores the pipeline, preserving the high AUC score (≈0.93) and allows a proper `submission.csv` to be generated.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import gc
import numpy as np
import pandas as pd
from glob import glob
from skimage.io import imread
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

print(os.listdir("../input"))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
base_tile_dir = "../input/train/"
df = pd.DataFrame({"path": glob(os.path.join(base_tile_dir, "*.tif"))})
df["id"] = df.path.map(lambda x: x.split("/")[-1].split(".")[0])
labels = pd.read_csv("../input/train_labels.csv")
df = df.merge(labels, on="id")

SAMPLES_N = 10000
SAMPLES_P = 10000
df0 = df[df.label == 0].sample(SAMPLES_N, random_state=42)
df1 = df[df.label == 1].sample(SAMPLES_P, random_state=42)
df = pd.concat([df0, df1], ignore_index=True)
df = df[["path", "id", "label"]]

image_list = [imread(p) for p in df["path"].values]
X = np.stack(image_list).astype(np.float32)
y = to_categorical(df["label"].values)

del df, image_list, labels
gc.collect()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

train_mean = X_train.mean()
train_std = X_train.std()
X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std




## === cell 2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

kernel_size = (3, 3)
pool_size = (2, 2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.3

model = Sequential(
    [
        Conv2D(first_filters, kernel_size, activation="relu", input_shape=(96, 96, 3)),
        Conv2D(first_filters, kernel_size, activation="relu"),
        Conv2D(first_filters, kernel_size, activation="relu"),
        MaxPool2D(pool_size=pool_size),
        Dropout(dropout_conv),
        Conv2D(second_filters, kernel_size, activation="relu"),
        Conv2D(second_filters, kernel_size, activation="relu"),
        Conv2D(second_filters, kernel_size, activation="relu"),
        MaxPool2D(pool_size=pool_size),
        Dropout(dropout_conv),
        Conv2D(third_filters, kernel_size, activation="relu"),
        Conv2D(third_filters, kernel_size, activation="relu"),
        Conv2D(third_filters, kernel_size, activation="relu"),
        MaxPool2D(pool_size=pool_size),
        Dropout(dropout_conv),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(dropout_dense),
        Dense(2, activation="softmax"),
    ]
)

optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

earlystopper = EarlyStopping(
    monitor="val_loss", patience=5, verbose=1, restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=64,
    callbacks=[earlystopper],
    verbose=2,
)




## === cell 3
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_pred_prob = model.predict(X_test, batch_size=64)[:, 1]
auc_score = roc_auc_score(y_test[:, 1], y_pred_prob)
fpr, tpr, _ = roc_curve(y_test[:, 1], y_pred_prob)

plt.figure(figsize=(6, 5))
plt.plot([0, 1], [0, 1], "k--")
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="best")
plt.show()




## === cell 4
base_test_dir = "../input/test/"
test_files = glob(os.path.join(base_test_dir, "*.tif"))

submission = pd.DataFrame()
file_batch = 5000
max_idx = len(test_files)

for idx in range(0, max_idx, file_batch):
    batch_files = test_files[idx : idx + file_batch]
    batch_ids = [p.split("/")[-1].split(".")[0] for p in batch_files]
    batch_images = [imread(p) for p in batch_files]
    K_test = np.stack(batch_images).astype(np.float32)
    K_test = (K_test - train_mean) / train_std  # reuse training stats
    preds = model.predict(K_test, batch_size=64)[:, 1]
    batch_df = pd.DataFrame({"id": batch_ids, "label": preds})
    submission = pd.concat([submission, batch_df], ignore_index=True)

submission.head()




## === cell 5
submission.to_csv("submission.csv", index=False, header=True)
