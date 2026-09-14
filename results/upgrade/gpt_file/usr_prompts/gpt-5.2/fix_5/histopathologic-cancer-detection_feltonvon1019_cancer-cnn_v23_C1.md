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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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

0.4574

# 6. Current score

0.65204

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.60789) has done: 'The timeout is dominated by two hotspots: (1) per-image Python loops for color balancing over up to 30k training images, and (2) slow, non-parallel test image loading (45k files) plus repeated `pd.concat` growth in a loop. I keep the exact same preprocessing math, model, and training loop, but vectorize the color-balance step using NumPy (identical formula) and parallelize image reading with OpenCV using a thread pool while preallocating arrays. I also avoid quadratic-time DataFrame concatenation by collecting batches and concatenating once at the end, preserving identical outputs. Finally, I reduce unnecessary work from plotting/extra reads by gating visualization behind a flag (doesn’t affect training/inference correctness).'
- What this solution (achieved 0.65926) has done: 'The crash happens before training because importing `keras` triggers a known protobuf compatibility issue in some Kaggle images (“MessageFactory has no attribute GetPrototype”). I fix that by switching the imports to `tf_keras` (already installed) while keeping the exact same model architecture, training loop, and inference logic. I also add a small, score-neutral safety step to ensure the submission rows align exactly with `sample_submission.csv` order (same predictions, just correct ordering), and keep the output filename as `submission.csv`. No changes are made to preprocessing math, model layers, optimizer, batch size, or epochs, so the score should remain essentially the same (still above target, but the primary goal here is to run end-to-end and produce a valid CSV).'
- What this solution (achieved 0.65204) has done: 'I fix the protobuf/keras import crash by forcing TensorFlow’s pure-Python protobuf implementation before any `tf_keras` import, which avoids the `MessageFactory.GetPrototype` error in Kaggle images. I also correct the notebook cell numbering so it runs as-provided (your original starts at cell 0, but I keep the same code order/content). To keep the score from drifting (it’s already well above the target), I won’t change any modeling, preprocessing, training, or inference logic—only the environment fix and making sure the submission is written properly as `submission.csv` with the right columns/order.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd

print(os.listdir("../input"))



## === cell 1
from glob import glob
import itertools
import shutil
from sklearn.utils import shuffle
import math

import tf_keras as keras
import cv2 as cv

from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tf_keras.layers import Conv2D, MaxPool2D

from tqdm import trange
import matplotlib.pyplot as plt
import gc  # garbage collection

import random

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(42)
random.seed(42)
try:
    keras.utils.set_random_seed(42)
except Exception:
    pass

DO_PLOTS = False

from concurrent.futures import ThreadPoolExecutor

_CPU = os.cpu_count() or 4
IO_WORKERS = min(8, max(2, _CPU))  # conservative to avoid oversubscription on Kaggle



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
if os.path.exists("../input/histopathologic-cancer-detection/train_labels.csv"):
    base_path = "../input/histopathologic-cancer-detection/"
else:
    base_path = "../input/"

train_path = os.path.join(base_path, "train")
test_path = os.path.join(base_path, "test")
labels_path = os.path.join(base_path, "train_labels.csv")

print("Using base_path:", base_path)
print("Train path exists:", os.path.exists(train_path))
print("Test path exists:", os.path.exists(test_path))
print("Labels path exists:", os.path.exists(labels_path))

df = pd.DataFrame({"path": glob(os.path.join(train_path, "*.tif"))})
df["id"] = df["path"].map(lambda x: os.path.splitext(os.path.basename(x))[0])
labels = pd.read_csv(labels_path)
df = df.merge(labels, on="id")
df.head(3)




## === cell 3
def _imread_or_zeros(path):
    im = cv.imread(path, cv.IMREAD_COLOR)
    if im is None:
        im = np.zeros((96, 96, 3), dtype=np.uint8)
    return im


def load_data(N, df_):
    """Load first N images from dataframe with columns: path, label."""
    N = int(min(N, len(df_)))
    paths = df_["path"].values[:N]
    y = df_["label"].to_numpy(dtype=np.uint8)[:N]

    with ThreadPoolExecutor(max_workers=IO_WORKERS) as ex:
        imgs = list(ex.map(_imread_or_zeros, paths))
    X = np.stack(imgs, axis=0).astype(np.uint8, copy=False)
    return X, y




## === cell 4
N = 10000
X, y = load_data(N=N, df_=df)



## === cell 5
if DO_PLOTS:
    fig = plt.figure(figsize=(10, 4), dpi=150)
    np.random.seed(100)
    for plotNr, idx in enumerate(np.random.randint(0, N, 8)):
        ax = fig.add_subplot(2, 8 // 2, plotNr + 1, xticks=[], yticks=[])
        plt.imshow(X[idx])
        ax.set_title("Label: " + str(y[idx]))



## === cell 6
if DO_PLOTS:
    fig = plt.figure(figsize=(4, 2), dpi=150)
    plt.bar([1, 0], [(y == 0).sum(), (y == 1).sum()])
    plt.xticks([1, 0], [f"Negative (N={(y==0).sum()})", f"Positive (N={(y==1).sum()})"])
    plt.ylabel("# of samples")



## === cell 7
if DO_PLOTS:
    demo_candidates = glob(os.path.join(train_path, "*.tif"))
    demo_path = os.path.join(train_path, "019ce31cc317087ca287f66ad757776952826594.tif")
    if not os.path.exists(demo_path) and len(demo_candidates) > 0:
        demo_path = demo_candidates[0]

    img = cv.imread(demo_path, cv.IMREAD_COLOR)
    r, g, b = cv.split(img)
    r_avg = cv.mean(r)[0]
    g_avg = cv.mean(g)[0]
    b_avg = cv.mean(b)[0]

    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg

    r = cv.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)

    balance_img = cv.merge([b, g, r])

    plt.figure(figsize=(25, 12))
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(balance_img)



## === cell 8
if DO_PLOTS:
    train_dir = train_path
    train_imgs = [
        os.path.join(train_dir, i) for i in os.listdir(train_dir) if i.endswith(".tif")
    ]

    plt.figure(figsize=(25, 12))
    for idx, train_img in enumerate(train_imgs[:15]):
        temp_img = cv.imread(train_img, cv.IMREAD_COLOR)
        plt.subplot(3, 5, idx + 1)
        plt.imshow(temp_img)



## === cell 9
if DO_PLOTS:
    plt.figure(figsize=(25, 12))
    for idx, train_img in enumerate(train_imgs[:15]):
        temp_img = cv.imread(train_img, cv.IMREAD_COLOR)
        r, g, b = cv.split(temp_img)
        r_avg = cv.mean(r)[0]
        g_avg = cv.mean(g)[0]
        b_avg = cv.mean(b)[0]
        k = (r_avg + g_avg + b_avg) / 3
        kr = k / r_avg
        kg = k / g_avg
        kb = k / b_avg
        r = cv.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
        g = cv.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
        b = cv.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
        balance_img = cv.merge([b, g, r])

        plt.subplot(3, 5, idx + 1)
        plt.imshow(balance_img)



## === cell 10
N = min(30000, len(df))
X, y = load_data(N=N, df_=df)



## === cell 11
Xf = X.astype(np.float32, copy=False)
means = Xf.reshape(Xf.shape[0], -1, 3).mean(axis=1)  # (N,3) means in BGR order
k = means.mean(axis=1, keepdims=True)  # (N,1)
scale = k / means  # (N,3)
Xf *= scale[:, None, None, :]  # broadcast multiply
X = np.clip(Xf, 0.0, 255.0).astype(np.uint8)
del Xf, means, k, scale
gc.collect()



## === cell 12
positives_samples = None
negative_samples = None
gc.collect()



## === cell 13
training_portion = 0.8
split_idx = int(np.round(training_portion * y.shape[0]))

np.random.seed(42)
idx = np.arange(y.shape[0])
np.random.shuffle(idx)
X = X[idx]
y = y[idx]



## === cell 14
kernel_size = (3, 3)
pool_size = (2, 2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.5

model = Sequential()

model.add(Conv2D(first_filters, kernel_size, input_shape=(96, 96, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(first_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(second_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(third_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(dropout_dense))

model.add(Dense(1, activation="sigmoid"))



## === cell 15
batch_size = 50

model.compile(
    loss=keras.losses.binary_crossentropy,
    optimizer=keras.optimizers.Adam(0.001),
    metrics=["accuracy"],
)



## === cell 16
epochs = 3
for epoch in range(epochs):
    iterations = int(np.floor(split_idx / batch_size))
    loss, acc = 0.0, 0.0
    with trange(iterations) as t:
        for i in t:
            start_idx = i * batch_size
            x_batch = X[start_idx : start_idx + batch_size]
            y_batch = y[start_idx : start_idx + batch_size]

            metrics = model.train_on_batch(x_batch, y_batch)

            loss += float(metrics[0])
            acc += float(metrics[1])
            t.set_description("Running training epoch " + str(epoch))
            t.set_postfix(
                loss="%.2f" % round(loss / (i + 1), 2),
                acc="%.2f" % round(acc / (i + 1), 2),
            )



## === cell 17
val_len = y.shape[0] - split_idx
iterations = int(np.floor(val_len / batch_size))
loss, acc = 0.0, 0.0

with trange(max(iterations, 1)) as t:
    for i in t:
        start = split_idx + i * batch_size
        end = start + batch_size
        if start >= y.shape[0]:
            break
        x_batch = X[start:end]
        y_batch = y[start:end]

        metrics = model.test_on_batch(x_batch, y_batch)

        loss += float(metrics[0])
        acc += float(metrics[1])
        t.set_description("Running validation")
        t.set_postfix(
            loss="%.2f" % round(loss / (i + 1), 2), acc="%.2f" % round(acc / (i + 1), 2)
        )

den = max(i + 1, 1)
print("Validation loss:", loss / den)
print("Validation accuracy:", acc / den)



## === cell 18
X = None
y = None
gc.collect()



## === cell 19
base_test_dir = test_path
test_files = glob(os.path.join(base_test_dir, "*.tif"))
test_files.sort()

file_batch = 5000
max_idx = len(test_files)

sub_parts = []
for idx in range(0, max_idx, file_batch):
    print("Indexes: %i - %i" % (idx, min(idx + file_batch, max_idx)))
    batch_paths = test_files[idx : idx + file_batch]
    batch_ids = [os.path.splitext(os.path.basename(p))[0] for p in batch_paths]

    with ThreadPoolExecutor(max_workers=IO_WORKERS) as ex:
        imgs = list(ex.map(_imread_or_zeros, batch_paths))
    K_test = np.stack(imgs, axis=0).astype(np.uint8, copy=False)

    predictions = model.predict(K_test, verbose=0).reshape(-1).astype(float)

    sub_parts.append(pd.DataFrame({"id": batch_ids, "label": predictions}))

submission = pd.concat(sub_parts, ignore_index=True)

sample_path = os.path.join(base_path, "sample_submission.csv")
if not os.path.exists(sample_path):
    sample_path = "../input/sample_submission.csv"
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    submission = sample[["id"]].merge(submission, on="id", how="left")
    submission["label"] = submission["label"].fillna(0.5)

submission.head()



## === cell 20
submission.to_csv("submission.csv", index=False, header=True)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
