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
Predict the individual whale species in images.

## Metric
Mean Average Precision @ 5 (MAP@5).

## Submission Format
For each `Image` in the test set, you may predict up to 5 labels for the whale `Id`. Whales that are not predicted to be one of the labels in the training data should be labeled as `new_whale`. The file should contain a header and have the following format:

```
Image,Id
00029b3a.jpg,new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46
0003c693.jpg,new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46
...
```

## Dataset
This training data contains thousands of images of humpback whale flukes. Individual whales have been identified by researchers and given an `Id`. The challenge is to predict the whale `Id` of images in the test set. What makes this such a challenge is that there are only a few examples for each of 3,000+ whale Ids.

- **train.zip** - a folder containing the training images
- **train.csv** - maps the training `Image` to the appropriate whale `Id`. Whales that are not predicted to have a label identified in the training data should be labeled as `new_whale`.
- **test.zip** - a folder containing the test images to predict the whale `Id`
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (51 lines)
            sample_submission.csv (2611 lines)
            sample_submission.csv.zip (15.5 kB)
            test.zip (72.1 MB)
            train.csv (7241 lines)
            train.csv.zip (68.3 kB)
            train.zip (200.4 MB)
            test/
                11482c0d.jpg (27.8 kB)
                68edd063.jpg (11.5 kB)
                ... and 2608 other files
                test/
            train/
                6fc790ac.jpg (17.1 kB)
                f4bb7bb0.jpg (11.6 kB)
                ... and 7238 other files
                train/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
        input/
            description.md (51 lines)
            sample_submission.csv (2611 lines)
            sample_submission.csv.zip (15.5 kB)
            test.zip (72.1 MB)
            train.csv (7241 lines)
            train.csv.zip (68.3 kB)
            train.zip (200.4 MB)
            test/
                11482c0d.jpg (27.8 kB)
                68edd063.jpg (11.5 kB)
                ... and 2608 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
            train/
                6fc790ac.jpg (17.1 kB)
                f4bb7bb0.jpg (11.6 kB)
                ... and 7238 other files
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
        working/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
```

-> data/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> data/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> data/whale-categorization-playground/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> data/whale-categorization-playground/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> input/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> input/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> (stopped after 10 files for performance)

# 5. Target score

0.00128

# 6. Current score

0.1148

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.1148) has done: 'Diagnosis: Cell 4 crashes because `test_images_dir` (picked in cell 1) points to a directory that exists but does not actually contain images in this environment (`../input/whale-categorization-playground/test/test`), so the fallback discovery still finds zero files and raises. The underlying issue is a mismatch between “directory exists” and “directory contains image files”, especially with the duplicated nested `test/test` structure in the dataset tree. We must not change cell 1, so the fix must make cell 4 robust by searching alternative known candidate directories when the chosen `test_images_dir` yields no images.  

Patch summary: In cell 4 only, replace the “discover images from `test_images_dir`” fallback with a deterministic search over `test_images_dir_candidates` (already defined in cell 1) plus the currently selected `test_images_dir`, and pick the first directory that contains image files (including one-level nested subfolders). Keep the rest of the prediction logic unchanged, and ensure `test` remains a list of full file paths so cell 5 continues to work.  

Updated cells: (cell 4 only)  

Compatibility notes for cell k+1: `test` remains a list of image file paths; `y_final` remains aligned with `test`, so `imageId = [os.path.basename(p) for p in test]` and submission creation in cell 5 work unchanged.  

Assumptions: Image files are `.jpg/.jpeg/.png` and are either directly inside a test directory or one level below it (as in Kaggle’s nested folder layouts).'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
import subprocess

try:
    import google.protobuf as _protobuf
    from packaging.version import Version

    _pb_ver = getattr(_protobuf, "__version__", "0")
    if Version(_pb_ver) >= Version("5"):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        import importlib

        importlib.invalidate_caches()
        if "google.protobuf" in sys.modules:
            importlib.reload(sys.modules["google.protobuf"])
except Exception:
    pass

import time
import tensorflow as tf
import pandas as pd
import numpy as np
from random import shuffle

folder = "../input/whale-categorization-playground/"

np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass



## === cell 1
train_images_dir_candidates = [
    os.path.join(folder, "train", "train"),
    os.path.join(folder, "train"),
    "../input/train/train",
    "../input/train",
]
test_images_dir_candidates = [
    os.path.join(folder, "test", "test"),
    os.path.join(folder, "test"),
    "../input/test/test",
    "../input/test",
]


def _pick_existing_dir(cands):
    for d in cands:
        if os.path.isdir(d):
            return d
    return None


train_images_dir = _pick_existing_dir(train_images_dir_candidates)
test_images_dir = _pick_existing_dir(test_images_dir_candidates)

if train_images_dir is None:
    raise FileNotFoundError(
        f"Could not find train images dir. Tried: {train_images_dir_candidates}"
    )
if test_images_dir is None:
    raise FileNotFoundError(
        f"Could not find test images dir. Tried: {test_images_dir_candidates}"
    )

idDict = {}
idx2id = {}

train = pd.read_csv(os.path.join(folder, "train.csv"))

img_paths = []
label_ids = []
for i in range(len(train)):
    whale_id = train.iloc[i, 1]
    if whale_id not in idDict:
        idDict[whale_id] = len(idDict)
        idx2id[idDict[whale_id]] = whale_id
    img_paths.append(os.path.join(train_images_dir, train.iloc[i, 0]))
    label_ids.append(idDict[whale_id])

x_train = pd.Series(img_paths)
y_train = pd.Series(label_ids)

sample = pd.read_csv(os.path.join(folder, "sample_submission.csv"))
test_files_set = set(os.listdir(test_images_dir))
missing = [im for im in sample["Image"].tolist() if im not in test_files_set]
if len(missing) > 0:
    test_images = sorted(list(test_files_set))
else:
    test_images = sample["Image"].tolist()

test = [os.path.join(test_images_dir, fn) for fn in test_images]

width, height = 150, 150

batchSize, iterations = 32, 1

num_classes = len(idDict)
num_classes



## === cell 2
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(2, 2),
            padding="Same",
            activation="relu",
            input_shape=(150, 150, 3),
        ),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="Same",
            activation="relu",
        ),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(5, 5),
            padding="Same",
            activation="relu",
        ),
        tf.keras.layers.MaxPool2D(pool_size=(5, 5)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()



## === cell 3
for k in range(iterations):
    indexes = list(range(len(x_train)))
    shuffle(indexes)
    i, iterationStartTime = 0, time.time()

    while i < len(indexes):
        batchStartTime = time.time()
        x, y = [], []
        for j in range(i, min(len(indexes), i + batchSize)):
            img_path = x_train.iloc[indexes[j]]

            if not os.path.exists(img_path):
                base = os.path.basename(img_path)
                resolved = None
                for d in train_images_dir_candidates:
                    cand = os.path.join(d, base)
                    if os.path.exists(cand):
                        resolved = cand
                        break
                if resolved is None:
                    raise FileNotFoundError(f"Image file not found: {img_path}")
                img_path = resolved

            image = tf.keras.preprocessing.image.load_img(
                img_path, target_size=(width, height)
            )
            image = (
                tf.keras.preprocessing.image.img_to_array(image).astype(np.float32)
                / 255.0
            )
            x.append(image)

            ans = np.zeros(num_classes, dtype=np.float32)
            ans[int(y_train.iloc[indexes[j]])] = 1.0
            y.append(ans)

        x, y = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

        model.fit(x, y, epochs=5, verbose=0, shuffle=False)

        i += batchSize
        if i % (batchSize * 50) == 0 or i >= len(indexes):
            print(
                "\tbatch: %0.2f%% - %0.2f seconds"
                % (100 * i / len(indexes), time.time() - batchStartTime)
            )

    print(
        "iteration: %0.2f%% - %0.2f seconds"
        % (100 * (k + 1) / iterations, time.time() - iterationStartTime)
    )


## === cell 4
y_final = []
pos = 0

if len(test) == 0:
    exts = (".jpg", ".jpeg", ".png")

    def _discover_images_in_dir(d):
        discovered_local = []
        try:
            for fn in os.listdir(d):
                p = os.path.join(d, fn)
                if os.path.isfile(p) and fn.lower().endswith(exts):
                    discovered_local.append(p)
        except FileNotFoundError:
            return []

        if len(discovered_local) > 0:
            return discovered_local

        try:
            for fn in os.listdir(d):
                sub = os.path.join(d, fn)
                if os.path.isdir(sub):
                    for subfn in os.listdir(sub):
                        if subfn.lower().endswith(exts):
                            discovered_local.append(os.path.join(sub, subfn))
        except FileNotFoundError:
            return []

        return discovered_local

    discovered = []
    search_dirs = []
    if test_images_dir is not None:
        search_dirs.append(test_images_dir)
    for d in test_images_dir_candidates:
        if d not in search_dirs:
            search_dirs.append(d)

    for d in search_dirs:
        imgs = _discover_images_in_dir(d)
        if len(imgs) > 0:
            discovered = imgs
            test_images_dir = d  # keep for error message consistency
            break

    test = sorted(discovered)

if len(test) == 0:
    raise ValueError(
        f"No test images found to predict on. Resolved test_images_dir={test_images_dir}"
    )

for path in test:
    image = tf.keras.preprocessing.image.load_img(path, target_size=(width, height))
    image = tf.keras.preprocessing.image.img_to_array(image).astype(np.float32) / 255.0
    x = np.array([image], dtype=np.float32)

    y_pred = model.predict(x, verbose=0)[0]  # shape: (num_classes,)

    topk_idx = np.argsort(-y_pred)[:4].tolist()
    topk_ids = [idx2id[int(ix)] for ix in topk_idx]

    preds5 = ["new_whale"] + topk_ids
    preds5 = preds5[:5]
    y_final.append(preds5)

    if pos % 1000 == 0:
        print(f"{100 * pos / len(test):0.2f}%")
    pos += 1

(len(y_final), y_final[0][:5]) if len(y_final) > 0 else (0, [])


## === cell 5
imageId = [os.path.basename(p) for p in test]
y_sub = [" ".join(preds) for preds in y_final]

submission = pd.DataFrame({"Image": imageId, "Id": y_sub})

submission = submission[["Image", "Id"]]
submission.to_csv("submission.csv", index=False)

submission.head()
