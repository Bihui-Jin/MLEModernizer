# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.8

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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import glob
import re
import random
import time
import gc
import concurrent.futures

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

start = time.time()
random.seed(558)



## === cell 1
BASE_PATH = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")

train_images = glob.glob(os.path.join(TRAIN_DIR, "**", "*.jpg"), recursive=True)
test_images = glob.glob(os.path.join(TEST_DIR, "**", "*.jpg"), recursive=True)


def txt_dig(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [txt_dig(c) for c in re.split(r"(\d+)", text)]


train_images.sort(key=natural_keys)
test_images.sort(key=natural_keys)

random.shuffle(train_images)  # shuffle training order



## === cell 2
IMG_WIDTH, IMG_HEIGHT = 128, 128
CACHE_DIR = "image_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def load_resize(path):
    """Read an image, resize it, and return a uint8 array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    return cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)


def load_images(paths, cache_name):
    """Load and resize images using a thread pool, caching on‑disk."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path, mmap_mode="r")
    max_workers = os.cpu_count() or 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        imgs = list(executor.map(load_resize, paths))
    arr = np.array(imgs, dtype=np.uint8)
    np.save(cache_path, arr)
    return arr


x = load_images(train_images, "train_images")
test = load_images(test_images, "test_images")
y = np.array(
    [1 if "dog" in os.path.basename(p) else 0 for p in train_images], dtype=np.uint8
)

gc.collect()
print("Train shape:", x.shape, "Test shape:", test.shape, "Labels shape:", y.shape)



## === cell 3
sns.countplot(x=y)
plt.title("Label distribution")
plt.show()
plt.close()



## === cell 4
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=2020, stratify=y
)

x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0

ns, h, w, c = x_train.shape
x_train_flat = x_train.reshape(ns, h * w * c)
x_val_flat = x_val.reshape(x_val.shape[0], h * w * c)



## === cell 5
logreg = LogisticRegression(
    solver="saga",
    max_iter=100,
    n_jobs=-1,
    C=1.0,
    penalty="l2",
    random_state=2020,
)
logreg.fit(x_train_flat, y_train)

val_preds = logreg.predict_proba(x_val_flat)[:, 1]
val_pred_cls = (val_preds > 0.5).astype(int)

print(f"Out‑of‑Fold Accuracy: {accuracy_score(y_val, val_pred_cls):.5f}")
print(f"Out‑of‑Fold Log Loss: {log_loss(y_val, val_preds):.5f}")



## === cell 6
test_norm = test.astype("float32") / 255.0
test_flat = test_norm.reshape(test_norm.shape[0], h * w * c)
test_pred = logreg.predict_proba(test_flat)[:, 1]

submission = pd.DataFrame(
    {"id": np.arange(1, len(test_images) + 1), "label": test_pred}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission written to {submission_path}")
print("Total runtime: {:.2f} seconds".format(time.time() - start))
