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

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import numpy as np
import pandas as pd

print("Listing ../input:", os.listdir("../input")[:20])


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        from importlib.metadata import version

        pb_ver = version("protobuf")
        print("Detected protobuf version:", pb_ver)
        if int(pb_ver.split(".")[0]) >= 5:
            print("Downgrading protobuf to 4.25.3 for TensorFlow compatibility...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
            os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        print("Protobuf compatibility check warning:", repr(e))


_ensure_protobuf_compatible()




## === cell 1
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout

print("TensorFlow:", tf.__version__)
print("Eager execution:", tf.executing_eagerly())




## === cell 2
BASE_DIR = "../input/aerial-cactus-identification"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test")
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.isdir(TRAIN_IMG_DIR), f"Missing train dir: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing test dir: {TEST_IMG_DIR}"
assert os.path.isfile(TRAIN_CSV_PATH), f"Missing train.csv: {TRAIN_CSV_PATH}"
assert os.path.isfile(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv: {SAMPLE_SUB_PATH}"

df = pd.read_csv(TRAIN_CSV_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print("train.csv:", df.shape, "sample_submission:", sample_sub.shape)
print(df.head())




## === cell 3
def _safe_list_images(img_dir):
    files = []
    for f in sorted(os.listdir(img_dir)):
        fp = os.path.join(img_dir, f)
        if os.path.isfile(fp) and f.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(f)
    return files


train_files = _safe_list_images(TRAIN_IMG_DIR)
test_files = _safe_list_images(TEST_IMG_DIR)

print("Train images found:", len(train_files), "Test images found:", len(test_files))
print("Example train files:", train_files[:3])
print("Example test files:", test_files[:3])

train_img_pathes = [os.path.join(TRAIN_IMG_DIR, img_id) for img_id in df["id"].tolist()]

missing_train = [p for p in train_img_pathes[:50] if not os.path.isfile(p)]
assert (
    len(missing_train) == 0
), f"Some train images are missing, e.g.: {missing_train[:3]}"




## === cell 4
img_size = 32


def read_and_prep_images(img_paths, img_height=img_size, img_width=img_size):
    img_load_batch_size = 900
    output = None

    for i in range(0, len(img_paths), img_load_batch_size):
        print("process batch %d/%d" % (i, len(img_paths)))
        batch_paths = img_paths[i : i + img_load_batch_size]

        batch_paths = [p for p in batch_paths if os.path.isfile(p)]

        tmp_imgs = [
            load_img(p, target_size=(img_height, img_width)) for p in batch_paths
        ]
        tmp_img_array = np.array(
            [img_to_array(img) for img in tmp_imgs], dtype=np.float32
        )

        if not isinstance(output, np.ndarray):
            output = preprocess_input(tmp_img_array)
        else:
            output = np.vstack((output, preprocess_input(tmp_img_array)))

    return output


train_imgs = read_and_prep_images(train_img_pathes)
print("train_imgs shape:", train_imgs.shape)




## === cell 5
num_classes = 2
out_y = tf.keras.utils.to_categorical(df["has_cactus"].values, num_classes)

print("One sample image shape:", train_imgs[0].shape)
print("Labels shape:", out_y.shape)

model = Sequential()
model.add(
    Conv2D(filters=50, kernel_size=(3, 3), input_shape=(32, 32, 3), activation="relu")
)
model.add(Dropout(0.5))
model.add(Conv2D(30, kernel_size=(3, 3), activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(54, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer="adam",
    metrics=["accuracy"],
)

model.summary()




## === cell 6
batch_size = int(17500 * 0.9 / 100)
batch_size = max(1, batch_size)
print("Using batch_size:", batch_size)

history = model.fit(
    train_imgs,
    out_y,
    batch_size=batch_size,
    epochs=4,
    validation_split=0.1,
    verbose=2,
)




## === cell 7
test_img_pathes = [os.path.join(TEST_IMG_DIR, f) for f in sorted(test_files)]
test_imgs = read_and_prep_images(test_img_pathes)
print("test_imgs shape:", test_imgs.shape)




## === cell 8
proba = model.predict(test_imgs, verbose=0)  # shape (N,2) softmax
has_cactus_proba = proba[:, 1].astype(float)

answer = pd.DataFrame(
    {
        "id": [os.path.basename(p) for p in test_img_pathes],
        "has_cactus": has_cactus_proba,
    }
)

answer = sample_sub[["id"]].merge(answer, on="id", how="left")
answer["has_cactus"] = answer["has_cactus"].fillna(0.5).astype(float)

print(answer.head())
print("Submission shape after merge:", answer.shape)
assert list(answer.columns) == ["id", "has_cactus"]
assert answer["id"].isna().sum() == 0




## === cell 9
answer.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", answer.shape)
print(answer.describe(include="all"))
print("Saved to:", os.path.abspath("submission.csv"))
