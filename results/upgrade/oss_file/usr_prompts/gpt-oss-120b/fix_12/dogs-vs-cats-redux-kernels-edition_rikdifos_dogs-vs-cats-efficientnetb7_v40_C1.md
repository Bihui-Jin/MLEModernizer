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
import os, gc, time, random, zipfile, re, warnings, pathlib

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = (
    "python"  # prevent protobuf AttributeError
)
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, mixed_precision
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB7
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
random.seed(558)
np.random.seed(558)
tf.random.set_seed(558)

tf.config.optimizer.set_jit(True)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

mixed_precision.set_global_policy("mixed_float16")

start = time.time()

IMG_WIDTH, IMG_HEIGHT = 128, 128
BATCH_SIZE = 64

possible_base_paths = [
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition",
    "/kaggle/working",
    os.getcwd(),
]
BASE_PATH = None
for p in possible_base_paths:
    if os.path.isdir(p):
        BASE_PATH = p
        break
if BASE_PATH is None:
    raise FileNotFoundError("Cannot locate base data directory.")


def find_train_dir(base):
    candidates = [
        base,
        os.path.join(base, "train"),
        os.path.join(base, "dogs-vs-cats-redux-kernels-edition", "train"),
        os.path.join(base, "train", "train"),
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            subfolders = [
                name
                for name in os.listdir(cand)
                if os.path.isdir(os.path.join(cand, name))
            ]
            if set(subfolders) >= {"cat", "dog"}:
                return cand
    return None


TRAIN_DIR = find_train_dir(BASE_PATH)
if TRAIN_DIR is None:
    raise FileNotFoundError("Cannot locate train directory with cat/dog subfolders.")


def find_test_dir(base):
    candidates = [
        os.path.join(base, "test"),
        os.path.join(base, "dogs-vs-cats-redux-kernels-edition", "test"),
        os.path.join(base, "test", "test"),
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            inner = os.path.join(cand, "test")
            if os.path.isdir(inner):
                return inner
            return cand
    return None


TEST_DIR = find_test_dir(BASE_PATH)
if TEST_DIR is None:
    raise FileNotFoundError("Cannot locate test directory.")




## === cell 1
train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["cat", "dog"],
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    shuffle=True,
    seed=558,
    validation_split=0.2,
    subset="training",
)

val_ds = image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["cat", "dog"],
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    shuffle=False,
    seed=558,
    validation_split=0.2,
    subset="validation",
)

normalizer = layers.Rescaling(1.0 / 255)

train_ds = (
    train_ds.map(lambda x, y: (normalizer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    val_ds.map(lambda x, y: (normalizer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)




## === cell 2
base_model = EfficientNetB7(
    weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
)

model = models.Sequential(
    [base_model, layers.GlobalAveragePooling2D(), layers.Dense(1, activation="sigmoid")]
)

opt = optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
model.summary()




## === cell 3
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
)

history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    callbacks=[early_stop, reduce_lr],
    verbose=2,
)




## === cell 4
val_labels = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
val_preds = model.predict(val_ds, verbose=0).ravel()
val_pred_class = (val_preds > 0.5).astype(int)

print(f"Validation Accuracy: {accuracy_score(val_labels, val_pred_class):.5f}")
print(f"Validation LogLoss: {log_loss(val_labels, val_preds):.5f}")
test_files = []
for root, _, files in os.walk(TEST_DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            test_files.append(os.path.join(root, f))


def extract_id(filepath):
    m = re.search(r"(\d+)", os.path.basename(filepath))
    return int(m.group(1)) if m else float("inf")


test_files.sort(key=extract_id)
test_ids = [extract_id(fp) for fp in test_files]

test_ds = tf.data.Dataset.from_tensor_slices(test_files)


def load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    img = tf.cast(img, tf.float32) / 255.0
    return img


test_ds = (
    test_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_pred = model.predict(test_ds, verbose=0).ravel()

submission = pd.DataFrame({"id": test_ids, "label": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission written to {submission_path}")
print(f"Runtime: {time.time() - start:.2f} seconds")
