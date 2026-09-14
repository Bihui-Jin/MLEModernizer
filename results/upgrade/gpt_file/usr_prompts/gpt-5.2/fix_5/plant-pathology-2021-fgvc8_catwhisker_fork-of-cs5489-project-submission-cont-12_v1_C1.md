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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.5919852262234518

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.29874) has done: 'Your notebook didn’t yield a score because it likely failed before writing a valid submission: the weights path points to a dataset that isn’t present in your `/kaggle/input/...` tree, and your code also iterates `os.listdir()` in arbitrary order which can misalign with `sample_submission.csv` expectations. I keep your model and thresholds unchanged, but add a safe fallback to load Xception ImageNet weights only if your custom weights aren’t found, so you always produce a submission and get a non-zero baseline score. I also generate the submission by iterating exactly over `sample_submission.csv["image"]` (correct row count/order), and ensure predictions are computed in inference mode (`training=False`) for stability. These are minimal changes aimed at producing a valid CSV and moving the score upward toward your target.'
- What this solution (achieved 0.29295) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by pinning `protobuf` to a TensorFlow-compatible version at runtime before importing TensorFlow; this is the root cause preventing any training/inference from running. Then we keep your exact model architecture and thresholding, but ensure Xception uses the correct preprocessing (`tf.keras.applications.xception.preprocess_input`) instead of `per_image_standardization`, which is a minimal metric-aligned fix that should improve score toward your target. Finally, we keep the stable submission ordering by iterating strictly over `sample_submission.csv["image"]` and always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import pandas as pd
import numpy as np

train_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/"
test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
print("train_images dir exists:", os.path.isdir(train_dir))
print("test_images dir exists:", os.path.isdir(test_dir))

training_csv = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
training_class = np.array([])
for labels in pd.unique(training_csv["labels"]):
    training_class = np.append(training_class, labels.split())
training_class = np.unique(training_class)
print("\nnumber of class")
print(training_class)
print(len(training_class))




## === cell 1
def predict2strings(pred, threshold):
    t = np.array(threshold)
    return " ".join(training_class[np.where(pred >= t)])


def predict2n_hot(pred, threshold):
    t = np.array(threshold)
    return np.where(pred >= t, np.ones(pred.shape), np.zeros(pred.shape))


def n_hot2string(n_hot):
    return " ".join(training_class[np.where(n_hot >= 1)])




## === cell 2
import pkgutil
import importlib


def _ensure_protobuf_compatible():
    try:
        import google.protobuf as protobuf  # noqa: F401
        import google.protobuf.__version__ as pbv  # type: ignore
    except Exception:
        pbv = None

    need_install = False
    try:
        import google.protobuf
        from packaging.version import Version

        v = Version(google.protobuf.__version__)
        if v.major >= 5:
            need_install = True
    except Exception:
        need_install = True

    if need_install:
        print("Pinning protobuf to 4.25.3 for TensorFlow compatibility...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        for m in list(sys.modules.keys()):
            if m.startswith("google.protobuf"):
                sys.modules.pop(m, None)


_ensure_protobuf_compatible()

import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)




## === cell 3
model_input = tf.keras.layers.Input(shape=(299, 299, 3))
xception_layer = tf.keras.applications.Xception(
    include_top=False, weights=None, input_shape=(299, 299, 3), input_tensor=model_input
).output
at = tf.keras.layers.Conv2D(2048, 3, padding="same", activation="relu")(xception_layer)
at = tf.keras.layers.BatchNormalization()(at)
at = tf.keras.layers.Conv2D(2048, 1, padding="same", activation="relu")(at)
at = tf.keras.layers.BatchNormalization()(at)
at = tf.keras.layers.Conv2D(1, 1, padding="same", activation="relu")(at)
attention_heatmap = tf.keras.layers.Softmax(name="attention_layer")(at)
attention = tf.keras.layers.Multiply()([attention_heatmap, xception_layer])
fusion_attention_spectrum = tf.keras.layers.Add()([attention, xception_layer])
fusion_attention_spectrum = tf.keras.layers.BatchNormalization()(
    fusion_attention_spectrum
)
fc_1 = tf.keras.layers.Flatten()(fusion_attention_spectrum)
fc_1 = tf.keras.layers.Dense(2048, activation="relu")(fc_1)
fc_1 = tf.keras.layers.BatchNormalization()(fc_1)
fc_2 = tf.keras.layers.Dense(4096 + 2048, activation="relu")(fc_1)
fc_2 = tf.keras.layers.BatchNormalization()(fc_2)
average_pooling = tf.keras.layers.GlobalAveragePooling2D()(fusion_attention_spectrum)
fc_2 = tf.keras.layers.concatenate([fc_2, average_pooling])
model_output = tf.keras.layers.Dense(len(training_class), activation="sigmoid")(fc_2)
model = tf.keras.Model(inputs=model_input, outputs=model_output)




## === cell 4
weights_path = "../input/fork-of-cs5489-project-train-cont-12/my_model/attention_1"

loaded = False
try:
    if tf.io.gfile.exists(weights_path) or tf.io.gfile.exists(weights_path + ".index"):
        model.load_weights(weights_path)
        loaded = True
        print(f"Loaded custom weights from: {weights_path}")
except Exception as e:
    print(
        "Failed to load custom weights; will fall back to ImageNet init for backbone."
    )
    print("Load error:", repr(e))

if not loaded:
    model_input = tf.keras.layers.Input(shape=(299, 299, 3))
    xception_layer = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(299, 299, 3),
        input_tensor=model_input,
    ).output
    at = tf.keras.layers.Conv2D(2048, 3, padding="same", activation="relu")(
        xception_layer
    )
    at = tf.keras.layers.BatchNormalization()(at)
    at = tf.keras.layers.Conv2D(2048, 1, padding="same", activation="relu")(at)
    at = tf.keras.layers.BatchNormalization()(at)
    at = tf.keras.layers.Conv2D(1, 1, padding="same", activation="relu")(at)
    attention_heatmap = tf.keras.layers.Softmax(name="attention_layer")(at)
    attention = tf.keras.layers.Multiply()([attention_heatmap, xception_layer])
    fusion_attention_spectrum = tf.keras.layers.Add()([attention, xception_layer])
    fusion_attention_spectrum = tf.keras.layers.BatchNormalization()(
        fusion_attention_spectrum
    )
    fc_1 = tf.keras.layers.Flatten()(fusion_attention_spectrum)
    fc_1 = tf.keras.layers.Dense(2048, activation="relu")(fc_1)
    fc_1 = tf.keras.layers.BatchNormalization()(fc_1)
    fc_2 = tf.keras.layers.Dense(4096 + 2048, activation="relu")(fc_1)
    fc_2 = tf.keras.layers.BatchNormalization()(fc_2)
    average_pooling = tf.keras.layers.GlobalAveragePooling2D()(
        fusion_attention_spectrum
    )
    fc_2 = tf.keras.layers.concatenate([fc_2, average_pooling])
    model_output = tf.keras.layers.Dense(len(training_class), activation="sigmoid")(
        fc_2
    )
    model = tf.keras.Model(inputs=model_input, outputs=model_output)
    print("Initialized Xception backbone with ImageNet weights (fallback).")




## === cell 5
preprocess = tf.keras.applications.xception.preprocess_input
AUTOTUNE = tf.data.AUTOTUNE

train_img_path = "../input/plant-pathology-2021-fgvc8/train_images/"
test_img_path = "../input/plant-pathology-2021-fgvc8/test_images/"

class2idx = {c: i for i, c in enumerate(training_class.tolist())}


def labels_to_multi_hot(labels_str):
    y = np.zeros((len(training_class),), dtype=np.float32)
    for lab in str(labels_str).split():
        if lab in class2idx:
            y[class2idx[lab]] = 1.0
    return y


df = training_csv.copy()
df["y"] = df["labels"].apply(labels_to_multi_hot)

rng = np.random.RandomState(42)
perm = rng.permutation(len(df))
df = df.iloc[perm].reset_index(drop=True)

val_frac = 0.1
val_n = int(len(df) * val_frac)
val_df = df.iloc[:val_n].reset_index(drop=True)
trn_df = df.iloc[val_n:].reset_index(drop=True)


def decode_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299), method="bilinear")
    img = tf.cast(img, tf.float32)
    img = preprocess(img)
    return img


def make_ds(dataframe, training):
    paths = (train_img_path + dataframe["image"].values).astype("U")
    ys = np.asarray(list(dataframe["y"].values), dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, ys))

    def _map(p, y):
        return decode_and_preprocess(p), y

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(2048, seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(16).prefetch(AUTOTUNE)
    return ds


if not loaded:
    trn_ds = make_ds(trn_df, training=True)
    val_ds = make_ds(val_df, training=False)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    )

    model.fit(
        trn_ds,
        validation_data=val_ds,
        epochs=2,
        verbose=2,
    )




## === cell 6
threshold = [
    0.9151549935340881,
    0.028693564236164093,
    0.22229868173599243,
    0.987423300743103,
    0.18797093629837036,
    0.6855853796005249,
]

sample_sub = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")


def write_csv_kaggle_tags():
    import csv

    fnames = sample_sub["image"].values.astype("U")
    paths = (test_img_path + fnames).astype("U")

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(decode_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(64).prefetch(AUTOTUNE)

    preds = model.predict(ds, verbose=0)
    labels_out = [
        predict2strings(preds[i], threshold=threshold) for i in range(len(fnames))
    ]

    tmp = [["image", "labels"]]
    tmp.extend([[fnames[i], labels_out[i]] for i in range(len(fnames))])

    with open("submission.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(tmp)


write_csv_kaggle_tags()

sub_check = pd.read_csv("submission.csv")
print(sub_check.head())
print("submission rows:", len(sub_check), "expected:", len(sample_sub))
print("submission columns:", list(sub_check.columns))
print("Saved to:", os.path.abspath("submission.csv"))

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
UFuncTypeError                            Traceback (most recent call last)
/tmp/ipykernel_55/662348995.py in <cell line: 0>()
     38 
     39 
---> 40 write_csv_kaggle_tags()
     41 
     42 sub_check = pd.read_csv("submission.csv")

/tmp/ipykernel_55/662348995.py in write_csv_kaggle_tags()
     18     # and lets TF parallelize I/O + run fewer, larger model calls.
     19     fnames = sample_sub["image"].values.astype("U")
---> 20     paths = (test_img_path + fnames).astype("U")
     21 
     22     ds = tf.data.Dataset.from_tensor_slices(paths)

UFuncTypeError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U48'), dtype('<U20')) -> None
