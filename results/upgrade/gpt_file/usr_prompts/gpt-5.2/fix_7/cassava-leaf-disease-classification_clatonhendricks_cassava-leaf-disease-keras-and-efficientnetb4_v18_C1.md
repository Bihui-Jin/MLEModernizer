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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

fastai==2.8.5
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
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.1000302206104563

# 6. Current score

0.65321

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.65583) has done: 'I remove the `fastai` import path that is currently crashing due to an incompatibility in this Kaggle environment, and replace the missing external `model.pkl` dependency with a small self-contained TensorFlow image classifier that trains from `train.csv` + `train_images` and then predicts on `test_images`. This keeps the pipeline end-to-end (no hidden inputs), fixes the `learn_inf`/`model.pkl` file-not-found issue, and guarantees a properly formatted `submission.csv` is written. I also ensure predictions are integer class IDs (0–4) and that the submission rows match `sample_submission.csv` ordering to avoid alignment issues. Since there is no current score, the goal is primarily to produce a valid submission reliably; the simple model should also achieve at least a non-trivial accuracy above the target baseline.'
- What this solution (achieved 0.65508) has done: 'The crash happens at `import tensorflow as tf` due to a known protobuf/TensorFlow compatibility issue in this environment (`MessageFactory.GetPrototype`). To fix it with minimal impact, I force TensorFlow to use the pure-Python protobuf implementation *before* importing TensorFlow, which avoids the failing C++ protobuf path. I also keep the rest of your pipeline (data loading, model, training loop, prediction, and submission formatting) unchanged, only renumbering cells to start from 1 and ensuring the submission is written as `submission.csv`. This should run end-to-end and produce a valid CSV submission.'
- What this solution (achieved 0.65471) has done: 'You’re hitting a TensorFlow import crash caused by a protobuf API mismatch (`MessageFactory.GetPrototype`) in this Kaggle image. The cleanest minimal fix is to pin protobuf to the pure‑Python v3 API by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` **and** `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3`, and to do it before any TensorFlow/protobuf-related import. I also add a small safety fallback: if TF still can’t import, the notebook still produce a valid `submission.csv` using the competition’s `sample_submission.csv` labels (runs end-to-end, score drop, but it guarantees a submission). No model/training logic is changed when TF imports successfully, so your score behavior remains essentially the same as before.'
- What this solution (achieved 0.64985) has done: 'I fix the TensorFlow import crash by forcing the pure‑Python protobuf implementation earlier (before any protobuf/TensorFlow modules can be loaded) and by actively clearing any already-imported `google.protobuf` modules from `sys.modules` before importing TensorFlow. This is a minimal, execution-unblocking change that keeps your model/training/inference logic identical when TensorFlow becomes available. I also keep the existing safe fallback that writes a valid `submission.csv` if TensorFlow still can’t import, ensuring the notebook always completes end-to-end. No score-targeting changes are needed since your current score (0.65471) is already far above the target.'
- What this solution (achieved 0.65508) has done: 'The immediate blocker is the TensorFlow import crash caused by an incompatible protobuf runtime (`MessageFactory.GetPrototype`), so I make the TensorFlow path robust by explicitly switching the protobuf runtime to the pure-Python implementation and forcing a safe reload order, then only proceed with TF if it imports cleanly. Since your current score (0.64985) is far above the target (0.1000), I avoid any model/training changes that could increase performance; instead, if TensorFlow cannot import, the fallback still produce a valid `submission.csv` deterministically. I also fix the missing `sys` import in the TF setup cell (it’s used there) to avoid NameError in environments where execution differs. The rest of the pipeline (data loading, model definition, training loop, inference, and submission formatting) is kept the same.'
- What this solution (achieved 0.65321) has done: 'I fix the TensorFlow/protobuf import crash that prevents training/inference from running by forcing the pure-Python protobuf implementation *and* proactively removing any already-imported protobuf modules before importing TensorFlow. If TensorFlow still cannot import, the script deterministically fall back to writing a valid `submission.csv` (using a constant label) so you always get an end-to-end run with a correctly formatted CSV. I keep your model, training loop, data pipeline, and submission formatting unchanged when TensorFlow imports successfully, so score behavior remains essentially the same (and already far above the target). I also renumber cells to start at 1 to match the required format.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os
import sys
import importlib

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf") or k.startswith("tensorflow"):
        del sys.modules[k]

import json
import glob

TF_AVAILABLE = True
try:
    import google.protobuf  # ensure it imports under the chosen implementation
    import tensorflow as tf

    print("TensorFlow:", tf.__version__)
    SEED = 42
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
except Exception as e:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow failed to import. Will fall back to a safe submission.")
    print("TF import error:", repr(e))
    SEED = 42
    np.random.seed(SEED)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"

train_csv_path = os.path.join(BASE_DIR, "train.csv")
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
train_img_dir = os.path.join(BASE_DIR, "train_images")
test_img_dir = os.path.join(BASE_DIR, "test_images")

assert os.path.exists(train_csv_path), train_csv_path
assert os.path.exists(sample_sub_path), sample_sub_path
assert os.path.isdir(train_img_dir), train_img_dir
assert os.path.isdir(test_img_dir), test_img_dir



## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json"), "r") as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=4))



## === cell 4
df_train = pd.read_csv(train_csv_path)
df_train["class_name"] = df_train["label"].astype(str).map(map_classes)

print(df_train.head())
print("Train rows:", len(df_train))
print("Class counts:\n", df_train["label"].value_counts().sort_index())



## === cell 5
sample_sub = pd.read_csv(sample_sub_path)

if not TF_AVAILABLE:
    submission = sample_sub.copy()
    submission["label"] = 0
    submission.to_csv("submission.csv", index=False)
    print(submission.head())
    print("Wrote submission.csv with shape:", submission.shape)
    with open("submission.csv", "r") as f:
        for _ in range(5):
            print(f.readline().strip())
else:
    IMG_SIZE = 224
    BATCH_SIZE = 32
    AUTOTUNE = tf.data.AUTOTUNE
    N_CLASSES = 5

    df_train["filepath"] = df_train["image_id"].apply(
        lambda x: os.path.join(train_img_dir, x)
    )
    missing = df_train.loc[~df_train["filepath"].apply(os.path.exists)]
    if len(missing) > 0:
        raise FileNotFoundError(
            f"Missing {len(missing)} training images, e.g. {missing.iloc[0]['filepath']}"
        )

    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df_train[["filepath", "label"]].copy(),
        test_size=0.15,
        random_state=SEED,
        stratify=df_train["label"],
    )

    def decode_and_resize(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(
            img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
        )
        img = tf.cast(img, tf.float32) / 255.0
        label = tf.cast(label, tf.int32)
        return img, label

    def make_ds(df, training: bool):
        paths = df["filepath"].values
        labels = df["label"].values.astype(np.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(
                buffer_size=min(len(df), 4096), seed=SEED, reshuffle_each_iteration=True
            )
        ds = ds.map(decode_and_resize, num_parallel_calls=AUTOTUNE)
        if training:

            def aug(img, label):
                img = tf.image.random_flip_left_right(img, seed=SEED)
                img = tf.image.random_flip_up_down(img, seed=SEED)
                return img, label

            ds = ds.map(aug, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
        return ds

    train_ds = make_ds(train_df, training=True)
    val_ds = make_ds(val_df, training=False)



## === cell 6
if TF_AVAILABLE:
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(N_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()



## === cell 7
if TF_AVAILABLE:
    EPOCHS = 5
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)



## === cell 8
if TF_AVAILABLE:
    sample_sub = pd.read_csv(sample_sub_path)
    test_paths = (
        sample_sub["image_id"].apply(lambda x: os.path.join(test_img_dir, x)).values
    )

    missing_test = [p for p in test_paths if not os.path.exists(p)]
    if missing_test:
        raise FileNotFoundError(
            f"Missing {len(missing_test)} test images, e.g. {missing_test[0]}"
        )

    def make_test_ds(paths):
        ds = tf.data.Dataset.from_tensor_slices(paths)

        def decode_only(path):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(
                img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
            )
            img = tf.cast(img, tf.float32) / 255.0
            return img

        ds = ds.map(decode_only, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
        return ds

    test_ds = make_test_ds(test_paths)
    probs = model.predict(test_ds, verbose=0)
    preds = np.argmax(probs, axis=1).astype(int)

    submission = pd.DataFrame(
        {"image_id": sample_sub["image_id"].values, "label": preds}
    )
    submission.to_csv("submission.csv", index=False)

    print(submission.head())
    print("Wrote submission.csv with shape:", submission.shape)

    with open("submission.csv", "r") as f:
        for _ in range(5):
            print(f.readline().strip())
