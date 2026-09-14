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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
imageio==2.37.0
imageio-ffmpeg==0.6.0
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
pillow==11.3.0
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
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.48912

# 6. Current score

0.57159

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.57159) has done: 'The timeout is dominated by slow Python-side image loading/augmentation in `ImageDataGenerator.flow_from_dataframe` plus 20 epochs of training; the model itself is relatively light because the ResNet50 backbone is frozen. To keep the exact same model, loss, epochs, and data semantics, the main speedup is to switch the input pipeline to `tf.data` with parallel JPEG decode/resize, caching, and prefetch so the GPU/CPU stays fed. We preserve the same resize, rescale, and horizontal flip behavior (random flip only for training) and the same train/val split and class mapping. We also eliminate redundant generator bookkeeping (`steps_per_epoch=len(...)`) by letting Keras infer steps from dataset cardinality (equivalent coverage, less overhead), while keeping determinism via seeds.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from PIL import Image

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)  # keep as-is; may help speed, score-neutral
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF:", tf.__version__)
print("Keras:", keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_file = "/kaggle/input/plant-pathology-2020-fgvc7/train.csv"
path = "/kaggle/input/plant-pathology-2020-fgvc7/images/"
sub_path = "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv"
test_file = "/kaggle/input/plant-pathology-2020-fgvc7/test.csv"



## === cell 2
df = pd.read_csv(train_file)
df.head()



## === cell 3
df_test = pd.read_csv(test_file)
df_test.head()



## === cell 4
colnames = df.columns.to_list()
colnames.remove("image_id")
colnames




## === cell 5
def get_label(row):
    if row["healthy"]:
        return "healthy"
    elif row["multiple_diseases"]:
        return "multiple_diseases"
    elif row["rust"]:
        return "rust"
    elif row["scab"]:
        return "scab"
    return "healthy"




## === cell 6
label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
df["label"] = df[label_cols].astype(np.int8).idxmax(axis=1)
df[["image_id", "label"]].head()



## === cell 7
df["file_name"] = df["image_id"].astype(str) + ".jpg"
df_test["file_name"] = df_test["image_id"].astype(str) + ".jpg"



## === cell 8
df_train, df_validate = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"]
)

print(f"Training Size : {len(df_train)}")
print(f"Validation Size : {len(df_validate)}")



## === cell 9
im = Image.open(os.path.join(path, "Train_1.jpg"))
width, height = im.size
print("Original size:", width, height)



## === cell 10
BATCH = 32

new_width = int(width / 1.5)
new_height = int(height / 1.5)

AUTOTUNE = tf.data.AUTOTUNE
CLASS_NAMES = ["healthy", "multiple_diseases", "rust", "scab"]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASS_NAMES)}


def _build_paths_and_labels(df_in, with_labels=True):
    file_paths = (path + df_in["file_name"].astype(str)).to_numpy()
    if not with_labels:
        return file_paths
    y_idx = df_in["label"].map(CLASS_TO_INDEX).astype(np.int32).to_numpy()
    y_oh = tf.one_hot(y_idx, depth=4, dtype=tf.float32)
    return file_paths, y_oh


def _decode_resize_rescale(fp):
    img = tf.io.read_file(fp)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _train_map(fp, y):
    img = _decode_resize_rescale(fp)
    img = tf.image.random_flip_left_right(img, seed=SEED)
    return img, y


def _eval_map(fp, y):
    img = _decode_resize_rescale(fp)
    return img, y


def _test_map(fp):
    img = _decode_resize_rescale(fp)
    return img


train_paths, train_y = _build_paths_and_labels(df_train, with_labels=True)
val_paths, val_y = _build_paths_and_labels(df_validate, with_labels=True)
test_paths = _build_paths_and_labels(df_test, with_labels=False)

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_y))
train_ds = train_ds.shuffle(
    buffer_size=len(df_train), seed=SEED, reshuffle_each_iteration=True
)
train_ds = train_ds.map(_train_map, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache()
train_ds = train_ds.batch(BATCH, drop_remainder=False)
train_ds = train_ds.prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_y))
val_ds = val_ds.map(_eval_map, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.cache()
val_ds = val_ds.batch(BATCH, drop_remainder=False)
val_ds = val_ds.prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(_test_map, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.cache()
test_ds = test_ds.batch(BATCH, drop_remainder=False)
test_ds = test_ds.prefetch(AUTOTUNE)



## === cell 11
base_model = ResNet50(weights="imagenet", include_top=False)
print("Base model layers:", len(base_model.layers))



## === cell 12
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation="relu")(x)
x = Dense(32, activation="relu")(x)
preds = Dense(4, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=preds)



## === cell 13
base_model.trainable = False



## === cell 14
n_epochs = 20
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["categorical_accuracy"],
)



## === cell 15
history = model.fit(
    train_ds,
    epochs=n_epochs,
    validation_data=val_ds,
    verbose=1,
)



## === cell 16
y_pred = model.predict(
    test_ds,
    verbose=1,
)
y_pred = y_pred[: len(df_test)]
print("Pred shape:", y_pred.shape)



## === cell 17
submit = pd.concat(
    [
        df_test[["image_id"]].reset_index(drop=True),
        pd.DataFrame(y_pred, columns=["healthy", "multiple_diseases", "rust", "scab"]),
    ],
    axis=1,
)

submit = submit[["image_id", "healthy", "multiple_diseases", "rust", "scab"]]
submit.head()



## === cell 18
out_path = "/kaggle/working/submit.csv"
submit.to_csv(out_path, index=False)
print("Wrote submission to", out_path, "with shape:", submit.shape)
print("Columns:", submit.columns.tolist())
