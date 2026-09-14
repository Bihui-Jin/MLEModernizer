# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MultiLabelBinarizer

try:
    import tensorflow_addons as tfa  # noqa: F401
except Exception as e:
    tfa = None
    print("tensorflow_addons unavailable (safe to ignore):", repr(e))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)



## === cell 1
DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
train.head()



## === cell 2
h_target = 384
w_target = 384
batch_size = 32

CPU_COUNT = os.cpu_count() or 2
GEN_WORKERS = min(8, max(2, CPU_COUNT // 2))
GEN_MAX_QUEUE_SIZE = 4 * GEN_WORKERS



## === cell 3
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
mlb.fit(label_split)

classes = list(mlb.classes_)
print("Num classes:", len(classes))
print("Classes:", classes)



## === cell 4
train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.1,
)

test_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0
)

train_generator = train_data_generator.flow_from_dataframe(
    dataframe=train,
    directory=TRAIN_IMG_DIR,
    x_col="image",
    y_col="labels",
    target_size=(h_target, w_target),
    color_mode="rgb",
    classes=classes,  # enforce consistent class index mapping
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    seed=SEED,
    subset="training",
)

valid_generator = train_data_generator.flow_from_dataframe(
    dataframe=train,
    directory=TRAIN_IMG_DIR,
    x_col="image",
    y_col="labels",
    target_size=(h_target, w_target),
    color_mode="rgb",
    classes=classes,
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=False,
    seed=SEED,
    subset="validation",
)

test_generator = test_data_generator.flow_from_dataframe(
    submissions,
    directory=TEST_IMG_DIR,
    x_col="image",
    y_col=None,
    target_size=(h_target, w_target),
    color_mode="rgb",
    classes=None,
    class_mode=None,
    shuffle=False,
    batch_size=batch_size,
)




## === cell 5
def find_first_h5_model(search_root="../input"):
    preferred_roots = [
        "../input/plant-pathology-2021-fgvc8",
        "../input",
    ]

    exts = (".h5", ".hdf5")

    for root in preferred_roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for f in filenames:
                fl = f.lower()
                if fl.endswith(exts):
                    return os.path.join(dirpath, f)
            rel = os.path.relpath(dirpath, root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth >= 3:
                dirnames[:] = []

    for root, dirs, files in os.walk(search_root):
        for f in files:
            fl = f.lower()
            if fl.endswith(exts):
                return os.path.join(root, f)
    return None


model_path = find_first_h5_model("../input")
print("Discovered model path:", model_path)

model = None
if model_path is not None:
    try:
        model = keras.models.load_model(model_path, compile=False)
        print("Loaded model from:", model_path)
    except Exception as e:
        print(
            "Failed to load discovered model; will train fallback model. Error:",
            repr(e),
        )
        model = None

if model is None:
    inputs = keras.layers.Input(shape=(h_target, w_target, 3))
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(len(classes), activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )

    EPOCHS = 2

    model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=EPOCHS,
        verbose=1,
    )



## === cell 6
preds = model.predict(
    test_generator,
    verbose=1,
)
preds = np.asarray(preds)

if preds.ndim == 1:
    preds = preds.reshape(-1, len(classes))
elif (
    preds.ndim == 2
    and preds.shape[1] != len(classes)
    and preds.shape[0] == len(classes)
):
    preds = preds.T

print("preds shape:", preds.shape)
print("num test images:", len(submissions))
assert preds.shape[0] == len(
    submissions
), "Prediction count mismatch with submission rows"



## === cell 7
thresh = 0.2

healthy_idx = classes.index("healthy") if "healthy" in classes else None

argmax_all = np.argmax(preds, axis=1)
max_all = np.max(preds, axis=1)

out_labels = []
for i in range(len(submissions)):
    p = preds[i]
    picked = np.where(p >= thresh)[0].tolist()

    if len(picked) == 0:
        picked = [int(argmax_all[i])]

    if healthy_idx is not None:
        if int(argmax_all[i]) == healthy_idx and (
            p[healthy_idx] >= float(max_all[i]) - 1e-12
        ):
            picked = [healthy_idx]

        if len(picked) > 1 and healthy_idx in picked:
            picked = [k for k in picked if k != healthy_idx]
            if len(picked) == 0:
                picked = [healthy_idx]

    label_str = " ".join([classes[k] for k in picked])
    out_labels.append(label_str)

submissions["labels"] = out_labels

submissions["labels"] = submissions["labels"].fillna("").astype(str)
submissions.loc[submissions["labels"].str.strip() == "", "labels"] = "healthy"

submissions = submissions[["image", "labels"]]
submissions.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submissions.shape)
submissions.head()



## === cell 8
assert os.path.exists("submission.csv"), "submission.csv was not created"
chk = pd.read_csv("submission.csv")
print(chk.columns.tolist())
print(chk.head(3))
print("Unique label strings (sample):", chk["labels"].head(10).tolist())
