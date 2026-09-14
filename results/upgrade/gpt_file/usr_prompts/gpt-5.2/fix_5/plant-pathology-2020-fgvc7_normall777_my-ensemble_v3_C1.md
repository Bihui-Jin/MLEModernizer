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

No external packages required in the script and installed.

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

0.969447776568138

# 6. Current score

0.44

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.43208) has done: 'I make the notebook runnable in the Kaggle offline environment by removing the failing `pip install` and notebook magics, and by replacing the unauthenticated `KaggleDatasets().get_gcs_path()` logic with the local `/kaggle/input/...` image path. I also fix the protobuf-related crash by avoiding `kaggle_datasets` usage entirely, and ensure all variables (paths, labels, datasets) are defined before they’re referenced. For the EfficientNetB7 model, I switch to the built-in `tf.keras.applications.EfficientNetB7` (same architecture) to avoid the broken external EfficientNet weight URL, while keeping the overall two-model ensemble logic unchanged. Finally, I make the submission strictly match `sample_submission.csv` columns and write `submission.csv` successfully.'
- What this solution (achieved 0.43407) has done: 'I fix the immediate runtime crash by pinning protobuf to a compatible pure-Python implementation before importing TensorFlow (this is the root cause of the `MessageFactory.GetPrototype` error in many Kaggle TF images). Then I keep your model/ensemble logic intact but correct the output activation/loss pairing for a multi-label AUC competition (use `sigmoid` + `binary_crossentropy` instead of `softmax` + categorical CE), which is a minimal semantic alignment likely responsible for the low 0.432 score. Finally, I ensure predictions are written in exactly the `sample_submission.csv` column order and that the submission is a valid `.csv` file.'
- What this solution (achieved 0.44) has done: 'I fix the protobuf/TensorFlow crash by forcing the pure-Python protobuf implementation before TensorFlow import and by ensuring TensorFlow is imported only after that environment variable is set. Then I correct the model input preprocessing to match each pretrained backbone (EfficientNetB7 and DenseNet201), which is a minimal, metric-aligned change that should significantly improve ROC AUC compared to feeding only `/255.0` normalized images. I also make the random augmentations truly random per-sample (instead of using a fixed seed every time), which improves generalization without changing the overall training approach. Finally, I keep the ensemble/submission logic intact and continue writing `submission.csv` with columns exactly matching `sample_submission.csv`.'
- What this solution (achieved 0.44) has done: 'I fix the protobuf/TensorFlow import crash that currently prevents any training/inference by forcing a compatible pure-Python protobuf at process start and ensuring TensorFlow is imported only after that. I also make the dataset path resolution robust to either `/kaggle/input/...` or the provided `/kaggle/data/...` layout so image loading doesn’t silently fail. Core model/ensemble logic (EfficientNetB7 + DenseNet201, preprocessing, sigmoid+BCE, alpha blend, submission format) is kept the same to preserve semantics while restoring end-to-end execution. The script always write a valid `submission.csv` with columns in the exact `sample_submission.csv` order.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import random
import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import DenseNet201, EfficientNetB7

print("Tensorflow version " + tf.__version__)

SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
AUTO = tf.data.experimental.AUTOTUNE

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

EPOCHS = 40
BATCH_SIZE = 8 * strategy.num_replicas_in_sync



## === cell 2
CANDIDATE_DIRS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]

DATA_DIR = None
for d in CANDIDATE_DIRS:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
        os.path.join(d, "images")
    ):
        DATA_DIR = d
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate dataset directory containing train.csv and images/. "
        f"Tried: {CANDIDATE_DIRS}"
    )

IMAGES_DIR = os.path.join(DATA_DIR, "images")
print("Using DATA_DIR:", DATA_DIR)
print("Using IMAGES_DIR:", IMAGES_DIR)


def format_path(st):
    return os.path.join(IMAGES_DIR, st + ".jpg")




## === cell 3
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

TARGET_COLS = [c for c in sub.columns if c != "image_id"]

train_paths = train["image_id"].apply(format_path).values
test_paths = test["image_id"].apply(format_path).values

train_labels = train[TARGET_COLS].values.astype(np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths,
    train_labels,
    test_size=0.15,
    random_state=SEED,
    stratify=np.argmax(train_labels, axis=1),
)

print(
    "Train size:",
    len(train_paths),
    "Valid size:",
    len(valid_paths),
    "Test size:",
    len(test_paths),
)
print("Targets:", TARGET_COLS)



## === cell 4
img_size = 768


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(
        image, tf.float32
    )  # keep 0..255 float, model-specific preprocessing later
    if label is None:
        return image
    return image, label


def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if label is None:
        return image
    return image, label




## === cell 5
train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)

STEPS_PER_EPOCH = int(np.ceil(len(train_paths) / BATCH_SIZE))
VALIDATION_STEPS = int(np.ceil(len(valid_paths) / BATCH_SIZE))

print(
    "BATCH_SIZE:",
    BATCH_SIZE,
    "STEPS_PER_EPOCH:",
    STEPS_PER_EPOCH,
    "VALIDATION_STEPS:",
    VALIDATION_STEPS,
)




## === cell 6
def get_model(use_model):
    if use_model.__name__ == "EfficientNetB7":
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    elif use_model.__name__ == "DenseNet201":
        preprocess = tf.keras.applications.densenet.preprocess_input
    else:
        preprocess = None

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = inputs
    if preprocess is not None:
        x = tf.keras.layers.Lambda(preprocess, name=f"{use_model.__name__}_preprocess")(
            x
        )

    base_model = use_model(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    x = base_model(x)

    predictions = Dense(train_labels.shape[1], activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    return model


with strategy.scope():
    model1 = get_model(EfficientNetB7)

w1 = "/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7.h5"
if os.path.exists(w1):
    model1.load_weights(w1)
    print("Loaded model1 weights:", w1)
else:
    print("WARNING: model1 weights not found, using imagenet base weights only:", w1)



## === cell 7
with strategy.scope():
    model2 = get_model(DenseNet201)

w2 = "/kaggle/input/tf-zoo-models-on-tpu-densenet201/my_dense_net_201.h5"
if os.path.exists(w2):
    model2.load_weights(w2)
    print("Loaded model2 weights:", w2)
else:
    print("WARNING: model2 weights not found, using imagenet base weights only:", w2)



## === cell 8
best_alpha = 0.52

print("Вычисляем предсказания...")
probabilities1 = model1.predict(test_dataset, verbose=1)
probabilities2 = model2.predict(test_dataset, verbose=1)

probabilities = best_alpha * probabilities1 + (1.0 - best_alpha) * probabilities2
probabilities = np.clip(probabilities, 0.0, 1.0)

sub.loc[:, TARGET_COLS] = probabilities.astype(np.float32)

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(sub.head())
