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

3.8

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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

0.83966

# 6. Current score

0.73366

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.74784) has done: 'The timeout is driven by doing 50 epochs of Xception training while repeatedly decoding/resizing JPEGs and running augmentation; on CPU this is too slow. I keep the exact model, loss, optimizer, LR schedule, and 50-epoch loop, but make the input pipeline much faster by (1) removing expensive file-based `.cache()` (which can thrash disk) in favor of in-memory caching (dataset is small enough), (2) using `tf.io.decode_jpeg` with `dct_method='INTEGER_FAST'` (equivalent decode semantics, faster), (3) enabling parallel batching and map optimizations, and (4) setting deterministic-related flags without forcing slow deterministic ops globally. These changes are provably equivalent for training semantics (same samples/labels/augmentations/epochs) with only negligible floating-point differences from faster JPEG decode internals, and they target the bottleneck directly.'
- What this solution (achieved 0.71212) has done: 'I fix the immediate runtime crash caused by an incompatible protobuf implementation by forcing the pure-Python protobuf runtime before importing TensorFlow/Keras, which avoids the `MessageFactory.GetPrototype` AttributeError in this environment. I also make the dataset directory detection compatible with your provided `/kaggle/data/...` layout so the code reliably finds `train.csv` and `images/`. To move the score toward your target (while preserving the same model/training loop), I correct the learning-rate schedule bug where `LR_MAX` is accidentally *lower* than `LR_START`, which prevents proper warmup and tends to underfit. All other core logic (Xception, softmax head, categorical crossentropy, 50 epochs, data augmentation, submission format) remains the same.'
- What this solution (achieved 0.71958) has done: 'I fix the TensorFlow import crash by setting the protobuf environment variables early and forcing a safe protobuf version range before importing TensorFlow, which addresses the `MessageFactory.GetPrototype` incompatibility in this Kaggle image. I also make the dataset directory detection robust to both `/kaggle/input/...` and `/kaggle/data/...` layouts, but keep all paths and core training logic unchanged. To move the score toward your target (without changing the model/loop/epochs), I keep your corrected learning-rate schedule and ensure the LR scheduler actually drives the Adam learning rate by explicitly using a float `learning_rate` in the optimizer. Finally, I keep the submission column ordering aligned to `sample_submission.csv` and guarantee a valid `submission.csv` is written.'
- What this solution (achieved 0.73366) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf runtime and (critically) importing `google.protobuf` once before importing TensorFlow; this avoids the `MessageFactory.GetPrototype` AttributeError seen in this environment. I also make the dataset directory detection accept both layouts you have (`.../plant-pathology-2020-fgvc7/` and the nested `.../plant-pathology-2020-fgvc7/plant-pathology-2020-fgvc7/`) so it always finds `train.csv` and `images/`. These changes are execution/stability fixes only and keep your model, training loop, augmentation, and LR schedule unchanged, so any score change should be negligible aside from being able to run end-to-end. Finally, the script still write a valid `submission.csv` with the exact column order from `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import google.protobuf  # noqa: F401

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import Xception

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
kaggle_dir_candidates = [
    "/kaggle/input/plant-pathology-2020-fgvc7/",
    "/kaggle/data/plant-pathology-2020-fgvc7/",
    "/kaggle/input/",
    "/kaggle/data/",
]

expanded_candidates = []
for d in kaggle_dir_candidates:
    expanded_candidates.append(d)
    expanded_candidates.append(os.path.join(d, "plant-pathology-2020-fgvc7/"))

kaggle_dir = None
for d in expanded_candidates:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
        os.path.join(d, "images")
    ):
        kaggle_dir = d
        break

if kaggle_dir is None:
    raise FileNotFoundError(
        f"Could not find train.csv + images/ in any of: {expanded_candidates}"
    )

images_dir = os.path.join(kaggle_dir, "images")

sample_submission = pd.read_csv(os.path.join(kaggle_dir, "sample_submission.csv"))
test = pd.read_csv(os.path.join(kaggle_dir, "test.csv"))
train = pd.read_csv(os.path.join(kaggle_dir, "train.csv"))

AUTO = tf.data.experimental.AUTOTUNE

print("Using kaggle_dir:", kaggle_dir)
print("images_dir:", images_dir)
print("train:", train.shape, "test:", test.shape)
print("submission columns:", sample_submission.columns.tolist())



## === cell 2
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS:", strategy.num_replicas_in_sync)



## === cell 3
IMG_SIZE = 300


def seed_everything(seed=0):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


seed = 2048
seed_everything(seed)


def format_path(st):
    return os.path.join(images_dir, f"{st}.jpg")


label_cols = ["healthy", "multiple_diseases", "rust", "scab"]

train_paths = train["image_id"].apply(format_path).values
test_paths = test["image_id"].apply(format_path).values
train_labels = train[label_cols].values.astype(np.float32)

from sklearn.model_selection import train_test_split

SPLIT_VALIDATION = True
if SPLIT_VALIDATION:
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=0.15,
        random_state=seed,
        stratify=np.argmax(train_labels, axis=1),
    )


@tf.function
def decode_image(filename, label=None, IMG_SIZE=(IMG_SIZE, IMG_SIZE)):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, IMG_SIZE)
    if label is None:
        return image
    return image, label


@tf.function
def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if label is None:
        return image
    return image, label


print("Example path:", train_paths[0])



## === cell 4
BATCH_SIZE = 32

options = tf.data.Options()
options.experimental_deterministic = False
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True

train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .with_options(options)
    .shuffle(512, seed=seed, reshuffle_each_iteration=True)
    .map(decode_image, num_parallel_calls=AUTO)
    .cache()
    .repeat()
    .map(data_augment, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## === cell 5
LR_START = 1e-4
LR_MAX = 5e-4 * strategy.num_replicas_in_sync
LR_MIN = 1e-6
LR_RAMPUP_EPOCHS = 4
LR_SUSTAIN_EPOCHS = 6
LR_EXP_DECAY = 0.8


def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (
            epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        ) + LR_MIN
    return float(lr)


lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)



## === cell 6
print(train[label_cols].sum())
print("Label prevalence:", train[label_cols].mean(numeric_only=True).to_dict())



## === cell 7
tf.config.optimizer.set_jit(True)

with strategy.scope():
    Dense_net = Xception(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        weights="imagenet",
        include_top=False,
    )
    x = Dense_net.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(4, activation="softmax")(x)
    model = keras.Model(inputs=Dense_net.input, outputs=x)

    opt = tf.keras.optimizers.Adam(learning_rate=float(LR_START))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"],
        jit_compile=True,
    )

model.summary()



## === cell 8
steps_per_epoch = max(1, train_labels.shape[0] // BATCH_SIZE)
validation_steps = None
if SPLIT_VALIDATION:
    validation_steps = int(np.ceil(valid_labels.shape[0] / BATCH_SIZE))

history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=valid_dataset if SPLIT_VALIDATION else None,
    validation_steps=validation_steps if SPLIT_VALIDATION else None,
    callbacks=[lr_callback],
)



## === cell 9
pred = model.predict(test_dataset, verbose=1)
pred = np.clip(pred, 0.0, 1.0)

sub_cols = [c for c in sample_submission.columns if c != "image_id"]
col_to_idx = {c: i for i, c in enumerate(label_cols)}
pred_ordered = np.stack([pred[:, col_to_idx[c]] for c in sub_cols], axis=1)

submission = pd.DataFrame(pred_ordered, columns=sub_cols)
submission.insert(0, "image_id", test["image_id"].values)

assert submission.shape[0] == test.shape[0]
assert submission.columns.tolist() == ["image_id"] + sub_cols

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
