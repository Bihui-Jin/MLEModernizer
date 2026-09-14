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

0.96078

# 6. Current score

0.46843

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.56522) has done: 'The timeout is dominated by very expensive 512×512 decoding/resizing done every step for 50 epochs, plus input pipeline overhead from small buffers and repeated JPEG decode work. I keep the same model, loss, epochs, and augmentation logic, but make the tf.data pipeline faster by caching *decoded/resized* images once (then augmenting on-the-fly), enabling non-blocking map/batch behavior, and using larger shuffle buffering with deterministic options preserved. I also switch the decode functions to use `tf.image.decode_jpeg(..., dct_method="INTEGER_FAST")` (equivalent decode output semantics) and remove redundant tf.function wrappers that can increase retracing overhead. These changes reduce repeated CPU work and input stalls without changing training/evaluation meaning.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash caused by an incompatibility between TensorFlow 2.18 and the installed protobuf 6.x by forcing protobuf to use the pure-Python implementation before importing TensorFlow. I also make the input image directory resolution robust to both `/kaggle/input/...` and `/kaggle/data/...` layouts so the notebook consistently finds the JPG files. To prevent silent dataset/missing-file issues (which can destroy AUC), I add a fast path existence check and keep the rest of the model/training logic unchanged. Finally, the script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.58523) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x incompatibility by forcing the pure-Python protobuf implementation *and* monkey-patching the missing `MessageFactory.GetPrototype` method before importing TensorFlow. This is the minimal change needed to make the notebook run end-to-end again in TF 2.18 without changing your model/training logic. I also make the environment setup happen before any TensorFlow-related import and keep the rest of your pipeline identical so score behavior is driven by the same training semantics. Finally, I ensure the submission is always written as `submission.csv` with the required columns and row alignment.'
- What this solution (achieved 0.59201) has done: 'The crash happens before TensorFlow imports because the protobuf monkey-patch is attached to the `MessageFactory` *class*, but the error is raised from an already-created `MessageFactory` *instance* that still lacks `GetPrototype`. I fix this by patching both the class and the default instance (and only if needed), while keeping the same “force python protobuf” approach. This is an execution-unblocking fix and is score-neutral; once TF imports, the rest of your training/inference pipeline stays identical and produce a valid `submission.csv`.'
- What this solution (achieved 0.47386) has done: 'I fix the TensorFlow import crash by ensuring the protobuf `GetPrototype` monkey-patch is applied to the actual `MessageFactory` instance used by protobuf (not just the class), and I do it before importing TensorFlow. This is an execution-unblocking change and should be score-neutral while allowing the rest of your pipeline (same model, same training loop, same preprocessing) to run end-to-end. I also keep the TPU/strategy logic unchanged and make sure the notebook always writes a valid `submission.csv` with the required columns. No changes are made to architecture, loss, epochs, or augmentation semantics.'
- What this solution (achieved 0.48337) has done: 'I fix the protobuf/TensorFlow import crash by applying the `GetPrototype` monkey-patch to the *actual* MessageFactory class and default instance used by protobuf in this environment, before importing TensorFlow. This is an execution-unblocking change and should be score-neutral, allowing your existing model/training pipeline to run as intended. I also keep paths/columns consistent with the competition (notably `multiple_diseases`) and ensure `submission.csv` is always written with the right row order and columns. No changes are made to the model architecture, loss, epochs, or augmentation logic.'
- What this solution (achieved 0.46843) has done: 'I fix the TensorFlow import crash by applying the protobuf `GetPrototype` patch to the *actual* `MessageFactory` instance used by protobuf’s internal `_message` module, which is where the error is raised in this environment. This is an execution-unblocking change and is score-neutral (it doesn’t touch the model, data, or training). I also keep the rest of your pipeline identical and ensure `submission.csv` is always written with the required columns and correct row alignment.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf import message_factory as _mf_mod

    def _patch_getprototype(obj_or_cls):
        if hasattr(obj_or_cls, "GetPrototype"):
            return True

        if hasattr(obj_or_cls, "GetMessageClass"):

            def _GetPrototype(self, descriptor):
                return self.GetMessageClass(descriptor)

            try:
                setattr(obj_or_cls, "GetPrototype", _GetPrototype)
                return True
            except Exception:
                return False
        return False

    _patch_getprototype(_mf_mod.MessageFactory)

    for attr in ("message_factory", "default_factory", "_DEFAULT_FACTORY"):
        inst = getattr(_mf_mod, attr, None)
        if inst is not None:
            _patch_getprototype(inst)

    get_mf = getattr(_mf_mod, "GetMessageFactory", None)
    if callable(get_mf):
        try:
            _patch_getprototype(get_mf())
        except Exception:
            pass

    try:
        from google.protobuf.pyext import _message as _message_mod  # type: ignore

        mf_inst = getattr(_message_mod, "default_message_factory", None)
        if mf_inst is not None:
            _patch_getprototype(mf_inst)
    except Exception:
        pass

except Exception:
    pass

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0

tf.config.optimizer.set_jit(True)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU", tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

EPOCHS = 50
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
AUTO = tf.data.experimental.AUTOTUNE

CANDIDATE_DATA_DIRS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_DIR = None
for d in CANDIDATE_DATA_DIRS:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
        os.path.join(d, "images")
    ):
        DATA_DIR = d
        break
if DATA_DIR is None:
    DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"

IMAGES_DIR = os.path.join(DATA_DIR, "images")
print("DATA_DIR:", DATA_DIR)
print("IMAGES_DIR exists:", os.path.isdir(IMAGES_DIR))




## === cell 2
def seed_everything(seed=0):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def format_path(image_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"{image_id}.jpg")


seed_everything(2048)



## === cell 3
train_csv = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_csv = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
submission_csv = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

train_paths = train_csv.image_id.apply(format_path).values
test_paths = test_csv.image_id.apply(format_path).values

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
train_labels = train_csv[target_cols].values.astype(np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=2048
)

missing_train = [p for p in train_paths[:50] if not os.path.exists(p)]
missing_test = [p for p in test_paths[:50] if not os.path.exists(p)]
if missing_train or missing_test:
    raise FileNotFoundError(
        "Some image files were not found. Example missing paths:\n"
        f"train: {missing_train[:3]}\n"
        f"test: {missing_test[:3]}\n"
        f"IMAGES_DIR={IMAGES_DIR}"
    )



## === cell 4
IMAGE_SIZE = (512, 512)


def _decode_and_resize(filename):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, IMAGE_SIZE)
    return image


def decode_image(filename, label=None):
    image = _decode_and_resize(filename)
    if label is None:
        return image
    return image, label


def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if label is None:
        return image
    return image, label


def decode_and_augment(filename, label):
    image = _decode_and_resize(filename)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image, label




## === cell 5
options = tf.data.Options()
options.experimental_deterministic = True

SHUFFLE_BUFFER = min(2048, len(train_paths))

train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .cache()
    .shuffle(SHUFFLE_BUFFER, reshuffle_each_iteration=True)
    .repeat()
    .map(data_augment, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO, deterministic=True)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .cache()
    .batch(BATCH_SIZE, num_parallel_calls=AUTO, deterministic=True)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(options)
    .map(lambda f: decode_image(f, None), num_parallel_calls=AUTO, deterministic=True)
    .cache()
    .batch(BATCH_SIZE, num_parallel_calls=AUTO, deterministic=True)
    .prefetch(AUTO)
)



## === cell 6
with strategy.scope():
    base = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=[512, 512, 3]
    )
    model = tf.keras.Sequential(
        [
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(train_labels.shape[1], activation="sigmoid"),
        ]
    )

model.compile(
    optimizer="Adam",
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(curve="ROC", multi_label=True, name="auc")],
)



## === cell 7
STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE
VALIDATION_STEPS = int(np.ceil(valid_labels.shape[0] / BATCH_SIZE))

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=valid_dataset,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
)




## === cell 8
def display_training_curves(training, validation, title, subplot):
    """
    Source: https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu
    """
    if subplot % 10 == 1:
        plt.subplots(figsize=(10, 10), facecolor="#F0F0F0")
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor("#F8F8F8")
    ax.plot(training)
    ax.plot(validation)
    ax.set_title("model " + title)
    ax.set_ylabel(title)
    ax.set_xlabel("epoch")
    ax.legend(["train", "valid."])


display_training_curves(
    history.history["loss"],
    history.history["val_loss"],
    "loss",
    211,
)
display_training_curves(
    history.history["auc"],
    history.history["val_auc"],
    "auc",
    212,
)



## === cell 9
probs = model.predict(test_dataset, verbose=1)

assert probs.shape[0] == submission_csv.shape[0], (probs.shape, submission_csv.shape)
assert probs.shape[1] == len(target_cols), (probs.shape, len(target_cols))

submission_csv[target_cols] = probs
submission_csv.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_csv.shape)
submission_csv.head()
