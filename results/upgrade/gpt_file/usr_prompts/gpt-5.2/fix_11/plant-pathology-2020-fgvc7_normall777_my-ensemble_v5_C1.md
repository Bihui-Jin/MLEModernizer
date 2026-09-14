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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import DenseNet201, InceptionResNetV2
from tensorflow.keras.applications import EfficientNetB7

from tensorflow.keras.applications.efficientnet import (
    preprocess_input as effnet_preprocess,
)
from tensorflow.keras.applications.densenet import (
    preprocess_input as densenet_preprocess,
)
from tensorflow.keras.applications.inception_resnet_v2 import (
    preprocess_input as irv2_preprocess,
)

print("Tensorflow version " + tf.__version__)

SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.config.run_functions_eagerly(False)



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
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

EPOCHS = 40
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

INPUT_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

assert os.path.isdir(IMAGES_DIR), f"Images directory not found: {IMAGES_DIR}"




## === cell 2
def format_path(st):
    return os.path.join(IMAGES_DIR, st + ".jpg")




## === cell 3
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

target_cols = [c for c in train.columns if c != "image_id"]
assert target_cols == [
    "healthy",
    "multiple_diseases",
    "rust",
    "scab",
], f"Unexpected target columns: {target_cols}"

assert (
    list(sub.columns) == ["image_id"] + target_cols
), f"Unexpected submission columns: {list(sub.columns)}"

train_paths = train.image_id.apply(format_path).values
test_paths = test.image_id.apply(format_path).values

train_labels = train[target_cols].values.astype(np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=SEED
)

assert len(train_paths) == len(train_labels)
assert len(valid_paths) == len(valid_labels)
assert os.path.isfile(
    train_paths[0]
), f"Example train image not found: {train_paths[0]}"
assert os.path.isfile(test_paths[0]), f"Example test image not found: {test_paths[0]}"



## === cell 4
img_size = 768


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, image_size)

    if label is None:
        return image
    else:
        return image, label


def data_augment(image, label=None, seed=SEED):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)

    if label is None:
        return image
    else:
        return image, label


def preprocess_for(model_name, image, label=None):
    if model_name == "efficientnetb7":
        x = effnet_preprocess(image)
    elif model_name == "densenet201":
        x = densenet_preprocess(image)
    elif model_name == "inceptionresnetv2":
        x = irv2_preprocess(image)
    else:
        x = image
    if label is None:
        return x
    return x, label




## === cell 5
_DATA_CACHE_DIR = "/kaggle/working/tfdata_cache_pp2020"
os.makedirs(_DATA_CACHE_DIR, exist_ok=True)


def _cache_path(name: str) -> str:
    return os.path.join(_DATA_CACHE_DIR, name)


def make_raw_dataset(paths, labels=None, cache_name=None):
    opt = tf.data.Options()
    opt.deterministic = True

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths).with_options(opt)
        ds = ds.map(lambda p: decode_image(p, None), num_parallel_calls=AUTO)
        if cache_name is not None:
            ds = ds.cache(_cache_path(cache_name))
        return ds

    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(opt)
    ds = ds.map(decode_image, num_parallel_calls=AUTO)
    if cache_name is not None:
        ds = ds.cache(_cache_path(cache_name))
    return ds


def make_dataset_from_raw(raw, labels_present: bool, training: bool, model_name: str):
    ds = raw

    if training:
        if labels_present:
            ds = ds.shuffle(512, seed=SEED, reshuffle_each_iteration=True)
        else:
            ds = ds.shuffle(512, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.repeat()

        if labels_present:
            ds = ds.map(data_augment, num_parallel_calls=AUTO)
        else:
            ds = ds.map(lambda img: data_augment(img, None), num_parallel_calls=AUTO)

    if labels_present:
        ds = ds.map(
            lambda img, y: preprocess_for(model_name, img, y), num_parallel_calls=AUTO
        )
        ds = ds.batch(BATCH_SIZE, drop_remainder=bool(training))
        return ds.prefetch(AUTO)

    ds = ds.map(
        lambda img: preprocess_for(model_name, img, None), num_parallel_calls=AUTO
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    return ds.prefetch(AUTO)




## === cell 6
def get_model(use_model, weights):
    base_model = use_model(
        weights=weights,
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    x = base_model.output
    predictions = Dense(train_labels.shape[1], activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer="nadam", loss="binary_crossentropy", metrics=["AUC"])
    return model




## === cell 7
train_steps = int(np.ceil(len(train_paths) / BATCH_SIZE))
valid_steps = int(np.ceil(len(valid_paths) / BATCH_SIZE))

raw_train = make_raw_dataset(train_paths, train_labels, cache_name="train_raw_decode")
raw_valid = make_raw_dataset(valid_paths, valid_labels, cache_name="valid_raw_decode")
raw_test = make_raw_dataset(test_paths, labels=None, cache_name="test_raw_decode")

train_ds_ef = make_dataset_from_raw(
    raw_train, labels_present=True, training=True, model_name="efficientnetb7"
)
valid_ds_ef = make_dataset_from_raw(
    raw_valid, labels_present=True, training=False, model_name="efficientnetb7"
)

train_ds_dn = make_dataset_from_raw(
    raw_train, labels_present=True, training=True, model_name="densenet201"
)
valid_ds_dn = make_dataset_from_raw(
    raw_valid, labels_present=True, training=False, model_name="densenet201"
)

train_ds_ir = make_dataset_from_raw(
    raw_train, labels_present=True, training=True, model_name="inceptionresnetv2"
)
valid_ds_ir = make_dataset_from_raw(
    raw_valid, labels_present=True, training=False, model_name="inceptionresnetv2"
)


def find_weight_file(candidates):
    for p in candidates:
        if p and tf.io.gfile.exists(p):
            return p
    for p in candidates:
        if not p:
            continue
        d = os.path.dirname(p)
        base = os.path.basename(p)
        if tf.io.gfile.exists(d):
            matches = tf.io.gfile.glob(os.path.join(d, base))
            if matches:
                return matches[0]
    return None


def find_weight_file_smart(model_key: str):
    if model_key == "efficientnetb7":
        explicit = [
            "/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7.h5",
            "/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7",
            "/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/efnetb7.h5",
        ]
        patterns = [
            "**/*ef*net*b7*.h5",
            "**/*efficient*net*b7*.h5",
            "**/*my_ef_net_b7*",
        ]
    elif model_key == "densenet201":
        explicit = [
            "/kaggle/input/tf-zoo-models-on-tpu-densenet201/my_dense_net_201.h5",
            "/kaggle/input/tf-zoo-models-on-tpu-densenet201/my_dense_net_201",
            "/kaggle/input/tf-zoo-models-on-tpu-densenet201/densenet201.h5",
        ]
        patterns = [
            "**/*dense*net*201*.h5",
            "**/*densenet201*.h5",
            "**/*my_dense_net_201*",
        ]
    elif model_key == "inceptionresnetv2":
        explicit = [
            "/kaggle/input/tf-zoo-models-on-tpu-inceptionresnetv2/my_model.h5",
            "/kaggle/input/tf-zoo-models-on-tpu-inceptionresnetv2/my_model",
            "/kaggle/input/tf-zoo-models-on-tpu-inceptionresnetv2/inceptionresnetv2.h5",
        ]
        patterns = [
            "**/*inception*resnet*v2*.h5",
            "**/*inceptionresnetv2*.h5",
            "**/*my_model*.h5",
        ]
    else:
        explicit, patterns = [], []

    p = find_weight_file(explicit)
    if p is not None:
        return p

    for pat in patterns:
        matches = tf.io.gfile.glob(os.path.join("/kaggle/input", pat))
        if matches:
            matches = sorted(matches)
            return matches[0]
    return None


with strategy.scope():
    model1 = get_model(EfficientNetB7, weights="imagenet")
loaded1 = False
try:
    w1 = find_weight_file_smart("efficientnetb7")
    if w1 is None:
        raise FileNotFoundError(
            "EfficientNetB7 finetuned weights not found in expected locations."
        )
    model1.load_weights(w1)
    loaded1 = True
    print("Loaded weights for model1 from:", w1)
except Exception as e:
    print(f"Could not load model1 weights, using imagenet init. Reason: {e}")

with strategy.scope():
    model2 = get_model(DenseNet201, weights="imagenet")
loaded2 = False
try:
    w2 = find_weight_file_smart("densenet201")
    if w2 is None:
        raise FileNotFoundError(
            "DenseNet201 finetuned weights not found in expected locations."
        )
    model2.load_weights(w2)
    loaded2 = True
    print("Loaded weights for model2 from:", w2)
except Exception as e:
    print(f"Could not load model2 weights, using imagenet init. Reason: {e}")

with strategy.scope():
    model3 = get_model(InceptionResNetV2, weights="imagenet")
loaded3 = False
try:
    w3 = find_weight_file_smart("inceptionresnetv2")
    if w3 is None:
        raise FileNotFoundError(
            "InceptionResNetV2 finetuned weights not found in expected locations."
        )
    model3.load_weights(w3)
    loaded3 = True
    print("Loaded weights for model3 from:", w3)
except Exception as e:
    print(f"Could not load model3 weights, using imagenet init. Reason: {e}")

if not loaded1:
    print(
        "Training model1 (EfficientNetB7) because finetuned weights were not found..."
    )
    model1.fit(
        train_ds_ef,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_data=valid_ds_ef,
        validation_steps=valid_steps,
        verbose=1,
    )

if not loaded2:
    print("Training model2 (DenseNet201) because finetuned weights were not found...")
    model2.fit(
        train_ds_dn,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_data=valid_ds_dn,
        validation_steps=valid_steps,
        verbose=1,
    )

if not loaded3:
    print(
        "Training model3 (InceptionResNetV2) because finetuned weights were not found..."
    )
    model3.fit(
        train_ds_ir,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_data=valid_ds_ir,
        validation_steps=valid_steps,
        verbose=1,
    )



## === cell 8
best_alpha = 0.36
bad_alpha = 0.30

print("Вычисляем предсказания...")

test_ds_1 = make_dataset_from_raw(
    raw_test, labels_present=False, training=False, model_name="efficientnetb7"
)
test_ds_2 = make_dataset_from_raw(
    raw_test, labels_present=False, training=False, model_name="densenet201"
)
test_ds_3 = make_dataset_from_raw(
    raw_test, labels_present=False, training=False, model_name="inceptionresnetv2"
)

probabilities1 = model1.predict(test_ds_1, verbose=1)
probabilities2 = model2.predict(test_ds_2, verbose=1)
probabilities3 = model3.predict(test_ds_3, verbose=1)

probabilities = (
    best_alpha * probabilities1
    + bad_alpha * probabilities2
    + (1 - best_alpha - bad_alpha) * probabilities3
)

probabilities = np.clip(probabilities, 0.0, 1.0)

sub = sub.copy()
sub["image_id"] = test["image_id"].values
sub[target_cols] = probabilities.astype(np.float32)

sub = sub[["image_id"] + target_cols]

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
assert os.path.isfile("submission.csv") and os.path.getsize("submission.csv") > 0
