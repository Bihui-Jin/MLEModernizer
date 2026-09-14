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

0.96833

# 6. Current score

0.52946

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.53882) has done: 'The timeout is overwhelmingly dominated by training EfficientNetB7 at 768×768 for 40 epochs; input pipeline tweaks alone won’t save enough time. To preserve the exact core model and training semantics while making runtime feasible, the key optimization is to precompute and cache the EfficientNetB7 “avg pooled” features once (for train/valid/test) and then train the same final Dense(sigmoid) head on those features for the same number of epochs/steps and with the same LR schedule/loss/metric. This is mathematically equivalent to training the head while keeping the base frozen (and it matches your current architecture’s split point exactly), but eliminates the expensive per-step convolutional forward/backward passes that cause the timeout. We also keep deterministic tf.data options and avoid repeated JPEG decode/resize by caching, and we keep the same checkpointing and submission logic.'
- What this solution (achieved 0.52946) has done: 'I fix the startup crash by disabling XLA JIT (the `MessageFactory.GetPrototype` protobuf error is triggered by `tf.config.optimizer.set_jit(True)` in this environment) while keeping determinism and the rest of the pipeline unchanged. I also make the checkpoint filename and reload robust across Keras 3/TF 2.18 by saving weights-only (avoids serialization edge cases) and explicitly rebuilding the head before loading weights. Finally, I add a small safety check to ensure predictions align with `test.csv`/`sample_submission.csv` row order and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.52946) has done: 'We fix the startup crash caused by a protobuf/TensorFlow/Keras import interaction that triggers `MessageFactory.GetPrototype` by setting the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` environment variable before importing TensorFlow. This change is score-neutral but unblocks the entire pipeline so training/inference can run and a valid `submission.csv` is always written. I also keep the existing feature-caching approach, model head, LR schedule, and training loop unchanged to preserve core logic and expected scoring behavior. Finally, I add a tiny safety fallback for `BASE_PATH` so it works in either `/kaggle/input/...` or `/kaggle/data/...` layouts without changing any I/O semantics.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow.keras.layers as L
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

print("TF version:", tf.__version__)
SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.keras.utils.set_random_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
AUTO = tf.data.experimental.AUTOTUNE

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU:", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS:", strategy.num_replicas_in_sync)

EPOCHS = 40
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

BASE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"
if not os.path.exists(BASE_PATH):
    alt = "/kaggle/data/plant-pathology-2020-fgvc7"
    if os.path.exists(alt):
        BASE_PATH = alt

IMAGES_DIR = os.path.join(BASE_PATH, "images")
print("BASE_PATH:", BASE_PATH)
print("IMAGES_DIR exists:", os.path.exists(IMAGES_DIR))




## === cell 2
def format_path(image_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"{image_id}.jpg")




## === cell 3
train = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
test = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
sub = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

train_paths_all = (IMAGES_DIR + "/" + train["image_id"].astype(str) + ".jpg").to_numpy()
test_paths = (IMAGES_DIR + "/" + test["image_id"].astype(str) + ".jpg").to_numpy()
train_labels_all = train[TARGET_COLS].values.astype(np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths_all,
    train_labels_all,
    test_size=0.06,
    random_state=SEED,
    shuffle=True,
)

print("Train/Valid sizes:", len(train_paths), len(valid_paths))
print("Example path exists:", train_paths[0], os.path.exists(train_paths[0]))



## === cell 4
if False:
    f, ax = plt.subplots(3, 6, figsize=(18, 7))
    ax = ax.flatten()
    for i in range(18):
        img_path = os.path.join(IMAGES_DIR, f"Train_{i}.jpg")
        if not os.path.exists(img_path):
            ax[i].axis("off")
            continue
        img = plt.imread(img_path)
        row = train[train["image_id"] == f"Train_{i}"][TARGET_COLS]
        if len(row) == 1:
            label_name = row.idxmax(axis=1).values[0]
        else:
            label_name = "unknown"
        ax[i].set_title(label_name)
        ax[i].imshow(img)
        ax[i].axis("off")
    plt.tight_layout()
    plt.show()



## === cell 5
img_size = 768


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    return image, label


def data_augment(image, label=None, seed=SEED):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    if label is None:
        return image
    return image, label




## === cell 6
options = tf.data.Options()
options.experimental_deterministic = True  # keep stable behavior
try:
    options.threading.private_threadpool_size = 16
except Exception:
    pass

train_base = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO)
    .cache()
)

train_dataset = (
    train_base.map(data_augment, num_parallel_calls=AUTO)
    .shuffle(512, seed=SEED, reshuffle_each_iteration=True)
    .repeat()
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)



## === cell 7
LR_START = 0.00001
LR_MAX = 0.0001 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
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

if False:
    rng = list(range(EPOCHS))
    y = [lrfn(x) for x in rng]
    plt.plot(rng, y)
    plt.title("Learning rate schedule")
    plt.show()
    print(
        "Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1])
    )




## === cell 8
def get_backbone():
    return tf.keras.applications.EfficientNetB7(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )


def get_head(num_features, num_labels):
    inputs = tf.keras.Input(shape=(num_features,), name="features")
    outputs = Dense(num_labels, activation="sigmoid", name="pred")(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="head")


with strategy.scope():
    backbone = get_backbone()
    backbone.trainable = False

num_features = int(backbone.output_shape[-1])
print("Backbone pooled feature dim:", num_features)




## === cell 9
def make_feature_ds(paths, labels=None, batch_size=BATCH_SIZE):
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths).with_options(options)
        ds = ds.map(decode_image, num_parallel_calls=AUTO)
        ds = ds.batch(batch_size).prefetch(AUTO)
        return ds
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(options)
    ds = ds.map(decode_image, num_parallel_calls=AUTO)
    ds = ds.batch(batch_size).prefetch(AUTO)
    return ds


train_feat_ds = make_feature_ds(train_paths, train_labels)
valid_feat_ds = make_feature_ds(valid_paths, valid_labels)
test_feat_ds = make_feature_ds(test_paths, labels=None)

train_features = backbone.predict(train_feat_ds, verbose=1)
valid_features = backbone.predict(valid_feat_ds, verbose=1)
test_features = backbone.predict(test_feat_ds, verbose=1)

train_features = train_features.astype(np.float32, copy=False)
valid_features = valid_features.astype(np.float32, copy=False)
test_features = test_features.astype(np.float32, copy=False)

print(
    "Feature arrays:", train_features.shape, valid_features.shape, test_features.shape
)



## === cell 10
feat_options = tf.data.Options()
feat_options.experimental_deterministic = True

train_head_ds = (
    tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    .with_options(feat_options)
    .shuffle(512, seed=SEED, reshuffle_each_iteration=True)
    .repeat()
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

valid_head_ds = (
    tf.data.Dataset.from_tensor_slices((valid_features, valid_labels))
    .with_options(feat_options)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_head_ds = (
    tf.data.Dataset.from_tensor_slices(test_features)
    .with_options(feat_options)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)



## === cell 11
ckpt_path = "my_ef_net_b7.weights.h5"

with strategy.scope():
    model = get_head(num_features=num_features, num_labels=train_labels.shape[1])
    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(
                multi_label=True, num_labels=len(TARGET_COLS), name="auc"
            )
        ],
        jit_compile=False,
    )

model.summary()

history = model.fit(
    train_head_ds,
    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
    validation_data=valid_head_ds,
    epochs=EPOCHS,
    callbacks=[
        lr_callback,
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ],
    verbose=1,
)



## === cell 12
if False:
    plt.plot(history.history.get("auc", []), label="AUC on the training set")
    plt.plot(history.history.get("val_auc", []), label="AUC on the validation set")
    plt.xlabel("epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.show()



## === cell 13
if False:
    plt.plot(history.history["loss"], label="loss on the training set")
    plt.plot(history.history["val_loss"], label="loss on the validation set")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()



## === cell 14
with strategy.scope():
    model = get_head(num_features=num_features, num_labels=train_labels.shape[1])
    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(
                multi_label=True, num_labels=len(TARGET_COLS), name="auc"
            )
        ],
        jit_compile=False,
    )
model.load_weights(ckpt_path)



## === cell 15
probs = model.predict(test_head_ds, verbose=1)
probs = np.clip(probs, 0.0, 1.0)

assert probs.shape[0] == len(
    test
), f"Prediction rows {probs.shape[0]} != test rows {len(test)}"
assert probs.shape[1] == len(
    TARGET_COLS
), f"Prediction cols {probs.shape[1]} != {len(TARGET_COLS)}"

submission = sub.copy()
submission = submission.set_index("image_id").loc[test["image_id"]].reset_index()
submission[TARGET_COLS] = probs

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
