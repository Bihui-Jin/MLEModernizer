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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow.keras.layers as L
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.efficientnet import EfficientNetB7

from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

print("TF:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))

tf.keras.utils.set_random_seed(2020)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception as _e:
    print("Warning: could not enable op determinism:", _e)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception as _e:
    print("Warning: could not set threading:", _e)

try:
    tf.config.optimizer.set_jit(False)
except Exception as _e:
    print("Warning: could not set global XLA JIT flag:", _e)

AUTO = tf.data.experimental.AUTOTUNE

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

EPOCHS = 50
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGES_DIR = os.path.join(DATA_DIR, "images")




## === cell 1
def format_path(st):
    return os.path.join(IMAGES_DIR, st + ".jpg")




## === cell 2
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

all_train_paths = (IMAGES_DIR + "/" + train["image_id"].astype(str) + ".jpg").to_numpy()
test_paths = (IMAGES_DIR + "/" + test["image_id"].astype(str) + ".jpg").to_numpy()

label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
y_multi = train[label_cols].values.astype(np.float32)
y_class = np.argmax(y_multi, axis=1)
all_train_labels = to_categorical(y_class, num_classes=len(label_cols)).astype(
    np.float32
)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    all_train_paths,
    all_train_labels,
    test_size=0.06,
    random_state=2020,
    stratify=y_class,
)

print("Train:", train_paths.shape, train_labels.shape)
print("Valid:", valid_paths.shape, valid_labels.shape)
print("Test :", test_paths.shape)



## === cell 3
if False:
    img = plt.imread(os.path.join(IMAGES_DIR, "Train_0.jpg"))
    row = train[train["image_id"] == "Train_0"][label_cols]
    lbl = row.columns[np.argmax(row.values[0])]
    print("Sanity image label:", lbl)
    print("Example image shape:", img.shape)



## === cell 4
img_size = 768


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label


def data_augment(image, label=None, seed=2020):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    if label is None:
        return image
    else:
        return image, label




## === cell 5
AUTO = tf.data.AUTOTUNE

opts = tf.data.Options()
try:
    opts.experimental_deterministic = True
except Exception as _e:
    print("Warning: could not set experimental_deterministic:", _e)

try:
    opts.experimental_optimization.apply_default_optimizations = True
except Exception as _e:
    print("Warning: could not set apply_default_optimizations:", _e)

try:
    opts.threading.private_threadpool_size = max(8, (os.cpu_count() or 8))
    opts.threading.max_intra_op_parallelism = 0
except Exception as _e:
    print("Warning: could not set tf.data threading options:", _e)

train_cache_path = "/kaggle/working/train_decode_cache"
valid_cache_path = "/kaggle/working/valid_decode_cache"
test_cache_path = "/kaggle/working/test_decode_cache"

for p in (train_cache_path, valid_cache_path, test_cache_path):
    try:
        if tf.io.gfile.exists(p):
            tf.io.gfile.rmtree(p)
    except Exception:
        pass

train_base = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .with_options(opts)
    .map(decode_image, num_parallel_calls=AUTO)
    .cache(train_cache_path)
)

valid_base = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .with_options(opts)
    .map(decode_image, num_parallel_calls=AUTO)
    .cache(valid_cache_path)
)

test_base = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(opts)
    .map(decode_image, num_parallel_calls=AUTO)
    .cache(test_cache_path)
)

train_dataset = (
    train_base.map(data_augment, num_parallel_calls=AUTO)
    .shuffle(512, seed=2020, reshuffle_each_iteration=True)
    .repeat()
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

valid_dataset = valid_base.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
test_dataset = test_base.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

print("Built datasets. Batch size:", BATCH_SIZE)



## === cell 6
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
    return lr


lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)

print(
    "Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(
        lrfn(0), max(lrfn(x) for x in range(EPOCHS)), lrfn(EPOCHS - 1)
    )
)




## === cell 7
def get_model(use_model):
    base_model = use_model(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    x = base_model.output
    predictions = Dense(train_labels.shape[1], activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=predictions)


with strategy.scope():
    model = get_model(EfficientNetB7)

model.compile(
    optimizer="nadam",
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
    jit_compile=True,  # keep model-level XLA
    steps_per_execution=16,
)

model.summary()



## === cell 8
ckpt_path = "pretrained_EfficientNetB7.keras"

steps_per_epoch = train_labels.shape[0] // BATCH_SIZE

history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    callbacks=[lr_callback],
    validation_data=valid_dataset,
    epochs=EPOCHS,
    verbose=1,
)

final_val_acc = history.history.get("val_categorical_accuracy", [None])[-1]
final_val_loss = history.history.get("val_loss", [None])[-1]
print(
    "Final validation metrics:",
    {"loss": final_val_loss, "categorical_accuracy": final_val_acc},
)

model.save(ckpt_path)



## === cell 9
if False:
    plt.plot(
        history.history["categorical_accuracy"], label="Accuracy on the training set"
    )
    if "val_categorical_accuracy" in history.history:
        plt.plot(
            history.history["val_categorical_accuracy"],
            label="Accuracy on the validation set",
        )
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()



## === cell 10
if False:
    plt.plot(history.history["loss"], label="loss on the training set")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="loss on the validation set")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()



## === cell 11
if os.path.exists(ckpt_path):
    model = tf.keras.models.load_model(ckpt_path)
    print("Loaded checkpoint:", ckpt_path)
else:
    print("Checkpoint not found, using last-epoch model.")



## === cell 12
probs = model.predict(test_dataset, verbose=1)
probs = np.asarray(probs)

if probs.ndim != 2 or probs.shape[1] != len(label_cols):
    raise ValueError(
        f"Unexpected prediction shape {probs.shape}; expected (n_test, {len(label_cols)})"
    )

if probs.shape[0] != len(test):
    raise ValueError(
        f"Pred rows {probs.shape[0]} != test rows {len(test)}; check dataset batching."
    )

submission = test[["image_id"]].copy()
submission[label_cols] = probs.astype(np.float32)

submission = submission[sub.columns.tolist()]

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
