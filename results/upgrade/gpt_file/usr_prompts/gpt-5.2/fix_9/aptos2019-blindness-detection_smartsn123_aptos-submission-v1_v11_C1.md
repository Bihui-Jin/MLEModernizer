# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import glob
import random
from copy import deepcopy
from functools import lru_cache

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)

try:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

print("Listing ../input:")
print(os.listdir("../input")[:20])




## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("tf.keras:", keras.__version__)




## === cell 2
plt.rcParams.update({"axes.titlesize": "small"})




## === cell 3
IMG_SIZE = 128




## === cell 4
DATA_DIR = "../input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

print("Train csv exists:", os.path.exists(TRAIN_CSV))
print("Test csv exists:", os.path.exists(TEST_CSV))
print("Train img dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test img dir exists:", os.path.isdir(TEST_IMG_DIR))




## === cell 5
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
print(train_df.head())
print(test_df.head())




## === cell 6
labels_df = train_df.set_index("id_code")
print("Num train:", len(labels_df))
print(labels_df["diagnosis"].value_counts())




## === cell 7
images = [f for f in glob.glob(os.path.join(TRAIN_IMG_DIR, "*.png"))]
labels = [
    int(labels_df.loc[os.path.splitext(os.path.basename(f))[0], "diagnosis"])
    for f in images
]

print("Found train images:", len(images))
print("Found labels:", len(labels))




## === cell 8
n = len(images)
k = min(6000, n)  # keeps "about 6000" intent while remaining valid
perm = np.arange(n)
rng = np.random.RandomState(SEED)
rng.shuffle(perm)

train_mask = np.zeros(n, dtype=bool)
train_mask[perm[:k]] = True

images_arr = np.array(images, dtype=object)
labels_arr = np.array(labels, dtype=np.int64)

train_images = images_arr[train_mask].tolist()
train_labels = labels_arr[train_mask].tolist()
test_images = images_arr[~train_mask].tolist()
test_labels = labels_arr[~train_mask].tolist()

print(len(train_labels), len(train_images))
print(len(test_labels), len(test_images))




## === cell 9
for img, lb in list(zip(images[:5], labels[:5])):
    print(img, lb)




## === cell 10
label_text = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

try:
    images_to_display = []
    for lb in range(5):
        candidates = [
            (images[ix], labels[ix]) for ix in range(len(images)) if labels[ix] == lb
        ]
        if len(candidates) > 0:
            images_to_display += random.sample(candidates, k=min(3, len(candidates)))

    fig = plt.figure(figsize=(12, 8))
    for ii, (img_path, label) in enumerate(images_to_display):
        ax = fig.add_subplot(2, 8, ii + 1, xticks=[], yticks=[])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(label_text[label])
    plt.show()
except Exception as e:
    print("Plot skipped:", repr(e))




## === cell 11
_GAMMA_GRID = np.round(np.linspace(0.8, 1.8, 101), 3)  # inclusive grid
_GAMMA_LUTS = {}
for g in _GAMMA_GRID:
    invGamma = 1.0 / float(g)
    lut = ((np.arange(256, dtype=np.float32) / 255.0) ** invGamma * 255.0).astype(
        np.uint8
    )
    _GAMMA_LUTS[float(g)] = lut


def _nearest_gamma(gamma: float) -> float:
    idx = int(
        np.clip(
            np.round((gamma - 0.8) / (1.8 - 0.8) * (len(_GAMMA_GRID) - 1)),
            0,
            len(_GAMMA_GRID) - 1,
        )
    )
    return float(_GAMMA_GRID[idx])


def adjust_gamma(image, gamma=1.0):
    g = _nearest_gamma(float(gamma))
    return cv2.LUT(image, _GAMMA_LUTS[g])


def add_contrast(img, contrast):
    buf = img.copy()
    f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
    alpha_c = f
    gamma_c = 127 * (1 - f)
    buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf


def preproces_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = adjust_gamma(img, 1.5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = add_contrast(img, 20)
    return img




## === cell 12
def random_rotation(image_array: np.ndarray):
    raise NotImplementedError("random_rotation is unused in this pipeline.")


def random_noise(image_array: np.ndarray):
    raise NotImplementedError("random_noise is unused in this pipeline.")


def horizontal_flip(image_array: np.ndarray):
    return image_array[:, ::-1]




## === cell 13
wts = [0.1, 0.4, 0.2, 0.90, 0.60]
num_of_class = 5




## === cell 14
img0 = cv2.imread(train_images[0])
img0p = preproces_image(img0)
plt.imshow(img0p)
plt.axis("off")
plt.show()




## === cell 15
def _to_model_input(batch_imgs_uint8):
    return batch_imgs_uint8.astype(np.float32) / 255.0


@lru_cache(maxsize=16384)
def _read_and_base_preprocess(path: str):
    im = cv2.imread(path)
    if im is None:
        im = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    else:
        im = preproces_image(im)
    return np.ascontiguousarray(im)


def _build_base_cache(image_paths):
    cache = np.empty((len(image_paths), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    read_pp = _read_and_base_preprocess
    for i, p in enumerate(image_paths):
        cache[i] = read_pp(p)
    return cache


train_base_cache = _build_base_cache(train_images)
val_base_cache = _build_base_cache(test_images)
print("Cached train/val base images:", train_base_cache.shape, val_base_cache.shape)


def _epoch_gammas(num_samples: int, epoch_seed: int) -> np.ndarray:
    rs = np.random.RandomState(int(epoch_seed))
    return rs.uniform(0.8, 1.8, size=num_samples).astype(np.float32)


def generate_training_images(cur_images, cur_tags, batch_size=500):
    cur_tags_arr = np.asarray(cur_tags, dtype=np.int64)
    onehot = tf.keras.utils.to_categorical(cur_tags_arr, num_of_class)
    sample_w = np.asarray([wts[int(t)] for t in cur_tags_arr], dtype=np.float32)

    to_inp = _to_model_input
    hflip = horizontal_flip

    base_rs = np.random.RandomState(SEED)

    base_cache = (
        train_base_cache
        if cur_images is train_images
        else _build_base_cache(cur_images)
    )
    n = len(cur_images)

    batch_x = np.empty((batch_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    batch_y = np.empty((batch_size, num_of_class), dtype=np.float32)
    batch_w = np.empty((batch_size,), dtype=np.float32)

    while True:
        epoch_seed = int(base_rs.randint(0, 2**31 - 1))
        gammas = _epoch_gammas(n, epoch_seed)

        bi = 0
        for ix in range(n):
            label_oh = onehot[ix]
            wt = sample_w[ix]
            img = base_cache[ix]

            batch_x[bi] = img
            batch_y[bi] = label_oh
            batch_w[bi] = wt
            bi += 1
            if bi == batch_size:
                yield to_inp(batch_x), batch_y, batch_w
                bi = 0

            batch_x[bi] = adjust_gamma(img, float(gammas[ix]))
            batch_y[bi] = label_oh
            batch_w[bi] = wt
            bi += 1
            if bi == batch_size:
                yield to_inp(batch_x), batch_y, batch_w
                bi = 0

            batch_x[bi] = hflip(img)
            batch_y[bi] = label_oh
            batch_w[bi] = wt
            bi += 1
            if bi == batch_size:
                yield to_inp(batch_x), batch_y, batch_w
                bi = 0

        if bi > 0:
            yield to_inp(batch_x[:bi]), batch_y[:bi], batch_w[:bi]




## === cell 16
def generate_testing_images(cur_images, cur_tags, batch_size=500):
    cur_tags_arr = np.asarray(cur_tags, dtype=np.int64)
    onehot = tf.keras.utils.to_categorical(cur_tags_arr, num_of_class)

    to_inp = _to_model_input
    hflip = horizontal_flip

    base_rs = np.random.RandomState(SEED + 1)

    base_cache = (
        val_base_cache if cur_images is test_images else _build_base_cache(cur_images)
    )
    n = len(cur_images)

    batch_x = np.empty((batch_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    batch_y = np.empty((batch_size, num_of_class), dtype=np.float32)

    while True:
        epoch_seed = int(base_rs.randint(0, 2**31 - 1))
        gammas = _epoch_gammas(n, epoch_seed)

        bi = 0
        for ix in range(n):
            label_oh = onehot[ix]
            img = base_cache[ix]

            batch_x[bi] = img
            batch_y[bi] = label_oh
            bi += 1
            if bi == batch_size:
                yield to_inp(batch_x), batch_y
                bi = 0

            batch_x[bi] = adjust_gamma(img, float(gammas[ix]))
            batch_y[bi] = label_oh
            bi += 1
            if bi == batch_size:
                yield to_inp(batch_x), batch_y
                bi = 0

            batch_x[bi] = hflip(img)
            batch_y[bi] = label_oh
            bi += 1
            if bi == batch_size:
                yield to_inp(batch_x), batch_y
                bi = 0

        if bi > 0:
            yield to_inp(batch_x[:bi]), batch_y[:bi]




## === cell 17
for batch in generate_training_images(train_images, train_labels, batch_size=50):
    tmp_images, tmp_labels, tmp_w = batch
    print(
        tmp_images.shape,
        tmp_labels.shape,
        tmp_w.shape,
        tmp_images.dtype,
        tmp_images.min(),
        tmp_images.max(),
    )
    plt.imshow((tmp_images[0] * 255).astype(np.uint8))
    plt.axis("off")
    plt.show()
    break




## === cell 18
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler




## === cell 19
def reset_tf_session():
    K.clear_session()
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass


reset_tf_session()
input_shape = (IMG_SIZE, IMG_SIZE, 3)




## === cell 20
inp = Input(shape=input_shape)
base = DenseNet121(include_top=False, weights="imagenet", input_tensor=inp)
x = GlobalAveragePooling2D()(base.output)
out = Dense(num_of_class, activation="softmax")(x)
model = Model(inputs=inp, outputs=out)

INIT_LR = 5e-3
BATCH_SIZE = 200
EPOCHS = 2  # keep core training loop approach unchanged


def lr_scheduler(epoch):
    return min(INIT_LR * 0.9**epoch, 0.00001)


model.compile(
    optimizer=Adam(learning_rate=INIT_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 21
steps_per_epoch = max(1, int(np.ceil((len(train_images) * 3) / BATCH_SIZE)))
val_steps = max(1, int(np.ceil((len(test_images) * 3) / BATCH_SIZE)))

output_signature_train = (
    tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None, num_of_class), dtype=tf.float32),
    tf.TensorSpec(shape=(None,), dtype=tf.float32),
)
output_signature_val = (
    tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None, num_of_class), dtype=tf.float32),
)

opts = tf.data.Options()
opts.deterministic = True

train_ds = (
    tf.data.Dataset.from_generator(
        lambda: generate_training_images(
            train_images, train_labels, batch_size=BATCH_SIZE
        ),
        output_signature=output_signature_train,
    )
    .with_options(opts)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_generator(
        lambda: generate_testing_images(
            test_images, test_labels, batch_size=BATCH_SIZE
        ),
        output_signature=output_signature_val,
    )
    .with_options(opts)
    .prefetch(tf.data.AUTOTUNE)
)

history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_ds,
    validation_steps=val_steps,
    callbacks=[LearningRateScheduler(lr_scheduler)],
    verbose=2,
)




## === cell 22
test_ids = test_df["id_code"].astype(str).tolist()
predict_images = [os.path.join(TEST_IMG_DIR, f"{id_code}.png") for id_code in test_ids]

missing = [p for p in predict_images if not os.path.exists(p)]
print("Num test images:", len(predict_images))
print("Missing test images:", len(missing))
if len(missing) > 0:
    print("Example missing:", missing[:5])

predict_base_cache = _build_base_cache(predict_images)


def generate_predict_images(cur_images, base_cache_uint8):
    n = len(cur_images)
    cur_names = [os.path.splitext(os.path.basename(p))[0] for p in cur_images]
    return _to_model_input(base_cache_uint8), cur_names


pred_batch, names = generate_predict_images(predict_images, predict_base_cache)
print(
    pred_batch.shape, len(names), pred_batch.dtype, pred_batch.min(), pred_batch.max()
)




## === cell 23
predictions = model.predict(pred_batch, batch_size=32, verbose=0)
print(predictions.shape)
print(predictions[:2])




## === cell 24
pred_labels = predictions.argmax(axis=1).astype(int)

subm = pd.DataFrame({"id_code": names, "diagnosis": pred_labels})
subm = test_df.merge(subm, on="id_code", how="left")
subm["diagnosis"] = subm["diagnosis"].fillna(0).astype(int)

submission_path = "submission.csv"
subm.to_csv(submission_path, index=False)
print("Wrote", submission_path, "with shape", subm.shape)
print(subm.head())
