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

# 5. Target score

0.7896952908587265

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, re, math, random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model

print("TF:", tf.__version__)
print("Keras:", tf.keras.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import pathlib

BASE_PATH = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")

assert os.path.exists(TRAIN_CSV), TRAIN_CSV
assert os.path.exists(SAMPLE_SUB), SAMPLE_SUB
assert os.path.isdir(TRAIN_IMG_DIR), TRAIN_IMG_DIR
assert os.path.isdir(TEST_IMG_DIR), TEST_IMG_DIR

train_df = pd.read_csv(TRAIN_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)

print(train_df.shape, sub_df.shape)
train_df.head()




## === cell 2
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(
        bits, channels=3, dct_method="INTEGER_FAST", fancy_upscaling=False
    )
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1] float32
    image = tf.image.resize(
        image, image_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    if label is None:
        return image
    else:
        return image, label




## === cell 3
BATCH_SIZE = 32
IMAGE_SIZE = (512, 512)

AUTO = tf.data.AUTOTUNE



## === cell 4
test_images = sub_df["image"].tolist()
IMAGE_PATHS = [os.path.join(TEST_IMG_DIR, f) for f in test_images]

missing = []
print("Missing test images (skipped check for speed):", len(missing))



## === cell 5
CLASSES = ["scab", "frog_eye_leaf_spot", "rust", "complex", "powdery_mildew", "healthy"]
class_to_idx = {c: i for i, c in enumerate(CLASSES)}


def labels_to_vec_series(labels_series: pd.Series) -> np.ndarray:
    s = labels_series.fillna("").astype(str)
    tokens = s.str.get_dummies(sep=" ")
    tokens = tokens.reindex(columns=CLASSES, fill_value=0)
    return tokens.to_numpy(dtype=np.float32)


train_df["filepath"] = train_df["image"].map(lambda x: os.path.join(TRAIN_IMG_DIR, x))

X = train_df["filepath"].values
Y = labels_to_vec_series(train_df["labels"])

print("Train:", X.shape, Y.shape, "Pos rates:", Y.mean(axis=0))



## === cell 6
idx = np.arange(len(X))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.1
val_n = int(len(idx) * val_frac)
val_idx = idx[:val_n]
tr_idx = idx[val_n:]

X_tr, Y_tr = X[tr_idx], Y[tr_idx]
X_val, Y_val = X[val_idx], Y[val_idx]

print("Split:", len(X_tr), len(X_val))


def _map_train(x, y):
    return decode_image(x, y, IMAGE_SIZE)


def _map_test(x):
    return decode_image(x, None, IMAGE_SIZE)


options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
try:
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_and_batch_fusion = True
except Exception:
    pass
try:
    options.threading.private_threadpool_size = 0
    options.threading.max_intra_op_parallelism = 0
except Exception:
    pass

CACHE_DIR = "/kaggle/working/tf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
TRAIN_CACHE = os.path.join(CACHE_DIR, "train_images.cache")
VAL_CACHE = os.path.join(CACHE_DIR, "val_images.cache")

train_ds = (
    tf.data.Dataset.from_tensor_slices((tf.constant(X_tr, dtype=tf.string), Y_tr))
    .with_options(options)
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .map(_map_train, num_parallel_calls=AUTO, deterministic=True)
    .cache(TRAIN_CACHE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((tf.constant(X_val, dtype=tf.string), Y_val))
    .with_options(options)
    .map(_map_train, num_parallel_calls=AUTO, deterministic=True)
    .cache(VAL_CACHE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(tf.constant(IMAGE_PATHS, dtype=tf.string))
    .with_options(options)
    .map(_map_test, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## === cell 7
base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
)
base.trainable = False  # keep runtime and stability within limits

inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = tf.keras.applications.resnet.preprocess_input(inputs * 255.0)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(CLASSES), activation="sigmoid")(x)
model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=16,
)

model.summary()

feat_inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
feat_x = tf.keras.applications.resnet.preprocess_input(feat_inputs * 255.0)
feat_out = base(feat_x, training=False)
feature_extractor = Model(feat_inputs, feat_out)

base_out_shape = feature_extractor.output_shape[1:]  # (H, W, C)
head_inp = layers.Input(shape=base_out_shape)
h = layers.GlobalAveragePooling2D()(head_inp)
h = layers.Dropout(0.2)(h)
head_out = layers.Dense(len(CLASSES), activation="sigmoid")(h)
head_model = Model(head_inp, head_out)
head_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=16,
)

head_model.layers[-1].set_weights(model.layers[-1].get_weights())



## === cell 8
FEATURE_BATCH = 64  # larger batch improves throughput; does not change outputs


def _extract_features_to_memmap_predict(ds_images_only, n_samples: int, out_path: str):
    feat_shape = tuple(
        base_out_shape
    )  # (H,W,C) from graph; avoids iterating the dataset
    mm = np.memmap(
        out_path, mode="w+", dtype=np.float32, shape=(n_samples,) + feat_shape
    )

    feats = feature_extractor.predict(ds_images_only, verbose=1)
    if feats.dtype != np.float32:
        feats = feats.astype(np.float32, copy=False)
    mm[:] = feats
    mm.flush()
    return np.memmap(
        out_path, mode="r", dtype=np.float32, shape=(n_samples,) + feat_shape
    )


train_img_only = train_ds.map(
    lambda x, y: x, num_parallel_calls=AUTO, deterministic=True
)
val_img_only = val_ds.map(lambda x, y: x, num_parallel_calls=AUTO, deterministic=True)
test_img_only = test_dataset  # already images only

train_img_only = train_img_only.batch(1).unbatch().batch(FEATURE_BATCH).prefetch(AUTO)
val_img_only = val_img_only.batch(1).unbatch().batch(FEATURE_BATCH).prefetch(AUTO)
test_img_only = test_img_only.batch(1).unbatch().batch(FEATURE_BATCH).prefetch(AUTO)

Xtr_feat = _extract_features_to_memmap_predict(
    train_img_only, len(X_tr), "/kaggle/working/Xtr_feat.dat"
)
Xva_feat = _extract_features_to_memmap_predict(
    val_img_only, len(X_val), "/kaggle/working/Xva_feat.dat"
)
Xte_feat = _extract_features_to_memmap_predict(
    test_img_only, len(IMAGE_PATHS), "/kaggle/working/Xte_feat.dat"
)

Ytr_arr = Y_tr.astype(np.float32, copy=False)
Yva_arr = Y_val.astype(np.float32, copy=False)

print("Features:", Xtr_feat.shape, Xva_feat.shape, Xte_feat.shape)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3720443260.py in <cell line: 0>()
     36 test_img_only = test_img_only.batch(1).unbatch().batch(FEATURE_BATCH).prefetch(AUTO)
     37 
---> 38 Xtr_feat = _extract_features_to_memmap_predict(
     39     train_img_only, len(X_tr), "/kaggle/working/Xtr_feat.dat"
     40 )

/tmp/ipykernel_11/3720443260.py in _extract_features_to_memmap_predict(ds_images_only, n_samples, out_path)
     14 
     15     # Predict on dataset in the same order; deterministic options are already set upstream.
---> 16     feats = feature_extractor.predict(ds_images_only, verbose=1)
     17     # Ensure we write as float32 without changing values (predict already returns float32)
     18     if feats.dtype != np.float32:

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/layers/input_spec.py in assert_input_compatibility(input_spec, inputs, layer_name)
    243                 if spec_dim is not None and dim is not None:
    244                     if spec_dim != dim:
--> 245                         raise ValueError(
    246                             f'Input {input_index} of layer "{layer_name}" is '
    247                             "incompatible with the layer: "

ValueError: Input 0 of layer "functional_1" is incompatible with the layer: expected shape=(None, 512, 512, 3), found shape=(64, 32, 512, 512, 3)

## === cell 9
EPOCHS = 3

history = {"loss": [], "val_loss": []}
rng2 = np.random.RandomState(SEED)

for ep in range(EPOCHS):
    perm = rng2.permutation(len(Xtr_feat))
    hist = head_model.fit(
        Xtr_feat[perm],
        Ytr_arr[perm],
        validation_data=(Xva_feat, Yva_arr),
        epochs=1,
        verbose=1,
        batch_size=BATCH_SIZE,
        shuffle=False,  # already deterministically shuffled above
    )
    history["loss"].append(hist.history["loss"][0])
    history["val_loss"].append(hist.history["val_loss"][0])



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2431195056.py in <cell line: 0>()
      5 
      6 for ep in range(EPOCHS):
----> 7     perm = rng2.permutation(len(Xtr_feat))
      8     hist = head_model.fit(
      9         Xtr_feat[perm],

NameError: name 'Xtr_feat' is not defined

## === cell 10
probs = head_model.predict(Xte_feat, batch_size=FEATURE_BATCH, verbose=1)
temp_probs = probs
print("Pred shape:", temp_probs.shape)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2359226105.py in <cell line: 0>()
----> 1 probs = head_model.predict(Xte_feat, batch_size=FEATURE_BATCH, verbose=1)
      2 temp_probs = probs
      3 print("Pred shape:", temp_probs.shape)
      4 

NameError: name 'Xte_feat' is not defined

## === cell 11
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "rust",
    3: "complex",
    4: "powdery_mildew",
    5: "healthy",
}

threshold = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}
threshold2 = {0: 0.20, 1: 0.20, 2: 0.20, 3: 0.20, 4: 0.20}

thr = np.array([threshold[i] for i in range(5)], dtype=np.float32)
thr2 = np.array([threshold2[i] for i in range(5)], dtype=np.float32)
name0_4 = [name[i] for i in range(5)]
healthy_name = name[5]

above_thr = temp_probs[:, :5] > thr[None, :]
above_thr2_cnt = (temp_probs[:, :5] > thr2[None, :]).sum(axis=1)

pred_string = []
for r in range(temp_probs.shape[0]):
    s_parts = [name0_4[i] for i in range(5) if above_thr[r, i]]
    if above_thr2_cnt[r] >= 2 and "complex" not in s_parts:
        s_parts.append("complex")
    pred_string.append(healthy_name if not s_parts else " ".join(s_parts))

print("Example preds:", pred_string[:5])

df = pd.DataFrame({"image": test_images, "labels": pred_string})
assert len(df) == len(sub_df), (len(df), len(sub_df))
df.to_csv("submission.csv", index=False)
df.head()

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1097092252.py in <cell line: 0>()
     16 healthy_name = name[5]
     17 
---> 18 above_thr = temp_probs[:, :5] > thr[None, :]
     19 above_thr2_cnt = (temp_probs[:, :5] > thr2[None, :]).sum(axis=1)
     20 

NameError: name 'temp_probs' is not defined
