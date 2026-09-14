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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.976958525345622

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.12221) has done: 'The timeout is most likely dominated by expensive image decoding/resize running in the input pipeline plus Python/TF overhead from repeated augmentation execution; training itself is fairly short. I keep the same model, epochs, losses, and augmentation layers, but make the input pipeline more efficient by fusing decode+resize via `tf.image.resize_with_pad` (same semantics for fixed-size resizing) and enabling the non-debug, graph-optimized `tf.data` execution path. I also ensure augmentation runs after batching (vectorized over the batch) to reduce per-element overhead while preserving identical augmentation logic and randomness (same layers/seeds). Finally, I add a small, correctness-preserving optimization to avoid repeated string concatenations and ensure the datasets use the most efficient deterministic settings available.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import random
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

BASE_CANDIDATES = [
    "/kaggle/input/paddy-disease-classification",
    "/kaggle/input/paddy-disease-classification/paddy-disease-classification",
    "/kaggle/data/paddy-disease-classification",
    "/kaggle/data/paddy-disease-classification/paddy-disease-classification",
]
BASE = None
for c in BASE_CANDIDATES:
    if os.path.exists(c):
        BASE = c
        break
if BASE is None:
    raise FileNotFoundError(
        "Could not locate paddy-disease-classification dataset folder in known locations."
    )

TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")

TRAIN_IMG_CANDIDATES = [
    os.path.join(BASE, "train_images"),
    os.path.join(BASE, "paddy-disease-classification", "train_images"),
]
TEST_IMG_CANDIDATES = [
    os.path.join(BASE, "test_images"),
    os.path.join(BASE, "paddy-disease-classification", "test_images"),
]

TRAIN_IMG_DIR = next((p for p in TRAIN_IMG_CANDIDATES if os.path.isdir(p)), None)
TEST_IMG_DIR = next((p for p in TEST_IMG_CANDIDATES if os.path.isdir(p)), None)
if TRAIN_IMG_DIR is None or TEST_IMG_DIR is None:
    raise FileNotFoundError(f"Could not find train_images/test_images under {BASE}")

print("BASE:", BASE)
print("TRAIN_CSV:", TRAIN_CSV)
print("TRAIN_IMG_DIR:", TRAIN_IMG_DIR)
print("TEST_IMG_DIR:", TEST_IMG_DIR)

DATA_OPTS = tf.data.Options()
DATA_OPTS.deterministic = True
DATA_OPTS.experimental_optimization.apply_default_optimizations = True
DATA_OPTS.experimental_optimization.autotune_buffers = True
DATA_OPTS.experimental_optimization.autotune_cpu_budget = True
DATA_OPTS.experimental_optimization.autotune_ram_budget = True
DATA_OPTS.experimental_optimization.map_parallelization = True
DATA_OPTS.experimental_optimization.map_and_batch_fusion = True
DATA_OPTS.experimental_optimization.parallel_batch = True
DATA_OPTS.experimental_threading.private_threadpool_size = max(8, (os.cpu_count() or 8))
DATA_OPTS.experimental_threading.max_intra_op_parallelism = 0  # 0 = let TF decide

train_df = pd.read_csv(TRAIN_CSV)

required_cols = {"image_id", "label"}
missing = required_cols - set(train_df.columns)
if missing:
    raise ValueError(f"train.csv missing columns: {missing}")

labels_sorted = sorted(train_df["label"].unique().tolist())
print("Num classes:", len(labels_sorted))
print("Classes:", labels_sorted)

train_df = train_df.copy()
train_df["filename"] = (
    train_df["label"].astype(str) + "/" + train_df["image_id"].astype(str)
)

for i in range(3):
    fp = os.path.join(TRAIN_IMG_DIR, train_df["filename"].iloc[i])
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Example train image not found: {fp}")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split

train_split, valid_split = train_test_split(
    train_df,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"],
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE

class_names = sorted(train_split["label"].unique().tolist())
class_indices = {name: i for i, name in enumerate(class_names)}
inverse_map = {v: k for k, v in class_indices.items()}
num_classes = len(class_names)
print("class_indices:", class_indices)

TRAIN_IMG_DIR_STR = str(TRAIN_IMG_DIR)


def _paths_and_intlabels(df: pd.DataFrame):
    paths = (TRAIN_IMG_DIR_STR + "/" + df["filename"].astype(str)).to_numpy(dtype=str)
    y = df["label"].map(class_indices).astype(np.int32).to_numpy()
    return paths, y


train_paths, train_y = _paths_and_intlabels(train_split)
valid_paths, valid_y = _paths_and_intlabels(valid_split)


@tf.function
def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img.set_shape([None, None, 3])
    img = tf.image.resize_with_pad(
        img,
        IMG_SIZE[0],
        IMG_SIZE[1],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.keras.applications.xception.preprocess_input(tf.cast(img, tf.float32))
    return img


@tf.function
def _load_and_preprocess(path, y):
    return _decode_resize_preprocess(path), y


augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(
            factor=10.0 / 360.0, fill_mode="reflect", seed=SEED
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.05, width_factor=0.05, fill_mode="reflect", seed=SEED
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            width_factor=(-0.1, 0.1),
            fill_mode="reflect",
            seed=SEED,
        ),
        tf.keras.layers.RandomFlip(mode="horizontal", seed=SEED),
    ],
    name="augment",
)


@tf.function
def _augment_only_map_batched(img, y):
    img = augment(img, training=True)
    return img, y


shuffle_buf = min(len(train_paths), 2048)

train_base = tf.data.Dataset.from_tensor_slices((train_paths, train_y)).with_options(
    DATA_OPTS
)
train_base = train_base.map(
    _load_and_preprocess, num_parallel_calls=AUTO, deterministic=True
)
train_base = train_base.cache()  # in-memory

train_ds = train_base.shuffle(
    buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
)
train_ds = train_ds.repeat()
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.map(
    _augment_only_map_batched, num_parallel_calls=AUTO, deterministic=True
)
train_ds = train_ds.prefetch(AUTO)

valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_y)).with_options(
    DATA_OPTS
)
valid_ds = valid_ds.map(
    _load_and_preprocess, num_parallel_calls=AUTO, deterministic=True
)
valid_ds = valid_ds.cache()  # in-memory
valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=False)
valid_ds = valid_ds.prefetch(AUTO)

steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))
validation_steps = int(np.ceil(len(valid_paths) / BATCH_SIZE))
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2576101669.py in <cell line: 0>()
      2 
      3 train_split, valid_split = train_test_split(
----> 4     train_df,
      5     test_size=0.1,
      6     random_state=SEED,

NameError: name 'train_df' is not defined

## === cell 2
back_bone = tf.keras.applications.Xception(
    weights="imagenet",
    input_shape=(224, 224, 3),
    include_top=False,
)

input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
x = back_bone(input_layer)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
output_layer = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.models.Model(input_layer, output_layer)

back_bone.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=["accuracy"],
    steps_per_execution=64,
)

model.summary()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/660583965.py in <cell line: 0>()
      8 x = back_bone(input_layer)
      9 x = tf.keras.layers.GlobalAveragePooling2D()(x)
---> 10 output_layer = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
     11 model = tf.keras.models.Model(input_layer, output_layer)
     12 

NameError: name 'num_classes' is not defined

## === cell 3
EPOCHS_HEAD = 3
history_head = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS_HEAD,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)

back_bone.trainable = True
for layer in back_bone.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=["accuracy"],
    steps_per_execution=64,
)

EPOCHS_FT = 2
history_ft = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS_FT,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2835521834.py in <cell line: 0>()
      1 EPOCHS_HEAD = 3
----> 2 history_head = model.fit(
      3     train_ds,
      4     validation_data=valid_ds,
      5     epochs=EPOCHS_HEAD,

NameError: name 'model' is not defined

## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB)
if "image_id" not in sample_sub.columns:
    raise ValueError("sample_submission.csv missing image_id column")

image_ids = sample_sub["image_id"].astype(str).to_numpy(dtype=str)

test_paths_arr = (str(TEST_IMG_DIR) + "/" + image_ids).astype(str)

for p in test_paths_arr[:3]:
    if not tf.io.gfile.exists(p):
        raise FileNotFoundError(f"Example test image not found: {p}")

test_ds = tf.data.Dataset.from_tensor_slices(test_paths_arr).with_options(DATA_OPTS)


@tf.function
def _test_load_and_preprocess(path):
    return _decode_resize_preprocess(path)


test_ds = test_ds.map(
    _test_load_and_preprocess, num_parallel_calls=AUTO, deterministic=True
)
test_ds = test_ds.cache()
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTO)

pred = model.predict(test_ds, verbose=1)
pred_idx = np.argmax(pred, axis=1)
pred_labels = [inverse_map[int(i)] for i in pred_idx]

sub = pd.DataFrame({"image_id": image_ids.tolist(), "label": pred_labels})

if len(sample_sub) == len(sub):
    sub = sample_sub[["image_id"]].merge(sub, on="image_id", how="left")
    if sub["label"].isna().any():
        fill_label = train_df["label"].mode().iloc[0]
        sub["label"] = sub["label"].fillna(fill_label)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv path:", os.path.abspath("submission.csv"))

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
UFuncTypeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1226932286.py in <cell line: 0>()
      7 # --- Fix: avoid np.char.add dtype-kind mismatch (object vs unicode)
      8 # Use robust vectorized concatenation that yields the exact same paths.
----> 9 test_paths_arr = (str(TEST_IMG_DIR) + "/" + image_ids).astype(str)
     10 
     11 for p in test_paths_arr[:3]:

UFuncTypeError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U55'), dtype('<U10')) -> None
