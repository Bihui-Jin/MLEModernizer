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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

18.87476603238

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

try:
    cpu = os.cpu_count() or 8
    tf.config.threading.set_intra_op_parallelism_threads(min(4, cpu))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

DATA_DIR = "/kaggle/input/petfinder-pawpularity-score"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")

IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

train_csv = pd.read_csv(TRAIN_CSV)
train_meta = train_csv.drop(columns=["Id", "Pawpularity"]).astype("float32").to_numpy()
train_label = train_csv["Pawpularity"].astype("float32").to_numpy()

train_ids = train_csv["Id"].astype(str).to_numpy()
train_paths = np.char.add(np.char.add(TRAIN_IMG_DIR + "/", train_ids), ".jpg")

opts = tf.data.Options()
opts.experimental_deterministic = True
opts.experimental_optimization.apply_default_optimizations = True

MMAP_DIR = "/kaggle/working/petfinder_mmaps"
os.makedirs(MMAP_DIR, exist_ok=True)

TRAIN_MMAP = os.path.join(MMAP_DIR, f"train_imgs_{IMG_SIZE}x{IMG_SIZE}_uint8.dat")
TEST_MMAP = os.path.join(MMAP_DIR, f"test_imgs_{IMG_SIZE}x{IMG_SIZE}_uint8.dat")


def _build_or_load_mmap_uint8(paths_np: np.ndarray, mmap_path: str) -> np.memmap:
    n = len(paths_np)
    shape = (n, IMG_SIZE, IMG_SIZE, 3)
    expected_bytes = int(np.prod(shape, dtype=np.int64) * np.dtype(np.uint8).itemsize)

    if os.path.exists(mmap_path) and os.path.getsize(mmap_path) == expected_bytes:
        return np.memmap(mmap_path, mode="r", dtype=np.uint8, shape=shape)

    mm = np.memmap(mmap_path, mode="w+", dtype=np.uint8, shape=shape)

    import cv2  # available on Kaggle
    from concurrent.futures import ThreadPoolExecutor

    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    try:
        cv2.setUseOptimized(True)
    except Exception:
        pass

    def _read_resize_one(p: str):
        img = cv2.imread(p, cv2.IMREAD_COLOR)  # BGR uint8
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        return img  # uint8 (H,W,3)

    max_workers = min(8, (os.cpu_count() or 2))
    block = 4096 if n >= 4096 else 1024

    paths_list = paths_np.tolist()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for start in range(0, n, block):
            end = min(n, start + block)
            chunk_paths = paths_list[start:end]
            imgs_list = list(ex.map(_read_resize_one, chunk_paths, chunksize=128))
            mm[start:end] = np.stack(imgs_list, axis=0)

    mm.flush()
    return np.memmap(mmap_path, mode="r", dtype=np.uint8, shape=shape)


train_imgs_mm = _build_or_load_mmap_uint8(train_paths, TRAIN_MMAP)


def _make_train_ds_from_mmap(
    imgs_mm: np.memmap, meta_np: np.ndarray, label_np: np.ndarray
):
    n = int(len(label_np))
    meta_np_c = np.ascontiguousarray(meta_np, dtype=np.float32)
    label_np_c = np.ascontiguousarray(label_np, dtype=np.float32)
    mm = imgs_mm

    meta_tf = tf.constant(meta_np_c)
    y_tf = tf.constant(label_np_c)

    def _fetch_batch_numpy(idx_batch):
        idx = np.asarray(idx_batch, dtype=np.int32)
        imgs = mm[idx]  # uint8, (B,H,W,3), memmap slice
        return imgs, idx  # return idx for TF gather

    def _fetch_batch_tf(idx_batch):
        imgs, idx = tf.numpy_function(
            _fetch_batch_numpy, [idx_batch], [tf.uint8, tf.int32]
        )
        imgs.set_shape([None, IMG_SIZE, IMG_SIZE, 3])
        idx.set_shape([None])

        imgs = tf.cast(imgs, tf.float32)
        meta = tf.gather(meta_tf, idx)
        y = tf.gather(y_tf, idx)
        return (imgs, meta), y

    ds = tf.data.Dataset.range(n).with_options(opts)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.map(_fetch_batch_tf, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_train_ds_from_mmap(train_imgs_mm, train_meta, train_label)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
eff = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)
eff.trainable = False

inputA = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.RandomFlip("horizontal")(inputA)
x = layers.GaussianNoise(0.1)(x)

x = keras.applications.efficientnet.preprocess_input(x)

x = eff(x)
x = layers.BatchNormalization()(x)

inputB = keras.Input(shape=(12,))

combined = layers.Concatenate()([x, inputB])
combined = layers.Dense(1)(combined)

model = keras.Model(inputs=[inputA, inputB], outputs=combined)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.RootMeanSquaredError()],
    steps_per_execution=32,
)

model.fit(train_ds, epochs=15, verbose=1)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4015761867.py in <cell line: 0>()
     30 )
     31 
---> 32 model.fit(train_ds, epochs=15, verbose=1)
     33 

NameError: name 'train_ds' is not defined

## === cell 2
test_csv = pd.read_csv(TEST_CSV)
test_meta = test_csv.drop(columns=["Id"]).astype("float32").to_numpy()

test_ids = test_csv["Id"].astype(str).to_numpy()
test_paths = np.char.add(np.char.add(TEST_IMG_DIR + "/", test_ids), ".jpg")

test_imgs_mm = _build_or_load_mmap_uint8(test_paths, TEST_MMAP)


def _make_test_ds_from_mmap(imgs_mm: np.memmap, meta_np: np.ndarray):
    n = int(len(meta_np))
    meta_np_c = np.ascontiguousarray(meta_np, dtype=np.float32)
    mm = imgs_mm

    meta_tf = tf.constant(meta_np_c)

    def _fetch_batch_numpy(idx_batch):
        idx = np.asarray(idx_batch, dtype=np.int32)
        imgs = mm[idx]  # uint8
        return imgs, idx

    def _fetch_batch_tf(idx_batch):
        imgs, idx = tf.numpy_function(
            _fetch_batch_numpy, [idx_batch], [tf.uint8, tf.int32]
        )
        imgs.set_shape([None, IMG_SIZE, IMG_SIZE, 3])
        idx.set_shape([None])
        imgs = tf.cast(imgs, tf.float32)
        meta = tf.gather(meta_tf, idx)
        return imgs, meta

    ds = tf.data.Dataset.range(n).with_options(opts)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.map(_fetch_batch_tf, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = _make_test_ds_from_mmap(test_imgs_mm, test_meta)

prediction = model.predict(test_ds, verbose=1).reshape(-1)
prediction = np.clip(prediction, 0.0, 100.0)

submission = pd.DataFrame({"Id": test_csv["Id"].values, "Pawpularity": prediction})
submission.to_csv("submission.csv", index=False)

submission.head()

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/835856042.py in <cell line: 0>()
      4 # --- SPEED: Vectorized numpy string ops for paths (less pandas overhead), same result.
      5 test_ids = test_csv["Id"].astype(str).to_numpy()
----> 6 test_paths = np.char.add(np.char.add(TEST_IMG_DIR + "/", test_ids), ".jpg")
      7 
      8 test_imgs_mm = _build_or_load_mmap_uint8(test_paths, TEST_MMAP)

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U47' and 'object' (the few cases where this used to work often lead to incorrect results).
