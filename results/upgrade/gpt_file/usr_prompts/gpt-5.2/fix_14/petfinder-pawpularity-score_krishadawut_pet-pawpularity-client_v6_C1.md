# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

TRAIN_DIR = "../input/petfinder-pawpularity-score/train"
TEST_DIR = "../input/petfinder-pawpularity-score/test"
TRAIN_CSV_PATH = "../input/petfinder-pawpularity-score/train.csv"
TEST_CSV_PATH = "../input/petfinder-pawpularity-score/test.csv"

IMG_SIZE = (224, 224)  # (H, W)

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF pick
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

train_csv = pd.read_csv(TRAIN_CSV_PATH)
test_csv = pd.read_csv(TEST_CSV_PATH)

train_ids = train_csv["Id"].values
train_label = train_csv["Pawpularity"].values.astype(np.float32)


def _build_paths_py(img_dir, ids_np):
    ids_np = ids_np.astype(str, copy=False)
    return (np.char.add(np.char.add(img_dir + os.sep, ids_np), ".jpg")).tolist()


def _stable_list_hash(items) -> str:
    h = hashlib.md5()
    for s in items:
        h.update(str(s).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def _ensure_memmap_images(
    cache_path,
    img_paths,
    img_size=(224, 224),
    verbose=True,
    workers=None,
    chunk_size=256,
):
    h, w = img_size
    n = len(img_paths)

    if os.path.exists(cache_path):
        try:
            arr = np.load(cache_path, mmap_mode="r")
            if arr.shape == (n, h, w, 3) and arr.dtype == np.uint8:
                if verbose:
                    print(
                        f"Reusing cached memmap: {cache_path} shape={arr.shape} dtype={arr.dtype}"
                    )
                return arr
        except Exception:
            pass

    if verbose:
        print(f"Building memmap cache: {cache_path} (one-time cost)")

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tmp_path = cache_path + ".tmp"
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    arr = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=np.uint8, shape=(n, h, w, 3)
    )

    try:
        import cv2  # type: ignore

        def _read_resize_rgb_u8(path: str):
            im = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR uint8
            if im is None:
                raise ValueError(f"Failed to read image: {path}")
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            return im  # (h,w,3) uint8

        from concurrent.futures import ProcessPoolExecutor

        if workers is None:
            workers = min(8, (os.cpu_count() or 2))

        offset = 0
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            for i in range(0, n, int(chunk_size)):
                batch_paths = img_paths[i : i + int(chunk_size)]
                imgs = list(ex.map(_read_resize_rgb_u8, batch_paths))
                b = len(imgs)
                arr[offset : offset + b] = np.stack(imgs, axis=0)
                offset += b
                if verbose and (offset % 2048 == 0 or offset == n):
                    print(f"  processed {offset}/{n}")

    except Exception:
        if verbose:
            print(
                "OpenCV multiprocessing path unavailable; falling back to TF decode/resize (slower)."
            )

        @tf.function(reduce_retracing=True)
        def _decode_resize_one(path):
            img_bytes = tf.io.read_file(path)
            img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
            img.set_shape([None, None, 3])
            img = tf.image.resize(
                img, (h, w), method=tf.image.ResizeMethod.BILINEAR, antialias=False
            )
            img = tf.cast(img, tf.uint8)
            img.set_shape([h, w, 3])
            return img

        paths_tf = tf.constant(img_paths)
        ds = tf.data.Dataset.from_tensor_slices(paths_tf)

        options = tf.data.Options()
        options.deterministic = True
        ds = ds.with_options(options)

        ds = ds.map(
            _decode_resize_one, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
        )
        ds = ds.batch(int(chunk_size), drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        offset = 0
        for batch_imgs in ds:
            b = int(batch_imgs.shape[0])
            arr[offset : offset + b] = batch_imgs.numpy()
            offset += b
            if verbose and (offset % 2048 == 0 or offset == n):
                print(f"  processed {offset}/{n}")

    arr.flush()
    try:
        os.replace(tmp_path, cache_path)
    except Exception:
        cache_path = tmp_path

    return np.load(cache_path, mmap_mode="r")


def make_memmap_label_ds(memmap_u8, labels_f32, batch_size, training):
    n = int(len(labels_f32))

    x_tf = tf.convert_to_tensor(np.asarray(memmap_u8), dtype=tf.uint8)
    y_tf = tf.convert_to_tensor(
        np.asarray(labels_f32, dtype=np.float32), dtype=tf.float32
    )

    idx_ds = tf.data.Dataset.range(n)
    options = tf.data.Options()
    options.deterministic = True
    idx_ds = idx_ds.with_options(options)
    if training:
        idx_ds = idx_ds.repeat()  # preserve original training semantics

    idx_ds = idx_ds.batch(batch_size, drop_remainder=False)

    def _gather(idxs):
        x_u8 = tf.gather(x_tf, idxs, axis=0)
        y = tf.gather(y_tf, idxs, axis=0)
        x = tf.cast(x_u8, tf.float32) * (1.0 / 255.0)
        return x, y

    ds = idx_ds.map(_gather, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_memmap_only_ds(memmap_u8, batch_size):
    n = int(memmap_u8.shape[0])
    x_tf = tf.convert_to_tensor(np.asarray(memmap_u8), dtype=tf.uint8)

    idx_ds = tf.data.Dataset.range(n)
    options = tf.data.Options()
    options.deterministic = True
    idx_ds = idx_ds.with_options(options)
    idx_ds = idx_ds.batch(batch_size, drop_remainder=False)

    def _gather(idxs):
        x_u8 = tf.gather(x_tf, idxs, axis=0)
        x = tf.cast(x_u8, tf.float32) * (1.0 / 255.0)
        return x

    ds = idx_ds.map(_gather, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 1
eff = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling="avg",
)
eff.trainable = False

inp = keras.Input(shape=(224, 224, 3))
x = eff(inp)  # dataset already scales to [0,1]
x = layers.BatchNormalization()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(1)(x)
model = keras.Model(inputs=inp, outputs=out)


def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


model.compile(
    optimizer="rmsprop",
    loss=root_mean_squared_error,
    metrics=[keras.metrics.RootMeanSquaredError()],
)

batch_size = 32

train_ids_tr = train_ids[:7930]
train_label_tr = train_label[:7930]
train_ids_val = train_ids[7930:]
train_label_val = train_label[7930:]

train_paths_all = _build_paths_py(TRAIN_DIR, train_ids)
train_cache_all = f"train_images_224_all_{_stable_list_hash(train_ids)}.npy"
train_imgs_all_u8 = _ensure_memmap_images(
    train_cache_all,
    train_paths_all,
    img_size=IMG_SIZE,
    verbose=True,
    workers=None,
    chunk_size=512,
)

train_imgs_tr_u8 = train_imgs_all_u8[:7930]
train_imgs_val_u8 = train_imgs_all_u8[7930:]

train_ds = make_memmap_label_ds(
    train_imgs_tr_u8, train_label_tr, batch_size=batch_size, training=True
)
val_ds = make_memmap_label_ds(
    train_imgs_val_u8, train_label_val, batch_size=batch_size, training=False
)

steps_per_epoch = int(np.ceil(len(train_ids_tr) / batch_size))
val_steps = int(np.ceil(len(train_ids_val) / batch_size))

model.fit(train_ds, epochs=10, steps_per_epoch=steps_per_epoch, verbose=2)
results = model.evaluate(val_ds, steps=val_steps, verbose=0)
print("test loss, test rmse:", results)




## === cell 2
test_ids = test_csv["Id"].values
test_paths = _build_paths_py(TEST_DIR, test_ids)

test_cache = f"test_images_224_{_stable_list_hash(test_ids)}.npy"
test_imgs_u8 = _ensure_memmap_images(
    test_cache,
    test_paths,
    img_size=IMG_SIZE,
    verbose=True,
    workers=None,
    chunk_size=512,
)

test_ds = make_memmap_only_ds(test_imgs_u8, batch_size=32)

prediction = model.predict(test_ds, verbose=0).reshape(-1)
prediction = np.clip(prediction, 1.0, 100.0)

submission = pd.DataFrame({"Id": test_ids, "Pawpularity": prediction})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
assert submission.shape[0] == len(test_ids)
assert list(submission.columns) == ["Id", "Pawpularity"]
assert os.path.exists("submission.csv")
assert "submission.csv".endswith(".csv")
