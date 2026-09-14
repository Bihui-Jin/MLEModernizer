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
import sys
import random
import subprocess

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import numpy as np
    import pandas as pd
    import tensorflow as tf
except Exception:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
    for m in list(sys.modules.keys()):
        if m.startswith("google.protobuf"):
            del sys.modules[m]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
    )
    for m in list(sys.modules.keys()):
        if m.startswith("google.protobuf"):
            del sys.modules[m]
    import numpy as np
    import pandas as pd
    import tensorflow as tf

from tensorflow.keras.optimizers import Adam

print("TF version:", tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 1
def processAndWriteDf(df):
    if "image_id" not in df.columns:
        raise ValueError("Expected 'image_id' column in dataframe.")
    df.to_csv("./submission.csv", index=False)
    print("write done -> ./submission.csv")
    return df




## === cell 2
def _resolve_pp2020_base():
    """
    Fix: choose the real existing Kaggle input path (varies by environment).
    Supports both Kaggle's canonical /kaggle/input and the provided /kaggle/data layouts.
    """
    candidates = [
        "/kaggle/input/plant-pathology-2020-fgvc7",
        "/kaggle/data/plant-pathology-2020-fgvc7",
        "/kaggle/input",
        "/kaggle/data",
        "../input/plant-pathology-2020-fgvc7",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
            os.path.join(c, "test.csv")
        ):
            return c

        nested = os.path.join(c, "plant-pathology-2020-fgvc7")
        if os.path.exists(os.path.join(nested, "train.csv")) and os.path.exists(
            os.path.join(nested, "test.csv")
        ):
            return nested

    raise FileNotFoundError(f"Could not find dataset base dir in: {candidates}")


def getPredictionFromTPUModel():
    base = _resolve_pp2020_base()
    base_dir = os.path.join(base, "images")
    train_path = os.path.join(base, "train.csv")
    test_path = os.path.join(base, "test.csv")
    sample_path = os.path.join(base, "sample_submission.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_path)

    label_cols = [c for c in sample_sub.columns if c != "image_id"]

    test_image_ids = test_df["image_id"].astype(str).to_numpy()

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["image_id"] = train_df["image_id"].astype(str) + ".jpg"
    test_df["image_id"] = test_df["image_id"].astype(str) + ".jpg"

    img_size = (320, 320)
    batch_size = 32
    epochs_stage1 = 18
    epochs_stage2 = 6

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    n = len(train_df)
    val_n = int(np.floor(0.1 * n))
    train_n = n - val_n
    train_part = train_df.iloc[:train_n]
    val_part = train_df.iloc[train_n:]

    train_files = (base_dir + "/" + train_part["image_id"]).to_numpy()
    val_files = (base_dir + "/" + val_part["image_id"]).to_numpy()
    test_files = (base_dir + "/" + test_df["image_id"]).to_numpy()

    y_train = train_part[label_cols].to_numpy(dtype=np.float32)
    y_val = val_part[label_cols].to_numpy(dtype=np.float32)

    AUTO = tf.data.AUTOTUNE

    def _decode_resize(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(
            img, img_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )
        img = preprocess_input(img)
        return img

    def _augment_no_tfa(img, seed):
        img = tf.image.stateless_random_flip_left_right(img, seed=seed)

        dx = tf.random.stateless_uniform(
            [], seed=seed + tf.constant([1, 0], tf.int32), minval=-0.06, maxval=0.06
        )
        dy = tf.random.stateless_uniform(
            [], seed=seed + tf.constant([0, 1], tf.int32), minval=-0.06, maxval=0.06
        )
        tx = tf.cast(tf.round(dx * tf.cast(img_size[1], tf.float32)), tf.int32)
        ty = tf.cast(tf.round(dy * tf.cast(img_size[0], tf.float32)), tf.int32)
        img = tf.roll(img, shift=[ty, tx], axis=[0, 1])

        z = tf.random.stateless_uniform(
            [],
            seed=seed + tf.constant([2, 2], tf.int32),
            minval=1.0 - 0.08,
            maxval=1.0 + 0.08,
        )
        new_h = tf.cast(tf.round(z * tf.cast(img_size[0], tf.float32)), tf.int32)
        new_w = tf.cast(tf.round(z * tf.cast(img_size[1], tf.float32)), tf.int32)
        img2 = tf.image.resize(
            img, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )
        img2 = tf.image.resize_with_crop_or_pad(img2, img_size[0], img_size[1])

        angle = tf.random.stateless_uniform(
            [], seed=seed + tf.constant([3, 3], tf.int32), minval=-12.0, maxval=12.0
        ) * (np.pi / 180.0)
        cos_a = tf.math.cos(angle)
        sin_a = tf.math.sin(angle)
        cx = (img_size[1] - 1) / 2.0
        cy = (img_size[0] - 1) / 2.0

        a0 = cos_a
        a1 = -sin_a
        a2 = cx - cos_a * cx + sin_a * cy
        b0 = sin_a
        b1 = cos_a
        b2 = cy - sin_a * cx - cos_a * cy
        transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]

        img2 = tf.raw_ops.ImageProjectiveTransformV3(
            images=img2[None, ...],
            transforms=transform,
            output_shape=tf.constant(img_size, dtype=tf.int32),
            interpolation="BILINEAR",
            fill_mode="NEAREST",
            fill_value=0.0,
        )[0]
        return img2

    def make_ds(files, labels=None, training=False):
        opt = tf.data.Options()
        opt.experimental_deterministic = True

        files_ds = tf.data.Dataset.from_tensor_slices(files).with_options(opt)

        decoded = (
            files_ds.map(_decode_resize, num_parallel_calls=AUTO, deterministic=True)
            .apply(tf.data.experimental.ignore_errors())
            .cache()
        )

        if labels is None:
            ds = decoded
            ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTO)
            return ds

        labels_ds = tf.data.Dataset.from_tensor_slices(labels).with_options(opt)
        ds = tf.data.Dataset.zip((files_ds, decoded, labels_ds)).with_options(opt)

        def _apply_aug(path, img, y):
            if training:
                h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
                seed = tf.stack(
                    [tf.cast(h % (2**16), tf.int32), tf.cast(SEED, tf.int32)]
                )
                img = _augment_no_tfa(img, seed)
            return img, y

        ds = ds.map(_apply_aug, num_parallel_calls=AUTO, deterministic=True)

        if training:
            ds = ds.shuffle(
                buffer_size=min(2048, len(files)),
                seed=SEED,
                reshuffle_each_iteration=True,
            )

        ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTO)
        return ds

    train_ds = make_ds(train_files, y_train, training=True)
    valid_ds = make_ds(val_files, y_val, training=False)
    test_ds = make_ds(test_files, labels=None, training=False)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(label_cols), activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        )
    ]

    model.compile(
        optimizer=Adam(learning_rate=2e-4),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01),
        metrics=[
            tf.keras.metrics.AUC(
                multi_label=True, num_labels=len(label_cols), name="auc"
            )
        ],
        jit_compile=True,
    )

    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs_stage1,
        verbose=1,
        callbacks=callbacks,
    )

    base_model.trainable = True
    for layer in base_model.layers[:-25]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01),
        metrics=[
            tf.keras.metrics.AUC(
                multi_label=True, num_labels=len(label_cols), name="auc"
            )
        ],
        jit_compile=True,
    )

    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs_stage2,
        verbose=1,
        callbacks=callbacks,
    )

    p = model.predict(test_ds, verbose=1)
    p = np.clip(p, 0.0, 1.0)

    out = pd.DataFrame(p, columns=label_cols)
    out.insert(0, "image_id", test_image_ids)
    out = out[sample_sub.columns.tolist()]
    return out




## === cell 3
isTPU = True

if isTPU:
    df = getPredictionFromTPUModel()
    print(df.head(5))
    df.to_csv("./submission.csv", index=False)
    print("write done -> ./submission.csv")
else:
    df = pd.read_csv("../input/notebook45bc751087/submission.csv")
    df.head()
    result = processAndWriteDf(df)
    result.head(5)
