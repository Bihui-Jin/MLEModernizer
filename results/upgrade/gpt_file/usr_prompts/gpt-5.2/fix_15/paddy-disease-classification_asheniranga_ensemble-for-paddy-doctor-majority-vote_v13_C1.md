# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "").lower() == "python":
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import random
import numpy as np
import pandas as pd
import scipy.stats as ss

random.seed(42)
np.random.seed(42)

import tensorflow as tf

tf.random.set_seed(42)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    _cpu = os.cpu_count() or 2
    tf.config.threading.set_intra_op_parallelism_threads(_cpu)
    tf.config.threading.set_inter_op_parallelism_threads(max(2, _cpu // 2))
except Exception as e:
    print("TF threading config skipped/failed:", repr(e))

print("TF version:", tf.__version__)
print("CPUs:", os.cpu_count())



## === cell 1
DATA_ROOT = "../input/paddy-disease-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test_images")

if not os.path.isdir(TEST_DIR):
    DATA_ROOT = "../input/paddy-disease-classification/paddy-disease-classification"
    TEST_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.isdir(TEST_DIR), f"Could not find test_images directory at {TEST_DIR}"
print("Using DATA_ROOT:", DATA_ROOT)
print("TEST_DIR:", TEST_DIR)

TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
assert os.path.isdir(TRAIN_DIR), f"Could not find train_images directory at {TRAIN_DIR}"

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
assert os.path.isfile(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Missing sample_submission.csv at {SAMPLE_SUB}"



## === cell 2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array

input_size = 256  # kept for compatibility with original code


def _np_pad_to_at_least(image, min_h, min_w):
    h, w = image.shape[0], image.shape[1]
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return image
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return np.pad(image, ((top, bottom), (left, right), (0, 0)), mode="reflect")


def _np_random_crop(image, crop_h, crop_w):
    image = _np_pad_to_at_least(image, crop_h, crop_w)
    h, w = image.shape[0], image.shape[1]
    top = np.random.randint(0, h - crop_h + 1) if h > crop_h else 0
    left = np.random.randint(0, w - crop_w + 1) if w > crop_w else 0
    out = image[top : top + crop_h, left : left + crop_w, :]
    if out.shape[0] != crop_h or out.shape[1] != crop_w:
        out = _np_pad_to_at_least(out, crop_h, crop_w)[:crop_h, :crop_w, :]
    return out


def _np_random_brightness(image, max_delta):
    delta = np.random.uniform(-max_delta, max_delta) * 255.0
    out = image.astype(np.float32) + delta
    return np.clip(out, 0.0, 255.0).astype(image.dtype, copy=False)


def _np_random_contrast(image, lower, upper):
    factor = np.random.uniform(lower, upper)
    x = image.astype(np.float32)
    mean = np.mean(x, axis=(0, 1), keepdims=True)
    out = (x - mean) * factor + mean
    return np.clip(out, 0.0, 255.0).astype(image.dtype, copy=False)


def _np_random_saturation(image, lower, upper):
    factor = np.random.uniform(lower, upper)
    x = image.astype(np.float32)
    gray = np.dot(x[..., :3], np.array([0.2989, 0.5870, 0.1140], dtype=np.float32))
    gray = gray[..., None]
    out = (x - gray) * factor + gray
    return np.clip(out, 0.0, 255.0).astype(image.dtype, copy=False)


def _np_random_hue(image, max_delta):
    x = image.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn

    h = np.zeros_like(mx)
    mask = diff != 0
    idx = (mx == r) & mask
    h[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6.0
    idx = (mx == g) & mask
    h[idx] = ((b[idx] - r[idx]) / diff[idx]) + 2.0
    idx = (mx == b) & mask
    h[idx] = ((r[idx] - g[idx]) / diff[idx]) + 4.0
    h = h / 6.0

    s = np.zeros_like(mx)
    s[mx != 0] = diff[mx != 0] / mx[mx != 0]
    v = mx

    delta = np.random.uniform(-max_delta, max_delta)
    h = (h + delta) % 1.0

    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    i_mod = i % 6
    out = np.empty_like(x)
    cond = i_mod == 0
    out[..., 0][cond], out[..., 1][cond], out[..., 2][cond] = v[cond], t[cond], p[cond]
    cond = i_mod == 1
    out[..., 0][cond], out[..., 1][cond], out[..., 2][cond] = q[cond], v[cond], p[cond]
    cond = i_mod == 2
    out[..., 0][cond], out[..., 1][cond], out[..., 2][cond] = p[cond], v[cond], t[cond]
    cond = i_mod == 3
    out[..., 0][cond], out[..., 1][cond], out[..., 2][cond] = p[cond], q[cond], v[cond]
    cond = i_mod == 4
    out[..., 0][cond], out[..., 1][cond], out[..., 2][cond] = t[cond], p[cond], v[cond]
    cond = i_mod == 5
    out[..., 0][cond], out[..., 1][cond], out[..., 2][cond] = v[cond], p[cond], q[cond]

    out = out * 255.0
    return np.clip(out, 0.0, 255.0).astype(image.dtype, copy=False)


def random_cutout(image, patch_size=16, patches=16):
    if random.choice([True, False]):
        anchors_x = []
        anchors_y = []

        for _ in range(patches):
            rv = np.random.randint(0, image.shape[0])
            if rv not in anchors_x:
                anchors_x.append(rv)

        for _ in range(patches):
            rv = np.random.randint(0, image.shape[1])
            if rv not in anchors_y:
                anchors_y.append(rv)

        for x, y in zip(anchors_x, anchors_y):
            image[x : x + patch_size, y : y + patch_size, :] = 0

        return image
    else:
        return image


def random_gaus_blur(image):
    if random.choice([True, False]):
        x = image.astype(np.float32)
        k = 7
        pad = k // 2
        xp = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        c = np.cumsum(xp, axis=0)
        c = c[k:, :, :] - c[:-k, :, :]
        c = np.cumsum(c, axis=1)
        c = c[:, k:, :] - c[:, :-k, :]
        out = c / float(k * k)
        return np.clip(out, 0.0, 255.0).astype(image.dtype, copy=False)
    else:
        return image


def random_displacment(image):
    if random.choice([True, False]):
        ax = random.choice([0, 1])
        if ax == 0:
            slices = np.array_split(image, 6, axis=ax)
            np.random.shuffle(slices)
            return np.vstack(slices)
        else:
            slices = np.array_split(image, 6, axis=ax)
            np.random.shuffle(slices)
            return np.hstack(slices)
    else:
        return image


def center_crop_and_random_augmentations_fn(image):
    image = _np_random_crop(image, input_size, input_size)
    image = random_cutout(image, 16, 16)
    image = random_displacment(image)
    image = random_gaus_blur(image)
    image = _np_random_brightness(image, 0.2)
    image = _np_random_contrast(image, 0.5, 2.0)
    image = _np_random_saturation(image, 0.75, 1.25)
    image = _np_random_hue(image, 0.1)
    if image.shape[0] != input_size or image.shape[1] != input_size:
        image = _np_pad_to_at_least(image, input_size, input_size)[
            :input_size, :input_size, :
        ]
    return image


def test_time_augmentation_fn_1(image):
    image = _np_random_crop(image, 256, 256)
    image = _np_random_brightness(image, 0.2)
    image = _np_random_contrast(image, 0.5, 2.0)
    return image


def test_time_augmentation_fn_2(image):
    image = _np_random_crop(image, 384, 384)
    image = _np_random_brightness(image, 0.2)
    image = _np_random_contrast(image, 0.5, 2.0)
    return image


def test_time_augmentation_fn_3(image):
    image = _np_random_crop(image, 300, 300)
    image = _np_random_brightness(image, 0.2)
    image = _np_random_contrast(image, 0.5, 2.0)
    return image




## === cell 3
generator_1 = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=test_time_augmentation_fn_1,
)

generator_2 = ImageDataGenerator(
    rescale=1.0 / 255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=test_time_augmentation_fn_2,
)

generator_3 = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=test_time_augmentation_fn_3,
)



## === cell 4
files_to_fit = [
    os.path.join(TRAIN_DIR, "bacterial_leaf_blight", "100049.jpg"),
    os.path.join(TRAIN_DIR, "bacterial_leaf_streak", "100042.jpg"),
    os.path.join(TRAIN_DIR, "bacterial_panicle_blight", "100068.jpg"),
    os.path.join(TRAIN_DIR, "blast", "100012.jpg"),
    os.path.join(TRAIN_DIR, "brown_spot", "100022.jpg"),
    os.path.join(TRAIN_DIR, "dead_heart", "100020.jpg"),
    os.path.join(TRAIN_DIR, "downy_mildew", "100059.jpg"),
    os.path.join(TRAIN_DIR, "hispa", "100139.jpg"),
    os.path.join(TRAIN_DIR, "normal", "100111.jpg"),
    os.path.join(TRAIN_DIR, "tungro", "100134.jpg"),
]

to_gen_fit = []
for file in files_to_fit:
    if os.path.exists(file):
        to_gen_fit.append(img_to_array(load_img(file), dtype="uint8"))

if len(to_gen_fit) == 0:
    train_df_tmp = pd.read_csv(TRAIN_CSV, usecols=["image_id", "label"])
    for img_id, label in train_df_tmp.head(50).itertuples(index=False, name=None):
        p = os.path.join(TRAIN_DIR, label, img_id)
        if os.path.exists(p):
            to_gen_fit.append(img_to_array(load_img(p), dtype="uint8"))
        if len(to_gen_fit) >= 10:
            break

if len(to_gen_fit) == 0:
    raise RuntimeError(
        "No images found to fit generator_2 statistics; cannot continue."
    )

generator_2.fit(np.asarray(to_gen_fit))
print("generator_2 fitted on", len(to_gen_fit), "images")



## === cell 5
test_filenames = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])
image_ids = test_filenames[:]  # filenames are the image_id values
abs_paths = [os.path.join(TEST_DIR, f) for f in test_filenames]
n_test = len(abs_paths)
print("n_test:", n_test)

_g2_mean = (
    generator_2.mean.astype(np.float32)
    if getattr(generator_2, "mean", None) is not None
    else None
)
_g2_std = (
    generator_2.std.astype(np.float32)
    if getattr(generator_2, "std", None) is not None
    else None
)
if _g2_mean is None or _g2_std is None:
    raise RuntimeError("generator_2 statistics (mean/std) not available after fit().")

AUTOTUNE = tf.data.AUTOTUNE


def _decode_jpeg_u8(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.clip_by_value(tf.round(img), 0.0, 255.0)
    img = tf.cast(img, tf.uint8)
    return img


def _resize_u8(img_u8, size_hw):
    img = tf.image.resize(img_u8, size_hw, method="bilinear", antialias=False)
    img = tf.clip_by_value(tf.round(img), 0.0, 255.0)
    img = tf.cast(img, tf.uint8)
    return img


def _np_standardize_g2_from_u8_batch(batch_u8):
    x = batch_u8.astype(np.float32) / 255.0
    x -= _g2_mean
    x /= _g2_std + 1e-7
    return x.astype(np.float32, copy=False)


def _make_base_ds_u8(paths, size_hw, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(_decode_jpeg_u8, num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x: _resize_u8(x, size_hw), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_tta_ds_float(base_ds_u8, preprocess_fn, mode):
    def _py_apply(batch_u8):
        b = batch_u8.shape[0]
        out_u8 = np.empty_like(batch_u8, dtype=np.uint8)
        for i in range(b):
            out_u8[i] = preprocess_fn(batch_u8[i])
        if mode in ("g1", "g3"):
            return (out_u8.astype(np.float32) / 255.0).astype(np.float32, copy=False)
        elif mode == "g2":
            return _np_standardize_g2_from_u8_batch(out_u8)
        else:
            raise ValueError("Unknown mode")

    def _tf_map(batch_u8):
        y = tf.numpy_function(_py_apply, [batch_u8], tf.float32)
        y.set_shape([None, None, None, 3])
        return y

    return base_ds_u8.map(_tf_map, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


base256_ds_u8 = _make_base_ds_u8(abs_paths, (256, 256), batch_size=32)
base300_ds_u8 = _make_base_ds_u8(abs_paths, (300, 300), batch_size=16)
base384_ds_u8 = _make_base_ds_u8(abs_paths, (384, 384), batch_size=16)

TTA = 5




## === cell 6
def safe_load_model(path):
    if not os.path.exists(path):
        return None
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        print(f"Failed to load model at {path}: {e}")
        return None


CANDIDATE_MODEL_PATHS = [
    ("m1", "../input/paddy-doc-ensemble-models/ensemble_estimators/resnet_150.hdf5"),
    ("m2", "../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_v2_m.hdf5"),
    ("m3", "../input/k/kasunpramodya/paddy-doctor-training/model_effnet_s.hdf5"),
    ("m4", "../input/notebooka9ca40495e/model_effnet_s.hdf5"),
    ("m5", "../input/paddy-doctor-training/model_effnet_b4.hdf5"),
]

loaded_models = {}
for name, path in CANDIDATE_MODEL_PATHS:
    model = safe_load_model(path)
    if model is not None:
        loaded_models[name] = model
        print(f"Loaded {name} from {path}")

m1 = loaded_models.get("m1")
m2 = loaded_models.get("m2")
m3 = loaded_models.get("m3")
m4 = loaded_models.get("m4")
m5 = loaded_models.get("m5")

if len(loaded_models) == 0:
    print(
        "No external .hdf5 models found; training an offline fallback model from train_images."
    )

    train_df = pd.read_csv(TRAIN_CSV).copy()
    train_df["filepath"] = (
        TRAIN_DIR.rstrip("/")
        + "/"
        + train_df["label"].astype(str)
        + "/"
        + train_df["image_id"].astype(str)
    )
    train_df = train_df[train_df["filepath"].map(os.path.exists)].reset_index(drop=True)
    assert len(train_df) > 0, "No training images found on disk."

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=center_crop_and_random_augmentations_fn,
    )

    train_flow = train_gen.flow_from_dataframe(
        train_df,
        x_col="filepath",
        y_col="label",
        target_size=(256, 256),
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        subset="training",
        seed=42,
    )
    val_flow = train_gen.flow_from_dataframe(
        train_df,
        x_col="filepath",
        y_col="label",
        target_size=(256, 256),
        class_mode="categorical",
        batch_size=32,
        shuffle=False,
        subset="validation",
        seed=42,
    )

    inputs = tf.keras.Input(shape=(256, 256, 3))
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.SeparableConv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    fallback_model = tf.keras.Model(inputs, outputs)

    fallback_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    fallback_model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=8,
        verbose=1,
    )

    m1 = fallback_model




## === cell 7
def predict_tta_from_base_ds(
    model, base_ds_u8, preprocess_fn, mode, batch_size, tta=5, n_classes=10
):
    acc = np.zeros((n_test, n_classes), dtype=np.float32)
    for _ in range(tta):
        tta_ds = _make_tta_ds_float(base_ds_u8, preprocess_fn, mode)
        preds = model.predict(tta_ds, batch_size=None, verbose=0)
        acc += preds.astype(np.float32, copy=False)
    acc /= float(tta)
    return acc


test_encodings = []

for model in [m1, m3, m4]:
    if model is None:
        continue
    test_encodings.append(
        predict_tta_from_base_ds(
            model,
            base256_ds_u8,
            test_time_augmentation_fn_1,
            mode="g1",
            batch_size=32,
            tta=TTA,
            n_classes=10,
        )
    )

if m2 is not None:
    test_encodings.append(
        predict_tta_from_base_ds(
            m2,
            base300_ds_u8,
            test_time_augmentation_fn_3,
            mode="g3",
            batch_size=16,
            tta=TTA,
            n_classes=10,
        )
    )

if m5 is not None:
    test_encodings.append(
        predict_tta_from_base_ds(
            m5,
            base384_ds_u8,
            test_time_augmentation_fn_2,
            mode="g2",
            batch_size=16,
            tta=TTA,
            n_classes=10,
        )
    )

if len(test_encodings) == 0:
    raise RuntimeError(
        "No models available for inference; cannot generate predictions."
    )

print("Ensemble members used:", len(test_encodings))



## === cell 8
predict_max = [np.argmax(test_enc, axis=1) for test_enc in test_encodings]

class_indices = {
    "bacterial_leaf_blight": 0,
    "bacterial_leaf_streak": 1,
    "bacterial_panicle_blight": 2,
    "blast": 3,
    "brown_spot": 4,
    "dead_heart": 5,
    "downy_mildew": 6,
    "hispa": 7,
    "normal": 8,
    "tungro": 9,
}
inverse_map = {v: k for k, v in class_indices.items()}

predictions = []
for enc in predict_max:
    predictions.append([inverse_map[int(k)] for k in enc])

s_full = pd.DataFrame({"image_id": image_ids})
for i, pres in enumerate(predictions):
    s_full[f"m{i+1}"] = pres

s_full = s_full.set_index("image_id")
s_full_mode = ss.mode(s_full, axis=1, keepdims=False)
final = pd.DataFrame({"image_id": s_full.index.values, "label": s_full_mode.mode})



## === cell 9
sample = pd.read_csv(SAMPLE_SUB)

final = sample[["image_id"]].merge(final, on="image_id", how="left")
final["label"] = final["label"].fillna("normal")

out_path = "model_submission_v21.csv"
final.to_csv(out_path, index=False)

print(final.head())
print("Saved submission to", out_path, "with shape:", final.shape)
print("Submission columns:", list(final.columns))
assert list(final.columns) == ["image_id", "label"]
assert out_path.endswith(".csv")
assert final.shape[0] == sample.shape[0]
