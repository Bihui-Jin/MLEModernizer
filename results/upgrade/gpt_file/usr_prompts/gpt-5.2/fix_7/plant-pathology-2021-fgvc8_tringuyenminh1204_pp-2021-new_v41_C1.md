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

# 5. Code solution

## === cell 0
import os, re, math, random, pathlib
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K

print("TF:", tf.__version__)
print("Keras:", tf.keras.__version__ if hasattr(tf.keras, "__version__") else "bundled")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
except Exception:
    pass




## === cell 1
import pathlib




## === cell 2
@tf.function
def _decode_image_core(bits, image_size):
    image = tf.image.decode_jpeg(bits, channels=3)  # JPEG-only for speed (unchanged)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    return image


def make_decode_fn(image_size=(512, 512)):
    image_size_t = tf.constant(image_size, dtype=tf.int32)

    def _decode(filename, label=None):
        bits = tf.io.read_file(filename)
        image = _decode_image_core(bits, image_size_t)
        if label is None:
            return image
        else:
            return image, label

    return _decode




## === cell 3
BATCH_SIZE = 32




## === cell 4
source = "../input/plant-pathology-2021-fgvc8/test_images"

IMAGE_PATHS = tf.io.gfile.glob(os.path.join(source, "*.jpg"))
if not IMAGE_PATHS:
    IMAGE_PATHS = tf.io.gfile.glob(os.path.join(source, "*.JPG"))
IMAGE_PATHS = sorted(IMAGE_PATHS)

IMAGE_FILES = [os.path.basename(p) for p in IMAGE_PATHS]

print("Num test images found:", len(IMAGE_PATHS))
print("First 3:", IMAGE_FILES[:3])




## === cell 5
IMAGE_PATHS[:5]




## === cell 6
AUTO = tf.data.experimental.AUTOTUNE




## === cell 7
options = tf.data.Options()
options.experimental_deterministic = True  # preserve determinism
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_fusion = True
options.experimental_optimization.map_parallelization = True

IMAGE_SIZE = (512, 512)
decode_image = make_decode_fn(image_size=IMAGE_SIZE)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)




## === cell 8
import tensorflow as tf
from tensorflow import keras




## === cell 9
class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)




## === cell 10
def find_model_file():
    candidates = [
        "../input/3smresnet50/3SMResNet50.h5",
        "/kaggle/input/3smresnet50/3SMResNet50.h5",
        "../input/3smresnet50/3SMResNet50.hdf5",
        "/kaggle/input/3smresnet50/3SMResNet50.hdf5",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    roots = [
        "../input",
        "/kaggle/input",
    ]
    target_names = {
        "3SMResNet50.h5",
        "3SMResNet50.hdf5",
        "3smresnet50.h5",
        "3smresnet50.hdf5",
    }

    for root in roots:
        if not os.path.isdir(root):
            continue
        try:
            for ds in os.listdir(root):
                ds_path = os.path.join(root, ds)
                if not os.path.isdir(ds_path):
                    continue
                for fn in target_names:
                    p = os.path.join(ds_path, fn)
                    if os.path.exists(p):
                        return p
        except Exception:
            pass

    shallow_roots = [
        "../input/3smresnet50",
        "/kaggle/input/3smresnet50",
    ]
    for root in shallow_roots:
        if os.path.isdir(root):
            for fn in target_names:
                p = os.path.join(root, fn)
                if os.path.exists(p):
                    return p
    return None


model_path = find_model_file()
print("Found external model:", model_path)




## === cell 11
TRAIN_CSV = "../input/plant-pathology-2021-fgvc8/train.csv"
TRAIN_IMG_DIR = "../input/plant-pathology-2021-fgvc8/train_images"

train_df = None
classes = None
class_to_idx = None
num_classes = None
Y = None




## === cell 12
def build_fallback_model(input_shape=(512, 512, 3), num_classes=6):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg"
    )
    base.trainable = False  # keep fast; avoids exceeding 600s

    inp = tf.keras.Input(shape=input_shape)
    x = inp
    x = tf.keras.applications.efficientnet.preprocess_input(x * 255.0)
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.2, seed=SEED)(x)
    out = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    return model




## === cell 13
if model_path is not None:
    print("Loading model from:", model_path)
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"FixedDropout": FixedDropout},
    )
    print("Loaded external model. Output units:", model.output_shape)
else:
    train_df = pd.read_csv(TRAIN_CSV)
    train_df["image_path"] = train_df["image"].apply(
        lambda x: os.path.join(TRAIN_IMG_DIR, x)
    )

    all_labels = set()
    for s in train_df["labels"].astype(str).tolist():
        for t in s.split():
            if t:
                all_labels.add(t)

    preferred_order = [
        "scab",
        "frog_eye_leaf_spot",
        "complex",
        "rust",
        "powdery_mildew",
        "healthy",
    ]
    classes = [c for c in preferred_order if c in all_labels] + sorted(
        [c for c in all_labels if c not in preferred_order]
    )

    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    def encode_labels(label_str):
        y = np.zeros((num_classes,), dtype=np.float32)
        for t in str(label_str).split():
            if t in class_to_idx:
                y[class_to_idx[t]] = 1.0
        return y

    Y = np.stack([encode_labels(s) for s in train_df["labels"].tolist()], axis=0)

    print("Train rows:", len(train_df))
    print("Num classes:", num_classes)
    print("Classes:", classes)

    idx = np.arange(len(train_df))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)

    val_frac = 0.1
    val_size = int(len(idx) * val_frac)
    val_idx = idx[:val_size]
    trn_idx = idx[val_size:]

    trn_paths = train_df.iloc[trn_idx]["image_path"].tolist()
    val_paths = train_df.iloc[val_idx]["image_path"].tolist()

    trn_y = Y[trn_idx]
    val_y = Y[val_idx]

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((trn_paths, trn_y))
        .with_options(options)
        .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        .map(
            lambda p, y: (decode_image(p, y)[0], decode_image(p, y)[1]),
            num_parallel_calls=AUTO,
        )
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTO)
    )

    valid_dataset = (
        tf.data.Dataset.from_tensor_slices((val_paths, val_y))
        .with_options(options)
        .map(
            lambda p, y: (decode_image(p, y)[0], decode_image(p, y)[1]),
            num_parallel_calls=AUTO,
        )
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTO)
    )

    model = build_fallback_model(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=num_classes
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )
    EPOCHS = 2
    print("Training fallback model for", EPOCHS, "epochs...")
    model.fit(train_dataset, validation_data=valid_dataset, epochs=EPOCHS, verbose=1)




## === cell 14
probs = model.predict(test_dataset, verbose=1)
temp_probs = np.asarray(probs, dtype=np.float32)
print("Pred shape:", temp_probs.shape)




## === cell 15
default_name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",
}
threshold = {0: 0.35, 1: 0.35, 2: 0.35, 3: 0.35, 4: 0.35}

if model_path is None:
    desired = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew"]
    disease_indices = [class_to_idx[c] for c in desired if c in class_to_idx]
    healthy_idx = class_to_idx["healthy"] if "healthy" in class_to_idx else None
else:
    disease_indices = [0, 1, 2, 3, 4]
    healthy_idx = 5 if temp_probs.shape[1] > 5 else None

if len(disease_indices) != 5 or temp_probs.shape[1] <= max(disease_indices):
    C = temp_probs.shape[1]
    disease_indices = list(range(min(5, C)))
    healthy_idx = C - 1  # best-effort

p5 = temp_probs[:, disease_indices]  # (N,5)
thr = np.array([threshold[i] for i in range(min(5, p5.shape[1]))], dtype=np.float32)[
    None, :
]
mask = p5 > thr  # (N,5) boolean

complex_col = 2 if p5.shape[1] > 2 else None
cnt = mask.sum(axis=1)
need_complex = (
    (cnt >= 2) & (~mask[:, complex_col])
    if complex_col is not None
    else np.zeros((mask.shape[0],), dtype=bool)
)

slot_names = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew"][
    : p5.shape[1]
]

labels_per_row = []
for row_mask, add_c in zip(mask, need_complex):
    labs = [slot_names[i] for i in np.flatnonzero(row_mask)]
    if add_c and "complex" not in labs:
        labs.append("complex")
    if not labs:
        labs = ["healthy"]
    labels_per_row.append(" ".join(labs))

pred_string = labels_per_row




## === cell 16
pred_string[:10], len(pred_string)




## === cell 17
df = pd.DataFrame({"image": IMAGE_FILES, "labels": pred_string})
df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("Submission path:", os.path.abspath("submission.csv"))
