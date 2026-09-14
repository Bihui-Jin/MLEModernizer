# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
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
scipy==1.15.3
seaborn==0.12.2
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Target score

0.95717

# 6. Current score

0.37087

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.16817) has done: 'The timeout is dominated by training VGG16 for 50 epochs at 224×224 with heavy augmentation and a relatively slow input pipeline. I keep the exact same model and training loop, but make the data pipeline faster by fusing map steps, moving augmentation into the compiled graph, enabling prefetch-to-device, and ensuring the on-disk caches are reused across runs without extra work. I also enable XLA JIT (safe for this model) and ensure the base VGG16 is executed in inference mode during training (trainable=False behavior), which reduces overhead without changing the architecture or loss. These changes preserve semantics (same inputs/labels, same augment ops, same epochs, same callbacks) while reducing per-step latency substantially.'
- What this solution (achieved 0.37087) has done: 'I fix the immediate TensorFlow import crash caused by an incompatible protobuf runtime by forcing TensorFlow to use the pure-Python protobuf implementation before importing it. Then I fix a major training logic bug where almost the entire VGG16 backbone was unintentionally left trainable (causing slow/unstable training and very poor score) by correctly freezing the whole base model as intended by the previous optimization plan. Finally, I keep the same model/loop and ensure we still write a correctly formatted `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.37087) has done: 'You’re currently crashing before any training because TensorFlow 2.18 is tripping over an incompatible protobuf runtime (`MessageFactory.GetPrototype` missing). I fix this by forcing a compatible protobuf path in-process (prefer the pure-Python protobuf and disable the C++ implementation) and by pinning `protobuf<5` at runtime if needed, before importing TensorFlow. This is score-neutral but unblocks the entire pipeline so it can train and create `submission.csv`. I also make a tiny robustness fix in weight loading (use `load_model` when `ModelCheckpoint` saved a full model) to ensure best checkpoint is actually used at inference, which should improve score toward the target without changing the model/training logic.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            print(
                f"Detected protobuf {pb_ver}; installing protobuf<5 for TF compatibility..."
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
            )
            import importlib

            importlib.invalidate_caches()
    except Exception as e:
        print("Warning: protobuf compatibility step encountered an issue:", repr(e))


_ensure_protobuf_compat()

import numpy as np
import pandas as pd
import tensorflow as tf

print("TF version:", tf.__version__)

SEED = 1337
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("Warning: could not enable XLA JIT:", e)



## === cell 1
base_candidates = [
    "/kaggle/input/plant-seedlings-classification",
    "/kaggle/input/plant-seedlings-classification/plant-seedlings-classification",
]
base_dir = None
for cand in base_candidates:
    if os.path.isdir(os.path.join(cand, "train")) and os.path.isdir(
        os.path.join(cand, "test")
    ):
        base_dir = cand
        break
if base_dir is None:
    raise FileNotFoundError(
        "Could not locate train/test directories under expected Kaggle input paths."
    )

train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

img_size = 224
batch_size = 32

AUTOTUNE = tf.data.AUTOTUNE

data_opts = tf.data.Options()
data_opts.experimental_deterministic = True  # keep stable ordering/repro
data_opts.experimental_optimization.map_parallelization = True
data_opts.experimental_optimization.parallel_batch = True
data_opts.experimental_optimization.map_and_batch_fusion = True

try:
    data_opts.experimental_prefetch_to_device = "/device:GPU:0"
except Exception:
    pass

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="categorical",
    validation_split=0.25,
    subset="training",
    seed=SEED,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="categorical",
    validation_split=0.25,
    subset="validation",
    seed=SEED,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=False,
)

class_names = list(train_ds.class_names)
num_classes = len(class_names)
class_indices = {name: i for i, name in enumerate(class_names)}
print("Detected num_classes:", num_classes)
print("Class indices:", class_indices)

train_ds = train_ds.with_options(data_opts)
val_ds = val_ds.with_options(data_opts)

vgg_preprocess = tf.keras.applications.vgg16.preprocess_input

augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(factor=30.0 / 180.0, seed=SEED),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), seed=SEED
        ),
        tf.keras.layers.RandomFlip(mode="horizontal", seed=SEED),
        tf.keras.layers.RandomBrightness(
            factor=(0.5, 1.2), value_range=(0.0, 255.0), seed=SEED
        ),
    ],
    name="augment",
)

cache_dir = "/kaggle/working/tf_cache"
os.makedirs(cache_dir, exist_ok=True)
train_cache_path = os.path.join(cache_dir, "train_base.cache")
val_cache_path = os.path.join(cache_dir, "val_base.cache")


@tf.function
def _train_map_fused(x, y):
    x = tf.cast(x, tf.float32)
    x = vgg_preprocess(x)
    x = tf.ensure_shape(x, (None, img_size, img_size, 3))
    y = tf.ensure_shape(y, (None, num_classes))
    x = augment(x, training=True)
    x = tf.ensure_shape(x, (None, img_size, img_size, 3))
    return x, y


@tf.function
def _val_map(x, y):
    x = tf.cast(x, tf.float32)
    x = vgg_preprocess(x)
    x = tf.ensure_shape(x, (None, img_size, img_size, 3))
    y = tf.ensure_shape(y, (None, num_classes))
    return x, y


train_ds = (
    train_ds.map(_train_map_fused, num_parallel_calls=AUTOTUNE)
    .cache(train_cache_path)
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds.map(_val_map, num_parallel_calls=AUTOTUNE)
    .cache(val_cache_path)
    .prefetch(AUTOTUNE)
)

test_files = tf.io.gfile.glob(os.path.join(test_dir, "*.png"))
test_files = sorted(test_files)  # preserve deterministic order


@tf.function
def _load_test(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(
        img, (img_size, img_size), method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    img = vgg_preprocess(img)
    img = tf.ensure_shape(img, (img_size, img_size, 3))
    return img


test_ds = (
    tf.data.Dataset.from_tensor_slices(test_files)
    .with_options(data_opts)
    .map(_load_test, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

test_filenames = [os.path.basename(p) for p in test_files]
print("Num test files:", len(test_filenames))



## === cell 2
pass



## === cell 3
base_model_vgg16 = tf.keras.applications.VGG16(
    include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
)



## === cell 4
pass



## === cell 5
base_model_vgg16.trainable = False



## === cell 6
pass



## === cell 7
model_vgg16 = tf.keras.models.Sequential(
    layers=[
        base_model_vgg16,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)



## === cell 8
model_vgg16.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=10,
)



## === cell 9
model_name = "model_vgg16.h5"
Checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_name, monitor="val_loss", mode="min", save_best_only=True, verbose=1
)

lrr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", patience=3, verbose=1, factor=0.3, min_lr=0.00000001
)
cb_List = [Checkpoint, lrr]



## === cell 10
pass



## === cell 11
EPOCH = 50

history_vgg16 = model_vgg16.fit(
    train_ds,
    epochs=EPOCH,
    validation_data=val_ds,
    callbacks=cb_List,
)



## === cell 12
pass



## === cell 13
pass



## === cell 14
pass



## === cell 15
pass



## === cell 16
if os.path.exists(model_name):
    try:
        loaded = tf.keras.models.load_model(model_name, compile=False)
        model_vgg16 = loaded
        model_vgg16.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
            steps_per_execution=10,
        )
        print(f"Loaded best model from {model_name} for inference.")
    except Exception as e:
        print(f"Warning: could not load full model from {model_name}: {e}")
        try:
            model_vgg16.load_weights(model_name)
            print(f"Loaded best weights from {model_name} for inference (fallback).")
        except Exception as e2:
            print(f"Warning: could not load weights from {model_name}: {e2}")

idx_to_class = {v: k for k, v in class_indices.items()}

preds = model_vgg16.predict(test_ds, verbose=1)
pred_idx = np.argmax(preds, axis=1)
class_list = [idx_to_class[int(i)] for i in pred_idx]

submission = pd.DataFrame({"file": test_filenames, "species": class_list})

sample_path_candidates = [
    os.path.join(base_dir, "sample_submission.csv"),
    "/kaggle/input/sample_submission.csv",
]
sample_path = None
for p in sample_path_candidates:
    if os.path.exists(p):
        sample_path = p
        break

if sample_path is not None:
    sample_sub = pd.read_csv(sample_path)
    if "file" in sample_sub.columns:
        submission = sample_sub[["file"]].merge(submission, on="file", how="left")
        if submission["species"].isna().any():
            fallback = pd.Series(class_list).mode().iloc[0]
            submission["species"] = submission["species"].fillna(fallback)



## === cell 17
submission.head(5)



## === cell 18
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.columns.tolist())
print("Unique predicted species:", submission["species"].nunique())
print("First rows:\n", submission.head())
