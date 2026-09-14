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
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

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
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Target score

0.9542400357733768

# 6. Current score

0.84501

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the immediate TensorFlow import crash caused by an incompatibility between the installed protobuf version and TensorFlow by forcing the pure-Python protobuf implementation before importing TensorFlow. Then I make the model-loading path robust: instead of a hardcoded `../input/ranzcr-tpu-weights/model.h5` (which doesn’t exist in your dataset), the script search common Kaggle input locations for a `.h5/.keras/SavedModel` and load it if found; if not, it fall back to producing a valid submission using safe constant probabilities. Finally, I ensure the submission has exactly the columns from `sample_submission.csv`, the prediction shape matches, and the output is written as `submission.csv` in the working directory.'
- What this solution (achieved 0.5) has done: 'We fix the TensorFlow import crash by pinning protobuf to the compatible pure-Python + version used by TF2.18 at runtime (via `pip`), and only then import TensorFlow. Then we fix the submission-column mismatch by ensuring we always submit the exact 11 target columns required by the competition (adding any missing ones like `CVC - Normal` and `Swan Ganz Catheter Present` when the provided `sample_submission.csv` has only 9 targets). Finally, we keep your fallback behavior (constant predictions) but make it produce correctly-shaped predictions aligned to the corrected target list so the pipeline runs end-to-end and yields a valid `submission.csv`.'
- What this solution (achieved 0.50196) has done: 'The timeout is dominated by input pipeline overhead (Python-side file existence checks; JPEG decode/resize done every epoch without caching) and by slow single-threaded-ish data feeding relative to GPU/TPU compute. I keep the exact same model/training setup, but (1) replace the TensorFlow `gfile.exists` loop with a much faster `os.path.exists` vectorized-ish loop, (2) switch the `tf.data` pipeline to non-deterministic map ordering (same random seed still controls shuffling/augment RNG) to unlock higher throughput, and (3) enable on-disk caching for decoded images (before augmentation) so epoch 2 and validation/prediction don’t redo decode+resize work. These changes preserve evaluation semantics (same data, same labels, same architecture/loss/epochs/steps) and should reduce wall time substantially under the 600s limit.'
- What this solution (achieved 0.81737) has done: 'Your score is far below the target (0.50196 vs 0.95424), so we should improve performance with the smallest possible change that keeps the same model and training approach. The current biggest issue is that the model trains only a frozen ImageNet backbone for 2 epochs, which typically won’t learn this domain well; we keep the same architecture but unfreeze only the EfficientNetB0 top blocks after a short warmup and continue training with a lower learning rate (a standard fine-tuning step that preserves core logic). We also switch preprocessing to use EfficientNet’s `preprocess_input` (instead of simple `/255`) to better match the ImageNet initialization while keeping the same inputs and labels. These changes should move AUC substantially upward toward the target without changing the overall pipeline or submission semantics.'
- What this solution (achieved 0.84501) has done: 'Your gap to the target is large (0.817 → 0.954), so we need a modest, low-risk lift without changing the overall pipeline. The biggest lever within your existing logic is fine-tuning duration: right now you only fine-tune for 1 extra epoch, which is typically insufficient; I increase only the fine-tune phase to a small number of epochs while keeping the same EfficientNetB0 head, losses, data pipeline, and warmup intact. I also make the fine-tune unfreezing slightly less restrictive (unfreeze more top layers) while still keeping BatchNorm frozen to preserve stability. These two changes usually improve AUC materially on this task and should move you closer to the target while staying within runtime limits.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as pb_ver

    major = int(pb_ver.split(".")[0])
    if major >= 5:
        raise RuntimeError(f"Incompatible protobuf version detected: {pb_ver}")
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

import numpy as np
import pandas as pd
import tensorflow as tf

print("Python:", sys.version.split()[0])
print("TensorFlow:", tf.__version__)

SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

img_size = 224


def auto_select_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except Exception:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy


def build_decoder(with_labels=True, target_size=(img_size, img_size), ext="jpg"):
    preprocess = tf.keras.applications.efficientnet.preprocess_input

    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == "png":
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ["jpg", "jpeg"]:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")

        img = tf.image.resize(img, target_size)
        img = tf.cast(img, tf.float32)
        img = preprocess(img)
        return img

    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.2)
        return img

    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment


def build_dataset(
    paths,
    labels=None,
    bsize=32,
    cache=True,
    decode_fn=None,
    augment_fn=None,
    augment=True,
    repeat=True,
    shuffle=1024,
    cache_dir="",
):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)

    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    AUTO = tf.data.AUTOTUNE
    slices = paths if labels is None else (paths, labels)

    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.parallel_batch = True
    options.threading.private_threadpool_size = 0
    options.threading.max_intra_op_parallelism = 0

    dset = tf.data.Dataset.from_tensor_slices(slices).with_options(options)
    dset = dset.map(decode_fn, num_parallel_calls=AUTO, deterministic=False)
    dset = dset.cache(cache_dir) if cache else dset
    dset = (
        dset.map(augment_fn, num_parallel_calls=AUTO, deterministic=False)
        if augment
        else dset
    )
    dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle, seed=SEED) if shuffle else dset
    dset = dset.batch(bsize, drop_remainder=False).prefetch(AUTO)
    return dset




## === cell 1
COMPETITION_NAME = "ranzcr-clip-catheter-line-classification"
load_dir = "/kaggle/input/ranzcr-clip-catheter-line-classification/"

train_csv_path = os.path.join(load_dir, "train.csv")
sample_sub_path = os.path.join(load_dir, "sample_submission.csv")
train_img_dir = os.path.join(load_dir, "train")
test_img_dir = os.path.join(load_dir, "test")

df_train = pd.read_csv(train_csv_path)
df_sub = pd.read_csv(sample_sub_path)

REQUIRED_TARGET_COLS = [
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "Swan Ganz Catheter Present",
]

if "StudyInstanceUID" not in df_sub.columns:
    raise ValueError("sample_submission.csv missing StudyInstanceUID column")

for c in REQUIRED_TARGET_COLS:
    if c not in df_sub.columns:
        df_sub[c] = 0.5
df_sub = df_sub[["StudyInstanceUID"] + REQUIRED_TARGET_COLS]

train_paths = (
    train_img_dir + "/" + df_train["StudyInstanceUID"].astype(str) + ".jpg"
).values

exists_mask = np.fromiter(
    (os.path.exists(p) for p in train_paths), dtype=np.bool_, count=len(train_paths)
)
df_train = df_train.loc[exists_mask].reset_index(drop=True)
train_paths = (
    train_img_dir + "/" + df_train["StudyInstanceUID"].astype(str) + ".jpg"
).values

y = df_train[REQUIRED_TARGET_COLS].astype("float32").values

print("Train rows used:", len(df_train))
print("Targets:", REQUIRED_TARGET_COLS)

strategy = auto_select_accelerator()
batch_size = strategy.num_replicas_in_sync * 32
print("batch size", batch_size)

patients = df_train["PatientID"].astype(str).values
uniq_patients = np.unique(patients)
rng = np.random.default_rng(SEED)
rng.shuffle(uniq_patients)

val_frac = 0.1
n_val = max(1, int(len(uniq_patients) * val_frac))
val_patients = set(uniq_patients[:n_val])

is_val = np.isin(patients, np.fromiter(val_patients, dtype=object))
train_idx = np.where(~is_val)[0]
val_idx = np.where(is_val)[0]

train_paths_split = train_paths[train_idx]
y_train = y[train_idx]
val_paths_split = train_paths[val_idx]
y_val = y[val_idx]

print("Train split:", len(train_idx), "Val split:", len(val_idx))

train_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size))
val_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size))
test_decoder = build_decoder(with_labels=False, target_size=(img_size, img_size))

cache_root = "/kaggle/working/tfdata_cache_ranzcr"
train_cache = os.path.join(cache_root, f"train_img_{img_size}.cache")
val_cache = os.path.join(cache_root, f"val_img_{img_size}.cache")
test_cache = os.path.join(cache_root, f"test_img_{img_size}.cache")

dtrain = build_dataset(
    train_paths_split,
    y_train,
    bsize=batch_size,
    repeat=True,
    shuffle=2048,
    augment=True,
    cache=True,
    cache_dir=train_cache,
    decode_fn=train_decoder,
)
dval = build_dataset(
    val_paths_split,
    y_val,
    bsize=batch_size,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=True,
    cache_dir=val_cache,
    decode_fn=val_decoder,
)

test_paths = (
    test_img_dir + "/" + df_sub["StudyInstanceUID"].astype(str) + ".jpg"
).values
dtest = build_dataset(
    test_paths,
    bsize=batch_size,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=True,
    cache_dir=test_cache,
    decode_fn=test_decoder,
)



## === cell 2
with strategy.scope():
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
    )
    base.trainable = False  # warmup unchanged

    inp = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = inp
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(len(REQUIRED_TARGET_COLS), activation="sigmoid")(x)

    model = tf.keras.Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        jit_compile=True,
    )

steps_per_epoch = max(1, len(train_paths_split) // batch_size)
val_steps = max(1, int(np.ceil(len(val_paths_split) / batch_size)))

warmup_epochs = 2  # unchanged
finetune_epochs = 3

print(
    "steps_per_epoch:",
    steps_per_epoch,
    "val_steps:",
    val_steps,
    "warmup_epochs:",
    warmup_epochs,
    "finetune_epochs:",
    finetune_epochs,
)

model.fit(
    dtrain,
    validation_data=dval,
    epochs=warmup_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    jit_compile=True,
)

model.fit(
    dtrain,
    validation_data=dval,
    epochs=warmup_epochs + finetune_epochs,
    initial_epoch=warmup_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)

test_steps = int(np.ceil(len(df_sub) / batch_size))
y_preds = model.predict(dtest, steps=test_steps, verbose=1)
y_preds = np.asarray(y_preds, dtype=np.float32)

n_targets = len(REQUIRED_TARGET_COLS)
if y_preds.ndim == 1:
    y_preds = y_preds.reshape(-1, 1)
if y_preds.shape[0] != len(df_sub):
    min_len = min(y_preds.shape[0], len(df_sub))
    y_preds = y_preds[:min_len]
    df_sub = df_sub.iloc[:min_len].reset_index(drop=True)
if y_preds.shape[1] != n_targets:
    if y_preds.shape[1] > n_targets:
        y_preds = y_preds[:, :n_targets]
    else:
        pad = np.full(
            (y_preds.shape[0], n_targets - y_preds.shape[1]), 0.5, dtype=np.float32
        )
        y_preds = np.concatenate([y_preds, pad], axis=1)

y_preds = np.clip(y_preds, 0.0, 1.0)

df_sub.loc[:, REQUIRED_TARGET_COLS] = y_preds
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print(df_sub.head())
