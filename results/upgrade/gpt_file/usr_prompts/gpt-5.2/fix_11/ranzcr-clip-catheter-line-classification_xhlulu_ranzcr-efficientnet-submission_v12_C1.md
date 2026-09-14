# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    pass

import numpy as np
import pandas as pd
import tensorflow as tf



## === cell 1
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.optimizer.set_jit(True)  # allow XLA where supported
try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 2
def auto_select_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except (ValueError, tf.errors.NotFoundError):
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy


def build_decoder(with_labels=True, target_size=(300, 300), ext="jpg"):
    h, w = target_size
    ext_l = ext.lower()

    @tf.function(reduce_retracing=True)
    def decode_only(path):
        file_bytes = tf.io.read_file(path)
        if ext_l == "png":
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext_l in ["jpg", "jpeg"]:
            img = tf.io.decode_jpeg(file_bytes, channels=3, dct_method="INTEGER_FAST")
        else:
            raise ValueError("Image extension not supported")
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, target_size)
        img.set_shape((h, w, 3))
        return img

    if with_labels:

        @tf.function(reduce_retracing=True)
        def decode_with_labels(path, label):
            return decode_only(path), label

        return decode_with_labels

    return decode_only


def build_augmenter(with_labels=True):
    @tf.function(reduce_retracing=True)
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img

    @tf.function(reduce_retracing=True)
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
    drop_remainder=False,
    deterministic=True,
):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)

    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    AUTO = tf.data.AUTOTUNE
    slices = paths if labels is None else (paths, labels)
    dset = tf.data.Dataset.from_tensor_slices(slices)

    options = tf.data.Options()
    options.experimental_deterministic = bool(deterministic)
    options.experimental_optimization.apply_default_optimizations = True
    options.autotune.enabled = True
    options.autotune.cpu_budget = 0  # let TF decide
    options.autotune.ram_budget = 0  # let TF decide
    try:
        options.threading.private_threadpool_size = 0
        options.threading.max_intra_op_parallelism = 0
    except Exception:
        pass
    dset = dset.with_options(options)

    if shuffle:
        dset = dset.shuffle(shuffle, seed=SEED, reshuffle_each_iteration=True)

    if repeat:
        dset = dset.repeat()

    dset = dset.map(decode_fn, num_parallel_calls=AUTO, deterministic=deterministic)
    dset = dset.apply(tf.data.experimental.ignore_errors())

    if cache:
        dset = dset.cache(cache_dir) if cache_dir != "" else dset.cache()

    if augment:
        dset = dset.map(
            augment_fn, num_parallel_calls=AUTO, deterministic=deterministic
        )

    dset = dset.batch(bsize, drop_remainder=drop_remainder)
    dset = dset.prefetch(AUTO)
    return dset




## === cell 3
COMPETITION_NAME = "ranzcr-clip-catheter-line-classification"
strategy = auto_select_accelerator()

BATCH_SIZE = strategy.num_replicas_in_sync * 32



## === cell 4
load_dir = f"/kaggle/input/{COMPETITION_NAME}/"

TARGET_COLS = [
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

sub_df = pd.read_csv(os.path.join(load_dir, "sample_submission.csv"))
for c in TARGET_COLS:
    if c not in sub_df.columns:
        sub_df[c] = 0.0
sub_df = sub_df[["StudyInstanceUID"] + TARGET_COLS]

test_paths = (
    load_dir + "test/" + sub_df["StudyInstanceUID"].astype(str) + ".jpg"
).values

IMSIZE = (224, 240, 260, 300, 380, 456, 528, 600)
IMG_SIZE = (IMSIZE[5], IMSIZE[5])  # keep original choice (456x456)

test_decoder = build_decoder(with_labels=False, target_size=IMG_SIZE)

dtest = build_dataset(
    test_paths,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=False,
    cache_dir="",
    decode_fn=test_decoder,
    drop_remainder=False,
    deterministic=False,
)



## === cell 5
pretrained_path = "/kaggle/input/ranzcr-efficientnet-tpu-training/model.h5"
NEED_TRAIN = not tf.io.gfile.exists(pretrained_path)

if NEED_TRAIN:
    train_csv_path = os.path.join(load_dir, "train.csv")
    train_df = pd.read_csv(train_csv_path)

    missing_train_cols = [c for c in TARGET_COLS if c not in train_df.columns]
    if missing_train_cols:
        raise ValueError(
            f"train.csv missing expected target columns: {missing_train_cols}"
        )

    train_df = train_df[["StudyInstanceUID", "PatientID"] + TARGET_COLS].copy()
    train_paths = (
        load_dir + "train/" + train_df["StudyInstanceUID"].astype(str) + ".jpg"
    ).values
    train_labels = train_df[TARGET_COLS].astype("float32").values

    n = len(train_df)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)

    val_frac = 0.1
    val_n = int(n * val_frac)
    val_idx = idx[:val_n]
    trn_idx = idx[val_n:]

    trn_paths = train_paths[trn_idx]
    trn_labels = train_labels[trn_idx]
    val_paths = train_paths[val_idx]
    val_labels = train_labels[val_idx]

    train_decoder = build_decoder(with_labels=True, target_size=IMG_SIZE)
    val_decoder = build_decoder(with_labels=True, target_size=IMG_SIZE)

    dtrain = build_dataset(
        trn_paths,
        trn_labels,
        bsize=BATCH_SIZE,
        repeat=True,
        shuffle=2048,
        augment=True,
        cache=False,
        cache_dir="",
        decode_fn=train_decoder,
        drop_remainder=True,
        deterministic=True,
    )
    dval = build_dataset(
        val_paths,
        val_labels,
        bsize=BATCH_SIZE,
        repeat=False,
        shuffle=False,
        augment=False,
        cache=False,
        cache_dir="",
        decode_fn=val_decoder,
        drop_remainder=False,
        deterministic=True,
    )

    steps_per_epoch = max(1, len(trn_paths) // BATCH_SIZE)
    val_steps = max(1, int(np.ceil(len(val_paths) / BATCH_SIZE)))

with strategy.scope():
    if tf.io.gfile.exists(pretrained_path):
        model = tf.keras.models.load_model(pretrained_path)
    else:
        inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(len(TARGET_COLS), activation="sigmoid")(x)
        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            jit_compile=True,
        )

model.summary()

if not tf.io.gfile.exists(pretrained_path):
    EPOCHS = 2  # keep original training choice
    _ = model.fit(
        dtrain,
        validation_data=dval,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        verbose=1,
    )

pred = model.predict(
    dtest, verbose=0, max_queue_size=256, workers=1, use_multiprocessing=False
)
pred = np.asarray(pred, dtype=np.float32)

if pred.ndim != 2 or pred.shape[1] != len(TARGET_COLS):
    raise ValueError(
        f"Prediction shape mismatch. Got {pred.shape}, expected (N, {len(TARGET_COLS)})"
    )

sub_df[TARGET_COLS] = pred
sub_df.to_csv("submission.csv", index=False)

sub_df.head()
