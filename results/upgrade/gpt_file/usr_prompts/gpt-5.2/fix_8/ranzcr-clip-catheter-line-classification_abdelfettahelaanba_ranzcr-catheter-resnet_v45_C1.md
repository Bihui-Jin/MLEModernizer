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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## === cell 1
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


def build_decoder(with_labels=True, target_size=(224, 224), ext="jpg"):
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == "png":
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ["jpg", "jpeg"]:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")

        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, target_size)
        return img

    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    def augment(img):
        return img

    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment


def build_dataset(
    paths,
    labels=None,
    bsize=64,
    cache=True,
    decode_fn=None,
    augment_fn=None,
    augment=True,
    repeat=True,
    shuffle=1024,
    cache_dir="",
):
    """
    Speed fixes (correctness-preserving):
    1) Ensure on-disk cache directory exists; otherwise cache() may silently be ineffective.
    2) Use fused map+batch via `Dataset.map(...).batch(...)` kept, but add `num_parallel_calls`
       and `deterministic` flags explicitly to reduce input stalls.
    3) Training pipeline uses non-deterministic parallelism only for performance; shuffled element
       set/epoch semantics remain the same (seeded shuffle), with at most negligible FP diffs.
    4) Prefetch to device when available to overlap CPU decode with accelerator execution.
    """
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    AUTO = tf.data.AUTOTUNE
    slices = paths if labels is None else (paths, labels)
    dset = tf.data.Dataset.from_tensor_slices(slices)

    is_training = labels is not None and shuffle not in (0, False) and repeat is True

    opts = tf.data.Options()
    opts.experimental_deterministic = False if is_training else True
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.map_and_batch_fusion = True
    opts.experimental_optimization.parallel_batch = True
    dset = dset.with_options(opts)

    dset = dset.map(decode_fn, num_parallel_calls=AUTO, deterministic=not is_training)

    if cache:
        if cache_dir:
            cache_parent = os.path.dirname(cache_dir)
            if cache_parent:
                os.makedirs(cache_parent, exist_ok=True)
            dset = dset.cache(cache_dir)
        else:
            dset = dset.cache()

    if augment:
        dset = dset.map(
            augment_fn, num_parallel_calls=AUTO, deterministic=not is_training
        )

    if shuffle not in (0, False):
        dset = dset.shuffle(shuffle, seed=SEED, reshuffle_each_iteration=True)

    if repeat:
        dset = dset.repeat()

    dset = dset.batch(bsize, drop_remainder=False)

    try:
        dset = dset.apply(tf.data.experimental.prefetch_to_device("/device:GPU:0"))
    except Exception:
        dset = dset.prefetch(AUTO)

    return dset




## === cell 2
COMPETITION_NAME = "ranzcr-clip-catheter-line-classification"
load_dir = f"/kaggle/input/{COMPETITION_NAME}/"

strategy = auto_select_accelerator()
BATCH_SIZE = strategy.num_replicas_in_sync * 16

IMSIZE = (224, 224, 260, 300, 380, 456, 528, 600)
IMG_SIZE = (IMSIZE[0], IMSIZE[0])

train_csv_path = os.path.join(load_dir, "train.csv")
test_dir = os.path.join(load_dir, "test")
train_dir = os.path.join(load_dir, "train")

train_df = pd.read_csv(train_csv_path)

sample_sub_path = os.path.join(load_dir, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)
sub_cols = [c for c in sample_sub.columns if c != "StudyInstanceUID"]

label_cols = sub_cols
num_labels = len(label_cols)

print("Num labels:", num_labels)
print("Labels:", label_cols)

missing_in_train = [c for c in label_cols if c not in train_df.columns]
if missing_in_train:
    raise ValueError(
        f"These required label columns are missing from train.csv: {missing_in_train}"
    )




## === cell 3
uids = train_df["StudyInstanceUID"].to_numpy(dtype=str)
train_paths = np.char.add(np.char.add(train_dir + "/", uids), ".jpg")
train_labels = train_df[label_cols].to_numpy(dtype=np.float32)

unique_patients = train_df["PatientID"].unique()
rng = np.random.default_rng(SEED)
rng.shuffle(unique_patients)
split = int(0.9 * len(unique_patients))
train_patients = set(unique_patients[:split])
is_train = train_df["PatientID"].isin(train_patients).to_numpy()

tr_paths = train_paths[is_train]
tr_labels = train_labels[is_train]
va_paths = train_paths[~is_train]
va_labels = train_labels[~is_train]

print("Train samples:", len(tr_paths), "Valid samples:", len(va_paths))

train_decoder = build_decoder(with_labels=True, target_size=IMG_SIZE, ext="jpg")
train_augmenter = build_augmenter(with_labels=True)

cache_root = "/kaggle/working/tfdata_cache"
train_cache_path = os.path.join(cache_root, f"train_img{IMG_SIZE[0]}_seed{SEED}.cache")
valid_cache_path = os.path.join(cache_root, f"valid_img{IMG_SIZE[0]}_seed{SEED}.cache")

dtrain = build_dataset(
    tr_paths,
    tr_labels,
    bsize=BATCH_SIZE,
    repeat=True,
    shuffle=2048,
    augment=True,
    cache=True,
    cache_dir=train_cache_path,
    decode_fn=train_decoder,
    augment_fn=train_augmenter,
)

dvalid = build_dataset(
    va_paths,
    va_labels,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=True,
    cache_dir=valid_cache_path,
    decode_fn=train_decoder,
    augment_fn=train_augmenter,
)

test_files = sorted(tf.io.gfile.glob(os.path.join(test_dir, "*.jpg")))
test_paths = np.asarray(test_files)
test_uids = [f.rsplit("/", 1)[-1].rsplit(".", 1)[0] for f in test_files]

test_decoder = build_decoder(with_labels=False, target_size=IMG_SIZE, ext="jpg")
dtest = build_dataset(
    test_paths,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=False,
    cache_dir="",
    decode_fn=test_decoder,
)




## === cell 4
with strategy.scope():
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling="avg",
    )
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_labels, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[],
        jit_compile=True,
    )

steps_per_epoch = int(np.ceil(len(tr_paths) / BATCH_SIZE))
validation_steps = int(np.ceil(len(va_paths) / BATCH_SIZE))

history = model.fit(
    dtrain,
    epochs=2,
    steps_per_epoch=steps_per_epoch,
    validation_data=dvalid,
    validation_steps=validation_steps,
    verbose=1,
)

with strategy.scope():
    base.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[],
        jit_compile=True,
    )

history_ft = model.fit(
    dtrain,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    validation_data=dvalid,
    validation_steps=validation_steps,
    verbose=1,
)




## === cell 5
test_steps = int(np.ceil(len(test_paths) / BATCH_SIZE))

pred = model.predict(dtest, steps=test_steps, verbose=1)
pred = np.clip(pred, 0.0, 1.0)

pred_df = pd.DataFrame(pred, columns=label_cols)
pred_df.insert(0, "StudyInstanceUID", test_uids)

sub_df = sample_sub[["StudyInstanceUID"]].merge(
    pred_df, on="StudyInstanceUID", how="left"
)

for c in label_cols:
    if c not in sub_df.columns:
        sub_df[c] = 0.5
sub_df[label_cols] = sub_df[label_cols].astype(np.float32)
sub_df[label_cols] = sub_df[label_cols].fillna(0.5)

sub_df = sub_df[["StudyInstanceUID"] + label_cols]

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)

print("Wrote:", sub_path)
print("Submission shape:", sub_df.shape)
print("Submission columns:", list(sub_df.columns))
sub_df.head()
