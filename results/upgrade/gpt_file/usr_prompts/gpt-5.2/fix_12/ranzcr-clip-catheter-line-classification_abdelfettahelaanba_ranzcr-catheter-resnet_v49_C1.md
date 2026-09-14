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

3.10

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

# 5. Target score

0.9310826595540482

# 6. Current score

0.72471

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.72458) has done: 'I remove the biggest sources of overhead that don’t change the model or training semantics: (1) stop forcing the slow pure-Python protobuf implementation, (2) speed up the input pipeline by using static shapes, enabling deterministic execution while still using parallel map/prefetch, and (3) eliminate redundant dataset transformations (repeat/shuffle ordering) that add CPU overhead. I also freeze the ImageNet backbone (set `trainable=False`) since you already run it with `training=False`, which preserves the intended behavior but avoids building/maintaining trainable variables and reduces backprop work. Finally, I ensure `model.predict` uses an explicit `steps` count to avoid any extra bookkeeping and keep the run bounded.'
- What this solution (achieved 0.72466) has done: 'The timeout is dominated by JPEG decode + resize for ~24k training images and ~3k val images per epoch, plus suboptimal input pipeline settings (no parallel reads/interleave, no fused map, and deterministic ordering forces slower scheduling). I keep the exact same model, loss, split, and one-epoch `model.fit`, but make the `tf.data` pipeline faster by (1) using `Dataset.list_files` + `interleave` for parallel file reading, (2) enabling `map_and_batch`/fused batching via `map(...).batch(...)` with AUTOTUNE and `deterministic` only where required, and (3) caching validation decode (small enough) to avoid decoding it twice (during val and potentially any re-iteration). These changes are provably equivalent in semantics (same images/labels, same preprocessing, same ordering for val/test) and typically reduce wall time substantially without changing the learning setup.'
- What this solution (achieved 0.72471) has done: 'The main bottleneck is JPEG decode+resize at a large 380×380 resolution for ~27k training images (and again for val/test), plus avoidable Python-protobuf overhead. To stay within 600s without changing the model or training semantics, I keep the exact same architecture/training loop but speed up the input pipeline by (1) switching protobuf back to the default C++ implementation, (2) using `tf.io.decode_jpeg(..., dct_method='INTEGER_FAST')` (same image content, faster decode), and (3) enabling non-deterministic tf.data execution only for the training pipeline to improve throughput while keeping seeds/determinism for eval/predict. I also avoid expensive on-disk `.cache()` writes for test/val (which often costs more than it saves in a single-epoch run) while preserving identical samples/order. These are provably equivalent wrt core logic (same data, same model, same loss/optimizer/steps), with only negligible floating-point differences possible from parallelism scheduling.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

print("TF:", tf.__version__)

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

tf.config.optimizer.set_jit(False)  # keep as-is (avoid compile overhead surprises)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
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


def build_decoder(with_labels=True, target_size=(380, 380), ext="jpg"):
    ext = ext.lower()

    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == "png":
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ("jpg", "jpeg"):
            img = tf.io.decode_jpeg(file_bytes, channels=3, dct_method="INTEGER_FAST")
        else:
            raise ValueError("Image extension not supported")

        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, target_size, method="bilinear", antialias=False)
        img.set_shape([target_size[0], target_size[1], 3])
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


def _safe_setattr(obj, name, value):
    try:
        getattr(obj, name)
    except Exception:
        return
    try:
        setattr(obj, name, value)
    except Exception:
        pass


def build_dataset(
    paths,
    labels=None,
    bsize=64,
    cache=False,
    decode_fn=None,
    augment_fn=None,
    augment=True,
    repeat=True,
    shuffle=1024,
    cache_dir="",
    deterministic=True,
    interleave_files=False,  # kept for API compatibility; not used
    cycle_length=None,  # kept for API compatibility; not used
    drop_remainder=False,
):
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    AUTO = tf.data.AUTOTUNE

    options = tf.data.Options()
    _safe_setattr(
        options.experimental_optimization, "apply_default_optimizations", True
    )
    _safe_setattr(options.experimental_optimization, "map_parallelization", True)
    _safe_setattr(options.experimental_optimization, "map_and_batch_fusion", True)
    _safe_setattr(options.experimental_optimization, "parallel_batch", True)

    _safe_setattr(options.experimental_optimization, "autotune_buffers", True)
    _safe_setattr(options.experimental_optimization, "autotune_cpu_budget", True)
    _safe_setattr(options.experimental_optimization, "autotune_ram_budget", True)

    _safe_setattr(options, "experimental_slack", True)
    _safe_setattr(options, "deterministic", deterministic)
    _safe_setattr(options, "experimental_deterministic", deterministic)

    _safe_setattr(
        options, "experimental_distribute", tf.data.experimental.DistributeOptions()
    )
    try:
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
    except Exception:
        pass

    if labels is None:
        dset = tf.data.Dataset.from_tensor_slices(paths)
    else:
        dset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dset = dset.shuffle(shuffle, reshuffle_each_iteration=True)

    dset = dset.with_options(options)

    dset = dset.map(decode_fn, num_parallel_calls=AUTO, deterministic=deterministic)
    dset = dset.apply(tf.data.experimental.ignore_errors())

    if cache:
        if isinstance(cache_dir, str) and cache_dir:
            os.makedirs(os.path.dirname(cache_dir) or ".", exist_ok=True)
            dset = dset.cache(cache_dir)
        else:
            dset = dset.cache()

    if augment:
        dset = dset.map(
            augment_fn, num_parallel_calls=AUTO, deterministic=deterministic
        )

    if repeat:
        dset = dset.repeat()

    dset = dset.batch(bsize, drop_remainder=drop_remainder)
    dset = dset.prefetch(AUTO)
    return dset


COMPETITION_NAME = "ranzcr-clip-catheter-line-classification"
load_dir = f"/kaggle/input/{COMPETITION_NAME}/"

strategy = auto_select_accelerator()
BATCH_SIZE = strategy.num_replicas_in_sync * 16

IMSIZE = (224, 224, 260, 300, 380, 456, 528, 600)
TARGET_SIZE = (IMSIZE[4], IMSIZE[4])

sub_df = pd.read_csv(load_dir + "sample_submission.csv")
test_paths = (
    load_dir + "test/" + sub_df["StudyInstanceUID"].astype(str) + ".jpg"
).values
label_cols = list(sub_df.columns[1:])

test_decoder = build_decoder(with_labels=False, target_size=TARGET_SIZE)

dtest = build_dataset(
    test_paths,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=False,
    cache_dir="",
    decode_fn=test_decoder,
    deterministic=True,
    interleave_files=False,
    drop_remainder=False,
)

print("Submission columns:", label_cols)
print("Num test images:", len(test_paths))




## === cell 2
train_df = pd.read_csv(load_dir + "train.csv")

required_targets = [
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
train_targets = [c for c in required_targets if c in train_df.columns]

train_paths = (
    load_dir + "train/" + train_df["StudyInstanceUID"].astype(str) + ".jpg"
).values
y = train_df[train_targets].astype("float32").values

rng = np.random.RandomState(42)
patients = train_df["PatientID"].astype(str).values
unique_patients = np.unique(patients)
rng.shuffle(unique_patients)
split = int(0.9 * len(unique_patients))
train_p = unique_patients[:split]
is_train = np.isin(patients, train_p)

tr_paths = train_paths[is_train]
va_paths = train_paths[~is_train]
tr_y = y[is_train]
va_y = y[~is_train]

print("Train/Val sizes:", tr_paths.shape[0], va_paths.shape[0])
print("Num targets in training:", len(train_targets), train_targets)

train_decoder = build_decoder(with_labels=True, target_size=TARGET_SIZE)

steps_per_epoch = max(1, int(np.ceil(len(tr_paths) / BATCH_SIZE)))
val_steps = max(1, int(np.ceil(len(va_paths) / BATCH_SIZE)))

dtr = build_dataset(
    tr_paths,
    tr_y,
    bsize=BATCH_SIZE,
    repeat=True,
    shuffle=2048,
    augment=False,
    cache=False,
    decode_fn=train_decoder,
    deterministic=False,
    interleave_files=False,
    drop_remainder=True,
)

dva = build_dataset(
    va_paths,
    va_y,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=False,
    cache_dir="",
    decode_fn=train_decoder,
    deterministic=True,
    interleave_files=False,
    drop_remainder=False,
)

with strategy.scope():
    backbone = tf.keras.applications.DenseNet169(
        include_top=False,
        weights="imagenet",
        input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
        pooling=None,
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    x = inputs
    x = tf.keras.applications.densenet.preprocess_input(x * 255.0)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(len(train_targets), activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        run_eagerly=False,
    )

history = model.fit(
    dtr,
    validation_data=dva,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)




## === cell 3
pred = model.predict(dtest, verbose=1)
pred = np.asarray(pred, dtype=np.float32)

pred_df = pd.DataFrame(pred, columns=train_targets)
pred_df.insert(0, "StudyInstanceUID", sub_df["StudyInstanceUID"].values)

out = sub_df[["StudyInstanceUID"]].copy()
for c in label_cols:
    if c in pred_df.columns:
        out[c] = pred_df[c].values
    else:
        out[c] = 0.0

out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print(out.head())
