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

# 5. Target score

0.7600051610686219

# 6. Current score

0.47214

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.52395) has done: 'The timeout is dominated by CPU-side JPEG decode/resize overhead and an input pipeline that can’t keep the model fed fast enough, plus extra `ceil()` steps that re-traverse repeated samples each epoch. I keep the exact model and training loop semantics, but optimize `tf.data` by (1) moving resizing into a fused decode+resize op, (2) enabling parallel file reads, (3) setting pipeline options to reduce input stalls while remaining deterministic, and (4) using exact step counts for `drop_remainder=True` training to avoid unnecessary extra steps. These changes are provably equivalent in terms of data/labels seen (up to the existing shuffle/repeat behavior) and don’t alter architecture, loss, optimizer, or epochs.'
- What this solution (achieved 0.56576) has done: 'The crash happens before any training because TensorFlow import triggers a protobuf incompatibility (`MessageFactory.GetPrototype`), which is caused by the installed `protobuf==6.x` being too new for the TensorFlow build in this environment. I fix this by forcing TensorFlow to use the pure-Python protobuf implementation via environment variables set *before* importing TensorFlow (score-neutral, just unblocks execution). Next, I make the submission columns exactly match `train.csv` label columns (including `CVC - Normal` and `Swan Ganz Catheter Present`), because your `sample_submission.csv` in this environment is missing some targets and would otherwise lead to a structurally invalid submission for the competition. Finally, I keep your model/training loop intact and only add a small safety check to ensure predictions align with the expected number of labels and that `submission.csv` is always written.'
- What this solution (achieved 0.53061) has done: 'You’re still crashing on TensorFlow import because `protobuf==6.x` is incompatible with TF 2.18 in this environment, and setting the pure-Python protobuf env vars is not sufficient here. I fix this by additionally forcing protobuf to use the pure-Python backend at runtime via `google.protobuf.internal.api_implementation._set_implementation_type("python")` before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error. I also make the test ID source robust by reading test filenames from the `/test` directory (since your provided `sample_submission.csv` is missing two required label columns in this environment), while still using `train.csv` to define the exact 11 target columns required for a valid submission. Finally, I keep your model and training loop intact, but unfreeze EfficientNet for fine-tuning (same architecture, same loss/optimizer) to improve AUC toward the target score.'
- What this solution (achieved 0.47214) has done: 'We fix the TensorFlow import crash by forcing protobuf to the pure-Python implementation *before anything else imports protobuf/tensorflow*, and by removing the brittle private `_set_implementation_type()` call that no longer works with protobuf 6.x. Then we keep your model/training logic identical, but ensure the submission has exactly the 11 label columns required by `train.csv` (your environment’s `sample_submission.csv` is missing some targets) and that `test_ids` are aligned to `test_paths` deterministically. These changes are score-neutral for the model itself, but unblock training/inference and guarantee a valid `submission.csv` output.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import numpy as np
import pandas as pd

import tensorflow as tf

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass

print("TF version:", tf.__version__)




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


def build_decoder(with_labels=True, target_size=(224, 224), ext="jpg"):
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == "png":
            img = tf.image.decode_png(file_bytes, channels=3)
            img = tf.image.resize(
                img, target_size, method=tf.image.ResizeMethod.BILINEAR
            )
        elif ext in ["jpg", "jpeg"]:
            img = tf.io.decode_jpeg(file_bytes, channels=3, dct_method="INTEGER_FAST")
            img = tf.image.resize(
                img, target_size, method=tf.image.ResizeMethod.BILINEAR
            )
        else:
            raise ValueError("Image extension not supported")

        img = tf.cast(img, tf.float32) * (1.0 / 255.0)
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
    drop_remainder=False,
    deterministic=True,
):
    use_disk_cache = (cache_dir != "") and (cache is True)
    if use_disk_cache:
        os.makedirs(cache_dir, exist_ok=True)

    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    AUTO = tf.data.AUTOTUNE
    slices = paths if labels is None else (paths, labels)

    dset = tf.data.Dataset.from_tensor_slices(slices)

    options = tf.data.Options()
    options.deterministic = deterministic
    options.experimental_slack = True
    try:
        options.autotune.enabled = True
    except Exception:
        pass
    dset = dset.with_options(options)

    dset = dset.map(decode_fn, num_parallel_calls=AUTO, deterministic=deterministic)

    if cache:
        dset = dset.cache(cache_dir) if use_disk_cache else dset.cache()

    dset = (
        dset.map(augment_fn, num_parallel_calls=AUTO, deterministic=deterministic)
        if augment
        else dset
    )
    dset = dset.repeat() if repeat else dset
    dset = (
        dset.shuffle(shuffle, seed=SEED, reshuffle_each_iteration=True)
        if shuffle
        else dset
    )

    dset = dset.batch(bsize, drop_remainder=drop_remainder).prefetch(AUTO)
    return dset




## === cell 2
COMPETITION_NAME = "ranzcr-clip-catheter-line-classification"
strategy = auto_select_accelerator()
BATCH_SIZE = strategy.num_replicas_in_sync * 16

IMSIZE = (224, 224, 260, 300, 380, 456, 528, 600)

load_dir = f"/kaggle/input/{COMPETITION_NAME}/"

train_df = pd.read_csv(load_dir + "train.csv")

label_cols = [c for c in train_df.columns if c not in ["StudyInstanceUID", "PatientID"]]

test_files = tf.io.gfile.glob(load_dir + "test/*.jpg")
test_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_files]
test_ids = np.array(sorted(test_ids), dtype=object)
test_paths = np.array(
    [load_dir + "test/" + sid + ".jpg" for sid in test_ids], dtype=object
)

test_decoder = build_decoder(with_labels=False, target_size=(IMSIZE[0], IMSIZE[0]))
dtest = build_dataset(
    test_paths,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=False,
    decode_fn=test_decoder,
    drop_remainder=False,
    deterministic=True,
)

train_paths = (load_dir + "train/") + train_df["StudyInstanceUID"].astype(str) + ".jpg"
y = train_df[label_cols].astype("float32").values

n = len(train_df)
idx = np.arange(n)
rng = np.random.default_rng(SEED)
rng.shuffle(idx)

split = int(n * 0.9)
tr_idx, va_idx = idx[:split], idx[split:]

train_decoder = build_decoder(with_labels=True, target_size=(IMSIZE[0], IMSIZE[0]))

dtrain = build_dataset(
    train_paths.iloc[tr_idx].values,
    labels=y[tr_idx],
    bsize=BATCH_SIZE,
    repeat=True,
    shuffle=2048,
    augment=False,
    cache=False,
    cache_dir="",
    decode_fn=train_decoder,
    drop_remainder=True,
    deterministic=True,
)

dvalid = build_dataset(
    train_paths.iloc[va_idx].values,
    labels=y[va_idx],
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=True,
    cache_dir="",
    decode_fn=train_decoder,
    drop_remainder=False,
    deterministic=True,
)

steps_per_epoch = len(tr_idx) // BATCH_SIZE
valid_steps = int(np.ceil(len(va_idx) / BATCH_SIZE))

print("Num labels:", len(label_cols))
print("Train/valid:", len(tr_idx), len(va_idx))
print("BATCH_SIZE:", BATCH_SIZE, "steps_per_epoch:", steps_per_epoch)
print("Num test:", len(test_paths))
print("Label cols:", label_cols)



## === cell 3
with strategy.scope():
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMSIZE[0], IMSIZE[0], 3),
        pooling="avg",
    )

    base.trainable = True

    x_in = tf.keras.Input(shape=(IMSIZE[0], IMSIZE[0], 3))
    x = base(x_in, training=True)
    x = tf.keras.layers.Dropout(0.2)(x)
    x_out = tf.keras.layers.Dense(len(label_cols), activation="sigmoid")(x)
    model = tf.keras.Model(x_in, x_out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        jit_compile=True,
    )

EPOCHS = 2
history = model.fit(
    dtrain,
    validation_data=dvalid,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=valid_steps,
    verbose=1,
)



## === cell 4
pred = model.predict(dtest, verbose=1)

if pred.ndim != 2 or pred.shape[1] != len(label_cols):
    raise ValueError(
        f"Unexpected pred shape {pred.shape}; expected (n_test, {len(label_cols)})"
    )

sub_df = pd.DataFrame({"StudyInstanceUID": test_ids})
for j, c in enumerate(label_cols):
    sub_df[c] = pred[:, j].astype(np.float32)

sub_df = sub_df[["StudyInstanceUID"] + label_cols]
sub_df.to_csv("submission.csv", index=False)

print(sub_df.shape)
print(sub_df.columns.tolist())
print(sub_df.head())
print("Wrote submission.csv")
