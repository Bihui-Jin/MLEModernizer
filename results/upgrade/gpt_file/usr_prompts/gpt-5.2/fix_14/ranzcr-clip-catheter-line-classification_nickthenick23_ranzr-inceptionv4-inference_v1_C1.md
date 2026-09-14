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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.9075610321771016

# 6. Current score

0.65689

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.62898) has done: 'The timeout is dominated by repeatedly decoding and resizing 512×512 JPGs on the CPU while the model trains, plus extra overhead from `ignore_errors()` and deterministic `tf.data` settings that limit pipeline parallelism. I keep the exact same model, loss, optimizer, steps, and epochs, but make the input pipeline much faster by (1) switching to fused JPEG decode+resize ops, (2) caching the training decode stage to a local disk cache (so epoch 2 reuses already-decoded tensors), and (3) removing `ignore_errors()` (it adds overhead and is unnecessary if files exist). These changes preserve semantics (same images/labels, same augmentation, same batches/steps) and typically bring runtime well under 600s.'
- What this solution (achieved 0.62898) has done: 'We need to fix the immediate runtime crash happening before any training: the `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` is a known protobuf/TensorFlow incompatibility that can be resolved by forcing the Python protobuf implementation before importing TensorFlow. I apply that environment fix at the very top (before `import tensorflow as tf`) and keep the rest of the pipeline/model/training loop unchanged. I also make the TPU strategy selection compatible with TF 2.18 by using the non-experimental `TPUStrategy` API when available (fallback unchanged). Finally, I keep the submission column handling as-is (it correctly adds missing target columns to match `train.csv`) to ensure a valid `submission.csv` is always produced.'
- What this solution (achieved 0.62898) has done: 'The immediate crash happens before training because TensorFlow 2.18 with protobuf 6.x is incompatible with forcing the pure-Python protobuf implementation; removing those environment overrides (or forcing the C++ implementation) fixes the `MessageFactory.GetPrototype` error. I make that change at the very top, before importing TensorFlow, while keeping your model, loss, optimizer, steps/epochs, and data pipeline logic the same. I also ensure the submission columns exactly match `train.csv` targets by continuing to add any missing columns to `sample_submission.csv` (your approach is correct) and write `submission.csv` in the working directory.'
- What this solution (achieved 0.62898) has done: 'You’re crashing before training because TensorFlow 2.18 + protobuf 6.x can trigger `MessageFactory.GetPrototype` failures depending on the protobuf runtime; the most reliable fix in Kaggle is to force the pure-Python protobuf implementation *before* importing TensorFlow. I add that environment setting at the very top and keep the rest of your pipeline/model/training loop intact. I also make the submission column handling robust to the known issue that some provided `sample_submission.csv` copies are missing target columns, ensuring the output always matches the 11 label columns from `train.csv` and writes a valid `submission.csv`. These changes are execution-stability fixes and should also help score by allowing the model to actually train and generate non-zero predictions.'
- What this solution (achieved 0.62898) has done: 'I fix the immediate TensorFlow/protobuf crash by removing the forced pure-Python protobuf environment override that is incompatible with TF 2.18 + protobuf 6.x in this Kaggle image. I also make the submission-column logic robust to the known issue that some `sample_submission.csv` copies are missing a few target columns by always aligning to the full label set from `train.csv` (score-neutral but prevents invalid submissions). Finally, I keep your model/training loop unchanged, but I add a lightweight sanity check that all referenced image files exist (to avoid silent empty/NaN batches) while preserving the same data and semantics.'
- What this solution (achieved 0.65689) has done: 'The immediate blocker is the TensorFlow import crash caused by forcing the protobuf C++ implementation (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="cpp"`) in an environment where `google.protobuf.pyext._message` is not available; removing that override (and not forcing any implementation) fixes the runtime so the rest of the notebook can execute. Once TF imports, the downstream `NameError`s disappear because `load_dir/IMAGE_SIZE/...` be defined by the earlier cell again. I keep your model, loss, optimizer, epochs/steps, and dataset semantics the same, only making the minimal environment fix plus a small guard to always align the submission columns to the full label set from `train.csv` (some sample_submissions are missing columns), ensuring a valid `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

IMAGE_SIZE = 512


def auto_select_accelerator():
    """
    TPU if available; otherwise default strategy. Keep semantics close to original.
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        try:
            strategy = tf.distribute.TPUStrategy(tpu)
        except Exception:
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except Exception:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy


def build_decoder(with_labels=True, target_size=(IMAGE_SIZE, IMAGE_SIZE), ext="jpg"):
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == "png":
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ["jpg", "jpeg"]:
            img = tf.io.decode_jpeg(file_bytes, channels=3, dct_method="INTEGER_FAST")
        else:
            raise ValueError("Image extension not supported")

        img = tf.image.resize(
            img, target_size, method=tf.image.ResizeMethod.BILINEAR, antialias=True
        )
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img

    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment


def build_dataset(
    paths,
    labels=None,
    bsize=32,
    cache=False,
    decode_fn=None,
    augment_fn=None,
    augment=True,
    repeat=True,
    shuffle=1024,
    cache_dir="",
    deterministic=True,
    prefetch_to_device=None,
    drop_remainder=False,
):
    if cache_dir != "" and cache is True:
        os.makedirs(os.path.dirname(cache_dir), exist_ok=True)

    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    AUTO = tf.data.AUTOTUNE
    slices = paths if labels is None else (paths, labels)

    options = tf.data.Options()
    options.experimental_deterministic = bool(deterministic)
    options.autotune.enabled = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True

    dset = tf.data.Dataset.from_tensor_slices(slices)
    dset = dset.with_options(options)

    if shuffle:
        dset = dset.shuffle(shuffle, reshuffle_each_iteration=True)

    dset = dset.map(decode_fn, num_parallel_calls=AUTO)

    if cache:
        if cache_dir:
            dset = dset.cache(cache_dir)
        else:
            dset = dset.cache()

    if augment:
        dset = dset.map(augment_fn, num_parallel_calls=AUTO)

    if repeat:
        dset = dset.repeat()

    dset = dset.batch(bsize, drop_remainder=bool(drop_remainder))

    if prefetch_to_device is not None:
        dset = dset.apply(
            tf.data.experimental.prefetch_to_device(
                prefetch_to_device, buffer_size=AUTO
            )
        )
    dset = dset.prefetch(AUTO)
    return dset


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass


seed_everything(42)

COMPETITION_NAME = "ranzcr-clip-catheter-line-classification"
strategy = auto_select_accelerator()
BATCH_SIZE = strategy.num_replicas_in_sync * 16


def find_competition_root(name: str) -> str:
    candidates = [
        f"/kaggle/input/{name}/",
        f"/kaggle/data/{name}/",
        f"/kaggle/input/{name}/{name}/",
        f"/kaggle/data/{name}/{name}/",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.exists(
            os.path.join(c, "sample_submission.csv")
        ):
            return c
    raise FileNotFoundError(
        "Could not locate competition data. Tried: " + ", ".join(candidates)
    )


load_dir = find_competition_root(COMPETITION_NAME)
load_dir



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
sub_path = os.path.join(load_dir, "sample_submission.csv")
train_path = os.path.join(load_dir, "train.csv")

sub_df = pd.read_csv(sub_path)
train_df = pd.read_csv(train_path)

label_cols = [c for c in train_df.columns if c not in ["StudyInstanceUID", "PatientID"]]

required_cols = ["StudyInstanceUID"] + label_cols
for c in required_cols:
    if c not in sub_df.columns:
        sub_df[c] = 0.0
sub_df = sub_df[required_cols]

train_img_dir = os.path.join(load_dir, "train")
test_img_dir = os.path.join(load_dir, "test")

train_paths = (
    train_img_dir + "/" + train_df["StudyInstanceUID"].astype(str) + ".jpg"
).to_numpy()
test_paths = (
    test_img_dir + "/" + sub_df["StudyInstanceUID"].astype(str) + ".jpg"
).to_numpy()

for p in (train_paths[0], train_paths[len(train_paths) // 2], train_paths[-1]):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Train image not found: {p}")
for p in (test_paths[0], test_paths[len(test_paths) // 2], test_paths[-1]):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Test image not found: {p}")

y = train_df[label_cols].astype(np.float32).values

unique_patients = train_df["PatientID"].unique()
train_p, val_p = train_test_split(unique_patients, test_size=0.1, random_state=42)
train_idx = train_df["PatientID"].isin(train_p).values
val_idx = ~train_idx

tr_paths, va_paths = train_paths[train_idx], train_paths[val_idx]
tr_y, va_y = y[train_idx], y[val_idx]

train_decoder = build_decoder(with_labels=True, target_size=(IMAGE_SIZE, IMAGE_SIZE))
test_decoder = build_decoder(with_labels=False, target_size=(IMAGE_SIZE, IMAGE_SIZE))
augmenter = build_augmenter(with_labels=True)

device_prefetch = None

train_cache_path = "/kaggle/working/tf_cache/ranzcr_train_decode.cache"

dtrain = build_dataset(
    tr_paths,
    tr_y,
    bsize=BATCH_SIZE,
    repeat=True,
    shuffle=2048,
    augment=True,
    cache=True,
    cache_dir=train_cache_path,
    decode_fn=train_decoder,
    augment_fn=augmenter,
    deterministic=False,
    prefetch_to_device=device_prefetch,
    drop_remainder=True,
)
dvalid = build_dataset(
    va_paths,
    va_y,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=True,  # in-memory cache (fast, small split)
    cache_dir="",
    decode_fn=train_decoder,
    deterministic=True,
    prefetch_to_device=device_prefetch,
    drop_remainder=False,
)
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
    prefetch_to_device=device_prefetch,
    drop_remainder=False,
)

len(tr_paths), len(va_paths), len(test_paths), len(label_cols)




## === cell 2
def build_model(num_classes: int, image_size: int = IMAGE_SIZE):
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)


with strategy.scope():
    model = build_model(num_classes=len(label_cols), image_size=IMAGE_SIZE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.AUC(
                multi_label=True, num_labels=len(label_cols), name="auc"
            )
        ],
        jit_compile=True,
    )

steps_per_epoch = max(1, int(np.ceil(len(tr_paths) / BATCH_SIZE)))
val_steps = max(1, int(np.ceil(len(va_paths) / BATCH_SIZE)))

history = model.fit(
    dtrain,
    validation_data=dvalid,
    epochs=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)



## === cell 3
test_steps = max(1, int(np.ceil(len(test_paths) / BATCH_SIZE)))

preds = model.predict(dtest, steps=test_steps, verbose=1)
preds = np.clip(preds, 0.0, 1.0)

sub_out = sub_df.copy()
sub_out[label_cols] = preds.astype(np.float32)

out_path = "submission.csv"
sub_out.to_csv(out_path, index=False)

sub_out.head()
