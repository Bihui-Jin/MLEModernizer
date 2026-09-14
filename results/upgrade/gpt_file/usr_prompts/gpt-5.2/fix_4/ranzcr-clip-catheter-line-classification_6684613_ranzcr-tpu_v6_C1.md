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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## === cell 1
def auto_select_accelerator():
    """
    Reference:
        * https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu
        * https://www.kaggle.com/xhlulu/ranzcr-efficientnet-tpu-training
    """
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


def build_decoder(with_labels=True, target_size=(256, 256), ext="jpg"):
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
    cache=True,
    decode_fn=None,
    augment_fn=None,
    augment=True,
    repeat=True,
    shuffle=1024,
    cache_dir="",
):
    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass

    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)

    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    AUTO = tf.data.AUTOTUNE
    slices = paths if labels is None else (paths, labels)

    dset = tf.data.Dataset.from_tensor_slices(slices)
    dset = dset.with_options(options)

    dset = dset.map(decode_fn, num_parallel_calls=AUTO, deterministic=True)
    dset = dset.cache(cache_dir) if cache else dset
    dset = (
        dset.map(augment_fn, num_parallel_calls=AUTO, deterministic=True)
        if augment
        else dset
    )
    dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle) if shuffle else dset
    dset = dset.batch(bsize, drop_remainder=False).prefetch(AUTO)

    return dset




## === cell 2
COMPETITION_NAME = "ranzcr-clip-catheter-line-classification"
strategy = auto_select_accelerator()
BATCH_SIZE = strategy.num_replicas_in_sync * 16




## === cell 3
load_dir = f"/kaggle/input/{COMPETITION_NAME}/"

sub_df = pd.read_csv(os.path.join(load_dir, "sample_submission.csv"))

REQUIRED_LABELS = [
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

for c in REQUIRED_LABELS:
    if c not in sub_df.columns:
        sub_df[c] = 0.0

label_cols = REQUIRED_LABELS

test_paths = (
    os.path.join(load_dir, "test")
    + "/"
    + sub_df["StudyInstanceUID"].astype(str)
    + ".jpg"
)

IMSIZE = (224, 240, 260, 300, 380, 456, 528, 600)
IMG_SIZE = IMSIZE[7]

test_decoder = build_decoder(with_labels=False, target_size=(IMG_SIZE, IMG_SIZE))
dtest = build_dataset(
    test_paths.values,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=False,
    decode_fn=test_decoder,
)




## === cell 4
def find_any_keras_model_path():
    """
    Try to locate a saved model file/folder in typical Kaggle locations.
    Preference order:
      1) User's originally referenced path if exists
      2) A small set of likely /kaggle/input immediate subfolders (fast)
      3) Fallback recursive search (bounded to avoid massive filesystem scans)
    """
    orig = "/kaggle/input/modeltpu2/bestmodel_tpu (2).h5"
    if os.path.exists(orig):
        return orig

    fast_candidates = []
    try:
        for ds in os.listdir("/kaggle/input"):
            base = os.path.join("/kaggle/input", ds)
            if not os.path.isdir(base):
                continue
            fast_candidates.extend(glob.glob(os.path.join(base, "*.keras")))
            fast_candidates.extend(glob.glob(os.path.join(base, "*.h5")))
            fast_candidates.extend(glob.glob(os.path.join(base, "*.hdf5")))
            for marker in ("saved_model.pb", "keras_metadata.pb"):
                marker_paths = glob.glob(
                    os.path.join(base, "**", marker), recursive=True
                )
                for mp in marker_paths[:5]:
                    fast_candidates.append(os.path.dirname(mp))
            if fast_candidates:
                break
    except Exception:
        fast_candidates = []

    seen = set()
    fast_uniq = []
    for p in fast_candidates:
        if p not in seen and os.path.exists(p):
            fast_uniq.append(p)
            seen.add(p)
    if fast_uniq:
        return fast_uniq[0]

    patterns = [
        "/kaggle/input/**/*.keras",
        "/kaggle/input/**/*.h5",
        "/kaggle/input/**/*.hdf5",
        "/kaggle/input/**/saved_model.pb",
        "/kaggle/input/**/keras_metadata.pb",
    ]
    candidates = []
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        for p in matches[:2000]:
            if os.path.basename(p) in ("saved_model.pb", "keras_metadata.pb"):
                candidates.append(os.path.dirname(p))
            else:
                candidates.append(p)

    seen = set()
    uniq = []
    for p in candidates:
        if p not in seen:
            uniq.append(p)
            seen.add(p)

    return uniq[0] if uniq else None


def compute_label_priors(train_csv_path, label_columns):
    train_df = pd.read_csv(train_csv_path)
    priors = []
    for c in label_columns:
        if c in train_df.columns:
            priors.append(float(train_df[c].mean()))
        else:
            priors.append(0.0)
    return np.array(priors, dtype=np.float32)


def build_simple_cnn(img_size, n_classes):
    inp = tf.keras.Input(shape=(img_size, img_size, 3))
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(n_classes, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
    )
    return model


model_path = find_any_keras_model_path()
print("Resolved model path:", model_path)

model_p = None
if model_path is not None:
    with strategy.scope():
        try:
            model_p = tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            print("Model load failed, will try to train a small CNN. Error:", repr(e))
            model_p = None




## === cell 5
if model_p is None:
    train_csv_path = os.path.join(load_dir, "train.csv")
    train_df = pd.read_csv(train_csv_path)

    for c in label_cols:
        if c not in train_df.columns:
            train_df[c] = 0.0

    train_paths = (
        os.path.join(load_dir, "train")
        + "/"
        + train_df["StudyInstanceUID"].astype(str)
        + ".jpg"
    )
    y = train_df[label_cols].values.astype(np.float32)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    tr_idx, va_idx = next(
        splitter.split(train_df, y, groups=train_df["PatientID"].values)
    )

    x_tr, y_tr = train_paths.values[tr_idx], y[tr_idx]
    x_va, y_va = train_paths.values[va_idx], y[va_idx]

    train_decoder = build_decoder(with_labels=True, target_size=(IMG_SIZE, IMG_SIZE))
    valid_decoder = build_decoder(with_labels=True, target_size=(IMG_SIZE, IMG_SIZE))

    dtrain = build_dataset(
        x_tr,
        labels=y_tr,
        bsize=BATCH_SIZE,
        repeat=True,
        shuffle=2048,
        augment=True,
        cache=False,
        decode_fn=train_decoder,
    )
    dvalid = build_dataset(
        x_va,
        labels=y_va,
        bsize=BATCH_SIZE,
        repeat=False,
        shuffle=False,
        augment=False,
        cache=False,
        decode_fn=valid_decoder,
    )

    steps_per_epoch = max(1, int(np.ceil(len(x_tr) / BATCH_SIZE)))
    val_steps = max(1, int(np.ceil(len(x_va) / BATCH_SIZE)))

    with strategy.scope():
        model_p = build_simple_cnn(IMG_SIZE, len(label_cols))

    model_p.fit(
        dtrain,
        epochs=2,
        steps_per_epoch=steps_per_epoch,
        validation_data=dvalid,
        validation_steps=val_steps,
        verbose=1,
    )




## === cell 6
if model_p is not None:
    try:
        model_p.run_eagerly = False
    except Exception:
        pass

    preds = model_p.predict(dtest, verbose=1)
    preds = np.asarray(preds)

    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    if preds.shape[1] != len(label_cols):
        train_csv_path = os.path.join(load_dir, "train.csv")
        priors = compute_label_priors(train_csv_path, label_cols)

        if preds.shape[1] < len(label_cols):
            pad = np.tile(priors[preds.shape[1] :], (preds.shape[0], 1))
            preds = np.concatenate([preds, pad], axis=1)
        else:
            preds = preds[:, : len(label_cols)]
else:
    train_csv_path = os.path.join(load_dir, "train.csv")
    priors = compute_label_priors(train_csv_path, label_cols)
    preds = np.tile(priors, (len(sub_df), 1))

preds = np.clip(preds, 0.0, 1.0)

sub_df[label_cols] = preds.astype(np.float32)

sub_out = sub_df[["StudyInstanceUID"] + label_cols].copy()
sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print("Submission columns:", list(sub_out.columns))

sub_out.head()
