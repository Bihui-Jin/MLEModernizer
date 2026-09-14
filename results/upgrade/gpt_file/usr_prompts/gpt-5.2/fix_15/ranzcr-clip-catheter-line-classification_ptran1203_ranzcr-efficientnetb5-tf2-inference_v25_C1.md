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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
import os, sys, subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit

SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("Warning: could not enable XLA JIT:", e)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

W = H = 338
BATCH_SIZE = 16
autotune = tf.data.AUTOTUNE

DATA_DIR = "../input/ranzcr-clip-catheter-line-classification"
train_csv_path = os.path.join(DATA_DIR, "train.csv")
test_dir = os.path.join(DATA_DIR, "test")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

target_cols_full = [
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

sample_sub = pd.read_csv(sample_sub_path)
sub_cols = [c for c in sample_sub.columns if c != "StudyInstanceUID"]

train_df = pd.read_csv(train_csv_path)
train_label_cols = [c for c in target_cols_full if c in train_df.columns]

if len(train_label_cols) == 0:
    raise ValueError("No target label columns found in train.csv; cannot train.")

N_CLASSES = len(train_label_cols)

print("Submission columns (from sample_submission.csv):", sub_cols)
print("Training label columns (from train.csv):", train_label_cols)
print("N_CLASSES for model output:", N_CLASSES)



## === cell 1
gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
groups = train_df["PatientID"].values
idx_train, idx_val = next(gss.split(train_df, groups=groups))

df_tr = train_df.iloc[idx_train].reset_index(drop=True)
df_va = train_df.iloc[idx_val].reset_index(drop=True)

train_dir = os.path.join(DATA_DIR, "train")


def _path_from_uid(uid):
    return os.path.join(train_dir, f"{uid}.jpg")


df_tr["path"] = df_tr["StudyInstanceUID"].map(_path_from_uid)
df_va["path"] = df_va["StudyInstanceUID"].map(_path_from_uid)

try:
    _train_jpg_names = set(tf.io.gfile.listdir(train_dir))
except Exception:
    _train_jpg_names = set(os.listdir(train_dir))

df_tr = df_tr[
    df_tr["StudyInstanceUID"].map(lambda uid: f"{uid}.jpg" in _train_jpg_names)
].reset_index(drop=True)
df_va = df_va[
    df_va["StudyInstanceUID"].map(lambda uid: f"{uid}.jpg" in _train_jpg_names)
].reset_index(drop=True)

print("Train/Val sizes (after existence filtering):", len(df_tr), len(df_va))



## === cell 2
preprocess_input = tf.keras.applications.efficientnet.preprocess_input


def _load_jpg_with_label(path, y):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, (H, W))  # keep identical resize semantics
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # critical for EfficientNet
    y = tf.cast(y, tf.float32)
    return img, y


_ds_options = tf.data.Options()
_ds_options.experimental_deterministic = True
_ds_options.experimental_optimization.apply_default_optimizations = True
try:
    _ds_options.experimental_optimization.map_parallelization = True
except Exception:
    pass
try:
    _ds_options.experimental_optimization.map_and_batch_fusion = True
except Exception:
    pass
try:
    _ds_options.experimental_optimization.parallel_batch = True
except Exception:
    pass


def _make_ds(df, training, cache_in_memory=True):
    paths = df["path"].values.astype(str)
    y = df[train_label_cols].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.with_options(_ds_options)

    if training:
        ds = ds.shuffle(min(len(df), 4096), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_load_jpg_with_label, num_parallel_calls=autotune)

    if cache_in_memory:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.prefetch(autotune)
    return ds


ds_tr = _make_ds(df_tr, training=True, cache_in_memory=True)
ds_va = _make_ds(df_va, training=False, cache_in_memory=True)




## === cell 3
def get_model(
    base_model,
    baseline_weight=None,
    init_weight=None,
    lr=0.001,
    optimizer=tf.optimizers.Adam,
):
    base_model = base_model(
        include_top=False, input_shape=(W, H, 3), pooling="avg", weights=baseline_weight
    )
    base_out = base_model.output
    out = tf.keras.layers.Dropout(0.3)(base_out)
    out = tf.keras.layers.Dense(N_CLASSES, activation="sigmoid")(out)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=out)

    model.compile(
        optimizer=optimizer(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[],
        jit_compile=True,
    )
    if init_weight:
        try:
            model.load_weights(init_weight)
            print(f"Weight loaded from {init_weight}")
        except Exception as e:
            print(f"Load weight from {init_weight} failed, {e}")
    return model


base_model_fn = tf.keras.applications.EfficientNetB5
model = get_model(base_model_fn, baseline_weight="imagenet", init_weight=None)



## === cell 4
EPOCHS = 2


class ValAUC(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, num_labels):
        super().__init__()
        self.val_ds = val_ds
        self.num_labels = num_labels
        self._auc = tf.keras.metrics.AUC(multi_label=True, num_labels=self.num_labels)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        y_true_parts = []
        for _, yb in self.val_ds:
            y_true_parts.append(tf.cast(yb, tf.float32))
        y_true = tf.concat(y_true_parts, axis=0)

        y_pred = self.model.predict(
            self.val_ds.map(lambda x, y: x, num_parallel_calls=autotune), verbose=0
        )
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        self._auc.reset_state()
        self._auc.update_state(y_true, y_pred)
        val_auc = float(self._auc.result().numpy())

        logs["val_auc"] = val_auc
        print(f"\nval_auc: {val_auc:.6f}")


history = model.fit(
    ds_tr,
    validation_data=None,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[ValAUC(ds_va, N_CLASSES)],
)



## === cell 5
test_files = tf.io.gfile.glob(os.path.join(test_dir, "*.jpg"))
test_files = sorted(test_files)


def _load_test_jpg(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, (H, W))
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # critical for EfficientNet
    uid = tf.strings.regex_replace(tf.strings.split(path, os.sep)[-1], r"\.jpg$", "")
    return img, uid


test_data = tf.data.Dataset.from_tensor_slices(test_files).with_options(_ds_options)
test_data = test_data.map(_load_test_jpg, num_parallel_calls=autotune)

test_data = test_data.cache()

test_data = test_data.batch(BATCH_SIZE, drop_remainder=False)
test_data = test_data.apply(tf.data.experimental.ignore_errors())
test_data = test_data.prefetch(autotune)



## === cell 6
uids = [os.path.splitext(os.path.basename(p))[0] for p in test_files]

test_images = test_data.map(lambda x, uid: x, num_parallel_calls=autotune)

preds = model.predict(test_images, verbose=0)
preds = np.clip(preds, 0.0, 1.0)

pred_map = {c: preds[:, i] for i, c in enumerate(train_label_cols)}

test_df = pd.DataFrame({"StudyInstanceUID": uids})
for c in sub_cols:
    if c in pred_map:
        test_df[c] = pred_map[c]
    else:
        test_df[c] = 0.5

test_df = (
    test_df.set_index("StudyInstanceUID")
    .reindex(sample_sub["StudyInstanceUID"])
    .reset_index()
)

test_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", test_df.shape)
print(test_df.head())



## === cell 7
assert os.path.exists("submission.csv")
out = pd.read_csv("submission.csv")
assert list(out.columns) == list(sample_sub.columns), (
    out.columns.tolist(),
    sample_sub.columns.tolist(),
)
assert len(out) == len(sample_sub)
out.describe(include="all")
