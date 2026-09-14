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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

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

0.9003505419349426

# 6. Current score

0.78904

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.67381) has done: 'I fix the import-time crash by pinning protobuf to the pure-Python implementation before TensorFlow/Keras import, which resolves the `MessageFactory.GetPrototype` error seen in this environment. Since the referenced external trained model file does not exist, I replace the missing `load_model(...)` step with a minimal in-notebook training of the same kind of multi-label image classifier (sigmoid outputs) using the provided `train/` JPGs and `train.csv`, so `model` is always defined. I also make the submission generation robust to the sample submission’s missing columns by aligning predictions to exactly the sample submission columns (required by Kaggle), while still training on all available labels for better generalization. Finally, I keep the inference loop logic the same (read image → resize → scale → predict) but vectorize it with a `tf.data` pipeline for speed so it completes within the timeout and writes `submission.csv`.'
- What this solution (achieved 0.76027) has done: 'I fix the protobuf/TensorFlow import-time crash by forcing the pure-Python protobuf implementation *before* any TensorFlow-related import and by restarting the protobuf module state in-process when possible. I also make the data path resolution robust to either `/kaggle/input/...` or the provided `/kaggle/data/...` layout so the notebook runs end-to-end in this environment. Finally, I keep your exact model and training loop, but train a bit longer (same architecture/loss/optimizer) to move the AUC toward your target, and I ensure the submission columns exactly match `sample_submission.csv` so Kaggle accepts it.'
- What this solution (achieved 0.77624) has done: 'Main bottlenecks are (1) forcing pure-Python protobuf (very slow) and even running a `pip install` at runtime, and (2) an input pipeline that repeatedly resizes with slower ops and doesn’t cache decoded/resized images, making 10 epochs I/O-bound. I remove the protobuf override/pip-install (TensorFlow 2.18 works with the provided protobuf), enable deterministic + faster TF runtime settings, and make the `tf.data` pipeline more efficient by using `decode_jpeg(..., dct_method='INTEGER_FAST')`, `tf.image.resize(..., antialias=False)`, caching (memory) for train/val/test, and a couple of low-risk pipeline options (prefetch, deterministic). This preserves the same model, loss, epochs, split logic, and exact training loop semantics—only speeds up environment setup and data loading/decoding.'
- What this solution (achieved 0.7774) has done: 'I fix the import-time crash by forcing protobuf to use the pure-Python implementation *before* importing TensorFlow (this is the direct cause of the `MessageFactory.GetPrototype` error in this environment). I keep your exact dataset logic, model architecture, loss/optimizer, patient split, and inference/submission logic unchanged, only moving the environment variables and imports to the correct order so the notebook runs end-to-end. I also add a small safety fallback to ensure the sample submission columns are handled robustly (some copies of the sample file in your environment appear truncated), without changing prediction semantics. This should restore execution and keep (or improve) score by allowing the full 11-label model to train and then be aligned to the required submission columns.'
- What this solution (achieved 0.79511) has done: 'I fix the protobuf/TensorFlow import crash by removing the forced pure-Python protobuf environment variables (they are what triggers the `MessageFactory.GetPrototype` error here) and importing TensorFlow normally in this Kaggle environment. I also make the submission-column handling robust by always building the submission from `train.csv` label columns (11 targets) and only falling back if a truly truncated `sample_submission.csv` is encountered, so Kaggle gets the exact required header. Finally, to move the score upward toward your target without changing the core model/training approach, I train the same architecture for a few more epochs (no early stopping or other training-logic changes), which is the smallest expected performance nudge.'
- What this solution (achieved 0.78904) has done: 'I fix the import-time crash by pinning protobuf to the pure-Python implementation *before* importing TensorFlow, which avoids the `MessageFactory.GetPrototype` AttributeError in this environment. I also ensure all required imports are present in the data-loading cell (the current script reads CSVs before importing pandas due to the failed first cell). Finally, I keep your exact model, split, training loop, and inference logic unchanged so scoring behavior stays comparable while producing a valid `submission.csv` with the correct 11 target columns.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import os

CANDIDATE_DIRS = [
    "/kaggle/input/ranzcr-clip-catheter-line-classification",
    "/kaggle/data/ranzcr-clip-catheter-line-classification",
    "/kaggle/input/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification",
    "/kaggle/data/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification",
]

DATA_DIR = None
for d in CANDIDATE_DIRS:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
        os.path.join(d, "sample_submission.csv")
    ):
        DATA_DIR = d
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find dataset directory. Checked: " + ", ".join(CANDIDATE_DIRS)
    )

TRAIN_CSV = f"{DATA_DIR}/train.csv"
SAMPLE_SUB = f"{DATA_DIR}/sample_submission.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train"
TEST_IMG_DIR = f"{DATA_DIR}/test"

train_df = pd.read_csv(TRAIN_CSV)
sub = pd.read_csv(SAMPLE_SUB)

all_label_cols = [
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

expected_sub_label_cols = all_label_cols[:]
sub_label_cols = [c for c in sub.columns if c != "StudyInstanceUID"]
if set(sub_label_cols) != set(expected_sub_label_cols):
    print(
        "WARNING: sample_submission label columns do not match expected 11 targets.\n"
        f"sample has {len(sub_label_cols)}: {sub_label_cols}\n"
        f"expected {len(expected_sub_label_cols)}: {expected_sub_label_cols}\n"
        "Will write submission with expected 11 target columns."
    )
    sub_label_cols = expected_sub_label_cols

print("DATA_DIR:", DATA_DIR)
print("Train rows:", len(train_df), "Test rows:", len(sub))
print("Train label cols:", len(all_label_cols))
print("Submission label cols:", len(sub_label_cols), sub_label_cols)




## === cell 2
BS = 16
IMG_SIZE = 260
AUTOTUNE = tf.data.AUTOTUNE


def _read_decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def make_train_dataset(df, batch_size=BS, shuffle=True):
    paths = tf.strings.join(
        [TRAIN_IMG_DIR, "/", tf.constant(df["StudyInstanceUID"].values), ".jpg"]
    )
    y = df[all_label_cols].to_numpy(dtype="float32", copy=False)

    ds = tf.data.Dataset.from_tensor_slices((paths, y))

    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(df), 4096), seed=42, reshuffle_each_iteration=True
        )

    ds = ds.map(lambda p, yy: (_read_decode_resize(p), yy), num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_dataset(uids, batch_size=BS):
    paths = tf.strings.join([TEST_IMG_DIR, "/", tf.constant(uids), ".jpg"])

    ds = tf.data.Dataset.from_tensor_slices(paths)
    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    ds = ds.map(_read_decode_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds




## === cell 3
patients = train_df["PatientID"].values
unique_patients = pd.unique(patients)
rng = np.random.RandomState(42)
rng.shuffle(unique_patients)
val_frac = 0.1
n_val = int(len(unique_patients) * val_frac)
val_patients = set(unique_patients[:n_val])

is_val = train_df["PatientID"].isin(val_patients)
trn_df = train_df.loc[~is_val].reset_index(drop=True)
val_df = train_df.loc[is_val].reset_index(drop=True)

train_ds = make_train_dataset(trn_df, batch_size=BS, shuffle=True)
val_ds = make_train_dataset(val_df, batch_size=BS, shuffle=False)

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(all_label_cols), activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[],
)

model.summary()

EPOCHS = 14
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)




## === cell 4
test_uids = sub["StudyInstanceUID"].values
test_ds = make_test_dataset(test_uids, batch_size=BS)

pred = model.predict(test_ds, verbose=1)  # shape: (N, 11)
pred = np.asarray(pred, dtype=np.float32)

pred_df_all = pd.DataFrame(pred, columns=all_label_cols)
pred_df_all.insert(0, "StudyInstanceUID", test_uids)

submission = pd.DataFrame({"StudyInstanceUID": test_uids})
for c in sub_label_cols:
    if c in pred_df_all.columns:
        submission[c] = pred_df_all[c].values
    else:
        submission[c] = 0.5

submission = submission[["StudyInstanceUID"] + sub_label_cols]

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head())
print("Shape:", submission.shape)
