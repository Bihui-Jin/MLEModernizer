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
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

-1.0

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.60706) has done: 'I remove/guard the imports that trigger the protobuf `MessageFactory.GetPrototype` crash (coming from optional memory/protobuf-related deps) so the notebook can import cleanly. Then I replace the missing external pretrained-model loads with a minimal in-notebook fallback that preserves the “ensemble of multiple models over 6 T2 slices” core semantics by training the same small Keras CNN architecture multiple times with different seeds on the available train set (T2 only). I also fix the data-loading bugs (missing `resize`, list/array normalization, and the incorrect `p406` using `pixels_5`), and make prediction aggregation robust in case some subjects yield fewer valid slices. Finally, I ensure the submission has exactly the required columns and writes `submission.csv`.'
- What this solution (achieved 0.59176) has done: 'I fix the import crash by removing the standalone `keras` import and using only `tf.keras`, which avoids the protobuf `MessageFactory.GetPrototype` issue in this Kaggle environment. I also make the split deterministic at the *case* level (instead of slice level) to prevent leakage where slices from the same patient can appear in both train/valid, which should improve AUC without changing the model architecture or training loop semantics. Finally, I keep the submission writing logic intact but ensure all IDs are handled as 5-digit strings consistently and that `submission.csv` is always produced.'
- What this solution (achieved 0.5) has done: 'The timeout is dominated by extremely slow per-slice DICOM decoding done in Python loops with repeated TensorFlow graph breaks (`.numpy()` checks) and by forcing eager execution globally (`run_functions_eagerly(True)`), which prevents TensorFlow from optimizing the decode/resize pipeline. I keep the exact slice-selection rules, model, training loop, and evaluation semantics, but make the decoding pipeline run as compiled `tf.function` and avoid decoding each slice twice by returning both the stacked image and the quick “sum” check from a single decode. I also reduce filesystem overhead by using `os.scandir` efficiently and parallelize DICOM decoding across slices/cases using a small thread pool (same deterministic ordering, same filtering logic). These changes are provably equivalent in outputs (up to negligible float differences) while removing the main Python-level bottlenecks that cause the 10-minute timeout.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(42)

DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

bad_ids = set(["00109", "00123", "00709"])  # per competition note

print("TensorFlow:", tf.__version__)
print("DATA_ROOT exists:", os.path.isdir(DATA_ROOT))

try:
    tf.config.run_functions_eagerly(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
    tf.config.threading.set_inter_op_parallelism_threads(max(1, os.cpu_count() // 4))
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
@tf.function
def _decode_resize_norm_stack_and_stats(dcm_path, img_px_size: int):
    b = tf.io.read_file(dcm_path)
    img = tf.io.decode_dicom_image(
        b,
        dtype=tf.uint16,
        color_dim=False,
        scale="auto",
        expand_animations=True,
    )
    img = tf.squeeze(img)  # [H,W] or [H,W,1]
    img = tf.cond(tf.equal(tf.rank(img), 2), lambda: img[..., None], lambda: img)
    img = tf.cast(img, tf.float32)  # [H,W,1]

    img2d = tf.squeeze(img, axis=-1)  # [H,W]
    raw_sum = tf.reduce_sum(img2d)

    img = tf.image.resize(
        img, (img_px_size, img_px_size), method="bilinear", antialias=True
    )  # [S,S,1]
    img2d_rs = tf.squeeze(img, axis=-1)  # [S,S]

    mx = tf.reduce_max(img2d_rs)
    norm2d = tf.math.divide_no_nan(img2d_rs, mx)
    stacked = tf.stack([norm2d, norm2d, norm2d], axis=-1)  # [S,S,3]
    stacked_sum = tf.reduce_sum(stacked)
    return mx, raw_sum, stacked_sum, stacked


def load_case_slices_t2(case_dir: str, img_px_size: int = 150, n_slices: int = 6):
    """
    Loads up to n_slices T2w slices for one subject.
    Returns: list of (H,W,3) float32 in [0,1], length <= n_slices
    """
    t2_dir = os.path.join(case_dir, "T2w")
    if not os.path.isdir(t2_dir):
        return []

    entries = []
    with os.scandir(t2_dir) as it:
        for e in it:
            if e.is_file() and e.name.lower().endswith(".dcm"):
                entries.append(e.path)
    entries.sort()

    out = []
    count = 0
    for p in entries:
        if count >= n_slices:
            break
        try:
            mx, raw_sum, stacked_sum, stacked = _decode_resize_norm_stack_and_stats(
                tf.constant(p), img_px_size
            )

            if float(raw_sum.numpy()) <= 100000.0:
                continue
            if float(mx.numpy()) <= 0.0:
                continue
            if float(stacked_sum.numpy()) <= 2000.0:
                continue

            out.append(stacked.numpy().astype(np.float32))
            count += 1
        except Exception:
            continue

    return out




## === cell 2
from concurrent.futures import ThreadPoolExecutor


def load_dataset_t2(
    root_dir: str,
    labels_df: pd.DataFrame = None,
    img_px_size: int = 150,
    n_slices: int = 6,
):
    """
    Builds an image dataset by treating each selected slice as one example.
    Returns:
      X: (N, H, W, 3) float32
      y: (N,) float32 (if labels_df provided else None)
      meta: list of tuples (case_id_str, slice_index)
    """
    case_dirs = []
    with os.scandir(root_dir) as it:
        for e in it:
            if e.is_dir():
                case_dirs.append(e.path)
    case_dirs.sort()

    label_map = None
    if labels_df is not None:
        ids = labels_df["BraTS21ID"].to_numpy(dtype=np.int32)
        vals = labels_df["MGMT_value"].to_numpy(dtype=np.float32)
        label_map = {f"{int(i):05d}": float(v) for i, v in zip(ids, vals)}

    filtered = []
    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)
        if labels_df is not None:
            if case_id in bad_ids:
                continue
            if label_map is not None and case_id not in label_map:
                continue
        filtered.append(case_dir)

    def _process_one(case_dir_):
        case_id_ = os.path.basename(case_dir_)
        slices_ = load_case_slices_t2(
            case_dir_, img_px_size=img_px_size, n_slices=n_slices
        )
        if not slices_:
            return case_id_, [], []
        X_loc = []
        meta_loc = []
        for si, img in enumerate(slices_):
            if img is None or img.shape != (img_px_size, img_px_size, 3):
                continue
            X_loc.append(img)
            meta_loc.append((case_id_, si))
        return case_id_, X_loc, meta_loc

    X_list, y_list, meta = [], [], []

    max_workers = min(8, max(1, (os.cpu_count() or 4) // 2))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_process_one, filtered))

    for case_id, X_loc, meta_loc in results:
        if not X_loc:
            continue
        X_list.extend(X_loc)
        meta.extend(meta_loc)
        if label_map is not None:
            y_list.extend([label_map[case_id]] * len(X_loc))

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32) if labels_df is not None else None
    return X, y, meta




## === cell 3
train_labels = pd.read_csv(TRAIN_LABELS_CSV)
train_labels["BraTS21ID"] = train_labels["BraTS21ID"].astype(int)

IMG_PX_SIZE = 150
N_SLICES = 6

X_train, y_train, train_meta = load_dataset_t2(
    TRAIN_DIR, labels_df=train_labels, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)
print("Train slice dataset:", X_train.shape, y_train.shape)

X_test, _, test_meta = load_dataset_t2(
    TEST_DIR, labels_df=None, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)
print("Test slice dataset:", X_test.shape, len(test_meta))

assert X_train.shape[0] == y_train.shape[0], "X_train/y_train length mismatch"
assert X_test.shape[0] == len(test_meta), "X_test/test_meta length mismatch"




## === cell 4
def build_model(input_shape=(150, 150, 3)):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model




## === cell 5
EPOCHS = 3
BATCH_SIZE = 32

ensemble_models = []  # ensure always defined

if X_train.shape[0] == 0:
    print("Warning: X_train is empty (no decoded slices). Will skip training.")
else:
    case_ids_all = np.array([cid for (cid, _si) in train_meta], dtype=object)
    unique_case_ids = np.array(sorted(set(case_ids_all.tolist())), dtype=object)

    rng = np.random.RandomState(42)
    rng.shuffle(unique_case_ids)
    split_case = int(0.9 * len(unique_case_ids))
    train_case_ids = unique_case_ids[:split_case]
    valid_case_ids = unique_case_ids[split_case:]

    tr_mask = np.isin(case_ids_all, train_case_ids, assume_unique=False)
    va_mask = ~tr_mask

    X_tr, y_tr = X_train[tr_mask], y_train[tr_mask]
    X_va, y_va = X_train[va_mask], y_train[va_mask]

    print("Train/valid slices:", X_tr.shape, X_va.shape)
    print("Train/valid unique cases:", len(train_case_ids), len(valid_case_ids))

    AUTOTUNE = tf.data.AUTOTUNE

    if len(X_tr) > 0:
        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
            .shuffle(
                buffer_size=max(1, len(X_tr)), seed=42, reshuffle_each_iteration=True
            )
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(AUTOTUNE)
        )
    else:
        print("Warning: Training split has 0 slices; training will be skipped.")
        train_ds = None

    valid_ds = (
        tf.data.Dataset.from_tensor_slices((X_va, y_va))
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
        if len(X_va) > 0
        else None
    )

    seeds = [11, 22, 33, 44, 55]
    if train_ds is not None:
        for s in seeds:
            seed_everything(s)
            m = build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3))
            m.fit(
                train_ds,
                validation_data=valid_ds,
                epochs=EPOCHS,
                verbose=2,
            )
            ensemble_models.append(m)




## === cell 6
AUTOTUNE = tf.data.AUTOTUNE
test_pred_models = np.zeros((max(1, len(ensemble_models)), 0), dtype=np.float32)

if X_test.shape[0] > 0 and len(ensemble_models) > 0:
    test_ds = tf.data.Dataset.from_tensor_slices(X_test).batch(64).prefetch(AUTOTUNE)
    preds = []
    for m in ensemble_models:
        p = m.predict(test_ds, verbose=0).reshape(-1).astype(np.float32)
        preds.append(p)
    test_pred_models = np.stack(preds, axis=0)  # (n_models, n_slices_total)
    print("Test pred stack:", test_pred_models.shape)
elif X_test.shape[0] == 0:
    print("Warning: X_test is empty; will output 0.5 for all cases.")
else:
    print("Warning: No trained models available; will output 0.5 for all cases.")




## === cell 7
if isinstance(test_pred_models, list):
    test_pred_models = np.asarray(test_pred_models)

slice_pred_mean = (
    test_pred_models.mean(axis=0).astype(np.float32)
    if getattr(test_pred_models, "shape", (0, 0))[1] > 0
    else np.zeros((0,), dtype=np.float32)
)

case_to_preds = {}
for (case_id, slice_idx), pr in zip(test_meta, slice_pred_mean):
    case_to_preds.setdefault(case_id, []).append(float(pr))

test_case_ids = sorted([f.name for f in os.scandir(TEST_DIR) if f.is_dir()])
case_pred = {}
for cid in test_case_ids:
    preds = case_to_preds.get(cid, [])
    if len(preds) == 0:
        case_pred[cid] = 0.5
    else:
        case_pred[cid] = float(np.mean(preds))

print("Aggregated cases:", len(case_pred))




## === cell 8
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(int)

preds_out = []
for brats_id in sample_sub["BraTS21ID"].tolist():
    cid = f"{int(brats_id):05d}"
    preds_out.append(case_pred.get(cid, 0.5))

sub_df = pd.DataFrame(
    {
        "BraTS21ID": sample_sub["BraTS21ID"],
        "MGMT_value": np.asarray(preds_out, dtype=np.float32),
    }
)
sub_df["MGMT_value"] = sub_df["MGMT_value"].clip(0.0, 1.0)

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.dtypes)
print(sub_df.head())
