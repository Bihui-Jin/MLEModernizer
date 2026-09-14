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

None

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.46588) has done: 'I fix the import/runtime issue caused by `pympler` (it’s not needed and triggers a protobuf-related crash) and remove other unused heavyweight imports to keep the environment stable. Since the referenced pretrained models are not available in your provided `/kaggle/input` tree, I replace the failing `load_model()` calls with a minimal TensorFlow/Keras training pipeline that preserves the “image slices → CNN → probability” semantics and still produces a valid submission. I also fix the missing `resize` symbol by avoiding `skimage` entirely and using `tf.image.resize`, which is available and fast. Finally, I correct the submission creation logic (it was computing `prediction` inside a loop but only keeping the last value) and ensure `BraTS21ID` formatting matches the sample submission.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

print("Train dir exists:", os.path.isdir(TRAIN_DIR))
print("Test dir exists:", os.path.isdir(TEST_DIR))
print("Labels csv exists:", os.path.isfile(LABELS_CSV))
print("Sample sub exists:", os.path.isfile(SAMPLE_SUB))

labels_df = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

print(labels_df.head())
print(sample_sub.head())




## === cell 2
IMG_PX_SIZE = 150
CHANNELS = 3


@tf.function
def _resize_to_150_tf(x2d):
    x = tf.cast(x2d[..., None], tf.float32)
    x = tf.image.resize(x, (IMG_PX_SIZE, IMG_PX_SIZE), method="bilinear")
    return tf.squeeze(x, axis=-1)


def _read_dicom_pixel_array(dcm_path: str):
    """Return float32 2D image or None if unreadable."""
    try:
        raw = tf.io.read_file(dcm_path)
        img = tf.io.decode_dicom_image(
            raw,
            dtype=tf.uint16,
            scale="preserve",
            color_dim=False,
            on_error="skip",
        )
        if img is None:
            return None
        img = tf.convert_to_tensor(img)
        if tf.size(img) == 0:
            return None

        img = tf.squeeze(img)
        if img.shape.rank != 2:
            img = tf.reshape(img, [tf.shape(img)[-2], tf.shape(img)[-1]])

        arr = img.numpy().astype(np.float32)
        return arr
    except Exception:
        return None


def _normalize01(x: np.ndarray):
    x = x.astype(np.float32, copy=False)
    x = x - np.min(x)
    mx = np.max(x)
    if mx > 0:
        x = x / mx
    return x


def _resize_to_150(x2d: np.ndarray):
    x = _resize_to_150_tf(tf.convert_to_tensor(x2d, dtype=tf.float32)).numpy()
    return x


def _to_3ch(x2d: np.ndarray):
    x2d = x2d.astype(np.float32, copy=False)
    return np.repeat(x2d[..., None], 3, axis=-1)




## === cell 3
MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]

CACHE_DIR = "/kaggle/working/dcm_cache_t2w_150"
os.makedirs(CACHE_DIR, exist_ok=True)


def _get_modality_dir(case_dir: str, modality: str):
    p = os.path.join(case_dir, modality)
    if os.path.isdir(p):
        return p
    for entry in os.scandir(case_dir):
        if entry.is_dir() and entry.name.lower() == modality.lower():
            return entry.path
    return None


def _select_first_n_dcms(modality_dir: str, n: int):
    numeric = []
    fallback = []
    for e in os.scandir(modality_dir):
        if not (e.is_file() and e.name.lower().endswith(".dcm")):
            continue
        name = e.name
        i = name.rfind("-")
        j = name.lower().rfind(".dcm")
        if i != -1 and j != -1 and i < j:
            token = name[i + 1 : j]
            if token.isdigit():
                numeric.append((int(token), e.path))
                continue
        fallback.append(e.path)

    if numeric:
        numeric.sort(key=lambda t: (t[0], t[1]))
        return [p for _, p in numeric[:n]]
    fallback.sort()
    return fallback[:n]


def load_case_slices(case_dir: str, modality: str, n_slices: int = 6):
    """
    Load up to n_slices 'informative' slices (similar to original filtering),
    returns list of HWC float32 images in [0,1], each (150,150,3).
    """
    modality_dir = _get_modality_dir(case_dir, modality)
    if modality_dir is None:
        return []

    dcm_files = _select_first_n_dcms(modality_dir, n=max(n_slices * 6, 64))
    if not dcm_files:
        return []

    out = []
    count = 0
    for p in dcm_files:
        arr = _read_dicom_pixel_array(p)
        if arr is None:
            continue

        if float(np.sum(arr)) <= 100000.0:
            continue

        arr = _resize_to_150(arr)
        arr = _normalize01(arr)

        img = _to_3ch(arr)
        if float(np.sum(img)) <= 2000.0:
            continue

        out.append(img)
        count += 1
        if count >= n_slices:
            break

    return out


def load_case_slices_cached(case_id: str, case_dir: str, modality: str, n_slices: int):
    cache_path = os.path.join(CACHE_DIR, f"{case_id}_{modality}_n{n_slices}.npz")
    if os.path.isfile(cache_path):
        try:
            data = np.load(cache_path)
            arr = data["arr_0"].astype(np.float32, copy=False)
            return [arr[i] for i in range(arr.shape[0])]
        except Exception:
            pass  # if cache corrupted, fall through and regenerate

    slices = load_case_slices(case_dir, modality=modality, n_slices=n_slices)
    try:
        if slices:
            np.savez_compressed(cache_path, np.asarray(slices, dtype=np.float32))
        else:
            np.savez_compressed(
                cache_path, np.zeros((0, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
            )
    except Exception:
        pass
    return slices




## === cell 4
BAD_CASES = set(["00109", "00123", "00709"])

train_case_ids = sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
train_case_ids = [cid for cid in train_case_ids if cid not in BAD_CASES]
labels_map = dict(
    zip(labels_df["BraTS21ID"], labels_df["MGMT_value"].astype(np.float32))
)

SLICES_PER_CASE = 2  # minimal but stable
MAX_CASES = None  # use all available cases


def make_slice_dataset(
    case_ids, base_dir, modality="T2w", slices_per_case=2, max_cases=None
):
    X, y = [], []
    X_append = X.append
    y_append = y.append

    used = 0
    for cid in case_ids:
        if max_cases is not None and used >= max_cases:
            break
        case_dir = os.path.join(base_dir, cid)

        slices = load_case_slices_cached(
            cid, case_dir, modality=modality, n_slices=slices_per_case
        )
        if not slices:
            used += 1
            continue

        if base_dir == TRAIN_DIR:
            label = labels_map.get(cid, None)
            if label is None:
                used += 1
                continue
            label = float(label)
            for s in slices:
                X_append(s)
                y_append(label)
        else:
            for s in slices:
                X_append((cid, s))
        used += 1

    if base_dir == TRAIN_DIR:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        return X, y
    return X


X_all, y_all = make_slice_dataset(
    train_case_ids,
    TRAIN_DIR,
    modality="T2w",
    slices_per_case=SLICES_PER_CASE,
    max_cases=MAX_CASES,
)
print(
    "Train slices:",
    X_all.shape,
    "Labels:",
    y_all.shape,
    "Pos rate:",
    float(y_all.mean()) if len(y_all) else None,
)




## === cell 5
from sklearn.model_selection import train_test_split

if len(X_all) < 10:
    raise RuntimeError("Too few training slices were extracted; cannot train reliably.")

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=SEED, stratify=(y_all > 0.5)
)

print("Train:", X_train.shape, "Val:", X_val.shape)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_12/2222281706.py in <cell line: 0>()
      2 
      3 if len(X_all) < 10:
----> 4     raise RuntimeError("Too few training slices were extracted; cannot train reliably.")
      5 
      6 X_train, X_val, y_train, y_val = train_test_split(

RuntimeError: Too few training slices were extracted; cannot train reliably.

## === cell 6
def build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, CHANNELS)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model = build_model()
model.summary()




## === cell 7
BATCH_SIZE = 32
EPOCHS = 5

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/1914312873.py in <cell line: 0>()
      3 
      4 history = model.fit(
----> 5     X_train,
      6     y_train,
      7     validation_data=(X_val, y_val),

NameError: name 'X_train' is not defined

## === cell 8
test_case_ids = sorted([d.name for d in os.scandir(TEST_DIR) if d.is_dir()])

prior = float(y_all.mean())
final_preds = []

for cid in test_case_ids:
    case_dir = os.path.join(TEST_DIR, cid)
    slices = load_case_slices_cached(
        cid, case_dir, modality="T2w", n_slices=SLICES_PER_CASE
    )
    if not slices:
        final_preds.append(prior)
        continue

    imgs = np.asarray(slices, dtype=np.float32)
    p = model.predict(imgs, verbose=0).reshape(-1)
    final_preds.append(float(np.mean(p)))

print("Test cases:", len(test_case_ids), "Preds:", len(final_preds))




## === cell 9
pred_df = pd.DataFrame(
    {"BraTS21ID": [str(x).zfill(5) for x in test_case_ids], "MGMT_value": final_preds}
)

sub_df = sample_sub[["BraTS21ID"]].merge(pred_df, on="BraTS21ID", how="left")

sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(np.float32)
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(prior).clip(0.0, 1.0)

print(sub_df.head())
print("Submission rows:", len(sub_df), "Expected:", len(sample_sub))
print("Pred stats:", sub_df["MGMT_value"].describe())




## === cell 10
out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("File size (bytes):", os.path.getsize(out_path))
