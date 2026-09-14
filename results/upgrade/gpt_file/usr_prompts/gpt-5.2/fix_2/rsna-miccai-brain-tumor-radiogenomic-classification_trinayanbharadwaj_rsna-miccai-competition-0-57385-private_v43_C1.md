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

0.48941

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.48941) has done: 'I remove/guard the problematic imports that trigger the protobuf `MessageFactory.GetPrototype` error, and add fallbacks so the notebook runs even if `pydicom`, `skimage`, or `cv2` aren’t available. Because the referenced pre-trained `.h5` models aren’t present, I keep the same “single-slice CNN probability averaging” core idea but replace model loading with a tiny Keras model trained quickly on the provided training set using the same image extraction logic, so we can generate real probabilities and a valid submission. I also fix multiple logic/runtime bugs: missing `resize`, invalid list/float division, an indentation error in `create_sub`, and incorrect placement of prediction computation inside the loop. Finally, I ensure the submission IDs match `sample_submission.csv` exactly (including zero-padding) and write `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import seaborn as sns
except Exception:
    sns = None

try:
    import pydicom as dicom
except Exception as e:
    raise ImportError(
        "pydicom is required to read the provided DICOM files but could not be imported."
    ) from e

try:
    from skimage.transform import resize as sk_resize
except Exception:
    sk_resize = None
try:
    from PIL import Image
except Exception:
    Image = None

import tensorflow as tf
from tensorflow import keras
from keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

INPUT_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(INPUT_ROOT, "train")
TEST_DIR = os.path.join(INPUT_ROOT, "test")
LABELS_CSV = os.path.join(INPUT_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(INPUT_ROOT, "sample_submission.csv")

print("Train dir exists:", os.path.isdir(TRAIN_DIR))
print("Test dir exists:", os.path.isdir(TEST_DIR))
print("Labels csv exists:", os.path.isfile(LABELS_CSV))
print("Sample submission exists:", os.path.isfile(SAMPLE_SUB))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _resize_2d(img2d: np.ndarray, out_size: int) -> np.ndarray:
    """Resize 2D image to (out_size, out_size) with safe fallbacks."""
    img2d = img2d.astype(np.float32)
    if sk_resize is not None:
        return sk_resize(
            img2d, (out_size, out_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
    if Image is None:
        raise ImportError("Neither skimage nor PIL are available for resizing.")
    pil = Image.fromarray(img2d)
    pil = pil.resize((out_size, out_size), resample=Image.BILINEAR)
    return np.asarray(pil).astype(np.float32)


def _normalize_stack(img2d: np.ndarray) -> np.ndarray:
    """Stack to 3 channels and normalize robustly to [0,1]."""
    img = img2d.astype(np.float32)
    mn = np.min(img)
    img = img - mn
    mx = np.max(img)
    if mx > 0:
        img = img / mx
    stacked = np.stack([img, img, img], axis=-1).astype(np.float32)
    return stacked


def _sorted_case_dirs(path_root: str):
    return sorted([f.path for f in os.scandir(path_root) if f.is_dir()])


def _sorted_mri_dirs(case_dir: str):
    return sorted([f.path for f in os.scandir(case_dir) if f.is_dir()])


def _sorted_dcm_files(mri_dir: str):
    files = [f.path for f in os.scandir(mri_dir) if f.is_file()]
    return sorted(files)


def _load_n_slices_for_modality(
    path_root: str,
    modality_index: int,
    n_slices: int,
    img_px_size: int = 299,
    pixel_sum_thresh: float = 100000.0,
    norm_sum_thresh: float = 100.0,
):
    """
    For each case, load up to n_slices "valid" slices from a modality folder by scanning DICOMs.
    Returns list of length n_slices, each element is (num_cases, H, W, 3) float32.
    """
    case_dirs = _sorted_case_dirs(path_root)
    arrays = [[] for _ in range(n_slices)]

    for case_dir in case_dirs:
        mri_dirs = _sorted_mri_dirs(case_dir)
        if len(mri_dirs) <= modality_index:
            for s in range(n_slices):
                arrays[s].append(
                    np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
                )
            continue

        dcm_files = _sorted_dcm_files(mri_dirs[modality_index])
        count = 0
        for fp in dcm_files:
            if count >= n_slices:
                break
            try:
                dcm = dicom.dcmread(fp, force=True)
                px = dcm.pixel_array
            except Exception:
                continue

            if px is None:
                continue
            if float(np.sum(px)) <= pixel_sum_thresh:
                continue

            resized = _resize_2d(px, img_px_size)
            stacked = _normalize_stack(resized)

            if float(np.sum(stacked)) <= norm_sum_thresh:
                continue

            arrays[count].append(stacked)
            count += 1

        while count < n_slices:
            arrays[count].append(
                np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
            )
            count += 1

    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    return arrays


def load_test_flair_images(path_test):
    arrays = _load_n_slices_for_modality(
        path_test, modality_index=0, n_slices=6, img_px_size=299
    )
    print(
        "Number of FLAIR images loaded are", ", ".join(str(a.shape[0]) for a in arrays)
    )
    return tuple(arrays)


def load_test_T2W_images(path_test):
    arrays = _load_n_slices_for_modality(
        path_test, modality_index=3, n_slices=6, img_px_size=299
    )
    print("Number of T2w images loaded are", ", ".join(str(a.shape[0]) for a in arrays))
    return tuple(arrays)




## === cell 2
test = TEST_DIR
train = TRAIN_DIR

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)

sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

bad_cases = {"00109", "00123", "00709"}
labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_cases)].reset_index(drop=True)

print("Train labels:", labels_df.shape, "Test ids:", sample_sub.shape)



## === cell 3
train_flair = load_test_flair_images(train)
train_t2w = load_test_T2W_images(train)

pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_flair_images(
    test
)
pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_T2W_images(
    test
)

train_case_dirs = sorted([f.name for f in os.scandir(train) if f.is_dir()])
train_ids = pd.Series(train_case_dirs, dtype=str).str.zfill(5).tolist()

id_to_label = dict(
    zip(
        labels_df["BraTS21ID"].tolist(),
        labels_df["MGMT_value"].astype(np.float32).tolist(),
    )
)
y_train = np.array([id_to_label.get(i, np.nan) for i in train_ids], dtype=np.float32)

keep_idx = ~np.isnan(y_train)
y_train = y_train[keep_idx].astype(np.float32)

train_flair = tuple(a[keep_idx] for a in train_flair)
train_t2w = tuple(a[keep_idx] for a in train_t2w)

print("Effective training cases:", y_train.shape[0])




## === cell 4
def build_model(input_shape=(299, 299, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


model_1 = build_model()

X_train_all = np.concatenate(list(train_flair) + list(train_t2w), axis=0)
y_train_all = np.concatenate([y_train] * 12, axis=0)

perm = np.random.RandomState(SEED).permutation(len(y_train_all))
X_train_all = X_train_all[perm]
y_train_all = y_train_all[perm]

BATCH_SIZE = 16
EPOCHS = 2

model_1.fit(X_train_all, y_train_all, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)


def _predict_proba(model, X):
    p = model.predict(X, batch_size=16, verbose=0).reshape(-1)
    return np.clip(p.astype(np.float32), 1e-6, 1 - 1e-6)


prediction_1 = _predict_proba(model_1, pixels_1)
prediction_2 = _predict_proba(model_1, pixels_2)
prediction_3 = _predict_proba(model_1, pixels_3)
prediction_4 = _predict_proba(model_1, pixels_4)
prediction_5 = _predict_proba(model_1, pixels_5)
prediction_6 = _predict_proba(model_1, pixels_6)

prediction_7 = _predict_proba(model_1, pixels_7)
prediction_8 = _predict_proba(model_1, pixels_8)
prediction_9 = _predict_proba(model_1, pixels_9)
prediction_10 = _predict_proba(model_1, pixels_10)
prediction_11 = _predict_proba(model_1, pixels_11)
prediction_12 = _predict_proba(model_1, pixels_12)

print("Pred shapes:", prediction_1.shape, prediction_12.shape)




## === cell 5
def create_sub(path_test, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12):
    case_dirs = _sorted_case_dirs(path_test)
    cases = [os.path.basename(p).zfill(5) for p in case_dirs]

    prediction = (
        p1.astype(np.float32)
        + p2.astype(np.float32)
        + p3.astype(np.float32)
        + p4.astype(np.float32)
        + p5.astype(np.float32)
        + p6.astype(np.float32)
        + p7.astype(np.float32)
        + p8.astype(np.float32)
        + p9.astype(np.float32)
        + p10.astype(np.float32)
        + p11.astype(np.float32)
        + p12.astype(np.float32)
    ) / 12.0

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction.astype(np.float32)})
    return df


sub_df = create_sub(
    test,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
    prediction_7,
    prediction_8,
    prediction_9,
    prediction_10,
    prediction_11,
    prediction_12,
)

sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")

sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(np.float32)

sub_df.head(), sub_df.shape



## === cell 6
if sns is not None:
    try:
        sns.displot(sub_df["MGMT_value"])
        if plt is not None:
            plt.show()
    except Exception as e:
        print("Plotting skipped due to:", repr(e))



## === cell 7
out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub_df.columns.tolist())
print(sub_df.head(3).to_string(index=False))
