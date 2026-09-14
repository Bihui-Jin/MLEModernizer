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

- What this solution (achieved 0.53765) has done: 'I fix the two root runtime blockers: the protobuf/pydicom import crash and the empty-modality directory causing an IndexError in the test loader. Then I make the image loader return a prediction-aligned list of case IDs and guarantee each case contributes exactly 6 slices (with safe fallbacks) so predictions and `BraTS21ID` stay correctly aligned. Finally, I ensure a valid `submission.csv` is always written in the required format by merging onto `sample_submission.csv` and filling any missing predictions with 0.5 (score-neutral fallback).'
- What this solution (achieved 0.44706) has done: 'I fix the runtime crash happening at import time (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by hardening the protobuf/pydicom initialization so it reliably uses the pure-Python protobuf implementation before any library triggers protobuf loading. I also make the DICOM import path more robust by falling back to TensorFlow-based resizing when `skimage` isn’t available (score-neutral) and keeping your existing model/prediction logic unchanged. Finally, I keep the submission creation exactly the same but ensure `BraTS21ID` is consistently treated as a zero-padded 5-character string to avoid any accidental merge misalignment with `sample_submission.csv` (score-neutral correctness).'
- What this solution (achieved 0.55647) has done: 'I fix the import-time crash by forcing the pure-Python protobuf implementation before anything that may indirectly import protobuf (including TensorFlow and pydicom), and by defensively importing pydicom via its stable submodule path. I also add a safe fallback so that if pydicom still can’t be imported/used at runtime, the loader won’t crash and the notebook still produce a valid `submission.csv` (with neutral 0.5 predictions if no images load). These changes are execution/stability focused and keep your existing model, slice selection, prediction averaging, and submission formatting logic unchanged, so score behavior should remain consistent aside from no longer failing at import time.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash (`MessageFactory` / protobuf) by forcing the pure-Python protobuf runtime before TensorFlow loads and by avoiding indirect protobuf usage at import time; this unblocks the whole pipeline. Then I make the DICOM reader path more robust by importing `pydicom.dcmread` directly (fallback-safe) so image loading works reliably in Kaggle. These changes are execution/stability focused and preserve your existing model, slice selection, prediction averaging, and submission formatting logic, so the score should stay in the same range (or improve only insofar as more images load instead of falling back to 0.5). Finally, I keep the submission merge-on-sample approach to guarantee the correct row order and a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
import pandas as pd

np.random.seed(0)

import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(0)

try:
    import pydicom  # noqa: F401
    from pydicom.dcmread import dcmread as _dcmread  # stable import path
except Exception:
    _dcmread = None

try:
    from skimage.transform import resize as sk_resize  # type: ignore
except Exception:
    sk_resize = None


def resize_image(img2d: np.ndarray, size: int) -> np.ndarray:
    """Resize a 2D image to (size, size) with float32 output."""
    img2d = img2d.astype(np.float32, copy=False)
    if sk_resize is not None:
        out = sk_resize(
            img2d, (size, size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        return out
    x = tf.convert_to_tensor(img2d[None, ..., None], dtype=tf.float32)  # (1,H,W,1)
    x = tf.image.resize(x, (size, size), method="bilinear", antialias=True)
    return x[0, ..., 0].numpy().astype(np.float32)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def build_prob_model(input_shape=(150, 150, 3), seed=0):
    tf.random.set_seed(seed)
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    outputs = keras.layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


model_T2 = build_prob_model(seed=1)
model_T2_2 = build_prob_model(seed=2)




## === cell 2
def load_test_T2W_images(path_test):
    """
    Loads up to 6 slices per case (T2w preferred) and returns:
      kept_cases, array_1..array_6 each with shape (N,150,150,3)
    Safeguards:
      - skip empty/broken cases
      - ensure exactly 6 images per kept case (pad by last valid slice)
    """
    IMG_PX_SIZE = 150
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    kept_cases = []

    def normalize_to_3ch(px2d: np.ndarray) -> np.ndarray:
        resized_img = resize_image(px2d, IMG_PX_SIZE).astype(np.float32)
        stacked = np.stack((resized_img,) * 3, axis=-1)  # (H,W,3)
        mx = float(np.max(stacked))
        if mx > 0:
            stacked = stacked / mx
        return stacked.astype(np.float32)

    def safe_read_pixel_array(dcm_path: str):
        if _dcmread is None:
            return None
        try:
            dcm = _dcmread(dcm_path, force=True)
            px = getattr(dcm, "pixel_array", None)
            if px is None:
                return None
            return px
        except Exception:
            return None

    for case_path in path_cases:
        case_id = os.path.basename(case_path)

        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) == 0:
            continue

        t2_candidates = [p for p in mri_type if os.path.basename(p).lower() == "t2w"]
        modality_path = t2_candidates[0] if len(t2_candidates) else mri_type[-1]

        img_files = sorted([f.path for f in os.scandir(modality_path) if f.is_file()])
        if len(img_files) == 0:
            continue

        selected_imgs = []
        for p in img_files:
            px = safe_read_pixel_array(p)
            if px is None:
                continue
            try:
                if float(np.sum(px)) <= 100000:
                    continue
            except Exception:
                continue

            img3 = normalize_to_3ch(px)
            if float(np.sum(img3)) > 2000:
                selected_imgs.append(img3)
                if len(selected_imgs) >= 6:
                    break

        if len(selected_imgs) == 0:
            continue
        if len(selected_imgs) < 6:
            selected_imgs = selected_imgs + [selected_imgs[-1]] * (
                6 - len(selected_imgs)
            )

        array_1.append(selected_imgs[0])
        array_2.append(selected_imgs[1])
        array_3.append(selected_imgs[2])
        array_4.append(selected_imgs[3])
        array_5.append(selected_imgs[4])
        array_6.append(selected_imgs[5])
        kept_cases.append(case_id)

    def finalize(arr_list):
        arr = np.asarray(arr_list, dtype=np.float32)
        if arr.size == 0:
            return arr
        denom = float(np.max(arr))
        if denom > 0:
            arr = arr / denom
        return arr

    array_1 = finalize(array_1)
    array_2 = finalize(array_2)
    array_3 = finalize(array_3)
    array_4 = finalize(array_4)
    array_5 = finalize(array_5)
    array_6 = finalize(array_6)

    print(
        "Number of T2 images loaded are ",
        len(array_1),
        ",",
        len(array_2),
        ",",
        len(array_3),
        ",",
        len(array_4),
        ",",
        len(array_5),
        ",",
        len(array_6),
    )
    print("Number of kept test cases:", len(kept_cases))

    return kept_cases, array_1, array_2, array_3, array_4, array_5, array_6




## === cell 3
test = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
if not os.path.isdir(test):
    alt = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
    if os.path.isdir(alt):
        test = alt



## === cell 4
case_ids, pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = (
    load_test_T2W_images(test)
)


def ensure_batch(x):
    return np.asarray(x, dtype=np.float32)


pixels_1 = ensure_batch(pixels_1)
pixels_2 = ensure_batch(pixels_2)
pixels_3 = ensure_batch(pixels_3)
pixels_4 = ensure_batch(pixels_4)
pixels_5 = ensure_batch(pixels_5)
pixels_6 = ensure_batch(pixels_6)

lengths = [
    len(case_ids),
    len(pixels_1),
    len(pixels_2),
    len(pixels_3),
    len(pixels_4),
    len(pixels_5),
    len(pixels_6),
]
min_len = min(lengths) if len(lengths) else 0

if min_len == 0:
    case_ids = []
    pixels_1 = pixels_2 = pixels_3 = pixels_4 = pixels_5 = pixels_6 = np.zeros(
        (0, 150, 150, 3), dtype=np.float32
    )
else:
    case_ids = case_ids[:min_len]
    pixels_1 = pixels_1[:min_len]
    pixels_2 = pixels_2[:min_len]
    pixels_3 = pixels_3[:min_len]
    pixels_4 = pixels_4[:min_len]
    pixels_5 = pixels_5[:min_len]
    pixels_6 = pixels_6[:min_len]



## === cell 5
if len(case_ids) == 0:
    prediction_1 = prediction_2 = prediction_3 = prediction_4 = prediction_5 = (
        prediction_6
    ) = np.array([], dtype=float)
    prediction_101 = prediction_102 = prediction_103 = prediction_104 = (
        prediction_105
    ) = prediction_106 = np.array([], dtype=float)
else:
    preds_1 = model_T2.predict(pixels_1, verbose=0)
    prediction_1 = preds_1[:, 1]

    preds_2 = model_T2.predict(pixels_2, verbose=0)
    prediction_2 = preds_2[:, 1]

    preds_3 = model_T2.predict(pixels_3, verbose=0)
    prediction_3 = preds_3[:, 1]

    preds_4 = model_T2.predict(pixels_4, verbose=0)
    prediction_4 = preds_4[:, 1]

    preds_5 = model_T2.predict(pixels_5, verbose=0)
    prediction_5 = preds_5[:, 1]

    preds_6 = model_T2.predict(pixels_6, verbose=0)
    prediction_6 = preds_6[:, 1]

    preds_101 = model_T2_2.predict(pixels_1, verbose=0)
    prediction_101 = preds_101[:, 1]

    preds_102 = model_T2_2.predict(pixels_2, verbose=0)
    prediction_102 = preds_102[:, 1]

    preds_103 = model_T2_2.predict(pixels_3, verbose=0)
    prediction_103 = preds_103[:, 1]

    preds_104 = model_T2_2.predict(pixels_4, verbose=0)
    prediction_104 = preds_104[:, 1]

    preds_105 = model_T2_2.predict(pixels_5, verbose=0)
    prediction_105 = preds_105[:, 1]

    preds_106 = model_T2_2.predict(pixels_6, verbose=0)
    prediction_106 = preds_106[:, 1]




## === cell 6
def create_sub(case_ids, p1, p2, p3, p101, p102, p103):
    n = min(len(case_ids), len(p1), len(p2), len(p3), len(p101), len(p102), len(p103))
    cases = case_ids[:n]

    prediction = (
        p1[:n].astype(float)
        + p2[:n].astype(float)
        + p3[:n].astype(float)
        + p101[:n].astype(float)
        + p102[:n].astype(float)
        + p103[:n].astype(float)
    ) / 6.0

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    return df




## === cell 7
sub_df = create_sub(
    case_ids,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_101,
    prediction_102,
    prediction_103,
)

sample_path = (
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)

if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path, dtype={"BraTS21ID": str})
    sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)
    sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
    sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).fillna(0.5)
else:
    sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).fillna(0.5)



## === cell 8
sub_df.head(), sub_df.shape, sub_df["MGMT_value"].describe()



## === cell 9
pass



## === cell 10
sub_df = sub_df[["BraTS21ID", "MGMT_value"]]
sub_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
