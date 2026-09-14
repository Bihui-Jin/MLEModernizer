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

3.10

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

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the import/runtime crash caused by an incompatible `protobuf` version by removing unused heavy/fragile imports (not needed for inference) and setting a safe protobuf environment flag before importing TensorFlow/Keras. I also make the script robust to the missing external pretrained model files by falling back to a deterministic baseline prediction that still produces a valid `submission.csv` in the required format. Next, I fix the image loader `resize` NameError and a few array-type issues (lists being divided by scalars) so the DICOM loading functions can run without exceptions. Finally, I correct the submission creation logic so predictions align per-case and the output IDs are exactly the same as `sample_submission.csv` ordering.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.isdir(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.isfile(SAMPLE_SUB_PATH), f"Missing sample submission: {SAMPLE_SUB_PATH}"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
print("sample_submission shape:", sample_sub.shape)
print(sample_sub.head())



## === cell 2
MODEL_DIR = "/kaggle/input/trained-model-for-rsnamiccai"
MODEL_FILES = [
    "rsna_miccai_114_epochs_T2W_7k_imgs.h5",
    "rsna_miccai_200_epochs_T2W_7k_imgs.h5",
    "rsna_miccai_15_b400_flair_5k_0.73auc_imgs.h5",
    "rsna_miccai_20_b600_t1wce_7k_0.73auc_imgs.h5",
    "rsna_miccai_10_b600_T2w_7k_0.62auc_imgs.h5",
    "rsna_miccai_15_b600_T2w_7k_0.74auc_imgs.h5",
]

models = []
missing = []
for mf in MODEL_FILES:
    mp = os.path.join(MODEL_DIR, mf)
    if os.path.exists(mp):
        try:
            m = keras.models.load_model(mp, compile=False)
            models.append(m)
            print("Loaded model:", mp)
        except Exception as e:
            missing.append((mp, f"load_error: {repr(e)}"))
    else:
        missing.append((mp, "not_found"))

USE_MODELS = len(models) == len(MODEL_FILES)
if not USE_MODELS:
    print("Pretrained models not fully available; will use fallback submission.")
    print("Missing/failed:")
    for x in missing:
        print(" -", x[0], "=>", x[1])
else:
    print("All pretrained models loaded; will run original inference path.")




## === cell 3
def _safe_norm_img(stacked_img: np.ndarray) -> np.ndarray:
    mx = float(np.max(stacked_img))
    if mx <= 0:
        return stacked_img.astype(np.float32)
    return (stacked_img / mx).astype(np.float32)


def _collect_case_ids(path_test: str):
    case_dirs = sorted([f.name for f in os.scandir(path_test) if f.is_dir()])
    return case_dirs


def load_test_images_by_modality(
    path_test: str, modality_index: int, img_px_size: int = 150, per_case_imgs: int = 6
):
    """
    modality_index matches original code's sorted modality folder order:
      0: FLAIR, 1: T1w, 2: T1wCE, 3: T2w (as used in the provided code).
    Returns: list of length per_case_imgs, each an (N, H, W, 3) float32 array.
    """
    arrays = [[] for _ in range(per_case_imgs)]
    case_ids = _collect_case_ids(path_test)

    for case_id in case_ids:
        case_path = os.path.join(path_test, case_id)
        modalities = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(modalities) <= modality_index:
            continue

        img_dir = modalities[modality_index]
        img_paths = sorted([f.path for f in os.scandir(img_dir) if f.is_file()])
        count = 0

        for ip in img_paths:
            try:
                dcm = dicom.dcmread(ip)
                px = dcm.pixel_array
            except Exception:
                continue

            if px.sum() > 100000:
                resized_img = resize(
                    px,
                    (img_px_size, img_px_size),
                    anti_aliasing=True,
                    preserve_range=True,
                )
                img = np.array(resized_img, dtype=np.float32)
                stacked_img = np.stack((img,) * 3, axis=-1)
                stacked_img_normalize = _safe_norm_img(stacked_img)

                if stacked_img_normalize.sum() > 2000:
                    if count < per_case_imgs:
                        arrays[count].append(stacked_img_normalize)
                        count += 1
                    if count >= per_case_imgs:
                        break

    out = []
    for a in arrays:
        a = np.asarray(a, dtype=np.float32)
        if a.size == 0:
            out.append(a)
            continue
        mx = float(np.max(a))
        if mx > 0:
            a = a / mx
        out.append(a)
    return out




## === cell 4
def load_test_T2W_images(path_test):
    arrays = load_test_images_by_modality(
        path_test, modality_index=3, img_px_size=150, per_case_imgs=6
    )
    print("Number of T2 images loaded are", ", ".join(str(len(x)) for x in arrays))
    return tuple(arrays)


def load_test_flair_images(path_test):
    arrays = load_test_images_by_modality(
        path_test, modality_index=0, img_px_size=150, per_case_imgs=6
    )
    print("Number of flair images loaded are", ", ".join(str(len(x)) for x in arrays))
    return tuple(arrays)


def load_test_T1wce_images(path_test):
    arrays = load_test_images_by_modality(
        path_test, modality_index=2, img_px_size=150, per_case_imgs=6
    )
    print("Number of T1wce images loaded are", ", ".join(str(len(x)) for x in arrays))
    return tuple(arrays)




## === cell 5
if USE_MODELS:
    pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(
        TEST_DIR
    )
    pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = (
        load_test_flair_images(TEST_DIR)
    )
    pixels_13, pixels_14, pixels_15, pixels_16, pixels_17, pixels_18 = (
        load_test_T1wce_images(TEST_DIR)
    )




## === cell 6
def _predict_class1_prob(model, x: np.ndarray) -> np.ndarray:
    """Return P(class=1). Supports sigmoid (N,1) or softmax (N,2)."""
    pred = model.predict(x, verbose=0)
    pred = np.asarray(pred)
    if pred.ndim == 2 and pred.shape[1] == 2:
        return pred[:, 1].astype(np.float32)
    if pred.ndim == 2 and pred.shape[1] == 1:
        return pred[:, 0].astype(np.float32)
    return pred.reshape(-1).astype(np.float32)




## === cell 7
def create_sub(
    path_test,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p101,
    p102,
    p103,
    p104,
    p105,
    p106,
    p401,
    p402,
    p403,
    p404,
    p405,
    p406,
    p501,
    p502,
    p503,
    p504,
    p505,
    p506,
):
    """
    Fix: original code recomputed 'prediction' inside loop but used full-length vectors,
    producing a mismatched submission. Here we compute per-case average across available
    per-slice predictions and then align to sample_submission order.
    """
    case_ids = _collect_case_ids(path_test)
    n = len(case_ids)

    pred_matrix = np.vstack(
        [
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p101,
            p102,
            p103,
            p104,
            p105,
            p106,
            p401,
            p402,
            p403,
            p404,
            p405,
            p406,
            p501,
            p502,
            p503,
            p504,
            p505,
            p506,
        ]
    ).astype(
        np.float32
    )  # (24, n)

    prediction = pred_matrix.mean(axis=0)  # (n,)

    df = pd.DataFrame(
        {"BraTS21ID": [int(x) for x in case_ids], "MGMT_value": prediction}
    )
    df = sample_sub[["BraTS21ID"]].merge(df, on="BraTS21ID", how="left")
    return df




## === cell 8
if USE_MODELS:
    model_T2, model_T2_2, model_T2_3, model_T2_4, model_T2_5, model_T2_6 = models

    prediction_1 = _predict_class1_prob(model_T2, pixels_1)
    prediction_2 = _predict_class1_prob(model_T2, pixels_2)
    prediction_3 = _predict_class1_prob(model_T2, pixels_3)
    prediction_4 = _predict_class1_prob(model_T2, pixels_4)
    prediction_5 = _predict_class1_prob(model_T2, pixels_5)
    prediction_6 = _predict_class1_prob(model_T2, pixels_6)

    prediction_101 = _predict_class1_prob(model_T2_2, pixels_1)
    prediction_102 = _predict_class1_prob(model_T2_2, pixels_2)
    prediction_103 = _predict_class1_prob(model_T2_2, pixels_3)
    prediction_104 = _predict_class1_prob(model_T2_2, pixels_4)
    prediction_105 = _predict_class1_prob(model_T2_2, pixels_5)
    prediction_106 = _predict_class1_prob(model_T2_2, pixels_6)


    prediction_401 = _predict_class1_prob(model_T2_5, pixels_1)
    prediction_402 = _predict_class1_prob(model_T2_5, pixels_2)
    prediction_403 = _predict_class1_prob(model_T2_5, pixels_3)
    prediction_404 = _predict_class1_prob(model_T2_5, pixels_4)
    prediction_405 = _predict_class1_prob(model_T2_5, pixels_5)
    prediction_406 = _predict_class1_prob(model_T2_5, pixels_6)

    prediction_501 = _predict_class1_prob(model_T2_6, pixels_1)
    prediction_502 = _predict_class1_prob(model_T2_6, pixels_2)
    prediction_503 = _predict_class1_prob(model_T2_6, pixels_3)
    prediction_504 = _predict_class1_prob(model_T2_6, pixels_4)
    prediction_505 = _predict_class1_prob(model_T2_6, pixels_5)
    prediction_506 = _predict_class1_prob(model_T2_6, pixels_6)

    sub_df = create_sub(
        TEST_DIR,
        prediction_1,
        prediction_2,
        prediction_3,
        prediction_4,
        prediction_5,
        prediction_6,
        prediction_101,
        prediction_102,
        prediction_103,
        prediction_104,
        prediction_105,
        prediction_106,
        prediction_401,
        prediction_402,
        prediction_403,
        prediction_404,
        prediction_405,
        prediction_406,
        prediction_501,
        prediction_502,
        prediction_503,
        prediction_504,
        prediction_505,
        prediction_506,
    )
else:
    sub_df = sample_sub.copy()
    sub_df["MGMT_value"] = 0.5

print(sub_df.head())
print("Submission shape:", sub_df.shape)
print(
    "MGMT_value min/max:",
    float(sub_df["MGMT_value"].min()),
    float(sub_df["MGMT_value"].max()),
)



## === cell 9
sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(np.float32).clip(0.0, 1.0)
sub_df = sub_df[["BraTS21ID", "MGMT_value"]]
sub_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(pd.read_csv("submission.csv").head())
