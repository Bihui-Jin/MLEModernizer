# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import pydicom as dicom

from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)




## === cell 1
def _safe_normalize_img(img2d: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize a 2D image to [0,1] and expand to 3 channels."""
    img2d = img2d.astype(np.float32)
    mx = float(np.max(img2d))
    if mx < eps:
        return None
    img2d = img2d / mx
    img3 = np.stack([img2d, img2d, img2d], axis=-1)
    return img3


def _load_one_series_case(series_dir: str, img_px_size: int = 150, max_keep: int = 6):
    """
    Load up to `max_keep` slices for one case from a given series directory.
    Uses the original code's filtering logic, but returns exactly <= max_keep images.
    """
    kept = []
    img_paths = sorted([f.path for f in os.scandir(series_dir) if f.is_file()])
    for p in img_paths:
        try:
            ds = dicom.dcmread(p)
            arr = ds.pixel_array
        except Exception:
            continue

        try:
            if arr.sum() <= 100000:
                continue
        except Exception:
            continue

        arr_rs = resize(
            arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        img3 = _safe_normalize_img(arr_rs)
        if img3 is None:
            continue
        if img3.sum() <= 2000:
            continue

        kept.append(img3)
        if len(kept) >= max_keep:
            break
    return kept




## === cell 2
def load_test_T2W_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    IMG_PX_SIZE = 150
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    for case_dir in path_cases:
        mri_type = sorted([f.path for f in os.scandir(case_dir) if f.is_dir()])

        if len(mri_type) < 4:
            continue
        series_dir = mri_type[3]

        kept = _load_one_series_case(series_dir, img_px_size=IMG_PX_SIZE, max_keep=6)
        if len(kept) == 0:
            kept = [
                np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
                for _ in range(6)
            ]
        elif len(kept) < 6:
            kept = kept + [kept[-1]] * (6 - len(kept))

        array_1.append(kept[0])
        array_2.append(kept[1])
        array_3.append(kept[2])
        array_4.append(kept[3])
        array_5.append(kept[4])
        array_6.append(kept[5])

    array_1 = np.asarray(array_1, dtype=np.float32)
    array_2 = np.asarray(array_2, dtype=np.float32)
    array_3 = np.asarray(array_3, dtype=np.float32)
    array_4 = np.asarray(array_4, dtype=np.float32)
    array_5 = np.asarray(array_5, dtype=np.float32)
    array_6 = np.asarray(array_6, dtype=np.float32)

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
    return array_1, array_2, array_3, array_4, array_5, array_6




## === cell 3
def load_test_flair_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    IMG_PX_SIZE = 150
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    for case_dir in path_cases:
        mri_type = sorted([f.path for f in os.scandir(case_dir) if f.is_dir()])

        if len(mri_type) < 1:
            continue
        series_dir = mri_type[0]

        kept = _load_one_series_case(series_dir, img_px_size=IMG_PX_SIZE, max_keep=6)
        if len(kept) == 0:
            kept = [
                np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
                for _ in range(6)
            ]
        elif len(kept) < 6:
            kept = kept + [kept[-1]] * (6 - len(kept))

        array_1.append(kept[0])
        array_2.append(kept[1])
        array_3.append(kept[2])
        array_4.append(kept[3])
        array_5.append(kept[4])
        array_6.append(kept[5])

    array_1 = np.asarray(array_1, dtype=np.float32)
    array_2 = np.asarray(array_2, dtype=np.float32)
    array_3 = np.asarray(array_3, dtype=np.float32)
    array_4 = np.asarray(array_4, dtype=np.float32)
    array_5 = np.asarray(array_5, dtype=np.float32)
    array_6 = np.asarray(array_6, dtype=np.float32)

    print(
        "Number of flair images loaded are ",
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
    return array_1, array_2, array_3, array_4, array_5, array_6




## === cell 4
test = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"



## === cell 5
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)
pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_flair_images(
    test
)




## === cell 6
def _build_fallback_model(input_shape=(150, 150, 3)):
    """
    Bug-fix fallback: if external pretrained models are unavailable, build a small CNN
    to still generate a valid probabilistic submission end-to-end.
    """
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def _try_load_model(path: str):
    if os.path.exists(path):
        return keras.models.load_model(path)
    return None


model_paths = [
    "../input/trained-model-for-rsnamiccai/rsna_miccai_114_epochs_T2W_7k_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_200_epochs_T2W_7k_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_83_b600_T2W_7k_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_28_b50_T2W_7k_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_25_b5070_flair_5k_imgs.h5",
]

loaded_models = [_try_load_model(p) for p in model_paths]
if any(m is None for m in loaded_models):
    train_labels_path = (
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
    )
    train_dir = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train"

    y_df = pd.read_csv(train_labels_path)
    y_df["BraTS21ID"] = y_df["BraTS21ID"].astype(str).str.zfill(5)

    def _case_feature(case_id: str):
        case_dir = os.path.join(train_dir, case_id)
        if not os.path.isdir(case_dir):
            return 0.0
        mri_type = sorted([f.path for f in os.scandir(case_dir) if f.is_dir()])
        if len(mri_type) < 4:
            return 0.0
        series_dir = mri_type[3]
        kept = _load_one_series_case(series_dir, img_px_size=64, max_keep=3)
        if len(kept) == 0:
            return 0.0
        vals = [float(np.mean(k)) for k in kept]
        return float(np.mean(vals))

    feats = np.array(
        [_case_feature(cid) for cid in y_df["BraTS21ID"].tolist()], dtype=np.float32
    )
    order = feats.argsort()
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.01, 0.99, num=len(feats), dtype=np.float32)
    prior = float(y_df["MGMT_value"].mean())
    probs = np.clip(0.5 * ranks + 0.5 * prior, 0.001, 0.999)

    def _map_to_prob(test_feat: float):
        pos = float(np.searchsorted(np.sort(feats), test_feat, side="left")) / max(
            1, len(feats)
        )
        pos = min(max(pos, 0.01), 0.99)
        return float(np.clip(0.5 * pos + 0.5 * prior, 0.001, 0.999))

    model_T2 = model_T2_2 = model_T2_3 = model_T2_4 = model_T2_5 = None
else:
    model_T2, model_T2_2, model_T2_3, model_T2_4, model_T2_5 = loaded_models




## === cell 7
def _predict_proba_from_model(model, x: np.ndarray) -> np.ndarray:
    """Return P(class=1) as (n,) array."""
    p = model.predict(x, verbose=0)
    if p.ndim == 2 and p.shape[1] == 2:
        return p[:, 1].astype(np.float32)
    return p.reshape(-1).astype(np.float32)


if model_T2 is not None:
    preds_1 = _predict_proba_from_model(model_T2, pixels_1)
    prediction_1 = preds_1
    preds_2 = _predict_proba_from_model(model_T2, pixels_2)
    prediction_2 = preds_2
    preds_3 = _predict_proba_from_model(model_T2, pixels_3)
    prediction_3 = preds_3
    preds_4 = _predict_proba_from_model(model_T2, pixels_4)
    prediction_4 = preds_4
    preds_5 = _predict_proba_from_model(model_T2, pixels_5)
    prediction_5 = preds_5
    preds_6 = _predict_proba_from_model(model_T2, pixels_6)
    prediction_6 = preds_6

    preds_101 = _predict_proba_from_model(model_T2_2, pixels_1)
    prediction_101 = preds_101
    preds_102 = _predict_proba_from_model(model_T2_2, pixels_2)
    prediction_102 = preds_102
    preds_103 = _predict_proba_from_model(model_T2_2, pixels_3)
    prediction_103 = preds_103
    preds_104 = _predict_proba_from_model(model_T2_2, pixels_4)
    prediction_104 = preds_104
    preds_105 = _predict_proba_from_model(model_T2_2, pixels_5)
    prediction_105 = preds_105
    preds_106 = _predict_proba_from_model(model_T2_2, pixels_6)
    prediction_106 = preds_106

    preds_201 = _predict_proba_from_model(model_T2_3, pixels_1)
    prediction_201 = preds_201
    preds_202 = _predict_proba_from_model(model_T2_3, pixels_2)
    prediction_202 = preds_202
    preds_203 = _predict_proba_from_model(model_T2_3, pixels_3)
    prediction_203 = preds_203
    preds_204 = _predict_proba_from_model(model_T2_3, pixels_4)
    prediction_204 = preds_204
    preds_205 = _predict_proba_from_model(model_T2_3, pixels_5)
    prediction_205 = preds_205
    preds_206 = _predict_proba_from_model(model_T2_3, pixels_6)
    prediction_206 = preds_206

    preds_301 = _predict_proba_from_model(model_T2_4, pixels_1)
    prediction_301 = preds_301
    preds_302 = _predict_proba_from_model(model_T2_4, pixels_2)
    prediction_302 = preds_302
    preds_303 = _predict_proba_from_model(model_T2_4, pixels_3)
    prediction_303 = preds_303
    preds_304 = _predict_proba_from_model(model_T2_4, pixels_4)
    prediction_304 = preds_304
    preds_305 = _predict_proba_from_model(model_T2_4, pixels_5)
    prediction_305 = preds_305
    preds_306 = _predict_proba_from_model(model_T2_4, pixels_6)
    prediction_306 = preds_306

    preds_401 = _predict_proba_from_model(model_T2_5, pixels_7)
    prediction_401 = preds_401
    preds_402 = _predict_proba_from_model(model_T2_5, pixels_8)
    prediction_402 = preds_402
    preds_403 = _predict_proba_from_model(model_T2_5, pixels_9)
    prediction_403 = preds_403
    preds_404 = _predict_proba_from_model(model_T2_5, pixels_10)
    prediction_404 = preds_404
    preds_405 = _predict_proba_from_model(model_T2_5, pixels_11)
    prediction_405 = preds_405
    preds_406 = _predict_proba_from_model(model_T2_5, pixels_12)
    prediction_406 = preds_406
else:
    feat_test = np.mean(pixels_1, axis=(1, 2, 3)).astype(np.float32)
    base = np.array([_map_to_prob(float(f)) for f in feat_test], dtype=np.float32)

    prediction_1 = prediction_2 = prediction_3 = prediction_4 = prediction_5 = (
        prediction_6
    ) = base
    prediction_101 = prediction_102 = prediction_103 = prediction_104 = (
        prediction_105
    ) = prediction_106 = base
    prediction_201 = prediction_202 = prediction_203 = prediction_204 = (
        prediction_205
    ) = prediction_206 = base
    prediction_301 = prediction_302 = prediction_303 = prediction_304 = (
        prediction_305
    ) = prediction_306 = base
    prediction_401 = prediction_402 = prediction_403 = prediction_404 = (
        prediction_405
    ) = prediction_406 = base




## === cell 8
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
    p201,
    p202,
    p203,
    p204,
    p205,
    p206,
    p301,
    p302,
    p303,
    p304,
    p305,
    p306,
    p401,
    p402,
    p403,
    p404,
    p405,
    p406,
):
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    cases = [int(os.path.basename(p)) for p in path_cases]

    preds_list = [
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
        p201,
        p202,
        p203,
        p204,
        p205,
        p206,
        p301,
        p302,
        p303,
        p304,
        p305,
        p306,
        p401,
        p402,
        p403,
        p404,
        p405,
        p406,
    ]
    preds_list = [np.asarray(p, dtype=np.float32).reshape(-1) for p in preds_list]

    n = len(cases)
    for i, p in enumerate(preds_list):
        if len(p) != n:
            raise ValueError(
                f"Prediction vector {i} length {len(p)} does not match number of test cases {n}."
            )

    prediction = np.mean(np.vstack(preds_list), axis=0)
    prediction = np.clip(prediction, 0.0, 1.0)

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    df["BraTS21ID"] = df["BraTS21ID"].astype(int).astype(str).str.zfill(5)
    df = df.sort_values("BraTS21ID").reset_index(drop=True)
    return df




## === cell 9
sub_df = create_sub(
    test,
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
    prediction_201,
    prediction_202,
    prediction_203,
    prediction_204,
    prediction_205,
    prediction_206,
    prediction_301,
    prediction_302,
    prediction_303,
    prediction_304,
    prediction_305,
    prediction_306,
    prediction_401,
    prediction_402,
    prediction_403,
    prediction_404,
    prediction_405,
    prediction_406,
)

sub_df.head()



## === cell 10
try:
    import seaborn as sns

    sns.displot(sub_df.MGMT_value)
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 11
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
