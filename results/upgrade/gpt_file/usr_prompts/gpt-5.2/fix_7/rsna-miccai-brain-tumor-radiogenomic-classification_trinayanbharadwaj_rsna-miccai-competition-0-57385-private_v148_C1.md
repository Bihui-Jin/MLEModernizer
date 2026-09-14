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

0.34588

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.65882) has done: 'I fix the early runtime crash caused by an incompatible protobuf/pydicom interaction by importing TensorFlow first and forcing the pure-Python protobuf implementation before pydicom is imported. Then I fix the core logic bug that causes a row-count mismatch: your loader aggregates images across all patients, so predictions are per-image while the submission must be per-patient; I minimally change the loader to return exactly one stacked 6-slice T2W tensor per case (or a safe fallback), preserving the “6 slices per case” idea and the same 6-model averaging. Finally, I make `create_sub` align predictions to the exact folder order and ensure the output has the exact number of test IDs from `sample_submission.csv`, writing a valid `submission.csv`.'
- What this solution (achieved 0.37059) has done: 'I fix the immediate runtime crash by ensuring TensorFlow is imported before anything that might indirectly trigger protobuf/pydicom interactions, and by setting the protobuf env var *before* importing TensorFlow (so it actually takes effect). I also make the pydicom import robust: if pydicom still fails to import in this environment, the code fall back to producing a valid submission using the sample IDs with 0.5 predictions rather than crashing. These changes keep your existing core logic (same loader idea, same 6-model averaging, same submission alignment) and are score-neutral except in the rare case pydicom cannot be used (then it at least produce a valid CSV).'
- What this solution (achieved 0.63765) has done: 'I fix the crash happening before any model/data code runs by addressing the protobuf/TensorFlow incompatibility that causes `MessageFactory` to miss `GetPrototype` in this Kaggle image. Concretely, I force the pure-Python protobuf implementation and also set the TensorFlow flag that disables C++ protobuf parsing before importing TensorFlow, plus add a safe fallback that keeps the notebook running and still writes a valid `submission.csv` if TensorFlow cannot be imported. These changes are execution/stability fixes and do not change your core inference logic (6-model averaging, 6-slice T2W tensor per case, and sample_submission alignment). The rest of the pipeline remains the same and produce a valid submission CSV.'
- What this solution (achieved 0.64588) has done: 'I fix the immediate TensorFlow import crash by forcing TensorFlow to use the pure-Python protobuf implementation in a way that actually takes effect, and by applying a small compatibility shim for the `MessageFactory.GetPrototype` attribute that is missing in some protobuf builds in this Kaggle image. This keeps your existing inference logic intact (same 6-model averaging, same per-case 6-slice T2W tensor construction, same sample_submission alignment), but makes the notebook run end-to-end reliably. I also make the DICOM loader’s “T2w folder selection” robust by selecting the `T2w` directory by name (instead of assuming it is always index 3), which is a minimal correctness fix that can improve score if folder ordering differs. Finally, I ensure the submission is always written as `submission.csv` with the exact required columns and row count.'
- What this solution (achieved 0.34588) has done: 'I fix the crash happening at import time by applying the protobuf `MessageFactory.GetPrototype` compatibility shim in a way that actually affects TensorFlow’s already-imported `google.protobuf` module (the current shim targets a different class and still triggers the AttributeError). I keep your inference logic identical (same 6-slice per case loader, same 6-model averaging, same sample_submission alignment), only making the import/shim ordering robust so the pipeline runs end-to-end reliably. I also add a safe guard so if TensorFlow still fails, the code cleanly falls back to 0.5 predictions and still writes a valid `submission.csv` with the correct row count/columns. These changes should be score-neutral when TF works, while preventing the runtime error that currently blocks execution.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_PROTOBUF_DISABLE_CPP_IMPLEMENTATION"] = "1"

import numpy as np
import pandas as pd


def _apply_protobuf_getprototype_shim():
    try:
        import google.protobuf.message_factory as _mf

        MF = getattr(_mf, "MessageFactory", None)
        if MF is None:
            return

        if (not hasattr(MF, "GetPrototype")) and hasattr(MF, "GetMessageClass"):

            def _GetPrototype(self, descriptor):
                return self.GetMessageClass(descriptor)

            setattr(MF, "GetPrototype", _GetPrototype)
    except Exception:
        pass


_apply_protobuf_getprototype_shim()

TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow import keras

    _apply_protobuf_getprototype_shim()
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    keras = None
    print(
        "TensorFlow import failed; will fall back to constant predictions. Error:",
        repr(e),
    )

DICOM_AVAILABLE = True
try:
    import pydicom as dicom
except Exception:
    DICOM_AVAILABLE = False
    dicom = None

from skimage.transform import resize  # ensure resize is defined

try:
    import seaborn as sns
except Exception:
    sns = None

if TF_AVAILABLE:
    tf.random.set_seed(42)
np.random.seed(42)

print("TF_AVAILABLE:", TF_AVAILABLE, "DICOM_AVAILABLE:", DICOM_AVAILABLE)
if TF_AVAILABLE:
    print("TensorFlow:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _build_fallback_model(input_shape=(150, 150, 6)):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0)(inputs)
    x = keras.layers.Conv2D(8, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    outputs = keras.layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def _safe_load_model(path):
    if os.path.exists(path):
        try:
            return keras.models.load_model(path, compile=False)
        except Exception:
            return None
    return None


model_paths = [
    "../input/trained-model-for-rsnamiccai/rsna_miccai_114_epochs_T2W_7k_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_200_epochs_T2W_7k_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_83_b600_T2W_7k_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_28_b50_T2W_7k_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_10_b600_T2w_7k_0.62auc_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_15_b600_T2w_7k_0.74auc_imgs.h5",
]

if TF_AVAILABLE:
    loaded_models = []
    for p in model_paths:
        m = _safe_load_model(p)
        loaded_models.append(m)

    for i, m in enumerate(loaded_models):
        if m is None:
            loaded_models[i] = _build_fallback_model()

    model_T2, model_T2_2, model_T2_3, model_T2_4, model_T2_5, model_T2_6 = loaded_models
else:
    model_T2 = model_T2_2 = model_T2_3 = model_T2_4 = model_T2_5 = model_T2_6 = None




## === cell 2
def _read_and_preprocess_dcm(dcm_path, img_px_size=150):
    if not DICOM_AVAILABLE:
        return None

    try:
        img = dicom.dcmread(dcm_path)
        px = img.pixel_array
    except Exception:
        return None

    if px is None:
        return None

    if float(np.sum(px)) <= 100000:
        return None

    resized_img = resize(
        px,
        (img_px_size, img_px_size),
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)

    mx = float(np.max(resized_img))
    if mx > 0:
        resized_img = resized_img / mx

    if float(np.sum(resized_img)) <= 2000:
        return None

    return resized_img


def _find_t2w_folder(case_path):
    subdirs = [f.path for f in os.scandir(case_path) if f.is_dir()]
    if not subdirs:
        return None
    by_name = {os.path.basename(p).lower(): p for p in subdirs}
    if "t2w" in by_name:
        return by_name["t2w"]
    for p in subdirs:
        if "t2" in os.path.basename(p).lower():
            return p
    subdirs_sorted = sorted(subdirs)
    return subdirs_sorted[3] if len(subdirs_sorted) >= 4 else None


def load_test_T2W_images(path_test, img_px_size=150, n_slices=6):
    """
    Return one tensor per case.
    Each case becomes a single (150,150,6) array by stacking up to 6 informative T2 slices.
    """
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    X = []
    used_cases = []

    for case_path in path_cases:
        t2_path = _find_t2w_folder(case_path)

        if t2_path is None or (not os.path.isdir(t2_path)):
            X.append(np.zeros((img_px_size, img_px_size, n_slices), dtype=np.float32))
            used_cases.append(os.path.basename(case_path))
            continue

        img_paths = sorted([f.path for f in os.scandir(t2_path) if f.is_file()])

        slices = []
        for p in img_paths:
            sl = _read_and_preprocess_dcm(p, img_px_size=img_px_size)
            if sl is None:
                continue
            slices.append(sl)
            if len(slices) == n_slices:
                break

        if len(slices) == 0:
            stacked = np.zeros((img_px_size, img_px_size, n_slices), dtype=np.float32)
        else:
            while len(slices) < n_slices:
                slices.append(slices[-1])
            stacked = np.stack(slices[:n_slices], axis=-1).astype(np.float32)

            mx = float(np.max(stacked))
            if mx > 0:
                stacked = stacked / mx

        X.append(stacked)
        used_cases.append(os.path.basename(case_path))

    X = np.asarray(X, dtype=np.float32)
    print("Loaded cases:", len(X), "X shape:", X.shape)
    return X, used_cases




## === cell 3
test = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"

if DICOM_AVAILABLE:
    X_test, test_case_ids = load_test_T2W_images(test)
else:
    X_test, test_case_ids = np.empty((0, 150, 150, 6), dtype=np.float32), []




## === cell 4
def _predict_class1(model, x):
    if (x is None) or (len(x) == 0) or (model is None) or (not TF_AVAILABLE):
        return np.array([], dtype=np.float32)
    preds = model.predict(x, verbose=0)
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[1] >= 2:
        return preds[:, 1].astype(np.float32)
    return preds.reshape(-1).astype(np.float32)


prediction_1 = _predict_class1(model_T2, X_test)
prediction_101 = _predict_class1(model_T2_2, X_test)
prediction_201 = _predict_class1(model_T2_3, X_test)
prediction_301 = _predict_class1(model_T2_4, X_test)
prediction_401 = _predict_class1(model_T2_5, X_test)
prediction_501 = _predict_class1(model_T2_6, X_test)




## === cell 5
def create_sub_from_case_preds(case_ids, *pred_arrays):
    """
    Create exactly one prediction per case_id by averaging the provided model predictions.
    """
    preds_list = [np.asarray(p, dtype=np.float32) for p in pred_arrays]
    n = len(case_ids)

    min_len = min([len(p) for p in preds_list]) if preds_list else 0

    final_preds = []
    for i in range(n):
        if i < min_len and len(preds_list) > 0:
            s = 0.0
            for p in preds_list:
                s += float(p[i])
            pred = s / float(len(preds_list))
        else:
            pred = 0.5
        final_preds.append(float(np.clip(pred, 0.0, 1.0)))

    df = pd.DataFrame({"BraTS21ID": case_ids, "MGMT_value": final_preds})
    return df


sub_df = create_sub_from_case_preds(
    test_case_ids,
    prediction_1,
    prediction_101,
    prediction_201,
    prediction_301,
    prediction_401,
    prediction_501,
)

print(sub_df.head())
print("Raw sub_df shape:", sub_df.shape)



## === cell 6
sample_path = (
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
sample = pd.read_csv(sample_path)

if len(sub_df) > 0:
    sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
else:
    sub_df = pd.DataFrame({"BraTS21ID": [], "MGMT_value": []})

sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)

merged = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
merged["MGMT_value"] = (
    merged["MGMT_value"].astype(np.float32).fillna(0.5).clip(0.0, 1.0)
)

sub_df = merged
print("Aligned to sample_submission:", sub_df.shape)
print(sub_df.head())



## === cell 7
if sns is not None:
    try:
        sns.displot(sub_df["MGMT_value"])
    except Exception:
        pass



## === cell 8
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
print("Columns:", list(sub_df.columns))
