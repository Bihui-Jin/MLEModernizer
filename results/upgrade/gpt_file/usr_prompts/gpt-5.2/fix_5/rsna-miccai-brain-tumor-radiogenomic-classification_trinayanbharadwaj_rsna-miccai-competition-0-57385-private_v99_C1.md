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

0.30941

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.55294) has done: 'I remove/avoid the imports that crash the environment (`pympler`, `sklearn`, `cv2`, `skimage`) and replace `skimage.transform.resize` with a small, reliable PIL-based resize so image loading works. Since the referenced external pretrained `.h5` model is not available, I keep the same “predict from T2W slices then average” core inference semantics but build a tiny Keras CNN and train it quickly on the provided training set’s T2W slices, then run test inference. I also fix the submission creation bug (it was assigning an entire vector to every row) by averaging the six per-case predictions elementwise and aligning IDs with `sample_submission.csv`. Finally, I ensure `submission.csv` is always written with the required columns and formatting.'
- What this solution (achieved 0.38471) has done: 'The crash happens before any modeling due to a known protobuf incompatibility triggered during `pydicom` import/use in some Kaggle images; switching to a DICOM reader that doesn’t depend on that protobuf path (SimpleITK) fixes execution while keeping the same “load T2w slices → resize → tiny CNN → per-slice predict → per-case mean” core logic. I add a robust DICOM read function that tries SimpleITK first and only falls back to pydicom if available, and ensure deterministic behavior stays the same. I also make sure IDs are consistently handled as ints during loading and zero-padded strings only in the final submission, to avoid any silent misalignment. No changes are made to the model architecture, training loop, or aggregation semantics beyond the IO fix needed to run end-to-end and produce `submission.csv`.'
- What this solution (achieved 0.53882) has done: 'The crash happens at import time before any training because `SimpleITK` triggers a protobuf incompatibility (`MessageFactory.GetPrototype`) in this Kaggle image; to unblock execution while preserving the exact “load T2w DICOM slices → resize → tiny CNN → mean over slices” logic, I remove the `SimpleITK` import path and use a safe local `pydicom` import inside the DICOM reader. To keep the pipeline robust, the DICOM loader handle MONOCHROME inversion and missing/failed reads by falling back to zero slices exactly as your code already expects. I also keep ID typing consistent (ints internally, zero-padded strings only for submission) and ensure `submission.csv` is always written with the required columns and row alignment to `sample_submission.csv`. No changes are made to the model architecture, training loop, loss, or aggregation semantics.'
- What this solution (achieved 0.30941) has done: 'The import-time protobuf crash is happening before any of your code executes because `tensorflow` pulls in a protobuf build that’s incompatible with the environment; to fix this cleanly, I set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow so it uses the pure-Python protobuf implementation. I also make the DICOM reader more deterministic by sorting slices numerically by the `Image-XXX.dcm` index (instead of lexicographic sorting), which is a small correctness fix that tends to improve slice consistency without changing the overall “load T2w → resize → tiny CNN → mean over slices” approach. Finally, I keep the submission creation exactly aligned to `sample_submission.csv` and always write `submission.csv` with the required columns and zero-padded IDs.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import re
import numpy as np
import pandas as pd

from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def pil_resize(arr2d: np.ndarray, out_hw=(150, 150)) -> np.ndarray:
    """Replacement for skimage.transform.resize using PIL (no extra dependency)."""
    arr = arr2d.astype(np.float32)
    amin, amax = float(np.min(arr)), float(np.max(arr))
    if amax > amin:
        arr01 = (arr - amin) / (amax - amin)
    else:
        arr01 = np.zeros_like(arr, dtype=np.float32)

    img = Image.fromarray((arr01 * 255.0).astype(np.uint8), mode="L")
    img = img.resize((out_hw[1], out_hw[0]), resample=Image.BILINEAR)
    out = np.asarray(img).astype(np.float32) / 255.0
    return out


def safe_dcm_pixel_array(dcm_path: str):
    """
    Read DICOM safely; return None on failures.

    Use pydicom via a local import and handle common DICOM nuances (MONOCHROME1 inversion).
    """
    try:
        import pydicom

        d = pydicom.dcmread(dcm_path, force=True)
        arr = d.pixel_array

        try:
            if getattr(d, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
                arr = np.max(arr) - arr
        except Exception:
            pass

        return arr
    except Exception:
        return None




## === cell 2
def _get_modality_dir(case_dir: str, modality: str = "T2w") -> str:
    mod_dir = os.path.join(case_dir, modality)
    if os.path.isdir(mod_dir):
        return mod_dir
    for name in os.listdir(case_dir):
        if name.lower() == modality.lower() and os.path.isdir(
            os.path.join(case_dir, name)
        ):
            return os.path.join(case_dir, name)
    raise FileNotFoundError(f"Modality folder {modality} not found under {case_dir}")


_dcm_num_re = re.compile(r"(\d+)")


def _numeric_dcm_sort_key(path: str) -> int:
    base = os.path.basename(path)
    m = _dcm_num_re.findall(base)
    return int(m[-1]) if m else 0


def load_case_T2W_slices(case_dir: str, n_slices: int = 6, img_px_size: int = 150):
    """
    Load up to n_slices informative T2W slices from a case folder.
    Returns: list length n_slices of (H,W,3) float32 in [0,1]. If not enough, pads with last/zeros.
    """
    t2_dir = _get_modality_dir(case_dir, "T2w")
    dcm_files = [
        os.path.join(t2_dir, f)
        for f in os.listdir(t2_dir)
        if f.lower().endswith(".dcm")
    ]
    dcm_files = sorted(dcm_files, key=_numeric_dcm_sort_key)

    if not dcm_files:
        z = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
        return [z for _ in range(n_slices)]

    chosen = []
    for fp in dcm_files:
        px = safe_dcm_pixel_array(fp)
        if px is None:
            continue
        try:
            if float(np.sum(px)) <= 200000:
                continue
            img2d = pil_resize(px, out_hw=(img_px_size, img_px_size))
            stacked = np.stack([img2d, img2d, img2d], axis=-1).astype(np.float32)
            if float(np.sum(stacked)) > 3000:
                chosen.append(stacked)
                if len(chosen) >= n_slices:
                    break
        except Exception:
            continue

    if len(chosen) == 0:
        z = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
        chosen = [z]

    while len(chosen) < n_slices:
        chosen.append(chosen[-1].copy())

    return chosen[:n_slices]




## === cell 3
def load_dataset_slices(
    root_dir: str, ids, labels_df=None, n_slices: int = 6, img_px_size: int = 150
):
    """
    Creates slice-level dataset:
      X: (num_cases*n_slices, H, W, 3)
      y: (num_cases*n_slices,) if labels_df provided else None
      groups: (num_cases*n_slices,) case id for aggregation
    """
    X_list, y_list, g_list = [], [], []
    for brats_id in ids:
        brats_id_int = int(brats_id)
        case_dir = os.path.join(root_dir, f"{brats_id_int:05d}")
        slices = load_case_T2W_slices(
            case_dir, n_slices=n_slices, img_px_size=img_px_size
        )
        X_list.extend(slices)
        g_list.extend([brats_id_int] * n_slices)
        if labels_df is not None:
            val = float(labels_df.loc[brats_id_int, "MGMT_value"])
            y_list.extend([val] * n_slices)

    X = np.stack(X_list).astype(np.float32)
    if labels_df is not None:
        y = np.array(y_list, dtype=np.float32)
    else:
        y = None
    groups = np.array(g_list, dtype=np.int32)
    return X, y, groups




## === cell 4
BASE = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")
LABELS_CSV = os.path.join(BASE, "train_labels.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")

labels = pd.read_csv(LABELS_CSV)
labels["BraTS21ID"] = labels["BraTS21ID"].astype(int)
labels = labels.set_index("BraTS21ID")

bad_cases = {109, 123, 709}
train_ids = [int(i) for i in labels.index.values.tolist() if int(i) not in bad_cases]

sample_sub = pd.read_csv(SAMPLE_SUB)
test_ids = sample_sub["BraTS21ID"].astype(int).tolist()

len(train_ids), len(test_ids), sample_sub.head()



## === cell 5
rng = np.random.RandomState(SEED)
perm = rng.permutation(len(train_ids))
split = int(0.85 * len(train_ids))
tr_ids = [train_ids[i] for i in perm[:split]]
va_ids = [train_ids[i] for i in perm[split:]]

IMG_PX_SIZE = 150
N_SLICES = 6

X_tr, y_tr, g_tr = load_dataset_slices(
    TRAIN_DIR, tr_ids, labels_df=labels, n_slices=N_SLICES, img_px_size=IMG_PX_SIZE
)
X_va, y_va, g_va = load_dataset_slices(
    TRAIN_DIR, va_ids, labels_df=labels, n_slices=N_SLICES, img_px_size=IMG_PX_SIZE
)

X_tr.shape, X_va.shape, y_tr.mean(), y_va.mean()




## === cell 6
def build_model(input_shape=(150, 150, 3)):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


model_T2 = build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3))

BATCH = 32
EPOCHS = 2
history = model_T2.fit(
    X_tr, y_tr, validation_data=(X_va, y_va), epochs=EPOCHS, batch_size=BATCH, verbose=2
)




## === cell 7
def predict_cases_from_slices(
    model,
    root_dir: str,
    ids,
    n_slices: int = 6,
    img_px_size: int = 150,
    batch_size: int = 32,
):
    """
    For each case: load n_slices, predict per-slice probabilities, return mean per case.
    """
    case_preds = []
    for brats_id in ids:
        brats_id_int = int(brats_id)
        case_dir = os.path.join(root_dir, f"{brats_id_int:05d}")
        slices = load_case_T2W_slices(
            case_dir, n_slices=n_slices, img_px_size=img_px_size
        )
        X = np.stack(slices).astype(np.float32)
        p = model.predict(X, batch_size=batch_size, verbose=0).reshape(-1)
        case_preds.append(float(np.mean(p)))
    return np.array(case_preds, dtype=np.float32)


test_case_preds = predict_cases_from_slices(
    model_T2,
    TEST_DIR,
    test_ids,
    n_slices=N_SLICES,
    img_px_size=IMG_PX_SIZE,
    batch_size=BATCH,
)

test_case_preds[:10], test_case_preds.min(), test_case_preds.max()




## === cell 8
def create_sub_from_sample(
    sample_sub_df: pd.DataFrame, preds: np.ndarray
) -> pd.DataFrame:
    out = sample_sub_df.copy()
    if len(preds) != len(out):
        raise ValueError(
            f"Prediction length {len(preds)} != sample_submission length {len(out)}"
        )
    out["MGMT_value"] = preds.astype(float)
    out["BraTS21ID"] = out["BraTS21ID"].astype(str).str.zfill(5)
    return out[["BraTS21ID", "MGMT_value"]]


sub_df = create_sub_from_sample(sample_sub, test_case_preds)
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"Wrote {sub_path} with shape={sub_df.shape} and columns={list(sub_df.columns)}")
print(sub_df.head())
