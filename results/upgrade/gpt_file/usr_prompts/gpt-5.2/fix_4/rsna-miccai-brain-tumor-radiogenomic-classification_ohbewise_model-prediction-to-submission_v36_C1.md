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

nibabel==5.3.2
protobuf==6.33.0
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

- What this solution (achieved 0.5) has done: 'I remove the `torchio` dependency (it is not available in this environment) and replace its preprocessing with a minimal, deterministic DICOM→numpy pipeline using `tensorflow-io` that keeps the overall flow the same (load scan → preprocess → predict). I also fix the `Path`/file-writing issues by writing exactly one submission file after collecting predictions for all test patients, ensuring the row count matches `sample_submission.csv` and the IDs are in the correct format/order. To avoid repeated expensive model loads and potential inconsistencies, I load the TensorFlow model once per scan type outside the patient loop (same model/architecture, same predictions). Finally, I add safe fallbacks so the script always produces a valid `submission.csv` even if a case has irregular slices.'
- What this solution (achieved 0.5) has done: 'I fix the immediate runtime import crash caused by an incompatibility between `tensorflow-io` and `protobuf==6.x` by removing the `tensorflow_io` dependency and switching DICOM reading to a minimal, deterministic `pydicom`-based loader (pydicom is available on Kaggle). Next, I make the model-loading path robust by automatically locating a TensorFlow SavedModel/`.h5`/`.keras` under the provided model base directory, instead of hard-failing on a non-existent hardcoded subfolder. Finally, if no model artifact is found, the script still complete and write a valid `submission.csv` using safe default probabilities (score-neutral vs your current 0.5 baseline, but now guaranteed to run end-to-end).'
- What this solution (achieved 0.5) has done: 'The crash happens before any modeling because `pydicom` imports protobuf internals that are incompatible with `protobuf==6.33.0` in this Kaggle image, triggering the `MessageFactory.GetPrototype` error. I remove the `pydicom` dependency entirely and replace DICOM loading with a small, deterministic `nibabel`-based reader that is already installed and compatible. This keeps the same overall flow (load series → preprocess → predict) and should improve score beyond the 0.5 baseline when a model is found, while still guaranteeing a valid `submission.csv` even if some cases are irregular. I also add a tiny robustness tweak to sort slices by `InstanceNumber`/`ImagePositionPatient` when available (still deterministic) to avoid volume scrambling.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

import nibabel as nib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.random.set_seed(0)
np.random.seed(0)

print(
    "Using nibabel for DICOM loading (pydicom removed due to protobuf incompatibility)."
)
print("TensorFlow:", tf.__version__)
print("Nibabel:", nib.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_dir = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/"
test_dir = os.path.join(data_dir, "test")
sample_path = os.path.join(data_dir, "sample_submission.csv")

sample_sub = pd.read_csv(sample_path)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)
test_ids = sample_sub["BraTS21ID"].tolist()

print("Sample submission rows:", len(sample_sub))
print("First IDs:", test_ids[:5])




## === cell 2
def _list_dcm_files(case_dir):
    if not os.path.isdir(case_dir):
        return []
    files = [
        os.path.join(case_dir, f)
        for f in os.listdir(case_dir)
        if f.lower().endswith(".dcm")
    ]
    files.sort()
    return files


def read_dicom_series(case_dir):
    """
    Read a DICOM series (folder of .dcm files) into a 3D numpy volume: (H, W, D).
    Deterministic ordering:
      - Prefer sorting by (InstanceNumber) or (ImagePositionPatient[2]) when available,
        otherwise fallback to filename sort.
    Uses nibabel's DICOM reader to avoid pydicom/protobuf incompatibility in this environment.
    """
    dcm_files = _list_dcm_files(case_dir)
    if len(dcm_files) == 0:
        raise FileNotFoundError(f"No DICOM files found in {case_dir}")

    slices = []
    sort_keys = []
    for fp in dcm_files:
        img = nib.load(fp)
        arr = np.asanyarray(img.dataobj)

        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim != 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Unexpected DICOM array shape {arr.shape} for {fp}")

        key = None
        try:
            hdr = img.header
            inst = None
            ippz = None
            if hasattr(hdr, "get"):
                inst = hdr.get("InstanceNumber", None)
                ipp = hdr.get("ImagePositionPatient", None)
                if ipp is not None:
                    try:
                        ippz = float(ipp[2])
                    except Exception:
                        ippz = None
            if inst is not None:
                key = (0, int(inst))
            elif ippz is not None:
                key = (1, ippz)
        except Exception:
            key = None

        if key is None:
            key = (2, os.path.basename(fp))

        slices.append(arr.astype(np.float32))
        sort_keys.append(key)

    order = np.argsort(np.array(sort_keys, dtype=object))
    slices = [slices[i] for i in order]

    vol = np.stack(slices, axis=-1).astype(np.float32)  # [H, W, D]
    return vol


def preprocess_volume(vol, target_shape=(128, 128, 64)):
    """
    Minimal deterministic preprocessing:
    - center-crop or pad to target_shape
    - z-normalize (mean/std over all voxels)
    - rescale to [-1, 1] using robust percentiles
    Output: (1, 128, 128, 64, 1) float32
    """
    th, tw, td = target_shape
    h, w, d = vol.shape

    out = np.zeros((th, tw, td), dtype=np.float32)

    def _compute_src_dst(src_len, dst_len):
        if src_len >= dst_len:
            src_start = (src_len - dst_len) // 2
            src_end = src_start + dst_len
            dst_start, dst_end = 0, dst_len
        else:
            src_start, src_end = 0, src_len
            dst_start = (dst_len - src_len) // 2
            dst_end = dst_start + src_len
        return src_start, src_end, dst_start, dst_end

    hs, he, hd_s, hd_e = _compute_src_dst(h, th)
    ws, we, wd_s, wd_e = _compute_src_dst(w, tw)
    ds, de, dd_s, dd_e = _compute_src_dst(d, td)

    out[hd_s:hd_e, wd_s:wd_e, dd_s:dd_e] = vol[hs:he, ws:we, ds:de].astype(np.float32)

    mean = float(out.mean())
    std = float(out.std())
    if std < 1e-6:
        std = 1.0
    out = (out - mean) / std

    p1, p99 = np.percentile(out, [1, 99])
    if abs(p99 - p1) < 1e-6:
        out = np.clip(out, -3, 3) / 3.0
    else:
        out = (out - p1) / (p99 - p1)  # ~[0,1]
        out = out * 2.0 - 1.0  # ~[-1,1]
        out = np.clip(out, -1.0, 1.0)

    out = out[..., None]  # (H,W,D,1)
    out = out[None, ...]  # (1,H,W,D,1)
    return out.astype(np.float32)


def _find_model_artifact(model_base_dir, scan_type):
    """
    Search common TF model formats:
    - SavedModel directory containing saved_model.pb
    - .keras / .h5 file
    Preference: scan_type-named artifacts first, then any model under base dir.
    """
    candidates = []

    st_dir = os.path.join(model_base_dir, scan_type)
    if os.path.isdir(st_dir):
        candidates.append(st_dir)

    for ext in (".keras", ".h5", ".hdf5"):
        fp = os.path.join(model_base_dir, f"{scan_type}{ext}")
        if os.path.isfile(fp):
            candidates.append(fp)

    if os.path.isdir(model_base_dir):
        for name in os.listdir(model_base_dir):
            p = os.path.join(model_base_dir, name)
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "saved_model.pb")):
                candidates.append(p)
            elif os.path.isfile(p) and os.path.splitext(p)[1].lower() in (
                ".keras",
                ".h5",
                ".hdf5",
            ):
                candidates.append(p)
        for root, dirs, files in os.walk(model_base_dir):
            if "saved_model.pb" in files:
                candidates.append(root)
            for f in files:
                if os.path.splitext(f)[1].lower() in (".keras", ".h5", ".hdf5"):
                    candidates.append(os.path.join(root, f))

    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)

    for c in uniq:
        base = os.path.basename(c)
        if base == scan_type:
            return c
        if os.path.isfile(c) and os.path.splitext(os.path.basename(c))[0] == scan_type:
            return c

    return uniq[0] if uniq else None




## === cell 3
scan_types = ["T1w"]

model_base_dir = "/kaggle/input/dataset-to-model-with-tensorflow/models"
preds = {sid: 0.5 for sid in test_ids}  # safe default (baseline)

for scan_type in scan_types:
    model_artifact = _find_model_artifact(model_base_dir, scan_type)

    if model_artifact is None:
        print(
            f"WARNING: No model artifact found under {model_base_dir}. Using default 0.5 predictions."
        )
        model = None
    else:
        model = tf.keras.models.load_model(model_artifact, compile=False)
        print(f"Loaded model for {scan_type} from {model_artifact}")

    if model is None:
        continue

    for sid in test_ids:
        case_dir = os.path.join(test_dir, sid, scan_type)
        try:
            vol = read_dicom_series(case_dir)
            x = preprocess_volume(vol, target_shape=(128, 128, 64))
            y = model.predict(x, verbose=0)
            p = float(np.asarray(y).reshape(-1)[0])
            p = max(0.0, min(1.0, p))
            preds[sid] = p
        except Exception:
            preds[sid] = 0.5

print("Predictions generated:", len(preds))




## === cell 4
submission = sample_sub.copy()
submission["MGMT_value"] = submission["BraTS21ID"].map(preds).astype(float)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Row count:", len(submission), " NaNs:", submission["MGMT_value"].isna().sum())
assert len(submission) == len(
    sample_sub
), "Submission row count mismatch vs sample_submission.csv"
assert submission["MGMT_value"].between(0, 1).all(), "Predictions must be within [0, 1]"
assert out_path.endswith(".csv")
