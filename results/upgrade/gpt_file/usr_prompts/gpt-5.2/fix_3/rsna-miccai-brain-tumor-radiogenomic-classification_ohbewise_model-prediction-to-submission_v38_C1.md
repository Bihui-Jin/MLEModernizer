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

0.4792622811490736

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio

DATA_DIR = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TEST_DIR = os.path.join(DATA_DIR, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

print("TensorFlow:", tf.__version__)
print("TensorFlow-IO:", tfio.__version__)
print("Test dir exists:", os.path.isdir(TEST_DIR))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TARGET_SHAPE = (128, 128, 64)  # (H, W, D)


def _sorted_dicom_files(series_dir: str):
    files = [
        os.path.join(series_dir, f)
        for f in os.listdir(series_dir)
        if f.lower().endswith(".dcm")
    ]

    def _key(p):
        b = os.path.basename(p)
        nums = "".join([c if c.isdigit() else " " for c in b]).split()
        return int(nums[-1]) if nums else b

    files.sort(key=_key)
    return files


def load_dicom_volume(series_dir: str):
    """Loads a DICOM series directory into a float32 tensor (H, W, D)."""
    files = _sorted_dicom_files(series_dir)
    if len(files) == 0:
        raise FileNotFoundError(f"No DICOM files found in {series_dir}")

    slices = []
    for fp in files:
        bs = tf.io.read_file(fp)
        img = tfio.image.decode_dicom_image(
            bs, dtype=tf.float32
        )  # typically (1, H, W, 1)
        img = tf.squeeze(img)  # (H, W)
        slices.append(img)
    vol = tf.stack(slices, axis=0)  # (D, H, W)
    vol = tf.transpose(vol, perm=[1, 2, 0])  # (H, W, D)
    return vol


def normalize_volume(vol_hwd: tf.Tensor):
    vol = tf.cast(vol_hwd, tf.float32)
    mean = tf.reduce_mean(vol)
    std = tf.math.reduce_std(vol)
    vol = (vol - mean) / (std + 1e-6)
    vol = tf.clip_by_value(vol, -5.0, 5.0)
    return vol


def resize_volume_hwd(vol_hwd: tf.Tensor, target_shape=TARGET_SHAPE):
    """Resize (H, W, D) -> target_shape using tf.image.resize for H,W and linear resampling for D."""
    tgt_h, tgt_w, tgt_d = target_shape
    vol = tf.cast(vol_hwd, tf.float32)

    vol_dhw = tf.transpose(vol, [2, 0, 1])  # (D, H, W)
    vol_dhw = vol_dhw[..., tf.newaxis]  # (D, H, W, 1)
    vol_dhw = tf.image.resize(vol_dhw, (tgt_h, tgt_w), method="bilinear")
    vol = tf.squeeze(vol_dhw, axis=-1)  # (D, tgt_h, tgt_w)
    vol = tf.transpose(vol, [1, 2, 0])  # (tgt_h, tgt_w, D)

    vol_whd = tf.transpose(vol, [1, 0, 2])  # (W, H, D)
    vol_whd = vol_whd[..., tf.newaxis]  # (W, H, D, 1)
    vol_whd = tf.image.resize(vol_whd, (tgt_h, tgt_d), method="bilinear")
    vol_res = tf.squeeze(vol_whd, axis=-1)  # (W, tgt_h, tgt_d)
    vol_res = tf.transpose(vol_res, [1, 0, 2])  # (tgt_h, W, tgt_d)

    if vol_res.shape[1] != tgt_w:
        v = tf.transpose(vol_res, [2, 0, 1])[..., tf.newaxis]  # (D, H, W, 1)
        v = tf.image.resize(v, (tgt_h, tgt_w), method="bilinear")
        vol_res = tf.transpose(tf.squeeze(v, axis=-1), [1, 2, 0])

    vol_res.set_shape((tgt_h, tgt_w, tgt_d))
    return vol_res


def add_batch_channel(volume_hwd: tf.Tensor):
    """Add channel and batch dims: (H, W, D) -> (1, H, W, D, 1)."""
    v = volume_hwd[..., tf.newaxis]  # (H, W, D, 1)
    v = v[tf.newaxis, ...]  # (1, H, W, D, 1)
    return v


def process_series(series_dir: str):
    vol = load_dicom_volume(series_dir)
    vol = normalize_volume(vol)
    vol = resize_volume_hwd(vol, TARGET_SHAPE)
    vol = add_batch_channel(vol)
    return vol




## === cell 2
scan_type = "T1wCE"
MODEL_DIR = f"../input/dataset-to-model-with-tensorflow/models/{scan_type}/"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_ids = sample_sub["BraTS21ID"].astype(str).str.zfill(5).tolist()

print("N test IDs (from sample_submission):", len(test_ids))
print("Loading model from:", MODEL_DIR)


def _find_savedmodel_dir(base_dir: str) -> str:
    base_dir = os.path.abspath(base_dir)
    if os.path.exists(os.path.join(base_dir, "saved_model.pb")):
        return base_dir
    if os.path.isdir(base_dir):
        for name in sorted(os.listdir(base_dir)):
            cand = os.path.join(base_dir, name)
            if os.path.isdir(cand) and os.path.exists(
                os.path.join(cand, "saved_model.pb")
            ):
                return cand
    return base_dir  # fallback (will error later with clear message)


def _pick_endpoint(savedmodel_dir: str) -> str:
    sm = tf.saved_model.load(savedmodel_dir)
    sigs = list(getattr(sm, "signatures", {}).keys())
    if "serving_default" in sigs:
        return "serving_default"
    if len(sigs) > 0:
        return sigs[0]
    return "serving_default"


savedmodel_dir = _find_savedmodel_dir(MODEL_DIR)
endpoint = _pick_endpoint(savedmodel_dir)
print("Resolved SavedModel dir:", savedmodel_dir)
print("Using call_endpoint:", endpoint)

inference_layer = tf.keras.layers.TFSMLayer(savedmodel_dir, call_endpoint=endpoint)
model = tf.keras.Model(
    inputs=tf.keras.Input(shape=(TARGET_SHAPE[0], TARGET_SHAPE[1], TARGET_SHAPE[2], 1)),
    outputs=inference_layer,
)

preds = []
missing = 0

for pid in test_ids:
    series_dir = os.path.join(TEST_DIR, pid, scan_type)
    try:
        x = process_series(series_dir)
        y = model(x, training=False)
        if isinstance(y, dict):
            for k in ("outputs", "output_0", "predictions", "probabilities", "logits"):
                if k in y:
                    y = y[k]
                    break
            if isinstance(y, dict):
                y = list(y.values())[0]
        p = float(tf.reshape(y, [-1])[0].numpy())
        if p < 0.0 or p > 1.0:
            p = float(tf.math.sigmoid(p).numpy())
    except Exception:
        missing += 1
        p = 0.5
    preds.append(p)

sub = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": preds})
sub.to_csv("/kaggle/working/submission.csv", index=False)

print("Wrote /kaggle/working/submission.csv")
print("Missing/failed cases filled with 0.5:", missing)
print(sub.head())
print("Submission rows:", len(sub), "Expected:", len(sample_sub))
assert len(sub) == len(sample_sub)
assert list(sub.columns) == ["BraTS21ID", "MGMT_value"]

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/2403011831.py in <cell line: 0>()
     41 
     42 savedmodel_dir = _find_savedmodel_dir(MODEL_DIR)
---> 43 endpoint = _pick_endpoint(savedmodel_dir)
     44 print("Resolved SavedModel dir:", savedmodel_dir)
     45 print("Using call_endpoint:", endpoint)

/tmp/ipykernel_55/2403011831.py in _pick_endpoint(savedmodel_dir)
     28 
     29 def _pick_endpoint(savedmodel_dir: str) -> str:
---> 30     sm = tf.saved_model.load(savedmodel_dir)
     31     sigs = list(getattr(sm, "signatures", {}).keys())
     32     # Prefer standard serving signature if present

/usr/local/lib/python3.11/dist-packages/tensorflow/python/saved_model/load.py in load(export_dir, tags, options)
    910   if isinstance(export_dir, os.PathLike):
    911     export_dir = os.fspath(export_dir)
--> 912   result = load_partial(export_dir, None, tags, options)["root"]
    913   return result
    914 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/saved_model/load.py in load_partial(export_dir, filters, tags, options)
   1014     tags = nest.flatten(tags)
   1015   saved_model_proto, debug_info = (
-> 1016       loader_impl.parse_saved_model_with_debug_info(export_dir))
   1017 
   1018   loader = None

/usr/local/lib/python3.11/dist-packages/tensorflow/python/saved_model/loader_impl.py in parse_saved_model_with_debug_info(export_dir)
     57     parsed. Missing graph debug info file is fine.
     58   """
---> 59   saved_model = parse_saved_model(export_dir)
     60 
     61   debug_info_path = file_io.join(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/saved_model/loader_impl.py in parse_saved_model(export_dir)
    117       raise IOError(f"Cannot parse file {path_to_pbtxt}: {str(e)}.") from e
    118   else:
--> 119     raise IOError(
    120         f"SavedModel file does not exist at: {export_dir}{os.path.sep}"
    121         f"{{{constants.SAVED_MODEL_FILENAME_PBTXT}|"

OSError: SavedModel file does not exist at: /kaggle/input/dataset-to-model-with-tensorflow/models/T1wCE/{saved_model.pbtxt|saved_model.pb}
