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

- What this solution (achieved 0.5) has done: 'I remove the protobuf environment override that is triggering the `MessageFactory.GetPrototype` crash with TF 2.18 + protobuf 6, so imports work normally. Then I make the external SavedModel dependency optional: if the `/kaggle/input/dataset-to-model-with-tensorflow/...` model folder doesn’t exist, the script fall back to training the same simple 3D CNN on-the-fly from the provided train DICOMs (excluding the known-bad cases) and use it for test inference. This fixes the missing-model runtime error and guarantees a valid `submission.csv` is produced end-to-end in the Kaggle environment. The training/inference pipeline is kept minimal and deterministic, and outputs are clipped to valid probabilities for the AUC metric.'
- What this solution (achieved 0.5) has done: 'I fix the two runtime blockers: (1) the protobuf crash happens because the env var is removed after importing TensorFlow, so I move that `os.environ.pop(...)` to run before any TF/Keras import; (2) TensorFlow-IO’s DICOM decoder is failing to load its shared library, so I remove the tfio dependency and replace DICOM reading with a lightweight `pydicom` reader (installed in Kaggle) while keeping the same preprocessing shapes and the same 3D CNN/training loop. These changes are execution/stability fixes and keep the model/training semantics the same, so score should be similar or slightly better than the all-0.5 fallback. The script still optionally load the external SavedModel if present, otherwise it trains locally and always writes `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.5) has done: 'I fix two runtime blockers: the protobuf `MessageFactory.GetPrototype` crash (caused by unsetting the protobuf implementation env var) and the invalid reshape coming from the custom “trilinear resize” function. The protobuf fix is score-neutral but required to even import TF reliably in this environment; the resize fix preserves the same preprocessing intent (resize to 128×128×64) but makes it correct and stable by using proper 3D resizing. With these changes, the local fallback training run end-to-end and produce a valid `/kaggle/working/submission.csv`; score should move above the constant-0.5 baseline because the model actually train instead of failing. I keep the model and training loop unchanged otherwise.'

# 9. Code solution

## === cell 0
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

print("TF:", tf.__version__)
print("Keras:", keras.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = f"{DATA_DIR}/train"
TEST_DIR = f"{DATA_DIR}/test"
LABELS_CSV = f"{DATA_DIR}/train_labels.csv"
SAMPLE_SUB = f"{DATA_DIR}/sample_submission.csv"

BAD_CASES = {"00109", "00123", "00709"}  # known broken cases per competition note



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import pydicom


def _sorted_dicom_files(series_dir: str):
    if not os.path.isdir(series_dir):
        return []
    files = []
    for fn in os.listdir(series_dir):
        if fn.lower().endswith(".dcm"):
            files.append(os.path.join(series_dir, fn))
    files.sort()
    return files


def read_dicom_series(series_dir: str):
    """
    Reads a DICOM series folder into a float32 tensor [H, W, D].
    Uses pydicom to avoid TF-IO shared library issues.
    """
    fns = _sorted_dicom_files(series_dir)
    if len(fns) == 0:
        raise FileNotFoundError(f"No DICOM files found in: {series_dir}")

    slices = []
    for fp in fns:
        ds = pydicom.dcmread(fp, force=True)
        arr = ds.pixel_array.astype(np.float32)  # [H, W]
        slices.append(arr)

    vol = np.stack(slices, axis=-1)  # [H, W, D]
    return tf.convert_to_tensor(vol, dtype=tf.float32)


def z_normalize(volume: tf.Tensor, eps: float = 1e-6):
    mean = tf.reduce_mean(volume)
    std = tf.math.reduce_std(volume)
    return (volume - mean) / (std + eps)


def resize_volume_trilinear(volume: tf.Tensor, target_hw=(128, 128), target_d=64):
    """
    BUGFIX (logic/runtime): Previous implementation used an incorrect reshape path that can
    fail when D is unknown, causing:
      Input to reshape is a tensor with 64 values, but the requested shape has 1048576
    Implement true 3D resize by treating depth as a spatial dimension and using tf.image.resize
    on a 5D tensor [N, D, H, W, C], which is supported for 3D resize in TF (resize last 3 dims).
    Input:  [H, W, D]
    Output: [Ht, Wt, Dt]
    """
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    target_d = int(target_d)

    vol = tf.convert_to_tensor(volume, dtype=tf.float32)  # [H,W,D]
    vol = tf.transpose(vol, [2, 0, 1])  # [D,H,W]
    vol = vol[tf.newaxis, ..., tf.newaxis]  # [1,D,H,W,1]

    vol = tf.image.resize(vol, size=(target_d, target_h, target_w), method="bilinear")

    vol = tf.squeeze(vol, axis=0)  # [target_d,target_h,target_w,1]
    vol = tf.squeeze(vol, axis=-1)  # [target_d,target_h,target_w]
    vol = tf.transpose(vol, [1, 2, 0])  # [target_h,target_w,target_d]
    return vol


def center_crop_or_pad_depth(volume: tf.Tensor, target_d: int = 64):
    """Ensure depth == target_d with symmetric crop/pad on the last axis."""
    d = tf.shape(volume)[-1]

    def _crop():
        start = (d - target_d) // 2
        return volume[..., start : start + target_d]

    def _pad():
        pad_total = target_d - d
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return tf.pad(volume, paddings=[[0, 0], [0, 0], [pad_before, pad_after]])

    return tf.cond(d >= target_d, _crop, _pad)


def process_scan_from_dicom(series_dir: str):
    """
    Creates model input tensor with shape [1, 128, 128, 64, 1]
    from a DICOM series directory.
    """
    vol = read_dicom_series(series_dir)  # [H, W, D]
    vol = z_normalize(vol)
    vol = resize_volume_trilinear(vol, target_hw=(128, 128), target_d=64)
    vol = center_crop_or_pad_depth(vol, target_d=64)
    vol = vol[..., tf.newaxis]  # [H, W, D, 1]
    vol = vol[tf.newaxis, ...]  # [1, H, W, D, 1]
    return vol


def _model_predict_single_prob(model, case_tensor: tf.Tensor) -> float:
    """
    SavedModel via TFSMLayer may return dict or tensor; handle both robustly.
    Returns scalar probability.
    """
    out = model(case_tensor, training=False)
    if isinstance(out, dict):
        for k in (
            "outputs",
            "predictions",
            "logits",
            "output_0",
            "dense",
            "activation",
        ):
            if k in out:
                out = out[k]
                break
        else:
            out = next(iter(out.values()))
    out = tf.convert_to_tensor(out)
    out = tf.reshape(out, [-1])
    return float(out[0].numpy())




## === cell 2
def build_3d_cnn(input_shape=(128, 128, 64, 1)):
    inputs = keras.Input(shape=input_shape, dtype=tf.float32, name="input")
    x = keras.layers.Conv3D(16, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPool3D(pool_size=2)(x)
    x = keras.layers.Conv3D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPool3D(pool_size=2)(x)
    x = keras.layers.Conv3D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling3D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def try_load_external_savedmodel(scan_type: str):
    candidates = [
        f"../input/dataset-to-model-with-tensorflow/models/{scan_type}/",
        f"/kaggle/input/dataset-to-model-with-tensorflow/models/{scan_type}/",
    ]
    for model_dir in candidates:
        sm_pb = os.path.join(model_dir, "saved_model.pb")
        sm_pbtxt = os.path.join(model_dir, "saved_model.pbtxt")
        if os.path.isdir(model_dir) and (
            os.path.exists(sm_pb) or os.path.exists(sm_pbtxt)
        ):
            endpoint = "serving_default"
            try:
                tfsml = keras.layers.TFSMLayer(model_dir, call_endpoint=endpoint)
            except Exception:
                loaded = tf.saved_model.load(model_dir)
                sigs = list(getattr(loaded, "signatures", {}).keys())
                if len(sigs) == 0:
                    continue
                endpoint = sigs[0]
                tfsml = keras.layers.TFSMLayer(model_dir, call_endpoint=endpoint)

            inp = keras.Input(shape=(128, 128, 64, 1), dtype=tf.float32, name="input")
            out = tfsml(inp)
            inference_model = keras.Model(inputs=inp, outputs=out)
            return inference_model, model_dir, endpoint
    return None, None, None




## === cell 3
def make_tf_dataset_from_ids(
    ids, labels_map, scan_type="T1w", batch_size=1, shuffle=False
):
    def _load_one(brats_id):
        brats_id = brats_id.numpy().decode("utf-8")
        series_dir = f"{TRAIN_DIR}/{brats_id}/{scan_type}/"
        x = process_scan_from_dicom(series_dir)  # [1,128,128,64,1]
        x = tf.squeeze(x, axis=0)  # [128,128,64,1]
        y = np.float32(labels_map[brats_id])
        return x, y

    def _tf_load_one(brats_id):
        x, y = tf.py_function(_load_one, [brats_id], Tout=(tf.float32, tf.float32))
        x.set_shape((128, 128, 64, 1))
        y.set_shape(())
        return x, y

    ds = tf.data.Dataset.from_tensor_slices(ids)
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(ids), 256), seed=SEED, reshuffle_each_iteration=True
        )
    ds = ds.map(_tf_load_one, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 4
labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
labels_df = labels_df[~labels_df["BraTS21ID"].isin(BAD_CASES)].reset_index(drop=True)

labels_map = dict(
    zip(labels_df["BraTS21ID"].tolist(), labels_df["MGMT_value"].astype(int).tolist())
)
all_ids = labels_df["BraTS21ID"].tolist()

rng = np.random.RandomState(SEED)
perm = rng.permutation(len(all_ids))
split = int(0.85 * len(all_ids))
train_ids = [all_ids[i] for i in perm[:split]]
val_ids = [all_ids[i] for i in perm[split:]]

print("Train/Val sizes:", len(train_ids), len(val_ids))



## === cell 5
scan_type = "T1w"

inference_model, model_dir, endpoint = try_load_external_savedmodel(scan_type)

if inference_model is not None:
    print(f"Loaded external SavedModel from: {model_dir}")
    print(f"Using SavedModel endpoint: {endpoint}")
else:
    print(
        "External SavedModel not found; training a small 3D CNN locally (minimal fallback)."
    )
    model = build_3d_cnn(input_shape=(128, 128, 64, 1))

    train_ds = make_tf_dataset_from_ids(
        train_ids, labels_map, scan_type=scan_type, batch_size=1, shuffle=True
    )
    val_ds = make_tf_dataset_from_ids(
        val_ids, labels_map, scan_type=scan_type, batch_size=1, shuffle=False
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
        verbose=2,
    )
    inference_model = model



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_55/452153114.py in <cell line: 0>()
     19     )
     20 
---> 21     history = model.fit(
     22         train_ds,
     23         validation_data=val_ds,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node EagerPyFunc defined at (most recent call last):
<stack traces unavailable>
Detected at node EagerPyFunc defined at (most recent call last):
<stack traces unavailable>
2 root error(s) found.
  (0) INVALID_ARGUMENT:  ValueError: 'images' must have either 3 or 4 dimensions.
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 267, in __call__
    return func(device, token, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 145, in __call__
    outputs = self._call(device, args)
              ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 152, in _call
    ret = self._func(*args)
          ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_55/3986340791.py", line 7, in _load_one
    x = process_scan_from_dicom(series_dir)  # [1,128,128,64,1]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_55/1013377414.py", line 90, in process_scan_from_dicom
    vol = resize_volume_trilinear(vol, target_hw=(128, 128), target_d=64)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_55/1013377414.py", line 58, in resize_volume_trilinear
    vol = tf.image.resize(vol, size=(target_d, target_h, target_w), method="bilinear")
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    raise e.with_traceback(filtered_tb) from None

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/image_ops_impl.py", line 1469, in _resize_images_common
    raise ValueError('\'images\' must have either 3 or 4 dimensions.')

ValueError: 'images' must have either 3 or 4 dimensions.


	 [[{{node EagerPyFunc}}]]
	 [[IteratorGetNext]]
	 [[IteratorGetNext/_2]]
  (1) INVALID_ARGUMENT:  ValueError: 'images' must have either 3 or 4 dimensions.
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 267, in __call__
    return func(device, token, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 145, in __call__
    outputs = self._call(device, args)
              ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 152, in _call
    ret = self._func(*args)
          ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_55/3986340791.py", line 7, in _load_one
    x = process_scan_from_dicom(series_dir)  # [1,128,128,64,1]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_55/1013377414.py", line 90, in process_scan_from_dicom
    vol = resize_volume_trilinear(vol, target_hw=(128, 128), target_d=64)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_55/1013377414.py", line 58, in resize_volume_trilinear
    vol = tf.image.resize(vol, size=(target_d, target_h, target_w), method="bilinear")
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    raise e.with_traceback(filtered_tb) from None

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/image_ops_impl.py", line 1469, in _resize_images_common
    raise ValueError('\'images\' must have either 3 or 4 dimensions.')

ValueError: 'images' must have either 3 or 4 dimensions.


	 [[{{node EagerPyFunc}}]]
	 [[IteratorGetNext]]
0 successful operations.
0 derived errors ignored. [Op:__inference_multi_step_on_iterator_2362]

## === cell 6
sub = pd.read_csv(SAMPLE_SUB)
sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)

preds = []
missing = 0

for brats_id in sub["BraTS21ID"].tolist():
    series_dir = f"{TEST_DIR}/{brats_id}/{scan_type}/"
    try:
        case = process_scan_from_dicom(series_dir)
        p = _model_predict_single_prob(inference_model, case)
        if not np.isfinite(p):
            p = 0.5
        p = float(np.clip(p, 0.0, 1.0))
    except Exception:
        missing += 1
        p = 0.5
    preds.append(p)

sub["MGMT_value"] = preds

out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)

print(f"Wrote {out_path} with {len(sub)} rows. Fallback(0.5) used for {missing} cases.")
print(sub.head())
print("Submission columns:", list(sub.columns))
print("MGMT_value min/max:", float(np.min(preds)), float(np.max(preds)))
