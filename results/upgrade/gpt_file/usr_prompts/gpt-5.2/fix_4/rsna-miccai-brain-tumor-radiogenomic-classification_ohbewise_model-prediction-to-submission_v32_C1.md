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

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I remove the protobuf environment override that is triggering the `MessageFactory.GetPrototype` crash with TF 2.18 + protobuf 6, so imports work normally. Then I make the external SavedModel dependency optional: if the `/kaggle/input/dataset-to-model-with-tensorflow/...` model folder doesn’t exist, the script fall back to training the same simple 3D CNN on-the-fly from the provided train DICOMs (excluding the known-bad cases) and use it for test inference. This fixes the missing-model runtime error and guarantees a valid `submission.csv` is produced end-to-end in the Kaggle environment. The training/inference pipeline is kept minimal and deterministic, and outputs are clipped to valid probabilities for the AUC metric.'

# 9. Code solution

## === cell 0
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import keras

print("TF:", tf.__version__)
print("Keras:", keras.__version__)
print("TF-IO:", tfio.__version__)

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
    Uses tensorflow-io (available) so we don't need pydicom/torchio.
    """
    fns = _sorted_dicom_files(series_dir)
    if len(fns) == 0:
        raise FileNotFoundError(f"No DICOM files found in: {series_dir}")

    slices = []
    for fp in fns:
        raw = tf.io.read_file(fp)
        img = tfio.image.decode_dicom_image(
            raw,
            dtype=tf.uint16,
            color_dim=False,
            scale="auto",
        )
        img = tf.cast(img, tf.float32)
        img = tf.squeeze(img)  # [H, W]
        slices.append(img)

    vol = tf.stack(slices, axis=-1)  # [H, W, D]
    return vol


def z_normalize(volume: tf.Tensor, eps: float = 1e-6):
    mean = tf.reduce_mean(volume)
    std = tf.math.reduce_std(volume)
    return (volume - mean) / (std + eps)


def resize_volume_trilinear(volume: tf.Tensor, target_hw=(128, 128), target_d=64):
    """
    Resize a [H,W,D] volume to [target_h, target_w, target_d] using trilinear interpolation
    implemented via two bilinear resizes (HW then D).
    """
    vol = tf.transpose(volume, [2, 0, 1])  # [D, H, W]
    vol = vol[..., tf.newaxis]  # [D, H, W, 1]
    vol = tf.image.resize(vol, target_hw, method="bilinear")  # [D, Ht, Wt, 1]
    vol = tf.squeeze(vol, axis=-1)  # [D, Ht, Wt]

    vol = tf.transpose(vol, [1, 2, 0])  # [Ht, Wt, D]
    vol = tf.reshape(vol, [-1, tf.shape(vol)[-1], 1])  # [Ht*Wt, D, 1]
    vol = tf.image.resize(vol, [target_d, 1], method="bilinear")  # [Ht*Wt, target_d, 1]
    vol = tf.squeeze(vol, axis=-1)  # [Ht*Wt, target_d]
    vol = tf.reshape(vol, [target_hw[0], target_hw[1], target_d])  # [Ht, Wt, target_d]
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
UnimplementedError                        Traceback (most recent call last)
/tmp/ipykernel_55/424812023.py in <cell line: 0>()
     23     )
     24 
---> 25     history = model.fit(
     26         train_ds,
     27         validation_data=val_ds,

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

UnimplementedError: Graph execution error:

Detected at node EagerPyFunc defined at (most recent call last):
<stack traces unavailable>
Detected at node EagerPyFunc defined at (most recent call last):
<stack traces unavailable>
2 root error(s) found.
  (0) UNIMPLEMENTED:  NotImplementedError: unable to open file: libtensorflow_io.so, from paths: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so']
caused by: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so: undefined symbol: _ZN3tsl8str_util9LowercaseB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEE']
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

  File "/tmp/ipykernel_55/942361435.py", line 8, in _load_one
    x = process_scan_from_dicom(series_dir)  # [1,128,128,64,1]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_55/3627583518.py", line 84, in process_scan_from_dicom
    vol = read_dicom_series(series_dir)  # [H, W, D]
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_55/3627583518.py", line 24, in read_dicom_series
    img = tfio.image.decode_dicom_image(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/dicom_ops.py", line 84, in decode_dicom_image
    return core_ops.io_decode_dicom_image(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 88, in __getattr__
    return getattr(self._load(), attrb)
                   ^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 84, in _load
    self._mod = _load_library(self._library)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 69, in _load_library
    raise NotImplementedError(

NotImplementedError: unable to open file: libtensorflow_io.so, from paths: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so']
caused by: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so: undefined symbol: _ZN3tsl8str_util9LowercaseB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEE']


	 [[{{node EagerPyFunc}}]]
	 [[IteratorGetNext]]
	 [[IteratorGetNext/_2]]
  (1) UNIMPLEMENTED:  NotImplementedError: unable to open file: libtensorflow_io.so, from paths: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so']
caused by: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so: undefined symbol: _ZN3tsl8str_util9LowercaseB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEE']
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

  File "/tmp/ipykernel_55/942361435.py", line 8, in _load_one
    x = process_scan_from_dicom(series_dir)  # [1,128,128,64,1]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_55/3627583518.py", line 84, in process_scan_from_dicom
    vol = read_dicom_series(series_dir)  # [H, W, D]
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_55/3627583518.py", line 24, in read_dicom_series
    img = tfio.image.decode_dicom_image(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/dicom_ops.py", line 84, in decode_dicom_image
    return core_ops.io_decode_dicom_image(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 88, in __getattr__
    return getattr(self._load(), attrb)
                   ^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 84, in _load
    self._mod = _load_library(self._library)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 69, in _load_library
    raise NotImplementedError(

NotImplementedError: unable to open file: libtensorflow_io.so, from paths: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so']
caused by: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so: undefined symbol: _ZN3tsl8str_util9LowercaseB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEE']


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
