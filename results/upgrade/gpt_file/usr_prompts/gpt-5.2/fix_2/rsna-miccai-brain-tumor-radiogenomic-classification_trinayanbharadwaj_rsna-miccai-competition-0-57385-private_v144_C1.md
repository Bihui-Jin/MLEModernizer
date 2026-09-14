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

0.45059

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.45059) has done: 'I remove/guard imports that trigger the protobuf `MessageFactory.GetPrototype` crash and keep only what’s needed for inference. Since the referenced pre-trained `.h5` files are not present in your `/kaggle/input`, I replace that block with a small TensorFlow/Keras model that preserves the same “predict then ensemble/average” semantics, so the notebook runs end-to-end and outputs `submission.csv`. I also fix the missing `resize` symbol and multiple logic/runtime issues in `load_test_T2W_images` and `create_sub` (lists vs arrays, prediction computed inside the loop, and ID formatting/alignment). Finally, I ensure the submission matches `sample_submission.csv` ordering and has the correct columns and a `.csv` suffix.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)
sample_sub.head()




## === cell 2
def load_test_T2W_images(path_test, img_px_size=150, n_slices=7):
    """
    Loads up to n_slices per case from the T2w series (expects folder names FLAIR/T1w/T1wCE/T2w).
    Returns:
      pixels_list: list of length n_slices; each element is an array (N, H, W, 3)
      case_ids: list of BraTS21ID strings (zero-padded length 5) in the same order used for pixels_list arrays
    """
    case_dirs = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    case_ids = [
        os.path.basename(p) for p in case_dirs
    ]  # already zero-padded in folder name

    arrays = [[] for _ in range(n_slices)]

    for case_path in case_dirs:
        t2_dir = os.path.join(case_path, "T2w")
        if not os.path.isdir(t2_dir):
            modalities = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
            if len(modalities) == 0:
                for si in range(n_slices):
                    arrays[si].append(
                        np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
                    )
                continue
            t2_dir = modalities[-1]

        dcm_paths = sorted([f.path for f in os.scandir(t2_dir) if f.is_file()])
        count = 0

        for dp in dcm_paths:
            if count >= n_slices:
                break
            try:
                ds = dicom.dcmread(dp, force=True)
                px = ds.pixel_array.astype(np.float32)
            except Exception:
                continue

            if px.size == 0:
                continue
            if px.sum() <= 100000:
                continue

            px_rs = resize(
                px, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            stacked = np.stack((px_rs, px_rs, px_rs), axis=-1)

            mx = float(stacked.max())
            if mx <= 0:
                continue
            stacked_norm = stacked / mx

            if stacked_norm.sum() <= 2000:
                continue

            arrays[count].append(stacked_norm)
            count += 1

        while count < n_slices:
            arrays[count].append(
                np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
            )
            count += 1

    pixels_list = [np.asarray(a, dtype=np.float32) for a in arrays]

    print(
        "Loaded T2 slices per position:",
        [x.shape[0] for x in pixels_list],
        "cases:",
        len(case_ids),
    )
    return pixels_list, case_ids




## === cell 3
pixels_list, case_ids = load_test_T2W_images(TEST_DIR, img_px_size=150, n_slices=7)

pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6, pixels_7 = pixels_list




## === cell 4
def build_light_model(input_shape=(150, 150, 3), seed=42):
    tf.random.set_seed(seed)
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model


model_T2 = build_light_model(seed=SEED + 1)
model_T2_2 = build_light_model(seed=SEED + 2)
model_T2_3 = build_light_model(seed=SEED + 3)
model_T2_4 = build_light_model(seed=SEED + 4)




## === cell 5
def predict_class1(model, x, batch_size=32):
    preds = model.predict(x, batch_size=batch_size, verbose=0)
    return preds[:, 1].astype(np.float32)


prediction_1 = predict_class1(model_T2, pixels_1)
prediction_2 = predict_class1(model_T2, pixels_2)
prediction_3 = predict_class1(model_T2, pixels_3)
prediction_4 = predict_class1(model_T2, pixels_4)
prediction_5 = predict_class1(model_T2, pixels_5)
prediction_6 = predict_class1(model_T2, pixels_6)
prediction_7 = predict_class1(model_T2, pixels_7)

prediction_101 = predict_class1(model_T2_2, pixels_1)
prediction_102 = predict_class1(model_T2_2, pixels_2)
prediction_103 = predict_class1(model_T2_2, pixels_3)
prediction_104 = predict_class1(model_T2_2, pixels_4)
prediction_105 = predict_class1(model_T2_2, pixels_5)
prediction_106 = predict_class1(model_T2_2, pixels_6)
prediction_107 = predict_class1(model_T2_2, pixels_7)

prediction_201 = predict_class1(model_T2_3, pixels_1)
prediction_202 = predict_class1(model_T2_3, pixels_2)
prediction_203 = predict_class1(model_T2_3, pixels_3)
prediction_204 = predict_class1(model_T2_3, pixels_4)
prediction_205 = predict_class1(model_T2_3, pixels_5)
prediction_206 = predict_class1(model_T2_3, pixels_6)
prediction_207 = predict_class1(model_T2_3, pixels_7)

prediction_301 = predict_class1(model_T2_4, pixels_1)
prediction_302 = predict_class1(model_T2_4, pixels_2)
prediction_303 = predict_class1(model_T2_4, pixels_3)
prediction_304 = predict_class1(model_T2_4, pixels_4)
prediction_305 = predict_class1(model_T2_4, pixels_5)
prediction_306 = predict_class1(model_T2_4, pixels_6)
prediction_307 = predict_class1(model_T2_4, pixels_7)




## === cell 6
def create_sub(
    case_ids,
    p1,
    p2,
    p3,
    p4,
    p5,
    p101,
    p102,
    p103,
    p104,
    p105,
    p201,
    p202,
    p203,
    p204,
    p205,
    p301,
    p302,
    p303,
    p304,
    p305,
):
    """
    Fixes original logic bug: prediction must be computed once (vectorized) rather than inside the case loop.
    Keeps the original averaging scheme over 20 prediction vectors.
    """
    prediction = (
        p1.astype(np.float32)
        + p2.astype(np.float32)
        + p3.astype(np.float32)
        + p4.astype(np.float32)
        + p5.astype(np.float32)
        + p101.astype(np.float32)
        + p102.astype(np.float32)
        + p103.astype(np.float32)
        + p104.astype(np.float32)
        + p105.astype(np.float32)
        + p201.astype(np.float32)
        + p202.astype(np.float32)
        + p203.astype(np.float32)
        + p204.astype(np.float32)
        + p205.astype(np.float32)
        + p301.astype(np.float32)
        + p302.astype(np.float32)
        + p303.astype(np.float32)
        + p304.astype(np.float32)
        + p305.astype(np.float32)
    ) / 20.0

    prediction = np.clip(prediction, 0.0, 1.0)

    df = pd.DataFrame(
        {
            "BraTS21ID": pd.Series(case_ids, dtype=str).str.zfill(5),
            "MGMT_value": prediction,
        }
    )
    return df


sub_df = create_sub(
    case_ids,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_101,
    prediction_102,
    prediction_103,
    prediction_104,
    prediction_105,
    prediction_201,
    prediction_202,
    prediction_203,
    prediction_204,
    prediction_205,
    prediction_301,
    prediction_302,
    prediction_303,
    prediction_304,
    prediction_305,
)

sub_df.head()



## === cell 7
sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")

sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(np.float32).fillna(0.5)

assert list(sub_df.columns) == ["BraTS21ID", "MGMT_value"]
assert len(sub_df) == len(sample_sub)
sub_df.describe()



## === cell 8
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
