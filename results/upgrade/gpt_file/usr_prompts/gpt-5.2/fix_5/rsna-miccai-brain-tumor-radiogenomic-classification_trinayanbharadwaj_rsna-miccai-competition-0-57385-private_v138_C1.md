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

0.42471

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.56118) has done: 'I fix the early crash caused by a protobuf incompatibility (triggered during imports) by forcing the pure‑Python protobuf implementation before importing TensorFlow/pydicom. Then I fix the core logic bug that creates a row-count mismatch: your image loaders skip cases with missing/short sequences, but `create_sub()` still uses *all* test folders and then falls back to constant 0.5 vectors, producing the wrong number/order of predictions. To keep the architecture/ensembling intact while making submission valid, I have both loaders return the exact list of case IDs they successfully processed, compute per‑case predictions, and build the submission by mapping predictions back onto the full sample_submission ID list (defaulting to 0.5 for any missing case). This preserves your inference approach but ensures the output CSV has the correct rows and alignment.'
- What this solution (achieved 0.54941) has done: 'The current failure happens immediately on import because the Kaggle environment’s protobuf version is incompatible with how TensorFlow/pydicom trigger protobuf message creation, so we prevent that crash by enforcing a compatible protobuf runtime setting and import order before TensorFlow is imported. I also add a safe fallback if TensorFlow still cannot be imported (so the notebook always produces a valid `submission.csv`), while keeping your existing prediction/ensembling logic intact when TensorFlow loads successfully. Finally, I keep the submission row alignment based on `sample_submission.csv` exactly as your plan describes, defaulting missing-case predictions to 0.5 (score-neutral for missing rows and ensures a valid file). No model/training logic is changed beyond making imports robust and guaranteeing end-to-end execution.'
- What this solution (achieved 0.42471) has done: 'I fix the immediate import-time crash (`MessageFactory.GetPrototype`) by forcing the safer pure-Python protobuf implementation *and* preventing `pydicom` from importing `google.protobuf` before TensorFlow has a chance to initialize (by importing TensorFlow first, and only then importing pydicom). I also add a robust second-line fallback: if TensorFlow still fails to import due to protobuf in this environment, the script continue and generate a valid `submission.csv` with 0.5 defaults (score-neutral for validity). These changes keep your model/ensembling logic intact and only touch import order / safety. Finally, I keep the existing sample-submission-based alignment to ensure the submission row count/order is always correct.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception:
    sns = None

TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(42)
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    keras = None
    print(
        "WARNING: TensorFlow failed to import; will fall back to constant predictions."
    )
    print("TensorFlow import error:", repr(e))

try:
    import pydicom as dicom
except Exception as e:
    dicom = None
    print("WARNING: pydicom failed to import; will fall back to constant predictions.")
    print("pydicom import error:", repr(e))

np.random.seed(42)

try:
    from skimage.transform import resize as _sk_resize

    def resize(img2d, out_hw):
        return _sk_resize(
            img2d, out_hw, preserve_range=True, anti_aliasing=True
        ).astype(np.float32)

except Exception:

    def resize(img2d, out_hw):
        if TF_AVAILABLE:
            x = tf.convert_to_tensor(img2d, dtype=tf.float32)
            x = tf.expand_dims(x, axis=-1)  # H W 1
            x = tf.image.resize(x, out_hw, method="bilinear", antialias=True)
            x = tf.squeeze(x, axis=-1)
            return x.numpy().astype(np.float32)
        in_h, in_w = img2d.shape[:2]
        out_h, out_w = out_hw
        ys = (np.linspace(0, in_h - 1, out_h)).astype(np.int32)
        xs = (np.linspace(0, in_w - 1, out_w)).astype(np.int32)
        return img2d[np.ix_(ys, xs)].astype(np.float32)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _build_fallback_model(input_shape=(150, 150, 3)):
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0)(inp)
    x = keras.layers.Conv2D(8, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    out = keras.layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inp, out)
    return model


def safe_load_model(path, input_shape=(150, 150, 3)):
    if not TF_AVAILABLE:
        return None
    if path and os.path.exists(path):
        return keras.models.load_model(path)
    return _build_fallback_model(input_shape=input_shape)


model_T2 = safe_load_model(
    "../input/trained-model-for-rsnamiccai/rsna_miccai_114_epochs_T2W_7k_imgs.h5"
)
model_T2_2 = safe_load_model(
    "../input/trained-model-for-rsnamiccai/rsna_miccai_200_epochs_T2W_7k_imgs.h5"
)
model_T2_3 = safe_load_model(
    "../input/trained-model-for-rsnamiccai/rsna_miccai_83_b600_T2W_7k_imgs.h5"
)
model_T2_4 = safe_load_model(
    "../input/trained-model-for-rsnamiccai/rsna_miccai_28_b50_T2W_7k_imgs.h5"
)
model_T2_5 = safe_load_model(
    "../input/trained-model-for-rsnamiccai/rsna_miccai_50_b20_T1wce_6.8k_imgs.h5"
)  # used on T1w in original




## === cell 2
def load_test_T2W_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    used_cases = []
    IMG_PX_SIZE = 150

    if dicom is None:
        return (
            used_cases,
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])

        if len(mri_type) < 4:
            continue

        img_path = sorted([f.path for f in os.scandir(mri_type[3]) if f.is_file()])

        for pth in img_path:
            try:
                img = dicom.dcmread(pth)
                px = img.pixel_array
            except Exception:
                continue

            if px.sum() > 100000:
                resized_img = resize(px, (IMG_PX_SIZE, IMG_PX_SIZE))
                stacked_img = np.stack((resized_img,) * 3, axis=-1).astype(np.float32)
                mx = float(np.max(stacked_img)) if np.max(stacked_img) != 0 else 1.0
                stacked_img_normalize = stacked_img / mx

                if stacked_img_normalize.sum() > 2000:
                    if count == 0:
                        array_1.append(stacked_img_normalize)
                    elif count == 1:
                        array_2.append(stacked_img_normalize)
                    elif count == 2:
                        array_3.append(stacked_img_normalize)
                    elif count == 3:
                        array_4.append(stacked_img_normalize)
                    elif count == 4:
                        array_5.append(stacked_img_normalize)
                    elif count == 5:
                        array_6.append(stacked_img_normalize)
                    count += 1
                    if count == 6:
                        break

        if count == 6:
            used_cases.append(os.path.basename(case_path).zfill(5))
        else:
            if count >= 1:
                array_1 = array_1[:-1]
            if count >= 2:
                array_2 = array_2[:-1]
            if count >= 3:
                array_3 = array_3[:-1]
            if count >= 4:
                array_4 = array_4[:-1]
            if count >= 5:
                array_5 = array_5[:-1]
            if count >= 6:
                array_6 = array_6[:-1]

    def _to_norm(a):
        a = np.asarray(a, dtype=np.float32)
        if a.size == 0:
            return a
        m = float(np.max(a))
        return a / (m if m != 0 else 1.0)

    array_1 = _to_norm(array_1)
    array_2 = _to_norm(array_2)
    array_3 = _to_norm(array_3)
    array_4 = _to_norm(array_4)
    array_5 = _to_norm(array_5)
    array_6 = _to_norm(array_6)

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
    print("Number of T2 cases used:", len(used_cases))

    return used_cases, array_1, array_2, array_3, array_4, array_5, array_6




## === cell 3
def load_test_T1w_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    used_cases = []
    IMG_PX_SIZE = 150

    if dicom is None:
        return (
            used_cases,
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])

        if len(mri_type) < 2:
            continue

        img_path = sorted([f.path for f in os.scandir(mri_type[1]) if f.is_file()])
        for pth in img_path:
            try:
                img = dicom.dcmread(pth)
                px = img.pixel_array
            except Exception:
                continue

            if px.sum() > 100000:
                resized_img = resize(px, (IMG_PX_SIZE, IMG_PX_SIZE))
                stacked_img = np.stack((resized_img,) * 3, axis=-1).astype(np.float32)
                mx = float(np.max(stacked_img)) if np.max(stacked_img) != 0 else 1.0
                stacked_img_normalize = stacked_img / mx

                if stacked_img_normalize.sum() > 2000:
                    if count == 0:
                        array_1.append(stacked_img_normalize)
                    elif count == 1:
                        array_2.append(stacked_img_normalize)
                    elif count == 2:
                        array_3.append(stacked_img_normalize)
                    elif count == 3:
                        array_4.append(stacked_img_normalize)
                    elif count == 4:
                        array_5.append(stacked_img_normalize)
                    elif count == 5:
                        array_6.append(stacked_img_normalize)
                    count += 1
                    if count == 6:
                        break

        if count == 6:
            used_cases.append(os.path.basename(case_path).zfill(5))
        else:
            if count >= 1:
                array_1 = array_1[:-1]
            if count >= 2:
                array_2 = array_2[:-1]
            if count >= 3:
                array_3 = array_3[:-1]
            if count >= 4:
                array_4 = array_4[:-1]
            if count >= 5:
                array_5 = array_5[:-1]
            if count >= 6:
                array_6 = array_6[:-1]

    def _to_norm(a):
        a = np.asarray(a, dtype=np.float32)
        if a.size == 0:
            return a
        m = float(np.max(a))
        return a / (m if m != 0 else 1.0)

    array_1 = _to_norm(array_1)
    array_2 = _to_norm(array_2)
    array_3 = _to_norm(array_3)
    array_4 = _to_norm(array_4)
    array_5 = _to_norm(array_5)
    array_6 = _to_norm(array_6)

    print(
        "Number of T1w images loaded are ",
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
    print("Number of T1w cases used:", len(used_cases))

    return used_cases, array_1, array_2, array_3, array_4, array_5, array_6




## === cell 4
test = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
sample_path = (
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)




## === cell 5
t2_cases, pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = (
    load_test_T2W_images(test)
)




## === cell 6
t1_cases, pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = (
    load_test_T1w_images(test)
)




## === cell 7
def _predict_class1(model, x):
    if x is None or len(x) == 0:
        return np.array([], dtype=np.float32)
    if (model is None) or (not TF_AVAILABLE):
        return np.full((len(x),), 0.5, dtype=np.float32)

    preds = model.predict(x, verbose=0)
    preds = np.asarray(preds)
    if preds.ndim == 1:
        return preds.astype(np.float32)
    if preds.shape[1] == 1:
        return preds[:, 0].astype(np.float32)
    return preds[:, 1].astype(np.float32)


prediction_1 = _predict_class1(model_T2, pixels_1)
prediction_2 = _predict_class1(model_T2, pixels_2)
prediction_3 = _predict_class1(model_T2, pixels_3)
prediction_4 = _predict_class1(model_T2, pixels_4)
prediction_5 = _predict_class1(model_T2, pixels_5)
prediction_6 = _predict_class1(model_T2, pixels_6)

prediction_101 = _predict_class1(model_T2_2, pixels_1)
prediction_102 = _predict_class1(model_T2_2, pixels_2)
prediction_103 = _predict_class1(model_T2_2, pixels_3)
prediction_104 = _predict_class1(model_T2_2, pixels_4)
prediction_105 = _predict_class1(model_T2_2, pixels_5)
prediction_106 = _predict_class1(model_T2_2, pixels_6)

prediction_201 = _predict_class1(model_T2_3, pixels_1)
prediction_202 = _predict_class1(model_T2_3, pixels_2)
prediction_203 = _predict_class1(model_T2_3, pixels_3)
prediction_204 = _predict_class1(model_T2_3, pixels_4)
prediction_205 = _predict_class1(model_T2_3, pixels_5)
prediction_206 = _predict_class1(model_T2_3, pixels_6)

prediction_301 = _predict_class1(model_T2_4, pixels_1)
prediction_302 = _predict_class1(model_T2_4, pixels_2)
prediction_303 = _predict_class1(model_T2_4, pixels_3)
prediction_304 = _predict_class1(model_T2_4, pixels_4)
prediction_305 = _predict_class1(model_T2_4, pixels_5)
prediction_306 = _predict_class1(model_T2_4, pixels_6)

prediction_401 = _predict_class1(model_T2_5, pixels_7)
prediction_402 = _predict_class1(model_T2_5, pixels_8)
prediction_403 = _predict_class1(model_T2_5, pixels_9)
prediction_404 = _predict_class1(model_T2_5, pixels_10)
prediction_405 = _predict_class1(model_T2_5, pixels_11)
prediction_406 = _predict_class1(model_T2_5, pixels_12)




## === cell 8
def create_sub_from_sample(
    sample_submission_path,
    t2_cases,
    t1_cases,
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
    sub = pd.read_csv(sample_submission_path)
    sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)

    def _mean_stack(preds_list, expected_n):
        fixed = []
        for p in preds_list:
            p = np.asarray(p, dtype=np.float32)
            if p.shape[0] != expected_n:
                p = np.full((expected_n,), 0.5, dtype=np.float32)
            fixed.append(p)
        return np.mean(np.stack(fixed, axis=1), axis=1).astype(np.float32)

    t2_pred = _mean_stack(
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
        ],
        expected_n=len(t2_cases),
    )

    t1_pred = _mean_stack(
        [p401, p402, p403, p404, p405, p406], expected_n=len(t1_cases)
    )

    pred_map = {}
    for cid, val in zip(t2_cases, t2_pred):
        pred_map[cid] = float(np.clip(val, 0.0, 1.0))

    for cid, val in zip(t1_cases, t1_pred):
        v = float(np.clip(val, 0.0, 1.0))
        if cid in pred_map:
            pred_map[cid] = 0.5 * pred_map[cid] + 0.5 * v
        else:
            pred_map[cid] = v

    sub["MGMT_value"] = sub["BraTS21ID"].map(pred_map).fillna(0.5).astype(np.float32)
    sub["MGMT_value"] = sub["MGMT_value"].clip(0.0, 1.0)
    return sub




## === cell 9
sub_df = create_sub_from_sample(
    sample_path,
    t2_cases,
    t1_cases,
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

print("Submission preview:")
print(sub_df.head())
print("Submission shape:", sub_df.shape)
print("Any NA MGMT_value:", sub_df["MGMT_value"].isna().any())




## === cell 10
if sns is not None:
    try:
        sns.displot(sub_df["MGMT_value"])
    except Exception:
        pass




## === cell 11
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.dtypes)
print(sub_df.head())
print("Unique IDs:", sub_df["BraTS21ID"].nunique(), " / rows:", len(sub_df))
