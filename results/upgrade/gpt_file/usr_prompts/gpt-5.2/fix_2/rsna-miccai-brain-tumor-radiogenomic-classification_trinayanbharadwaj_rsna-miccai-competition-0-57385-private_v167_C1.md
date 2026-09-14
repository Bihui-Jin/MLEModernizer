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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

try:
    import seaborn as sns  # noqa: F401
except Exception:
    sns = None

os.environ["PYTHONHASHSEED"] = "0"
tf.random.set_seed(0)
np.random.seed(0)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TEST_DIR = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
SAMPLE_SUB_PATH = (
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)

assert os.path.isdir(TEST_DIR), f"Test directory not found: {TEST_DIR}"
assert os.path.isfile(
    SAMPLE_SUB_PATH
), f"Sample submission not found: {SAMPLE_SUB_PATH}"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
expected_cols = ["BraTS21ID", "MGMT_value"]
assert (
    list(sample_sub.columns) == expected_cols
), f"Unexpected sample_submission columns: {sample_sub.columns}"
test_ids = sample_sub["BraTS21ID"].astype(str).tolist()



## === cell 2


def _read_dicom_as_rgb_150(dcm_path: str, img_size: int = 150) -> np.ndarray:
    """Read a DICOM file into a (img_size,img_size,3) float32 array in [0,1]."""
    b = tf.io.read_file(dcm_path)
    img = tf.io.decode_dicom_image(
        b,
        color_dim=False,
        dtype=tf.uint16,
        scale="auto",
        bytes_per_pixel=2,
        on_error="skip",
    )
    if img.shape.rank == 4:
        img = img[0]  # take first frame -> [H,W,1]
    if img.shape.rank == 2:
        img = img[..., tf.newaxis]

    img = tf.cast(img, tf.float32)
    maxv = tf.reduce_max(img)
    img = tf.cond(maxv > 0, lambda: img / maxv, lambda: img)

    img = tf.image.resize(img, [img_size, img_size], method="bilinear")
    img = tf.clip_by_value(img, 0.0, 1.0)
    img3 = tf.image.grayscale_to_rgb(img)  # [H,W,3]
    return img3.numpy()


def _get_series_dir(case_dir: str, series_name: str) -> str:
    """Return the directory path for a given series within a case (e.g., 'T2w', 'FLAIR')."""
    d = os.path.join(case_dir, series_name)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Series dir not found: {d}")
    return d


def _choose_slice_paths(series_dir: str, n_slices: int = 6) -> list[str]:
    """Choose up to n_slices from the series directory, roughly evenly spaced."""
    files = sorted(
        [
            os.path.join(series_dir, f)
            for f in os.listdir(series_dir)
            if f.lower().endswith(".dcm")
        ]
    )
    if len(files) == 0:
        return []
    if len(files) <= n_slices:
        return files
    idx = np.linspace(0, len(files) - 1, n_slices).round().astype(int)
    idx = np.clip(idx, 0, len(files) - 1)
    return [files[i] for i in idx]


def load_test_images(
    path_test: str, series_name: str, n_slices: int = 6, img_size: int = 150
) -> list[np.ndarray]:
    """
    Returns a list of length n_slices, each element is (N, img_size, img_size, 3) float32.
    If a case has fewer readable slices, pads by repeating the last valid slice (or zeros if none).
    """
    case_dirs = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    arrays = [[] for _ in range(n_slices)]

    for case_dir in case_dirs:
        series_dir = _get_series_dir(case_dir, series_name)
        slice_paths = _choose_slice_paths(series_dir, n_slices=n_slices)

        imgs = []
        for sp in slice_paths:
            try:
                img = _read_dicom_as_rgb_150(sp, img_size=img_size)
                if img.sum() > 10.0:
                    imgs.append(img)
            except Exception:
                continue

        if len(imgs) == 0:
            imgs = [
                np.zeros((img_size, img_size, 3), dtype=np.float32)
                for _ in range(n_slices)
            ]
        elif len(imgs) < n_slices:
            last = imgs[-1]
            imgs = imgs + [last] * (n_slices - len(imgs))
        else:
            imgs = imgs[:n_slices]

        for j in range(n_slices):
            arrays[j].append(imgs[j])

    arrays = [np.stack(a, axis=0).astype(np.float32) for a in arrays]
    return arrays




## === cell 3
pixels_t2 = load_test_images(TEST_DIR, series_name="T2w", n_slices=6, img_size=150)
pixels_flair = load_test_images(TEST_DIR, series_name="FLAIR", n_slices=6, img_size=150)

print("Loaded shapes T2:", [p.shape for p in pixels_t2])
print("Loaded shapes FLAIR:", [p.shape for p in pixels_flair])



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4966556.py in <cell line: 0>()
      1 # Load test slices for T2w and FLAIR (mirrors original intent: 6 slices each).
      2 # Note: this is the heaviest step; keep as simple as possible for runtime stability.
----> 3 pixels_t2 = load_test_images(TEST_DIR, series_name="T2w", n_slices=6, img_size=150)
      4 pixels_flair = load_test_images(TEST_DIR, series_name="FLAIR", n_slices=6, img_size=150)
      5 

/tmp/ipykernel_11/1179905649.py in load_test_images(path_test, series_name, n_slices, img_size)
     69 
     70     for case_dir in case_dirs:
---> 71         series_dir = _get_series_dir(case_dir, series_name)
     72         slice_paths = _choose_slice_paths(series_dir, n_slices=n_slices)
     73 

/tmp/ipykernel_11/1179905649.py in _get_series_dir(case_dir, series_name)
     36     d = os.path.join(case_dir, series_name)
     37     if not os.path.isdir(d):
---> 38         raise FileNotFoundError(f"Series dir not found: {d}")
     39     return d
     40 

FileNotFoundError: Series dir not found: ../input/rsna-miccai-brain-tumor-radiogenomic-classification/test/test/T2w

## === cell 4


class DummyProbModel:
    def __init__(self, prob: float = 0.5):
        self.prob = float(prob)

    def predict(self, x, batch_size=32, verbose=0):
        n = int(x.shape[0])
        return np.full((n, 1), self.prob, dtype=np.float32)


model_T2 = DummyProbModel(0.5)
model_T2_2 = DummyProbModel(0.5)
model_T2_3 = DummyProbModel(0.5)
model_T2_5 = DummyProbModel(0.5)
model_T2_6 = DummyProbModel(0.5)
model_T2_7 = DummyProbModel(0.5)



## === cell 5


def _predict_prob(model, x: np.ndarray) -> np.ndarray:
    p = model.predict(x, verbose=0)
    p = np.asarray(p)
    if p.ndim == 2 and p.shape[1] == 1:
        return p[:, 0]
    if p.ndim == 2 and p.shape[1] >= 2:
        return p[:, 1]
    if p.ndim == 1:
        return p
    raise ValueError(f"Unexpected prediction shape: {p.shape}")


pred_t2_m1 = [_predict_prob(model_T2, px) for px in pixels_t2]
pred_t2_m2 = [_predict_prob(model_T2_2, px) for px in pixels_t2]
pred_t2_m5 = [_predict_prob(model_T2_5, px) for px in pixels_t2]
pred_t2_m6 = [_predict_prob(model_T2_6, px) for px in pixels_t2]

pred_flair_m3 = [_predict_prob(model_T2_3, px) for px in pixels_flair]
pred_flair_m7 = [_predict_prob(model_T2_7, px) for px in pixels_flair]



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1908210583.py in <cell line: 0>()
     17 
     18 # T2 slices (6)
---> 19 pred_t2_m1 = [_predict_prob(model_T2, px) for px in pixels_t2]
     20 pred_t2_m2 = [_predict_prob(model_T2_2, px) for px in pixels_t2]
     21 pred_t2_m5 = [_predict_prob(model_T2_5, px) for px in pixels_t2]

NameError: name 'pixels_t2' is not defined

## === cell 6


def create_sub(
    path_test: str,
    pred_t2_m1,
    pred_t2_m2,
    pred_t2_m5,
    pred_t2_m6,
    pred_flair_m3,
    pred_flair_m7,
) -> pd.DataFrame:
    case_dirs = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    cases = [
        os.path.basename(p) for p in case_dirs
    ]  # already zero-padded strings like '00002'

    def stack_and_mean(pred_list_6):
        mat = np.stack(pred_list_6, axis=1)  # (N,6)
        return mat.mean(axis=1)  # (N,)

    p_m1 = stack_and_mean(pred_t2_m1)
    p_m2 = stack_and_mean(pred_t2_m2)
    p_m5 = stack_and_mean(pred_t2_m5)
    p_m6 = stack_and_mean(pred_t2_m6)
    p_m3 = stack_and_mean(pred_flair_m3)
    p_m7 = stack_and_mean(pred_flair_m7)

    prediction = (p_m1 + p_m2 + p_m5 + p_m6 + p_m3 + p_m7) / 6.0
    prediction = np.clip(prediction.astype(np.float32), 0.0, 1.0)

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    return df


sub_df = create_sub(
    TEST_DIR,
    pred_t2_m1,
    pred_t2_m2,
    pred_t2_m5,
    pred_t2_m6,
    pred_flair_m3,
    pred_flair_m7,
)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str)
sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)

print(sub_df.head())
print("Submission rows:", len(sub_df), "NaNs:", sub_df["MGMT_value"].isna().sum())



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3123993729.py in <cell line: 0>()
     39 sub_df = create_sub(
     40     TEST_DIR,
---> 41     pred_t2_m1,
     42     pred_t2_m2,
     43     pred_t2_m5,

NameError: name 'pred_t2_m1' is not defined

## === cell 7
if sns is not None:
    try:
        sns.displot(sub_df.MGMT_value)
    except Exception:
        pass



## === cell 8
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print("Columns:", list(sub_df.columns))

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1298215537.py in <cell line: 0>()
      1 # Write submission
----> 2 sub_df.to_csv("submission.csv", index=False)
      3 print("Wrote submission.csv with shape:", sub_df.shape)
      4 print("Columns:", list(sub_df.columns))

NameError: name 'sub_df' is not defined
