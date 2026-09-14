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
Predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (`FVC`), i.e. the volume of air exhaled.

## Metric
A modified version of the Laplace Log Likelihood. 

For each true FVC measurement, you will predict both an FVC and a confidence measure (standard deviation 𝜎𝜎). The metric is computed as:

$$
\begin{gathered}
\sigma_{\text {clipped }}=\max (\sigma, 70), \\
\Delta=\min \left(\left|F V C_{\text {true }}-F V C_{\text {predicted }}\right|, 1000\right), \\
\text { metric }=-\frac{\sqrt{2} \Delta}{\sigma_{\text {clipped }}}-\ln \left(\sqrt{2} \sigma_{\text {clipped }}\right) .
\end{gathered}
$$

The error is thresholded at 1000 ml to avoid large errors adversely penalizing results, while the confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty in FVC. The final score is calculated by averaging the metric across all test set `Patient_Week`s (three per patient). 

Metric values will be negative and higher is better.

## Submission Format
For each `Patient_Week`, you must predict the `FVC` and a confidence. You are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

The file should contain a header and have the following format:

```
Patient_Week,FVC,Confidence
ID00002637202176704235138_1,2000,100
ID00002637202176704235138_2,2000,100
ID00002637202176704235138_3,2000,100
etc.

```

## Dataset
In the dataset, you are provided with a baseline chest CT scan and associated clinical information for a set of patients. A patient has an image acquired at time `Week = 0` and has numerous follow up visits over the course of approximately 1-2 years, at which time their `FVC` is measured.

- In the training set, you are provided with an anonymized, baseline CT scan and the entire history of FVC measurements.
- In the test set, you are provided with a baseline CT scan and only the initial FVC measurement. **You are asked to predict the final three `FVC` measurements for each patient, as well as a confidence value in your prediction.**

- **train.csv** - the training set, contains full history of clinical information
- **test.csv** - the test set, contains only the baseline measurement
- **train/** - contains the training patients' baseline CT scan in DICOM format
- **test/** - contains the test patients' baseline CT scan in DICOM format
- **sample_submission.csv** - demonstrates the submission format

**train.csv and test.csv**

- `Patient`a unique Id for each patient (also the name of the patient's DICOM folder)
- `Weeks`the relative number of weeks pre/post the baseline CT (may be negative)
- `FVC` - the recorded lung capacity in ml
- `Percent`a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
- `Age`
- `Sex`
- `SmokingStatus`

**sample submission.csv**

- `Patient_Week` - a unique Id formed by concatenating the `Patient` and `Weeks` columns (i.e. ABC_22 is a prediction for patient ABC at week 22)
- `FVC` - the predicted FVC in ml
- `Confidence` - a confidence value of your prediction (also has units of ml)

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 5. Target score

-6.972037427099178

# 6. Current score

-8.76216

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.76216) has done: 'The timeout is dominated by (1) reading/resizing every DICOM slice for all train patients and (2) an extremely long TF training schedule (5-fold CV × 800 epochs + final 800 epochs). To stay within 600 seconds without changing the model/training semantics, the key speedups are: cache CT-derived pixel vectors to disk (so DICOM parsing happens once), parallelize CT feature extraction across CPU cores, and make TensorFlow run faster per-epoch via `tf.data` pipelines (no Python batching) and `steps_per_execution` while keeping the same epochs, batch size, folds, loss, and architecture. These changes are provably equivalent in outputs (up to negligible FP differences) because they only alter input pipeline and reuse deterministic cached features; the optimization does not change any training hyperparameters or the network. I also avoid redundant work (duplicate CT concat/merge, repeated scaler fits) and add deterministic settings to keep results stable.'
- What this solution (achieved -8.76216) has done: 'I fix two execution-stopping issues: (1) the `pydicom` import crash caused by an incompatible protobuf runtime by forcing Python protobuf implementation before importing `pydicom`, and (2) the `None values not supported` error in `tf.data` by correctly constructing the inference dataset (it was producing a nested tuple with a `None` target). I also ensure the submission is always written with the exact required columns and `.csv` suffix. These changes are score-neutral (they don’t alter the model, features, or training schedule) and should let the pipeline run end-to-end so you can get a new score closer to the target.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import re
import gc
import random
import warnings
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import pydicom

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn import preprocessing

from skimage.transform import resize
from skimage import morphology, measure

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Reshape,
    BatchNormalization,
    Activation,
    Add,
    Conv1D,
    MaxPooling1D,
    concatenate,
)

warnings.filterwarnings("ignore")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def make_lungmask(img, display=False):
    row_size = img.shape[0]
    col_size = img.shape[1]

    mean = np.mean(img)
    std = np.std(img) if np.std(img) > 0 else 1.0
    img = img - mean
    img = img / std

    middle = img[
        int(col_size / 5) : int(col_size / 5 * 4),
        int(row_size / 5) : int(row_size / 5 * 4),
    ]
    mean_mid = np.mean(middle)
    maxv = np.max(img)
    minv = np.min(img)

    img[img == maxv] = mean_mid
    img[img == minv] = mean_mid

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(
        np.reshape(middle, [np.prod(middle.shape), 1])
    )
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)

    eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
    dilation = morphology.dilation(eroded, np.ones([8, 8]))

    labels = measure.label(dilation)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if (
            B[2] - B[0] < row_size / 10 * 9
            and B[3] - B[1] < col_size / 10 * 9
            and B[0] > row_size / 5
            and B[2] < col_size / 5 * 4
        ):
            good_labels.append(prop.label)

    mask = np.zeros([row_size, col_size], dtype=np.int8)
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)

    mask = morphology.dilation(mask, np.ones([10, 10]))

    if display:
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap="gray")
        ax[0, 0].axis("off")
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap="gray")
        ax[0, 1].axis("off")
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap="gray")
        ax[1, 0].axis("off")
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis("off")
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap="gray")
        ax[2, 0].axis("off")
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask * img, cmap="gray")
        ax[2, 1].axis("off")
        plt.show()

    return mask * img




## === cell 2
ROOT = "../input/osic-pulmonary-fibrosis-progression"
TRAIN_CT_DIR = f"{ROOT}/train"
TEST_CT_DIR = f"{ROOT}/test"

train_patients = sorted(
    [
        p
        for p in os.listdir(TRAIN_CT_DIR)
        if os.path.isdir(os.path.join(TRAIN_CT_DIR, p))
    ]
)
test_patients = sorted(
    [p for p in os.listdir(TEST_CT_DIR) if os.path.isdir(os.path.join(TEST_CT_DIR, p))]
)

len(train_patients), len(test_patients)




## === cell 3
def _sorted_dcm_files(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".dcm")]

    def _key(f):
        s = re.sub(r"\D", "", f)
        return int(s) if s else 0

    files.sort(key=_key)
    return [os.path.join(folder, f) for f in files]


def extract_patient_pixels(patient_folder, size=64, use_lungmask=False):
    dcm_files = _sorted_dcm_files(patient_folder)
    if len(dcm_files) == 0:
        return np.zeros((size * size,), dtype=np.float32)

    imgs = []
    for fp in dcm_files:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=False, force=True)
            arr = ds.pixel_array.astype(np.float32, copy=False)
        except Exception:
            continue
        arr = resize(arr, (size, size), anti_aliasing=True, preserve_range=True).astype(
            np.float32,
            copy=False,
        )
        if use_lungmask:
            arr = make_lungmask(arr, display=False).astype(np.float32, copy=False)
        imgs.append(arr)

    if len(imgs) == 0:
        out = np.zeros((size * size,), dtype=np.float32)
    else:
        mean_img = np.mean(np.stack(imgs, axis=0), axis=0).astype(
            np.float32, copy=False
        )
        out = mean_img.reshape(-1)

    mx = float(np.max(out))
    if mx > 0:
        out = out / mx
    return out.astype(np.float32, copy=False)




## === cell 4
SIZE_IMG = 64

CACHE_DIR = "./ct_cache_npz"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(split, patient, size, use_lungmask):
    lm = "lm1" if use_lungmask else "lm0"
    return os.path.join(CACHE_DIR, f"{split}_{patient}_s{size}_{lm}.npz")


def _extract_with_cache(split, base_dir, patient, size=64, use_lungmask=False):
    cp = _cache_path(split, patient, size, use_lungmask)
    if os.path.exists(cp):
        try:
            return patient, np.load(cp)["pix"].astype(np.float32, copy=False)
        except Exception:
            pass
    pix = extract_patient_pixels(
        os.path.join(base_dir, patient), size=size, use_lungmask=use_lungmask
    )
    try:
        np.savez_compressed(cp, pix=pix)
    except Exception:
        pass
    return patient, pix


def build_pixels_df(
    patients, split, base_dir, size=64, use_lungmask=False, max_workers=None
):
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 2) - 1)

    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(_extract_with_cache, split, base_dir, p, size, use_lungmask)
            for p in patients
        ]
        for fut in as_completed(futs):
            patient, pix = fut.result()
            results[patient] = pix

    ordered_pixels = [results[p] for p in patients]
    return pd.DataFrame({"Patient": patients, "Pixels": ordered_pixels})


train_pixels_df = build_pixels_df(
    train_patients, "train", TRAIN_CT_DIR, size=SIZE_IMG, use_lungmask=False
)
test_pixels_df = build_pixels_df(
    test_patients, "test", TEST_CT_DIR, size=SIZE_IMG, use_lungmask=False
)

df = train_pixels_df
df_test2 = test_pixels_df
df_test = df_test2.copy()

df.shape, df_test2.shape




## === cell 5
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(42)

BATCH_SIZE = 128

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 6
train = pd.read_csv(f"{ROOT}/train.csv")
train.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])
test = pd.read_csv(f"{ROOT}/test.csv")

sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
pw = sub["Patient_Week"].str.split("_", n=1, expand=True)
sub["Patient"] = pw[0].values
sub["Weeks"] = pw[1].astype(int).values
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub = sub.merge(test.drop("Weeks", axis=1), on="Patient")



## === cell 7
df["WHERE"] = "train"
df_test["WHERE"] = "val"
df_test2["WHERE"] = "test"

ct_data = pd.concat([df, df_test, df_test2], ignore_index=True)



## === cell 8
train["WHERE"] = "train"
test["WHERE"] = "val"
sub["WHERE"] = "test"

data = pd.concat([train, test, sub], ignore_index=True)

data["min_week"] = data["Weeks"]
data.loc[data.WHERE == "test", "min_week"] = np.nan
data["min_week"] = data.groupby("Patient")["min_week"].transform("min")

data["max_week"] = data["Weeks"]
data.loc[data.WHERE == "test", "max_week"] = np.nan
data["max_week"] = data.groupby("Patient")["max_week"].transform("max")

data["DLCO"] = data["FVC"]
data["min_FEV1"] = data["FVC"]



## === cell 9
base = data.loc[data.Weeks == data.min_week]
base = base[["Patient", "FVC", "Percent", "DLCO", "min_FEV1"]].copy()
base.columns = ["Patient", "min_FVC", "min_Percent", "min_DLCO", "min_FEV"]
base["nb"] = 1
base["nb"] = base.groupby("Patient")["nb"].transform("cumsum")
base = base[base.nb == 1]
base.drop("nb", axis=1, inplace=True)
data = data.merge(base, on="Patient", how="left")

base1 = data.loc[data.Weeks == data.max_week]
base1 = base1[["Patient", "FVC", "Percent"]].copy()
base1.columns = ["Patient", "max_FVC", "max_Percent"]
base1["nb"] = 1
base1["nb"] = base1.groupby("Patient")["nb"].transform("cumsum")
base1 = base1[base1.nb == 1]
base1.drop("nb", axis=1, inplace=True)
data = data.merge(base1, on="Patient", how="left")



## === cell 10
data = data.merge(ct_data, on=["Patient", "WHERE"], how="left")
missing_pixels = int(data["Pixels"].isna().sum())
if missing_pixels > 0:
    zero_pix = np.zeros((SIZE_IMG * SIZE_IMG,), dtype=np.float32)
    data.loc[data["Pixels"].isna(), "Pixels"] = [zero_pix] * missing_pixels



## === cell 11
data["min_FVC1"] = 0.84 - 0.3 / data["min_FVC"]
data["FEV1"] = 0.84 - 0.3 / data["FVC"]
data["base_week"] = data["Weeks"] - data["min_week"]
data["Neg_Age"] = -1 / (data["Age"])

COLS = ["Sex", "SmokingStatus"]
FE = []
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)

min_max_scaler = preprocessing.MinMaxScaler()

data["age"] = min_max_scaler.fit_transform(data["Age"].values.reshape(-1, 1))
data["BASE"] = min_max_scaler.fit_transform(data["min_FVC"].values.reshape(-1, 1))
data["week"] = min_max_scaler.fit_transform(data["base_week"].values.reshape(-1, 1))
data["org_week"] = min_max_scaler.fit_transform(data["Weeks"].values.reshape(-1, 1))
data["percent"] = min_max_scaler.fit_transform(data["Percent"].values.reshape(-1, 1))
data["min_fev"] = min_max_scaler.fit_transform(data["min_FVC1"].values.reshape(-1, 1))
data["neg_age"] = min_max_scaler.fit_transform(data["Neg_Age"].values.reshape(-1, 1))

FE = [
    "Ex-smoker",
    "org_week",
    "Never smoked",
    "week",
    "percent",
    "Currently smokes",
    "age",
    "Female",
    "Male",
    "BASE",
    "min_fev",
    "neg_age",
]
FE2 = ["Pixels"]



## === cell 12
train_df = data.loc[data.WHERE == "train"].copy()
val_df = data.loc[data.WHERE == "val"].copy()
sub_df = data.loc[data.WHERE == "test"].copy()

for c in FE:
    if c not in train_df.columns:
        train_df[c] = 0
        val_df[c] = 0
        sub_df[c] = 0

y = train_df["FVC"].values.astype(float).reshape(-1, 1)
z1 = train_df[FE].values.astype(np.float32)
ze1 = sub_df[FE].values.astype(np.float32)

z2 = np.ascontiguousarray(np.stack(train_df["Pixels"].values).astype(np.float32))
ze2 = np.ascontiguousarray(np.stack(sub_df["Pixels"].values).astype(np.float32))

z1.shape, z2.shape, ze1.shape, ze2.shape, y.shape



## === cell 13
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.cast(2.0, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.mean(metric)


def qloss(y_true, y_pred):
    qs = [0.2, 0.50, 0.80]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss


def make_model():
    z1_in = L.Input((len(FE),))
    z2_in = L.Input((64 * 64,), name="Patient")

    init = tf.keras.initializers.VarianceScaling(
        scale=0.1, mode="fan_in", distribution="uniform", seed=42
    )

    x = L.Dense(
        100, kernel_initializer=init, bias_initializer="zeros", activation="relu"
    )(z1_in)
    x = L.Dense(
        100, kernel_initializer=init, bias_initializer="zeros", activation="relu"
    )(x)
    x = M.Model(inputs=z1_in, outputs=x)

    y_ = Reshape((64 * 64, 1))(z2_in)
    shortcut = y_

    y_ = Conv1D(
        kernel_initializer=init,
        activation="relu",
        padding="same",
        filters=4,
        kernel_size=8,
    )(y_)
    y_ = Conv1D(kernel_initializer=init, padding="same", filters=4, kernel_size=8)(y_)

    y_ = Add()([y_, shortcut])
    y_ = Activation("relu")(y_)

    y_ = MaxPooling1D(pool_size=4)(y_)
    y_ = Flatten()(y_)

    y_ = L.Dense(
        256, kernel_initializer=init, bias_initializer="zeros", activation="relu"
    )(y_)
    y_ = L.Dense(
        128, kernel_initializer=init, bias_initializer="zeros", activation="relu"
    )(y_)
    y_ = L.Dense(
        64, kernel_initializer=init, bias_initializer="zeros", activation="relu"
    )(y_)
    y_ = L.Dense(
        32, kernel_initializer=init, bias_initializer="zeros", activation="relu"
    )(y_)
    y_ = L.Dense(
        16, kernel_initializer=init, bias_initializer="zeros", activation="relu"
    )(y_)
    y_ = L.Dense(
        8, kernel_initializer=init, bias_initializer="zeros", activation="relu"
    )(y_)
    y_ = M.Model(inputs=z2_in, outputs=y_)

    combined = concatenate([x.output, y_.output])

    p1 = L.Dense(3, activation="relu", name="p1")(combined)
    p2 = L.Dense(3, activation="linear", name="p2")(combined)

    preds = L.Lambda(lambda t: t[0] + tf.cumsum(t[1], axis=1), name="preds")([p1, p2])

    model = M.Model([z1_in, z2_in], preds, name="CNN")

    model.compile(
        loss=mloss(0.80),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.11,
            beta_1=0.92,
            beta_2=0.998,
            epsilon=None,
            decay=0.01,
            amsgrad=False,
        ),
        metrics=[score],
        steps_per_execution=32,
    )
    return model




## === cell 14
net = make_model()
print(net.summary())
print("params:", net.count_params())



## === cell 15
NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=42)
seed_everything(42)

pe = np.zeros((ze2.shape[0], 3), dtype=np.float32)
pred = np.zeros((z2.shape[0], 3), dtype=np.float32)

results_t = []
results_v = []

AUTOTUNE = tf.data.AUTOTUNE


def make_ds(x1, x2, y=None, training=False):
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices((x1, x2))
    else:
        ds = tf.data.Dataset.from_tensor_slices(((x1, x2), y))
    if training:
        ds = ds.shuffle(buffer_size=len(x1), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


cnt = 0
for tr_idx, val_idx in kf.split(z1):
    cnt += 1
    print(f"FOLD {cnt}")
    net = make_model()

    tr_ds = make_ds(z1[tr_idx], z2[tr_idx], y[tr_idx], training=True)
    va_ds = make_ds(z1[val_idx], z2[val_idx], y[val_idx], training=False)

    net.fit(
        tr_ds,
        epochs=800,
        validation_data=va_ds,
        verbose=0,
    )
    tr_eval = net.evaluate(tr_ds, verbose=0)
    va_eval = net.evaluate(va_ds, verbose=0)
    print("train", tr_eval)
    print("val", va_eval)
    results_t.append(tr_eval)
    results_v.append(va_eval)

    pred[val_idx] = net.predict(
        make_ds(z1[val_idx], z2[val_idx], y=None, training=False), verbose=0
    )

full_ds = make_ds(z1, z2, y, training=True)
net.fit(full_ds, epochs=800, verbose=0)

test_ds = make_ds(ze1, ze2, y=None, training=False)
pe = net.predict(test_ds, verbose=0)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2757204548.py in <cell line: 0>()
     36     va_ds = make_ds(z1[val_idx], z2[val_idx], y[val_idx], training=False)
     37 
---> 38     net.fit(
     39         tr_ds,
     40         epochs=800,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py in convert_to_tensor(x, dtype, sparse)
    135             x = tf.convert_to_tensor(x)
    136             return tf.cast(x, dtype)
--> 137         return tf.convert_to_tensor(x, dtype=dtype)
    138     elif dtype is not None and not x.dtype == dtype:
    139         if isinstance(x, tf.SparseTensor):

ValueError: None values not supported.

## === cell 16
print("training loss + score: ", np.mean(results_t, axis=0))
print("validation loss + score: ", np.mean(results_v, axis=0))
print("Final loss + score: ", np.mean(results_t, axis=0) * np.mean(results_v, axis=0))

sigma_opt = mean_absolute_error(y[:, 0], pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = float(np.mean(unc))
print("sigma_opt, sigma_mean:", sigma_opt, sigma_mean)



## === cell 17
sub_out = sub_df.copy()
sub_out["FVC1"] = 0.995 * pe[:, 1]
sub_out["Confidence1"] = pe[:, 2] - pe[:, 0]

subm = sub_out[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()

subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]

if sigma_mean < 70:
    subm["Confidence"] = sigma_opt
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[
        ~subm.FVC1.isnull(), "Confidence1"
    ]

otest = pd.read_csv(f"{ROOT}/test.csv")
otest_pw = otest["Patient"].astype(str) + "_" + otest["Weeks"].astype(str)
fix_df = pd.DataFrame({"Patient_Week": otest_pw.values, "FVC_fix": otest["FVC"].values})
subm = subm.merge(fix_df, on="Patient_Week", how="left")
mask = subm["FVC_fix"].notna()
subm.loc[mask, "FVC"] = subm.loc[mask, "FVC_fix"]
subm.loc[mask, "Confidence"] = 0.1
subm.drop(columns=["FVC_fix"], inplace=True)

subm["Confidence"] = subm["Confidence"].astype(float).clip(lower=0.1)

submission = subm[["Patient_Week", "FVC", "Confidence"]].copy()
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
