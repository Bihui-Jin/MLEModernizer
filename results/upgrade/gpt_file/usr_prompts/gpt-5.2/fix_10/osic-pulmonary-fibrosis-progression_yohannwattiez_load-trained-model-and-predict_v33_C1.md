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

-7.016106790691332

# 6. Current score

-9.90201

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -25.66776) has done: 'The timeout is dominated by DICOM I/O + 3D resampling: your script preloads and resizes *all* test patient volumes up front, then reads them again inside the generator, and the `zoom` call runs at full 30×256×256 for every patient. I remove the redundant full preload, keep your caching but make it effective (avoid duplicate work across batches), and speed up volume reading by using a fast DICOM path (`stop_before_pixels=True` + `pydicom.pixels.pixel_array`) and precomputing resize factors. I also reduce Python overhead in the generator by avoiding `np.unique` sorting cost and returning only what’s needed without changing the model calls or outputs. These changes preserve identical evaluation semantics (same inputs, same models, same computations) while cutting the heavy constant-factor costs that cause the 10-minute timeout.'
- What this solution (achieved -23.69527) has done: 'I fix the TensorFlow import crash by making TensorFlow an optional dependency and using a deterministic NumPy fallback path when TF can’t be imported in this environment. I also fix the DICOM reading bug by removing the incorrect `stop_before_pixels=True` usage (it prevents pixel decoding) and reading pixel data correctly with `pydicom.dcmread(...).pixel_array`. Finally, I keep your core inference structure (CNN features + MLP output semantics) but implement a lightweight deterministic “fallback CNN+MLP” using NumPy so the notebook runs end-to-end and writes a valid `submission.csv` without relying on broken TF/protobuf; this should substantially improve score versus the current broken/invalid run while remaining stable within the time limit.'
- What this solution (achieved -9.90201) has done: 'I fix the crash in the TensorFlow import block by catching the protobuf `AttributeError` and reliably falling back to the NumPy path (so the pipeline runs end-to-end and always writes `submission.csv`). I also fix a merge/rename logic bug that can produce missing `Base_week`/`Weeks` columns by explicitly renaming the test-week column before merging, preserving the intended feature semantics. Finally, to move the score toward the target (your current score is far worse than target), I make a minimal, metric-aligned improvement in the fallback path only: replace the random fallback “model” outputs with a simple per-patient linear extrapolation fitted from train (plus a robust confidence derived from train residuals), without changing any TF/SavedModel behavior when available.'

# 9. Code solution

## === cell 0
import os
import random
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

tf = None
keras = None
L = None
TF_AVAILABLE = False

try:
    import tensorflow as tf  # noqa: F401

    TF_AVAILABLE = True
    tf.random.set_seed(SEED)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

    from tensorflow import keras  # noqa: E402
    from tensorflow.keras import layers as L  # noqa: E402

    print("TensorFlow version:", tf.__version__)
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    keras = None
    L = None
    print("TensorFlow unavailable; using NumPy fallback. Root error:", repr(e))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
raw_test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
X_prediction = pd.read_csv(
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)



## === cell 2
TEST_PATH = "/kaggle/input/osic-pulmonary-fibrosis-progression/test"

DESIRED_SIZE = (30, 256, 256)
BATCH_SIZE = 256



## === cell 3
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)
X_prediction = X_prediction[["Patient", "Weeks", "Patient_Week"]]

raw_test_base = raw_test.rename(
    columns={
        "Weeks": "Base_week",
        "Percent": "Base_percent",
        "FVC": "Base_FVC",
    }
)

X_prediction = X_prediction.merge(raw_test_base, how="left", on="Patient")[
    [
        "Patient",
        "Base_week",
        "Base_FVC",
        "Base_percent",
        "Age",
        "Sex",
        "SmokingStatus",
        "Weeks",
        "Patient_Week",
    ]
].reset_index(drop=True)



## === cell 4
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError


class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, categories, index="", name="", **kwargs):
        sparse_matrix = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_new_columns(X=X, name=name, categories=categories)
        d_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=index)
        return d_out

    def fit_transform(self, X, categories, index, name, **kwargs):
        self.fit(X)
        return self.transform(X, categories=categories, index=index, name=name)

    def get_new_columns(self, X, name, categories):
        new_columns = []
        for j in range(len(categories)):
            new_columns.append("{}_{}".format(name, categories[j]))
        return new_columns


class data_preparation:
    def __init__(self):
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        self.onehotenc_smok = OneHotEncoder()

    def __call__(self, data_untransformed):
        data = data_untransformed.copy(deep=True)

        try:
            data["Sex"] = self.enc_sex.transform(data["Sex"].values)
            data["SmokingStatus"] = self.enc_smok.transform(
                data["SmokingStatus"].values
            )
            data = pd.concat(
                [
                    data.drop(columns=["SmokingStatus"]),
                    self.onehotenc_smok.transform(
                        data["SmokingStatus"].values.reshape(-1, 1),
                        categories=self.enc_smok.classes_,
                        name="",
                        index=data.index,
                    ).astype(int),
                ],
                axis=1,
            )
        except NotFittedError:
            data["Sex"] = self.enc_sex.fit_transform(data["Sex"].values)
            data["SmokingStatus"] = self.enc_smok.fit_transform(
                data["SmokingStatus"].values
            )
            data = pd.concat(
                [
                    data.drop(columns=["SmokingStatus"]),
                    self.onehotenc_smok.fit_transform(
                        data["SmokingStatus"].values.reshape(-1, 1),
                        categories=self.enc_smok.classes_,
                        name="",
                        index=data.index,
                    ).astype(int),
                ],
                axis=1,
            )

        return data




## === cell 5
data_prep = data_preparation()

raw_train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
_fit_df = pd.concat(
    [
        raw_train[["Sex", "SmokingStatus"]].dropna(),
        raw_test[["Sex", "SmokingStatus"]].dropna(),
    ],
    axis=0,
    ignore_index=True,
)

_dummy = pd.DataFrame(
    {"Sex": _fit_df["Sex"].values, "SmokingStatus": _fit_df["SmokingStatus"].values}
)
data_prep(_dummy)

X_prediction = data_prep(X_prediction).sort_values("Patient").reset_index(drop=True)

for col in ["_Currently smokes", "_Ex-smoker", "_Never smoked"]:
    if col not in X_prediction.columns:
        X_prediction[col] = 0




## === cell 6
def sort_function(x):
    return int(x.split(".")[0])


_VOL_CACHE = {}  # (path, patient, desired_size) -> np.ndarray [D,H,W] float32
_PATIENT_FILES_CACHE = {}  # (path, patient) -> list[str] sorted filenames
_RESIZE_FACTORS_CACHE = {}  # (patient, desired_size) -> factors np.ndarray float32


def _get_sorted_patient_files(path, patient):
    key = (path, patient)
    files = _PATIENT_FILES_CACHE.get(key)
    if files is None:
        files = sorted(os.listdir(os.path.join(path, patient)), key=sort_function)
        _PATIENT_FILES_CACHE[key] = files
    return files


def _read(path, patients=[], desired_size=(60, 512, 512)):
    import pydicom
    from scipy.ndimage import zoom

    patients = list(patients)
    n_pat = len(patients)

    desired_size = tuple(int(x) for x in desired_size)

    X = np.empty(
        (n_pat, 1, desired_size[0], desired_size[1], desired_size[2]),
        dtype=np.float32,
    )

    for i, patient in enumerate(patients):
        key = (path, patient, desired_size)
        arr = _VOL_CACHE.get(key)
        if arr is None:
            list_patient_files = _get_sorted_patient_files(path, patient)
            folder = os.path.join(path, patient)

            fp0 = os.path.join(folder, list_patient_files[0])
            ds0 = pydicom.dcmread(fp0, force=True)
            first = ds0.pixel_array.astype(np.float32, copy=False)

            n_slices = len(list_patient_files)
            stack = np.empty((n_slices,) + first.shape, dtype=np.float32)
            stack[0] = first

            for j in range(1, n_slices):
                fp = os.path.join(folder, list_patient_files[j])
                ds = pydicom.dcmread(fp, force=True)
                stack[j] = ds.pixel_array.astype(np.float32, copy=False)

            arr = stack
            arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-6)

            fkey = (patient, desired_size)
            factors = _RESIZE_FACTORS_CACHE.get(fkey)
            if factors is None:
                factors = np.asarray(desired_size, dtype=np.float32) / np.asarray(
                    arr.shape, dtype=np.float32
                )
                _RESIZE_FACTORS_CACHE[fkey] = factors

            arr = zoom(arr, factors, mode="nearest").astype(np.float32, copy=False)
            _VOL_CACHE[key] = arr

        X[i, 0] = arr

    return X




## === cell 7
temp_SELECTED_COLUMNS = [
    "Weeks",
    "Base_week",
    "Base_FVC",
    "Base_percent",
    "Age",
    "Sex",
    "_Currently smokes",
    "_Ex-smoker",
    "_Never smoked",
]

from math import ceil

if TF_AVAILABLE:
    _BaseSeq = keras.utils.Sequence
else:
    _BaseSeq = object


class DataGenerator(_BaseSeq):
    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_IDs))

    def __len__(self):
        return int(ceil(len(self.indices) / self.batch_size))

    def __init__(
        self,
        train,
        list_IDs,
        batch_size=1,
        desired_size=(10, 512, 512),
        img_path=TEST_PATH,
        *args,
        **kwargs
    ):
        self.train = train
        self.list_IDs = np.asarray(list_IDs, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.desired_size = desired_size
        self.img_path = img_path

        self._patients = self.train["Patient"].to_numpy()
        self._tab = np.asarray(self.train[temp_SELECTED_COLUMNS], dtype=np.float32)

        self.on_epoch_end()

    def __getitem__(self, index):
        idx = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        row_ids = self.list_IDs[idx]

        batch_patients = self._patients[row_ids]

        mapping = {}
        patients_in_batch = []
        inv = np.empty(batch_patients.shape[0], dtype=np.int64)
        for k, p in enumerate(batch_patients):
            pos = mapping.get(p)
            if pos is None:
                pos = len(patients_in_batch)
                mapping[p] = pos
                patients_in_batch.append(p)
            inv[k] = pos

        imgs = _read(
            self.img_path, patients=patients_in_batch, desired_size=self.desired_size
        )
        imgs = np.moveaxis(imgs, 1, -1)  # [P,D,H,W,1]

        tab = self._tab[row_ids]  # [B, n_tab]
        return batch_patients, tab, imgs, inv

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]




## === cell 8
pred_SELECTED_COLUMNS = [
    "Weeks",
    "Base_week",
    "Patient",
    "Base_FVC",
    "Base_percent",
    "Age",
    "Sex",
    "_Currently smokes",
    "_Ex-smoker",
    "_Never smoked",
]

pred_generator = DataGenerator(
    X_prediction[pred_SELECTED_COLUMNS],
    X_prediction.index.tolist(),
    batch_size=BATCH_SIZE,
    desired_size=DESIRED_SIZE,
    img_path=TEST_PATH,
)



## === cell 9
MODEL_DIR = "/kaggle/input/cnn-for-latent-features"


def _has_saved_model(p):
    return os.path.isdir(p) and (
        os.path.exists(os.path.join(p, "saved_model.pb"))
        or os.path.exists(os.path.join(p, "saved_model.pbtxt"))
    )


def _tfsmlayer_call(layer, x):
    out = layer(x)
    if isinstance(out, dict):
        out = out[next(iter(out.keys()))]
    return out


cnn_path = os.path.join(MODEL_DIR, "CNN_0")
mlp_path = os.path.join(MODEL_DIR, "model_0")

use_external = (
    TF_AVAILABLE and _has_saved_model(cnn_path) and _has_saved_model(mlp_path)
)

CNN = None
model = None

if use_external:

    def _get_serving_endpoint(model_path):
        imported = tf.saved_model.load(model_path)
        if "serving_default" in imported.signatures:
            return "serving_default"
        return list(imported.signatures.keys())[0]

    cnn_endpoint = _get_serving_endpoint(cnn_path)
    mlp_endpoint = _get_serving_endpoint(mlp_path)
    CNN = keras.layers.TFSMLayer(cnn_path, call_endpoint=cnn_endpoint)
    model = keras.layers.TFSMLayer(mlp_path, call_endpoint=mlp_endpoint)
    print("Loaded external SavedModels.")
else:
    train_df = raw_train.copy()
    train_df = train_df.dropna(subset=["Patient", "Weeks", "FVC"])
    train_df["Weeks"] = train_df["Weeks"].astype(int)

    slopes = {}
    intercepts = {}
    global_slope = -10.0  # mild decline prior; used if patient history insufficient
    global_intercept_shift = 0.0

    wk = train_df["Weeks"].to_numpy(np.float32)
    fvc = train_df["FVC"].to_numpy(np.float32)
    x = wk - wk.mean()
    denom = float((x * x).sum())
    if denom > 0:
        global_slope = float(((x) * (fvc - fvc.mean())).sum() / denom)
        global_intercept_shift = float(fvc.mean() - global_slope * wk.mean())

    for pid, g in train_df.groupby("Patient", sort=False):
        if len(g) < 2:
            continue
        w = g["Weeks"].to_numpy(np.float32)
        y = g["FVC"].to_numpy(np.float32)
        x = w - w.mean()
        denom = float((x * x).sum())
        if denom <= 0:
            continue
        a = float((x * (y - y.mean())).sum() / denom)  # slope
        b = float(y.mean() - a * w.mean())  # intercept
        slopes[pid] = a
        intercepts[pid] = b

    pred_train = []
    true_train = []
    for pid, g in train_df.groupby("Patient", sort=False):
        w = g["Weeks"].to_numpy(np.float32)
        y = g["FVC"].to_numpy(np.float32)
        if pid in slopes:
            a = slopes[pid]
            b = intercepts[pid]
        else:
            a = global_slope
            b = global_intercept_shift
        pred = a * w + b
        pred_train.append(pred)
        true_train.append(y)
    pred_train = np.concatenate(pred_train).astype(np.float32, copy=False)
    true_train = np.concatenate(true_train).astype(np.float32, copy=False)
    resid = np.abs(true_train - pred_train)
    base_sigma = float(np.median(resid)) if resid.size else 200.0
    base_sigma = float(np.clip(base_sigma, 70.0, 500.0))

    patient_slope = {}
    for pid in X_prediction["Patient"].unique():
        if pid in slopes:
            patient_slope[pid] = slopes[pid]
        else:
            patient_slope[pid] = global_slope

    def fallback_predict(tab, patients):
        weeks = tab[:, 0]
        base_week = tab[:, 1]
        base_fvc = tab[:, 2]
        out = np.empty((tab.shape[0], 3), dtype=np.float32)
        for i, pid in enumerate(patients):
            a = float(patient_slope.get(pid, global_slope))
            mu = float(base_fvc[i] + a * (weeks[i] - base_week[i]))
            sigma = base_sigma + 0.25 * abs(float(weeks[i] - base_week[i]))
            sigma = float(np.clip(sigma, 70.0, 1000.0))

            out[i, 1] = mu
            out[i, 0] = mu - sigma / 2.0
            out[i, 2] = mu + sigma / 2.0
        return out

    print("Using calibrated linear NumPy fallback (TF/SavedModel not available).")



## === cell 10
y_prediction = np.empty((X_prediction.shape[0], 3), dtype=np.float32)

write_pos = 0
for batch_patients, tab, imgs, inv in pred_generator:
    if use_external:
        out_imgs = _tfsmlayer_call(CNN, tf.convert_to_tensor(imgs, dtype=tf.float32))
        out_imgs_np = out_imgs.numpy()  # [P, latent]
        X_imgs = out_imgs_np[inv]  # [B, latent] aligned to rows
        mlp_inp = np.concatenate([X_imgs, tab], axis=1)
        out = _tfsmlayer_call(
            model, tf.convert_to_tensor(mlp_inp, dtype=tf.float32)
        ).numpy()
    else:
        out = fallback_predict(tab.astype(np.float32, copy=False), batch_patients)

    bs = out.shape[0]
    y_prediction[write_pos : write_pos + bs] = out
    write_pos += bs

assert write_pos == X_prediction.shape[0], (write_pos, X_prediction.shape[0])



## === cell 11
pred_fvc = y_prediction[:, 1].astype(np.float32)
pred_conf = (y_prediction[:, 2] - y_prediction[:, 0]).astype(np.float32)
pred_conf = np.clip(pred_conf, 70.0, None)

sub = pd.DataFrame(
    data={
        "Patient_Week": X_prediction["Patient_Week"].values,
        "FVC": pred_fvc,
        "Confidence": pred_conf,
    }
)

sample_sub = pd.read_csv(
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
sub = sample_sub[["Patient_Week"]].merge(sub, on="Patient_Week", how="left")
assert sub.shape[0] == sample_sub.shape[0]
assert list(sub.columns) == ["Patient_Week", "FVC", "Confidence"]

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("FVC range:", float(np.nanmin(sub["FVC"])), float(np.nanmax(sub["FVC"])))
print(
    "Confidence range:",
    float(np.nanmin(sub["Confidence"])),
    float(np.nanmax(sub["Confidence"])),
)
