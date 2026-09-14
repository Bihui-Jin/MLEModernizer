# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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

import tensorflow as tf

tf.random.set_seed(SEED)
from tensorflow import keras
from tensorflow.keras import layers as L

from scipy.ndimage import zoom
from math import ceil

print("TensorFlow version:", tf.__version__)



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

rename_cols = {
    "Weeks_y": "Base_week",
    "Weeks_x": "Weeks",
    "Percent": "Base_percent",
    "FVC": "Base_FVC",
}
X_prediction = (
    X_prediction.merge(raw_test, how="left", left_on="Patient", right_on="Patient")
    .rename(columns=rename_cols)[
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
    ]
    .reset_index(drop=True)
)



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


def _get_sorted_patient_files(path, patient):
    key = (path, patient)
    files = _PATIENT_FILES_CACHE.get(key)
    if files is None:
        files = sorted(os.listdir(os.path.join(path, patient)), key=sort_function)
        _PATIENT_FILES_CACHE[key] = files
    return files


def _read(path, patients=[], desired_size=(60, 512, 512)):
    import pydicom

    patients = list(patients)
    X = np.empty(
        (len(patients), 1, desired_size[0], desired_size[1], desired_size[2]),
        dtype=np.float32,
    )

    for i, patient in enumerate(patients):
        key = (path, patient, desired_size)
        arr = _VOL_CACHE.get(key)
        if arr is None:
            list_patient_files = _get_sorted_patient_files(path, patient)
            folder = os.path.join(path, patient)

            stack = []
            for f in list_patient_files:
                fp = os.path.join(folder, f)
                ds = pydicom.dcmread(fp)  # single read; pixel_array triggers decode
                stack.append(ds.pixel_array)

            arr = np.asarray(stack, dtype=np.float32)

            arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-6)

            factors = np.asarray(desired_size, dtype=np.float32) / np.asarray(
                arr.shape, dtype=np.float32
            )
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


class DataGenerator(keras.utils.Sequence):
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

        patients_in_batch, inv = np.unique(batch_patients, return_inverse=True)

        imgs = _read(
            self.img_path, patients=patients_in_batch, desired_size=self.desired_size
        )
        imgs = np.transpose(imgs, (0, 2, 3, 4, 1))  # [P,D,H,W,1]

        tab = self._tab[row_ids]  # [B, n_tab]
        return batch_patients, tab, imgs, inv




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

use_external = _has_saved_model(cnn_path) and _has_saved_model(mlp_path)

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
    latent_dim = 64
    CNN = keras.Sequential(
        [
            L.Input(shape=(DESIRED_SIZE[0], DESIRED_SIZE[1], DESIRED_SIZE[2], 1)),
            L.GlobalAveragePooling3D(),
            L.Dense(latent_dim, activation="relu"),
        ],
        name="fallback_cnn",
    )
    model = keras.Sequential(
        [
            L.Input(shape=(latent_dim + len(temp_SELECTED_COLUMNS),)),
            L.Dense(64, activation="relu"),
            L.Dense(3, activation=None),
        ],
        name="fallback_mlp",
    )
    _ = CNN(
        tf.zeros(
            (1, DESIRED_SIZE[0], DESIRED_SIZE[1], DESIRED_SIZE[2], 1), dtype=tf.float32
        )
    )
    _ = model(tf.zeros((1, latent_dim + len(temp_SELECTED_COLUMNS)), dtype=tf.float32))
    print("External SavedModels not found; using deterministic fallback models.")



## === cell 10
all_test_patients = pd.Index(X_prediction["Patient"].values).unique().to_list()
_ = _read(TEST_PATH, patients=all_test_patients, desired_size=DESIRED_SIZE)

y_prediction_chunks = []

for batch_patients, tab, imgs, inv in pred_generator:
    X2_tf = tf.convert_to_tensor(imgs, dtype=tf.float32)

    out_imgs = _tfsmlayer_call(CNN, X2_tf)
    out_imgs_np = out_imgs.numpy()  # [P, latent]

    X_imgs = out_imgs_np[inv]  # [B, latent] aligned to rows

    mlp_inp = np.concatenate([X_imgs, tab], axis=1)

    out = _tfsmlayer_call(
        model, tf.convert_to_tensor(mlp_inp, dtype=tf.float32)
    ).numpy()
    y_prediction_chunks.append(out)

y_prediction = np.concatenate(y_prediction_chunks, axis=0)

assert y_prediction.shape[0] == X_prediction.shape[0], (
    y_prediction.shape,
    X_prediction.shape,
)



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
