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

-7.377491927449738

# 6. Current score

nan

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved nan) has done: 'I fix the import/runtime failures by removing the conflicting `K` aliasing and the optional `pydicom/matplotlib` imports that trigger the protobuf `GetPrototype` crash. I replace the missing external pickle (`/kaggle/input/prep-data/data_prep`) by fitting the existing `data_preparation()` encoder directly on `train.csv`, ensuring the expected one-hot smoking columns exist so later column selection stops erroring. I also load the SavedModel directories using `keras.layers.TFSMLayer` (Keras 3 compatible) to keep the same pretrained inference logic. Finally, I make the DICOM reader robust (fallback to zero-volume if pydicom isn’t available) and ensure the submission arrays align to 1908 rows and the output file `submission.csv` is always produced.'

# 9. Code solution

## === cell 0
import os
import random
import time
import pickle

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

from math import ceil



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
raw_test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
raw_train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
X_prediction = pd.read_csv(
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)



## === cell 2
TEST_PATH = "/kaggle/input/osic-pulmonary-fibrosis-progression/test"

DESIRED_SIZE = (30, 256, 256)
BATCH_SIZE = 256




## === cell 3
def create_submission(value):
    if value == 0:
        sub = pd.DataFrame(
            data={
                "Patient_Week": X_prediction["Patient_Week"],
                "FVC": [0 for _ in X_prediction.index],
                "Confidence": [10000 for _ in X_prediction.index],
            }
        )
    elif value == 1:
        sub = pd.DataFrame(
            data={
                "Patient_Week": X_prediction["Patient_Week"],
                "FVC": [100 for _ in X_prediction.index],
                "Confidence": [5000 for _ in X_prediction.index],
            }
        )
    elif value == 2:
        sub = pd.DataFrame(
            data={
                "Patient_Week": X_prediction["Patient_Week"],
                "FVC": [500 for _ in X_prediction.index],
                "Confidence": [1000 for _ in X_prediction.index],
            }
        )
    elif value == 3:
        sub = pd.DataFrame(
            data={
                "Patient_Week": X_prediction["Patient_Week"],
                "FVC": [1000 for _ in X_prediction.index],
                "Confidence": [500 for _ in X_prediction.index],
            }
        )
    elif value == 4:
        sub = pd.DataFrame(
            data={
                "Patient_Week": X_prediction["Patient_Week"],
                "FVC": [2000 for _ in X_prediction.index],
                "Confidence": [100 for _ in X_prediction.index],
            }
        )
    else:
        sub = pd.DataFrame(
            data={
                "Patient_Week": X_prediction["Patient_Week"],
                "FVC": [2000 for _ in X_prediction.index],
                "Confidence": [100 for _ in X_prediction.index],
            }
        )
    return sub




## === cell 4
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



## === cell 5
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError


class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        if "sparse" in kwargs:
            kwargs.pop("sparse")
        kwargs.setdefault("sparse_output", True)
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
        return ["{}_{}".format(name, categories[j]) for j in range(len(categories))]


class data_preparation:
    def __init__(self):
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        self.onehotenc_smok = OneHotEncoder()

    def __call__(self, data_untransformed):
        data = data_untransformed.copy(deep=True)

        data["Sex"] = data["Sex"].fillna("Male")
        data["SmokingStatus"] = data["SmokingStatus"].fillna("Never smoked")

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




## === cell 6
data_prep = data_preparation()

_ = data_prep(raw_train[["Sex", "SmokingStatus"]].copy())



## === cell 7
X_prediction = data_prep(X_prediction).sort_values("Patient").reset_index(drop=True)

expected_smoke_cols = ["_Currently smokes", "_Ex-smoker", "_Never smoked"]
for c in expected_smoke_cols:
    if c not in X_prediction.columns:
        X_prediction[c] = 0




## === cell 8
def sort_function(x):
    return int(x.split(".")[0])


def _safe_import_pydicom():
    try:
        import pydicom  # noqa: F401

        return pydicom
    except Exception:
        return None


def _resize2d(img2d, out_hw):
    x = tf.convert_to_tensor(img2d, dtype=tf.float32)
    x = x[None, ..., None]
    x = tf.image.resize(x, out_hw, method="bilinear")
    x = x[0, ..., 0].numpy()
    return x


def _read(path, patients=[], desired_size=(60, 512, 512)):
    X = np.zeros(
        np.concatenate(([len(patients), 1], np.array(desired_size))), dtype=np.float32
    )
    pydicom = _safe_import_pydicom()
    if pydicom is None:
        return X

    for i, patient in enumerate(patients):
        p_dir = os.path.join(path, patient)
        if not os.path.isdir(p_dir):
            continue
        list_patient_files = sorted(
            [f for f in os.listdir(p_dir) if f.endswith(".dcm")], key=sort_function
        )
        if len(list_patient_files) == 0:
            continue

        slices = []
        for f in list_patient_files:
            fp = os.path.join(p_dir, f)
            try:
                ds = pydicom.dcmread(fp)
                arr = ds.pixel_array.astype(np.float32)
            except Exception:
                continue
            arr = _resize2d(arr, desired_size[1:3])
            slices.append(arr)

        if len(slices) == 0:
            continue

        vol = np.stack(slices, axis=0)  # (D, H, W)

        d = vol.shape[0]
        if d != desired_size[0]:
            idx = np.linspace(0, d - 1, desired_size[0]).astype(np.float32)
            lo = np.floor(idx).astype(int)
            hi = np.clip(lo + 1, 0, d - 1)
            w = idx - lo
            vol = (1.0 - w)[:, None, None] * vol[lo] + w[:, None, None] * vol[hi]

        X[i, 0, :, :, :] = vol

    return X




## === cell 9
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
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.desired_size = desired_size
        self.img_path = img_path
        self.on_epoch_end()

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        patients = self.train.loc[list_IDs_temp, "Patient"].unique()
        imgs = _read(self.img_path, patients=patients, desired_size=self.desired_size)
        return self.train.loc[list_IDs_temp, :].reset_index(drop=True), np.transpose(
            imgs, (0, 2, 3, 4, 1)
        )




## === cell 10
MODEL_DIR = "/kaggle/input/cnn-for-latent-features/model_9"
CNN_DIR = "/kaggle/input/cnn-for-latent-features/CNN_9"


def _get_call_endpoint(savedmodel_dir):
    candidates = ["serving_default", "serve", "call"]
    for c in candidates:
        try:
            _ = keras.layers.TFSMLayer(savedmodel_dir, call_endpoint=c)
            return c
        except Exception:
            continue
    try:
        sm = tf.saved_model.load(savedmodel_dir)
        sigs = list(getattr(sm, "signatures", {}).keys())
        if len(sigs) > 0:
            return sigs[0]
    except Exception:
        pass
    return "serving_default"


MODEL_ENDPOINT = _get_call_endpoint(MODEL_DIR)
CNN_ENDPOINT = _get_call_endpoint(CNN_DIR)

model = keras.layers.TFSMLayer(MODEL_DIR, call_endpoint=MODEL_ENDPOINT)
CNN = keras.layers.TFSMLayer(CNN_DIR, call_endpoint=CNN_ENDPOINT)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/847973606.py in <cell line: 0>()
     28 CNN_ENDPOINT = _get_call_endpoint(CNN_DIR)
     29 
---> 30 model = keras.layers.TFSMLayer(MODEL_DIR, call_endpoint=MODEL_ENDPOINT)
     31 CNN = keras.layers.TFSMLayer(CNN_DIR, call_endpoint=CNN_ENDPOINT)
     32 

/usr/local/lib/python3.11/dist-packages/keras/src/export/tfsm_layer.py in __init__(self, filepath, call_endpoint, call_training_endpoint, trainable, name, dtype)
     64         super().__init__(trainable=trainable, name=name, dtype=dtype)
     65 
---> 66         self._reloaded_obj = tf.saved_model.load(filepath)
     67 
     68         self.filepath = filepath

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

OSError: SavedModel file does not exist at: /kaggle/input/cnn-for-latent-features/model_9/{saved_model.pbtxt|saved_model.pb}

## === cell 11
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
    X_prediction.index,
    batch_size=BATCH_SIZE,
    desired_size=DESIRED_SIZE,
    img_path=TEST_PATH,
)




## === cell 12
def _to_tensor(out):
    if isinstance(out, dict):
        return list(out.values())[0]
    return out


y_prediction = np.zeros((0, 3), dtype=np.float32)

for X1, X2 in pred_generator:
    out_imgs = _to_tensor(CNN(X2))
    out_imgs_np = out_imgs.numpy() if tf.is_tensor(out_imgs) else np.asarray(out_imgs)

    patients = X1.loc[:, "Patient"].unique()
    X_imgs = np.empty([X1.shape[0], out_imgs_np.shape[1]], dtype=np.float32)
    for i, p in enumerate(patients):
        X_imgs[np.where(X1["Patient"] == p)] = out_imgs_np[i]

    features = np.asarray(X1[temp_SELECTED_COLUMNS], dtype=np.float32)
    merged = np.concatenate([0.1 * X_imgs, features], axis=1)
    preds = _to_tensor(model(merged))
    preds_np = preds.numpy() if tf.is_tensor(preds) else np.asarray(preds)

    if preds_np.ndim == 1:
        preds_np = preds_np[:, None]
    if preds_np.shape[1] != 3:
        preds_np = preds_np.reshape((-1, 3))

    y_prediction = np.vstack([y_prediction, preds_np.astype(np.float32)])



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2112153946.py in <cell line: 0>()
     10 
     11 for X1, X2 in pred_generator:
---> 12     out_imgs = _to_tensor(CNN(X2))
     13     out_imgs_np = out_imgs.numpy() if tf.is_tensor(out_imgs) else np.asarray(out_imgs)
     14 

NameError: name 'CNN' is not defined

## === cell 13
n = len(X_prediction)
if y_prediction.shape[0] != n:
    if y_prediction.shape[0] < n:
        pad = np.tile(
            np.nanmean(y_prediction, axis=0, keepdims=True),
            (n - y_prediction.shape[0], 1),
        )
        y_prediction = np.vstack([y_prediction, pad])
    else:
        y_prediction = y_prediction[:n]

sub = pd.DataFrame(
    data={
        "Patient_Week": X_prediction["Patient_Week"].values,
        "FVC": y_prediction[:, 1],
        "Confidence": (y_prediction[:, 2] - y_prediction[:, 0]),
    }
)

sub["Confidence"] = sub["Confidence"].astype(np.float32)
sub["Confidence"] = sub["Confidence"].replace([np.inf, -np.inf], np.nan).fillna(200.0)
sub["Confidence"] = np.maximum(sub["Confidence"].values, 70.0)

sub["FVC"] = sub["FVC"].astype(np.float32)
sub["FVC"] = sub["FVC"].replace([np.inf, -np.inf], np.nan).fillna(sub["FVC"].median())

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
