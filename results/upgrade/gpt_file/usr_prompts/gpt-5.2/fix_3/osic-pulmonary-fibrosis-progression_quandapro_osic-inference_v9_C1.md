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

-6.976746592794775

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
import random
import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
MIN_MAX = {
    "Weeks": (-5.0, 133.0),
    "FVC": (827.0, 6399.0),
    "Percent": (28.877577, 153.145378),
    "Age": (49.0, 88.0),
    "typical_fvc": (827.0, 6399.0),
}



## === cell 2
IMG_SIZE = 128
NUM_OF_SCANS = 12
BATCH_SIZE = 4  # as provided
EPOCHS = 3

INPUT_ROOT = "../input/osic-pulmonary-fibrosis-progression"
TRAIN_CSV = os.path.join(INPUT_ROOT, "train.csv")
TEST_CSV = os.path.join(INPUT_ROOT, "test.csv")
SAMPLE_CSV = os.path.join(INPUT_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(INPUT_ROOT, "train")
TEST_IMG_DIR = os.path.join(INPUT_ROOT, "test")

TRAIN_DF = pd.read_csv(TRAIN_CSV)
TEST_DF = pd.read_csv(TEST_CSV)
SAMPLE_SUBMISSION = pd.read_csv(SAMPLE_CSV)

training_features = [
    "Weeks",
    "min_week",
    "min_week_FVC",
    "typical_fvc",
    "Age",
    "Male",
    "Female",
    "Never smoked",
    "Currently smokes",
    "Ex-smoker",
]
num_of_features = len(training_features)

print(
    "train/test/sample shapes:", TRAIN_DF.shape, TEST_DF.shape, SAMPLE_SUBMISSION.shape
)




## === cell 3
def create_typical_fvc(df):
    df = df.copy()
    df["typical_fvc"] = df["FVC"] / df["Percent"] * 100.0
    return df


def normalize(df):
    df = df.copy()
    for feature, min_max in MIN_MAX.items():
        df[feature] = (df[feature] - min_max[0]) / (min_max[1] - min_max[0])
    return df


def add_onehots_and_minweek(df):
    df = df.copy()
    df = create_typical_fvc(df)

    df["Never smoked"] = (df["SmokingStatus"] == "Never smoked").astype("uint8")
    df["Currently smokes"] = (df["SmokingStatus"] == "Currently smokes").astype("uint8")
    df["Ex-smoker"] = (df["SmokingStatus"] == "Ex-smoker").astype("uint8")

    df["Male"] = (df["Sex"] == "Male").astype("uint8")
    df["Female"] = (df["Sex"] == "Female").astype("uint8")

    df["min_week"] = 0.0
    df["min_week_FVC"] = 0.0

    for patient in df["Patient"].unique():
        m = df["Patient"] == patient
        min_w = df.loc[m, "Weeks"].min()
        base_fvc = df.loc[m].sort_values("Weeks")["FVC"].iloc[0]
        df.loc[m, "min_week"] = min_w
        df.loc[m, "min_week_FVC"] = base_fvc

    df = normalize(df)
    return df


TRAIN_DF_PROC = add_onehots_and_minweek(TRAIN_DF)
TEST_DF_PROC = add_onehots_and_minweek(TEST_DF)

print("Processed shapes:", TRAIN_DF_PROC.shape, TEST_DF_PROC.shape)




## === cell 4
def read_dicom_pixel_array(path):
    try:
        import pydicom

        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array
        return arr
    except Exception:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            with open(path, "rb") as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read DICOM image at {path}.")
        return img


def window_and_scale_to_uint8(image, window_center=-200, window_width=2000):
    image = image.astype(np.int16, copy=False)
    image_min = window_center - window_width // 2
    image_max = window_center + window_width // 2
    image = np.clip(image, image_min, image_max).astype(np.float32)
    image = (image - image_min) / (image_max - image_min) * 255.0
    return image.astype(np.uint8)


def load_patient_volume(
    patient_id, base_dir, num_scans=NUM_OF_SCANS, img_size=IMG_SIZE
):
    folder = os.path.join(base_dir, patient_id)
    files = sorted(
        os.listdir(folder),
        key=lambda x: (
            int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x
        ),
    )
    files = np.asarray(files)
    if len(files) == 0:
        raise RuntimeError(f"No DICOM files found for patient {patient_id} in {folder}")

    mid = len(files) // 2
    start = max(0, mid - num_scans // 2)
    end = min(len(files), start + num_scans)
    sel = files[start:end]

    if len(sel) < num_scans:
        sel = list(sel) + [sel[-1]] * (num_scans - len(sel))

    images = []
    for fn in sel:
        path = os.path.join(folder, fn)
        arr = read_dicom_pixel_array(path)
        arr = window_and_scale_to_uint8(arr)
        arr = cv2.resize(arr, (img_size, img_size), interpolation=cv2.INTER_AREA)
        images.append(arr)

    vol = np.asarray(images, dtype=np.float32)
    return vol


def existing_patients(df_proc, img_dir):
    patients = df_proc["Patient"].unique()
    ok = []
    for p in patients:
        folder = os.path.join(img_dir, p)
        if os.path.isdir(folder):
            ok.append(p)
    return set(ok)


train_ok = existing_patients(TRAIN_DF_PROC, TRAIN_IMG_DIR)
test_ok = existing_patients(TEST_DF_PROC, TEST_IMG_DIR)

TRAIN_DF_PROC = TRAIN_DF_PROC[TRAIN_DF_PROC["Patient"].isin(train_ok)].reset_index(
    drop=True
)
TEST_DF_PROC = TEST_DF_PROC[TEST_DF_PROC["Patient"].isin(test_ok)].reset_index(
    drop=True
)

print("Patients with images:", len(train_ok), len(test_ok))
print("Filtered processed shapes:", TRAIN_DF_PROC.shape, TEST_DF_PROC.shape)

train_volumes = {}
for pid in TRAIN_DF_PROC["Patient"].unique():
    train_volumes[pid] = load_patient_volume(pid, TRAIN_IMG_DIR)

test_volumes = {}
for pid in TEST_DF_PROC["Patient"].unique():
    test_volumes[pid] = load_patient_volume(pid, TEST_IMG_DIR)

print("Loaded volumes:", len(train_volumes), len(test_volumes))




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3090088600.py in read_dicom_pixel_array(path)
      7         ds = pydicom.dcmread(path, force=True)
----> 8         arr = ds.pixel_array
      9         return arr

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in pixel_array(self)
   2192         """
-> 2193         self.convert_pixel_data()
   2194         return cast("numpy.ndarray", self._pixel_array)

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in convert_pixel_data(self, handler_name)
   1725             opts["decoding_plugin"] = name
-> 1726             self._pixel_array = pixel_array(self, **opts)
   1727             self._pixel_id = get_image_pixel_ids(self)

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py in pixel_array(src, ds_out, specific_tags, index, raw, decoding_plugin, **kwargs)
   1429         opts = as_pixel_options(ds, **kwargs)
-> 1430         return decoder.as_array(
   1431             ds,

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in as_array(self, src, index, validate, raw, decoding_plugin, **kwargs)
    981                 dict[str, "DecodeFunction"],
--> 982                 self._validate_plugins(decoding_plugin),
    983             ),

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/common.py in _validate_plugins(self, plugin)
    256         if self._decoder:
--> 257             raise RuntimeError(
    258                 f"Unable to decompress '{self.UID.name}' pixel data because all "

RuntimeError: Unable to decompress 'JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])' pixel data because all plugins are missing dependencies:
	gdcm - requires gdcm>=3.0.10
	pylibjpeg - requires pylibjpeg>=2.0 and pylibjpeg-libjpeg>=2.1

During handling of the above exception, another exception occurred:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3090088600.py in <cell line: 0>()
     90 train_volumes = {}
     91 for pid in TRAIN_DF_PROC["Patient"].unique():
---> 92     train_volumes[pid] = load_patient_volume(pid, TRAIN_IMG_DIR)
     93 
     94 test_volumes = {}

/tmp/ipykernel_11/3090088600.py in load_patient_volume(patient_id, base_dir, num_scans, img_size)
     54     for fn in sel:
     55         path = os.path.join(folder, fn)
---> 56         arr = read_dicom_pixel_array(path)
     57         arr = window_and_scale_to_uint8(arr)
     58         arr = cv2.resize(arr, (img_size, img_size), interpolation=cv2.INTER_AREA)

/tmp/ipykernel_11/3090088600.py in read_dicom_pixel_array(path)
     16             img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
     17         if img is None:
---> 18             raise RuntimeError(f"Failed to read DICOM image at {path}.")
     19         return img
     20 

RuntimeError: Failed to read DICOM image at ../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/10.dcm.

## === cell 5
class TrainDataset(Sequence):
    def __init__(self, df_proc, volumes_dict, batch_size=BATCH_SIZE, shuffle=True):
        self.df = df_proc.reset_index(drop=True)
        self.volumes = volumes_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_idx = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        bdf = self.df.iloc[batch_idx]

        imgs = (
            np.asarray(
                [self.volumes[p] for p in bdf["Patient"].values], dtype=np.float32
            )
            / 255.0
        )
        imgs = np.expand_dims(imgs, axis=-1)

        tabs = bdf[training_features].values.astype(np.float32)

        y = bdf["FVC"].values.astype(np.float32)
        y3 = np.stack([y, y, y], axis=1)
        return [imgs, tabs], y3


class TestDataset(Sequence):
    def __init__(self, sample_sub, test_df_proc, volumes_dict, batch_size=BATCH_SIZE):
        self.sample = sample_sub.reset_index(drop=True)
        self.test_df = test_df_proc
        self.volumes = volumes_dict
        self.batch_size = batch_size
        self.indices = np.arange(len(self.sample))

        self._tab_cache = {}
        for p in self.test_df["Patient"].unique():
            self._tab_cache[p] = self.test_df[self.test_df["Patient"] == p][
                training_features[1:]
            ].values[0]

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def get_tabular(self, patient, week):
        week = (float(week) - MIN_MAX["Weeks"][0]) / (
            MIN_MAX["Weeks"][1] - MIN_MAX["Weeks"][0]
        )
        tabular = [week]
        tabular += list(self._tab_cache[patient])
        return np.asarray(tabular, dtype="float32")

    def __getitem__(self, index):
        batch_idx = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        Patient_Week = self.sample.loc[batch_idx, "Patient_Week"].values
        patient = [x.split("_")[0] for x in Patient_Week]
        week = [x.split("_")[1] for x in Patient_Week]

        images = (
            np.asarray([self.volumes[pid] for pid in patient], dtype=np.float32) / 255.0
        )
        images = np.expand_dims(images, axis=-1)

        tabulars = np.asarray(
            [self.get_tabular(patient[i], week[i]) for i in range(len(patient))],
            dtype="float32",
        )
        return [images, tabulars]




## === cell 6
def swish(x):
    return x * K.sigmoid(x)


def conv_block(x, num_of_filters):
    x = Conv3D(
        num_of_filters,
        kernel_size=(3, 1, 1),
        padding="same",
        kernel_initializer="he_uniform",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv3D(
        num_of_filters,
        kernel_size=(1, 3, 3),
        padding="same",
        kernel_initializer="he_uniform",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv3D(
        num_of_filters,
        kernel_size=(1, 1, 1),
        padding="same",
        kernel_initializer="he_uniform",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def residual_block(x, num_of_filters):
    x1 = conv_block(x, num_of_filters)
    x2 = conv_block(x1, num_of_filters)
    return Add()([x1, x2])


def build_3d_resnet(input_tensor):
    c1 = Conv3D(64, kernel_size=(5, 7, 7), strides=(1, 2, 2), padding="same")(
        input_tensor
    )
    b1 = BatchNormalization()(c1)
    a1 = Activation("relu")(b1)
    p1 = MaxPool3D(pool_size=(3, 3, 3), strides=(1, 2, 2))(a1)

    r1 = residual_block(p1, 64)
    r1 = residual_block(r1, 64)
    p1 = MaxPool3D(pool_size=(3, 3, 3), strides=(1, 2, 2))(r1)

    r2 = residual_block(p1, 128)
    r2 = residual_block(r2, 128)
    p2 = MaxPool3D(pool_size=(3, 3, 3), strides=(1, 2, 2))(r2)

    r3 = residual_block(p2, 256)
    r3 = residual_block(r3, 256)

    return r3


def build_model():
    input_img = Input(shape=(NUM_OF_SCANS, IMG_SIZE, IMG_SIZE, 1))
    x = build_3d_resnet(input_img)
    input_latent = GlobalAveragePooling3D()(x)

    input_tabular = Input(shape=(num_of_features,))
    x = Concatenate()([input_latent, input_tabular])
    x = Dense(100)(x)
    x = Dense(100)(x)
    out = Dense(3)(x)

    model = tf.keras.Model(
        inputs=[input_img, input_tabular], outputs=out, name="tabular_model"
    )
    return model




## === cell 7
model = build_model()
model.compile(optimizer=Adam(1e-3), loss="mse")

train_gen = TrainDataset(
    TRAIN_DF_PROC, train_volumes, batch_size=BATCH_SIZE, shuffle=True
)
history = model.fit(train_gen, epochs=EPOCHS, verbose=1)

gc.collect()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/4267008495.py in <cell line: 0>()
      5     TRAIN_DF_PROC, train_volumes, batch_size=BATCH_SIZE, shuffle=True
      6 )
----> 7 history = model.fit(train_gen, epochs=EPOCHS, verbose=1)
      8 
      9 gc.collect()

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/238383844.py in __getitem__(self, index)
     23         imgs = (
     24             np.asarray(
---> 25                 [self.volumes[p] for p in bdf["Patient"].values], dtype=np.float32
     26             )
     27             / 255.0

/tmp/ipykernel_11/238383844.py in <listcomp>(.0)
     23         imgs = (
     24             np.asarray(
---> 25                 [self.volumes[p] for p in bdf["Patient"].values], dtype=np.float32
     26             )
     27             / 255.0

KeyError: 'ID00307637202282126172865'

## === cell 8
test_gen = TestDataset(
    SAMPLE_SUBMISSION, TEST_DF_PROC, test_volumes, batch_size=BATCH_SIZE
)
predictions = model.predict(test_gen, verbose=1)

if predictions.shape[0] != len(SAMPLE_SUBMISSION):
    raise RuntimeError(
        f"Prediction rows {predictions.shape[0]} != sample rows {len(SAMPLE_SUBMISSION)}"
    )




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3760977147.py in <cell line: 0>()
      1 test_gen = TestDataset(
----> 2     SAMPLE_SUBMISSION, TEST_DF_PROC, test_volumes, batch_size=BATCH_SIZE
      3 )
      4 predictions = model.predict(test_gen, verbose=1)
      5 

NameError: name 'test_volumes' is not defined

## === cell 9
def denormalize(y):
    return y * (MIN_MAX["FVC"][1] - MIN_MAX["FVC"][0]) + MIN_MAX["FVC"][0]


predictions = denormalize(predictions.astype(np.float32))
FVC = predictions[:, 1]
Confidence = predictions[:, 2] - predictions[:, 0]
Confidence = np.maximum(np.abs(Confidence), 70.0)

SAMPLE_SUBMISSION["FVC"] = np.round(FVC).astype(int)
SAMPLE_SUBMISSION["Confidence"] = np.round(Confidence).astype(int)

SAMPLE_SUBMISSION = SAMPLE_SUBMISSION[["Patient_Week", "FVC", "Confidence"]]
SAMPLE_SUBMISSION.to_csv("submission.csv", index=False)

print(SAMPLE_SUBMISSION.head())
print("Wrote submission.csv with shape:", SAMPLE_SUBMISSION.shape)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2767396405.py in <cell line: 0>()
      3 
      4 
----> 5 predictions = denormalize(predictions.astype(np.float32))
      6 FVC = predictions[:, 1]
      7 Confidence = predictions[:, 2] - predictions[:, 0]

NameError: name 'predictions' is not defined
