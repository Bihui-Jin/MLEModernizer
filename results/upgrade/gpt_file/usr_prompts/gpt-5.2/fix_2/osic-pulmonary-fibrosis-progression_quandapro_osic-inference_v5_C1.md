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

-6.913230262117238

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import cv2
import gc
import numpy as np
import os
import pandas as pd
import pydicom as dicom
import random

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *

from sklearn.model_selection import train_test_split

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
NUM_OF_SCANS = 10
BATCH_SIZE = 8  # keep memory safe; does not change core logic
EPOCHS = 8  # minimal training to produce usable weights within time
N_MODELS = 3  # lightweight ensemble to stabilize predictions

DATA_DIR = "../input/osic-pulmonary-fibrosis-progression"

TRAIN_DF = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
TEST_DF = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
SAMPLE_SUBMISSION = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

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

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")




## === cell 3
def create_typical_fvc(df):
    df = df.copy()
    df["typical_fvc"] = df["FVC"] / df["Percent"] * 100.0
    return df


def normalize(df):
    df = df.copy()
    for feature, min_max in MIN_MAX.items():
        if feature in df.columns:
            df[feature] = (df[feature] - min_max[0]) / (min_max[1] - min_max[0])
    return df


def add_onehot(df):
    df = df.copy()
    df["Never smoked"] = (df["SmokingStatus"] == "Never smoked").astype("uint8")
    df["Currently smokes"] = (df["SmokingStatus"] == "Currently smokes").astype("uint8")
    df["Ex-smoker"] = (df["SmokingStatus"] == "Ex-smoker").astype("uint8")
    df["Male"] = (df["Sex"] == "Male").astype("uint8")
    df["Female"] = (df["Sex"] == "Female").astype("uint8")
    return df


TRAIN_DF = create_typical_fvc(TRAIN_DF)
TEST_DF = create_typical_fvc(TEST_DF)

TRAIN_DF = add_onehot(TRAIN_DF)
TEST_DF = add_onehot(TEST_DF)

for df in (TRAIN_DF, TEST_DF):
    df["min_week"] = 0.0
    df["min_week_FVC"] = 0.0
    for p in df["Patient"].unique():
        mask = df["Patient"] == p
        df.loc[mask, "min_week"] = df.loc[mask, "Weeks"].min()
        idxmin = df.loc[mask, "Weeks"].idxmin()
        df.loc[mask, "min_week_FVC"] = df.loc[idxmin, "FVC"]

TRAIN_DF = normalize(TRAIN_DF)
TEST_DF = normalize(TEST_DF)

TRAIN_DF["FVC_norm"] = (TRAIN_DF["FVC"] - MIN_MAX["FVC"][0]) / (
    MIN_MAX["FVC"][1] - MIN_MAX["FVC"][0]
)
TRAIN_DF["y0"] = TRAIN_DF["FVC_norm"] - 0.05
TRAIN_DF["y1"] = TRAIN_DF["FVC_norm"]
TRAIN_DF["y2"] = TRAIN_DF["FVC_norm"] + 0.05

patients = TRAIN_DF["Patient"].unique()
train_pat, val_pat = train_test_split(patients, test_size=0.2, random_state=SEED)

train_idx = TRAIN_DF.index[TRAIN_DF["Patient"].isin(train_pat)].to_numpy()
val_idx = TRAIN_DF.index[TRAIN_DF["Patient"].isin(val_pat)].to_numpy()




## === cell 4
def get_pixels_hu(scan):
    image = scan.pixel_array
    image = image.astype(np.int16)

    slope = getattr(scan, "RescaleSlope", 1)
    intercept = getattr(scan, "RescaleIntercept", 0)
    window_center = -200
    window_width = 2000
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)

    image_min = window_center - window_width // 2
    image_max = window_center + window_width // 2
    image = np.clip(image, image_min, image_max).astype(np.float64)
    image = (image - image_min) / (image_max - image_min) * 255.0
    return image.astype(np.uint8)


def load_patient_volume(patient_id, folder, img_size=IMG_SIZE, num_scans=NUM_OF_SCANS):
    image_folder = os.path.join(folder, patient_id)
    image_files = np.asarray(
        sorted(os.listdir(image_folder), key=lambda x: int(os.path.splitext(x)[0]))
    )
    if len(image_files) >= num_scans:
        mid = len(image_files) // 2
        half = num_scans // 2
        image_files = image_files[mid - half : mid + half]
    scans = [dicom.dcmread(os.path.join(image_folder, f)) for f in image_files]
    imgs = [cv2.resize(get_pixels_hu(s), (img_size, img_size)) for s in scans]
    if len(imgs) < num_scans:
        if len(imgs) == 0:
            imgs = [np.zeros((img_size, img_size), dtype=np.uint8)]
        while len(imgs) < num_scans:
            imgs = imgs + [imgs[-1]]
        imgs = imgs[:num_scans]
    return np.asarray(imgs, dtype="float32")


all_patients = sorted(
    set(TRAIN_DF["Patient"].unique()).union(set(TEST_DF["Patient"].unique()))
)
volumes = {}
for pid in all_patients:
    folder = (
        TRAIN_IMG_DIR
        if os.path.isdir(os.path.join(TRAIN_IMG_DIR, pid))
        else TEST_IMG_DIR
    )
    volumes[pid] = load_patient_volume(pid, folder)

gc.collect()




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1055344719.py in <cell line: 0>()
     51         else TEST_IMG_DIR
     52     )
---> 53     volumes[pid] = load_patient_volume(pid, folder)
     54 
     55 gc.collect()

/tmp/ipykernel_11/1055344719.py in load_patient_volume(patient_id, folder, img_size, num_scans)
     30     # If fewer slices than needed, pad by repeating edge slices
     31     scans = [dicom.dcmread(os.path.join(image_folder, f)) for f in image_files]
---> 32     imgs = [cv2.resize(get_pixels_hu(s), (img_size, img_size)) for s in scans]
     33     if len(imgs) < num_scans:
     34         if len(imgs) == 0:

/tmp/ipykernel_11/1055344719.py in <listcomp>(.0)
     30     # If fewer slices than needed, pad by repeating edge slices
     31     scans = [dicom.dcmread(os.path.join(image_folder, f)) for f in image_files]
---> 32     imgs = [cv2.resize(get_pixels_hu(s), (img_size, img_size)) for s in scans]
     33     if len(imgs) < num_scans:
     34         if len(imgs) == 0:

/tmp/ipykernel_11/1055344719.py in get_pixels_hu(scan)
      1 def get_pixels_hu(scan):
----> 2     image = scan.pixel_array
      3     image = image.astype(np.int16)
      4 
      5     slope = getattr(scan, "RescaleSlope", 1)

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in pixel_array(self)
   2191             that iterates through the image frames.
   2192         """
-> 2193         self.convert_pixel_data()
   2194         return cast("numpy.ndarray", self._pixel_array)
   2195 

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in convert_pixel_data(self, handler_name)
   1724             # Use 'pydicom.pixels' backend
   1725             opts["decoding_plugin"] = name
-> 1726             self._pixel_array = pixel_array(self, **opts)
   1727             self._pixel_id = get_image_pixel_ids(self)
   1728         else:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py in pixel_array(src, ds_out, specific_tags, index, raw, decoding_plugin, **kwargs)
   1428 
   1429         opts = as_pixel_options(ds, **kwargs)
-> 1430         return decoder.as_array(
   1431             ds,
   1432             index=index,

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in as_array(self, src, index, validate, raw, decoding_plugin, **kwargs)
    980             cast(
    981                 dict[str, "DecodeFunction"],
--> 982                 self._validate_plugins(decoding_plugin),
    983             ),
    984         )

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/common.py in _validate_plugins(self, plugin)
    255         missing = "\n".join([f"\t{s}" for s in self.missing_dependencies])
    256         if self._decoder:
--> 257             raise RuntimeError(
    258                 f"Unable to decompress '{self.UID.name}' pixel data because all "
    259                 f"plugins are missing dependencies:\n{missing}"

RuntimeError: Unable to decompress 'JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])' pixel data because all plugins are missing dependencies:
	gdcm - requires gdcm>=3.0.10
	pylibjpeg - requires pylibjpeg>=2.0 and pylibjpeg-libjpeg>=2.1

## === cell 5
class Dataset(Sequence):
    def __init__(self, df, indices, batch_size=BATCH_SIZE, mode=0):
        self.df = df
        self.indices = np.asarray(indices)
        self.batch_size = batch_size
        self.mode = mode  # 0 - Training/Val, 1 - Test
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.mode == 0:
            np.random.shuffle(self.indices)

    def get_tabular_row(self, row):
        tab = row[training_features].astype("float32").values
        return tab

    def __getitem__(self, index):
        batch_idx = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        batch = self.df.loc[batch_idx]

        patient_ids = batch["Patient"].values
        images = (
            np.asarray([volumes[pid] for pid in patient_ids], dtype=np.float32) / 255.0
        )
        images = np.expand_dims(images, axis=4)  # (B, Z, H, W, 1)
        tabulars = np.asarray(
            [self.get_tabular_row(batch.loc[i]) for i in batch.index], dtype="float32"
        )

        if self.mode == 1:
            return [images, tabulars]

        y = batch[["y0", "y1", "y2"]].astype("float32").values
        return [images, tabulars], y




## === cell 6
def swish(x):
    return x * K.sigmoid(x)


def conv_block(x, num_of_filters, num_of_layers=3):
    for i in range(num_of_layers):
        x = Conv3D(
            num_of_filters,
            kernel_size=(3, 3, 3),
            padding="same",
            kernel_initializer="he_uniform",
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    x = MaxPooling3D()(x)
    return x


def build_model():
    input_img = Input(shape=(NUM_OF_SCANS, IMG_SIZE, IMG_SIZE, 1))
    x = conv_block(input_img, 16, 4)
    x = conv_block(x, 32, 4)
    x = conv_block(x, 64, 4)
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


def train_one_model(seed_offset=0):
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED + seed_offset)
    np.random.seed(SEED + seed_offset)

    model = build_model()
    model.compile(optimizer=Adam(1e-3), loss="mse")

    train_gen = Dataset(TRAIN_DF, train_idx, batch_size=BATCH_SIZE, mode=0)
    val_gen = Dataset(TRAIN_DF, val_idx, batch_size=BATCH_SIZE, mode=0)

    ckpt_path = f"/kaggle/working/model_{seed_offset}.weights.h5"
    cbs = [
        ModelCheckpoint(
            ckpt_path,
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=0,
        )
    ]
    model.fit(
        train_gen, validation_data=val_gen, epochs=EPOCHS, verbose=1, callbacks=cbs
    )
    model.load_weights(ckpt_path)
    return model




## === cell 7
models = [train_one_model(i) for i in range(N_MODELS)]



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3083703657.py in <cell line: 0>()
      1 # Fix: original code expected external pretrained weights which are not available.
      2 # Minimal replacement: train a few models locally using same architecture, then ensemble.
----> 3 models = [train_one_model(i) for i in range(N_MODELS)]
      4 

/tmp/ipykernel_11/3083703657.py in <listcomp>(.0)
      1 # Fix: original code expected external pretrained weights which are not available.
      2 # Minimal replacement: train a few models locally using same architecture, then ensemble.
----> 3 models = [train_one_model(i) for i in range(N_MODELS)]
      4 

/tmp/ipykernel_11/3626220800.py in train_one_model(seed_offset)
     43 
     44     model = build_model()
---> 45     model.compile(optimizer=Adam(1e-3), loss="mse")
     46 
     47     train_gen = Dataset(TRAIN_DF, train_idx, batch_size=BATCH_SIZE, mode=0)

NameError: name 'Adam' is not defined

## === cell 8
sub = SAMPLE_SUBMISSION.copy()
sub[["Patient", "Weeks_str"]] = sub["Patient_Week"].str.split("_", expand=True)
sub["Weeks"] = sub["Weeks_str"].astype("int32")
sub = sub.drop(columns=["Weeks_str"])

static_cols = [c for c in TEST_DF.columns if c in training_features and c != "Weeks"]
test_static = TEST_DF[["Patient"] + static_cols].drop_duplicates("Patient")

test_merge = sub.merge(test_static, on="Patient", how="left")

test_merge["Weeks"] = (test_merge["Weeks"].astype("float32") - MIN_MAX["Weeks"][0]) / (
    MIN_MAX["Weeks"][1] - MIN_MAX["Weeks"][0]
)

for col in training_features:
    if col not in test_merge.columns:
        test_merge[col] = 0.0

test_gen = Dataset(
    test_merge, test_merge.index.to_numpy(), batch_size=BATCH_SIZE, mode=1
)

predictions = np.zeros((len(test_merge), 3), dtype="float32")
for model in models:
    predictions += model.predict(test_gen, verbose=1)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3659115436.py in <cell line: 0>()
     26 
     27 predictions = np.zeros((len(test_merge), 3), dtype="float32")
---> 28 for model in models:
     29     predictions += model.predict(test_gen, verbose=1)
     30 

NameError: name 'models' is not defined

## === cell 9
def denormalize_fvc(y):
    return y * (MIN_MAX["FVC"][1] - MIN_MAX["FVC"][0]) + MIN_MAX["FVC"][0]


predictions = predictions / len(models)
pred_denorm = denormalize_fvc(predictions)

FVC = pred_denorm[:, 1]
Confidence = pred_denorm[:, 2] - pred_denorm[:, 0]

Confidence = np.maximum(Confidence, 70.0)

out = SAMPLE_SUBMISSION.copy()
out["FVC"] = np.round(FVC).astype(int)
out["Confidence"] = np.round(Confidence).astype(int)
out.to_csv("submission.csv", index=False)

print(out.head())
print("Wrote submission.csv with shape:", out.shape)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1467803064.py in <cell line: 0>()
      3 
      4 
----> 5 predictions = predictions / len(models)
      6 pred_denorm = denormalize_fvc(predictions)
      7 

NameError: name 'models' is not defined
