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

-6.905616663638874

# 6. Current score

-14.80647

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.78572) has done: 'You’re failing immediately on import due to an incompatibility between `pydicom` and the system `protobuf`; the safest minimal fix is to force protobuf’s pure-Python implementation before importing `pydicom`. Next, your notebook expects an external weights dataset (`../input/osic-resnet-v3`) that isn’t present, so `model_weights` never gets defined and everything downstream breaks; I add a robust fallback that builds the same model and produces predictions without those weights. Finally, there are a couple of Pandas chained-assignment issues and a `Sequence` import issue that can break or silently misbehave; I fix those and ensure we always write a valid `submission.csv` with correct columns and row count.'
- What this solution (achieved -8.78572) has done: 'I fix the immediate import crash caused by the `protobuf`/`pydicom` incompatibility by forcing protobuf’s pure-Python implementation *before* importing TensorFlow (and by avoiding importing `pydicom` at module import-time). I also make DICOM loading more robust (use `pydicom.dcmread(force=True)` and skip non-DICOM files) so volume creation doesn’t fail mid-run. To move score upward toward the target with minimal semantic change, I load the provided ResNet weights when present and otherwise still run end-to-end (but warn that score be lower without them). Finally, I ensure the submission is always written as `submission.csv` with the exact required columns and row count.'
- What this solution (achieved -8.78572) has done: 'We fix the import-time crash (`MessageFactory.GetPrototype`) by forcing protobuf’s pure-Python implementation *and* preventing TensorFlow from loading the protobuf C++ implementation (via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus a safe TF import). Next, we make the DICOM import more robust by importing `pydicom` only after the protobuf env vars are set, and by guarding against any remaining protobuf-related failures with a clear error. Finally, we ensure the generator length covers all rows (so predictions align with `sample_submission.csv`) and keep the existing prediction/denormalization logic intact, still writing a valid `submission.csv` with the required columns.'
- What this solution (achieved -14.80647) has done: 'We fix the import-time crash (`MessageFactory.GetPrototype`) by ensuring the pure-Python protobuf implementation is selected early and by importing TensorFlow in a safer way for this environment. Next, we keep the existing model/prediction logic intact but add a tiny, metric-aligned calibration step for `Confidence` (a single global scaling based on the model’s predicted spread) to nudge the score upward toward your target without changing the core architecture or inference flow. Finally, we ensure DICOM loading remains robust and that `submission.csv` is always written with the exact required columns and row count.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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
from tensorflow.keras.callbacks import *

from sklearn.model_selection import KFold, train_test_split

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)



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
IMG_SIZE = 256
NUM_OF_SCANS = 10
BATCH_SIZE = 32

TEST_DF = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
SAMPLE_SUBMISSION = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)

model_path = "../input/osic-resnet-v3"
if os.path.isdir(model_path):
    model_weights = [os.path.join(model_path, x) for x in os.listdir(model_path)]
    model_weights = sorted([w for w in model_weights if os.path.isfile(w)])
else:
    model_weights = []  # fallback: run with randomly initialized weights

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


TEST_DF = create_typical_fvc(TEST_DF)

TEST_DF["min_week"] = np.zeros(len(TEST_DF), dtype="float32")
TEST_DF["min_week_FVC"] = np.zeros(len(TEST_DF), dtype="float32")

TEST_DF["Never smoked"] = (TEST_DF["SmokingStatus"] == "Never smoked").astype("uint8")
TEST_DF["Currently smokes"] = (TEST_DF["SmokingStatus"] == "Currently smokes").astype(
    "uint8"
)
TEST_DF["Ex-smoker"] = (TEST_DF["SmokingStatus"] == "Ex-smoker").astype("uint8")

TEST_DF["Male"] = (TEST_DF["Sex"] == "Male").astype("uint8")
TEST_DF["Female"] = (TEST_DF["Sex"] == "Female").astype("uint8")

TEST_DF = normalize(TEST_DF)

for patient in np.unique(TEST_DF["Patient"]):
    m = TEST_DF["Patient"] == patient
    TEST_DF.loc[m, "min_week"] = TEST_DF.loc[m, "Weeks"].min()
    TEST_DF.loc[m, "min_week_FVC"] = TEST_DF.loc[m, "FVC"].values[0]

TEST_DF.head()



## === cell 4
try:
    import pydicom
except Exception as e:
    raise RuntimeError(
        "Failed to import pydicom. This is usually caused by protobuf incompatibility. "
        "Ensure PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is set before importing."
    ) from e


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
    image[image < image_min] = image_min
    image[image > image_max] = image_max

    image = image.astype(np.float64)
    image = (image - image_min) / (image_max - image_min) * 255.0
    return image.astype(np.uint8)


def safe_dcmread(path):
    try:
        return pydicom.dcmread(path, force=True)
    except Exception:
        return None


volumes = {}
for patient_id in np.unique(TEST_DF["Patient"]):
    image_folder = f"../input/osic-pulmonary-fibrosis-progression/test/{patient_id}"
    image_files = np.asarray(sorted(os.listdir(image_folder)))
    image_files = [f for f in image_files if not f.startswith(".")]

    mid = len(image_files) // 2
    half = NUM_OF_SCANS // 2
    image_files = image_files[max(0, mid - half) : min(len(image_files), mid + half)]

    scans = []
    for image_file in image_files:
        scan = safe_dcmread(os.path.join(image_folder, image_file))
        if scan is not None:
            scans.append(scan)

    if len(scans) == 0:
        raise RuntimeError(f"No readable DICOM slices found for patient {patient_id}")
    while len(scans) < NUM_OF_SCANS:
        scans.append(scans[-1])
    if len(scans) > NUM_OF_SCANS:
        scans = scans[:NUM_OF_SCANS]

    images = np.asarray(
        [cv2.resize(get_pixels_hu(scan), (IMG_SIZE, IMG_SIZE)) for scan in scans],
        dtype="float32",
    )
    volumes[patient_id] = images

len(volumes)




## === cell 5
class Dataset(Sequence):
    def __init__(self, batch_size=BATCH_SIZE, mode=0):
        self.indices = np.arange(0, len(SAMPLE_SUBMISSION), 1)
        self.batch_size = batch_size
        self.mode = mode  # 0 - Training, 1 - Test

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def get_tabular(self, patient, week):
        week = (float(week) - MIN_MAX["Weeks"][0]) / (
            MIN_MAX["Weeks"][1] - MIN_MAX["Weeks"][0]
        )
        tabular = [week]
        tabular += list(
            TEST_DF.loc[TEST_DF["Patient"] == patient, training_features[1:]].values[0]
        )
        return np.asarray(tabular, dtype="float32")

    def get_volume(self, patient_id):
        return volumes[patient_id]

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.indices))
        indices = self.indices[start:end]

        Patient_Week = np.asarray(SAMPLE_SUBMISSION["Patient_Week"][indices])
        patient = [x.split("_")[0] for x in Patient_Week]
        week = [x.split("_")[1] for x in Patient_Week]
        images = (
            np.asarray(
                [self.get_volume(patient_id) for patient_id in patient],
                dtype=np.float32,
            )
            / 255.0
        )
        images = np.expand_dims(images, axis=4)
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


def build_model(weights=None):
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

    if weights is not None and isinstance(weights, str) and os.path.isfile(weights):
        model.load_weights(weights)

    return model




## === cell 7
if len(model_weights) == 0:
    models = [build_model(weights=None)]
else:
    models = [build_model(weights=w) for w in model_weights]

len(models)



## === cell 8
test_gen = Dataset(mode=1)

predictions = np.zeros((len(SAMPLE_SUBMISSION), 3), dtype="float32")

batch_size = BATCH_SIZE
n = len(SAMPLE_SUBMISSION)
for model in models:
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        Patient_Week = SAMPLE_SUBMISSION["Patient_Week"].iloc[start:end].values
        patient = [x.split("_")[0] for x in Patient_Week]
        week = [x.split("_")[1] for x in Patient_Week]
        images = np.asarray([volumes[pid] for pid in patient], dtype=np.float32) / 255.0
        images = np.expand_dims(images, axis=4)
        tabulars = np.asarray(
            [test_gen.get_tabular(patient[i], week[i]) for i in range(len(patient))],
            dtype="float32",
        )
        preds = model.predict([images, tabulars], verbose=0)
        predictions[start:end] += preds.astype("float32")
        start = end




## === cell 9
def denormalize(y):
    return y * (MIN_MAX["FVC"][1] - MIN_MAX["FVC"][0]) + MIN_MAX["FVC"][0]


predictions = predictions / len(models)
predictions = denormalize(predictions)

FVC = predictions[:, 1]

raw_conf = predictions[:, 2] - predictions[:, 0]
raw_conf = np.maximum(raw_conf, 1.0)

median_conf = float(np.median(raw_conf))
scale = 150.0 / median_conf if median_conf > 0 else 1.0
Confidence = raw_conf * scale
Confidence = np.maximum(Confidence, 70.0)

SAMPLE_SUBMISSION["FVC"] = np.round(FVC).astype(int)
SAMPLE_SUBMISSION["Confidence"] = np.round(Confidence).astype(int)

SAMPLE_SUBMISSION = SAMPLE_SUBMISSION[["Patient_Week", "FVC", "Confidence"]]
SAMPLE_SUBMISSION.to_csv("submission.csv", index=False)

SAMPLE_SUBMISSION.head()
