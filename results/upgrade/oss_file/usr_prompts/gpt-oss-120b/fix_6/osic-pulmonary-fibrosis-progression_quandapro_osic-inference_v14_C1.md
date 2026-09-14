# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

-6.906376124026119

# 6. Current score

-9.11719

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.5179) has done: 'I remove the failing TensorFlow imports and model‑weight loading, replace the model ensemble with a simple deterministic predictor based on the normalized “typical_fvc” values already computed, and adjust the post‑processing so it no longer depends on a nonexistent model list. This fixes the import error, the missing‑file error, and the NameError, while still producing a valid `submission.csv` that follows the required format.'
- What this solution (achieved -9.30752) has done: 'I fixed the TensorFlow import error by removing the heavy TensorFlow imports (they’re not needed for the current deterministic predictor) and added a simple baseline‑FVC predictor that uses each patient’s initial measured FVC for all future weeks with a constant confidence of 100. This keeps the original data‑processing steps unchanged, avoids the previous import‑related crash, and provides a more sensible prediction that should raise the score toward the target.'
- What this solution (achieved -9.30752) has done: 'I added an import for `Sequence` to resolve the NameError, loaded the training data and computed a simple per‑patient linear trend (slope) from weeks to FVC. The prediction now uses the patient’s baseline FVC plus this slope multiplied by the week offset, which gives a more realistic forecast than a constant value while keeping the rest of the pipeline unchanged. Confidence remains constant at 100 and the script still writes a correctly‑named `submission.csv`.'
- What this solution (achieved -8.10575) has done: 'I improve the predictions by using an overall average slope for patients that lack a personal trend (most test patients) and increase the confidence value to a higher level (2000) which, given the Laplace Log Likelihood, reduces the penalty from the error term and moves the score closer to the target. These adjustments are minimal, keep the core logic intact, and only affect the final prediction loop.'
- What this solution (achieved -9.11719) has done: 'I lower the constant confidence value from an excessively large 2000 to the minimum allowed 70 (the clipping threshold). This reduces the penalty from the logarithmic term in the Laplace log‑likelihood, moving the score higher (closer to the target) while keeping all other prediction logic unchanged.'

# 9. Code solution

## === cell 0
import cv2
import gc
import numpy as np
import os
import pandas as pd
import pydicom as dicom
import random

from collections.abc import Sequence  # added to fix NameError
from sklearn.model_selection import KFold, train_test_split




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
BATCH_SIZE = 32
TEST_DF = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
SAMPLE_SUBMISSION = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
model_path = "../input/linear-model-osic-v1"
if os.path.isdir(model_path):
    model_weights = [os.path.join(model_path, x) for x in os.listdir(model_path)]
else:
    model_weights = []  # No pretrained weights available.
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

TRAIN_DF = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
TRAIN_DF = TRAIN_DF.copy()




## === cell 3
def create_typical_fvc(df):
    df["typical_fvc"] = df["FVC"] / df["Percent"] * 100.0
    return df


def normalize(df):
    for feature, min_max in MIN_MAX.items():
        df[feature] = (df[feature] - min_max[0]) / (min_max[1] - min_max[0])
    return df


TEST_DF = create_typical_fvc(TEST_DF)
TEST_DF["min_week"] = np.zeros(len(TEST_DF), dtype="int")
TEST_DF["min_week_FVC"] = np.zeros(len(TEST_DF), dtype="int")

TEST_DF["Never smoked"] = (TEST_DF["SmokingStatus"] == "Never smoked").astype("uint8")
TEST_DF["Currently smokes"] = (TEST_DF["SmokingStatus"] == "Currently smokes").astype(
    "uint8"
)
TEST_DF["Ex-smoker"] = (TEST_DF["SmokingStatus"] == "Ex-smoker").astype("uint8")

TEST_DF["Male"] = (TEST_DF["Sex"] == "Male").astype("uint8")
TEST_DF["Female"] = (TEST_DF["Sex"] == "Female").astype("uint8")

TEST_DF = normalize(TEST_DF)
for patient in np.unique(TEST_DF["Patient"]):
    TEST_DF["min_week"][TEST_DF["Patient"] == patient] = TEST_DF["Weeks"][
        TEST_DF["Patient"] == patient
    ].min()
    TEST_DF["min_week_FVC"][TEST_DF["Patient"] == patient] = TEST_DF["FVC"][
        TEST_DF["Patient"] == patient
    ].values[0]

TEST_DF.head()




## === cell 4
def get_pixels_hu(scan):
    image = scan.pixel_array
    image = image.astype(np.int16)

    slope = scan.RescaleSlope
    intercept = scan.RescaleIntercept
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


volumes = {}
for patient_id in np.unique(TEST_DF["Patient"]):
    image_folder = f"../input/osic-pulmonary-fibrosis-progression/test/{patient_id}"
    image_files = np.asarray(os.listdir(image_folder))
    image_files = image_files[
        len(image_files) // 2
        - NUM_OF_SCANS // 2 : len(image_files) // 2
        + NUM_OF_SCANS // 2
    ]
    scans = [
        dicom.dcmread(os.path.join(image_folder, image_file))
        for image_file in image_files
    ]
    images = np.asarray(
        [cv2.resize(get_pixels_hu(scan), (IMG_SIZE, IMG_SIZE)) for scan in scans],
        dtype="float32",
    )
    volumes[patient_id] = images




## === cell 5
class Dataset(Sequence):
    def __init__(self, batch_size=BATCH_SIZE, mode=0):
        self.indices = np.arange(0, len(SAMPLE_SUBMISSION), 1)
        self.batch_size = batch_size
        self.mode = mode  # 0 - Training, 1 - Test

    def __len__(self):
        return len(self.indices) // self.batch_size

    def get_tabular(self, patient, week):
        week = (float(week) - MIN_MAX["Weeks"][0]) / (
            MIN_MAX["Weeks"][1] - MIN_MAX["Weeks"][0]
        )
        tabular = [week]
        tabular += list(
            TEST_DF[training_features[1:]][TEST_DF["Patient"] == patient].values[0]
        )
        return np.asarray(tabular, dtype="float32")

    def get_volume(self, patient_id):
        return volumes[patient_id]

    def __getitem__(self, index):
        if index == self.__len__() - 1:
            indices = self.indices[index * self.batch_size :]
        else:
            indices = self.indices[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
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


def residual_block(x, num_of_filters):
    x1 = Conv3D(
        num_of_filters,
        kernel_size=(3, 1, 1),
        padding="same",
        kernel_initializer="he_uniform",
    )(x)
    b1 = BatchNormalization()(x1)
    a1 = Activation("relu")(b1)
    x2 = Conv3D(
        num_of_filters,
        kernel_size=(1, 3, 3),
        padding="same",
        kernel_initializer="he_uniform",
    )(a1)
    b2 = BatchNormalization()(x2)
    a2 = Activation("relu")(b2)
    x3 = Conv3D(
        num_of_filters,
        kernel_size=(1, 1, 1),
        padding="same",
        kernel_initializer="he_uniform",
    )(a2)
    b3 = BatchNormalization()(x3)
    a3 = Activation("relu")(b3)
    return Add()([a1, a3])


def dense_block(x, num_of_filters):
    x1 = residual_block(x, num_of_filters)
    x2 = residual_block(x1, num_of_filters)
    return Concatenate()([x1, x2])


def build_3d_cnn(input_tensor):
    c1 = Conv3D(64, kernel_size=(5, 7, 7), strides=(1, 2, 2), padding="same")(
        input_tensor
    )
    b1 = BatchNormalization()(c1)
    a1 = Activation("relu")(b1)
    p1 = MaxPool3D(pool_size=(3, 3, 3), strides=(1, 2, 2))(a1)

    r1 = dense_block(p1, 64)
    r1 = dense_block(r1, 64)
    p1 = MaxPool3D(pool_size=(3, 3, 3), strides=(1, 2, 2))(r1)

    r2 = dense_block(p1, 128)
    r2 = dense_block(r2, 128)
    p2 = MaxPool3D(pool_size=(3, 3, 3), strides=(1, 2, 2))(r2)

    r3 = dense_block(p2, 256)
    r3 = dense_block(r3, 256)

    return r3


def build_model(weights):
    input_img = Input(shape=(NUM_OF_SCANS, IMG_SIZE, IMG_SIZE, 1))
    x_img = build_3d_cnn(input_img)
    x_img = GlobalAveragePooling3D()(x_img)
    alpha = Dense(3)(x_img)

    input_tabular = Input(shape=(num_of_features,))

    x_tab = Dense(100)(input_tabular)
    x = Dense(3)(x_tab)
    beta = Dense(3)(x_tab)

    out = Lambda(lambda x: x[0] * x[1] + x[2])([alpha, x, beta])

    model = tf.keras.Model(
        inputs=[input_img, input_tabular], outputs=out, name="tabular_model"
    )

    model.load_weights(weights)

    return model




## === cell 7
models = []  # kept for compatibility with downstream code.




## === cell 8
def compute_patient_trends(df):
    slopes = {}
    baseline_weeks = {}
    baseline_fvc = {}
    for patient, grp in df.groupby("Patient"):
        weeks = grp["Weeks"].values.astype(float)
        fvc = grp["FVC"].values.astype(float)
        if len(weeks) > 1:
            slope, intercept = np.polyfit(weeks, fvc, 1)
        else:
            slope, intercept = 0.0, fvc[0]
        slopes[patient] = slope
        min_idx = weeks.argmin()
        baseline_weeks[patient] = weeks[min_idx]
        baseline_fvc[patient] = fvc[min_idx]
    return slopes, baseline_weeks, baseline_fvc


patient_slopes, patient_base_weeks, patient_base_fvc = compute_patient_trends(TRAIN_DF)

global_slope = np.mean(list(patient_slopes.values()))

raw_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
test_baseline_fvc = raw_test.groupby("Patient")["FVC"].first().to_dict()
test_baseline_week = raw_test.groupby("Patient")["Weeks"].first().to_dict()

FVC_preds = []
Confidence_preds = []

for pw in SAMPLE_SUBMISSION["Patient_Week"]:
    patient, week_str = pw.split("_")
    week = float(week_str)
    base_fvc = test_baseline_fvc.get(patient, patient_base_fvc.get(patient, 0))
    base_week = test_baseline_week.get(patient, patient_base_weeks.get(patient, 0))
    slope = patient_slopes.get(patient, global_slope)
    pred_fvc = base_fvc + slope * (week - base_week)
    pred_fvc = max(0, pred_fvc)
    FVC_preds.append(pred_fvc)
    Confidence_preds.append(70)

SAMPLE_SUBMISSION["FVC"] = np.round(FVC_preds).astype(int)
SAMPLE_SUBMISSION["Confidence"] = np.round(Confidence_preds).astype(int)
SAMPLE_SUBMISSION.to_csv("submission.csv", index=False)
SAMPLE_SUBMISSION.head()
