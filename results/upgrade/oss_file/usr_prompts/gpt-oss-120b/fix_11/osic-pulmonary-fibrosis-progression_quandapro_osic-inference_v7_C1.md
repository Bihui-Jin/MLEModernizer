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

-6.940493497033245

# 6. Current score

-12.20239

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -19.09388) has done: 'I fixed the import error, handled missing model files, and added a safe fallback that predicts each patient’s baseline FVC (from the provided test.csv) with a constant confidence of 100. This removes the crashing TensorFlow weight loading while still producing a correctly‑formatted `submission.csv`, keeping the original pipeline structure intact.'
- What this solution (achieved -18.99672) has done: 'I add preprocessing for the training set, fit a simple linear regression on the normalized tabular features, and use its predictions (with a fixed confidence of 100) instead of the fallback baseline. This fixes the earlier misuse of denormalization, keeps the original pipeline structure, and should raise the score toward the target while still writing a valid `submission.csv`.'
- What this solution (achieved -18.99672) has done: 'The fix addresses the batch‑size mismatch that caused a shape error when writing predictions.  
- A `models` list is defined to avoid a NameError.  
- The inference loop now computes the exact start/end indices for each batch using `ceil` division, ensuring the slice size always matches the predicted vector size, even for the final partial batch.  
- This change keeps the original linear‑regression logic intact while producing a correctly‑shaped `submission.csv`.'
- What this solution (achieved -9.30752) has done: 'I fixed the batch‑size handling in the custom Dataset so the last batch returns only its remaining rows, eliminating the shape‑mismatch error. I also replaced the linear‑regression predictions with each patient’s baseline FVC (taken from the provided test.csv) after normalising it, which aligns the predictions with the true measurement scale and should raise the score toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved -18.99672) has done: 'I added safe optional imports for cv2 and pydicom so the script runs even when those packages are unavailable, providing simple fallback resizing and dummy DICOM reads. I also refined the baseline‑FVC replacement to only apply to the baseline week (week 0) instead of all weeks, letting the linear‑regression model predict the later visits and thereby improving the score toward the target. The rest of the pipeline remains unchanged and a correct submission.csv is written.'
- What this solution (achieved -18.99672) has done: 'I bypass the fragile TensorFlow import by skipping it entirely and setting `tf = None`, which lets the custom `Dataset` fallback to a plain Python object. I also slightly increase the regularisation λ in the linear‑regression solution (from 1e‑5 to 1e‑3) to obtain more stable predictions, which should move the score nearer the target while keeping the core logic unchanged. The rest of the pipeline is left intact and the script now reliably writes a proper `submission.csv`.'
- What this solution (achieved -12.66738) has done: 'Increasing the constant confidence value (from 100 ml to 200 ml) raises the metric because the Laplace‑Log‑Likelihood rewards larger σ values up to the clipping point.  
A slightly weaker regularisation (λ = 1e‑4) lets the linear‑regression model capture more variance in the tabular data, which should also improve the predicted FVCs without altering the overall pipeline. Both tweaks are minimal, keep the core logic intact, and aim to move the score upward toward the target.'
- What this solution (achieved -24.65932) has done: 'I lower the constant confidence to the minimum allowed (70 ml) because a smaller σ improves the Laplace‑Log‑Likelihood, and I weaken the regularisation (λ = 1e‑5) and add a simple quadratic “Weeks_sq” feature to the linear model, which can capture non‑linear trends without changing the overall pipeline. These minimal tweaks keep the core logic intact while moving the score upward toward the target.'
- What this solution (achieved -23.33078) has done: 'I will (1) weaken the ridge regularisation slightly (λ = 1e‑6) so the linear model can capture more signal, (2) clip the normalized predictions to the valid [0, 1] range before denormalising, and (3) clip the final FVC values to the original FVC bounds. These minimal tweaks keep the overall pipeline unchanged while reducing large‑error outliers, which should raise the Laplace‑Log‑Likelihood score toward the target.'
- What this solution (achieved -12.20239) has done: 'I raise the constant confidence from the minimum 70 ml to 200 ml (larger σ reduces the error term for most samples) and weaken the ridge regularisation (λ from 1e‑6 to 1e‑7) so the linear model can capture more signal. I also remove the premature clipping of the normalized predictions, letting the final denormalisation step handle any out‑of‑range values. These minimal edits keep the core pipeline unchanged while moving the metric toward the target.'

# 9. Code solution

## === cell 0
import gc
import numpy as np
import os
import pandas as pd
import random

try:
    import cv2
except Exception:
    cv2 = None

    def _fallback_resize(img, size):
        return np.resize(img, size)

    class _CV2Dummy:
        @staticmethod
        def resize(img, size):
            return _fallback_resize(img, size)

    cv2 = _CV2Dummy()

try:
    import pydicom as dicom
except Exception:
    from types import SimpleNamespace

    class _DicomDummy:
        @staticmethod
        def dcmread(filepath):
            return SimpleNamespace(
                pixel_array=np.zeros((128, 128), dtype=np.int16),
                RescaleSlope=1,
                RescaleIntercept=0,
            )

    dicom = _DicomDummy()

tf = None
K = None

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
NUM_OF_SCANS = 10
BATCH_SIZE = 32

TEST_DF = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
RAW_TEST_DF = TEST_DF.copy()  # keep original values for baseline FVC lookup
SAMPLE_SUBMISSION = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)

model_path = "../input/osic-model-weights-resnet-v2"
if os.path.isdir(model_path):
    model_weights = [os.path.join(model_path, x) for x in os.listdir(model_path)]
else:
    model_weights = []  # no weights available; will use fallback predictions

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
    "Weeks_sq",  # new quadratic feature
]
num_of_features = len(training_features)




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

TEST_DF["Weeks_sq"] = TEST_DF["Weeks"] ** 2

for patient in np.unique(TEST_DF["Patient"]):
    mask = TEST_DF["Patient"] == patient
    TEST_DF.loc[mask, "min_week"] = TEST_DF.loc[mask, "Weeks"].min()
    TEST_DF.loc[mask, "min_week_FVC"] = TEST_DF.loc[mask, "FVC"].values[0]




## === cell 4
TRAIN_DF = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
RAW_TRAIN_DF = TRAIN_DF.copy()

TRAIN_DF = create_typical_fvc(TRAIN_DF)
TRAIN_DF["min_week"] = np.zeros(len(TRAIN_DF), dtype="int")
TRAIN_DF["min_week_FVC"] = np.zeros(len(TRAIN_DF), dtype="int")

TRAIN_DF["Never smoked"] = (TRAIN_DF["SmokingStatus"] == "Never smoked").astype("uint8")
TRAIN_DF["Currently smokes"] = (TRAIN_DF["SmokingStatus"] == "Currently smokes").astype(
    "uint8"
)
TRAIN_DF["Ex-smoker"] = (TRAIN_DF["SmokingStatus"] == "Ex-smoker").astype("uint8")

TRAIN_DF["Male"] = (TRAIN_DF["Sex"] == "Male").astype("uint8")
TRAIN_DF["Female"] = (TRAIN_DF["Sex"] == "Female").astype("uint8")

TRAIN_DF = normalize(TRAIN_DF)

TRAIN_DF["Weeks_sq"] = TRAIN_DF["Weeks"] ** 2

for patient in np.unique(TRAIN_DF["Patient"]):
    mask = TRAIN_DF["Patient"] == patient
    TRAIN_DF.loc[mask, "min_week"] = TRAIN_DF.loc[mask, "Weeks"].min()
    TRAIN_DF.loc[mask, "min_week_FVC"] = TRAIN_DF.loc[mask, "FVC"].values[0]




## === cell 5
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
    return cv2.resize(image, (IMG_SIZE, IMG_SIZE)).astype(np.uint8)




## === cell 6
volumes = {}
for patient_id in np.unique(TEST_DF["Patient"]):
    image_folder = f"../input/osic-pulmonary-fibrosis-progression/test/{patient_id}"
    if not os.path.isdir(image_folder):
        continue
    image_files = np.asarray(os.listdir(image_folder))
    image_files = image_files[
        len(image_files) // 2
        - NUM_OF_SCANS // 2 : len(image_files) // 2
        + NUM_OF_SCANS // 2
    ]
    scans = [dicom.dcmread(os.path.join(image_folder, f)) for f in image_files]
    images = np.asarray(
        [cv2.resize(get_pixels_hu(s), (IMG_SIZE, IMG_SIZE)) for s in scans],
        dtype="float32",
    )
    volumes[patient_id] = images




## === cell 7
class Dataset(tf.keras.utils.Sequence if tf else object):
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
            TEST_DF[training_features[1:]][TEST_DF["Patient"] == patient].values[0]
        )
        return np.asarray(tabular, dtype="float32")

    def get_volume(self, patient_id):
        return volumes.get(
            patient_id, np.zeros((NUM_OF_SCANS, IMG_SIZE, IMG_SIZE), dtype="float32")
        )

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.indices))
        indices = self.indices[start:end]

        Patient_Week = np.asarray(SAMPLE_SUBMISSION["Patient_Week"][indices])
        patient = [x.split("_")[0] for x in Patient_Week]
        week = [x.split("_")[1] for x in Patient_Week]

        images = (
            np.asarray([self.get_volume(p) for p in patient], dtype=np.float32) / 255.0
        )
        images = np.expand_dims(images, axis=4)

        tabulars = np.asarray(
            [self.get_tabular(patient[i], week[i]) for i in range(len(patient))],
            dtype="float32",
        )
        return [images, tabulars]




## === cell 8
X_train_tab = TRAIN_DF[training_features].values.astype("float32")
X_train = np.concatenate(
    [np.ones((X_train_tab.shape[0], 1), dtype="float32"), X_train_tab], axis=1
)

y_train_norm = (TRAIN_DF["FVC"].values - MIN_MAX["FVC"][0]) / (
    MIN_MAX["FVC"][1] - MIN_MAX["FVC"][0]
)
y_train_norm = y_train_norm.astype("float32")

lam = 1e-7
A = X_train.T @ X_train + lam * np.eye(X_train.shape[1])
b = X_train.T @ y_train_norm
w = np.linalg.solve(A, b)  # shape (num_features+1,)

test_gen = Dataset(mode=1)

models = []  # placeholder to keep later blending code functional

predictions = np.zeros((len(SAMPLE_SUBMISSION), 3), dtype="float32")
baseline_conf = 200.0  # use a higher constant confidence

num_batches = int(np.ceil(len(SAMPLE_SUBMISSION) / BATCH_SIZE))
for idx in range(num_batches):
    batch_start = idx * BATCH_SIZE
    batch_end = min(batch_start + BATCH_SIZE, len(SAMPLE_SUBMISSION))

    images, tabulars = test_gen[idx]  # images are unused for linear model
    X_batch = np.concatenate(
        [np.ones((tabulars.shape[0], 1), dtype="float32"), tabulars], axis=1
    )
    pred_norm = X_batch @ w  # normalized FVC

    predictions[batch_start:batch_end, 1] = pred_norm
    predictions[batch_start:batch_end, 2] = baseline_conf

baseline_fvc_dict = RAW_TEST_DF.set_index("Patient")["FVC"].to_dict()
patient_ids = SAMPLE_SUBMISSION["Patient_Week"].str.split("_", expand=True)[0]
weeks_series = (
    SAMPLE_SUBMISSION["Patient_Week"].str.split("_", expand=True)[1].astype(int)
)

baseline_vals = patient_ids.map(baseline_fvc_dict)
norm_vals = (baseline_vals - MIN_MAX["FVC"][0]) / (
    MIN_MAX["FVC"][1] - MIN_MAX["FVC"][0]
)

mask = norm_vals.notna() & (weeks_series == 0)
predictions[mask.values, 1] = norm_vals[mask].values

if models:
    model_preds = np.zeros_like(predictions)
    for model in models:
        model_preds += model.predict(test_gen, verbose=1)
    model_preds /= max(1, len(models))
    predictions[:, 1] = model_preds[:, 1]
    predictions[:, 2] = model_preds[:, 2]




## === cell 9
def denormalize_fvc(y):
    return y * (MIN_MAX["FVC"][1] - MIN_MAX["FVC"][0]) + MIN_MAX["FVC"][0]


predictions[:, 1] = denormalize_fvc(predictions[:, 1])

predictions[:, 1] = np.clip(
    predictions[:, 1],
    MIN_MAX["FVC"][0],
    MIN_MAX["FVC"][1],
)

FVC = predictions[:, 1]
Confidence = predictions[:, 2]  # already set to higher confidence for better metric

SAMPLE_SUBMISSION["FVC"] = FVC.astype("int")
SAMPLE_SUBMISSION["Confidence"] = Confidence.astype("int")
SAMPLE_SUBMISSION.to_csv("submission.csv", index=False)

SAMPLE_SUBMISSION.head()
