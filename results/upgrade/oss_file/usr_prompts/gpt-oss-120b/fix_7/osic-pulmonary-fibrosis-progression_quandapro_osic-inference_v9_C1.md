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

-24.65932

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -18.49739) has done: 'I wrap the TensorFlow import to avoid the protobuf error, guard all model‑related code so it only runs when pretrained weights are present, and add a simple fallback that uses the normalized “typical_fvc” feature to create predictions when no model can be loaded. This ensures the script runs end‑to‑end and writes a valid `submission.csv` while keeping the original architecture untouched for cases where the weights are available.'
- What this solution (achieved -18.99672) has done: 'Implemented missing imports, safe TensorFlow handling, fallback for missing OpenCV, and defined all required classes/functions. Consolidated constants and ensured every variable is defined before use. Added a lightweight `Sequence` placeholder when TensorFlow isn’t available. The script now runs end‑to‑end, trains the fallback tabular model, loads test volumes if possible, and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved -24.65932) has done: 'I keep the existing architecture and imports, but add a simple calibration step for the confidence values. After training the GradientBoostingRegressor, I compute the median absolute error on the training data and use it (clipped at the required minimum of 70) as the confidence for every test prediction. This small change aligns the confidence with the model’s typical error, improving the Laplace Log Likelihood without altering the core logic.'
- What this solution (achieved -24.65932) has done: 'I guard the optional DICOM loading so the script runs without pydicom or TensorFlow, tighten the tabular model hyper‑parameters for better predictions, and set the confidence to the required minimum 70 ml (which is optimal for the given metric when the error isn’t huge). These minimal fixes keep the original architecture while ensuring a valid submission.csv is written and moving the score closer to the target.'
- What this solution (achieved -24.65932) has done: 'Implemented a confidence calibration step: after fitting the GradientBoostingRegressor, the script now computes the median absolute error on the training set and uses the larger of this error or the required minimum 70 as the confidence value for all predictions. This aligns the confidence with the model’s typical error, improving the Laplace Log Likelihood while keeping the core logic unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import (
        Conv3D,
        BatchNormalization,
        Activation,
        Add,
        MaxPool3D,
        GlobalAveragePooling3D,
        Input,
        Concatenate,
        Dense,
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.utils import Sequence
except Exception:  # covers ImportError and protobuf related errors
    tf = None
    K = None
    Conv3D = BatchNormalization = Activation = Add = MaxPool3D = None
    GlobalAveragePooling3D = Input = Concatenate = Dense = None
    Model = None

    class Sequence:
        pass


try:
    import pydicom as dicom
except ImportError:
    dicom = None

from sklearn.ensemble import GradientBoostingRegressor

MIN_MAX = {
    "Weeks": (-5.0, 133.0),
    "FVC": (827.0, 6399.0),
    "Percent": (28.877577, 153.145378),
    "Age": (49.0, 88.0),
    "typical_fvc": (827.0, 6399.0),
}
IMG_SIZE = 128
NUM_OF_SCANS = 12
BATCH_SIZE = 32

TEST_DF = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
SAMPLE_SUBMISSION = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)

model_path = "../input/osic-resnet-v4"
if tf is not None and os.path.isdir(model_path):
    model_weights = [os.path.join(model_path, x) for x in os.listdir(model_path)]
else:
    model_weights = []  # no weights available

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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TRAIN_DF = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")


def create_typical_fvc(df):
    df["typical_fvc"] = df["FVC"] / df["Percent"] * 100.0
    return df


def normalize(df):
    for feature, min_max in MIN_MAX.items():
        df[feature] = (df[feature] - min_max[0]) / (min_max[1] - min_max[0])
    return df


TRAIN_DF = create_typical_fvc(TRAIN_DF)
TRAIN_DF["min_week"] = 0  # placeholder (not used for training)
TRAIN_DF["min_week_FVC"] = 0  # placeholder
TRAIN_DF["Never smoked"] = (TRAIN_DF["SmokingStatus"] == "Never smoked").astype("uint8")
TRAIN_DF["Currently smokes"] = (TRAIN_DF["SmokingStatus"] == "Currently smokes").astype(
    "uint8"
)
TRAIN_DF["Ex-smoker"] = (TRAIN_DF["SmokingStatus"] == "Ex-smoker").astype("uint8")
TRAIN_DF["Male"] = (TRAIN_DF["Sex"] == "Male").astype("uint8")
TRAIN_DF["Female"] = (TRAIN_DF["Sex"] == "Female").astype("uint8")
TRAIN_DF = normalize(TRAIN_DF)

X_train = TRAIN_DF[training_features].values.astype("float32")
y_train = TRAIN_DF["FVC"].values.astype("float32")  # original scale for later denorm

tabular_regressor = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
)
tabular_regressor.fit(X_train, y_train)

train_pred = tabular_regressor.predict(X_train)
median_error = np.median(np.abs(train_pred - y_train))
estimated_confidence = float(
    max(70.0, median_error)
)  # respect minimum required confidence




## === cell 2
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
    mask = TEST_DF["Patient"] == patient
    TEST_DF.loc[mask, "min_week"] = TEST_DF.loc[mask, "Weeks"].min()
    TEST_DF.loc[mask, "min_week_FVC"] = TEST_DF.loc[mask, "FVC"].values[0]

volumes = {}
if tf is not None and len(model_weights) > 0 and dicom is not None:

    def get_pixels_hu(scan):
        image = scan.pixel_array.astype(np.int16)

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
        image = np.clip(image, image_min, image_max)

        image = (image - image_min) / (image_max - image_min) * 255.0
        return image.astype(np.uint8)

    def resize_image(img):
        if cv2 is not None:
            return cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        factor_h = max(1, IMG_SIZE // img.shape[0])
        factor_w = max(1, IMG_SIZE // img.shape[1])
        resized = np.repeat(np.repeat(img, factor_h, axis=0), factor_w, axis=1)
        return resized[:IMG_SIZE, :IMG_SIZE]

    for patient_id in np.unique(TEST_DF["Patient"]):
        image_folder = f"../input/osic-pulmonary-fibrosis-progression/test/{patient_id}"
        if not os.path.isdir(image_folder):
            continue
        image_files = np.asarray(os.listdir(image_folder))
        start = len(image_files) // 2 - NUM_OF_SCANS // 2
        end = start + NUM_OF_SCANS
        image_files = image_files[start:end]

        scans = [
            dicom.dcmread(os.path.join(image_folder, image_file))
            for image_file in image_files
        ]
        images = np.asarray(
            [resize_image(get_pixels_hu(scan)) for scan in scans], dtype="float32"
        )
        volumes[patient_id] = images




## === cell 3
class Dataset(Sequence):
    def __init__(self, batch_size=BATCH_SIZE, mode=0):
        self.indices = np.arange(0, len(SAMPLE_SUBMISSION), 1)
        self.batch_size = batch_size
        self.mode = mode  # 0 – training (unused), 1 – test

    def __len__(self):
        return len(self.indices) // self.batch_size

    def get_tabular(self, patient, week):
        week_norm = (float(week) - MIN_MAX["Weeks"][0]) / (
            MIN_MAX["Weeks"][1] - MIN_MAX["Weeks"][0]
        )
        tabular = [week_norm]
        tabular += list(
            TEST_DF[training_features[1:]][TEST_DF["Patient"] == patient].values[0]
        )
        return np.asarray(tabular, dtype="float32")

    def get_volume(self, patient_id):
        return volumes.get(
            patient_id, np.zeros((NUM_OF_SCANS, IMG_SIZE, IMG_SIZE), dtype="float32")
        )

    def __getitem__(self, index):
        if index == self.__len__() - 1:
            indices = self.indices[index * self.batch_size :]
        else:
            indices = self.indices[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
        Patient_Week = np.asarray(SAMPLE_SUBMISSION["Patient_Week"].iloc[indices])
        patient = [x.split("_")[0] for x in Patient_Week]
        week = [x.split("_")[1] for x in Patient_Week]

        images = (
            np.asarray([self.get_volume(pid) for pid in patient], dtype=np.float32)
            / 255.0
        )
        images = np.expand_dims(images, axis=4)  # add channel dim

        tabulars = np.asarray(
            [self.get_tabular(patient[i], week[i]) for i in range(len(patient))],
            dtype="float32",
        )
        return [images, tabulars]




## === cell 4
if tf is not None:

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

    def build_model(weights):
        input_img = Input(shape=(NUM_OF_SCANS, IMG_SIZE, IMG_SIZE, 1))
        x = build_3d_resnet(input_img)
        x = GlobalAveragePooling3D()(x)

        input_tabular = Input(shape=(num_of_features,))

        x = Concatenate()([x, input_tabular])
        x = Dense(100, activation="relu")(x)
        x = Dense(100, activation="relu")(x)
        out = Dense(3)(x)

        model = Model(
            inputs=[input_img, input_tabular], outputs=out, name="fusion_model"
        )
        model.load_weights(weights)
        return model

else:

    def build_model(weights):
        raise RuntimeError("TensorFlow not available; cannot build model.")




## === cell 5
if tf is not None and len(model_weights) > 0:
    models = [build_model(w) for w in model_weights]
else:
    models = []  # fallback to tabular only




## === cell 6
test_gen = Dataset(mode=1)

if len(models) > 0:
    predictions = np.zeros((len(SAMPLE_SUBMISSION), 3), dtype="float32")
    for model in models:
        predictions += model.predict(test_gen, verbose=1)
    predictions = predictions / len(models)
else:
    predictions = np.zeros((len(SAMPLE_SUBMISSION), 3), dtype="float32")
    for idx, row in SAMPLE_SUBMISSION.iterrows():
        patient = row["Patient_Week"].split("_")[0]
        feat = TEST_DF[TEST_DF["Patient"] == patient][training_features].values[0]
        pred_fvc = tabular_regressor.predict(feat.reshape(1, -1))[0]
        predictions[idx, 1] = pred_fvc  # column 1 holds FVC

FVC = predictions[:, 1]
Confidence = np.full_like(FVC, estimated_confidence)

SAMPLE_SUBMISSION["FVC"] = FVC.astype("int")
SAMPLE_SUBMISSION["Confidence"] = Confidence.astype("int")
SAMPLE_SUBMISSION.to_csv("submission.csv", index=False)

SAMPLE_SUBMISSION.head()
