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

-8.1338

# 6. Current score

-24.65927

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -9.30752) has done: 'The fix updates imports, corrects deprecated pandas and TensorFlow API usage, replaces the removed pydicom.read_file with pydicom.dcmread, uses pd.concat instead of the removed DataFrame.append, adjusts the Adam optimizer constructor, and trims the training epochs so the notebook runs quickly while still producing a valid submission.csv that follows the required format.'
- What this solution (achieved -9.30752) has done: 'The update fixes the import errors, adds missing libraries, corrects the TensorFlow model construction (using proper `tf.keras` classes), adjusts the training target shape, imports the timing utility, and ensures the script runs end‑to‑end while keeping the original modelling approach. These changes also improve the metric‑compatible loss, helping the score move closer to the target.'
- What this solution (achieved -9.30752) has done: 'I added an environment setting to avoid the protobuf import error and rewrote the custom loss/metric functions so they return a true scalar, fixing the shape mismatch during model training. These minimal changes keep the original neural‑net architecture and workflow while allowing the script to run end‑to‑end and produce a valid `submission.csv`.'
- What this solution (achieved -9.30752) has done: 'The fix removes the shape‑mismatch in the custom loss by setting `_lambda = 0.0`, so the loss uses only the compatible `score` component (which works with a [batch, 3] prediction and a [batch] target). This eliminates the invalid tensor operations that caused the training crash while keeping the original model architecture and prediction logic unchanged, allowing the script to run end‑to‑end and produce a valid `submission.csv`.'
- What this solution (achieved -9.30752) has done: 'Implemented fixes to eliminate runtime errors and improve the competition score:  
1. Corrected the custom loss input shape by using a 1‑dimensional target vector (`y_train = y_fvc`).  
2. Modified the `score` function to return the *negative* Laplace Log Likelihood (the actual competition metric) so the loss minimisation drives the metric upward.  
3. Updated comments accordingly while preserving the original model architecture and training workflow.'
- What this solution (achieved -9.30752) has done: 'I added a compatibility patch for protobuf’s `MessageFactory` so TensorFlow imports correctly, and simplified the custom loss to use only the score component (avoiding the shape‑mismatch caused by the unused quantile loss). These fixes resolve the import error and the training shape error, allowing the model to run and produce a valid `submission.csv`, which should move the score closer to the target.'
- What this solution (achieved -24.65922) has done: 'I removed the unnecessary protobuf compatibility block that caused an import error and adjusted the post‑processing so that predictions are used for all non‑baseline weeks while keeping the baseline weeks at the provided FVC with a fixed confidence of 70. This fixes the runtime crash and improves the metric by correctly using the model’s outputs.'
- What this solution (achieved -24.65928) has done: 'I added a compatibility patch for the protobuf MessageFactory before TensorFlow is imported to stop the import‑time AttributeError, and I simplified the confidence handling so every prediction uses the competition‑recommended confidence of 70 ml (which improves the score). The rest of the pipeline is unchanged, preserving the original model and feature engineering.'
- What this solution (achieved -24.65928) has done: 'I adjust the post‑processing so that the predicted confidence (σ) is taken from the model’s output, clipped at the required minimum of 70 ml, instead of forcing a constant 70 ml for all non‑baseline weeks. This keeps the baseline weeks at the official 70 ml (where the true FVC is known) but provides a more realistic confidence for the forecasted weeks, which should raise the Laplace Log Likelihood score toward the target.'
- What this solution (achieved -10.81761) has done: 'The change only adjusts the post‑processing step: instead of using the model’s predictions for the future weeks, we now predict each patient’s FVC as their baseline value (the first measurement in the test set) and set a constant confidence of 70 ml for every entry. This aligns the submission with the competition’s recommended confidence floor and typically yields a higher (less negative) Laplace Log Likelihood, moving the score toward the target while leaving the core model and training untouched.'
- What this solution (achieved -24.65927) has done: 'I revise the post‑processing step so that baseline weeks keep the true FVC and a confidence of 70, while all other weeks use the model’s predictions (with confidence clipped to the required minimum of 70). This keeps the core model unchanged but leverages its forecasts to raise the Laplace Log Likelihood toward the target score.'

# 9. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")

import os, glob, sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def GetPrototype(self, descriptor):
            return descriptor._concrete_class

        message_factory.MessageFactory.GetPrototype = GetPrototype
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import timeit

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, optimizers

import pydicom

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error




## === cell 1
train_x = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
print(
    "the no of rows is {} and the no of columns is {} ".format(
        train_x.shape[0], train_x.shape[1]
    )
)




## === cell 2
train_x.describe()




## === cell 3
test_x = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
print(
    "the no of rows is {} and the no of columns is {} ".format(
        test_x.shape[0], test_x.shape[1]
    )
)




## === cell 4
test_x.describe()




## === cell 5
sns.countplot(x="Sex", data=train_x)




## === cell 6
new_df = train_x.groupby(
    [train_x.Patient, train_x.Age, train_x.Sex, train_x.SmokingStatus]
)["Patient"].count()
new_df.index = new_df.index.set_names(["id", "Age", "Sex", "SmokingStatus"])
new_df = new_df.reset_index()
new_df.rename(columns={"Patient": "freq"}, inplace=True)
fig = px.bar(new_df, x="id", y="freq", color="freq")
fig.update_layout(
    xaxis={"categoryorder": "total ascending"},
    title="Distribution of images for each patient",
)
fig.update_xaxes(showticklabels=False)
fig.show()




## === cell 7
fig = px.histogram(new_df, x="Age", nbins=42)
fig.update_traces(
    marker_color="rgb(158,202,225)",
    marker_line_color="rgb(8,48,107)",
    marker_line_width=1.5,
    opacity=0.6,
)
fig.update_layout(title="Distribution of Age")
fig.show()




## === cell 8
sns.countplot(x="SmokingStatus", data=train_x)




## === cell 9
fig = px.histogram(
    train_x,
    x="Age",
    color="SmokingStatus",
    color_discrete_map={
        "Never smoked": "yellow",
        "Currently smokes": "cyan",
        "Ex-smoker": "green",
    },
    hover_data=train_x.columns,
)
fig.update_layout(title="Distribution of Age w.r.t. SmokingStatus for unique patients")
fig.update_traces(marker_line_color="black", marker_line_width=1.5, opacity=0.85)
fig.show()




## === cell 10
plt.figure(figsize=(5, 5))
sns.countplot(x="Sex", hue="SmokingStatus", data=train_x)




## === cell 11
fig = px.histogram(
    train_x,
    x="Age",
    color="Sex",
    color_discrete_map={"Male": "blue", "Female": "mediumturquoise"},
    hover_data=train_x.columns,
)
fig.update_layout(title="Distribution of Age w.r.t. sex for unique patients")
fig.update_traces(marker_line_color="black", marker_line_width=1.5, opacity=0.85)
fig.show()




## === cell 12
sns.heatmap(
    train_x.select_dtypes(include=[np.number]).corr(), annot=True, cmap=plt.cm.cool
)




## === cell 13
a = sns.distplot(
    train_x["FVC"],
    color="r",
)
a.set_title("Distribution plot of SVC ", color="g", fontsize=18)




## === cell 14
b = sns.distplot(train_x["Percent"], color="g")
b.set_title("Distribution plot of Percent", color="r", fontsize=18)




## === cell 15
data = px.bar(
    x=list(train_x["Weeks"].value_counts().keys()),
    y=list(train_x["Weeks"].value_counts().values),
)
data




## === cell 16
fig = px.line(
    train_x,
    "Weeks",
    "FVC",
    line_group="Patient",
    color="Sex",
    title="Pulmonary Condition Progression by Sex",
)
fig.update_traces(mode="lines + markers")




## === cell 17
fig = px.line(
    train_x,
    "Weeks",
    "FVC",
    line_group="Patient",
    color="SmokingStatus",
    title="Pulmonary Condition Progression by Smoking Status",
)
fig.update_traces(mode="lines+markers")




## === cell 18
print(
    "The Number of Unique Patients in training data are : {}".format(
        len(train_x["Patient"].unique())
    )
)




## === cell 19
data_path = "../input/osic-pulmonary-fibrosis-progression/train/"
output_path = "../input/output/"
train_image_files = sorted(glob.glob(os.path.join(data_path, "*", "*.dcm")))
patients = os.listdir(data_path)
patients.sort()
print("Some sample Patient ID's :", len(train_image_files))
print("\n".join(train_image_files[:5]))




## === cell 20
def load_scan(path):
    """Loads scans from a folder and into a list."""
    slices = [pydicom.dcmread(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]
        )
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu(scans):
    """Converts raw images to Hounsfield Units (HU)."""
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)




## === cell 21
def set_lungwin(img, hu=[-1200.0, 600.0]):
    lungwin = np.array(hu)
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype("uint8")
    return newimg




## === cell 22
train_x.shape




## === cell 23
test_x.shape




## === cell 24
def eval_metric(FVC, FVC_Pred, sigma):
    n = len(sigma)
    a = np.empty(n)
    a.fill(70)
    sigma_clipped = np.maximum(sigma, a)
    delta = np.minimum(np.abs(FVC - FVC_Pred), 1000)
    eval_metric = -np.sqrt(2) * delta / sigma_clipped - np.log(
        np.sqrt(2) * sigma_clipped
    )
    return eval_metric




## === cell 25
sub_df = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
print(
    f"The sample submission contains: {sub_df.shape[0]} rows and {sub_df.shape[1]} columns."
)




## === cell 26
sub_df[["Patient", "Weeks"]] = sub_df.Patient_Week.str.split("_", expand=True)
sub_df = sub_df[["Patient", "Weeks", "Confidence", "Patient_Week"]]




## === cell 27
sub_df = sub_df.merge(test_x.drop("Weeks", axis=1), on="Patient", how="left")




## === cell 28
train_x["Source"] = "train"
sub_df["Source"] = "test"
data_df = pd.concat([train_x, sub_df], ignore_index=True)
data_df.reset_index(inplace=True, drop=True)
data_df.head()




## === cell 29
def get_baseline_week(df):
    _df = df.copy()
    _df["Weeks"] = _df["Weeks"].astype(int)
    _df.loc[_df.Source == "test", "min_week"] = np.nan
    _df["min_week"] = _df.groupby("Patient")["Weeks"].transform("min")
    _df["baselined_week"] = _df["Weeks"] - _df["min_week"]
    return _df




## === cell 30
data_df = get_baseline_week(data_df)
data_df.head()




## === cell 31
def get_baseline_FVC(df):
    _df = df.copy()
    base = _df.loc[_df.Weeks == _df.min_week]
    base = base[["Patient", "FVC"]].copy()
    base.columns = ["Patient", "base_FVC"]
    base["nb"] = 1
    base["nb"] = base.groupby("Patient")["nb"].transform("cumsum")
    base = base[base.nb == 1]
    base.drop("nb", axis=1, inplace=True)
    _df = _df.merge(base, on="Patient", how="left")
    _df.drop(["min_week"], axis=1, inplace=True)
    return _df




## === cell 32
duration_old = timeit.timeit(lambda: get_baseline_FVC(data_df), number=1)
duration_new = timeit.timeit(lambda: get_baseline_FVC(data_df), number=1)
print(f"Old version took {duration_old:.2f}s, new version {duration_new:.2f}s")




## === cell 33
data_df = get_baseline_FVC(data_df)
data_df.head()




## === cell 34
no_transform_attribs = ["Patient", "Weeks", "min_week"]
num_attribs = ["FVC", "Percent", "Age", "baselined_week", "base_FVC"]
cat_attribs = ["Sex", "SmokingStatus"]




## === cell 35
def own_MinMaxColumnScaler(df, columns):
    for col in columns:
        new_col_name = col + "_scld"
        col_min = df[col].min()
        col_max = df[col].max()
        df[new_col_name] = (df[col] - col_min) / (col_max - col_min)




## === cell 36
def own_OneHotColumnCreator(df, columns):
    for col in columns:
        for value in df[col].unique():
            df[value] = (df[col] == value).astype(int)




## === cell 37
own_MinMaxColumnScaler(data_df, num_attribs)
own_OneHotColumnCreator(data_df, cat_attribs)




## === cell 38
train_df = data_df.loc[data_df.Source == "train"]
sub = data_df.loc[data_df.Source == "test"]




## === cell 39
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def score(y_true, y_pred):
    """Custom metric matching the competition's scoring (negative value for minimisation)."""
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true - fvc_pred)
    delta = tf.minimum(delta, C2)
    sqrt2 = tf.sqrt(tf.constant(2.0))
    metric = -((delta / sigma_clip) * sqrt2 + tf.math.log(sigma_clip * sqrt2))
    return tf.reduce_mean(metric)


def mloss(_lambda):
    """Return a loss that uses only the score component when _lambda == 0."""
    if _lambda == 0.0:
        return lambda y_true, y_pred: score(y_true, y_pred)
    else:

        def qloss(y_true, y_pred):
            qs = [0.2, 0.50, 0.8]
            q = tf.constant(np.array([qs]), dtype=tf.float32)  # shape (1,3)
            e = y_true - y_pred
            v = tf.maximum(q * e, (q - 1) * e)
            return tf.reduce_mean(v)

        def loss(y_true, y_pred):
            return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(
                y_true, y_pred
            )

        return loss




## === cell 40
features_list = [
    "baselined_week_scld",
    "Percent_scld",
    "Age_scld",
    "base_FVC_scld",
    "Male",
    "Female",
    "Ex-smoker",
    "Never smoked",
    "Currently smokes",
]


def get_model():
    inp = layers.Input((len(features_list),), name="Patient")
    x = layers.Dense(128, activation="relu", name="d1")(inp)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(128, activation="relu", name="d2")(x)
    x = layers.Dropout(0.2)(x)
    p1 = layers.Dense(3, activation="relu", name="p1")(x)
    p2 = layers.Dense(3, activation="relu", name="p2")(x)
    preds = layers.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds")(
        [p1, p2]
    )
    model = models.Model(inp, preds, name="NeuralNet")
    model.compile(
        loss=mloss(0.0), optimizer=optimizers.Adam(learning_rate=0.001), metrics=[score]
    )
    return model




## === cell 41
neuralNet = get_model()
neuralNet.summary()




## === cell 42
y_fvc = train_df["FVC"].values.astype(float)
y_train = y_fvc
X_train = train_df[features_list].values
X_test = sub[features_list].values
train_preds = np.zeros((X_train.shape[0], 3))
test_preds = np.zeros((X_test.shape[0], 3))

NFOLDS = 5
gkf = GroupKFold(n_splits=NFOLDS)
groups = train_df["Patient"].values

for fold, (train_idx, val_idx) in enumerate(
    gkf.split(X_train, y_fvc, groups=groups), 1
):
    print(f"FOLD {fold}:")
    net = get_model()
    net.fit(
        X_train[train_idx],
        y_train[train_idx],
        batch_size=128,
        epochs=10,
        validation_data=(X_train[val_idx], y_train[val_idx]),
        verbose=0,
    )
    train_preds[val_idx] = net.predict(X_train[val_idx], batch_size=128, verbose=0)
    test_preds += net.predict(X_test, batch_size=128, verbose=0) / NFOLDS




## === cell 43
sigma_opt = mean_absolute_error(y_fvc, train_preds[:, 1])
sigma_uncertain = test_preds[:, 2] - test_preds[:, 0]
sigma_mean = np.mean(sigma_uncertain)
print(sigma_opt, sigma_mean)




## === cell 44
sub["FVC1"] = test_preds[:, 1]
sub["Confidence1"] = test_preds[:, 2] - test_preds[:, 0]




## === cell 45
submission = sub[["Patient_Week", "FVC", "Confidence"]].copy()

org_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
baseline_keys = set(org_test["Patient"] + "_" + org_test["Weeks"].astype(str))
baseline_fvc = org_test.groupby("Patient")["FVC"].first().to_dict()

for key in baseline_keys:
    mask = submission["Patient_Week"] == key
    submission.loc[mask, "FVC"] = org_test.loc[
        (org_test["Patient"] + "_" + org_test["Weeks"].astype(str)) == key, "FVC"
    ].values[0]
    submission.loc[mask, "Confidence"] = 70

mask = ~submission["Patient_Week"].isin(baseline_keys)
submission.loc[mask, "FVC"] = sub.loc[mask, "FVC1"].values
pred_conf = sub.loc[mask, "Confidence1"]
submission.loc[mask, "Confidence"] = np.clip(pred_conf, 70, None)

submission = submission[["Patient_Week", "FVC", "Confidence"]]




## === cell 46
submission.head()




## === cell 47
submission.to_csv("submission.csv", index=False)
