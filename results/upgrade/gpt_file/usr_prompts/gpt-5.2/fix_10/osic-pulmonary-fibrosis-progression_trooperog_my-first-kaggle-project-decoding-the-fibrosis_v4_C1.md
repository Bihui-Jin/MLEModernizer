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

-7.28953

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -24.19477) has done: 'I remove the broken/unused `dicom` installation/import that triggers the protobuf `MessageFactory` error, and update the DICOM reader to the current `pydicom.dcmread` API so scans load correctly. I also fix pandas deprecations (`DataFrame.append`) so `data_df` is created, and make `corr()`/plotting cells robust to non-numeric columns so they don’t crash the run. Finally, I correct a few logic bugs that prevent training/inference (Weeks dtype, missing optimizer due to earlier failure, and the custom metric helper using `np.abs` incorrectly), and ensure a valid `submission.csv` with the required columns is always written.'
- What this solution (achieved -24.19686) has done: 'I fix the environment/runtime issue that triggers the protobuf `MessageFactory.GetPrototype` error by avoiding the unnecessary `pydicom` import when not essential to model training, and making DICOM-dependent demo cells safely skippable if DICOM loading fails. I also fix the `NotImplementedError: numpy() is only available when eager execution is enabled` by forcing TensorFlow eager execution and compiling the model with `run_eagerly=True`, which restores compatibility with the custom loss/metric and KFold training loop. Finally, I keep the model and training logic intact, but adjust only the optimizer learning rate from an obviously too-large `0.1` to `1e-3` (same Adam optimizer) to move the score upward toward the target, and ensure `submission.csv` is always written with the correct columns and row count.'
- What this solution (achieved -7.28806) has done: 'The timeout is dominated by TensorFlow running eagerly plus doing 10-fold cross-validation with 1000 epochs each, which makes training orders of magnitude slower than necessary. I keep the exact same model, losses, folds, epochs, and data/feature logic, but switch execution back to graph mode (`run_eagerly=False`, disable `run_functions_eagerly`) and wrap the custom losses/metric in `@tf.function` to preserve identical semantics while drastically reducing Python overhead. I also remove/guard heavy EDA plotting and `os.walk`/DICOM demo work (which doesn’t affect training/inference) so runtime is spent only on what impacts the submission. Finally, I avoid redundant per-fold evaluate calls (they are extra forward passes not used downstream) while keeping predictions and fold training identical.'
- What this solution (achieved -7.29468) has done: 'I fix the earliest blocker: TensorFlow fails to import because the code forces the protobuf C++ implementation (`cpp`), which isn’t available in this environment; switching to the pure-Python protobuf implementation allows TF to import and unblocks the whole pipeline. After that, the downstream NameErrors disappear because the later cells depend on variables defined after the TF import. I also restore the missing `timeit` import used later and keep the rest of the modeling/training logic unchanged so the score behavior stays consistent. Finally, I ensure the notebook always writes a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved -7.30334) has done: 'I fix the protobuf/TensorFlow import crash that’s causing the `MessageFactory.GetPrototype` error by forcing the pure-Python protobuf implementation early and, if needed, falling back to a safe TF import order. To move the score *downward toward the target* (your current score is better than target), I make only a minimal calibration change in post-processing: blend the final FVC prediction slightly toward each patient’s baseline FVC using the already-available baseline column, which typically reduces overfitting/variance and should nudge the metric downward without changing the model/training. I keep the model, loss, folds, and epochs unchanged, and ensure a valid `submission.csv` with correct columns/rowcount is always written. All other changes are guards to keep the notebook running end-to-end in the Kaggle environment.'
- What this solution (achieved -7.28953) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf backend earlier and preventing any accidental early TensorFlow import before that environment variable is set. Then I keep the modeling/training logic intact, but make a tiny, controlled post-processing adjustment (slightly stronger baseline blending) to nudge the score downward toward your target (your current score is better than target, and higher is better). Finally, I add a safe fallback so even if TF still can’t import for any reason, the script still produces a valid `submission.csv` in the required format (score be worse, but it guarantees an end-to-end run and a valid CSV).'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

np.random.seed(42)

import glob as glob

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px, go = None, None

try:
    import imageio
except Exception:
    imageio = None

try:
    from IPython.display import Image
except Exception:
    Image = None

pydicom = None
try:
    from skimage import morphology, segmentation, measure
except Exception:
    morphology = segmentation = measure = None

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from timeit import timeit

INPUT_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"
for fn in ["train.csv", "test.csv", "sample_submission.csv"]:
    p = os.path.join(INPUT_DIR, fn)
    if os.path.exists(p):
        print(p)

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf
    import tensorflow.keras.backend as K
    import tensorflow.keras.layers as Layers
    import tensorflow.keras.models as Models

    tf.random.set_seed(42)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    tf.config.run_functions_eagerly(False)
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)
    print(
        "WARNING: TensorFlow import failed; will use fallback submission. Error:",
        TF_IMPORT_ERROR,
    )




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = INPUT_DIR

train_x = pd.read_csv(f"{DATA_DIR}/train.csv")
print(
    "the no of rows is {} and the no of columns is {} ".format(
        train_x.shape[0], train_x.shape[1]
    )
)
train_x.head()




## === cell 2
train_x.describe()




## === cell 3
test_x = pd.read_csv(f"{DATA_DIR}/test.csv")
print(
    "the no of rows is {} and the no of columns is {} ".format(
        test_x.shape[0], test_x.shape[1]
    )
)
test_x.head()




## === cell 4
test_x.describe()




## === cell 5
if os.environ.get("RUN_EDA", "0") == "1" and plt is not None and sns is not None:
    plt.figure(figsize=(5, 4))
    sns.countplot(x="Sex", data=train_x)
    plt.tight_layout()
    plt.show()




## === cell 6
new_df = train_x.groupby(
    [train_x.Patient, train_x.Age, train_x.Sex, train_x.SmokingStatus]
)["Patient"].count()

new_df.index = new_df.index.set_names(["id", "Age", "Sex", "SmokingStatus"])

new_df = new_df.reset_index()
new_df.rename(columns={"Patient": "freq"}, inplace=True)

if os.environ.get("RUN_EDA", "0") == "1" and px is not None:
    fig = px.bar(new_df, x="id", y="freq", color="freq")
    fig.update_layout(
        xaxis={"categoryorder": "total ascending"},
        title="Distribution of images for each patient",
    )
    fig.update_xaxes(showticklabels=False)
    fig.show()




## === cell 7
if os.environ.get("RUN_EDA", "0") == "1" and px is not None:
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
if os.environ.get("RUN_EDA", "0") == "1" and plt is not None and sns is not None:
    plt.figure(figsize=(6, 4))
    sns.countplot(x="SmokingStatus", data=train_x)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()




## === cell 9
if os.environ.get("RUN_EDA", "0") == "1" and px is not None:
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

    fig.update_layout(
        title="Distribution of Age w.r.t. SmokingStatus for unique patients"
    )

    fig.update_traces(marker_line_color="black", marker_line_width=1.5, opacity=0.85)

    fig.show()




## === cell 10
if os.environ.get("RUN_EDA", "0") == "1" and plt is not None and sns is not None:
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Sex", hue="SmokingStatus", data=train_x)
    plt.tight_layout()
    plt.show()




## === cell 11
if os.environ.get("RUN_EDA", "0") == "1" and px is not None:
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
if os.environ.get("RUN_EDA", "0") == "1" and plt is not None and sns is not None:
    plt.figure(figsize=(7, 5))
    corr = train_x.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap=plt.cm.cool)
    plt.tight_layout()
    plt.show()




## === cell 13
if os.environ.get("RUN_EDA", "0") == "1" and plt is not None and sns is not None:
    a = sns.histplot(train_x["FVC"], kde=True, color="r")
    a.set_title("Distribution plot of FVC", color="g", fontsize=18)
    plt.tight_layout()
    plt.show()




## === cell 14
if os.environ.get("RUN_EDA", "0") == "1" and plt is not None and sns is not None:
    b = sns.histplot(train_x["Percent"], kde=True, color="g")
    b.set_title("Distribution plot of Percent", color="r", fontsize=18)
    plt.tight_layout()
    plt.show()




## === cell 15
if os.environ.get("RUN_EDA", "0") == "1" and px is not None:
    data = px.bar(
        x=list(train_x["Weeks"].value_counts().keys()),
        y=list(train_x["Weeks"].value_counts().values),
    )
    data.show()




## === cell 16
if os.environ.get("RUN_EDA", "0") == "1" and px is not None:
    fig = px.line(
        train_x,
        "Weeks",
        "FVC",
        line_group="Patient",
        color="Sex",
        title="Pulmonary Condition Progression by Sex",
    )
    fig.update_traces(mode="lines+markers")
    fig.show()




## === cell 17
if os.environ.get("RUN_EDA", "0") == "1" and px is not None:
    fig = px.line(
        train_x,
        "Weeks",
        "FVC",
        line_group="Patient",
        color="SmokingStatus",
        title="Pulmonary Condition Progression by Smoking Status",
    )
    fig.update_traces(mode="lines+markers")
    fig.show()




## === cell 18
print(
    "The Number of Unique Patients in training data are : {}".format(
        len(train_x["Patient"].unique())
    )
)




## === cell 19
data_path = f"{DATA_DIR}/train/"
patients = os.listdir(data_path)
patients.sort()
print("Number of patient folders:", len(patients))
print("First patient folder:", patients[0])




## === cell 20
def load_scan(path):
    """
    Loads scans from a folder and into a list.
    """
    global pydicom
    if pydicom is None:
        try:
            import pydicom as _pydicom

            pydicom = _pydicom
        except Exception as e:
            raise RuntimeError(f"pydicom import failed: {e}")

    files = [os.path.join(path, s) for s in os.listdir(path)]
    slices = [pydicom.dcmread(f, force=True) for f in files]
    slices.sort(key=lambda x: int(getattr(x, "InstanceNumber", 0)))

    if len(slices) >= 2:
        try:
            slice_thickness = np.abs(
                float(slices[0].ImagePositionPatient[2])
                - float(slices[1].ImagePositionPatient[2])
            )
        except Exception:
            slice_thickness = np.abs(
                float(getattr(slices[0], "SliceLocation", 0.0))
                - float(getattr(slices[1], "SliceLocation", 0.0))
            )
    else:
        slice_thickness = 1.0

    for s in slices:
        try:
            s.SliceThickness = slice_thickness
        except Exception:
            pass
    return slices


def get_pixels_hu(scans):
    """
    Converts raw images to Hounsfield Units (HU).
    """
    image = np.stack([s.pixel_array for s in scans]).astype(np.int16)
    image[image == -2000] = 0

    intercept = float(getattr(scans[0], "RescaleIntercept", 0.0))
    slope = float(getattr(scans[0], "RescaleSlope", 1.0))

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)




## === cell 21
if os.environ.get("RUN_DICOM_DEMO", "0") == "1":
    try:
        test_patient_scans = load_scan(os.path.join(data_path, patients[0]))
        test_patient_images = get_pixels_hu(test_patient_scans)
        print(
            "Loaded patient:", patients[0], "volume shape:", test_patient_images.shape
        )

        if plt is not None:
            plt.figure(figsize=(5, 5))
            plt.imshow(
                test_patient_images[len(test_patient_images) // 2], cmap=plt.cm.bone
            )
            plt.title("Middle slice")
            plt.axis("off")
            plt.show()
    except Exception as e:
        print("Skipping DICOM demo cell due to error:", repr(e))




## === cell 22
def set_lungwin(img, hu=[-1200.0, 600.0]):
    lungwin = np.array(hu)
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype("uint8")
    return newimg


if os.environ.get("RUN_DICOM_DEMO", "0") == "1":
    try:
        if imageio is None or Image is None:
            raise RuntimeError("imageio/IPython not available")
        scans = load_scan(f"{DATA_DIR}/train/ID00007637202177411956430/")
        scan_array = set_lungwin(get_pixels_hu(scans))
        imageio.mimsave("/tmp/gif.gif", scan_array[:30], duration=0.02)
        Image(filename="/tmp/gif.gif", format="png")
    except Exception as e:
        print("Skipping GIF creation due to error:", repr(e))




## === cell 23
print("train_x.shape:", train_x.shape)




## === cell 24
print("test_x.shape:", test_x.shape)




## === cell 25
def eval_metric(FVC, FVC_Pred, sigma):
    sigma = np.asarray(sigma, dtype=float)
    FVC = np.asarray(FVC, dtype=float)
    FVC_Pred = np.asarray(FVC_Pred, dtype=float)

    sigma_clipped = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(FVC - FVC_Pred), 1000.0)
    metric = -np.sqrt(2) * delta / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)
    return metric




## === cell 26
sub_df = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
print(
    f"The sample submission contains: {sub_df.shape[0]} rows and {sub_df.shape[1]} columns."
)
sub_df.head()




## === cell 27
sub_df[["Patient", "Weeks"]] = sub_df.Patient_Week.str.split("_", expand=True)
sub_df = sub_df[["Patient", "Weeks", "Confidence", "Patient_Week"]].copy()
sub_df["Weeks"] = sub_df["Weeks"].astype(int)
sub_df.head()




## === cell 28
sub_df = sub_df.merge(test_x.drop("Weeks", axis=1), on="Patient", how="left")
sub_df.head()




## === cell 29
train_x = train_x.copy()
sub_df = sub_df.copy()

train_x["Source"] = "train"
sub_df["Source"] = "test"

data_df = pd.concat([train_x, sub_df], ignore_index=True)
data_df.head()




## === cell 30
def get_baseline_week(df):
    _df = df.copy()
    _df["Weeks"] = _df["Weeks"].astype(int)
    _df["min_week"] = _df.groupby("Patient")["Weeks"].transform("min")
    _df["baselined_week"] = _df["Weeks"] - _df["min_week"]
    return _df


data_df = get_baseline_week(data_df)
data_df.head()




## === cell 31
def get_baseline_FVC_old(df):
    _df = df.copy()
    baseline = _df.loc[_df.Weeks == _df.min_week]
    baseline = baseline[["Patient", "FVC"]].copy()
    baseline.columns = ["Patient", "base_FVC"]

    for idx in _df.index:
        patient_id = _df.at[idx, "Patient"]
        _df.at[idx, "base_FVC"] = baseline.loc[
            baseline.Patient == patient_id, "base_FVC"
        ].iloc[0]
    _df.drop(["min_week"], axis=1, inplace=True, errors="ignore")

    return _df




## === cell 32
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
    _df.drop(["min_week"], axis=1, inplace=True, errors="ignore")

    return _df




## === cell 33
def old_baseline_FVC():
    return get_baseline_FVC_old(data_df)


def new_baseline_FVC():
    return get_baseline_FVC(data_df)


duration_old = timeit(old_baseline_FVC, number=1)
duration_new = timeit(new_baseline_FVC, number=1)
print(
    f"Old baseline FVC took {duration_old:.2f} sec, vectorized version took {duration_new:.3f} sec."
)




## === cell 34
data_df = get_baseline_FVC(data_df)
data_df.head()




## === cell 35
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

no_transform_attribs = ["Patient", "Weeks"]
num_attribs = ["FVC", "Percent", "Age", "baselined_week", "base_FVC"]
cat_attribs = ["Sex", "SmokingStatus"]




## === cell 36
def own_MinMaxColumnScaler(df, columns):
    """Adds columns with scaled numeric values to range [0, 1]."""
    for col in columns:
        new_col_name = col + "_scld"
        col_min = df[col].min()
        col_max = df[col].max()
        denom = (col_max - col_min) if (col_max - col_min) != 0 else 1.0
        df[new_col_name] = (df[col] - col_min) / denom




## === cell 37
def own_OneHotColumnCreator(df, columns):
    """OneHot encodes categorical features. Adds a column for each unique value per column."""
    for col in columns:
        for value in df[col].dropna().unique():
            df[str(value)] = (df[col] == value).astype(int)




## === cell 38
own_MinMaxColumnScaler(data_df, num_attribs)
own_OneHotColumnCreator(data_df, cat_attribs)

data_df[data_df.Source != "train"].head()




## === cell 39
train_df = data_df.loc[data_df.Source == "train"].copy()
sub = data_df.loc[data_df.Source == "test"].copy()
print(train_df.shape, sub.shape)




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
for c in features_list:
    if c not in data_df.columns:
        data_df[c] = 0
        train_df[c] = 0
        sub[c] = 0

EPOCHS = 1000
BATCH_SIZE = 128
_lambda = 0.8  # default as provided

BASE_LR = 1e-3

lr_start = 0.0001
lr_max = 0.0001 * BATCH_SIZE
lr_min = 0.00001
lr_ramp_ep = EPOCHS * 0.3
lr_sus_ep = 0
lr_decay = 0.992


def test_the_scheduler(epoch):
    if epoch < lr_ramp_ep:
        lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
    elif epoch < lr_ramp_ep + lr_sus_ep:
        lr = lr_max
    else:
        lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min
    return lr


if os.environ.get("RUN_EDA", "0") == "1" and plt is not None:
    rng = list(range(EPOCHS))
    y_lr = [test_the_scheduler(x) for x in rng]
    plt.plot(rng, y_lr)
    plt.title("Learning rate schedule (not applied)")
    plt.show()
    print(
        "Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(
            y_lr[0], max(y_lr), y_lr[-1]
        )
    )




## === cell 41
if TF_AVAILABLE:
    C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")

    @tf.function
    def score(y_true, y_pred):
        """Calculate the competition metric-like term used in loss (as originally intended)."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        sigma = y_pred[:, 2] - y_pred[:, 0]
        fvc_pred = y_pred[:, 1]

        sigma_clip = tf.maximum(sigma, C1)
        delta = tf.abs(y_true[:, 0] - fvc_pred)
        delta = tf.minimum(delta, C2)
        sq2 = tf.sqrt(tf.cast(2.0, dtype=tf.float32))
        metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
        return K.mean(metric)

    @tf.function
    def qloss(y_true, y_pred):
        """Calculate Pinball loss."""
        qs = [0.2, 0.50, 0.8]
        q = tf.constant(np.array([qs]), dtype=tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        e = y_true - y_pred
        v = tf.maximum(q * e, (q - 1) * e)
        return K.mean(v)

    def mloss(_lambda):
        """Combine Score and qloss."""

        @tf.function
        def loss(y_true, y_pred):
            return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(
                y_true, y_pred
            )

        return loss




## === cell 42
if TF_AVAILABLE:

    def get_model():
        """Creates and returns a model (preserved architecture)."""
        inp = Layers.Input((len(features_list),), name="Patient")
        x = Layers.Dense(128, activation="relu", name="d1")(inp)
        x = Layers.Dropout(0.25)(x)
        x = Layers.Dense(128, activation="relu", name="d2")(x)
        x = Layers.Dropout(0.2)(x)
        p1 = Layers.Dense(3, activation="relu", name="p1")(x)
        p2 = Layers.Dense(3, activation="relu", name="p2")(x)
        preds = Layers.Lambda(lambda z: z[0] + tf.cumsum(z[1], axis=1), name="preds")(
            [p1, p2]
        )

        model = Models.Model(inp, preds, name="NeuralNet")

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=BASE_LR, beta_1=0.9, beta_2=0.999
        )

        model.compile(
            loss=mloss(_lambda),
            optimizer=optimizer,
            metrics=[score],
            run_eagerly=False,
            steps_per_execution=32,
        )
        return model




## === cell 43
if TF_AVAILABLE:
    neuralNet = get_model()
    neuralNet.summary()




## === cell 44
if TF_AVAILABLE:
    y = train_df["FVC"].values.astype(float).reshape(-1, 1)
    y = np.repeat(y, 3, axis=1)

    X_train = train_df[features_list].values.astype(np.float32)
    X_test = sub[features_list].values.astype(np.float32)

    train_preds = np.zeros((X_train.shape[0], 3), dtype=np.float32)
    test_preds = np.zeros((X_test.shape[0], 3), dtype=np.float32)

    print("X_train:", X_train.shape, "y:", y.shape, "X_test:", X_test.shape)




## === cell 45
if TF_AVAILABLE:
    AUTOTUNE = tf.data.AUTOTUNE

    def make_ds(X, Y=None, batch_size=128, shuffle=False, seed=42):
        if Y is None:
            ds = tf.data.Dataset.from_tensor_slices(X)
            ds = ds.batch(batch_size, drop_remainder=False)
            ds = ds.cache()
            ds = ds.prefetch(AUTOTUNE)
            return ds
        ds = tf.data.Dataset.from_tensor_slices((X, Y))
        if shuffle:
            ds = ds.shuffle(
                buffer_size=len(X), seed=seed, reshuffle_each_iteration=True
            )
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.cache()
        ds = ds.prefetch(AUTOTUNE)
        return ds

    test_ds = make_ds(X_test, None, batch_size=BATCH_SIZE, shuffle=False)




## === cell 46
if TF_AVAILABLE:
    NFOLDS = 10
    gkf = GroupKFold(n_splits=NFOLDS)
    groups = train_df["Patient"].values

    count = 0
    for train_idx, val_idx in gkf.split(X_train, y, groups=groups):
        count += 1
        print(f"FOLD {count}:")

        net = get_model()

        train_ds = make_ds(
            X_train[train_idx], y[train_idx], batch_size=BATCH_SIZE, shuffle=False
        )
        val_ds = make_ds(
            X_train[val_idx], y[val_idx], batch_size=BATCH_SIZE, shuffle=False
        )

        net.fit(
            train_ds,
            epochs=EPOCHS,
            validation_data=val_ds,
            verbose=0,
        )

        train_preds[val_idx] = net.predict(val_ds, verbose=0)
        test_preds += net.predict(test_ds, verbose=0) / NFOLDS




## === cell 47
if TF_AVAILABLE:
    y_mid = y[:, 1]
    sigma_opt = mean_absolute_error(y_mid, train_preds[:, 1])
    sigma_uncertain = train_preds[:, 2] - train_preds[:, 0]
    sigma_mean = np.mean(sigma_uncertain)
    print(
        "sigma_opt (MAE):", sigma_opt, "sigma_mean (predicted spread mean):", sigma_mean
    )
else:
    sigma_opt = 200.0  # fallback, only used if TF unavailable




## === cell 48
sub.head()




## === cell 49
if TF_AVAILABLE:
    sub["FVC1"] = test_preds[:, 1]

    pred_spread = (test_preds[:, 2] - test_preds[:, 0]).astype(np.float32)
    conf_blend = 0.5 * pred_spread + 0.5 * float(sigma_opt)
    sub["Confidence1"] = conf_blend

    submission = sub[["Patient_Week", "FVC1", "Confidence1"]].copy()
    submission.rename(
        columns={"FVC1": "FVC", "Confidence1": "Confidence"}, inplace=True
    )

    submission["Confidence"] = submission["Confidence"].clip(lower=70)
else:
    submission = sub[["Patient_Week"]].copy()
    if "base_FVC" in sub.columns:
        submission["FVC"] = sub["base_FVC"].astype(float).values
    else:
        submission["FVC"] = 2000.0
    submission["Confidence"] = 200.0
    submission["Confidence"] = submission["Confidence"].clip(lower=70)

submission.head(10)




## === cell 50
submission.describe().T




## === cell 51
BLEND_TO_BASELINE = 0.25  # was 0.15

if "base_FVC" in sub.columns:
    submission["FVC"] = (1.0 - BLEND_TO_BASELINE) * submission["FVC"].astype(
        float
    ) + BLEND_TO_BASELINE * sub["base_FVC"].astype(float).values




## === cell 52
org_test = pd.read_csv(f"{DATA_DIR}/test.csv")
org_test["Weeks"] = org_test["Weeks"].astype(int)
org_test["Patient_Week"] = (
    org_test["Patient"].astype(str) + "_" + org_test["Weeks"].astype(str)
)

submission = submission.merge(
    org_test[["Patient_Week", "FVC"]].rename(columns={"FVC": "FVC_base"}),
    on="Patient_Week",
    how="left",
)
mask = submission["FVC_base"].notnull()
submission.loc[mask, "FVC"] = submission.loc[mask, "FVC_base"].astype(float)
submission.loc[mask, "Confidence"] = 70.0
submission.drop(columns=["FVC_base"], inplace=True)




## === cell 53
assert (
    submission.shape[0] == sub_df.shape[0]
), "Row count mismatch vs sample_submission."
assert list(submission.columns) == [
    "Patient_Week",
    "FVC",
    "Confidence",
], "Wrong submission columns."
assert submission["Patient_Week"].isnull().sum() == 0, "Missing Patient_Week."
assert submission["FVC"].isnull().sum() == 0, "Missing FVC predictions."
assert submission["Confidence"].isnull().sum() == 0, "Missing Confidence predictions."

submission.head()




## === cell 54
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
if not TF_AVAILABLE:
    print(
        "NOTE: TensorFlow was unavailable; wrote fallback submission. TF error was:",
        TF_IMPORT_ERROR,
    )
