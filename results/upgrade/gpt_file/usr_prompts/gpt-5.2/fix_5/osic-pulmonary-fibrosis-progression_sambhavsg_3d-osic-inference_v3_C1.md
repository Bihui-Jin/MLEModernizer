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

-6.873693639104417

# 6. Current score

-24.19818

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -24.16078) has done: 'I fix the environment-breaking imports first by removing the optional `pydicom/cv2/PIL/matplotlib` imports (they trigger the protobuf `MessageFactory.GetPrototype` error here) and by not using CT images at all, since the current pipeline never successfully reaches that stage anyway. Next, I repair the pandas compatibility bug (`DataFrame.append` removed) and a few column-name/typing issues so the tabular feature pipeline runs end-to-end. Because the referenced external pretrained model file is missing, I keep the same “tabular → regression” core idea but train a small Keras MLP inside this notebook and then output predictions for every `Patient_Week`. Finally, I always write a valid `submission.csv` with exactly `Patient_Week,FVC,Confidence` and ensure prediction alignment matches the sample submission order.'
- What this solution (achieved -24.13413) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by forcing the pure-Python protobuf implementation before importing TensorFlow (this is the root cause of the `MessageFactory.GetPrototype` error). Then I keep your tabular-only pipeline and Keras MLP architecture unchanged, but make the training target consistent with the evaluation objective by predicting per-row FVC (using the per-row `Weeks` and `Percent` already present in the training data) and adding `Percent` back into the model features (it was computed/scaled but not used), which should move the score substantially toward the target without changing the overall approach. Finally, I ensure the submission is written in the exact sample order with valid numeric types and the required `.csv` suffix.'
- What this solution (achieved -24.28982) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *and* ensuring it’s set before any protobuf/TensorFlow import, plus adding a safe fallback so the notebook still runs even if TF remains unusable. If TF can’t be imported, I keep the same tabular-only “regression from clinical features” approach but switch to a lightweight NumPy least-squares linear regressor to produce valid predictions. I also fix a current feature bug: your `Base_Week/Base_FVC/Base_Percent` columns are being median-imputed in train because they’re NaN for most training rows; I correctly propagate each patient’s baseline values to all that patient’s weeks (score-improving while preserving the same core modeling intent). Finally, I always write a valid `submission.csv` with `Patient_Week,FVC,Confidence` in the exact sample order.'
- What this solution (achieved -24.19818) has done: 'I fix the crash in the first cell by moving the protobuf environment variables to the very top of the script (before any library imports) and by adding a more robust TensorFlow import fallback that cleanly disables TF when the protobuf incompatibility persists. Then, to move the score toward your target while preserving the same “tabular clinical regression” core logic, I correct the feature bug where `Base_Week/Base_FVC/Base_Percent/Typical_FVC` are missing for most training rows by computing each patient’s baseline from that patient’s `Weeks==0` row (or closest-to-0 if missing) and merging it back to all their weeks. Finally, I keep the existing model choices (Keras MLP if available, otherwise NumPy least-squares), but ensure the submission is fully aligned to `sample_submission.csv` and always writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

TF_AVAILABLE = False
try:
    import tensorflow as tf
    import tensorflow.keras.layers as L
    import tensorflow.keras.models as M

    tf.random.set_seed(SEED)
    TF_AVAILABLE = True
except Exception as e:
    print("WARNING: TensorFlow import failed; falling back to NumPy regressor.")
    print("TF import error:", repr(e))
    TF_AVAILABLE = False

from sklearn.preprocessing import MinMaxScaler



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
EPOCHS = 5
NUM_IMAGES = 140
BATCH_SIZE = 4
FOLDS = 5
IMAGE_DIM = (NUM_IMAGES, 55, 55)

COMP_DIR = "../input/osic-pulmonary-fibrosis-progression/"
TRAIN_PATH = "../input/osic-pulmonary-fibrosis-progression/train"
TEST_PATH = "../input/osic-pulmonary-fibrosis-progression/test"
SUB_PATH = "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"



## === cell 2
comp_dir = "../input/osic-pulmonary-fibrosis-progression"

train_data = pd.read_csv(os.path.join(comp_dir, "train.csv"))
sub = pd.read_csv(os.path.join(comp_dir, "sample_submission.csv"))
test_data = pd.read_csv(os.path.join(comp_dir, "test.csv"))

train_data.drop_duplicates(keep="first", inplace=True, subset=["Patient", "Weeks"])




## === cell 3
def make_patient_baseline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["abs_week"] = (df["Weeks"].astype(float)).abs()
    df = df.sort_values(["Patient", "abs_week", "Weeks"])
    base = df.drop_duplicates(subset=["Patient"], keep="first").copy()
    base = base.rename(
        columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
    )[["Patient", "Base_Week", "Base_FVC", "Base_Percent"]]
    base["Typical_FVC"] = (
        base["Base_FVC"].values / base["Base_Percent"].values
    ) * 100.0
    return base


train_base = make_patient_baseline(train_data)
train_data = train_data.merge(train_base, on="Patient", how="left")



## === cell 4
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub.drop(["Confidence"], axis=1)
sub = sub[["Patient", "Weeks", "Patient_Week"]]



## === cell 5
test_data = test_data.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
)
test_data["Typical_FVC"] = (
    test_data.Base_FVC.values / test_data.Base_Percent.values
) * 100.0
sub = sub.merge(test_data, how="left", on="Patient")



## === cell 6
train_data["Type"] = "train"
sub["Type"] = "test"



## === cell 7
data = pd.concat([train_data, sub], axis=0, ignore_index=True)



## === cell 8
prediction_col = ["FVC"]
Continuos_cols = [
    "Weeks",
    "Base_Week",
    "Base_FVC",
    "Typical_FVC",
    "Age",
    "Percent",
    "Base_Percent",
]
Categorical_cols = ["Sex", "SmokingStatus"]



## === cell 9
for c in Continuos_cols + Categorical_cols:
    if c not in data.columns:
        data[c] = np.nan

train_medians = data.loc[data["Type"] == "train", Continuos_cols].median(
    numeric_only=True
)
data[Continuos_cols] = data[Continuos_cols].fillna(train_medians)

for c in Categorical_cols:
    mode_val = data.loc[data["Type"] == "train", c].mode(dropna=True)
    mode_val = mode_val.iloc[0] if len(mode_val) else "Unknown"
    data[c] = data[c].fillna(mode_val)



## === cell 10
scaler = MinMaxScaler()
data[Continuos_cols] = scaler.fit_transform(data[Continuos_cols])



## === cell 11
data["Sex"] = data["Sex"].map({"Male": 0, "Female": 1}).fillna(0).astype(np.float32)

smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}
data["SmokingStatus"] = (
    data["SmokingStatus"].map(smoke_map).fillna(0).astype(np.float32)
)



## === cell 12
x_cols = ["Weeks", "Base_Week", "Base_FVC", "Percent", "Sex", "Age"]

x_train = data.loc[data["Type"] == "train", x_cols].values.astype(np.float32)
y_train = (
    data.loc[data["Type"] == "train", prediction_col]
    .values.astype(np.float32)
    .reshape(-1)
)
x_test = data.loc[data["Type"] == "test", x_cols].values.astype(np.float32)



## === cell 13
if TF_AVAILABLE:
    inp = L.Input(shape=(x_train.shape[1],), name="tabular")
    x = L.Dense(64, activation="relu")(inp)
    x = L.Dense(64, activation="relu")(x)
    out = L.Dense(1, name="fvc")(x)

    model = M.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32, verbose=0)

    pred_fvc = (
        model.predict(x_test, batch_size=256, verbose=0).reshape(-1).astype(np.float32)
    )
else:
    X = np.concatenate(
        [np.ones((x_train.shape[0], 1), dtype=np.float32), x_train], axis=1
    )
    XtX = X.T @ X
    Xty = X.T @ y_train.astype(np.float32)
    beta = np.linalg.pinv(XtX) @ Xty
    Xte = np.concatenate(
        [np.ones((x_test.shape[0], 1), dtype=np.float32), x_test], axis=1
    )
    pred_fvc = (Xte @ beta).astype(np.float32)



## === cell 14
pred_conf = np.full_like(pred_fvc, 70.0, dtype=np.float32)

sub_pred = sub.copy()
sub_pred["FVC"] = pred_fvc
sub_pred["Confidence"] = pred_conf

sample_pw = pd.read_csv(os.path.join(comp_dir, "sample_submission.csv"))[
    ["Patient_Week"]
]
subm = sample_pw.merge(
    sub_pred[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week", how="left"
)

if subm["FVC"].isna().any():
    subm["FVC"] = subm["FVC"].fillna(float(np.median(y_train)))
if subm["Confidence"].isna().any():
    subm["Confidence"] = subm["Confidence"].fillna(70.0)

subm["FVC"] = subm["FVC"].astype(float)
subm["Confidence"] = subm["Confidence"].astype(float)

subm.to_csv("submission.csv", index=False)

print(subm.head())
print("Wrote submission.csv with shape:", subm.shape)
print("TF_AVAILABLE:", TF_AVAILABLE)
