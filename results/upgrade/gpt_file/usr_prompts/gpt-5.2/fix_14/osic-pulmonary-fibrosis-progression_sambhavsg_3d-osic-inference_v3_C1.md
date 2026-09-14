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

-6.873693639104417

# 6. Current score

-9.03796

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.16078) has done: 'I fix the environment-breaking imports first by removing the optional `pydicom/cv2/PIL/matplotlib` imports (they trigger the protobuf `MessageFactory.GetPrototype` error here) and by not using CT images at all, since the current pipeline never successfully reaches that stage anyway. Next, I repair the pandas compatibility bug (`DataFrame.append` removed) and a few column-name/typing issues so the tabular feature pipeline runs end-to-end. Because the referenced external pretrained model file is missing, I keep the same “tabular → regression” core idea but train a small Keras MLP inside this notebook and then output predictions for every `Patient_Week`. Finally, I always write a valid `submission.csv` with exactly `Patient_Week,FVC,Confidence` and ensure prediction alignment matches the sample submission order.'
- What this solution (achieved -24.13413) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by forcing the pure-Python protobuf implementation before importing TensorFlow (this is the root cause of the `MessageFactory.GetPrototype` error). Then I keep your tabular-only pipeline and Keras MLP architecture unchanged, but make the training target consistent with the evaluation objective by predicting per-row FVC (using the per-row `Weeks` and `Percent` already present in the training data) and adding `Percent` back into the model features (it was computed/scaled but not used), which should move the score substantially toward the target without changing the overall approach. Finally, I ensure the submission is written in the exact sample order with valid numeric types and the required `.csv` suffix.'
- What this solution (achieved -24.28982) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *and* ensuring it’s set before any protobuf/TensorFlow import, plus adding a safe fallback so the notebook still runs even if TF remains unusable. If TF can’t be imported, I keep the same tabular-only “regression from clinical features” approach but switch to a lightweight NumPy least-squares linear regressor to produce valid predictions. I also fix a current feature bug: your `Base_Week/Base_FVC/Base_Percent` columns are being median-imputed in train because they’re NaN for most training rows; I correctly propagate each patient’s baseline values to all that patient’s weeks (score-improving while preserving the same core modeling intent). Finally, I always write a valid `submission.csv` with `Patient_Week,FVC,Confidence` in the exact sample order.'
- What this solution (achieved -24.19818) has done: 'I fix the crash in the first cell by moving the protobuf environment variables to the very top of the script (before any library imports) and by adding a more robust TensorFlow import fallback that cleanly disables TF when the protobuf incompatibility persists. Then, to move the score toward your target while preserving the same “tabular clinical regression” core logic, I correct the feature bug where `Base_Week/Base_FVC/Base_Percent/Typical_FVC` are missing for most training rows by computing each patient’s baseline from that patient’s `Weeks==0` row (or closest-to-0 if missing) and merging it back to all their weeks. Finally, I keep the existing model choices (Keras MLP if available, otherwise NumPy least-squares), but ensure the submission is fully aligned to `sample_submission.csv` and always writes a valid `submission.csv`.'
- What this solution (achieved -24.30643) has done: 'I fix the crash in the first cell by setting the protobuf environment variables before any library imports and additionally forcing the pure-Python protobuf module early (this prevents the `MessageFactory.GetPrototype` error). Then I make one minimal, score-improving correction to the feature pipeline: you already scale and impute `Typical_FVC`/`Base_Percent` but they were accidentally omitted from `x_cols`, so I include them without changing the model architecture or training loop. Finally, I keep the same submission-building logic but ensure the output is always numeric and written to `submission.csv` with the required columns and order.'
- What this solution (achieved -8.69002) has done: 'I fix the environment-breaking protobuf/TensorFlow crash by ensuring the pure-Python protobuf implementation is forced before any protobuf-related import and by patching the missing `MessageFactory.GetPrototype` method (a known incompatibility in some Kaggle images). This unblocks the script so it always runs end-to-end, using your existing TensorFlow MLP when available and otherwise falling back to the existing NumPy regressor (core logic preserved). I also make one score-relevant but minimal calibration improvement: use an out-of-fold residual-based confidence estimate (clipped to the metric’s 70 minimum) instead of a constant 70, which should move the score toward your target without changing the model architecture or training approach. Submission writing/format/order be kept exactly aligned to `sample_submission.csv` and always saved as `submission.csv`.'
- What this solution (achieved -8.68459) has done: 'I fix the immediate crash by making the protobuf patch robust: instead of referencing `MessageFactory.GetPrototype` in a way that can still raise `AttributeError` during TensorFlow’s internal imports, I (1) force the pure-Python protobuf implementation early, and (2) safely add a `GetPrototype` shim on both `google.protobuf.message_factory.MessageFactory` and any already-imported factory instance, but only if missing. This is a correctness/stability fix that unblocks TensorFlow so your existing Keras MLP path can run (which should keep or improve score vs the NumPy fallback). I also keep the rest of the pipeline unchanged (features, model, training loop, and submission formatting), only adjusting the import/patch ordering so it runs end-to-end and writes `submission.csv`.'
- What this solution (achieved -8.69486) has done: 'You’re currently below the target (−8.68459 vs −6.87369; higher is better), so we make the smallest changes that should legitimately improve the metric without altering the model architecture or training approach. The key score lever in this competition is the `Confidence` (sigma): using a single global sigma is suboptimal, so we keep the same FVC predictor but estimate a per-patient sigma from out-of-fold residuals and then map it to each test patient (fallback to global median if unseen). We also ensure predictions are in sensible units by inverse-transforming the scaled FVC outputs back to ml (right now the model is trained on unscaled y but can still produce reasonable values; this just makes it robust if any scaling/typing mismatch occurs) and clip confidence to at least 70 per metric. These are calibration/post-processing changes aligned to the evaluation and should move the score toward your target while keeping core logic intact.'
- What this solution (achieved -13.69792) has done: 'Your current gap to the target is sizable (−8.69486 vs −6.87369; higher is better), so we should improve without changing the core model or training loop. The largest remaining lever is that you trained on raw FVC while your features were MinMax-scaled, which makes the regression numerically ill-conditioned and typically hurts generalization; we minimally fix this by scaling `y_train` with a separate scaler and inverse-transforming predictions back to ml. We also compute the confidence (sigma) in the same unit (ml) from out-of-fold residuals after inverse-transform, so the metric’s sigma term is correctly calibrated. Everything else (features, architecture, epochs, loss, submission format/order) stays the same.'
- What this solution (achieved -9.92151) has done: 'You’re currently far below the target (−13.70 vs −6.87; higher is better), so we should make the smallest changes that legitimately improve the metric without changing the model or training procedure. The biggest remaining issue is that your test-time `Percent` feature is missing for most rows (it’s NaN in `sample_submission` rows after merge and gets filled by a global median), which makes predictions much less patient-specific; we compute `Percent` for test rows from existing baseline-derived fields as `Percent = 100 * FVC / Typical_FVC`, consistent with the dataset definition. We also compute a more stable, competition-aligned confidence by using out-of-fold (FOLDS=5) residuals from the same MLP architecture (not changing layers/loss/epochs), aggregating per-patient sigmas and a global fallback, and then using that for all test predictions. These two changes preserve core logic (tabular MLP regression + residual-based sigma) but should move the score substantially toward your target.'
- What this solution (achieved -10.18503) has done: 'We’re currently below the target (−9.92 vs −6.87; higher is better), so the smallest safe way to improve is to better align training/inference with what the test actually contains and how the metric is scored. I keep the same tabular-MLP + residual-based confidence core, but (1) define each patient’s baseline strictly from `Weeks==0` when available (matching the test baseline measurement), and (2) train/predict only on “future” weeks (`Weeks >= 0`) since test asks for post-baseline progression; this reduces a train/test mismatch without changing architecture or loss. Finally, I compute `Percent` for test rows deterministically from baseline (avoiding unintended NaNs/infs) and keep the same per-patient/global sigma logic, just using residuals from the aligned training subset.'
- What this solution (achieved -9.69472) has done: 'Your current score is below the target (−10.185 vs −6.874; higher is better), so we should make small, metric-aligned improvements without changing the MLP architecture or training loop. The biggest low-risk gain here is to stop using “scaled Weeks” (0–1) as if it were in real week units at test time: we keep scaling for the model input, but compute the final FVC prediction using the patient’s baseline FVC plus a learned per-week slope and multiply by the *real* week delta (in weeks), which better matches the problem structure. We learn that slope from the same model by adding a tiny post-hoc linear correction using train residuals vs. (Weeks−Base_Week), preserving the core tabular regression while improving extrapolation to future weeks. We also make confidence (sigma) depend mildly on how far the predicted week is from baseline (farther weeks → higher uncertainty), which usually improves this competition’s LaplaceLL without changing loss/architecture.'
- What this solution (achieved -9.03796) has done: 'I fix the KeyError by removing the incorrect re-merge that creates duplicate baseline columns and instead derive baseline FVC in ml directly from the already-computed `train_base`/`test_data` dictionaries. This unblocks training/inference end-to-end without changing your model architecture, training loop, or feature set. I also keep the existing slope-correction and confidence-calibration logic intact, only making the baseline-ml retrieval robust for both train and test patients. The result run to completion and write a valid `submission.csv` with the required columns and order.'

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

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import message_factory as _message_factory

    def _ensure_getprototype():
        MF = getattr(_message_factory, "MessageFactory", None)
        if MF is None:
            return

        if not hasattr(MF, "GetPrototype"):
            if hasattr(MF, "GetMessageClass"):

                def _GetPrototype(self, descriptor):
                    return self.GetMessageClass(descriptor)

                setattr(MF, "GetPrototype", _GetPrototype)
            else:

                def _GetPrototype(self, descriptor):
                    return None

                setattr(MF, "GetPrototype", _GetPrototype)

        try:
            inst = _message_factory.MessageFactory()
            if not hasattr(inst, "GetPrototype") and hasattr(MF, "GetPrototype"):
                inst.GetPrototype = MF.GetPrototype.__get__(inst, MF)  # type: ignore[attr-defined]
        except Exception:
            pass

    _ensure_getprototype()
except Exception as _e:
    print("WARNING: protobuf patch/pre-import failed:", repr(_e))

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
    df["Weeks"] = df["Weeks"].astype(float)

    has_w0 = df["Weeks"].eq(0.0)
    df_w0 = df.loc[has_w0].copy()

    df_non = df.loc[~has_w0].copy()
    df_non["abs_week"] = df_non["Weeks"].abs()
    df_non = df_non.sort_values(["Patient", "abs_week", "Weeks"])
    base_fallback = df_non.drop_duplicates(subset=["Patient"], keep="first").copy()

    base = pd.concat([df_w0, base_fallback], axis=0, ignore_index=True)
    base = base.sort_values(["Patient", "Weeks"]).drop_duplicates(
        subset=["Patient"], keep="first"
    )

    base = base.rename(
        columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
    )[["Patient", "Base_Week", "Base_FVC", "Base_Percent"]]

    base["Base_FVC"] = base["Base_FVC"].astype(float)
    base["Base_Percent"] = base["Base_Percent"].astype(float)

    base["Typical_FVC"] = (
        base["Base_FVC"].values / np.maximum(base["Base_Percent"].values, 1e-6)
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
test_data["Base_Week"] = test_data["Base_Week"].astype(float)
test_data["Base_FVC"] = test_data["Base_FVC"].astype(float)
test_data["Base_Percent"] = test_data["Base_Percent"].astype(float)

test_data["Typical_FVC"] = (
    test_data.Base_FVC.values / np.maximum(test_data.Base_Percent.values, 1e-6)
) * 100.0
sub = sub.merge(test_data, how="left", on="Patient")

if "Percent" not in sub.columns:
    sub["Percent"] = np.nan
sub["Percent"] = sub["Percent"].astype(float)

mask_missing_percent = sub["Percent"].isna()
sub.loc[mask_missing_percent, "Percent"] = (
    100.0
    * sub.loc[mask_missing_percent, "Base_FVC"].astype(float)
    / np.maximum(sub.loc[mask_missing_percent, "Typical_FVC"].astype(float), 1e-6)
)
sub["Percent"] = sub["Percent"].replace([np.inf, -np.inf], np.nan)




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

data["Weeks"] = data["Weeks"].astype(float)
data["Type"] = data["Type"].astype(str)

data["UseForTrain"] = True
data.loc[(data["Type"] == "train") & (data["Weeks"] < 0), "UseForTrain"] = False

train_mask = (data["Type"] == "train") & (data["UseForTrain"])
train_medians = data.loc[train_mask, Continuos_cols].median(numeric_only=True)
data[Continuos_cols] = data[Continuos_cols].fillna(train_medians)

for c in Categorical_cols:
    mode_val = data.loc[train_mask, c].mode(dropna=True)
    mode_val = mode_val.iloc[0] if len(mode_val) else "Unknown"
    data[c] = data[c].fillna(mode_val)

data["DeltaWeeks_real"] = (
    data["Weeks"].astype(float) - data["Base_Week"].astype(float)
).astype(np.float32)




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
x_cols = [
    "Weeks",
    "Base_Week",
    "Base_FVC",
    "Typical_FVC",
    "Base_Percent",
    "Percent",
    "Sex",
    "Age",
]

train_rows = (data["Type"] == "train") & (data["UseForTrain"])
test_rows = data["Type"] == "test"

x_train = data.loc[train_rows, x_cols].values.astype(np.float32)

y_scaler = MinMaxScaler()
y_train_raw = (
    data.loc[train_rows, prediction_col].values.astype(np.float32).reshape(-1, 1)
)
y_train = y_scaler.fit_transform(y_train_raw).reshape(-1).astype(np.float32)

x_test = data.loc[test_rows, x_cols].values.astype(np.float32)

patient_train = data.loc[train_rows, "Patient"].values
patient_test = data.loc[test_rows, "Patient"].values

deltaweeks_train_real = data.loc[train_rows, "DeltaWeeks_real"].values.astype(
    np.float32
)
deltaweeks_test_real = data.loc[test_rows, "DeltaWeeks_real"].values.astype(np.float32)

basefvc_by_patient_train = train_base.set_index("Patient")["Base_FVC"].to_dict()
basefvc_train_values = np.asarray(
    list(basefvc_by_patient_train.values()), dtype=np.float32
)
basefvc_global_median = (
    float(np.nanmedian(basefvc_train_values)) if basefvc_train_values.size else 0.0
)

basefvc_train_ml = np.array(
    [basefvc_by_patient_train.get(p, basefvc_global_median) for p in patient_train],
    dtype=np.float32,
)

basefvc_by_patient_test = test_data.set_index("Patient")["Base_FVC"].to_dict()
basefvc_test_ml = np.array(
    [basefvc_by_patient_test.get(p, np.nan) for p in patient_test], dtype=np.float32
)
basefvc_test_ml = np.where(
    np.isfinite(basefvc_test_ml), basefvc_test_ml, basefvc_global_median
).astype(np.float32)




## === cell 13
def build_mlp(input_dim: int):
    inp = L.Input(shape=(input_dim,), name="tabular")
    x = L.Dense(64, activation="relu")(inp)
    x = L.Dense(64, activation="relu")(x)
    out = L.Dense(1, name="fvc")(x)
    model = M.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")
    return model


def fit_slope_correction(y_true_ml, y_pred_ml, deltaweeks_real):
    r = (y_true_ml - y_pred_ml).astype(np.float64)
    dw = deltaweeks_real.astype(np.float64)
    mask = np.isfinite(r) & np.isfinite(dw)
    mask = mask & (np.abs(dw) >= 1.0)
    if mask.sum() < 10:
        return 0.0
    denom = float(np.sum(dw[mask] * dw[mask]))
    if denom <= 1e-12:
        return 0.0
    a = float(np.sum(dw[mask] * r[mask]) / denom)
    return float(np.clip(a, -200.0, 200.0))


if TF_AVAILABLE:
    n = x_train.shape[0]
    rng = np.random.RandomState(SEED)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, FOLDS)

    oof_pred_scaled = np.zeros((n, 1), dtype=np.float32)

    for f in range(FOLDS):
        va_idx = folds[f]
        tr_idx = np.setdiff1d(idx, va_idx, assume_unique=False)

        model_f = build_mlp(x_train.shape[1])
        model_f.fit(
            x_train[tr_idx], y_train[tr_idx], epochs=EPOCHS, batch_size=32, verbose=0
        )
        oof_pred_scaled[va_idx, :] = model_f.predict(
            x_train[va_idx], batch_size=256, verbose=0
        ).astype(np.float32)

    oof_pred = (
        y_scaler.inverse_transform(oof_pred_scaled).reshape(-1).astype(np.float32)
    )
    y_true_ml = y_train_raw.reshape(-1).astype(np.float32)

    slope_a = fit_slope_correction(y_true_ml, oof_pred, deltaweeks_train_real)

    oof_pred_corr = (oof_pred + slope_a * deltaweeks_train_real).astype(np.float32)
    resid = np.abs(oof_pred_corr - y_true_ml)

    sigma_global = float(np.maximum(70.0, np.median(resid) + 1e-6))
    df_res = pd.DataFrame({"Patient": patient_train, "resid": resid})
    sigma_by_patient = (
        df_res.groupby("Patient")["resid"].median().clip(lower=70.0).to_dict()
    )

    model = build_mlp(x_train.shape[1])
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32, verbose=0)
    pred_scaled = model.predict(x_test, batch_size=256, verbose=0).reshape(-1, 1)
    pred_fvc = y_scaler.inverse_transform(pred_scaled).reshape(-1).astype(np.float32)

    pred_fvc = (pred_fvc + slope_a * deltaweeks_test_real).astype(np.float32)

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
    pred_scaled = (Xte @ beta).astype(np.float32).reshape(-1, 1)
    pred_fvc = y_scaler.inverse_transform(pred_scaled).reshape(-1).astype(np.float32)

    pred_tr_scaled = (X @ beta).astype(np.float32).reshape(-1, 1)
    pred_tr = y_scaler.inverse_transform(pred_tr_scaled).reshape(-1).astype(np.float32)
    y_train_ml = y_train_raw.reshape(-1).astype(np.float32)

    slope_a = fit_slope_correction(y_train_ml, pred_tr, deltaweeks_train_real)
    pred_tr_corr = (pred_tr + slope_a * deltaweeks_train_real).astype(np.float32)
    resid = np.abs(pred_tr_corr - y_train_ml)

    sigma_global = float(np.maximum(70.0, np.median(resid) + 1e-6))
    sigma_by_patient = {}

    pred_fvc = (pred_fvc + slope_a * deltaweeks_test_real).astype(np.float32)

base_conf = np.array(
    [sigma_by_patient.get(p, sigma_global) for p in patient_test], dtype=np.float32
)
conf_growth = (
    0.25
    * np.sqrt(np.maximum(np.abs(deltaweeks_test_real), 0.0)).astype(np.float32)
    * sigma_global
    / 10.0
)
pred_conf = (base_conf + conf_growth).astype(np.float32)

sub_pred = sub.copy()
sub_pred["FVC"] = pred_fvc.astype(np.float32)
sub_pred["Confidence"] = pred_conf.astype(np.float32)

sample_pw = pd.read_csv(os.path.join(comp_dir, "sample_submission.csv"))[
    ["Patient_Week"]
]
subm = sample_pw.merge(
    sub_pred[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week", how="left"
)

if subm["FVC"].isna().any():
    subm["FVC"] = subm["FVC"].fillna(float(np.median(y_train_raw.reshape(-1))))
if subm["Confidence"].isna().any():
    subm["Confidence"] = subm["Confidence"].fillna(70.0)

subm["Confidence"] = np.maximum(subm["Confidence"].astype(float), 70.0)
subm["FVC"] = subm["FVC"].astype(float)

subm.to_csv("submission.csv", index=False)

print(subm.head())
print("Wrote submission.csv with shape:", subm.shape)
print("TF_AVAILABLE:", TF_AVAILABLE)
print("Slope correction a (ml per week):", slope_a)
print("Confidence global (sigma_global):", sigma_global)
print("Num per-patient sigmas:", len(sigma_by_patient))
print(
    "Train rows used (Weeks>=0):",
    int(train_rows.sum()),
    "/",
    int((data["Type"] == "train").sum()),
)
