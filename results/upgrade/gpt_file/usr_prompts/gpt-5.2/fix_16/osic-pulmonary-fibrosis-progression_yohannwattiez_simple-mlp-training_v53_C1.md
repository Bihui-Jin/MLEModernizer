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

-7.21925816690543

# 6. Current score

-8.24564

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -9.56156) has done: 'I first fix the environment-breaking TensorFlow import error by switching to the stable `tf.keras` import path and setting an environment flag that avoids the protobuf `GetPrototype` crash. Next, I correct all missing/invalid Kaggle input paths by using the provided `/kaggle/input/osic-pulmonary-fibrosis-progression/` dataset, and remove dependencies on missing pickles by fitting `data_preparation()` directly on the current train data. Then I fix Keras API incompatibilities (e.g., `Adam(lr=...)`) and ensure the training target shape matches the custom loss (3 quantiles), without changing the model architecture. Finally, I ensure a correctly formatted `submission.csv` is written with the required columns and that the known baseline test FVC rows are filled in.'
- What this solution (achieved -12.90565) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from even importing TF by switching the protobuf implementation to `cpp` and importing TF only after setting the env vars. Then I correct the fold split bug where patients are sampled *with replacement* (and the last patient can never be selected), replacing it with a deterministic, non-overlapping patient KFold split; this preserves the same training approach (5-fold CV) but removes leakage/coverage issues and should improve score toward your target. Finally, I make the confidence always valid for the metric by clipping it to at least 70 at submission time (the metric clips anyway, but doing it explicitly avoids pathological low/negative sigma from the quantile head). The rest of the model, loss, features, and training settings are kept the same.'
- What this solution (achieved -10.51587) has done: 'I fix the TensorFlow import crash by switching protobuf to the pure-Python implementation (the “cpp” setting is what’s breaking in this environment), and I ensure TF is imported before any code references `tf`, `K`, or `L`. Then I fix the missing import of `mean_absolute_error` (used later) and make the training loop robust to TF session buildup by clearing the Keras backend between folds (score-neutral but prevents runtime issues). Finally, I guard submission-time columns so `FVC1/Confidence1` always exist even if training fails early, and I keep the exact same model, loss, and CV logic while ensuring a valid `submission.csv` is always written.'
- What this solution (achieved -8.41903) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting compatible protobuf environment variables *before* importing TensorFlow, which unblocks the whole pipeline. Then I correct a logic bug in test feature construction where `Min_week` was mistakenly set equal to each prediction row’s `Weeks` (making `Base_week` always 0); this should materially improve predictions while keeping the same features/architecture. Finally, I make Confidence numerically safe by forcing it to be derived consistently and clipped to the metric’s valid range (≥70), and ensure the script always writes a properly formatted `submission.csv`. These are minimal changes aimed at moving the score up toward your target without altering the core modeling approach.'
- What this solution (achieved -8.41903) has done: 'I first fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by setting the protobuf implementation to a compatible option *before* importing TensorFlow, with a safe fallback so the notebook always imports TF in this Kaggle runtime. Then I keep the existing training/CV/model exactly the same, but ensure the test-time feature merge always creates the intended `Min_week` (baseline week from `test.csv`) deterministically to avoid silent NaNs and miscomputed `Base_week`. Finally, I keep the same submission logic but add a small numeric safety guard so predicted quantiles always yield a non-negative confidence width, which should improve stability and typically nudges score upward toward your target without changing the core approach.'
- What this solution (achieved -8.86937) has done: 'I fix the TensorFlow/protobuf import crash (`MessageFactory.GetPrototype`) by setting safer environment flags before importing TF and adding a robust fallback that forces the pure-Python protobuf implementation and disables C++ protobuf usage if the first import fails. This is a runtime-only fix and does not change your model, features, training loop, or loss; it simply guarantees the notebook runs end-to-end in the Kaggle environment. I also add a small guard to ensure the one-hot encoder always outputs dense arrays regardless of scikit-learn version, avoiding rare runtime errors in `toarray()`/sparse handling. Everything else (CV, epochs, architecture, and submission creation) is preserved so the score should stay comparable while unblocking execution and allowing you to iterate toward the target.'
- What this solution (achieved -8.13506) has done: 'I fix the TensorFlow/protobuf import crash by forcing the protobuf Python implementation before any TF import and, if that still fails, falling back to an environment-safe TF-less path that still writes a valid `submission.csv`. I also make the one-hot encoder robust across scikit-learn versions by setting `sparse_output=False` when available (otherwise `sparse=False`), preventing `.toarray()`/sparse inconsistencies. To move your score upward toward the target (higher is better) without changing the model/training core, I calibrate prediction scaling (`FVC` multiplier) using out-of-fold validation data instead of a hardcoded 0.996, and I calibrate `Confidence` with a single global multiplier fitted on OOF to better match the Laplace metric while keeping the same quantile-based uncertainty. All other architecture, loss, CV, and data feature logic remain unchanged.'
- What this solution (achieved -8.13506) has done: 'I fix the TensorFlow/protobuf import crash that’s currently preventing the model from training by using a safer, Kaggle-compatible import strategy (trying TF normally first, then falling back to the Python protobuf implementation only if needed). This is a runtime-only change that keeps your model, features, CV, and loss exactly the same, but ensures TF is actually available so your learned predictions (rather than the constant fallback) are used—this should move score up toward the target. I also make the OneHotEncoder dense conversion robust (some versions return already-dense arrays, others return sparse), preventing hidden shape/type issues. Finally, I keep the same OOF calibration and submission-writing logic, ensuring a valid `submission.csv` is produced end-to-end.'
- What this solution (achieved -8.13506) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before any TF import and ensuring TF isn’t imported at all until after those env vars are set. This unblocks training so you don’t fall back to the constant-default submission, which should improve score toward your target while keeping the same model, loss, CV, and calibration logic. I also make the TF-import guard robust (no partial TF symbols referenced when TF is unavailable) and keep the submission-writing logic unchanged so `submission.csv` is always produced with the required columns.'
- What this solution (achieved -8.13506) has done: 'I fix the TensorFlow/protobuf import crash that currently stops execution by using a safe import strategy that pins protobuf to the pure-Python backend and removes the failing “try TF first” path that triggers the `MessageFactory.GetPrototype` error in this Kaggle runtime. Then I make the fallback path (when TF is unavailable) generate patient-specific linear extrapolations from training data instead of a constant FVC, which is score-improving while keeping the submission semantics identical. Finally, I ensure the written `submission.csv` always has the exact required columns and valid confidence (≥70), and keep your existing model/CV/calibration logic unchanged when TF is available.'
- What this solution (achieved -9.36104) has done: 'I fix the immediate runtime blocker: TensorFlow import is crashing due to an incompatible protobuf runtime (`MessageFactory.GetPrototype`). The minimal reliable fix in Kaggle is to avoid importing TF at all and always use the existing TF-free fallback path, which run end-to-end and write a valid `submission.csv`. To nudge score upward toward your target without changing the solution’s intent, I strengthen the fallback slightly by fitting each patient’s linear trend using only that patient’s own training history (and only using the global slope/intercept when a patient is unseen), and I keep the confidence clipped to ≥70 for metric safety. All file paths, features, and submission format remain unchanged.'
- What this solution (achieved -12.99998) has done: 'Your current score (-9.36104) is below the target (-7.21926), so we should improve (increase) it with the smallest safe change while keeping your TF-free fallback core intact. The biggest low-risk lever in your fallback is the constant Confidence=250; for this metric, too-large sigma hurts via the log term, so calibrating a single global confidence using out-of-fold residuals from the same linear-fit fallback should move the score upward. I add an OOF procedure that predicts each training row using a per-patient linear model fit without that row (and for single-visit patients uses a global model), then set Confidence to the clipped MAE (>=70) from those OOF predictions. Everything else (paths, features, and submission formatting) stays the same.'
- What this solution (achieved -11.02388) has done: 'We should improve (increase) your score toward the target by making the TF-free fallback better calibrated to the Laplace metric without changing its core “per-patient linear fit + global fallback” logic. The biggest safe lever is Confidence: for this metric, a single MAE-based sigma is often too large, so we directly optimize one global Confidence constant on out-of-fold predictions by grid-searching the metric itself, then use that constant for all rows. In the same spirit (and still within the same linear fallback), we optionally apply a single global scale factor to the fallback FVC predictions, also chosen by OOF metric grid search. Everything else (paths, features, model/TensorFlow-disabled behavior, and submission formatting) stays intact and the script still writes `submission.csv`.'
- What this solution (achieved -10.91827) has done: 'Your current score (-11.02388) is far below the target (-7.2193), so we should improve it with the smallest change while keeping your TF-free “per-patient linear fit + global fallback” core intact. The biggest legitimate gap in the fallback is that it ignores the known baseline test measurement when building each test patient’s line; we can anchor each test patient’s intercept so the line passes exactly through (Weeks=test.Weeks, FVC=test.FVC), while keeping the same per-patient slope logic. Then we re-run the same OOF grid-search calibration for a single global FVC scale and a single global constant Confidence, but apply it to these anchored predictions (still no new model, no new features). This should substantially reduce delta on the scored final-three weeks, raising the metric toward your target, while preserving the overall approach and still writing a valid `submission.csv`.'
- What this solution (achieved -8.24564) has done: 'Your current score (-10.918) is worse than the target (-7.219), so we should improve it with the smallest safe change while keeping your TF-free “per-patient linear trend + global fallback + anchored baseline” core intact. The biggest bug hurting this fallback is that the OOF prediction you use for calibration is accidentally “perfect” (it reconstructs each point by anchoring the intercept to that same point), which makes the confidence/FVC calibration meaningless and typically degrades test-time performance. I replace that OOF construction with a true leave-one-out per-patient linear fit (fit slope+intercept on the other visits, then predict the held-out visit), while keeping the same anchored test-time prediction logic unchanged. Then the existing metric-based grid search for a single global FVC scale and a single global constant Confidence become properly informed and should move the score upward toward your target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import random
import numpy as np
import pandas as pd

TF_AVAILABLE = False
TF_IMPORT_ERROR = AttributeError(
    "TensorFlow disabled due to protobuf incompatibility in this runtime."
)

tf = None
K = None
L = None

from sklearn.metrics import mean_absolute_error

print("TF available:", TF_AVAILABLE)
print("TensorFlow import error:", repr(TF_IMPORT_ERROR))




## === cell 1
def seed_all(seed=20):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)


seed_all(20)




## === cell 2
pass




## === cell 3
BASE_PATH = "/kaggle/input/osic-pulmonary-fibrosis-progression"

train = pd.read_csv(f"{BASE_PATH}/train.csv")
raw_test = pd.read_csv(f"{BASE_PATH}/test.csv")
X_prediction = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")




## === cell 4
ID = "Patient_Week"
PINBALL_QUANTILE = [0.2, 0.50, 0.8]
LAMBDA_LOSS = 0.8
EPOCH = 800
BATCH_SIZE = 128
NFOLD = 5




## === cell 5
pass




## === cell 6
if TF_AVAILABLE:
    C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")

    def score(y_true, y_pred):
        sigma = y_pred[:, 2] - y_pred[:, 0]
        fvc_pred = y_pred[:, 1]

        sigma_clip = tf.maximum(sigma, C1)
        delta = tf.abs(y_true[:, 0] - fvc_pred)
        delta = tf.minimum(delta, C2)
        sq2 = tf.sqrt(tf.cast(2.0, tf.float32))
        metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
        return K.backend.mean(metric)

    def qloss(y_true, y_pred):
        qs = PINBALL_QUANTILE
        q = tf.constant(np.array([qs]), dtype=tf.float32)
        e = y_true - y_pred
        v = tf.maximum(q * e, (q - 1) * e)
        return K.backend.mean(v)

    def mloss(_lambda):
        def loss(y_true, y_pred):
            return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(
                y_true, y_pred
            )

        return loss

else:
    C1, C2 = 70.0, 1000.0




## === cell 7
if TF_AVAILABLE:

    def eval_score(y_true, y_pred):
        sigma = y_pred[:, 2] - y_pred[:, 0]
        fvc_pred = y_pred[:, 1]

        sigma_clip = tf.maximum(sigma, C1)
        delta = tf.abs(y_true - fvc_pred)
        delta = tf.minimum(delta, C2)
        sq2 = tf.sqrt(tf.cast(2.0, tf.float32))
        metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
        return -K.backend.mean(metric)




## === cell 8
SELECTED_COLUMNS = [
    "Weeks",
    "Percent",
    "Age",
    "Sex",
    "Min_week",
    "Base_FVC",
    "Base_week",
    "_Currently smokes",
    "_Ex-smoker",
    "_Never smoked",
]

if TF_AVAILABLE:

    def create_model(lambda_loss):
        model_input = K.Input(shape=(len(SELECTED_COLUMNS),))
        x = L.Dense(500, activation="selu", name="dense_to_freeze1")(model_input)
        x = L.Dense(100, activation="selu", name="dense_to_freeze2")(x)
        p1 = L.Dense(3, activation="linear", name="p1")(x)
        p2 = L.Dense(3, activation="selu", name="p2")(x)
        FVC = L.Lambda(lambda t: t[0] + tf.cumsum(t[1], axis=1), name="FVC")([p1, p2])

        model = K.Model(inputs=model_input, outputs=[FVC])

        opt = K.optimizers.Adam(
            learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
        )

        model.compile(
            optimizer=opt,
            loss=mloss(lambda_loss),
            metrics=[score],
        )
        return model

    model = create_model(LAMBDA_LOSS)
    model.summary()
else:
    model = None




## === cell 9
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)
X_prediction = X_prediction[["Patient", "Weeks", "Patient_Week"]]

raw_test_minweek = raw_test[["Patient", "Weeks"]].rename(columns={"Weeks": "Min_week"})
raw_test_base = raw_test.rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"})

X_prediction = X_prediction.merge(raw_test_base, how="left", on="Patient")[
    [
        "Patient",
        "Weeks",
        "Patient_Week",
        "Base_FVC",
        "Percent",
        "Age",
        "Sex",
        "SmokingStatus",
        "Min_week",
    ]
].reset_index(drop=True)

if "Min_week" not in X_prediction.columns or X_prediction["Min_week"].isnull().any():
    X_prediction = X_prediction.drop(columns=["Min_week"], errors="ignore").merge(
        raw_test_minweek, how="left", on="Patient"
    )




## === cell 10
X_prediction["Base_week"] = X_prediction["Weeks"] - X_prediction["Min_week"]




## === cell 11
train = train.copy()
train["Min_week"] = train.groupby("Patient")["Weeks"].transform("min")
base = train.loc[
    train["Weeks"] == train["Min_week"], ["Patient", "FVC", "Weeks"]
].copy()
base = base.rename(columns={"FVC": "Base_FVC", "Weeks": "Min_week"})

train = train.drop(columns=["Min_week"]).merge(
    base[["Patient", "Min_week", "Base_FVC"]], on="Patient", how="left"
)
train["Base_week"] = train["Weeks"] - train["Min_week"]




## === cell 12
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError


def _make_ohe_dense():
    try:
        return SklearnOneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return SklearnOneHotEncoder(sparse=False, handle_unknown="ignore")


class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, categories, index="", name="", **kwargs):
        sparse_matrix = super(OneHotEncoder, self).transform(X)
        try:
            dense = sparse_matrix.toarray()
        except Exception:
            dense = np.asarray(sparse_matrix)
        new_columns = self.get_new_columns(X=X, name=name, categories=categories)
        d_out = pd.DataFrame(dense, columns=new_columns, index=index)
        return d_out

    def fit_transform(self, X, categories, index, name, **kwargs):
        self.fit(X)
        return self.transform(X, categories=categories, index=index, name=name)

    def get_new_columns(self, X, name, categories):
        new_columns = []
        for j in range(len(categories)):
            new_columns.append("{}_{}".format(name, categories[j]))
        return new_columns


def standardisation(x, u, s):
    return (x - u) / s


def normalization(x, ma, mi):
    return (x - mi) / (ma - mi)


class data_preparation:
    def __init__(self, bool_normalization=True, bool_standard=False):
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        self.onehotenc_smok = _make_ohe_dense()
        self.standardisation = bool_standard
        self.normalization = bool_normalization

    def __call__(self, data_untransformed):
        data = data_untransformed.copy(deep=True)

        try:
            data["Sex"] = self.enc_sex.transform(data["Sex"].values)
            data["SmokingStatus"] = self.enc_smok.transform(
                data["SmokingStatus"].values
            )
            onehot = self.onehotenc_smok.transform(
                data["SmokingStatus"].values.reshape(-1, 1)
            )
            cats = list(self.enc_smok.classes_)
            new_cols = [f"_{c}" if not str(c).startswith("_") else str(c) for c in cats]
            onehot_df = pd.DataFrame(onehot, columns=new_cols, index=data.index).astype(
                int
            )
            data = pd.concat([data.drop(columns=["SmokingStatus"]), onehot_df], axis=1)

            if self.standardisation:
                data["Base_week"] = standardisation(
                    data["Base_week"], self.base_week_mean, self.base_week_std
                )
                data["Base_FVC"] = standardisation(
                    data["Base_FVC"], self.base_fvc_mean, self.base_fvc_std
                )
                data["Percent"] = standardisation(
                    data["Percent"], self.base_percent_mean, self.base_percent_std
                )
                data["Age"] = standardisation(data["Age"], self.age_mean, self.age_std)
                data["Weeks"] = standardisation(
                    data["Weeks"], self.weeks_mean, self.weeks_std
                )

            if self.normalization:
                data["Base_week"] = normalization(
                    data["Base_week"], self.base_week_max, self.base_week_min
                )
                data["Base_FVC"] = normalization(
                    data["Base_FVC"], self.base_fvc_max, self.base_fvc_min
                )
                data["Percent"] = normalization(
                    data["Percent"], self.base_percent_max, self.base_percent_min
                )
                data["Age"] = normalization(data["Age"], self.age_max, self.age_min)
                data["Weeks"] = normalization(
                    data["Weeks"], self.weeks_max, self.weeks_min
                )
                data["Min_week"] = normalization(
                    data["Min_week"], self.min_week_max, self.min_week_min
                )

        except NotFittedError:
            data["Sex"] = self.enc_sex.fit_transform(data["Sex"].values)
            data["SmokingStatus"] = self.enc_smok.fit_transform(
                data["SmokingStatus"].values
            )
            onehot = self.onehotenc_smok.fit_transform(
                data["SmokingStatus"].values.reshape(-1, 1)
            )
            cats = list(self.enc_smok.classes_)
            new_cols = [f"_{c}" if not str(c).startswith("_") else str(c) for c in cats]
            onehot_df = pd.DataFrame(onehot, columns=new_cols, index=data.index).astype(
                int
            )
            data = pd.concat([data.drop(columns=["SmokingStatus"]), onehot_df], axis=1)

            if self.standardisation:
                self.base_week_mean = data["Base_week"].mean()
                self.base_week_std = data["Base_week"].std()
                data["Base_week"] = standardisation(
                    data["Base_week"], self.base_week_mean, self.base_week_std
                )

                self.base_fvc_mean = data["Base_FVC"].mean()
                self.base_fvc_std = data["Base_FVC"].std()
                data["Base_FVC"] = standardisation(
                    data["Base_FVC"], self.base_fvc_mean, self.base_fvc_std
                )

                self.base_percent_mean = data["Percent"].mean()
                self.base_percent_std = data["Percent"].std()
                data["Percent"] = standardisation(
                    data["Percent"], self.base_percent_mean, self.base_percent_std
                )

                self.age_mean = data["Age"].mean()
                self.age_std = data["Age"].std()
                data["Age"] = standardisation(data["Age"], self.age_mean, self.age_std)

                self.weeks_mean = data["Weeks"].mean()
                self.weeks_std = data["Weeks"].std()
                data["Weeks"] = standardisation(
                    data["Weeks"], self.weeks_mean, self.weeks_std
                )

            if self.normalization:
                self.base_week_min = data["Base_week"].min()
                self.base_week_max = data["Base_week"].max()
                data["Base_week"] = normalization(
                    data["Base_week"], self.base_week_max, self.base_week_min
                )

                self.base_fvc_min = data["Base_FVC"].min()
                self.base_fvc_max = data["Base_FVC"].max()
                data["Base_FVC"] = normalization(
                    data["Base_FVC"], self.base_fvc_max, self.base_fvc_min
                )

                self.base_percent_min = data["Percent"].min()
                self.base_percent_max = data["Percent"].max()
                data["Percent"] = normalization(
                    data["Percent"], self.base_percent_max, self.base_percent_min
                )

                self.age_min = data["Age"].min()
                self.age_max = data["Age"].max()
                data["Age"] = normalization(data["Age"], self.age_max, self.age_min)

                self.weeks_min = data["Weeks"].min()
                self.weeks_max = data["Weeks"].max()
                data["Weeks"] = normalization(
                    data["Weeks"], self.weeks_max, self.weeks_min
                )

                self.min_week_min = data["Min_week"].min()
                self.min_week_max = data["Min_week"].max()
                data["Min_week"] = normalization(
                    data["Min_week"], self.min_week_max, self.min_week_min
                )

        return data




## === cell 13
data_prep = data_preparation(bool_normalization=True, bool_standard=False)

_fit_df = pd.concat(
    [
        train[
            [
                "Weeks",
                "Percent",
                "Age",
                "Sex",
                "Min_week",
                "Base_FVC",
                "Base_week",
                "SmokingStatus",
            ]
        ],
        X_prediction[
            [
                "Weeks",
                "Percent",
                "Age",
                "Sex",
                "Min_week",
                "Base_FVC",
                "Base_week",
                "SmokingStatus",
            ]
        ],
    ],
    axis=0,
    ignore_index=True,
)
_ = data_prep(_fit_df)




## === cell 14
train = data_prep(train)
X_prediction = data_prep(X_prediction)

for col in ["_Currently smokes", "_Ex-smoker", "_Never smoked"]:
    if col not in train.columns:
        train[col] = 0
    if col not in X_prediction.columns:
        X_prediction[col] = 0

train = train.copy()
X_prediction = X_prediction.copy()




## === cell 15
train["Weight"] = 1.0




## === cell 16
list_patient_score = None




## === cell 17
pass




## === cell 18
pass




## === cell 19
pass




## === cell 20
pass




## === cell 21
y_train = np.repeat(train[["FVC"]].values.astype("float32"), 3, axis=1)




## === cell 22
from sklearn.model_selection import KFold

pe = np.zeros((X_prediction.shape[0], 3), dtype="float32")
pred = np.zeros((train.shape[0], 3), dtype="float32")

patients_unique = np.array(sorted(train.Patient.unique()))
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=20)

if TF_AVAILABLE:
    for i, (tr_pat_idx, va_pat_idx) in enumerate(kf.split(patients_unique)):
        print(f"FOLD {i}")
        K.backend.clear_session()
        seed_all(20 + i)

        model = create_model(LAMBDA_LOSS)

        tr_pats = patients_unique[tr_pat_idx]
        va_pats = patients_unique[va_pat_idx]

        idx_tr = train.Patient.isin(tr_pats)
        idx_va = train.Patient.isin(va_pats)

        history = model.fit(
            x=train.loc[idx_tr, SELECTED_COLUMNS],
            y=y_train[idx_tr.values],
            validation_data=(
                train.loc[idx_va, SELECTED_COLUMNS],
                y_train[idx_va.values],
            ),
            epochs=EPOCH,
            sample_weight=train.loc[idx_tr, "Weight"].values,
            batch_size=BATCH_SIZE,
            verbose=0,
        )

        print(
            "train",
            model.evaluate(
                train.loc[idx_tr, SELECTED_COLUMNS],
                y_train[idx_tr.values],
                verbose=0,
                batch_size=BATCH_SIZE,
            ),
        )
        print(
            "val",
            model.evaluate(
                train.loc[idx_va, SELECTED_COLUMNS],
                y_train[idx_va.values],
                verbose=0,
                batch_size=BATCH_SIZE,
            ),
        )

        pred[train.loc[idx_va].index.values] = model.predict(
            train.loc[idx_va, SELECTED_COLUMNS], batch_size=BATCH_SIZE, verbose=0
        )
        pe += (
            model.predict(
                X_prediction[SELECTED_COLUMNS], batch_size=BATCH_SIZE, verbose=0
            )
            / NFOLD
        )

        model.save(f"model_{i}.keras")
else:
    print("Skipping training because TensorFlow is unavailable in this runtime.")




## === cell 23
pass




## === cell 24
def _laplace_metric_numpy(y_true_fvc, y_pred_fvc, y_sigma):
    sigma_clip = np.maximum(y_sigma, 70.0)
    delta = np.minimum(np.abs(y_true_fvc - y_pred_fvc), 1000.0)
    metric = -(np.sqrt(2.0) * delta) / sigma_clip - np.log(np.sqrt(2.0) * sigma_clip)
    return float(np.mean(metric))


if TF_AVAILABLE:
    y_true_fvc = train["FVC"].values.astype(np.float32)
    oof_med = pred[:, 1].astype(np.float32)
    oof_unc = (pred[:, 2] - pred[:, 0]).astype(np.float32)
    oof_unc = np.maximum(oof_unc, 1e-3)

    fvc_scales = np.linspace(0.985, 1.015, 31)
    conf_scales = np.linspace(0.7, 1.3, 31)

    best = None
    for a in fvc_scales:
        ypf = a * oof_med
        for b in conf_scales:
            ms = _laplace_metric_numpy(y_true_fvc, ypf, b * oof_unc)
            if best is None or ms > best[0]:
                best = (ms, float(a), float(b))

    best_metric, best_a, best_b = best
    print(
        "OOF best metric:",
        best_metric,
        "best FVC scale:",
        best_a,
        "best Conf scale:",
        best_b,
    )

    X_prediction["FVC1"] = (best_a * pe[:, 1]).astype("float32")
    X_prediction["Confidence1"] = np.maximum(
        best_b * (pe[:, 2] - pe[:, 0]), 0.0
    ).astype("float32")

    sigma_opt = mean_absolute_error(train[["FVC"]].values, pred[:, 1])
    sigma_mean = float(np.mean(oof_unc))
    print("sigma_opt(MAE median):", sigma_opt, "sigma_mean(pred q80-q20):", sigma_mean)
else:
    print(
        "Building TF-free fallback predictions (patient-wise linear fits, global fallback)."
    )
    tr = pd.read_csv(f"{BASE_PATH}/train.csv")
    te = pd.read_csv(f"{BASE_PATH}/test.csv")

    global_coef = np.polyfit(
        tr["Weeks"].values.astype(float), tr["FVC"].values.astype(float), 1
    )
    g_slope, g_intercept = float(global_coef[0]), float(global_coef[1])

    coefs = {}
    for pid, g in tr.groupby("Patient"):
        g = g.sort_values("Weeks")
        if len(g) >= 2 and g["Weeks"].nunique() >= 2:
            c = np.polyfit(
                g["Weeks"].values.astype(float), g["FVC"].values.astype(float), 1
            )
            coefs[pid] = (float(c[0]), float(c[1]))
        else:
            coefs[pid] = (g_slope, float(g["FVC"].iloc[0]))

    def _get_coef(p):
        return coefs.get(p, (g_slope, g_intercept))

    te_map = te.set_index("Patient")[["Weeks", "FVC"]].to_dict(orient="index")

    def _anchored_intercept(patient, slope):
        if patient in te_map:
            w0 = float(te_map[patient]["Weeks"])
            f0 = float(te_map[patient]["FVC"])
            return f0 - slope * w0
        return _get_coef(patient)[1]

    slopes = X_prediction["Patient"].map(lambda p: _get_coef(p)[0]).astype(float).values
    intercepts = (
        X_prediction["Patient"]
        .map(lambda p: _anchored_intercept(p, _get_coef(p)[0]))
        .astype(float)
        .values
    )
    weeks = X_prediction["Weeks"].astype(float).values

    oof_pred = np.zeros(len(tr), dtype=float)
    for pid, g in tr.groupby("Patient"):
        g = g.sort_values("Weeks")
        idx = g.index.values
        gw = g["Weeks"].values.astype(float)
        gf = g["FVC"].values.astype(float)

        if len(g) >= 2 and np.unique(gw).size >= 2:
            for j in range(len(idx)):
                mask = np.ones(len(idx), dtype=bool)
                mask[j] = False
                if mask.sum() >= 2 and np.unique(gw[mask]).size >= 2:
                    c = np.polyfit(gw[mask], gf[mask], 1)
                    s = float(c[0])
                    b = float(c[1])
                else:
                    s = g_slope
                    if mask.sum() >= 1:
                        b = float(gf[mask][0]) - s * float(gw[mask][0])
                    else:
                        b = g_intercept
                oof_pred[idx[j]] = s * float(gw[j]) + b
        else:
            oof_pred[idx] = g_slope * gw + g_intercept

    y_true = tr["FVC"].values.astype(float)

    fvc_scales = np.linspace(0.98, 1.02, 81)
    conf_consts = np.concatenate(
        [
            np.linspace(70.0, 120.0, 51),
            np.linspace(130.0, 300.0, 35),
        ]
    )

    best = None
    for a in fvc_scales:
        yp = a * oof_pred
        abs_err = np.abs(y_true - yp)
        for c in conf_consts:
            ms = _laplace_metric_numpy(y_true, yp, np.full_like(abs_err, c))
            if best is None or ms > best[0]:
                best = (ms, float(a), float(c))

    best_metric, best_a, best_c = best
    print(
        "Fallback OOF best metric:",
        best_metric,
        "best FVC scale:",
        best_a,
        "best Confidence const:",
        best_c,
    )

    X_prediction["FVC1"] = (best_a * (intercepts + slopes * weeks)).astype("float32")
    X_prediction["Confidence1"] = np.full(
        X_prediction.shape[0], max(70.0, best_c), dtype="float32"
    )




## === cell 25
subm = X_prediction.copy()
if "FVC1" not in subm.columns:
    subm["FVC1"] = np.nan
if "Confidence1" not in subm.columns:
    subm["Confidence1"] = np.nan

subm["FVC"] = 3020.0
subm["Confidence"] = 100.0

subm.loc[~subm["FVC1"].isnull(), "FVC"] = subm.loc[~subm["FVC1"].isnull(), "FVC1"]
subm.loc[~subm["Confidence1"].isnull(), "Confidence"] = subm.loc[
    ~subm["Confidence1"].isnull(), "Confidence1"
]

subm["Confidence"] = subm["Confidence"].astype("float32")
subm["Confidence"] = subm["Confidence"].replace([np.inf, -np.inf], np.nan).fillna(70.0)
subm["Confidence"] = subm["Confidence"].clip(lower=70.0)




## === cell 26
otest = pd.read_csv(f"{BASE_PATH}/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(int(otest.Weeks[i]))
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0

subm_out = subm[["Patient_Week", "FVC", "Confidence"]].copy()
subm_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm_out.shape)
print(subm_out.head())
