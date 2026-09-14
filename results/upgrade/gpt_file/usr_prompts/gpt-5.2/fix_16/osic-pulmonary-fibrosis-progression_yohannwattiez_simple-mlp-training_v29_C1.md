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

-7.468159738011274

# 6. Current score

-8.76189

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.76189) has done: 'I fix the environment/import crash by forcing TensorFlow to use the Python implementation of protobuf (this resolves the `MessageFactory` error in many Kaggle images), then correct the broken input paths so train/test/sample_submission are read from the provided dataset. I remove the missing external pickle dependency (`data-preparation-for-osic/data_prep`) by fitting the existing `data_preparation` class directly on the loaded train data, preserving the original feature pipeline and selected columns. I update the Keras Adam optimizer arguments (`lr`→`learning_rate`) to match the installed Keras version, and fix a couple of column-name typos (`Base_percent` vs `Percent`) so preprocessing runs. Finally, I ensure the script always writes a valid `submission.csv` with the exact required columns.'
- What this solution (achieved -8.76189) has done: 'I fix two execution blockers: the protobuf/TensorFlow import crash by safely falling back to `tensorflow-cpu`/standard TF import without triggering the `MessageFactory.GetPrototype` path, and the Keras 3 compile error by passing `metrics=[score]` (same metric function, just in the required container type). These changes are score-neutral but allow the full training/inference pipeline to run end-to-end and write a valid `submission.csv`. Since your current score is below target (more negative), I also make a minimal, metric-aligned calibration fix: clip predicted `Confidence` to be at least 70 everywhere (matching the evaluation’s sigma clipping), which typically improves Laplace log likelihood without changing the model architecture/training loop. All paths and core model/training logic remain unchanged.'
- What this solution (achieved -8.76189) has done: 'I first fix the TensorFlow import crash by switching protobuf to the pure-Python implementation (the current `"cpp"` setting triggers the `_message` ImportError in this environment), which unblocks all downstream cells. Next, I restore missing imports (`KFold`, `mean_absolute_error`) and make `seed_all()` resilient so it doesn’t fail if TF isn’t imported yet. Finally, I keep the modeling/training logic unchanged, but ensure the submission is always created end-to-end with the exact required columns and a `.csv` suffix, including the metric-aligned `Confidence >= 70` clipping that is already present.'
- What this solution (achieved -8.76189) has done: 'I fix two execution blockers: the TensorFlow/protobuf import crash (by safely falling back to running without TF if import fails, and using a deterministic linear-regression baseline in that case), and the `None values not supported` error during Keras `.fit()` (by ensuring `Percent` uses the correct baseline column and filling any remaining NaNs before converting to numpy). These changes keep your core preprocessing/model logic intact when TF works, but guarantee an end-to-end run and a valid `submission.csv` even in environments where TF is broken. To move the score toward your target (higher is better), I also apply a metric-aligned correction: always clip `Confidence >= 70` and avoid setting confidence to `0.1` for the known baseline row (that hurts the Laplace log-likelihood). The rest of the pipeline, including the network architecture/loss definitions and cross-validation loop, is preserved.'
- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, and by clearing any preloaded `google.protobuf` modules so the env var takes effect. Then I fix the Keras `.fit()` “None values not supported” issue by ensuring the training labels have the expected shape for your custom loss/metric (tile `FVC` to 3 columns so `y_true[:,0]` is valid) and by removing any remaining NaNs in the model input arrays right before fitting. These changes preserve your model architecture, loss, and CV training loop, but make the pipeline run end-to-end reliably. Finally, I keep your metric-aligned `Confidence >= 70` clipping and ensure a valid `submission.csv` is always written.'
- What this solution (achieved -8.76189) has done: 'I fix the two blockers preventing an end-to-end run: (1) the TensorFlow/protobuf crash by removing the brittle protobuf module purging and enforcing the pure-Python protobuf implementation early, and (2) the Keras `.fit()` “None values not supported” by ensuring the model inputs and targets are strictly numeric float arrays with no object dtype/None (and aligning `y_true` shape with how `score()` indexes it). These changes preserve your model architecture, loss, CV loop, and feature pipeline; they only make execution stable and deterministic in the Kaggle runtime. To nudge the score upward toward the target without changing the model, I keep the metric-aligned `Confidence >= 70` clipping and also guard against negative/zero predicted uncertainty (`pe[:,2]-pe[:,0]`) by clipping it to a minimum of 70 (this matches evaluation’s sigma clipping and avoids pathological confidences). The script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.76189) has done: 'I fix two execution blockers: the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by making TensorFlow import optional and defaulting to the deterministic non-TF fallback (so the notebook always runs end-to-end), and the Keras `None values not supported` issue by ensuring any TF path never receives NaNs/None (while keeping the same model/loss if TF does load). To move the score upward toward your target with minimal semantic change, I also correct a key label alignment bug: `y_train` must be built from `train_` (the merged/ordered frame used for `train_p`) so folds learn the right targets. Finally, I keep the metric-aligned `Confidence >= 70` clipping and ensure the submission is written as `submission.csv` with the required columns.'
- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/protobuf crash that prevents your current TF path from running by forcing a compatible protobuf version early (and falling back to the deterministic non-TF baseline if TF still cannot import), so the notebook always finishes and writes `submission.csv`. Then I fix the `None values not supported` training error by ensuring `y_true` has the correct shape for your `qloss`/`score` functions (they expect `(N,3)`), without changing the model architecture or training loop. Finally, I keep your metric-aligned post-processing (including `Confidence >= 70`) and ensure the submission columns and file suffix are correct. These changes are execution-critical and should move your score upward toward the target by allowing the intended TF model to train instead of failing.'
- What this solution (achieved -8.76189) has done: 'I fix the `None values not supported` crash by ensuring `train_p` and `y_train` are built from the same aligned dataframe (currently `y_train` comes from `train_` but folds index `train_p`, which can introduce misalignment/NaNs). I also harden the model inputs/targets right before fitting by coercing to contiguous `float32` arrays and explicitly replacing any remaining NaN/inf values. These are minimal, score-positive correctness fixes (they let the intended TF model actually train rather than failing), while preserving your model architecture, loss/metric, and CV loop. The rest of the pipeline (including `Confidence >= 70` clipping and submission format) remains unchanged.'
- What this solution (achieved -8.76189) has done: 'I fix the `None values not supported` crash by ensuring Keras receives plain numeric `float32` NumPy arrays (not pandas objects) and by replacing any remaining non-finite values right before `.fit()`. I also make the fold split use `.to_numpy()` once and index those arrays directly, which avoids subtle object-dtype/None propagation that can happen with mixed pandas dtypes. These changes preserve your model, loss/metric, and CV loop, but unblock training so the intended TF model runs (which should improve score toward your target vs. falling back or failing). Finally, I keep the existing submission formatting and confidence clipping behavior unchanged.'
- What this solution (achieved -8.76189) has done: 'I fix the `None values not supported` crash during `model.fit()` by ensuring the arrays passed into Keras are strictly numeric dense `float32` tensors (no object dtype), and by converting the validation/training splits to `tf.Tensor` just-in-time (this is a minimal, score-neutral stability fix). I also make the `OneHotEncoder.transform()` robust to already-numeric inputs (so it won’t accidentally create object arrays) and add a defensive final `np.nan_to_num` right before training. These changes preserve your model, loss/metric, CV loop, and feature pipeline, but unblock end-to-end training so you don’t fall back or crash, which should move the score upward toward the target. The submission formatting and `Confidence >= 70` clipping are kept intact.'
- What this solution (achieved -8.76189) has done: 'I fix the crash in training by ensuring the arrays passed into Keras are pure numeric and contain no `None`/object dtypes (the current `None values not supported` comes from an object/None sneaking into the tensors). The smallest reliable fix is to coerce the selected feature matrix and targets to contiguous `float32` NumPy arrays, apply `np.nan_to_num`, and pass NumPy arrays directly to `model.fit()` (letting Keras handle conversion) instead of manually converting to tensors. I also ensure the prediction arrays are handled consistently and that the submission is still written in the required format with `Confidence >= 70` preserved (score-positive and metric-aligned, without changing the model/loss/core loop). No changes to the model architecture, loss, or CV logic are made.'
- What this solution (achieved -8.76189) has done: 'I fix the Keras training crash (`None values not supported`) by ensuring the custom `OneHotEncoder.transform()` never creates a DataFrame with a `None` index/name, which can propagate `None` into model inputs in this environment. I also explicitly drop non-feature columns (`Patient`, `Patient_Week`) after preprocessing so `train_p`/`X_pred_p` are purely numeric and consistent, while keeping your selected feature columns and model/loss/CV loop unchanged. Finally, I add a last-mile numeric coercion and `np.nan_to_num` right before `fit()`/`predict()` to guarantee no `None`/object values reach TensorFlow, which is execution-critical and should improve score by allowing the intended TF model to train instead of failing. Submission writing remains `submission.csv` with the required columns and `Confidence >= 70` clipping preserved.'
- What this solution (achieved -8.76189) has done: 'I fix the `None values not supported` crash in `model.fit()` by ensuring the arrays passed into Keras are strictly numeric and contain no `None`/object values at the source: we drop non-feature columns after preprocessing, coerce all selected feature columns to numeric, and rebuild `X_all/X_te/y_all` from those clean arrays. I also make the custom `OneHotEncoder.transform()` always return a purely numeric DataFrame (and avoid any weird index propagation) to prevent `None` sneaking into the tensors. These changes keep your model architecture, loss/metric, and CV training loop intact, but unblock training so the intended TF model runs (which should improve score toward the target). The submission writing and metric-aligned `Confidence >= 70` clipping remain unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

TF_AVAILABLE = False
TF_IMPORT_ERROR = None

try:
    import pkgutil, subprocess

    if pkgutil.find_loader("google.protobuf") is not None:
        import google.protobuf  # noqa: F401
        from importlib.metadata import version as _pkg_version

        try:
            pb_ver = _pkg_version("protobuf")
            major = int(pb_ver.split(".")[0])
            if major >= 5:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"],
                    check=False,
                )
        except Exception:
            pass
except Exception:
    pass

try:
    import tensorflow as tf
    from tensorflow import keras as K
    from tensorflow.keras import layers as L

    TF_AVAILABLE = True
except Exception as e:
    TF_IMPORT_ERROR = e
    TF_AVAILABLE = False

print("Python:", sys.version)
print("TF available:", TF_AVAILABLE)
if TF_AVAILABLE:
    print("TF version:", tf.__version__)
else:
    print("TF import error:", repr(TF_IMPORT_ERROR))




## === cell 1
def seed_all(seed=20):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        try:
            tf.random.set_seed(seed)
        except Exception as e:
            print("Warning: could not set TF seed:", repr(e))


seed_all(20)



## === cell 2
DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
raw_test = pd.read_csv(f"{DATA_DIR}/test.csv")
X_prediction = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print(train.shape, raw_test.shape, X_prediction.shape)
train.head()



## === cell 3
ID = "Patient_Week"
PINBALL_QUANTILE = [0.2, 0.50, 0.8]
LAMBDA_LOSS = 0.75
BATCH_SIZE = 128

NFOLD = 5
EPOCHS = 800



## === cell 4
if TF_AVAILABLE:
    C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")

    def score(y_true, y_pred):
        sigma = y_pred[:, 2] - y_pred[:, 0]
        fvc_pred = y_pred[:, 1]

        sigma_clip = tf.maximum(sigma, C1)
        delta = tf.abs(y_true[:, 0] - fvc_pred)
        delta = tf.minimum(delta, C2)
        sq2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))
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




## === cell 5
if TF_AVAILABLE:

    def create_model(lambda_loss):
        model_input = K.Input(shape=(9,))
        x = L.Dense(500, activation="relu", name="dense_to_freeze1")(model_input)
        x = L.Dense(100, activation="relu", name="dense_to_freeze2")(x)
        p1 = L.Dense(3, activation="linear", name="p1")(x)
        p2 = L.Dense(3, activation="relu", name="p2")(x)
        FVC = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="FVC")([p1, p2])

        model = K.Model(inputs=model_input, outputs=[FVC])

        model.compile(
            optimizer=K.optimizers.Adam(
                learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False
            ),
            loss=mloss(lambda_loss),
            metrics=[score],
        )
        return model

    model = create_model(LAMBDA_LOSS)
    model.summary()



## === cell 6
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)
X_prediction = X_prediction[["Patient", "Weeks", "Patient_Week"]]

raw_test_ = raw_test.copy()
raw_test_ = raw_test_.rename(columns={"Weeks": "Base_week", "FVC": "Base_FVC"})

X_prediction = (
    X_prediction.merge(
        raw_test_[
            [
                "Patient",
                "Base_week",
                "Base_FVC",
                "Percent",
                "Age",
                "Sex",
                "SmokingStatus",
            ]
        ],
        how="left",
        on="Patient",
    )
    .loc[
        :,
        [
            "Patient",
            "Base_week",
            "Base_FVC",
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
            "Patient_Week",
        ],
    ]
    .reset_index(drop=True)
)

X_prediction.head()



## === cell 7
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError


class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, categories, index=None, name="", **kwargs):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        sparse_matrix = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_new_columns(X=X, name=name, categories=categories)

        if index is None:
            index = pd.RangeIndex(0, X.shape[0])

        d_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=index)
        d_out = d_out.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.int32)
        return d_out

    def fit_transform(self, X, categories, index=None, name="", **kwargs):
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
        self.onehotenc_smok = OneHotEncoder()
        self.standardisation = bool_standard
        self.normalization = bool_normalization

    def __call__(self, data_untransformed):
        data = data_untransformed.copy(deep=True)

        if "Sex" in data.columns:
            data["Sex"] = data["Sex"].fillna("Unknown")
        if "SmokingStatus" in data.columns:
            data["SmokingStatus"] = data["SmokingStatus"].fillna("Unknown")

        try:
            data["Sex"] = self.enc_sex.transform(data["Sex"].values)
            data["SmokingStatus"] = self.enc_smok.transform(
                data["SmokingStatus"].values
            )
            data = pd.concat(
                [
                    data.drop(columns=["SmokingStatus"]),
                    self.onehotenc_smok.transform(
                        data["SmokingStatus"].values.reshape(-1, 1),
                        categories=self.enc_smok.classes_,
                        name="",
                        index=data.index,
                    ),
                ],
                axis=1,
            )

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

        except NotFittedError:
            data["Sex"] = self.enc_sex.fit_transform(data["Sex"].values)
            data["SmokingStatus"] = self.enc_smok.fit_transform(
                data["SmokingStatus"].values
            )
            data = pd.concat(
                [
                    data.drop(columns=["SmokingStatus"]),
                    self.onehotenc_smok.fit_transform(
                        data["SmokingStatus"].values.reshape(-1, 1),
                        categories=self.enc_smok.classes_,
                        name="",
                        index=data.index,
                    ),
                ],
                axis=1,
            )

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

        return data




## === cell 8
train_ = train.copy()

base = train_.loc[
    train_.groupby("Patient")["Weeks"].apply(lambda s: (s.abs().idxmin()))
].copy()
base = base.rename(
    columns={"Weeks": "Base_week", "FVC": "Base_FVC", "Percent": "Base_percent"}
)
base = base[["Patient", "Base_week", "Base_FVC", "Base_percent"]]

train_ = train_.merge(base, on="Patient", how="left")
train_["Percent"] = train_["Base_percent"]

train_["Patient_Week"] = (
    train_["Patient"].astype(str) + "_" + train_["Weeks"].astype(int).astype(str)
)

data_prep = data_preparation(bool_normalization=True, bool_standard=False)

train_p = data_prep(
    train_[
        [
            "Patient",
            "Base_week",
            "Base_FVC",
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
            "Patient_Week",
        ]
    ]
)
X_pred_p = data_prep(X_prediction)

for df in (train_p, X_pred_p):
    for col in ["Patient", "Patient_Week"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

y_train = train_.loc[train_p.index, "FVC"].to_numpy(dtype="float32").reshape(-1, 1)
y_train = np.repeat(y_train, 3, axis=1).astype("float32")

print(
    "train_p shape:",
    train_p.shape,
    "X_pred_p shape:",
    X_pred_p.shape,
    "y_train shape:",
    y_train.shape,
)



## === cell 9
SELECTED_COLUMNS = [
    "Base_week",
    "Base_FVC",
    "Percent",
    "Age",
    "Sex",
    "Weeks",
    "_Currently smokes",
    "_Ex-smoker",
    "_Never smoked",
]

missing_train = [c for c in SELECTED_COLUMNS if c not in train_p.columns]
missing_test = [c for c in SELECTED_COLUMNS if c not in X_pred_p.columns]
if missing_train or missing_test:
    raise ValueError(
        f"Missing columns. train missing={missing_train}, test missing={missing_test}"
    )


def make_numeric_no_nan(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


train_p = make_numeric_no_nan(train_p, SELECTED_COLUMNS)
X_pred_p = make_numeric_no_nan(X_pred_p, SELECTED_COLUMNS)

train_p[SELECTED_COLUMNS] = train_p[SELECTED_COLUMNS].astype("float32")
X_pred_p[SELECTED_COLUMNS] = X_pred_p[SELECTED_COLUMNS].astype("float32")

y_train = np.asarray(y_train, dtype=np.float32)
y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
y_train = np.ascontiguousarray(y_train)

print("Any NaNs in train inputs:", train_p[SELECTED_COLUMNS].isna().any().any())
print("Any NaNs in test inputs:", X_pred_p[SELECTED_COLUMNS].isna().any().any())
print("Any NaNs in y_train:", np.isnan(y_train).any())



## === cell 10
pe = np.zeros((X_pred_p.shape[0], 3), dtype=np.float32)
pred = np.zeros((train_p.shape[0], 3), dtype=np.float32)

if TF_AVAILABLE:
    X_all = train_p.loc[:, SELECTED_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_all = np.ascontiguousarray(X_all, dtype=np.float32)

    y_all = np.asarray(y_train, dtype=np.float32)
    y_all = np.nan_to_num(y_all, nan=0.0, posinf=0.0, neginf=0.0)
    y_all = np.ascontiguousarray(y_all, dtype=np.float32)

    X_te = X_pred_p.loc[:, SELECTED_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.ascontiguousarray(X_te, dtype=np.float32)

    if X_all.dtype == object or y_all.dtype == object or X_te.dtype == object:
        raise ValueError("Object dtype detected after cleaning; cannot train reliably.")

    if np.isnan(X_all).any() or np.isnan(y_all).any() or np.isnan(X_te).any():
        raise ValueError("NaNs remain after cleaning; would crash TF conversion.")

    kf = KFold(n_splits=NFOLD, shuffle=True, random_state=20)

    cnt = 0
    for tr_idx, val_idx in kf.split(X_all):
        cnt += 1
        print(f"FOLD {cnt}/{NFOLD}")
        net = create_model(LAMBDA_LOSS)

        X_tr = X_all[tr_idx]
        X_va = X_all[val_idx]
        y_tr = y_all[tr_idx]
        y_va = y_all[val_idx]

        net.fit(
            X_tr,
            y_tr,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_va, y_va),
            verbose=0,
        )

        tr_eval = net.evaluate(X_tr, y_tr, verbose=0, batch_size=BATCH_SIZE)
        va_eval = net.evaluate(X_va, y_va, verbose=0, batch_size=BATCH_SIZE)
        print("train eval (loss, score):", tr_eval)
        print("val   eval (loss, score):", va_eval)

        pred[val_idx] = net.predict(X_va, batch_size=BATCH_SIZE, verbose=0)
        pe += net.predict(X_te, batch_size=BATCH_SIZE, verbose=0) / NFOLD
else:
    X = train_p[SELECTED_COLUMNS].values.astype(np.float64)
    y = y_train[:, 0].astype(np.float64)
    X_aug = np.c_[np.ones(len(X)), X]
    coef, *_ = np.linalg.lstsq(X_aug, y, rcond=None)

    Xte = X_pred_p[SELECTED_COLUMNS].values.astype(np.float64)
    Xte_aug = np.c_[np.ones(len(Xte)), Xte]
    yhat_te = Xte_aug @ coef
    yhat_tr = X_aug @ coef

    resid = y - yhat_tr
    mad = float(np.median(np.abs(resid - np.median(resid))) * 1.4826)
    spread = max(mad, 150.0)

    pe[:, 1] = yhat_te.astype(np.float32)
    pe[:, 0] = (yhat_te - spread).astype(np.float32)
    pe[:, 2] = (yhat_te + spread).astype(np.float32)

    pred[:, 1] = yhat_tr.astype(np.float32)
    pred[:, 0] = (yhat_tr - spread).astype(np.float32)
    pred[:, 2] = (yhat_tr + spread).astype(np.float32)

print("pe shape:", pe.shape, "pred shape:", pred.shape)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2083998030.py in <cell line: 0>()
     36         y_va = y_all[val_idx]
     37 
---> 38         net.fit(
     39             X_tr,
     40             y_tr,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py in convert_to_tensor(x, dtype, sparse)
    135             x = tf.convert_to_tensor(x)
    136             return tf.cast(x, dtype)
--> 137         return tf.convert_to_tensor(x, dtype=dtype)
    138     elif dtype is not None and not x.dtype == dtype:
    139         if isinstance(x, tf.SparseTensor):

ValueError: None values not supported.

## === cell 11
sigma_opt = mean_absolute_error(
    train_.loc[train_p.index, "FVC"].values.reshape(-1, 1), pred[:, 1].reshape(-1, 1)
)
unc = pred[:, 2] - pred[:, 0]
sigma_mean = float(np.mean(unc))
print("sigma_opt:", sigma_opt, "sigma_mean:", sigma_mean)



## === cell 12
X_prediction_out = X_prediction.copy()
X_prediction_out["FVC1"] = 0.996 * pe[:, 1]
X_prediction_out["Confidence1"] = pe[:, 2] - pe[:, 0]

subm = X_prediction_out.copy()
subm["FVC"] = 3020.0
subm["Confidence"] = 100.0

mask = ~subm["FVC1"].isnull()
subm.loc[mask, "FVC"] = subm.loc[mask, "FVC1"]

if sigma_mean < 70:
    subm["Confidence"] = float(max(sigma_opt, 70.0))
else:
    subm.loc[mask, "Confidence"] = subm.loc[mask, "Confidence1"]

subm["Confidence"] = pd.to_numeric(subm["Confidence"], errors="coerce").fillna(70.0)
subm["Confidence"] = subm["Confidence"].clip(lower=70.0)



## === cell 13
otest = pd.read_csv(f"{DATA_DIR}/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(int(otest.Weeks[i]))
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0

subm["Confidence"] = subm["Confidence"].clip(lower=70.0)



## === cell 14
submission_path = "submission.csv"
subm[["Patient_Week", "FVC", "Confidence"]].to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(subm.head())
print("Submission shape:", subm[["Patient_Week", "FVC", "Confidence"]].shape)
print(
    "Any nulls:", subm[["Patient_Week", "FVC", "Confidence"]].isnull().any().to_dict()
)
