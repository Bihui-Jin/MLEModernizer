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

geopandas==0.14.4
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0

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

-6.8685

# 6. Current score

## error

Could not parse JSON from mlebench output.
Please be reminded that the grader relies on 'mle-bench' being installed and navigated.
Raw output (tail):
Traceback (most recent call last):
  File "/home/b27jin/miniconda3/envs/mle_env/bin/mlebench", line 5, in <module>
    from mlebench.cli import main
  File "/home/b27jin/mle-bench/mlebench/cli.py", line 5, in <module>
    from mlebench.data import download_and_prepare_dataset, ensure_leaderboard_exists
  File "/home/b27jin/mle-bench/mlebench/data.py", line 29, in <module>
    cache = dc.Cache("cache", size_limit=2**26)  # 64 MB
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/b27jin/miniconda3/envs/mle_env/lib/python3.11/site-packages/diskcache/core.py", line 499, in __init__
    sql(query, (key, value))
  File "/home/b27jin/miniconda3/envs/mle_env/lib/python3.11/site-packages/diskcache/core.py", line 666, in _execute_with_retry
    return sql(statement, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
sqlite3.OperationalError: disk I/O error

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.80859) has done: 'I fix the LightGBM v4 API break that causes training to crash: `lgb.train()` no longer accepts `fobj`/`feval` as keywords, so I pass the custom objective via `params["objective"]` and the metric via `callbacks` (keeping your same custom loss logic). I also make prediction shaping robust so `pred[:, 0]`/`pred[:, 1]` always work, which fixes the scalar/shape error in inference and prevents `pred_df` from being undefined downstream. Finally, I keep the submission-writing logic intact but ensure it always produces a valid `submission.csv` with the required columns and 1908 rows.'
- What this solution (achieved -19.70138) has done: 'I fix the LightGBM early-stopping crash by ensuring an evaluation metric is present during training while keeping your custom objective unchanged. I also make the wrapper prediction reshape robust for both `(n*2,)` and `(n,2)` outputs so inference always produces `pred[:, 0]` and `pred[:, 1]` correctly. Finally, I guard inference/submission generation so it always writes a valid `submission.csv` with the required columns and row count, even if something unexpected happens upstream. These changes are execution-stability fixes and should not materially alter the intended modeling logic.'
- What this solution (achieved -8.58542) has done: 'Your current score (-19.70) is far below the target (-6.87), so we need a small change that legitimately improves the metric without changing the model/feature core. The biggest issue is that your custom objective optimizes a Gaussian NLL with a softplus sigma, while the Kaggle metric uses a Laplace-style likelihood with sigma clipping at 70—this mismatch can severely hurt. I keep the same 2-output LightGBM setup and training loop, but adjust only the custom gradient/hessian to match the competition Laplace metric (with sigma clipping) and also ensure the evaluation callback uses the same transformed/clipped sigma. Finally, I keep submission generation identical but align confidence post-processing with the same sigma clipping used in training/eval.'
- What this solution (achieved -8.65543) has done: 'I make two minimal, score-relevant fixes while preserving your exact model/feature core: (1) correct the sign of the objective so LightGBM *minimizes* negative log-likelihood (your current objective/feval return the Kaggle metric which is higher-is-better, so training is being driven in the wrong direction), and (2) add a tiny deterministic post-fit calibration on the confidence output using out-of-fold residuals (a single global scaling) so the predicted Confidence better matches the Laplace metric’s tradeoff. These changes keep the same 2-output LightGBM setup, same folds, same features, same training loop, and same submission format, but should move your score up toward the target. The submission writing remains identical and still produces a valid `submission.csv` with 1908 rows and required columns.'
- What this solution (achieved -8.72075) has done: 'Your current score (-8.655) is below the target (-6.8685), so we should make a small, low-risk improvement that better matches the evaluation without changing the model/features/training loop. The biggest remaining mismatch is that the objective uses a clipped sigma (flat gradient when sigma<70), which makes the model unable to learn useful confidence values; instead we train on an unclipped sigma (still using softplus), and only apply the required sigma clipping at evaluation/submission time. This preserves the same 2-output LightGBM setup and loss family, but restores gradients for the confidence head so Confidence becomes informative, which typically improves the Laplace log-likelihood. We keep your existing global confidence scaling calibration and submission formatting intact.'
- What this solution (achieved -8.46038) has done: 'We make a small, score-relevant adjustment to better match how FVC evolves over weeks without changing your model/feature set or training loop: compute `passed_Weeks` as `pred_Weeks - current_Week` (time forward from the known visit), since the current sign makes the model infer trends in the wrong direction when extrapolating to later weeks in the submission. This is a minimal semantic fix in preprocessing that typically improves both FVC and Confidence behavior under the same Laplace NLL objective. We also add `min_data_in_leaf` and `lambda_l2` (light regularization) to reduce fold-to-fold instability for this extremely shallow tree setup, which tends to improve generalization slightly without changing core logic. Submission writing and the custom loss remain unchanged.'
- What this solution (achieved -8.44896) has done: 'Your current score (-8.46038) is worse than the target (-6.8685), so we should make a small, low-risk improvement without changing the model/core setup. The biggest remaining mismatch is that the FVC head is trained with an L1/Laplace-like loss but is evaluated after averaging fold predictions; for Laplace, the optimal ensemble for the location parameter is the median, not the mean. I change only the fold-ensemble aggregation for the FVC output from mean to median (keeping confidence aggregated as mean, then applying your existing softplus + scaling + clipping). This typically improves the Laplace log-likelihood while keeping architecture/training/features identical and still produces a valid `submission.csv`.'
- What this solution (achieved -8.44896) has done: 'Your current score (-8.44896) is below the target (-6.8685), so we should make a small, low-risk improvement that keeps your exact model/features/training loop intact. The biggest remaining score lever in this metric is Confidence calibration: a single global scale is often too crude, so we keep your same sigma head and softplus, but replace the one-number scaling with a patient-specific scaling learned from OOF residual-to-sigma ratios (then applied per patient at test time). This directly targets the Laplace log-likelihood tradeoff without changing the LightGBM architecture, features, folds, or objective. We also keep sigma clipping at 70 only at evaluation/submission time, as you already do, and still write a valid `submission.csv`.'
- What this solution (achieved -15.32033) has done: 'Your current score (-8.44896) is below the target (-6.8685), so we should make a small, low-risk improvement that keeps the same LightGBM setup, features, and training loop. The most direct lever left is aligning the *training-time* clipping behavior with the metric: the competition clips sigma at 70, so we apply a smooth “soft clip” toward 70 inside the objective (instead of no clipping), which helps the confidence head learn in the right regime without a hard zero-gradient region. We keep evaluation/submission clipping exactly as required, and we keep your fold aggregation (median for FVC, mean for raw sigma) and patient-specific calibration, but we re-fit the calibration on the new OOF outputs. This is a minimal semantic adjustment to the loss (within the same loss family) that should move the score upward toward the target band.'
- What this solution (achieved -8.44896) has done: 'Your current score (-15.32033) is far worse than the target (-6.8685), so we should undo the one recent change that most plausibly caused the regression while keeping your overall LightGBM + 2-head setup, folds, features, and training loop intact. The “smooth floor toward 70 inside the objective” can distort gradients for the confidence head and tends to push sigma too high, which hurts the Laplace log-likelihood; the metric only clips at evaluation time, so training should keep sigma *unclipped* for better learning signal. I revert the objective to use an unclipped softplus sigma (as you previously had when scoring around ~-8.4), keep the required hard clip at eval/submission time, and keep your patient-specific confidence calibration and median-ensemble for FVC unchanged. This is a minimal, score-relevant change that should move the score back up toward the target band while preserving core logic.'

# 9. Code solution

## === cell 0
import os
import typing as tp

import numpy as np
import pandas as pd

import cv2
import pydicom
from PIL import Image

import sklearn
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

try:
    get_ipython  # type: ignore
    from IPython import get_ipython as _get_ipython  # type: ignore

    _ip = _get_ipython()
    if _ip is not None:
        _ip.run_line_magic("matplotlib", "inline")
except Exception:
    pass



## === cell 1
SEED = 42

if os.path.exists("/kaggle/input"):
    DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression/"
else:
    DATA_DIR = "../data/raw/"




## === cell 2
def preprocessing(data: pd.DataFrame, is_test: bool = True) -> pd.DataFrame:
    features = pd.DataFrame()
    for patient, u_data in data.groupby("Patient"):
        feature = pd.DataFrame(
            {
                "current_FVC": u_data["FVC"],
                "current_Percent": u_data["Percent"],
                "current_Age": u_data["Age"],
                "current_Week": u_data["Weeks"],
                "Patient": u_data["Patient"],
                "Sex": u_data["Sex"].map({"Female": 0, "Male": 1}),
                "SmokingStatus": u_data["SmokingStatus"].map(
                    {"Currently smokes": 0, "Never smoked": 1, "Ex-smoker": 2}
                ),
            }
        )
        features = pd.concat([features, feature], axis=0, ignore_index=True)

    if is_test:
        label = pd.read_csv(
            os.path.join(DATA_DIR, "sample_submission.csv"), usecols=["Patient_Week"]
        )
        label["Patient"] = label["Patient_Week"].apply(lambda x: x.split("_")[0])
        label["pred_Weeks"] = (
            label["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
        )
        label["FVC"] = np.nan

        dst_data = pd.merge(label, features, how="left", on="Patient")
    else:
        label = pd.DataFrame(
            {
                "Patient_Week": data["Patient"].astype(str)
                + "_"
                + data["Weeks"].astype(str),
                "Patient": data["Patient"],
                "pred_Weeks": data["Weeks"],
                "FVC": data["FVC"],
            }
        )

        dst_data = pd.merge(label, features, how="outer", on="Patient")
        dst_data = dst_data.query("pred_Weeks!=current_Week")

    dst_data["passed_Weeks"] = dst_data["pred_Weeks"] - dst_data["current_Week"]
    return dst_data


train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
train = preprocessing(train, is_test=False)



## === cell 3
print(train.shape)
train.head()




## === cell 4
class OSICLossForLGBM:
    """
    Custom Loss for LightGBM (minimized).

    Keep sigma as softplus(raw) during training for clean gradients, and apply the
    required hard clip at eval/submission time (outside the objective), as done later.
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.name = "osic_nll"
        self.n_class = 2  # FVC & Confidence(raw)
        self.epsilon = float(epsilon)

    @staticmethod
    def _softplus(x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    def __call__(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> float:
        mu = preds[:, 0]
        sigma_raw = preds[:, 1]

        sigma = self._softplus(sigma_raw) + self.epsilon  # UNCLIPPED during training
        Delta = np.minimum(np.abs(mu - labels), 1000.0)

        nll_by_sample = np.sqrt(2.0) * Delta / sigma + np.log(np.sqrt(2.0) * sigma)
        return float(np.average(nll_by_sample, weights=weight))

    def _calc_grad_and_hess(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        mu = preds[:, 0]
        sigma_raw = preds[:, 1]

        sigma = self._softplus(sigma_raw) + self.epsilon
        dsigma_draw = 1.0 / (1.0 + np.exp(-sigma_raw))  # d softplus / d raw

        r = mu - labels
        abs_r = np.abs(r)
        Delta = np.minimum(abs_r, 1000.0)

        grad = np.zeros_like(preds, dtype=float)
        hess = np.zeros_like(preds, dtype=float)

        active = abs_r < 1000.0
        sign_r = np.sign(r)
        dDelta_dmu = np.where(active, sign_r, 0.0)

        grad_mu = np.sqrt(2.0) * dDelta_dmu / sigma
        hess_mu = np.full_like(grad_mu, self.epsilon)

        grad[:, 0] = grad_mu
        hess[:, 0] = hess_mu

        dloss_dsigma = (-np.sqrt(2.0) * Delta / (sigma**2)) + (1.0 / sigma)
        d2loss_dsigma2 = (2.0 * np.sqrt(2.0) * Delta / (sigma**3)) - (1.0 / (sigma**2))

        grad_sigma_raw = dloss_dsigma * dsigma_draw
        d2sigma_draw2 = dsigma_draw * (1.0 - dsigma_draw)

        hess_sigma_raw = (
            d2loss_dsigma2 * (dsigma_draw**2) + dloss_dsigma * d2sigma_draw2
        )
        hess_sigma_raw = np.maximum(hess_sigma_raw, self.epsilon)

        grad[:, 1] = grad_sigma_raw
        hess[:, 1] = hess_sigma_raw

        if weight is not None:
            grad = grad * weight[:, None]
            hess = hess * weight[:, None]

        return grad, hess

    def loss(self, preds: np.ndarray, data: lgb.Dataset) -> tp.Tuple[str, float, bool]:
        labels = data.get_label()
        weight = data.get_weight()
        n_example = len(labels)

        preds = preds.reshape(self.n_class, n_example).T
        loss = self(preds, labels, weight)
        return self.name, float(loss), False

    def grad_and_hess(
        self, preds: np.ndarray, data: lgb.Dataset
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        labels = data.get_label()
        weight = data.get_weight()
        n_example = len(labels)

        preds = preds.reshape(self.n_class, n_example).T
        grad, hess = self._calc_grad_and_hess(preds, labels, weight)

        grad = grad.T.reshape(n_example * self.n_class)
        hess = hess.T.reshape(n_example * self.n_class)
        return grad, hess




## === cell 5
class LGBM_Wrapper:
    def __init__(self):
        self.model = None
        self.importance = None

        self.train_bin_path = "tmp_train_set.bin"
        self.valid_bin_path = "tmp_valid_set.bin"

    def _remove_bin_file(self, filename):
        if os.path.exists(filename):
            os.remove(filename)

    def dataset_to_binary(self, train_dataset, valid_dataset):
        self._remove_bin_file(self.train_bin_path)
        self._remove_bin_file(self.valid_bin_path)
        train_dataset.save_binary(self.train_bin_path)
        valid_dataset.save_binary(self.valid_bin_path)
        train_dataset = lgb.Dataset(self.train_bin_path)
        valid_dataset = lgb.Dataset(self.valid_bin_path)
        return train_dataset, valid_dataset

    def fit(
        self,
        params,
        train_param,
        X_train,
        y_train,
        X_valid,
        y_valid,
        categorical=None,
        train_weight=None,
        valid_weight=None,
    ):
        train_dataset = lgb.Dataset(
            X_train,
            y_train,
            feature_name=X_train.columns.tolist(),
            categorical_feature=categorical,
            weight=train_weight,
        )
        valid_dataset = lgb.Dataset(
            X_valid,
            y_valid,
            weight=valid_weight,
            categorical_feature=categorical,
            reference=train_dataset,
        )

        train_dataset, valid_dataset = self.dataset_to_binary(
            train_dataset, valid_dataset
        )

        num_boost_round = train_param.get("num_boost_round", 10000)
        verbose_eval = train_param.get("verbose_eval", 100)
        early_stopping_rounds = train_param.get("early_stopping_rounds", 100)

        callbacks = []
        if early_stopping_rounds is not None:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=int(early_stopping_rounds),
                    verbose=bool(verbose_eval),
                )
            )
        if verbose_eval is not None and int(verbose_eval) > 0:
            callbacks.append(lgb.log_evaluation(period=int(verbose_eval)))

        params2 = dict(params)
        params2["objective"] = train_param.get("fobj", None)

        self.model = lgb.train(
            params2,
            train_dataset,
            num_boost_round=num_boost_round,
            valid_sets=[train_dataset, valid_dataset],
            valid_names=["train", "valid"],
            feval=train_param.get("feval", None),
            callbacks=callbacks,
        )

        self._remove_bin_file(self.train_bin_path)
        self._remove_bin_file(self.valid_bin_path)

    def predict(self, data):
        pred = self.model.predict(data, num_iteration=self.model.best_iteration)
        pred = np.asarray(pred)

        if pred.ndim == 1:
            if pred.size % 2 != 0:
                raise ValueError(
                    f"Unexpected prediction size {pred.size}; cannot reshape to (-1,2)."
                )
            pred = pred.reshape(-1, 2)
        elif pred.ndim == 2:
            if pred.shape[1] != 2 and pred.shape[0] == 2:
                pred = pred.T
        else:
            raise ValueError(
                f"Unexpected prediction ndim={pred.ndim}, shape={pred.shape}"
            )

        return pred

    def model_importance(self):
        imp_df = pd.DataFrame(
            [self.model.feature_importance()],
            columns=self.model.feature_name(),
            index=["Importance"],
        ).T
        imp_df.sort_values(by="Importance", inplace=True)
        return imp_df

    def plot_importance(self, filepath, max_num_features=50, figsize=(18, 25)):
        imp_df = self.model_importance()
        plt.figure(figsize=figsize)
        imp_df[-max_num_features:].plot(
            kind="barh",
            title="Feature importance",
            figsize=figsize,
            y="Importance",
            align="center",
        )
        plt.show()




## === cell 6
custom_loss = OSICLossForLGBM()

params = {
    "model_params": {
        "num_class": 2,
        "metric": "None",
        "boosting_type": "gbdt",
        "learning_rate": 5e-02,
        "seed": SEED,
        "subsample": 0.4,
        "subsample_freq": 1,
        "max_depth": 1,
        "verbosity": -1,
        "min_data_in_leaf": 30,
        "lambda_l2": 1.0,
    },
    "train_params": {
        "num_boost_round": 10000,
        "verbose_eval": 100,
        "early_stopping_rounds": 100,
        "fobj": custom_loss.grad_and_hess,
        "feval": custom_loss.loss,
    },
}

u_idx = train["Patient_Week"]

categorical_cols = ["Sex", "SmokingStatus"]
drop_cols = ["Patient", "Patient_Week", "FVC"]
features = [c for c in train.columns.tolist() if c not in drop_cols]

X = train[features]
y = train["FVC"]
groups = train["Patient"]

n_fold = 5
g_kfold = model_selection.GroupKFold(n_splits=n_fold)

models = []
oof = np.zeros((train.shape[0], 2), dtype=float)
for i, (train_idx, valid_idx) in enumerate(g_kfold.split(X, y, groups)):
    print("\n" + "#" * 20)
    print("#" * 5, f" {i+1}-Fold")
    print("#" * 20 + "\n")

    print(f"Train Size: {len(train_idx)}")
    print(f"Valid Size: {len(valid_idx)}", "\n")

    X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx, :], y.iloc[valid_idx]

    lgb_model = LGBM_Wrapper()
    lgb_model.fit(
        params["model_params"],
        params["train_params"],
        X_train,
        y_train,
        X_valid,
        y_valid,
        categorical_cols,
    )
    oof[valid_idx] = lgb_model.predict(X_valid)
    models.append(lgb_model)



## === cell 7
if len(models) > 0:
    f_imp = np.array([m.model_importance().sort_index().values for m in models])
    f_name = models[0].model_importance().sort_index().index

    imp_df = pd.DataFrame(f_imp.reshape(-1, len(f_name)).T, index=f_name)
    imp_df["AVG_importance"] = imp_df.iloc[:, : len(models)].mean(axis=1)
    imp_df["STD_importance"] = imp_df.iloc[:, : len(models)].std(axis=1)
    imp_df.sort_values(by="AVG_importance", inplace=True)

    imp_df.plot(
        kind="barh",
        y="AVG_importance",
        xerr="STD_importance",
        capsize=4,
        figsize=(5, 6),
    )
    plt.tight_layout()
    plt.show()




## === cell 8
def score(fvc_true, fvc_pred, sigma):
    sigma_clip = np.maximum(sigma, 70)
    delta = np.minimum(np.abs(fvc_true - fvc_pred), 1000)
    metric = -(np.sqrt(2) * delta / sigma_clip) - np.log(np.sqrt(2) * sigma_clip)
    return np.mean(metric)


sigma_oof = custom_loss._softplus(oof[:, 1]).astype(float)
sigma_oof = np.maximum(sigma_oof, 70.0)

resid_oof = np.abs(train["FVC"].to_numpy().astype(float) - oof[:, 0].astype(float))
resid_oof = np.minimum(resid_oof, 1000.0)

ratio = resid_oof / (sigma_oof + 1e-6)

oof_df = pd.DataFrame(
    {
        "Patient": train["Patient"].astype(str).values,
        "current_Week": train["current_Week"].astype(float).values,
        "ratio": ratio.astype(float),
    }
)

global_conf_scale = float(
    np.clip(
        np.median(ratio[np.isfinite(ratio)]) if np.isfinite(ratio).any() else 1.0,
        0.5,
        2.0,
    )
)

oof_df = oof_df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["ratio", "current_Week"]
)
oof_df = oof_df.sort_values(["Patient", "current_Week"], ascending=[True, True])

patient_conf_scale = (
    oof_df.groupby("Patient", sort=False)
    .head(1)
    .set_index("Patient")["ratio"]
    .astype(float)
    .clip(0.5, 2.0)
)

print(
    "OOF metric (uncalibrated):", score(train["FVC"].to_numpy(), oof[:, 0], sigma_oof)
)
print("Global Confidence scale factor (fallback):", global_conf_scale)

sigma_oof_patient_scaled = np.maximum(
    sigma_oof
    * train["Patient"]
    .astype(str)
    .map(patient_conf_scale)
    .fillna(global_conf_scale)
    .to_numpy()
    .astype(float),
    70.0,
)
print(
    "OOF metric (patient-calibrated, earliest-week):",
    score(train["FVC"].to_numpy(), oof[:, 0], sigma_oof_patient_scaled),
)



## === cell 9
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
test = preprocessing(test, is_test=True)



## === cell 10
print(test.shape)
test.head()



## === cell 11
if len(models) == 0:
    med_fvc = float(train["FVC"].median())
    pred = np.zeros((test.shape[0], 2), dtype=float)
    pred[:, 0] = med_fvc
    pred[:, 1] = 0.0
else:
    pred_stack = np.stack(
        [m.predict(test[features]).astype(float) for m in models], axis=0
    )  # (n_models, n_rows, 2)
    pred = np.zeros((pred_stack.shape[1], 2), dtype=float)
    pred[:, 0] = np.median(pred_stack[:, :, 0], axis=0)
    pred[:, 1] = np.mean(pred_stack[:, :, 1], axis=0)

pred_sigma = custom_loss._softplus(pred[:, 1]).astype(float)
pred_sigma = np.maximum(pred_sigma, 70.0)

test_patient_scale = (
    test["Patient"]
    .astype(str)
    .map(patient_conf_scale)
    .fillna(global_conf_scale)
    .to_numpy()
    .astype(float)
)
pred_sigma = np.maximum(pred_sigma * test_patient_scale, 70.0)

pred_df = pd.DataFrame(
    {
        "Patient_Week": test["Patient_Week"].astype(str).values,
        "FVC": pred[:, 0].astype(float),
        "Confidence": pred_sigma,
    }
)
pred_df.head()



## === cell 12
submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

sub_df = submission.drop(columns=["FVC", "Confidence"])
sub_df = sub_df.merge(
    pred_df[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week", how="left"
)
sub_df.columns = submission.columns

sub_df["FVC"] = (
    sub_df["FVC"].astype(float).fillna(submission["FVC"].astype(float).median())
)
sub_df["Confidence"] = sub_df["Confidence"].astype(float).fillna(70.0)
sub_df["Confidence"] = np.maximum(sub_df["Confidence"].astype(float), 70.0)

sub_df.to_csv("submission.csv", index=False)

print(sub_df.shape)
sub_df.head()
