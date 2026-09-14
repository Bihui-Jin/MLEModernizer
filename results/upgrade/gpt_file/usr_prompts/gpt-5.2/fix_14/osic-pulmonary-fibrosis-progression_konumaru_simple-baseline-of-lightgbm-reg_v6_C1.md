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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -19.70138) has done: 'I fix the LightGBM training call to be compatible with lightgbm==4.6.0 by passing the custom objective via `params["objective"]` (and using `feval`/callbacks) instead of the removed `objective=` argument to `lgb.train()`. This unblocks model training, which in turn resolves the downstream “no models to stack” and “pred_df not defined” errors. I also make prediction reshaping robust for LightGBM’s multiclass output and ensure `Confidence` is positive at submission time (consistent with the competition’s clipping rule) without changing the model’s core approach. Finally, the script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -19.61874) has done: 'Your current score is far below the target (higher is better), so we should improve it with minimal, metric-aligned tweaks without changing the model/feature core. The biggest likely issue is that the custom objective trains on an *unclipped, softplus sigma*, while the evaluation uses a *hard clip at 70*—so the model can “cheat” by predicting tiny sigma and get unstable gradients yet be scored with clipped sigma, harming generalization. I minimally adjust the custom objective to incorporate the same `max(sigma, 70)` behavior inside the loss/gradients (still the same approach: predicting FVC+Confidence with a custom objective), and I also enforce a minimum confidence of 70 at submission time (which is evaluation-equivalent and usually stabilizes the score). These changes keep architecture/loops intact and should move the score materially upward toward the -6.8685 target without introducing new training tricks.'
- What this solution (achieved -8.34528) has done: 'I fix the LightGBM custom objective reshape bug by correctly handling LightGBM’s raw prediction layout for multi-output training (it returns `n_samples * n_class`), and I make the reshape robust both in `grad_and_hess()` and `loss()` (feval). This unblocks training so the `models` list is populated, which in turn fixes the downstream “need at least one array to stack” and `pred_df not defined` errors. I also make `predict()` always return `(n_samples, 2)` for consistency, and keep the competition-equivalent confidence clipping at 70 in the final submission so the script always writes a valid `submission.csv`.'
- What this solution (achieved -24.65932) has done: 'Your current score (-8.34528) is below the target (-6.8685), so we should carefully improve it with minimal, metric-aligned changes. The biggest low-risk gain here is to make the custom objective/feval *exactly match* the competition’s Laplace log-likelihood: your gradients currently miss the required `sqrt(2)` factor and use `1/sigma^2` instead of `1/sigma` for the FVC (mu) term, which mis-trains the model for the true metric. I fix the loss/gradients to the correct Laplace form while keeping the same 2-output LightGBM setup and the same CV/training loop. I also ensure the evaluation function returns a value to *maximize* by setting `is_higher_better=True` (LightGBM uses this flag for early stopping), which should further move the score upward toward the target without changing the approach.'
- What this solution (achieved -8.24018) has done: 'Your score is far below the target (higher is better), so we should improve it with the smallest changes that keep the exact same LightGBM multi-output + custom Laplace objective approach. The main issue still limiting generalization is that the model is being trained on an unbounded raw sigma, while the evaluation *clips sigma at 70* and also effectively rewards well-calibrated confidence; predicting arbitrarily large sigma can overly “forgive” errors and hurt the score. I keep your objective/metric exactly the same, but add a single, metric-consistent stabilization: predict sigma in log-space via a smooth transform inside the objective (sigma = 70 + softplus(raw)), so sigma is always >= 70 and gradients behave well without changing the model/loop. I also apply the exact same transform at inference so the second output is interpreted consistently, which should move the score upward toward the target without altering the core pipeline.'
- What this solution (achieved -24.47244) has done: 'I fix the preprocessing crash by removing the duplicate `Patient` column (it currently appears in both the label and feature blocks, which breaks `DataFrame.query`) and by avoiding `query()` in favor of a boolean mask to prevent dtype resolver issues in pandas 2.2. Then I ensure the `Patient_Week` column is reliably present in the processed training frame so downstream training doesn’t KeyError. These are execution-blocking bugs; the modeling logic (LightGBM with the same custom Laplace objective and CV loop) is kept intact. After these fixes, the script train, run inference, and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -24.47244) has done: 'Your current score (-24.47) is far below the target (-6.87), so we should improve it with the smallest metric-aligned fixes that keep your LightGBM multi-output custom Laplace objective unchanged. The biggest low-risk issue is that your “passed_Weeks” sign is reversed (you’re effectively predicting *backwards in time*), which strongly degrades generalization; we flip it consistently in both train and test preprocessing. Next, we clip the *training* delta at 1000 inside the gradients (currently only the value is clipped but the gradient uses the unclipped sign), making the objective match the evaluation rule at the kink and stabilizing training. These are minimal, semantics-preserving corrections that should materially move the score upward toward the target band while keeping the same model/loop/loss.'

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
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass



## === cell 1
SEED = 42

if os.path.exists("/kaggle/input"):
    DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression/"
else:
    DATA_DIR = "../data/raw/"




## === cell 2
def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, score-relevant fix:
    - passed_Weeks must represent "weeks after baseline": pred_Weeks - current_Week.
      The previous sign (current_Week - pred_Weeks) makes the model learn the wrong direction.
    - Keep the rest of the feature/label construction identical.
    """
    dst_data = []

    data = data.sort_values(["Patient", "Weeks"]).reset_index(drop=True)

    for patient, u_data in data.groupby("Patient", sort=False):
        u_data = u_data.sort_values("Weeks").reset_index(drop=True)

        if (u_data["Weeks"] == 0).any():
            base_row = u_data.loc[u_data["Weeks"] == 0].iloc[0]
        else:
            base_row = u_data.iloc[0]

        label = pd.DataFrame(
            {
                "pred_Weeks": u_data["Weeks"].values,
                "FVC": u_data["FVC"].values,
            }
        )

        features = pd.DataFrame(
            {
                "Patient_Week": patient
                + "_"
                + label["pred_Weeks"].astype(int).astype(str),
                "current_FVC": float(base_row["FVC"]),
                "current_Percent": float(base_row["Percent"]),
                "current_Age": float(base_row["Age"]),
                "current_Week": int(base_row["Weeks"]),
                "Patient": patient,
                "Sex": {"Female": 0, "Male": 1}.get(str(base_row["Sex"]), np.nan),
                "SmokingStatus": {
                    "Currently smokes": 0,
                    "Never smoked": 1,
                    "Ex-smoker": 2,
                }.get(str(base_row["SmokingStatus"]), np.nan),
            }
        )

        dst_u_data = pd.concat(
            [label.reset_index(drop=True), features.reset_index(drop=True)], axis=1
        )

        mask = dst_u_data["pred_Weeks"].astype(int).values != int(base_row["Weeks"])
        dst_u_data = dst_u_data.loc[mask].reset_index(drop=True)

        dst_u_data["passed_Weeks"] = (
            dst_u_data["pred_Weeks"] - dst_u_data["current_Week"]
        )

        dst_data.append(dst_u_data)

    dst_data = pd.concat(dst_data, axis=0, ignore_index=True)

    if "Patient_Week" not in dst_data.columns:
        raise RuntimeError("preprocessing() must create 'Patient_Week' but did not.")
    if dst_data.columns.duplicated().any():
        dups = dst_data.columns[dst_data.columns.duplicated()].tolist()
        raise RuntimeError(f"Duplicate columns after preprocessing: {dups}")

    return dst_data


train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
train = preprocessing(train)



## === cell 3
print(train.shape)
train.head()




## === cell 4
class OSICLossForLGBM:
    """
    Custom Loss for LightGBM.

    - Predict 2 outputs: mu (FVC) and raw_sigma.
    - Interpret sigma as: sigma = sigma_clip + softplus(raw_sigma) so sigma >= sigma_clip.
    - Use the competition Laplace log-likelihood form for the metric (higher is better).
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        sigma_clip: float = 70.0,
        sigma_softplus_beta: float = 1.0,
    ) -> None:
        self.name = "osic_loss"
        self.n_class = 2
        self.epsilon = float(epsilon)
        self.sigma_clip = float(sigma_clip)
        self.beta = float(sigma_softplus_beta)

    @staticmethod
    def _reshape_preds(
        preds_1d: np.ndarray, n_example: int, n_class: int
    ) -> np.ndarray:
        preds_1d = np.asarray(preds_1d)
        if preds_1d.size != n_example * n_class:
            raise ValueError(
                f"Unexpected preds size: got {preds_1d.size}, expected {n_example*n_class} "
                f"(n_example={n_example}, n_class={n_class})"
            )
        return preds_1d.reshape(n_example, n_class)

    def _softplus(self, x: np.ndarray) -> np.ndarray:
        bx = self.beta * x
        return (1.0 / self.beta) * np.log1p(np.exp(-np.abs(bx))) + np.maximum(x, 0.0)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _sigma_from_raw(self, raw: np.ndarray) -> np.ndarray:
        return self.sigma_clip + self._softplus(raw)

    def __call__(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> float:
        mu = preds[:, 0]
        sigma = self._sigma_from_raw(preds[:, 1])

        delta = np.minimum(np.abs(mu - labels), 1000.0)
        metric_by_sample = -np.sqrt(2.0) * delta / (sigma + self.epsilon) - np.log(
            np.sqrt(2.0) * (sigma + self.epsilon)
        )
        metric = np.average(metric_by_sample, weights=weight)
        return metric

    def _calc_grad_and_hess(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        mu = preds[:, 0]
        raw_sigma = preds[:, 1]

        sigma = self._sigma_from_raw(raw_sigma)
        sigma_eps = sigma + self.epsilon

        diff = mu - labels
        abs_diff = np.abs(diff)

        delta = np.minimum(abs_diff, 1000.0)
        ddelta_dmu = np.where(abs_diff < 1000.0, np.sign(diff), 0.0)

        grad = np.zeros_like(preds)
        hess = np.zeros_like(preds)

        grad[:, 0] = (np.sqrt(2.0) / sigma_eps) * ddelta_dmu
        hess[:, 0] = 1e-6  # stable small hessian for LGBM

        dloss_dsigma = (-np.sqrt(2.0) * delta / (sigma_eps**2)) + (1.0 / sigma_eps)
        d2loss_dsigma2 = (2.0 * np.sqrt(2.0) * delta / (sigma_eps**3)) - (
            1.0 / (sigma_eps**2)
        )

        dsigma_draw = self._sigmoid(self.beta * raw_sigma)  # d softplus / d raw
        d2sigma_draw2 = self.beta * dsigma_draw * (1.0 - dsigma_draw)

        grad[:, 1] = dloss_dsigma * dsigma_draw
        hess[:, 1] = d2loss_dsigma2 * (dsigma_draw**2) + dloss_dsigma * d2sigma_draw2
        hess[:, 1] = np.maximum(hess[:, 1], 1e-6)

        if weight is not None:
            grad = grad * weight[:, None]
            hess = hess * weight[:, None]
        return grad, hess

    def loss(self, preds: np.ndarray, data: lgb.Dataset) -> tp.Tuple[str, float, bool]:
        raw_label = data.get_label()
        n_example = data.num_data()
        label = np.asarray(raw_label)
        if label.size == n_example * self.n_class:
            label = label.reshape(n_example, self.n_class)[:, 0]
        else:
            label = label.reshape(n_example)

        weight = data.get_weight()
        preds2d = self._reshape_preds(preds, n_example, self.n_class)
        metric = self(preds2d, label, weight)
        return self.name, metric, True

    def grad_and_hess(
        self, preds: np.ndarray, data: lgb.Dataset
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        raw_label = data.get_label()
        n_example = data.num_data()
        label = np.asarray(raw_label)
        if label.size == n_example * self.n_class:
            label = label.reshape(n_example, self.n_class)[:, 0]
        else:
            label = label.reshape(n_example)

        weight = data.get_weight()
        preds2d = self._reshape_preds(preds, n_example, self.n_class)
        grad2d, hess2d = self._calc_grad_and_hess(preds2d, label, weight)

        grad = grad2d.reshape(n_example * self.n_class)
        hess = hess2d.reshape(n_example * self.n_class)
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

    def dataset_to_binary(self, train_dataset, valid_dataset, params=None):
        self._remove_bin_file(self.train_bin_path)
        self._remove_bin_file(self.valid_bin_path)
        train_dataset.save_binary(self.train_bin_path)
        valid_dataset.save_binary(self.valid_bin_path)
        train_dataset = lgb.Dataset(self.train_bin_path, params=params)
        valid_dataset = lgb.Dataset(self.valid_bin_path, params=params)
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
            train_dataset, valid_dataset, params=params
        )

        train_param = dict(train_param)
        verbose_eval = train_param.pop("verbose_eval", 100)
        early_stopping_rounds = train_param.pop("early_stopping_rounds", None)

        custom_objective = train_param.pop("fobj", None)
        custom_feval = train_param.pop("feval", None)

        params = dict(params)
        if custom_objective is not None:
            params["objective"] = "none"

        callbacks = []
        if verbose_eval is not None:
            callbacks.append(lgb.log_evaluation(period=int(verbose_eval)))
        if early_stopping_rounds is not None:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=int(early_stopping_rounds), verbose=False
                )
            )

        self.model = lgb.train(
            params=params,
            train_set=train_dataset,
            valid_sets=[train_dataset, valid_dataset],
            valid_names=["train", "valid"],
            fobj=custom_objective,
            feval=custom_feval,
            callbacks=callbacks,
            **train_param,
        )
        self._remove_bin_file(self.train_bin_path)
        self._remove_bin_file(self.valid_bin_path)

    def predict(self, data):
        pred = self.model.predict(data, num_iteration=self.model.best_iteration)
        pred = np.asarray(pred)

        if pred.ndim == 1:
            if pred.size % 2 != 0:
                raise ValueError(
                    f"Unexpected prediction size {pred.size}, cannot reshape to (n,2)."
                )
            pred = pred.reshape(-1, 2)
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
custom_loss = OSICLossForLGBM(sigma_clip=70.0)

params = {
    "model_params": {
        "metric": "None",
        "boosting_type": "gbdt",
        "learning_rate": 5e-02,
        "seed": SEED,
        "subsample": 0.4,
        "subsample_freq": 1,
        "max_depth": 1,
        "verbosity": -1,
        "num_threads": -1,
        "num_class": 2,
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

y = train["FVC"].values.astype(np.float64)

groups = train["Patient"]

n_fold = 5
g_kfold = model_selection.GroupKFold(n_splits=n_fold)

models = []
oof = np.zeros((train.shape[0], 2), dtype=np.float64)

for i, (train_idx, valid_idx) in enumerate(g_kfold.split(X, train["FVC"], groups)):
    print("\n" + "#" * 20)
    print("#" * 5, f" {i+1} Fold")
    print("#" * 20 + "\n")

    print(f"Train Size: {len(train_idx)}")
    print(f"Valid Size: {len(valid_idx)}", "\n")

    X_train, y_train = X.iloc[train_idx, :], y[train_idx]
    X_valid, y_valid = X.iloc[valid_idx, :], y[valid_idx]

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
    pred_valid = lgb_model.predict(X_valid)
    oof[valid_idx] = pred_valid
    models.append(lgb_model)

len(models), oof.shape



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2608004632.py in <cell line: 0>()
     55 
     56     lgb_model = LGBM_Wrapper()
---> 57     lgb_model.fit(
     58         params["model_params"],
     59         params["train_params"],

/tmp/ipykernel_11/4128403605.py in fit(self, params, train_param, X_train, y_train, X_valid, y_valid, categorical, train_weight, valid_weight)
     74             )
     75 
---> 76         self.model = lgb.train(
     77             params=params,
     78             train_set=train_dataset,

TypeError: train() got an unexpected keyword argument 'fobj'

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


oof_sigma = custom_loss._sigma_from_raw(oof[:, 1])
print("OOF metric:", score(train["FVC"].values, oof[:, 0], oof_sigma))



## === cell 9
submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["pred_Weeks"] = (
    submission["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
)
submission.drop(["FVC", "Confidence"], axis=1, inplace=True)

test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
test.rename(
    columns={
        "Weeks": "current_Week",
        "FVC": "current_FVC",
        "Percent": "current_Percent",
        "Age": "current_Age",
    },
    inplace=True,
)

test["Sex"] = test["Sex"].map({"Female": 0, "Male": 1})
test["SmokingStatus"] = test["SmokingStatus"].map(
    {"Currently smokes": 0, "Never smoked": 1, "Ex-smoker": 2}
)

test = pd.merge(submission, test, how="left", on=["Patient"])

test["passed_Weeks"] = test["pred_Weeks"] - test["current_Week"]

test.shape, test.head()



## === cell 10
test_idx = test["Patient_Week"].to_numpy()

pred_folds_list = []
for m in models:
    p = m.predict(test[features])
    pred_folds_list.append(p)

pred_folds = np.stack(pred_folds_list, axis=0)
pred = pred_folds.mean(axis=0)

pred_conf = custom_loss._sigma_from_raw(pred[:, 1])

pred_df = pd.DataFrame(
    {
        "Patient_Week": test_idx,
        "FVC": pred[:, 0],
        "Confidence": pred_conf,
    }
)

pred_df.head(), pred_df.shape



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2820167494.py in <cell line: 0>()
      6     pred_folds_list.append(p)
      7 
----> 8 pred_folds = np.stack(pred_folds_list, axis=0)
      9 pred = pred_folds.mean(axis=0)
     10 

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in stack(arrays, axis, out, dtype, casting)
    443     arrays = [asanyarray(arr) for arr in arrays]
    444     if not arrays:
--> 445         raise ValueError('need at least one array to stack')
    446 
    447     shapes = {arr.shape for arr in arrays}

ValueError: need at least one array to stack

## === cell 11
submission_template = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

sub_df = submission_template.drop(columns=["FVC", "Confidence"])
sub_df = sub_df.merge(
    pred_df[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week", how="left"
)
sub_df.columns = submission_template.columns

sub_df["FVC"] = sub_df["FVC"].fillna(submission_template["FVC"])
sub_df["Confidence"] = sub_df["Confidence"].fillna(submission_template["Confidence"])

sub_df["Confidence"] = np.maximum(sub_df["Confidence"].values, 70.0)

sub_df.to_csv("submission.csv", index=False)

print(sub_df.shape)
sub_df.head()

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1575892905.py in <cell line: 0>()
      3 sub_df = submission_template.drop(columns=["FVC", "Confidence"])
      4 sub_df = sub_df.merge(
----> 5     pred_df[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week", how="left"
      6 )
      7 sub_df.columns = submission_template.columns

NameError: name 'pred_df' is not defined
