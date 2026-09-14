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

category_encoders==2.7.0
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

-6.8591

# 6. Current score

-10.17955

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.12241) has done: 'I fix the LightGBM training call to work with LightGBM 4.6.0, where `lgb.train()` no longer accepts `fobj`; instead we must pass the custom objective via `params["objective"]` and the custom metric via `feval`. I keep the model, folds, features, and loss logic the same, only adjusting the API usage and a couple of safety defaults (valid `best_iteration`, and ensuring sigma is positive at inference) so the pipeline runs end-to-end. Once training runs, the downstream `predictions/oof` NameErrors disappear and the submission creation code successfully write `submission.csv` with the required columns. These changes are score-neutral in intent (same objective/metric), but allow producing a valid submission and a reasonable score toward the target.'
- What this solution (achieved -24.65932) has done: 'Your current score is far below the target (gap = -24.12241 - (-6.8591) = -17.26), so we should improve it, but with minimal changes that preserve your core LightGBM + custom objective pipeline. The biggest issue is a semantic mismatch: you train a Gaussian NLL (mu + sigma) but evaluate with the competition’s Laplace log-likelihood; aligning the training loss to the Laplace form (while keeping the same 2-output setup and LightGBM loop) is a small, direct change that typically yields a large score jump. I also fix the fold metric computation to use the same sigma transform as used in the objective (so your “Comp Metric” logging matches what you are actually training), and keep the inference post-processing identical aside from ensuring sigma uses the correct transform consistently. These changes are tightly scoped to the objective/metric alignment and should move the score substantially toward the target band without altering features, folds, or training procedure.'
- What this solution (achieved -11.89564) has done: 'Your score is far below the target (gap ≈ -17.8), so we should improve it with minimal, metric-aligned tweaks while keeping your LightGBM 2-output (mu, sigma_raw) setup and the same folds/features/training loop. The largest remaining mismatch is that your training loss omits the competition’s required clipping (Δ capped at 1000 and σ floored at 70), so the model can “game” sigma/large errors differently than what the leaderboard scores. I incorporate those exact clip operations into the custom objective (grad/hess) consistently, and make the fold logging metric use the same clipping pipeline (so CV reflects LB behavior). I also ensure inference uses the same sigma transform and clipping (Confidence>=70) without changing any data processing or model hyperparameters.'
- What this solution (achieved -10.17955) has done: 'Your current score (-11.89564) is still well below the target (-6.8591), so we should improve it with small, metric-aligned changes while keeping your exact LightGBM 2-output setup, features, folds, and training loop intact. The biggest remaining issue is a scale mismatch: you train/predict on “virtual_FVC” units but your objective currently uses the competition’s clipping constants (70, 1000) in *ml* units, which makes the gradients inconsistent with your training label scale. I adjust the custom loss to operate in the same “virtual” scale during training (scale sigma_floor and delta_cap by `VIRTUAL_BASE_FVC/base_FVC` per row), and keep the evaluation metric unchanged in ml at the end. I also pass per-row weights into LightGBM so the objective and metric can access each sample’s `base_FVC` without changing features or data flow.'

# 9. Code solution

## === cell 0
import os
import operator
import typing as tp
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
from functools import partial

import numpy as np
import pandas as pd
import random
import math

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import mean_squared_error
import category_encoders as ce

from PIL import Image
import cv2
import pydicom

import torch

import lightgbm as lgb
from sklearn.linear_model import Ridge

import warnings

warnings.filterwarnings("ignore")




## === cell 1
def get_logger(filename="log"):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    if len(logger.handlers) == 0:
        handler1 = StreamHandler()
        handler1.setFormatter(Formatter("%(message)s"))
        handler2 = FileHandler(filename=f"{filename}.log")
        handler2.setFormatter(Formatter("%(message)s"))
        logger.addHandler(handler1)
        logger.addHandler(handler2)
    return logger


logger = get_logger()


def seed_everything(seed=777):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




## === cell 2
OUTPUT_DICT = "./"

ID = "Patient_Week"
TARGET = "virtual_FVC"
SEED = 42
VIRTUAL_BASE_FVC = 2000
seed_everything(seed=SEED)

N_FOLD = 4



## === cell 3
train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
train[ID] = train["Patient"].astype(str) + "_" + train["Weeks"].astype(str)
print(train.shape)
train.head()



## === cell 4
output = pd.DataFrame()
gb = train.groupby("Patient")
tk0 = tqdm(gb, total=len(gb))
for _, usr_df in tk0:
    usr_output = pd.DataFrame()
    for week, tmp in usr_df.groupby("Weeks"):
        rename_cols = {
            "Weeks": "base_Week",
            "FVC": "base_FVC",
            "Percent": "base_Percent",
            "Age": "base_Age",
        }
        tmp = tmp.drop(columns="Patient_Week").rename(columns=rename_cols)
        drop_cols = ["Age", "Sex", "SmokingStatus", "Percent"]
        _usr_output = (
            usr_df.drop(columns=drop_cols)
            .rename(columns={"Weeks": "predict_Week"})
            .merge(tmp, on="Patient")
        )
        _usr_output["Week_passed"] = (
            _usr_output["predict_Week"] - _usr_output["base_Week"]
        )
        usr_output = pd.concat([usr_output, _usr_output], axis=0)
    output = pd.concat([output, usr_output], axis=0)

train = output[output["Week_passed"] != 0].reset_index(drop=True)
print(train.shape)
train.head()



## === cell 5
train["virtual_FVC"] = train["FVC"] / train["base_FVC"] * VIRTUAL_BASE_FVC



## === cell 6
test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv").rename(
    columns={
        "Weeks": "base_Week",
        "FVC": "base_FVC",
        "Percent": "base_Percent",
        "Age": "base_Age",
    }
)
submission = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["predict_Week"] = (
    submission["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
)
test = submission.drop(columns=["FVC", "Confidence"]).merge(test, on="Patient")
test["Week_passed"] = test["predict_Week"] - test["base_Week"]
print(test.shape)
test.head()



## === cell 7
test["FVC"] = np.nan
test["virtual_FVC"] = np.nan



## === cell 8
train.head()



## === cell 9
test.head()



## === cell 10
submission = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
print(submission.shape)
submission.head()



## === cell 11
folds = train[[ID, "Patient", TARGET]].copy()
Fold = GroupKFold(n_splits=N_FOLD)
groups = folds["Patient"].values
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[TARGET], groups)):
    folds.loc[val_index, "fold"] = int(n)
folds["fold"] = folds["fold"].astype(int)
folds.head()




## === cell 12
class OSICLossForLGBM:
    """
    Custom Loss for LightGBM.

    Change (score-improving, minimal): make the objective operate in the SAME UNITS as the training
    label (virtual_FVC). The Kaggle metric's clipping constants (sigma>=70 ml, delta<=1000 ml) are
    in ml units; because we train on virtual-scaled FVC, we must scale these constants per-sample:
        scale_i = VIRTUAL_BASE_FVC / base_FVC_i
        sigma_floor_virtual_i = 70 * scale_i
        delta_cap_virtual_i   = 1000 * scale_i
    We pass `base_FVC` as LightGBM sample weights so the custom objective/metric can access it,
    without changing features, folds, architecture, or training loop.
    """

    def __init__(
        self,
        epsilon: float = 1e-09,
        sigma_floor_ml: float = 70.0,
        delta_cap_ml: float = 1000.0,
        virtual_base_fvc: float = 2000.0,
    ) -> None:
        self.name = "osic_loss"
        self.n_class = 2  # mu & sigma_raw
        self.epsilon = float(epsilon)
        self.sigma_floor_ml = float(sigma_floor_ml)
        self.delta_cap_ml = float(delta_cap_ml)
        self.virtual_base_fvc = float(virtual_base_fvc)

    def _softplus(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0) + self.epsilon

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def _get_scale_from_weight(
        self, weight: tp.Optional[np.ndarray], n: int
    ) -> np.ndarray:
        if weight is None:
            return np.full(n, 1.0, dtype=np.float64)
        w = np.asarray(weight, dtype=np.float64)
        w = np.clip(w, 1.0, None)  # avoid div-by-zero
        return self.virtual_base_fvc / w

    def __call__(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> float:
        mu = preds[:, 0]
        sigma_raw = preds[:, 1]
        sigma = self._softplus(sigma_raw)

        scale = self._get_scale_from_weight(weight, len(labels))
        sigma_floor_v = self.sigma_floor_ml * scale
        delta_cap_v = self.delta_cap_ml * scale

        sigma_clip = np.maximum(sigma, sigma_floor_v)
        delta = np.minimum(np.abs(labels - mu), delta_cap_v)
        loss_by_sample = (np.sqrt(2.0) * delta) / sigma_clip + np.log(sigma_clip)

        return float(np.mean(loss_by_sample))

    def _calc_grad_and_hess(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        mu = preds[:, 0]
        sigma_raw = preds[:, 1]

        sigma = self._softplus(sigma_raw)
        ds = self._sigmoid(sigma_raw)  # d softplus / d raw

        scale = self._get_scale_from_weight(weight, len(labels))
        sigma_floor_v = self.sigma_floor_ml * scale
        delta_cap_v = self.delta_cap_ml * scale

        diff = labels - mu
        abs_diff = np.abs(diff)

        cap_mask = abs_diff < delta_cap_v
        sign_mu = np.sign(mu - labels)  # = -sign(diff)

        sigma_clip = np.maximum(sigma, sigma_floor_v)
        sigma_clip_safe = np.maximum(sigma_clip, self.epsilon)

        grad_mu = (np.sqrt(2.0) * sign_mu) / sigma_clip_safe
        grad_mu = grad_mu * cap_mask.astype(grad_mu.dtype)
        hess_mu = np.full_like(grad_mu, 1e-6)

        sigma_active = (sigma >= sigma_floor_v).astype(sigma.dtype)
        delta = np.minimum(abs_diff, delta_cap_v)

        dL_dsigma_clip = (
            -(np.sqrt(2.0) * delta) / (sigma_clip_safe**2) + 1.0 / sigma_clip_safe
        )
        dL_dsigma = dL_dsigma_clip * sigma_active

        grad_sigraw = dL_dsigma * ds
        hess_sigraw = np.full_like(grad_sigraw, 1e-6)

        grad = np.zeros_like(preds)
        hess = np.zeros_like(preds)
        grad[:, 0] = grad_mu
        hess[:, 0] = hess_mu
        grad[:, 1] = grad_sigraw
        hess[:, 1] = hess_sigraw
        return grad, hess

    def calc_comp_metric(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> float:
        sigma_clip = np.maximum(preds[:, 1], self.sigma_floor_ml)
        delta = np.minimum(np.abs(preds[:, 0] - labels), self.delta_cap_ml)
        metric_by_sample = -np.sqrt(2) * delta / sigma_clip - np.log(
            np.sqrt(2) * sigma_clip
        )
        return float(np.average(metric_by_sample, weights=None))

    def lgb_native_objective(self, preds_1d: np.ndarray, train_data: lgb.Dataset):
        labels = train_data.get_label()
        weight = train_data.get_weight()  # base_FVC passed in as weights
        preds = preds_1d.reshape(self.n_class, -1).T  # (n,2): [mu, sigma_raw]
        grad, hess = self._calc_grad_and_hess(preds, labels, weight=weight)
        return grad.T.reshape(-1), hess.T.reshape(-1)

    def lgb_native_metric(self, preds_1d: np.ndarray, train_data: lgb.Dataset):
        labels = train_data.get_label()
        weight = train_data.get_weight()
        preds = preds_1d.reshape(self.n_class, -1).T
        mu = preds[:, 0]
        sigma_raw = preds[:, 1]
        sigma = self._softplus(sigma_raw)

        scale = self._get_scale_from_weight(weight, len(labels))
        sigma_floor_v = self.sigma_floor_ml * scale
        delta_cap_v = self.delta_cap_ml * scale

        sigma_clip = np.maximum(sigma, sigma_floor_v)
        delta = np.minimum(np.abs(mu - labels), delta_cap_v)
        metric_by_sample = -np.sqrt(2.0) * delta / sigma_clip - np.log(
            np.sqrt(2.0) * sigma_clip
        )
        metric = float(np.mean(metric_by_sample))
        return "osic_metric", metric, True




## === cell 13
def _ensure_2d_pred(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 2)
    return pred


def run_single_lightgbm(
    model_param,
    fit_param,
    train_df,
    test_df,
    folds,
    features,
    target,
    fold_num=0,
    categorical=[],
    my_loss=None,
):
    trn_idx = folds[folds.fold != fold_num].index
    val_idx = folds[folds.fold == fold_num].index
    logger.info(f"len(trn_idx) : {len(trn_idx)}")
    logger.info(f"len(val_idx) : {len(val_idx)}")

    X_tr = train_df.iloc[trn_idx][features]
    y_tr = target.iloc[trn_idx].values
    X_va = train_df.iloc[val_idx][features]
    y_va = target.iloc[val_idx].values

    oof = np.zeros((len(train_df), 2), dtype=np.float64)
    predictions = np.zeros((len(test_df), 2), dtype=np.float64)

    params = dict(model_param)
    params.setdefault("boosting_type", "gbdt")
    params.setdefault("seed", SEED)
    params.setdefault("verbosity", -1)
    params.setdefault("metric", "None")
    params["objective"] = my_loss.lgb_native_objective
    params.setdefault("num_class", 2)

    w_tr = train_df.iloc[trn_idx]["base_FVC"].astype(float).values
    w_va = train_df.iloc[val_idx]["base_FVC"].astype(float).values

    lgb_tr = lgb.Dataset(
        X_tr, label=y_tr, weight=w_tr, feature_name=features, free_raw_data=False
    )
    lgb_va = lgb.Dataset(
        X_va, label=y_va, weight=w_va, feature_name=features, free_raw_data=False
    )

    num_boost_round = int(fit_param.get("num_boost_round", 10000))
    early_stopping_rounds = int(fit_param.get("early_stopping_rounds", 100))
    verbose_eval = int(fit_param.get("verbose_eval", 100))

    booster = lgb.train(
        params=params,
        train_set=lgb_tr,
        num_boost_round=num_boost_round,
        valid_sets=[lgb_tr, lgb_va],
        valid_names=["train", "valid"],
        feval=my_loss.lgb_native_metric,
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=verbose_eval),
        ],
    )

    best_iter = int(getattr(booster, "best_iteration", 0) or num_boost_round)
    if best_iter <= 0:
        best_iter = num_boost_round

    oof[val_idx] = _ensure_2d_pred(booster.predict(X_va, num_iteration=best_iter))
    predictions += _ensure_2d_pred(
        booster.predict(test_df[features], num_iteration=best_iter)
    )

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = booster.feature_importance(
        importance_type="gain"
    )
    fold_importance_df["fold"] = fold_num

    mu_oof = oof[val_idx, 0]
    sigma_v_oof = my_loss._softplus(oof[val_idx, 1])

    scale_va = my_loss.virtual_base_fvc / np.clip(w_va, 1.0, None)
    sigma_floor_v = my_loss.sigma_floor_ml * scale_va
    delta_cap_v = my_loss.delta_cap_ml * scale_va
    sigma_clip = np.maximum(sigma_v_oof, sigma_floor_v)
    delta = np.minimum(np.abs(mu_oof - y_va), delta_cap_v)
    metric_by_sample = -np.sqrt(2.0) * delta / sigma_clip - np.log(
        np.sqrt(2.0) * sigma_clip
    )
    metric = float(np.mean(metric_by_sample))

    logger.info(
        "fold{} RMSE score: {:<8.5f}".format(
            fold_num, np.sqrt(mean_squared_error(y_va, mu_oof))
        )
    )
    logger.info("fold{} Comp Metric (virtual-space): {:<8.5f}".format(fold_num, metric))

    return oof, predictions, fold_importance_df


def run_kfold_lightgbm(
    model_param,
    fit_param,
    train,
    test,
    folds,
    features,
    target,
    n_fold=5,
    categorical=[],
    my_loss=None,
):
    logger.info(
        f"================================= {n_fold}fold lightgbm ================================="
    )

    oof = np.zeros((len(train), 2), dtype=np.float64)
    predictions = np.zeros((len(test), 2), dtype=np.float64)
    feature_importance_df = pd.DataFrame()

    for fold_ in range(n_fold):
        print("Fold {}".format(fold_))
        _oof, _predictions, fold_importance_df = run_single_lightgbm(
            model_param,
            fit_param,
            train,
            test,
            folds,
            features,
            target,
            fold_num=fold_,
            categorical=categorical,
            my_loss=my_loss,
        )
        feature_importance_df = pd.concat(
            [feature_importance_df, fold_importance_df], axis=0
        )

        val_idx = folds[folds.fold == fold_].index
        oof[val_idx] = _oof[val_idx]
        predictions += _predictions / n_fold

    mu_oof = oof[:, 0]
    sigma_v_oof = my_loss._softplus(oof[:, 1])
    logger.info(
        "CV RMSE score (virtual-space): {:<8.5f}".format(
            np.sqrt(mean_squared_error(target.values, mu_oof))
        )
    )
    logger.info(
        f"========================================================================================="
    )

    return feature_importance_df, predictions, oof


def show_feature_importance(feature_importance_df, name):
    cols = (
        feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:50]
        .index
    )
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

    plt.figure(figsize=(6, 4))
    sns.barplot(
        x="importance",
        y="Feature",
        data=best_features.sort_values(by="importance", ascending=False),
    )
    plt.title("Features importance (averaged/folds)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DICT + f"feature_importance_{name}.png")




## === cell 14
target = train[TARGET]

cat_features = ["Sex", "SmokingStatus"]
num_features = [
    c for c in test.columns if (test.dtypes[c] != "object") & (c not in cat_features)
]
features = num_features + cat_features
drop_features = [ID, TARGET, "predict_Week", "base_Week", "FVC", "base_FVC"]
features = [c for c in features if c not in drop_features]

if cat_features:
    ce_oe = ce.OrdinalEncoder(cols=cat_features, handle_unknown="impute")
    ce_oe.fit(train)
    train = ce_oe.transform(train)
    test = ce_oe.transform(test)

lgb_model_param = {
    "num_class": 2,
    "metric": "None",
    "boosting_type": "gbdt",
    "learning_rate": 1e-01,
    "seed": SEED,
    "max_depth": 1,
    "lambda_l2": 5e-03,
    "verbosity": -1,
}

lgb_fit_param = {
    "num_boost_round": 10000,
    "verbose_eval": 100,
    "early_stopping_rounds": 100,
}

my_loss = OSICLossForLGBM(
    sigma_floor_ml=70.0,
    delta_cap_ml=1000.0,
    virtual_base_fvc=VIRTUAL_BASE_FVC,
)

feature_importance_df, predictions, oof = run_kfold_lightgbm(
    lgb_model_param,
    lgb_fit_param,
    train,
    test,
    folds,
    features,
    target,
    n_fold=N_FOLD,
    categorical=cat_features,
    my_loss=my_loss,
)

show_feature_importance(feature_importance_df, TARGET)



## === cell 15
predictions[:5]



## === cell 16
oof_mu = oof[:, 0] / VIRTUAL_BASE_FVC * train[["base_FVC"]].values.reshape(-1)
pred_mu = predictions[:, 0] / VIRTUAL_BASE_FVC * test[["base_FVC"]].values.reshape(-1)

oof_sigma = (
    my_loss._softplus(oof[:, 1])
    / VIRTUAL_BASE_FVC
    * train[["base_FVC"]].values.reshape(-1)
)
pred_sigma = (
    my_loss._softplus(predictions[:, 1])
    / VIRTUAL_BASE_FVC
    * test[["base_FVC"]].values.reshape(-1)
)

oof_sigma = np.maximum(oof_sigma, 1.0)
pred_sigma = np.maximum(pred_sigma, 1.0)

oof_fvc = np.c_[oof_mu, oof_sigma]
pred_fvc = np.c_[pred_mu, pred_sigma]



## === cell 17
cv_metric = OSICLossForLGBM(virtual_base_fvc=VIRTUAL_BASE_FVC).calc_comp_metric(
    oof_fvc, train["FVC"].values
)
cv_metric



## === cell 18
pred_fvc[:5]



## === cell 19
train["FVC_pred"] = oof_fvc[:, 0]
train["Confidence"] = oof_fvc[:, 1]
test["FVC_pred"] = pred_fvc[:, 0]
test["Confidence"] = pred_fvc[:, 1]



## === cell 20
submission.head()



## === cell 21
if ID not in test.columns:
    test[ID] = test["Patient"].astype(str) + "_" + test["predict_Week"].astype(str)

test_out = test[[ID, "FVC_pred", "Confidence"]].copy()

test_out["Confidence"] = np.maximum(test_out["Confidence"].values, 70.0)

sub = submission.drop(columns=["FVC", "Confidence"]).merge(test_out, on=ID, how="left")
sub = sub.rename(columns={"FVC_pred": "FVC"})

if sub["FVC"].isna().any():
    fallback = (
        submission[[ID]]
        .merge(test[[ID, "base_FVC"]], on=ID, how="left")["base_FVC"]
        .fillna(VIRTUAL_BASE_FVC)
    )
    sub["FVC"] = sub["FVC"].fillna(fallback)

if sub["Confidence"].isna().any():
    sub["Confidence"] = sub["Confidence"].fillna(70.0)

sub = sub[submission.columns]
sub.to_csv("submission.csv", index=False)
sub.head()
