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

-6.8923

# 6. Current score

-11.74199

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -19.82007) has done: 'I fix the LightGBM v4+ API break that caused `lgb.train()` to reject `fobj` by switching to the supported `objective` and `feval` call pattern while keeping your custom loss/metric unchanged. I also correct a fold-aggregation logic bug where `oof` was being summed across folds (should be assigned per validation fold) to avoid distorted OOF predictions and downstream errors. Then I make sure the test dataframe retains `Patient_Week` and that prediction columns exist before merging, so the submission is generated successfully as `submission.csv` with the required columns. These changes are minimal, unblock end-to-end execution, and should yield a sensible score without altering the modeling approach.'
- What this solution (achieved -19.82007) has done: 'Your current score is far below the target, so we should improve it with minimal, low-risk changes that keep your model and training loop intact. The largest issue is that `run_kfold_lightgbm()` incorrectly *adds* OOF predictions across folds, which distorts the validation metric and typically harms generalization; we change that to assign per-fold OOF (while keeping test averaging the same). Next, your Confidence post-processing is inconsistent with your training parameterization (sigma is modeled via softplus inside the objective), so we apply the same `softplus` transform at inference before clipping to 70, which usually improves the Laplace log-likelihood without changing the core approach. Finally, we keep submission construction the same but ensure numeric casting and alignment remain stable.'
- What this solution (achieved -19.82007) has done: 'We keep your LightGBM setup and custom objective exactly as-is, but fix a key mismatch between training and evaluation that is likely depressing your score: your evaluation/metric path treats the raw second output as a sigma, while the objective actually trains sigma through a softplus transform. Specifically, we apply the same softplus+clip(70) transform inside `OSICLossForLGBM.__call__`, so the fold metric aligns with the loss you optimize and the competition metric definition. We also use a numerically stable softplus to avoid overflow and ensure the same transform is used consistently for both OOF logging and test-time `Confidence`. These are minimal, low-risk changes that should improve the Laplace log-likelihood (raise the score) without changing the model architecture or training loop.'
- What this solution (achieved -10.77033) has done: 'We keep your LightGBM multi-output setup and custom objective/metric intact, and make only two score-relevant fixes that typically lift OSIC scores substantially. First, we change the training target to match the objective’s expectation (the objective assumes per-row label is the *base-vs-predict delta*, not absolute FVC), by training on `FVC - base_FVC` and then reconstructing `FVC_pred = base_FVC + delta_pred` at inference. Second, we use the same delta reconstruction consistently for OOF/test and keep your existing softplus→clip(70) confidence post-processing, so the evaluation semantics align with the competition metric and what the model is optimizing. Everything else (folding, features, LightGBM params, training loop, submission merge) stays the same and still writes `submission.csv`.'
- What this solution (achieved -10.4756) has done: 'We fix the `category_encoders` transform crash by ensuring train and test have identical columns when fitting/transforming the `OrdinalEncoder` (the error comes from fitting on `train` with extra columns like `Patient_Week`/baseline merges that aren’t present in `test`). We do this with a minimal change: fit/transform on the same `features + drop_features + Patient` column set for both, then proceed with your existing LightGBM training exactly as written. After that, we restore the already-intended delta-target reconstruction for `FVC_pred` and apply the same stable softplus+clip(70) for `Confidence`, so the pipeline runs end-to-end. Finally, we guarantee the submission is written as `submission.csv` with the required columns and aligned `Patient_Week` keys.'
- What this solution (achieved -9.9201) has done: 'To move your score up toward the target with minimal risk, I keep the same LightGBM custom objective/metric and training loop, but fix a key mismatch: during training the objective uses an *unclipped* softplus sigma, while your metric (and Kaggle) uses `max(sigma, 70)`—so the model is incentivized to predict sigma < 70 that never helps the metric. I align the objective with the competition metric by applying the same `max(softplus(sigma), 70)` inside the gradient/hessian path (this changes only sigma parameterization, not the model architecture or loop). I also make the objective’s softplus numerically stable (to match inference) and add `force_col_wise` to avoid potential LightGBM performance variability, keeping everything deterministic. The submission writing remains identical and still produces `submission.csv`.'
- What this solution (achieved -10.81761) has done: 'Your current score (-9.9201) is substantially below the target (-6.8923), so we should improve it with a minimal, low-risk change that doesn’t alter the model/training loop. The biggest remaining issue is that the custom objective is mathematically inconsistent with the competition metric: it uses an L2/Gaussian-style gradient for `mu`, while the metric is Laplace (L1) based; aligning the `mu` gradient/hessian to Laplace improves the metric directly without changing the overall approach. I keep the same two-output LightGBM setup, same features, same folds, same inference post-processing, but update only the gradient/hessian formulas to the correct Laplace NLL (with sigma clipped at 70, as you already do). This should move the score upward toward the target while preserving evaluation semantics and still writing a valid `submission.csv`.'
- What this solution (achieved -11.74199) has done: 'We make one minimal, score-relevant fix in the custom objective: the gradient for `mu` currently uses the hard `sign(e)` which makes LightGBM see (almost) zero second-derivative and often trains poorly; we switch to a smooth approximation `e / sqrt(e^2 + eps)` (still Laplace-aligned) and provide a small positive hessian so boosting can optimize stably. We also clip `Delta` at 1000 inside the objective gradient path (to match the competition metric’s truncation) while keeping your inference and submission code unchanged. These are tiny changes that preserve your 2-output LightGBM setup, features, folds, and post-processing, but should move the score upward toward the target by making training better aligned and numerically stable. The script still runs end-to-end and writes `submission.csv` in the required format.'

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

from sklearn.model_selection import GroupKFold
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
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    if not logger.handlers:
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
    torch.backends.cudnn.deterministic = True




## === cell 2
OUTPUT_DICT = "./"

ID = "Patient_Week"
TARGET = "FVC"
SEED = 42
seed_everything(seed=SEED)

N_FOLD = 4



## === cell 3
train_raw = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
train_raw[ID] = train_raw["Patient"].astype(str) + "_" + train_raw["Weeks"].astype(str)
print(train_raw.shape)
train_raw.head()



## === cell 4
baseline = (
    train_raw.sort_values(["Patient", "Weeks"])
    .groupby("Patient", as_index=False)
    .first()[["Patient", "Weeks", "FVC", "Percent", "Age"]]
    .rename(
        columns={
            "Weeks": "base_Week",
            "FVC": "base_FVC",
            "Percent": "base_Percent",
            "Age": "base_Age",
        }
    )
)

train = train_raw.merge(baseline, on="Patient", how="left")
train = train.rename(columns={"Weeks": "predict_Week"})
train["Week_passed"] = train["predict_Week"] - train["base_Week"]
train = train[train["Week_passed"] != 0].reset_index(drop=True)

print(train.shape)
train.head()



## === cell 5
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



## === cell 6
submission = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
print(submission.shape)
submission.head()



## === cell 7
folds = train[[ID, "Patient", TARGET]].copy()
Fold = GroupKFold(n_splits=N_FOLD)
groups = folds["Patient"].values
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[TARGET], groups)):
    folds.loc[val_index, "fold"] = int(n)
folds["fold"] = folds["fold"].astype(int)
folds.head()




## === cell 8
class OSICLossForLGBM:
    """
    Custom Loss for LightGBM.

    * Objective: return grad & hess of competition-aligned Laplace NLL
    * Evaluation: return competition metric (Laplace log likelihood with clipping)
    """

    def __init__(self, epsilon: float = 1.0) -> None:
        self.name = "osic_loss"
        self.n_class = 2  # FVC & Confidence
        self.epsilon = float(epsilon)

    @staticmethod
    def _softplus_stable(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    def __call__(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> float:
        sigma = self._softplus_stable(preds[:, 1])
        sigma_clip = np.maximum(sigma, 70.0)

        Delta = np.minimum(np.abs(preds[:, 0] - labels), 1000.0)
        loss_by_sample = -np.sqrt(2.0) * Delta / sigma_clip - np.log(
            np.sqrt(2.0) * sigma_clip
        )
        loss = (
            np.average(loss_by_sample, weights=weight)
            if weight is not None
            else np.mean(loss_by_sample)
        )
        return float(loss)

    def _calc_grad_and_hess(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Change (score-relevant, minimal):
        - Use a smooth approximation to sign(e): e/sqrt(e^2+eps), so LightGBM gets a usable curvature signal.
        - Provide a small positive Hessian for mu consistent with the smooth L1 (pseudo-Huber-like) surrogate.
        - Clip Delta at 1000 inside the objective gradient path to match the competition metric truncation.
        These keep the same 2-output formulation and sigma softplus+clip(70) parameterization.
        """
        mu = preds[:, 0].astype(np.float64)
        sigma_raw = preds[:, 1].astype(np.float64)

        sp = self._softplus_stable(sigma_raw)

        sigmoid = 1.0 / (1.0 + np.exp(-sigma_raw))

        sigma_t = np.maximum(sp, 70.0)

        clip_mask = (sp > 70.0).astype(np.float64)
        grad_sigma_t = sigmoid * clip_mask
        hess_sigma_t = (sigmoid * (1.0 - sigmoid)) * clip_mask

        grad = np.zeros_like(preds, dtype=np.float64)
        hess = np.zeros_like(preds, dtype=np.float64)

        e = labels.astype(np.float64) - mu

        e_trunc = np.clip(e, -1000.0, 1000.0)

        denom = np.sqrt(e_trunc * e_trunc + self.epsilon)
        smooth_sign = e_trunc / denom  # ~sign(e) but smooth

        grad[:, 0] = np.sqrt(2.0) * (smooth_sign / sigma_t)

        d_smoothsign_dmu = self.epsilon / (
            denom**3
        )  # because d(e_trunc)/dmu = -1 within clip range
        in_clip = (np.abs(e) < 1000.0).astype(np.float64)
        hess[:, 0] = np.sqrt(2.0) * (d_smoothsign_dmu / sigma_t) * in_clip

        abs_e_trunc_smooth = denom  # ~|e_trunc| with smoothing
        dL_dsigma = -np.sqrt(2.0) * (abs_e_trunc_smooth / (sigma_t**2)) + (
            1.0 / sigma_t
        )

        grad[:, 1] = dL_dsigma * grad_sigma_t

        d2L_dsigma2 = 2.0 * np.sqrt(2.0) * (abs_e_trunc_smooth / (sigma_t**3)) - (
            1.0 / (sigma_t**2)
        )
        hess[:, 1] = d2L_dsigma2 * (grad_sigma_t**2) + dL_dsigma * hess_sigma_t

        if weight is not None:
            grad = grad * weight[:, None]
            hess = hess * weight[:, None]
        return grad, hess

    def return_loss(
        self, preds: np.ndarray, data: lgb.Dataset
    ) -> tp.Tuple[str, float, bool]:
        labels = data.get_label()
        weight = data.get_weight()
        n_example = len(labels)

        preds = preds.reshape(self.n_class, n_example).T
        loss = self(preds, labels, weight)

        return self.name, loss, True

    def return_grad_and_hess(
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




## === cell 9
def _predict_2col_from_lgbm(
    clf: lgb.Booster, X: pd.DataFrame, num_iteration=None
) -> np.ndarray:
    pred = clf.predict(X, num_iteration=num_iteration)
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

    if categorical == []:
        trn_data = lgb.Dataset(
            train_df.iloc[trn_idx][features], label=target.iloc[trn_idx]
        )
        val_data = lgb.Dataset(
            train_df.iloc[val_idx][features], label=target.iloc[val_idx]
        )
    else:
        trn_data = lgb.Dataset(
            train_df.iloc[trn_idx][features],
            label=target.iloc[trn_idx],
            categorical_feature=categorical,
        )
        val_data = lgb.Dataset(
            train_df.iloc[val_idx][features],
            label=target.iloc[val_idx],
            categorical_feature=categorical,
        )

    oof = np.zeros((len(train_df), 2), dtype=np.float64)
    predictions = np.zeros((len(test_df), 2), dtype=np.float64)

    callbacks = []
    if "verbose_eval" in fit_param:
        callbacks.append(lgb.log_evaluation(period=int(fit_param["verbose_eval"])))
    if (
        "early_stopping_rounds" in fit_param
        and fit_param["early_stopping_rounds"] is not None
    ):
        callbacks.append(
            lgb.early_stopping(
                stopping_rounds=int(fit_param["early_stopping_rounds"]), verbose=False
            )
        )

    model_param_local = dict(model_param)
    model_param_local["objective"] = my_loss.return_grad_and_hess

    clf = lgb.train(
        model_param_local,
        trn_data,
        num_boost_round=int(fit_param.get("num_boost_round", 10000)),
        valid_sets=[trn_data, val_data],
        valid_names=["train", "valid"],
        feval=my_loss.return_loss,
        callbacks=callbacks,
    )

    best_it = (
        clf.best_iteration
        if clf.best_iteration is not None
        else int(fit_param.get("num_boost_round", 10000))
    )

    oof[val_idx] = _predict_2col_from_lgbm(
        clf, train_df.iloc[val_idx][features], num_iteration=best_it
    )
    predictions = _predict_2col_from_lgbm(clf, test_df[features], num_iteration=best_it)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance(importance_type="gain")
    fold_importance_df["fold"] = fold_num

    logger.info(
        "fold{} RMSE score: {:<8.5f}".format(
            fold_num, np.sqrt(mean_squared_error(target.iloc[val_idx], oof[val_idx, 0]))
        )
    )
    logger.info(
        "fold{} Metric: {:<8.5f}".format(
            fold_num, my_loss(oof[val_idx], target.iloc[val_idx].values)
        )
    )

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

        oof = np.where(_oof != 0, _oof, oof)
        predictions += _predictions / n_fold

    logger.info(
        "CV RMSE score: {:<8.5f}".format(np.sqrt(mean_squared_error(target, oof[:, 0])))
    )
    logger.info("CV Metric: {:<8.5f}".format(my_loss(oof, target.values)))

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




## === cell 10
target = (train[TARGET] - train["base_FVC"]).astype(float)

test[TARGET] = np.nan

cat_features = ["Sex", "SmokingStatus"]
num_features = [
    c for c in test.columns if (test.dtypes[c] != "object") & (c not in cat_features)
]
features = num_features + cat_features
drop_features = [ID, TARGET, "predict_Week", "base_Week"]
features = [c for c in features if c not in drop_features]

encode_cols = ["Patient"] + drop_features + features
encode_cols = [c for c in encode_cols if c in train.columns and c in test.columns]

if cat_features:
    ce_oe = ce.OrdinalEncoder(cols=cat_features, handle_unknown="impute")
    ce_oe.fit(train[encode_cols])
    train_enc = ce_oe.transform(train[encode_cols])
    test_enc = ce_oe.transform(test[encode_cols])

    for c in cat_features:
        train[c] = train_enc[c].values
        test[c] = test_enc[c].values

lgb_model_param = {
    "num_class": 2,
    "metric": "None",
    "boosting_type": "gbdt",
    "learning_rate": 1e-02,
    "seed": SEED,
    "subsample": 0.4,
    "subsample_freq": 1,
    "feature_fraction": 0.75,
    "max_depth": 1,
    "verbosity": -1,
    "force_col_wise": True,
}

lgb_fit_param = {
    "num_boost_round": 10000,
    "verbose_eval": 100,
    "early_stopping_rounds": 1000,
}

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
    my_loss=OSICLossForLGBM(),
)

show_feature_importance(feature_importance_df, TARGET)



## === cell 11
oof[:5, :]



## === cell 12
predictions[:5]




## === cell 13
def _softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


train["FVC_pred"] = (train["base_FVC"].astype(float).values + oof[:, 0]).astype(float)
train["Confidence"] = np.maximum(_softplus(oof[:, 1]), 70.0)

test["FVC_pred"] = (test["base_FVC"].astype(float).values + predictions[:, 0]).astype(
    float
)
test["Confidence"] = np.maximum(_softplus(predictions[:, 1]), 70.0)

train[["FVC_pred", "Confidence"]].head()



## === cell 14
submission.head()



## === cell 15
assert (
    "Patient_Week" in test.columns
), "test must contain Patient_Week for submission merge."
assert (
    "FVC_pred" in test.columns and "Confidence" in test.columns
), "pred columns missing."

sub = submission.drop(columns=["FVC", "Confidence"]).merge(
    test[["Patient_Week", "FVC_pred", "Confidence"]], on="Patient_Week", how="left"
)
sub = sub.rename(columns={"FVC_pred": "FVC"})[["Patient_Week", "FVC", "Confidence"]]

sub["FVC"] = sub["FVC"].fillna(submission["FVC"]).astype(float)
sub["Confidence"] = sub["Confidence"].fillna(submission["Confidence"]).astype(float)

sub.to_csv("submission.csv", index=False)
print(sub.shape)
sub.head()
