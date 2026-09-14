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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

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

    * Objective: NLL of gaussian (mu, sigma_raw with softplus)
    * Evaluation: competition metric (Laplace log likelihood variant)
    """

    def __init__(self, epsilon: float = 1e-09) -> None:
        self.name = "osic_loss"
        self.n_class = 2  # mu & sigma_raw
        self.epsilon = epsilon

    def __call__(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> float:
        mu = preds[:, 0]
        sigma_raw = preds[:, 1]
        sigma_t = np.log1p(np.exp(sigma_raw)) + self.epsilon  # softplus
        loss_by_sample = ((labels - mu) / sigma_t) ** 2 / 2 + np.log(
            np.sqrt(2 * np.pi) * sigma_t
        )
        return np.average(loss_by_sample, weights=weight)

    def _calc_grad_and_hess(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        mu = preds[:, 0]
        sigma_raw = preds[:, 1]

        sigma_t = np.log1p(np.exp(sigma_raw)) + self.epsilon
        grad_sigma_t = 1 / (1 + np.exp(-sigma_raw))  # sigmoid
        hess_sigma_t = grad_sigma_t * (1 - grad_sigma_t)

        grad = np.zeros_like(preds)
        hess = np.zeros_like(preds)

        grad[:, 0] = -(labels - mu) / sigma_t**2
        hess[:, 0] = 1 / sigma_t**2

        tmp = ((labels - mu) / sigma_t) ** 2
        grad[:, 1] = (1 / sigma_t) * (1 - tmp) * grad_sigma_t
        hess[:, 1] = (
            -(1 / sigma_t**2) * (1 - 3 * tmp) * grad_sigma_t**2
            + (1 / sigma_t) * (1 - tmp) * hess_sigma_t
        )

        if weight is not None:
            grad = grad * weight[:, None]
            hess = hess * weight[:, None]
        return grad, hess

    def calc_comp_metric(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> float:
        sigma_clip = np.maximum(preds[:, 1], 70)
        delta = np.minimum(np.abs(preds[:, 0] - labels), 1000)
        metric_by_sample = -np.sqrt(2) * delta / sigma_clip - np.log(
            np.sqrt(2) * sigma_clip
        )
        return np.average(metric_by_sample, weights=weight)

    def lgb_sklearn_objective(self, y_true: np.ndarray, y_pred: np.ndarray):
        preds = y_pred.reshape(-1, self.n_class)
        grad, hess = self._calc_grad_and_hess(preds, y_true, weight=None)
        return grad, hess

    def lgb_sklearn_metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        preds = y_pred.reshape(-1, self.n_class)
        mu = preds[:, 0]
        sigma_raw = preds[:, 1]
        sigma_ml = np.log1p(np.exp(sigma_raw)) + self.epsilon
        metric = self.calc_comp_metric(np.c_[mu, sigma_ml], y_true, weight=None)
        return "osic_metric", metric, True




## === cell 13
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

    lgbm_param = dict(model_param)
    lgbm_param.setdefault("objective", "None")
    lgbm_param.setdefault("metric", "None")
    lgbm_param.setdefault("verbosity", -1)
    lgbm_param.setdefault("seed", SEED)

    clf = lgb.LGBMRegressor(**lgbm_param)
    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_tr, y_tr), (X_va, y_va)],
        eval_names=["train", "valid"],
        eval_metric=my_loss.lgb_sklearn_metric,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=int(fit_param.get("early_stopping_rounds", 100)),
                verbose=False,
            ),
            lgb.log_evaluation(period=int(fit_param.get("verbose_eval", 100))),
        ],
    )

    best_iter = int(
        getattr(clf, "best_iteration_", 0) or fit_param.get("num_boost_round", 10000)
    )

    oof[val_idx] = clf.predict(X_va, num_iteration=best_iter)
    predictions += clf.predict(test_df[features], num_iteration=best_iter)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.booster_.feature_importance(
        importance_type="gain"
    )
    fold_importance_df["fold"] = fold_num

    mu_oof = oof[val_idx, 0]
    sigma_ml_oof = np.log1p(np.exp(oof[val_idx, 1])) + my_loss.epsilon
    metric = my_loss.calc_comp_metric(np.c_[mu_oof, sigma_ml_oof], y_va)

    logger.info(
        "fold{} RMSE score: {:<8.5f}".format(
            fold_num, np.sqrt(mean_squared_error(y_va, mu_oof))
        )
    )
    logger.info("fold{} Comp Metric: {:<8.5f}".format(fold_num, metric))

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
    sigma_ml_oof = np.log1p(np.exp(oof[:, 1])) + my_loss.epsilon
    cv_metric = my_loss.calc_comp_metric(np.c_[mu_oof, sigma_ml_oof], target.values)

    logger.info(
        "CV RMSE score: {:<8.5f}".format(
            np.sqrt(mean_squared_error(target.values, mu_oof))
        )
    )
    logger.info("CV Comp Metric: {:<8.5f}".format(cv_metric))
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



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
LightGBMError                             Traceback (most recent call last)
/tmp/ipykernel_11/3624828380.py in <cell line: 0>()
     32 }
     33 
---> 34 feature_importance_df, predictions, oof = run_kfold_lightgbm(
     35     lgb_model_param,
     36     lgb_fit_param,

/tmp/ipykernel_11/4096222658.py in run_kfold_lightgbm(model_param, fit_param, train, test, folds, features, target, n_fold, categorical, my_loss)
    102     for fold_ in range(n_fold):
    103         print("Fold {}".format(fold_))
--> 104         _oof, _predictions, fold_importance_df = run_single_lightgbm(
    105             model_param,
    106             fit_param,

/tmp/ipykernel_11/4096222658.py in run_single_lightgbm(model_param, fit_param, train_df, test_df, folds, features, target, fold_num, categorical, my_loss)
     34 
     35     clf = lgb.LGBMRegressor(**lgbm_param)
---> 36     clf.fit(
     37         X_tr,
     38         y_tr,

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in fit(self, X, y, sample_weight, init_score, eval_set, eval_names, eval_sample_weight, eval_init_score, eval_metric, feature_name, categorical_feature, callbacks, init_model)
   1396     ) -> "LGBMRegressor":
   1397         """Docstring is inherited from the LGBMModel."""
-> 1398         super().fit(
   1399             X,
   1400             y,

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in fit(self, X, y, sample_weight, init_score, group, eval_set, eval_names, eval_sample_weight, eval_class_weight, eval_init_score, eval_group, eval_metric, feature_name, categorical_feature, callbacks, init_model)
   1047         callbacks.append(record_evaluation(evals_result))
   1048 
-> 1049         self._Booster = train(
   1050             params=params,
   1051             train_set=train_set,

/usr/local/lib/python3.11/dist-packages/lightgbm/engine.py in train(params, train_set, num_boost_round, valid_sets, valid_names, feval, init_model, keep_training_booster, callbacks)
    320             )
    321 
--> 322         booster.update(fobj=fobj)
    323 
    324         evaluation_result_list: List[_LGBM_BoosterEvalMethodResultType] = []

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in update(self, train_set, fobj)
   4152             if self.__set_objective_to_none:
   4153                 raise LightGBMError("Cannot update due to null objective function.")
-> 4154             _safe_call(
   4155                 _LIB.LGBM_BoosterUpdateOneIter(
   4156                     self._handle,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _safe_call(ret)
    311     """
    312     if ret != 0:
--> 313         raise LightGBMError(_LIB.LGBM_GetLastError().decode("utf-8"))
    314 
    315 

LightGBMError: No objective function provided

## === cell 15
predictions[:5]



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1829928051.py in <cell line: 0>()
----> 1 predictions[:5]
      2 

NameError: name 'predictions' is not defined

## === cell 16
my_loss = OSICLossForLGBM()

oof_mu = oof[:, 0] / VIRTUAL_BASE_FVC * train[["base_FVC"]].values.reshape(-1)
pred_mu = predictions[:, 0] / VIRTUAL_BASE_FVC * test[["base_FVC"]].values.reshape(-1)

oof_sigma = (
    (np.log1p(np.exp(oof[:, 1])) + my_loss.epsilon)
    / VIRTUAL_BASE_FVC
    * train[["base_FVC"]].values.reshape(-1)
)
pred_sigma = (
    (np.log1p(np.exp(predictions[:, 1])) + my_loss.epsilon)
    / VIRTUAL_BASE_FVC
    * test[["base_FVC"]].values.reshape(-1)
)

oof_sigma = np.maximum(oof_sigma, 1.0)
pred_sigma = np.maximum(pred_sigma, 1.0)

oof_fvc = np.c_[oof_mu, oof_sigma]
pred_fvc = np.c_[pred_mu, pred_sigma]



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3108416333.py in <cell line: 0>()
      1 my_loss = OSICLossForLGBM()
      2 
----> 3 oof_mu = oof[:, 0] / VIRTUAL_BASE_FVC * train[["base_FVC"]].values.reshape(-1)
      4 pred_mu = predictions[:, 0] / VIRTUAL_BASE_FVC * test[["base_FVC"]].values.reshape(-1)
      5 

NameError: name 'oof' is not defined

## === cell 17
cv_metric = OSICLossForLGBM().calc_comp_metric(oof_fvc, train["FVC"].values)
cv_metric



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3557031757.py in <cell line: 0>()
----> 1 cv_metric = OSICLossForLGBM().calc_comp_metric(oof_fvc, train["FVC"].values)
      2 cv_metric
      3 

NameError: name 'oof_fvc' is not defined

## === cell 18
pred_fvc[:5]



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/895425998.py in <cell line: 0>()
----> 1 pred_fvc[:5]
      2 

NameError: name 'pred_fvc' is not defined

## === cell 19
train["FVC_pred"] = oof_fvc[:, 0]
train["Confidence"] = oof_fvc[:, 1]
test["FVC_pred"] = pred_fvc[:, 0]
test["Confidence"] = pred_fvc[:, 1]



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1065024256.py in <cell line: 0>()
----> 1 train["FVC_pred"] = oof_fvc[:, 0]
      2 train["Confidence"] = oof_fvc[:, 1]
      3 test["FVC_pred"] = pred_fvc[:, 0]
      4 test["Confidence"] = pred_fvc[:, 1]
      5 

NameError: name 'oof_fvc' is not defined

## === cell 20
submission.head()



## === cell 21
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

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2372213614.py in <cell line: 0>()
      1 # BUGFIX: Ensure merge keys exist and submission is always complete/valid.
----> 2 test_out = test[[ID, "FVC_pred", "Confidence"]].copy()
      3 test_out["Confidence"] = np.maximum(test_out["Confidence"].values, 70.0)
      4 
      5 sub = submission.drop(columns=["FVC", "Confidence"]).merge(test_out, on=ID, how="left")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['FVC_pred', 'Confidence'] not in index"
