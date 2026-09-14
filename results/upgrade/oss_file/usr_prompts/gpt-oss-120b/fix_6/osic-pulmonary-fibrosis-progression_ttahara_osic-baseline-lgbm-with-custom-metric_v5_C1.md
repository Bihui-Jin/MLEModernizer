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
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True




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
        usr_output = pd.concat([usr_output, _usr_output])
    output = pd.concat([output, usr_output])

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

    * Objective: return grad & hess of NLL of gaussian
    * Evaluation: return competition metric
    """

    def __init__(self, epsilon: float = 1e-09) -> None:
        """Initialize."""
        self.name = "osic_loss"
        self.n_class = 2  # FVC & Confidence
        self.epsilon = epsilon

    def __call__(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> float:
        """Calc loss."""
        mu = preds[:, 0]
        sigma = preds[:, 1]
        sigma_t = np.log(1 + np.exp(sigma))
        loss_by_sample = ((labels - mu) / sigma_t) ** 2 / 2 + np.log(
            np.sqrt(2 * np.pi) * sigma_t
        )
        loss = np.average(loss_by_sample, weight)

        return loss

    def _calc_grad_and_hess(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> tp.Tuple[np.ndarray]:
        """Calc Grad and Hess"""
        mu = preds[:, 0]
        sigma = preds[:, 1]

        sigma_t = np.log(1 + np.exp(sigma))
        grad_sigma_t = 1 / (1 + np.exp(-sigma))
        hess_sigma_t = grad_sigma_t * (1 - grad_sigma_t)

        grad = np.zeros_like(preds)
        hess = np.zeros_like(preds)
        grad[:, 0] = -(labels - mu) / sigma_t**2
        hess[:, 0] = 1 / sigma_t**2

        tmp = ((labels - mu) / sigma_t) ** 2
        grad[:, 1] = 1 / sigma_t * (1 - tmp) * grad_sigma_t
        hess[:, 1] = (
            -1 / sigma_t**2 * (1 - 3 * tmp) * grad_sigma_t**2
            + 1 / sigma_t * (1 - tmp) * hess_sigma_t
        )
        if weight is not None:
            grad = grad * weight[:, None]
            hess = hess * weight[:, None]
        return grad, hess

    def return_loss(
        self, preds: np.ndarray, data: lgb.Dataset
    ) -> tp.Tuple[str, float, bool]:
        """Return Loss for lightgbm"""
        labels = data.get_label()
        weight = data.get_weight()
        n_example = len(labels)

        preds = preds.reshape(self.n_class, n_example).T
        loss = self(preds, labels, weight)

        return self.name, loss, False

    def return_grad_and_hess(
        self, preds: np.ndarray, data: lgb.Dataset
    ) -> tp.Tuple[np.ndarray]:
        """Return Grad and Hess for lightgbm"""
        labels = data.get_label()
        weight = data.get_weight()
        n_example = len(labels)

        preds = preds.reshape(self.n_class, n_example).T
        grad, hess = self._calc_grad_and_hess(preds, labels, weight)

        grad = grad.T.reshape(n_example * self.n_class)
        hess = hess.T.reshape(n_example * self.n_class)

        return grad, hess

    def calc_comp_metric(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> float:
        """Calc competition metric."""
        sigma_clip = np.maximum(preds[:, 1], 70)
        Delta = np.minimum(np.abs(preds[:, 0] - labels), 1000)
        loss_by_sample = -np.sqrt(2) * Delta / sigma_clip - np.log(
            np.sqrt(2) * sigma_clip
        )
        loss = np.average(loss_by_sample, weight)

        return loss




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

    if not categorical:
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

    oof = np.zeros((len(train_df), 2))
    predictions = np.zeros((len(test_df), 2))

    clf = lgb.train(
        model_param,
        trn_data,
        **fit_param,
        valid_sets=[trn_data, val_data],
        obj=my_loss.return_grad_and_hess,
        feval=my_loss.return_loss,
    )
    oof[val_idx] = clf.predict(
        train_df.iloc[val_idx][features], num_iteration=clf.best_iteration
    )

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance(importance_type="gain")
    fold_importance_df["fold"] = fold_num

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration)

    logger.info(
        "fold{} RMSE score: {:<8.5f}".format(
            fold_num, np.sqrt(mean_squared_error(target[val_idx], oof[val_idx, 0]))
        )
    )
    logger.info(
        "fold{} Metric: {:<8.5f}".format(
            fold_num, my_loss(oof[val_idx], target[val_idx])
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

    oof = np.zeros((len(train), 2))
    predictions = np.zeros((len(test), 2))
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
        oof += _oof
        predictions += _predictions / n_fold

    logger.info(
        "CV RMSE score: {:<8.5f}".format(np.sqrt(mean_squared_error(target, oof[:, 0])))
    )
    logger.info("CV Metric: {:<8.5f}".format(my_loss(oof, target)))

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
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1210622237.py in <cell line: 0>()
     29 }
     30 
---> 31 feature_importance_df, predictions, oof = run_kfold_lightgbm(
     32     lgb_model_param,
     33     lgb_fit_param,

/tmp/ipykernel_55/3762956712.py in run_kfold_lightgbm(model_param, fit_param, train, test, folds, features, target, n_fold, categorical, my_loss)
     95     for fold_ in range(n_fold):
     96         print("Fold {}".format(fold_))
---> 97         _oof, _predictions, fold_importance_df = run_single_lightgbm(
     98             model_param,
     99             fit_param,

/tmp/ipykernel_55/3762956712.py in run_single_lightgbm(model_param, fit_param, train_df, test_df, folds, features, target, fold_num, categorical, my_loss)
     39 
     40     # LightGBM 4.x uses 'obj' for a custom objective (instead of 'fobj')
---> 41     clf = lgb.train(
     42         model_param,
     43         trn_data,

TypeError: train() got an unexpected keyword argument 'obj'

## === cell 15
oof = oof / VIRTUAL_BASE_FVC * train[["base_FVC"]].values
predictions = predictions / VIRTUAL_BASE_FVC * test[["base_FVC"]].values



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2143576145.py in <cell line: 0>()
      1 # convert virtual FVC back to real FVC
----> 2 oof = oof / VIRTUAL_BASE_FVC * train[["base_FVC"]].values
      3 predictions = predictions / VIRTUAL_BASE_FVC * test[["base_FVC"]].values
      4 

NameError: name 'oof' is not defined

## === cell 16
print("Training metric:", OSICLossForLGBM().calc_comp_metric(oof, train["FVC"].values))



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3229913472.py in <cell line: 0>()
----> 1 print("Training metric:", OSICLossForLGBM().calc_comp_metric(oof, train["FVC"].values))
      2 

NameError: name 'oof' is not defined

## === cell 17
train["FVC_pred"] = oof[:, 0]
train["Confidence"] = oof[:, 1]
test["FVC_pred"] = predictions[:, 0]
test["Confidence"] = predictions[:, 1]



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1873625654.py in <cell line: 0>()
----> 1 train["FVC_pred"] = oof[:, 0]
      2 train["Confidence"] = oof[:, 1]
      3 test["FVC_pred"] = predictions[:, 0]
      4 test["Confidence"] = predictions[:, 1]
      5 

NameError: name 'oof' is not defined

## === cell 18
sub = submission.drop(columns=["FVC", "Confidence"]).merge(
    test[["Patient_Week", "FVC_pred", "Confidence"]], on="Patient_Week"
)
sub.columns = submission.columns
sub.to_csv("submission.csv", index=False)
sub.head()

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/3064380265.py in <cell line: 0>()
      1 sub = submission.drop(columns=["FVC", "Confidence"]).merge(
----> 2     test[["Patient_Week", "FVC_pred", "Confidence"]], on="Patient_Week"
      3 )
      4 sub.columns = submission.columns
      5 sub.to_csv("submission.csv", index=False)

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
