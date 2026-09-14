# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

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

import seaborn
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")



## === cell 1
SEED = 42

if os.path.exists("/kaggle/input"):
    DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression/"
else:
    DATA_DIR = "../data/raw/"




## === cell 2
def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    data["Patient_Week"] = data["Patient"].astype(str) + "_" + data["Weeks"].astype(str)
    data["base_Weeks"] = data.groupby(by="Patient")["Weeks"].transform(min)
    data = data.sort_values(by=["Patient", "Weeks"])

    data = data.assign(
        **{
            "Sex": data["Sex"].map({"Female": 0, "Male": 1}),
            "SmokingStatus": data["SmokingStatus"].map(
                {"Currently smokes": 0, "Never smoked": 1, "Ex-smoker": 2}
            ),
            "base_FVC": data.groupby(by="Patient")["FVC"].transform(
                lambda x: x.iloc[0]
            ),
            "base_Percent": data.groupby(by="Patient")["Percent"].transform(
                lambda x: x.iloc[0]
            ),
            "past_record_cumcnt": data.groupby(by="Patient").transform("cumcount"),
        }
    )

    data["Week_from_base"] = data["Weeks"] - data["base_Weeks"]

    non_numeric_cols = ["Patient", "Patient_Week"]
    numeric_cols = [c for c in data.columns.tolist() if c not in non_numeric_cols]
    data[numeric_cols] = data[numeric_cols].astype(float)
    return data


train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
train = preprocessing(train)



## === cell 3
print(train.shape)
train.head()




## === cell 4
class OSICLossForLGBM:
    """
    Custom Loss for LightGBM.

    * Objective: return grad & hess of NLL of gaussian
    * Evaluation: return competition metric
    """

    def __init__(self, epsilon: float = 1) -> None:
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
        sigma_clip = np.maximum(preds[:, 1], 70)
        Delta = np.minimum(np.abs(preds[:, 0] - labels), 1000)
        loss_by_sample = -np.sqrt(2) * Delta / sigma_clip - np.log(
            np.sqrt(2) * sigma_clip
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

    def loss(self, preds: np.ndarray, data: lgb.Dataset) -> tp.Tuple[str, float, bool]:
        """Return Loss for lightgbm"""
        labels = data.get_label()
        weight = data.get_weight()
        n_example = len(labels)

        preds = preds.reshape(self.n_class, n_example).T
        loss = self(preds, labels, weight)

        return self.name, loss, True

    def grad_and_hess(
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
        train_weight=None,
        valid_weight=None,
    ):
        train_dataset = lgb.Dataset(
            X_train, y_train, feature_name=X_train.columns.tolist(), weight=train_weight
        )
        valid_dataset = lgb.Dataset(
            X_valid, y_valid, weight=valid_weight, reference=train_dataset
        )

        train_dataset, valid_dataset = self.dataset_to_binary(
            train_dataset, valid_dataset
        )

        self.model = lgb.train(
            params,
            train_dataset,
            valid_sets=[train_dataset, valid_dataset],
            **train_param
        )
        self._remove_bin_file(self.train_bin_path)
        self._remove_bin_file(self.valid_bin_path)

    def predict(self, data):
        return self.model.predict(data, num_iteration=self.model.best_iteration)

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
print(train.shape)
train.head()



## === cell 7
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
        "objective": custom_loss.grad_and_hess,
    },
    "train_params": {
        "num_boost_round": 10000,
        "callbacks": [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=100),
        ],
        "feval": custom_loss.loss,
    },
}

u_idx = train["Patient_Week"]

drop_cols = ["Patient", "Patient_Week", "FVC"]
features = [c for c in train.columns.tolist() if c not in drop_cols]

X = train[features]

y2 = np.column_stack(
    [
        train["FVC"].values.astype(np.float32),
        np.full(train.shape[0], 70.0, dtype=np.float32),
    ]
)

groups = train["Patient"]

num_fold = 5
g_kfold = model_selection.GroupKFold(n_splits=num_fold)

models = []
oof = np.zeros((train.shape[0], 2), dtype=np.float32)

for fold, (train_idx, valid_idx) in enumerate(
    g_kfold.split(X, train["FVC"], groups), 1
):
    X_train, y_train = X.iloc[train_idx, :], y2[train_idx]
    X_valid, y_valid = X.iloc[valid_idx, :], y2[valid_idx]

    y_train = np.asarray(y_train, dtype=np.float32).T.reshape(-1)
    y_valid = np.asarray(y_valid, dtype=np.float32).T.reshape(-1)

    lgb_model = LGBM_Wrapper()
    lgb_model.fit(
        params["model_params"],
        params["train_params"],
        X_train,
        y_train,
        X_valid,
        y_valid,
    )

    pred_valid = lgb_model.predict(X_valid)
    pred_valid = np.asarray(pred_valid).reshape(2, X_valid.shape[0]).T
    oof[valid_idx] = pred_valid

    models.append(lgb_model)


## --- ERROR in cell 7, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mLightGBMError[0m                             Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/214374047.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     59[0m [0;34m[0m[0m
[1;32m     60[0m     [0mlgb_model[0m [0;34m=[0m [0mLGBM_Wrapper[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 61[0;31m     lgb_model.fit(
[0m[1;32m     62[0m         [0mparams[0m[0;34m[[0m[0;34m"model_params"[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     63[0m         [0mparams[0m[0;34m[[0m[0;34m"train_params"[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/4125927722.py[0m in [0;36mfit[0;34m(self, params, train_param, X_train, y_train, X_valid, y_valid, train_weight, valid_weight)[0m
[1;32m     39[0m         )
[1;32m     40[0m [0;34m[0m[0m
[0;32m---> 41[0;31m         train_dataset, valid_dataset = self.dataset_to_binary(
[0m[1;32m     42[0m             [0mtrain_dataset[0m[0;34m,[0m [0mvalid_dataset[0m[0;34m[0m[0;34m[0m[0m
[1;32m     43[0m         )

[0;32m/tmp/ipykernel_11/4125927722.py[0m in [0;36mdataset_to_binary[0;34m(self, train_dataset, valid_dataset)[0m
[1;32m     15[0m         [0mself[0m[0;34m.[0m[0m_remove_bin_file[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mtrain_bin_path[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m         [0mself[0m[0;34m.[0m[0m_remove_bin_file[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mvalid_bin_path[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 17[0;31m         [0mtrain_dataset[0m[0;34m.[0m[0msave_binary[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mtrain_bin_path[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     18[0m         [0mvalid_dataset[0m[0;34m.[0m[0msave_binary[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mvalid_bin_path[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     19[0m         [0mtrain_dataset[0m [0;34m=[0m [0mlgb[0m[0;34m.[0m[0mDataset[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mtrain_bin_path[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36msave_binary[0;34m(self, filename)[0m
[1;32m   2714[0m         _safe_call(
[1;32m   2715[0m             _LIB.LGBM_DatasetSaveBinary(
[0;32m-> 2716[0;31m                 [0mself[0m[0;34m.[0m[0mconstruct[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m_handle[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2717[0m                 [0m_c_str[0m[0;34m([0m[0mstr[0m[0;34m([0m[0mfilename[0m[0;34m)[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2718[0m             )

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36mconstruct[0;34m(self)[0m
[1;32m   2588[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2589[0m                 [0;31m# create train[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2590[0;31m                 self._lazy_init(
[0m[1;32m   2591[0m                     [0mdata[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mdata[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2592[0m                     [0mlabel[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mlabel[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36m_lazy_init[0;34m(self, data, label, reference, weight, group, init_score, predictor, feature_name, categorical_feature, params, position)[0m
[1;32m   2207[0m                 [0;32mraise[0m [0mTypeError[0m[0;34m([0m[0;34mf"Cannot initialize Dataset from {type(data).__name__}"[0m[0;34m)[0m [0;32mfrom[0m [0merr[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2208[0m         [0;32mif[0m [0mlabel[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2209[0;31m             [0mself[0m[0;34m.[0m[0mset_label[0m[0;34m([0m[0mlabel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2210[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mget_label[0m[0;34m([0m[0;34m)[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2211[0m             [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"Label should not be None"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36mset_label[0;34m(self, label)[0m
[1;32m   3076[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3077[0m                 [0mlabel_array[0m [0;34m=[0m [0m_list_to_1d_numpy[0m[0;34m([0m[0mlabel[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mnp[0m[0;34m.[0m[0mfloat32[0m[0;34m,[0m [0mname[0m[0;34m=[0m[0;34m"label"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3078[0;31m             [0mself[0m[0;34m.[0m[0mset_field[0m[0;34m([0m[0;34m"label"[0m[0;34m,[0m [0mlabel_array[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3079[0m             [0mself[0m[0;34m.[0m[0mlabel[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mget_field[0m[0;34m([0m[0;34m"label"[0m[0;34m)[0m  [0;31m# original values can be modified at cpp side[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3080[0m         [0;32mreturn[0m [0mself[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36mset_field[0;34m(self, field_name, data)[0m
[1;32m   2845[0m         [0;32mif[0m [0mtype_data[0m [0;34m!=[0m [0m_FIELD_TYPE_MAPPER[0m[0;34m[[0m[0mfield_name[0m[0;34m][0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2846[0m             [0;32mraise[0m [0mTypeError[0m[0;34m([0m[0;34m"Input type error for set_field"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2847[0;31m         _safe_call(
[0m[1;32m   2848[0m             _LIB.LGBM_DatasetSetField(
[1;32m   2849[0m                 [0mself[0m[0;34m.[0m[0m_handle[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36m_safe_call[0;34m(ret)[0m
[1;32m    311[0m     """
[1;32m    312[0m     [0;32mif[0m [0mret[0m [0;34m!=[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 313[0;31m         [0;32mraise[0m [0mLightGBMError[0m[0;34m([0m[0m_LIB[0m[0;34m.[0m[0mLGBM_GetLastError[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mdecode[0m[0;34m([0m[0;34m"utf-8"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    314[0m [0;34m[0m[0m
[1;32m    315[0m [0;34m[0m[0m

[0;31mLightGBMError[0m: Length of labels differs from the length of #data

## === cell 8
def score(fvc_true, fvc_pred, sigma):
    sigma_clip = np.maximum(sigma, 70)
    delta = np.minimum(np.abs(fvc_true - fvc_pred), 1000)
    metric = -(np.sqrt(2) * delta / sigma_clip) - np.log(np.sqrt(2) * sigma_clip)
    return np.mean(metric)


oof_fvc = oof[:, 0]
oof_sigma = np.log1p(np.exp(oof[:, 1]))
oof_sigma = np.maximum(oof_sigma, 70.0)

score(train["FVC"], oof_fvc, oof_sigma)
