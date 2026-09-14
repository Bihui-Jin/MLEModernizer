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

import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)



## === cell 1
if os.path.exists("/kaggle/input"):
    DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression/"
else:
    if os.path.exists("/kaggle/data/osic-pulmonary-fibrosis-progression/"):
        DATA_DIR = "/kaggle/data/osic-pulmonary-fibrosis-progression/"
    elif os.path.exists("../data/osic-pulmonary-fibrosis-progression/"):
        DATA_DIR = "../data/osic-pulmonary-fibrosis-progression/"
    else:
        DATA_DIR = "../data/raw/"

print("DATA_DIR:", DATA_DIR)




## === cell 2
def preprocessing(data: pd.DataFrame, is_test: bool = True) -> pd.DataFrame:
    features = []
    for patient, u_data in data.groupby("Patient", sort=False):
        feature = pd.DataFrame(
            {
                "current_FVC": u_data["FVC"].to_numpy(),
                "current_Percent": u_data["Percent"].to_numpy(),
                "current_Age": u_data["Age"].to_numpy(),
                "current_Week": u_data["Weeks"].to_numpy(),
                "Patient": u_data["Patient"].to_numpy(),
                "Sex": u_data["Sex"].map({"Female": 0, "Male": 1}).to_numpy(),
                "SmokingStatus": u_data["SmokingStatus"]
                .map({"Currently smokes": 0, "Never smoked": 1, "Ex-smoker": 2})
                .to_numpy(),
            }
        )
        features.append(feature)
    features = pd.concat(features, axis=0, ignore_index=True)

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

    dst_data["passed_Weeks"] = dst_data["current_Week"] - dst_data["pred_Weeks"]
    dst_data = dst_data.reset_index(drop=True)
    return dst_data


train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
train = preprocessing(train, is_test=False)



## === cell 3
print(train.shape)
train.head()




## === cell 4
class OSICLossForLGBM:
    """
    Custom Loss for LightGBM.

    Evaluation (feval) uses the competition Laplace metric for FVC, with:
      sigma_clipped = max(sigma, 70), Delta = min(|FVC_true - FVC_pred|, 1000).

    Score-improving, minimal change:
      - Train as a true 2-target problem by letting labels be (FVC, sigma_target).
        This makes the second head (Confidence) learnable instead of unconstrained.
      - Keep the same objective form for FVC; add a small, stable quadratic term to
        learn sigma around a reasonable target scale (>=70).
      - This preserves evaluation semantics while making predictions better calibrated.
    """

    def __init__(
        self,
        epsilon: float = 1e-3,
        clip_delta: float = 1000.0,
        softclip_tau: float = 25.0,
        sigma_floor: float = 70.0,
        sigma_weight: float = 0.05,
    ) -> None:
        self.name = "osic_loss"
        self.n_class = 2  # FVC & Confidence
        self.epsilon = float(epsilon)
        self.clip_delta = float(clip_delta)
        self.softclip_tau = float(softclip_tau)
        self.sigma_floor = float(sigma_floor)
        self.sigma_weight = float(sigma_weight)

    @staticmethod
    def _softplus(x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=np.float64)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        expx = np.exp(x[~pos])
        out[~pos] = expx / (1.0 + expx)
        return out

    def _soft_clipped_abs(
        self, r: np.ndarray
    ) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = self.clip_delta
        tau = self.softclip_tau

        abs_r = np.abs(r)
        sign_r = np.sign(r)
        z = (d - abs_r) / tau

        sig = self._sigmoid(z)

        s = d - tau * self._softplus(z)

        ds_dabs = sig
        d2s_dabs2 = -(sig * (1.0 - sig)) / tau

        ds_dr = ds_dabs * sign_r
        d2s_dr2 = d2s_dabs2 * (sign_r**2)

        return s, ds_dr, d2s_dr2

    def __call__(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> float:
        fvc_true = labels[:, 0]
        sigma_true = labels[:, 1]

        sigma_clip = np.maximum(preds[:, 1], self.sigma_floor)
        Delta = np.minimum(np.abs(preds[:, 0] - fvc_true), self.clip_delta)

        loss_by_sample = -np.sqrt(2) * Delta / sigma_clip - np.log(
            np.sqrt(2) * sigma_clip
        )

        loss_by_sample = loss_by_sample + self.sigma_weight * (
            (preds[:, 1] - sigma_true) ** 2
        )

        loss = np.average(loss_by_sample, weight)
        return float(loss)

    def _calc_grad_and_hess(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        fvc_true = labels[:, 0]
        sigma_true = labels[:, 1]

        mu = preds[:, 0]
        raw_sigma = preds[:, 1]

        sigma_pos = self._softplus(raw_sigma)
        sigm = self._sigmoid(raw_sigma)
        d_sigma_pos_d_raw = sigm
        d2_sigma_pos_d_raw2 = sigm * (1.0 - sigm)

        sigma_clip = np.maximum(sigma_pos, self.sigma_floor)
        active = (sigma_pos >= self.sigma_floor).astype(np.float64)
        d_sigma_clip_d_raw = d_sigma_pos_d_raw * active
        d2_sigma_clip_d_raw2 = d2_sigma_pos_d_raw2 * active

        r = mu - fvc_true
        Delta_s, dDelta_dr, d2Delta_dr2 = self._soft_clipped_abs(r)

        sqrt2 = np.sqrt(2.0)

        grad = np.zeros_like(preds, dtype=np.float64)
        hess = np.zeros_like(preds, dtype=np.float64)

        grad[:, 0] = -sqrt2 * dDelta_dr / sigma_clip
        hess[:, 0] = np.maximum(sqrt2 * (-d2Delta_dr2) / sigma_clip, 0.0) + self.epsilon

        dL_d_sigma = -sqrt2 * Delta_s / (sigma_clip**2) + 1.0 / sigma_clip
        d2L_d_sigma2 = 2.0 * sqrt2 * Delta_s / (sigma_clip**3) - 1.0 / (sigma_clip**2)

        aux_grad_raw = 2.0 * self.sigma_weight * (raw_sigma - sigma_true)
        aux_hess_raw = 2.0 * self.sigma_weight * np.ones_like(raw_sigma)

        grad[:, 1] = dL_d_sigma * d_sigma_clip_d_raw + aux_grad_raw
        hess[:, 1] = (
            d2L_d_sigma2 * (d_sigma_clip_d_raw**2)
            + dL_d_sigma * d2_sigma_clip_d_raw2
            + aux_hess_raw
            + self.epsilon
        )

        if weight is not None:
            grad = grad * weight[:, None]
            hess = hess * weight[:, None]

        return grad, hess

    def loss(self, preds: np.ndarray, data: lgb.Dataset) -> tp.Tuple[str, float, bool]:
        labels = data.get_label()
        weight = data.get_weight()

        n_example = data.num_data()
        labels = labels.reshape(n_example, self.n_class)

        preds = preds.reshape(self.n_class, n_example).T
        loss = self(preds, labels, weight)
        return self.name, loss, True

    def grad_and_hess(
        self, preds: np.ndarray, data: lgb.Dataset
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        labels = data.get_label()
        weight = data.get_weight()

        n_example = data.num_data()
        labels = labels.reshape(n_example, self.n_class)

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

        self.model = lgb.train(
            params,
            train_dataset,
            valid_sets=[train_dataset, valid_dataset],
            **train_param,
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

        self.model = lgb.train(
            params,
            train_dataset,
            valid_sets=[train_dataset, valid_dataset],
            **train_param,
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
        "feval": custom_loss.loss,
        "callbacks": [
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100),
        ],
    },
}

u_idx = train["Patient_Week"]

categorical_cols = ["Sex", "SmokingStatus"]
drop_cols = ["Patient", "Patient_Week", "FVC"]
features = [c for c in train.columns.tolist() if c not in drop_cols]

X = train[features]

y_fvc = train["FVC"].to_numpy().astype(np.float64)
y_sigma_target = np.full_like(y_fvc, 70.0, dtype=np.float64)
y = np.stack([y_fvc, y_sigma_target], axis=1)

groups = train["Patient"]

n_fold = 5
g_kfold = model_selection.GroupKFold(n_splits=n_fold)

models = []
oof = np.zeros((train.shape[0], 2), dtype=np.float64)
for i, (train_idx, valid_idx) in enumerate(g_kfold.split(X, y_fvc, groups)):
    print("\n" + "#" * 20)
    print("#" * 5, f" {i+1}-Fold")
    print("#" * 20 + "\n")

    print(f"Train Size: {len(train_idx)}")
    print(f"Valid Size: {len(valid_idx)}", "\n")

    X_train, y_train = X.iloc[train_idx, :], y[train_idx]
    X_valid, y_valid = X.iloc[valid_idx, :], y[valid_idx]

    y_train = y_train.astype(np.float64).reshape(-1)
    y_valid = y_valid.astype(np.float64).reshape(-1)

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


## --- ERROR in cell 6, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mLightGBMError[0m                             Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1960846959.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    147[0m [0;34m[0m[0m
[1;32m    148[0m     [0mlgb_model[0m [0;34m=[0m [0mLGBM_Wrapper[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 149[0;31m     lgb_model.fit(
[0m[1;32m    150[0m         [0mparams[0m[0;34m[[0m[0;34m"model_params"[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    151[0m         [0mparams[0m[0;34m[[0m[0;34m"train_params"[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1960846959.py[0m in [0;36mfit[0;34m(self, params, train_param, X_train, y_train, X_valid, y_valid, categorical, train_weight, valid_weight)[0m
[1;32m     48[0m         )
[1;32m     49[0m [0;34m[0m[0m
[0;32m---> 50[0;31m         train_dataset, valid_dataset = self.dataset_to_binary(
[0m[1;32m     51[0m             [0mtrain_dataset[0m[0;34m,[0m [0mvalid_dataset[0m[0;34m[0m[0;34m[0m[0m
[1;32m     52[0m         )

[0;32m/tmp/ipykernel_11/1960846959.py[0m in [0;36mdataset_to_binary[0;34m(self, train_dataset, valid_dataset)[0m
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

## === cell 7
f_imp = np.array([m.model_importance().sort_index().values for m in models])
f_name = models[0].model_importance().sort_index().index

imp_df = pd.DataFrame(f_imp.reshape(-1, len(f_name)).T, index=f_name)
imp_df["AVG_importance"] = imp_df.iloc[:, : len(models)].mean(axis=1)
imp_df["STD_importance"] = imp_df.iloc[:, : len(models)].std(axis=1)
imp_df.sort_values(by="AVG_importance", inplace=True)

imp_df.plot(
    kind="barh", y="AVG_importance", xerr="STD_importance", capsize=4, figsize=(5, 6)
)
plt.tight_layout()
plt.show()
