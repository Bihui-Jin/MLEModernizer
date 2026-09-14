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

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass



## === cell 1
SEED = 42

if os.path.exists("/kaggle/input"):
    DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression/"
else:
    DATA_DIR = "../data/osic-pulmonary-fibrosis-progression/"

np.random.seed(SEED)




## === cell 2
def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the same core idea (per-patient cross join) but add a metric-relevant weight:
    - OSIC test asks for final weeks; training on all week gaps uniformly overweights very large gaps.
    - We add 'sample_weight' that smoothly downweights large |passed_Weeks|, improving generalization
      without changing model/architecture/training loop.
    """
    dst_parts = []

    sex_map = {"Female": 0, "Male": 1}
    smoke_map = {"Currently smokes": 0, "Never smoked": 1, "Ex-smoker": 2}

    for patient, u_data in data.groupby("Patient"):
        u_data = u_data.copy()

        label = pd.DataFrame(
            {
                "Patient": patient,
                "pred_Weeks": u_data["Weeks"].values,
                "FVC": u_data["FVC"].values,
            }
        )

        features = pd.DataFrame(
            {
                "current_FVC": u_data["FVC"].values,
                "current_Percent": u_data["Percent"].values,
                "current_Age": u_data["Age"].values,
                "current_Week": u_data["Weeks"].values,
                "Patient": u_data["Patient"].values,
                "Sex": u_data["Sex"].map(sex_map).fillna(-1).astype(int).values,
                "SmokingStatus": u_data["SmokingStatus"]
                .map(smoke_map)
                .fillna(-1)
                .astype(int)
                .values,
            }
        )

        label["_tmp"] = 1
        features["_tmp"] = 1
        dst_u_data = label.merge(features, on=["Patient", "_tmp"], how="inner").drop(
            columns=["_tmp"]
        )

        dst_u_data["Patient_Week"] = (
            dst_u_data["Patient"].astype(str)
            + "_"
            + dst_u_data["pred_Weeks"].astype(str)
        )

        dst_u_data = dst_u_data.query("pred_Weeks != current_Week").copy()

        dst_u_data["passed_Weeks"] = (
            dst_u_data["current_Week"] - dst_u_data["pred_Weeks"]
        )

        abs_pw = np.abs(dst_u_data["passed_Weeks"].values).astype(np.float32)
        dst_u_data["sample_weight"] = 1.0 / (1.0 + (abs_pw / 26.0) ** 2)

        dst_parts.append(dst_u_data)

    dst_data = pd.concat(dst_parts, axis=0, ignore_index=True)
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
        """Calc loss (competition metric)."""
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

        sigma = np.clip(sigma, -20.0, 20.0)

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
y = train["FVC"]
groups = train["Patient"]

w = train["sample_weight"].astype(np.float32).values

n_fold = 5
g_kfold = model_selection.GroupKFold(n_splits=n_fold)

models = []
oof = np.zeros((train.shape[0], 2))
for i, (train_idx, valid_idx) in enumerate(g_kfold.split(X, y, groups)):
    print("\n" + "#" * 20)
    print("#" * 5, f" {i+1} Fold")
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
        train_weight=w[train_idx],
        valid_weight=w[valid_idx],
    )
    oof[valid_idx] = lgb_model.predict(X_valid)
    models.append(lgb_model)



## --- ERROR in cell 6, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/177480097.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     55[0m [0;34m[0m[0m
[1;32m     56[0m     [0mlgb_model[0m [0;34m=[0m [0mLGBM_Wrapper[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 57[0;31m     lgb_model.fit(
[0m[1;32m     58[0m         [0mparams[0m[0;34m[[0m[0;34m"model_params"[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     59[0m         [0mparams[0m[0;34m[[0m[0;34m"train_params"[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2866211468.py[0m in [0;36mfit[0;34m(self, params, train_param, X_train, y_train, X_valid, y_valid, categorical, train_weight, valid_weight)[0m
[1;32m     52[0m         )
[1;32m     53[0m [0;34m[0m[0m
[0;32m---> 54[0;31m         self.model = lgb.train(
[0m[1;32m     55[0m             [0mparams[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     56[0m             [0mtrain_dataset[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/engine.py[0m in [0;36mtrain[0;34m(params, train_set, num_boost_round, valid_sets, valid_names, feval, init_model, keep_training_booster, callbacks)[0m
[1;32m    326[0m         [0;32mif[0m [0mvalid_sets[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    327[0m             [0;32mif[0m [0mis_valid_contain_train[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 328[0;31m                 [0mevaluation_result_list[0m[0;34m.[0m[0mextend[0m[0;34m([0m[0mbooster[0m[0;34m.[0m[0meval_train[0m[0;34m([0m[0mfeval[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    329[0m             [0mevaluation_result_list[0m[0;34m.[0m[0mextend[0m[0;34m([0m[0mbooster[0m[0;34m.[0m[0meval_valid[0m[0;34m([0m[0mfeval[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    330[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36meval_train[0;34m(self, feval)[0m
[1;32m   4406[0m             [0mList[0m [0;32mwith[0m [0;34m([0m[0mtrain_dataset_name[0m[0;34m,[0m [0meval_name[0m[0;34m,[0m [0meval_result[0m[0;34m,[0m [0mis_higher_better[0m[0;34m)[0m [0mtuples[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4407[0m         """
[0;32m-> 4408[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m__inner_eval[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_train_data_name[0m[0;34m,[0m [0;36m0[0m[0;34m,[0m [0mfeval[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   4409[0m [0;34m[0m[0m
[1;32m   4410[0m     def eval_valid(

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36m__inner_eval[0;34m(self, data_name, data_idx, feval)[0m
[1;32m   5204[0m                 [0;32mif[0m [0meval_function[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   5205[0m                     [0;32mcontinue[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 5206[0;31m                 [0mfeval_ret[0m [0;34m=[0m [0meval_function[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m__inner_predict[0m[0;34m([0m[0mdata_idx[0m[0;34m)[0m[0;34m,[0m [0mcur_data[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   5207[0m                 [0;32mif[0m [0misinstance[0m[0;34m([0m[0mfeval_ret[0m[0;34m,[0m [0mlist[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   5208[0m                     [0;32mfor[0m [0meval_name[0m[0;34m,[0m [0mval[0m[0;34m,[0m [0mis_higher_better[0m [0;32min[0m [0mfeval_ret[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1035522307.py[0m in [0;36mloss[0;34m(self, preds, data)[0m
[1;32m     70[0m [0;34m[0m[0m
[1;32m     71[0m         [0mpreds[0m [0;34m=[0m [0mpreds[0m[0;34m.[0m[0mreshape[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mn_class[0m[0;34m,[0m [0mn_example[0m[0;34m)[0m[0;34m.[0m[0mT[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 72[0;31m         [0mloss[0m [0;34m=[0m [0mself[0m[0;34m([0m[0mpreds[0m[0;34m,[0m [0mlabels[0m[0;34m,[0m [0mweight[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     73[0m [0;34m[0m[0m
[1;32m     74[0m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mname[0m[0;34m,[0m [0mloss[0m[0;34m,[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1035522307.py[0m in [0;36m__call__[0;34m(self, preds, labels, weight)[0m
[1;32m     25[0m             [0mnp[0m[0;34m.[0m[0msqrt[0m[0;34m([0m[0;36m2[0m[0;34m)[0m [0;34m*[0m [0msigma_clip[0m[0;34m[0m[0;34m[0m[0m
[1;32m     26[0m         )
[0;32m---> 27[0;31m         [0mloss[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0maverage[0m[0;34m([0m[0mloss_by_sample[0m[0;34m,[0m [0mweight[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     28[0m [0;34m[0m[0m
[1;32m     29[0m         [0;32mreturn[0m [0mloss[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py[0m in [0;36maverage[0;34m(a, axis, weights, returned, keepdims)[0m
[1;32m    518[0m [0;34m[0m[0m
[1;32m    519[0m     [0;32mif[0m [0mweights[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 520[0;31m         [0mavg[0m [0;34m=[0m [0ma[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0maxis[0m[0;34m,[0m [0;34m**[0m[0mkeepdims_kw[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    521[0m         [0mavg_as_array[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0masanyarray[0m[0;34m([0m[0mavg[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    522[0m         [0mscl[0m [0;34m=[0m [0mavg_as_array[0m[0;34m.[0m[0mdtype[0m[0;34m.[0m[0mtype[0m[0;34m([0m[0ma[0m[0;34m.[0m[0msize[0m[0;34m/[0m[0mavg_as_array[0m[0;34m.[0m[0msize[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/numpy/core/_methods.py[0m in [0;36m_mean[0;34m(a, axis, dtype, out, keepdims, where)[0m
[1;32m    104[0m     [0mis_float16_result[0m [0;34m=[0m [0;32mFalse[0m[0;34m[0m[0;34m[0m[0m
[1;32m    105[0m [0;34m[0m[0m
[0;32m--> 106[0;31m     [0mrcount[0m [0;34m=[0m [0m_count_reduce_items[0m[0;34m([0m[0marr[0m[0;34m,[0m [0maxis[0m[0;34m,[0m [0mkeepdims[0m[0;34m=[0m[0mkeepdims[0m[0;34m,[0m [0mwhere[0m[0;34m=[0m[0mwhere[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    107[0m     [0;32mif[0m [0mrcount[0m [0;34m==[0m [0;36m0[0m [0;32mif[0m [0mwhere[0m [0;32mis[0m [0;32mTrue[0m [0;32melse[0m [0mumr_any[0m[0;34m([0m[0mrcount[0m [0;34m==[0m [0;36m0[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    108[0m         [0mwarnings[0m[0;34m.[0m[0mwarn[0m[0;34m([0m[0;34m"Mean of empty slice."[0m[0;34m,[0m [0mRuntimeWarning[0m[0;34m,[0m [0mstacklevel[0m[0;34m=[0m[0;36m2[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/numpy/core/_methods.py[0m in [0;36m_count_reduce_items[0;34m(arr, axis, keepdims, where)[0m
[1;32m     75[0m         [0mitems[0m [0;34m=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m     76[0m         [0;32mfor[0m [0max[0m [0;32min[0m [0maxis[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 77[0;31m             [0mitems[0m [0;34m*=[0m [0marr[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0mmu[0m[0;34m.[0m[0mnormalize_axis_index[0m[0;34m([0m[0max[0m[0;34m,[0m [0marr[0m[0;34m.[0m[0mndim[0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     78[0m         [0mitems[0m [0;34m=[0m [0mnt[0m[0;34m.[0m[0mintp[0m[0;34m([0m[0mitems[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     79[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: only integer scalar arrays can be converted to a scalar index

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
