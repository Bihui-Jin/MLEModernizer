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

-7.4531

# 6. Current score

-17.86777

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -17.86777) has done: 'The main runtime failure is from passing a Python method object into LightGBM’s `metric` parameter, which LightGBM 4.6 rejects; the fix is to keep `metric="None"` in params and pass the custom evaluator via `feval=` only. Because training currently aborts, `models` stays empty and inference crashes; fixing the LightGBM call allow folds to train and populate `models`, making prediction and stacking work. I also make the prediction wrapper robust to LightGBM’s output shape and ensure confidence is positive by applying `softplus` at inference (matching the loss’ internal sigma transform) and clipping to 70, which is score-consistent and avoids invalid negative confidences. Finally, the script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -17.86777) has done: 'Your current score is far below the target (gap = -17.86777 − (-7.4531) = -10.41, higher is better), so we should improve performance with minimal, score-relevant changes. The biggest safe gain without changing the model/training loop is to align features between train/test correctly: right now you force `test = test[train.columns.tolist()]`, which includes `FVC` and can scramble/duplicate columns, degrading predictions. I fix test feature construction to use the exact `features` list (and only those), keep the same LightGBM objective/feval and CV setup, and add a small, metric-consistent sigma calibration using OOF residuals to bring Confidence closer to the Laplace-LL optimum (this typically improves score without changing model structure). The script still write a valid `submission.csv` with the required columns.'

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

import seaborn
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

plt.rcParams["figure.dpi"] = 120



## === cell 1
SEED = 42
np.random.seed(SEED)

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
        """Calc loss (competition metric)."""
        sigma_clip = np.maximum(preds[:, 1], 70)
        Delta = np.minimum(np.abs(preds[:, 0] - labels), 1000)
        loss_by_sample = -np.sqrt(2) * Delta / sigma_clip - np.log(
            np.sqrt(2) * sigma_clip
        )
        loss = np.average(loss_by_sample, weights=weight)
        return loss

    def _calc_grad_and_hess(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        """Calc Grad and Hess"""
        mu = preds[:, 0]
        sigma = preds[:, 1]

        sigma_t = np.log1p(np.exp(sigma))
        grad_sigma_t = 1.0 / (1.0 + np.exp(-sigma))
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
        """Return Loss for lightgbm (feval callable)"""
        labels = data.get_label()
        weight = data.get_weight()
        n_example = len(labels)

        preds = preds.reshape(self.n_class, n_example).T
        loss = self(preds, labels, weight)

        return self.name, loss, True

    def grad_and_hess(
        self, preds: np.ndarray, data: lgb.Dataset
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        """Return Grad and Hess for lightgbm (objective callable)"""
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
    def __init__(self, bin_prefix: str = "tmp"):
        self.model = None
        self.importance = None

        self.train_bin_path = f"{bin_prefix}_train_set.bin"
        self.valid_bin_path = f"{bin_prefix}_valid_set.bin"

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
        fobj=None,
        feval=None,
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

        callbacks = []
        if (
            "early_stopping_rounds" in train_param
            and train_param["early_stopping_rounds"] is not None
        ):
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=train_param["early_stopping_rounds"], verbose=False
                )
            )
        if "verbose_eval" in train_param and train_param["verbose_eval"] is not None:
            callbacks.append(
                lgb.log_evaluation(period=int(train_param["verbose_eval"]))
            )

        train_param_clean = dict(train_param)
        train_param_clean.pop("early_stopping_rounds", None)
        train_param_clean.pop("verbose_eval", None)

        params_clean = dict(params)
        if fobj is not None:
            params_clean["objective"] = fobj

        self.model = lgb.train(
            params_clean,
            train_dataset,
            valid_sets=[train_dataset, valid_dataset],
            feval=feval,
            callbacks=callbacks,
            **train_param_clean,
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
        plt.tight_layout()
        plt.close()




## === cell 6
print(train.shape)
train.head()




## === cell 7
def _as_2d_pred(pred_raw: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Robustly convert LightGBM multiclass prediction output into shape (n_samples, 2).

    Depending on LightGBM version/settings, predict() may return:
    - (n_samples, 2)
    - (2 * n_samples,) flattened
    """
    pred_raw = np.asarray(pred_raw)
    if pred_raw.ndim == 2:
        if pred_raw.shape[0] != n_samples:
            raise ValueError(f"Expected {n_samples} rows, got {pred_raw.shape}")
        if pred_raw.shape[1] != 2:
            raise ValueError(f"Expected 2 columns, got {pred_raw.shape}")
        return pred_raw.astype(np.float32)
    if pred_raw.ndim == 1:
        if pred_raw.size != 2 * n_samples:
            raise ValueError(
                f"Expected size {2*n_samples} for flattened pred, got {pred_raw.size}"
            )
        return pred_raw.reshape(2, n_samples).T.astype(np.float32)
    raise ValueError(f"Unexpected pred shape: {pred_raw.shape}")


def _softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)




## === cell 8
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
    },
    "train_params": {
        "num_boost_round": 10000,
        "verbose_eval": 100,
        "early_stopping_rounds": 100,
    },
}

u_idx = train["Patient_Week"]

drop_cols = ["Patient", "Patient_Week", "FVC"]
features = [c for c in train.columns.tolist() if c not in drop_cols]

X = train[features]
y = train["FVC"]
groups = train["Patient"]

num_fold = 5
g_kfold = model_selection.GroupKFold(n_splits=num_fold)

models = []
oof = np.zeros((train.shape[0], 2), dtype=np.float32)

for fold, (train_idx, valid_idx) in enumerate(g_kfold.split(X, y, groups), 1):
    X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx, :], y.iloc[valid_idx]

    lgb_model = LGBM_Wrapper(bin_prefix=f"tmp_fold{fold}")
    lgb_model.fit(
        params["model_params"],
        params["train_params"],
        X_train,
        y_train,
        X_valid,
        y_valid,
        fobj=custom_loss.grad_and_hess,
        feval=custom_loss.loss,
    )
    pred_valid_raw = lgb_model.predict(X_valid)
    pred_valid = _as_2d_pred(pred_valid_raw, n_samples=X_valid.shape[0])

    pred_valid[:, 1] = _softplus(pred_valid[:, 1])

    oof[valid_idx] = pred_valid

    lgb_model.plot_importance(filepath="", max_num_features=80, figsize=(6, 8))
    models.append(lgb_model)




## === cell 9
def score(fvc_true, fvc_pred, sigma):
    sigma_clip = np.maximum(sigma, 70)
    delta = np.minimum(np.abs(fvc_true - fvc_pred), 1000)
    metric = -(np.sqrt(2) * delta / sigma_clip) - np.log(np.sqrt(2) * sigma_clip)
    return np.mean(metric)


oof_score = score(train["FVC"].values, oof[:, 0], oof[:, 1])
oof_score



## === cell 10
abs_err = np.abs(train["FVC"].values.astype(np.float32) - oof[:, 0].astype(np.float32))
sigma_target = np.sqrt(2.0) * abs_err
sigma_pred = oof[:, 1].astype(np.float32)

den = float(np.mean(sigma_pred**2) + 1e-9)
sigma_scale = float(np.mean(sigma_pred * sigma_target) / den)

sigma_scale = float(np.clip(sigma_scale, 0.5, 2.0))
sigma_scale



## === cell 11
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
test



## === cell 12
submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["Weeks"] = (
    submission["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
)

test_base = pd.read_csv(os.path.join(DATA_DIR, "test.csv")).drop(columns=["Weeks"])
test = submission.drop(columns=["FVC", "Confidence"]).merge(
    test_base, how="left", on="Patient"
)
test = preprocessing(test)

missing = [c for c in features if c not in test.columns]
if len(missing) > 0:
    raise ValueError(f"Test is missing expected feature columns: {missing}")

test_X = test[features].copy()

print(test.shape, test_X.shape)
test.head()



## === cell 13
train.head()



## === cell 14
pred_per_model = []
for m in models:
    p_raw = m.predict(test_X)
    p = _as_2d_pred(p_raw, n_samples=test_X.shape[0])  # (n_samples, 2)
    p[:, 1] = _softplus(p[:, 1])  # ensure positive sigma consistent with training
    pred_per_model.append(p)

pred = np.mean(np.stack(pred_per_model, axis=0), axis=0)  # (n_samples, 2)

pred[:, 1] = pred[:, 1] * sigma_scale

pred_df = pd.DataFrame(
    {
        "Patient_Week": test["Patient_Week"].astype(str).values,
        "FVC": pred[:, 0].astype(float),
        "Confidence": np.maximum(pred[:, 1].astype(float), 70.0),
    }
)

pred_df.head()



## === cell 15
submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

sub_df = submission.drop(columns=["FVC", "Confidence"])
sub_df = sub_df.merge(
    pred_df[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week", how="left"
)
sub_df = sub_df[submission.columns]

sub_df["FVC"] = sub_df["FVC"].fillna(submission["FVC"])
sub_df["Confidence"] = sub_df["Confidence"].fillna(submission["Confidence"])

sub_df.to_csv("submission.csv", index=False)

print(sub_df.shape)
sub_df.head()
