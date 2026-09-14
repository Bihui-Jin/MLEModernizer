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

-24.65251

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -19.70138) has done: 'The crash happens because `lgb.train()` does not accept an `objective=` keyword argument; instead, the objective must be passed either inside the `params` dict (as `"objective": callable`) or via the dedicated `fobj=` argument. In cell 6, the code incorrectly put the custom objective in `train_params` under `"objective"`, which ends up being forwarded as an unexpected keyword. The minimal fix is to move the custom objective callable into `model_params` as `"objective": custom_loss.grad_and_hess` and remove `"objective"` from `train_params`, keeping evaluation (`feval`) and callbacks unchanged. This preserves the same training semantics while making it compatible with LightGBM 4.6.0.'
- What this solution (achieved -19.70138) has done: 'Your score is far below the target, so we should improve it with minimal, low-risk fixes that keep the same LightGBM + custom objective approach. The biggest issue is that the model outputs “Confidence” without any constraint and the metric clips at 70, so negative/too-small sigmas can severely hurt the log-likelihood; we post-process predictions to enforce a strictly positive sigma and a reasonable floor (≥70) while leaving the training logic unchanged. We also make the train/test feature joins deterministic by resetting indexes after concatenations/merges, and ensure the submission always writes `submission.csv` (also outside Kaggle) with correct column types. These changes are directly aligned with the competition metric and typically improve scores substantially without changing the core model.'
- What this solution (achieved -24.65932) has done: 'To move your score up toward the target with minimal risk, I (1) align the model’s training objective with the competition metric by clipping the *label* error at 1000 and clipping sigma at 70 inside the custom objective (your current objective optimizes a different Gaussian NLL and ignores the metric’s clipping), and (2) make the submission-time Confidence post-processing strictly positive via a softplus transform before applying the 70 floor (preventing pathological negative/near-zero sigmas that can tank the log-likelihood). I not change the model family (LightGBM), the 2-output setup, the CV approach, or the overall pipeline; these are small, metric-consistent fixes. The script still run end-to-end and write `submission.csv` with the required columns.'
- What this solution (achieved -24.65251) has done: 'Your current score is far below the target, so we should improve it with very small, metric-aligned adjustments that don’t change your model family or training loop. The biggest issue is that your custom objective currently uses a non-smooth, piecewise gradient (and zero Hessian for `mu`), which makes LightGBM optimization unstable and often underfits; we replace the hard clip at `Δ=1000` with a smooth approximation (Huberized absolute error) while keeping the same clipping semantics in the *metric* (`feval`) and the same 2-output structure. We also make LightGBM treat this as a proper custom objective by ensuring the gradient/Hessian are well-behaved and strictly positive where needed, and we keep your submission-time sigma post-processing (softplus + floor at 70) intact. These changes are directly aimed at moving the score upward toward the target without changing the overall pipeline.'

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

    Keep evaluation EXACTLY as competition metric (hard clip on sigma & delta).

    Change (minimal, score-improving): use a smooth approximation of the clipped absolute
    error in the *objective* to provide stable gradients/Hessians for LightGBM.
    This keeps the same core logic (2 outputs: mu & sigma) and aligns with the metric,
    but avoids zero-Hessian / discontinuity around |r|=1000 which can underfit badly.
    """

    def __init__(self, epsilon: float = 1.0, huber_delta: float = 1000.0) -> None:
        self.name = "osic_loss"
        self.n_class = 2  # FVC & Confidence
        self.epsilon = float(epsilon)
        self.huber_delta = float(huber_delta)

    def __call__(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> float:
        sigma_clip = np.maximum(preds[:, 1], 70)
        Delta = np.minimum(np.abs(preds[:, 0] - labels), 1000)
        loss_by_sample = -np.sqrt(2) * Delta / sigma_clip - np.log(
            np.sqrt(2) * sigma_clip
        )
        loss = np.average(loss_by_sample, weight)
        return loss

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

    def _huber_abs(self, r: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Smooth abs with Huber:
          a = |r| if |r|<=d else d + (|r|-d)*0  (for abs itself we'd still be linear),
        but we need a smooth surrogate to the clipped |r|. Use:
          a = { 0.5*r^2/d            if |r|<=d
              { |r| - 0.5*d          else
        Then we cap a at d by using "soft cap" via min(a, d) is not smooth.
        Instead, we keep Huber itself as the surrogate for |r|, which matches |r|
        for large residuals but has smooth derivatives near 0. Since the metric caps at 1000,
        setting d=1000 makes the scale match and improves optimization stability.
        Returns: a, da/dr, d2a/dr2
        """
        d = self.huber_delta
        abs_r = np.abs(r)
        sign_r = np.sign(r)

        quad = abs_r <= d
        a = np.where(quad, 0.5 * (r**2) / d, abs_r - 0.5 * d)
        da_dr = np.where(quad, r / d, sign_r)
        d2a_dr2 = np.where(quad, 1.0 / d, 0.0)
        return a, da_dr, d2a_dr2

    def _calc_grad_and_hess(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        mu = preds[:, 0]
        raw_sigma = preds[:, 1]

        sigma_pos = self._softplus(raw_sigma)
        sigm = self._sigmoid(raw_sigma)
        d_sigma_pos_d_raw = sigm
        d2_sigma_pos_d_raw2 = sigm * (1.0 - sigm)

        sigma_clip = np.maximum(sigma_pos, 70.0)
        active = (sigma_pos >= 70.0).astype(np.float64)
        d_sigma_clip_d_raw = d_sigma_pos_d_raw * active
        d2_sigma_clip_d_raw2 = d2_sigma_pos_d_raw2 * active

        r = mu - labels

        Delta_s, dDelta_dr, d2Delta_dr2 = self._huber_abs(r)

        sqrt2 = np.sqrt(2.0)

        grad = np.zeros_like(preds, dtype=np.float64)
        hess = np.zeros_like(preds, dtype=np.float64)

        grad[:, 0] = sqrt2 * dDelta_dr / sigma_clip
        hess[:, 0] = sqrt2 * d2Delta_dr2 / sigma_clip + self.epsilon

        dL_d_sigma = -sqrt2 * Delta_s / (sigma_clip**2) + 1.0 / sigma_clip
        d2L_d_sigma2 = 2.0 * sqrt2 * Delta_s / (sigma_clip**3) - 1.0 / (sigma_clip**2)

        grad[:, 1] = dL_d_sigma * d_sigma_clip_d_raw
        hess[:, 1] = (
            d2L_d_sigma2 * (d_sigma_clip_d_raw**2)
            + dL_d_sigma * d2_sigma_clip_d_raw2
            + self.epsilon
        )

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
        return self.name, loss, True

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

n_fold = 5
g_kfold = model_selection.GroupKFold(n_splits=n_fold)

models = []
oof = np.zeros((train.shape[0], 2), dtype=np.float64)
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




## === cell 8
def score(fvc_true, fvc_pred, sigma):
    sigma_clip = np.maximum(sigma, 70)
    delta = np.minimum(np.abs(fvc_true - fvc_pred), 1000)
    metric = -(np.sqrt(2) * delta / sigma_clip) - np.log(np.sqrt(2) * sigma_clip)
    return np.mean(metric)


oof_pp = oof.copy()
oof_pp[:, 1] = np.log1p(np.exp(oof_pp[:, 1]))
oof_pp[:, 1] = np.maximum(oof_pp[:, 1], 70.0)

score(train["FVC"].to_numpy(), oof_pp[:, 0], oof_pp[:, 1])



## === cell 9
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
test = preprocessing(test, is_test=True)



## === cell 10
print(test.shape)
test.head()



## === cell 11
test_idx = test["Patient_Week"].to_numpy().reshape(-1, 1)
pred = np.mean([m.predict(test[features]) for m in models], axis=0)

pred = pred.astype(np.float64)

pred[:, 1] = np.log1p(np.exp(pred[:, 1]))
pred[:, 1] = np.maximum(pred[:, 1], 70.0)

pred_df = pd.DataFrame(
    np.concatenate((test_idx, pred), axis=1),
    columns=["Patient_Week", "FVC", "Confidence"],
)
pred_df["FVC"] = pred_df["FVC"].astype(float)
pred_df["Confidence"] = pred_df["Confidence"].astype(float)



## === cell 12
submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

sub_df = submission.drop(columns=["FVC", "Confidence"])
sub_df = sub_df.merge(
    pred_df[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week", how="left"
)
sub_df.columns = submission.columns

sub_df.to_csv("submission.csv", index=False)

print(sub_df.shape)
sub_df.head()
