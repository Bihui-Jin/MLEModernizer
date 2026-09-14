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

-9.90672

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -19.70138) has done: 'I fix the LightGBM training crash caused by an API change: in lightgbm==4.6.0, custom objective/eval must be passed as `objective` and `feval` (not `fobj`). I keep your custom loss/grad-hess logic identical, but wire it into `lgb.train()` in a version-compatible way and set `objective` to the custom callable. Then training complete, enabling inference and creation of `submission.csv` with the exact required columns and row alignment to `sample_submission.csv`. These changes are execution-critical and should also improve score versus a default/fallback because the intended custom metric/objective actually be used.'
- What this solution (achieved -19.70138) has done: 'Your custom objective optimizes a *softplus-transformed* sigma, but the evaluation metric (and your submission) uses sigma directly; this mismatch usually tanks the score, so I keep the model/training loop identical but apply the same softplus transform at inference (and for OOF scoring) before clipping to 70. I also add a tiny post-processing step to keep confidence positive and in a realistic range (only via monotonic transforms/clipping consistent with the metric), which typically improves the LaplaceLL without changing the core model. These are minimal changes focused on aligning training/prediction semantics with the competition metric, which should move your score up toward the target. The script still run end-to-end and write `submission.csv` with the correct columns/row order.'
- What this solution (achieved -15.22004) has done: 'Your score gap to the target is large (current -19.70 vs target -6.87, higher-is-better), so we need a meaningful but still “core-logic-preserving” fix. The biggest issue is that the model is trained on pairwise rows where `pred_Weeks` comes from a true visit week, but at test time you ask it to predict weeks that were never present as `pred_Weeks` in training (most sample_submission weeks), causing severe distribution shift and very poor predictions. To align train/inference semantics without changing the model/loop/loss, I rebuild the training pairs so that `pred_Weeks` spans the same per-patient week grid used by `sample_submission.csv`, while keeping the exact same feature set, LightGBM setup, and custom objective/eval. I also add a minimal “last measurement” filter (use the latest known visit as `current_*`) because that matches the test situation (only baseline info is known), which usually improves LaplaceLL a lot without altering the modeling approach.'
- What this solution (achieved -13.74132) has done: 'Your current gap to the target is large (−15.22 vs −6.87, higher is better), so we need a meaningful but still core-logic-preserving improvement. The biggest remaining mismatch is that training pairs are labeled by *absolute* FVC at `pred_Weeks` while features are anchored at the “current” visit, which makes the model learn absolute levels rather than the change relative to current; switching the label to `FVC_delta = FVC(pred) − current_FVC` preserves the exact model/training loop/custom loss but aligns the supervised target with the conditional prediction problem. At inference, we add back `current_FVC` to the predicted delta to recover absolute FVC for submission, leaving the confidence head and its softplus/clipping unchanged. This is a minimal semantic alignment step that typically improves LaplaceLL substantially without changing architecture, boosting setup, or the custom loss definition.'
- What this solution (achieved -10.41914) has done: 'Your current score (−13.741) is far below the target (−6.868), so we should improve, but with minimal changes and without altering the model/loss. The biggest remaining issue is that training uses the *last* available visit per patient as “current”, while test-time “current” is always the baseline visit (Week in `test.csv`), creating a train/test mismatch; we align training by anchoring each patient’s features at their baseline (Week==0 if present, otherwise the earliest week). Separately, your objective optimizes NLL using `sigma_t = softplus(raw_sigma)`, but the evaluation/feval currently uses raw `preds[:,1]` directly; we apply the same softplus inside the custom loss’ `__call__` so training-time evaluation matches inference and the competition metric semantics. These two alignment fixes typically yield a large score lift without changing architecture, boosting loop, or feature set, and still produce a valid `submission.csv`.'
- What this solution (achieved -10.37998) has done: 'Your score is far below the target (−10.42 vs −6.87, higher is better), so we should improve, but with minimal risk and without changing the model/loss/training loop. The biggest remaining correctness gap is the train/inference “passed_Weeks” sign mismatch: training uses `current_Week - pred_Weeks` while inference uses the same, which makes “future” weeks negative and hurts generalization; we align it to `pred_Weeks - current_Week` consistently (future positive) in both places. Second, the custom objective uses `sigma_t = log(1+exp(sigma))` but the metric/feval used a numerically-stable softplus; we make them identical by using the same stable softplus in the gradient/hessian too (same math, better behaved), which typically improves convergence toward the metric. These are small semantic/alignment fixes that preserve the core LightGBM setup and still produce a valid `submission.csv`.'
- What this solution (achieved -9.90672) has done: 'I make one minimal, score-relevant fix: ensure the custom objective’s gradients/hessians match the *competition metric* by clipping the softplus-transformed sigma at 70 **inside** the objective (currently you clip only in eval/inference, so the model is incentivized to push sigma below 70 which cannot help the metric). This preserves your exact model, features, data construction, and training loop, but aligns optimization with the metric’s sigma clipping and typically improves LaplaceLL toward your target. I also apply the same 70-clip inside the custom eval for perfect consistency (though it already effectively does), and keep everything else unchanged so runtime and semantics stay stable and it still writes a valid `submission.csv`.'

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

INLINE_PLOTTING = False



## === cell 1
SEED = 42
np.random.seed(SEED)

if os.path.exists("/kaggle/input"):
    DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression/"
else:
    DATA_DIR = "../data/raw/"




## === cell 2
def preprocessing(
    data: pd.DataFrame, pred_weeks_by_patient: tp.Optional[dict] = None
) -> pd.DataFrame:
    """
    Change (score-improving, core-logic-preserving):
    - Anchor "current_*" features to the baseline visit (Week==0 if available, else earliest),
      to match test.csv where only baseline info is provided.
    - Keep the same features, same target definition (FVC_delta), same pair construction over pred_Weeks.
    - IMPORTANT alignment fix: define passed_Weeks consistently as (pred_Weeks - current_Week),
      so future predictions are positive like the intuitive timeline (reduces train/test confusion).
    """
    dst_data = []

    for patient, u_data in data.groupby("Patient"):
        u_data = u_data.sort_values("Weeks").reset_index(drop=True)

        if pred_weeks_by_patient is None:
            pred_weeks = u_data["Weeks"].unique()
        else:
            pred_weeks = np.asarray(
                pred_weeks_by_patient.get(patient, u_data["Weeks"].unique()), dtype=int
            )

        if (u_data["Weeks"] == 0).any():
            current_row = u_data.loc[u_data["Weeks"] == 0].iloc[0]
        else:
            current_row = u_data.iloc[0]

        label = pd.DataFrame({"Patient": patient, "pred_Weeks": pred_weeks})
        label = label.merge(
            u_data[["Weeks", "FVC"]].rename(
                columns={"Weeks": "pred_Weeks", "FVC": "FVC"}
            ),
            on="pred_Weeks",
            how="left",
        )

        features = pd.DataFrame(
            {
                "Patient_Week": patient + "_" + label["pred_Weeks"].astype(str),
                "current_FVC": float(current_row["FVC"]),
                "current_Percent": float(current_row["Percent"]),
                "current_Age": float(current_row["Age"]),
                "current_Week": int(current_row["Weeks"]),
                "Patient": patient,
                "Sex": {"Female": 0, "Male": 1}.get(str(current_row["Sex"]), np.nan),
                "SmokingStatus": {
                    "Currently smokes": 0,
                    "Never smoked": 1,
                    "Ex-smoker": 2,
                }.get(str(current_row["SmokingStatus"]), np.nan),
                "pred_Weeks": label["pred_Weeks"].astype(int),
                "FVC": label["FVC"].astype(float),
            }
        )

        features = features.query("pred_Weeks != current_Week").copy()
        features = features[features["FVC"].notna()].copy()

        features["passed_Weeks"] = features["pred_Weeks"] - features["current_Week"]
        dst_data.append(features)

    if len(dst_data) == 0:
        return pd.DataFrame()

    return pd.concat(dst_data, axis=0, ignore_index=True)


train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
sample_sub["Patient"] = sample_sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sample_sub["pred_Weeks"] = (
    sample_sub["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
)
pred_weeks_by_patient = (
    sample_sub.groupby("Patient")["pred_Weeks"]
    .apply(lambda s: np.sort(s.unique()))
    .to_dict()
)

train = preprocessing(train_raw, pred_weeks_by_patient=pred_weeks_by_patient)



## === cell 3
print(train.shape)
train.head()



## === cell 4
"""
Reference: https://www.kaggle.com/ttahara/osic-baseline-lgbm-with-custom-metric?scriptVersionId=38578211
"""


class OSICLossForLGBM:
    """
    Custom Loss for LightGBM.

    * Objective: return grad & hess of NLL of gaussian
    * Evaluation: return competition metric
    """

    def __init__(self, epsilon: float = 1) -> None:
        self.name = "osic_loss"
        self.n_class = 2  # FVC & Confidence
        self.epsilon = epsilon

    def __call__(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> float:
        sigma = preds[:, 1]
        sigma_t = np.log1p(np.exp(-np.abs(sigma))) + np.maximum(sigma, 0)

        sigma_clip = np.maximum(sigma_t, 70)

        Delta = np.minimum(np.abs(preds[:, 0] - labels), 1000)
        loss_by_sample = -np.sqrt(2) * Delta / sigma_clip - np.log(
            np.sqrt(2) * sigma_clip
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
        mu = preds[:, 0]
        sigma = preds[:, 1]

        sigma_t = np.log1p(np.exp(-np.abs(sigma))) + np.maximum(sigma, 0)
        grad_sigma_t = 1.0 / (1.0 + np.exp(-sigma))
        hess_sigma_t = grad_sigma_t * (1.0 - grad_sigma_t)

        mask = (sigma_t >= 70.0).astype(np.float64)
        sigma_clip = np.maximum(sigma_t, 70.0)

        grad = np.zeros_like(preds)
        hess = np.zeros_like(preds)

        grad[:, 0] = -(labels - mu) / sigma_clip**2
        hess[:, 0] = 1.0 / sigma_clip**2

        tmp = ((labels - mu) / sigma_clip) ** 2
        grad[:, 1] = (1.0 / sigma_clip * (1.0 - tmp) * grad_sigma_t) * mask
        hess[:, 1] = (
            -1.0 / sigma_clip**2 * (1.0 - 3.0 * tmp) * grad_sigma_t**2
            + 1.0 / sigma_clip * (1.0 - tmp) * hess_sigma_t
        ) * mask

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

        cb = []
        verbose_eval = train_param.pop("verbose_eval", None)
        early_stopping_rounds = train_param.pop("early_stopping_rounds", None)
        if verbose_eval is not None:
            cb.append(lgb.log_evaluation(period=int(verbose_eval)))
        if early_stopping_rounds is not None:
            cb.append(
                lgb.early_stopping(
                    stopping_rounds=int(early_stopping_rounds), verbose=False
                )
            )

        fobj = train_param.pop("fobj", None)
        feval = train_param.pop("feval", None)
        if fobj is not None:
            params = dict(params)
            params["objective"] = fobj

        self.model = lgb.train(
            params,
            train_dataset,
            valid_sets=[train_dataset, valid_dataset],
            feval=feval,
            callbacks=cb,
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

    def plot_importance(self, filepath=None, max_num_features=50, figsize=(18, 25)):
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
        plt.show()




## === cell 6
def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


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
        "fobj": custom_loss.grad_and_hess,
        "feval": custom_loss.loss,
    },
}

categorical_cols = ["Sex", "SmokingStatus"]
drop_cols = ["Patient", "Patient_Week", "FVC"]
features = [c for c in train.columns.tolist() if c not in drop_cols]

X = train[features]
y = (train["FVC"] - train["current_FVC"]).astype(np.float64)
groups = train["Patient"]

n_fold = 5
g_kfold = model_selection.GroupKFold(n_splits=n_fold)

models = []
oof = np.zeros((train.shape[0], 2), dtype=np.float64)

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
        dict(params["train_params"]),  # per-fold copy (wrapper mutates via pop)
        X_train,
        y_train,
        X_valid,
        y_valid,
        categorical_cols,
    )

    pred_valid = np.asarray(lgb_model.predict(X_valid))
    pred_valid = pred_valid.reshape(len(valid_idx), 2)

    pred_valid[:, 1] = softplus(pred_valid[:, 1])
    pred_valid[:, 1] = np.maximum(pred_valid[:, 1], 70.0)

    oof[valid_idx] = pred_valid
    models.append(lgb_model)



## === cell 7
if len(models) > 0:
    f_imp = np.array([m.model_importance().sort_index().values for m in models])
    f_name = models[0].model_importance().sort_index().index

    imp_df = pd.DataFrame(f_imp.reshape(-1, len(f_name)).T, index=f_name)
    imp_df["AVG_importance"] = imp_df.iloc[:, : len(models)].mean(axis=1)
    imp_df["STD_importance"] = imp_df.iloc[:, : len(models)].std(axis=1)
    imp_df.sort_values(by="AVG_importance", inplace=True)

    plt.figure(figsize=(5, 6))
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
    return float(np.mean(metric))


oof_fvc_pred = oof[:, 0] + train["current_FVC"].to_numpy(dtype=np.float64)
print("OOF metric:", score(train["FVC"].to_numpy(), oof_fvc_pred, oof[:, 1]))



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



## === cell 10
if len(models) == 0:
    raise RuntimeError(
        "No trained models found; cannot run inference/predict submission."
    )

test_idx = test["Patient_Week"].astype(str).to_numpy()

pred_folds = []
for m in models:
    p = np.asarray(m.predict(test[features]))
    p = p.reshape(test.shape[0], 2)
    pred_folds.append(p)

pred = np.mean(np.stack(pred_folds, axis=0), axis=0)

pred[:, 1] = softplus(pred[:, 1])
pred[:, 1] = np.clip(pred[:, 1], 70.0, 1000.0)

pred_fvc_abs = pred[:, 0] + test["current_FVC"].to_numpy(dtype=np.float64)

pred_df = pd.DataFrame(
    {
        "Patient_Week": test_idx,
        "FVC": pred_fvc_abs.astype(np.float64),
        "Confidence": pred[:, 1].astype(np.float64),
    }
)



## === cell 11
submission_template = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

sub_df = submission_template.drop(columns=["FVC", "Confidence"])
sub_df = sub_df.merge(
    pred_df[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week", how="left"
)
sub_df.columns = submission_template.columns

sub_df["FVC"] = pd.to_numeric(sub_df["FVC"], errors="coerce")
sub_df["Confidence"] = pd.to_numeric(sub_df["Confidence"], errors="coerce")
sub_df["FVC"] = sub_df["FVC"].fillna(submission_template["FVC"])
sub_df["Confidence"] = (
    sub_df["Confidence"].fillna(submission_template["Confidence"]).clip(lower=70)
)

sub_df.to_csv("submission.csv", index=False)

print(sub_df.shape)
print(sub_df.head())
print("Wrote submission.csv:", os.path.exists("submission.csv"))
print("Submission columns:", list(sub_df.columns))
