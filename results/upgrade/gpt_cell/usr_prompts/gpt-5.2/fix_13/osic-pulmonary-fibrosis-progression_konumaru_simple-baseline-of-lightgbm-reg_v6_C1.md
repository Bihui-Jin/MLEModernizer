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

-13.89398

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -19.70138) has done: 'Diagnosis: The crash happens because LightGBM 4.6.0’s `lgb.train()` no longer accepts the legacy `eval_metric=` keyword; evaluation is now controlled via `feval=` (for custom metrics) and `callbacks`. Your `train_params` dict passes `eval_metric` into `lgb.train(**train_param)`, triggering `TypeError: train() got an unexpected keyword argument 'eval_metric'`.  
Patch summary: In cell 6 only, replace `eval_metric` with the correct `feval` key while keeping the same custom metric function (`custom_loss.loss`) and callbacks, so training/evaluation semantics remain unchanged.  
Updated cells: Cell 6 only.  
Compatibility notes for cell k+1: `models` remains a list of fitted `LGBM_Wrapper` objects; their `.model_importance()` interface is unchanged, so cell 7 work as-is.  
Assumptions: LightGBM’s custom objective (`objective=custom_loss.grad_and_hess`) remains supported; only the metric keyword name changed in this version.'
- What this solution (achieved -19.70138) has done: 'Your current score is far below the target (gap = -19.70138 − (-6.8685) = -12.83; higher is better), so we should improve model generalization with minimal, metric-aligned fixes. The biggest issue is that the training data construction is doing an `outer` merge then filtering, which creates many label/feature mismatches and noise; switching to a `cross` join per patient produces the intended “predict each week from each other week” pairs without NaNs and should substantially improve score while preserving the same modeling approach. I also make Confidence valid/stable for the competition metric by predicting sigma through a positive transform at inference and clipping to 70 (this doesn’t change the model, only post-processing to match evaluation semantics). Finally, I ensure categorical mappings don’t introduce NaNs by filling unknown categories with a safe value.'
- What this solution (achieved -19.99923) has done: 'Diagnosis: Cell 10 fails because `features` (built from the preprocessed train dataframe) includes the column `sample_weight`, but the `test` dataframe created in cell 9 does not have that column. Therefore `test[features]` raises a KeyError for `sample_weight`.  
Patch summary: In cell 10, build a safe feature matrix for prediction by adding any missing feature columns to `test` (specifically `sample_weight`) with a neutral default value (1.0) and then selecting `features` in the original order. This keeps the model input schema identical to training without changing model logic.  
Updated cells: Only cell 10 is modified.  
Compatibility notes for cell k+1: `pred_df` is still created with the same columns (`Patient_Week`, `FVC`, `Confidence`) and compatible dtypes/shapes, so cell 11 works unchanged.  
Assumptions: Missing `sample_weight` at inference time should default to 1.0 (neutral) since weights are only used during training/evaluation weighting, not as a real-world feature.'
- What this solution (achieved -16.28812) has done: 'Your gap to the target is large (current -19.999 vs target -6.8685; higher is better), so we should improve generalization without changing the model/loop. The biggest safe win is to align training features with test-time reality: your training currently uses every visit as a “current” anchor, but test has only the baseline (week 0) row, creating a train/test mismatch. I keep the same cross-join idea and LightGBM setup, but restrict training anchors to each patient’s baseline visit (Week==0 if available, else earliest), which typically boosts OSIC performance substantially. I also ensure the “passed_Weeks” sign matches the natural direction (pred_week - current_week) consistently across train/test; this is a minimal feature-correction that can materially affect predictions.'
- What this solution (achieved -11.11801) has done: 'Your current score is much worse than the target (gap ≈ -9.42), so we should improve it with minimal, metric-aligned tweaks while keeping the same LightGBM + cross-join + custom loss pipeline. The safest win is to make the model’s predicted “Confidence” better calibrated for the Laplace metric without changing the training approach: we add a single training feature `abs_passed_Weeks` (available at test time too) to help the model learn uncertainty grows with time distance, and we set the final submitted confidence to a blend of the model’s sigma and a simple time-based floor. This keeps your core logic intact (same model, same folds, same objective/feval), but typically improves score materially by avoiding overconfident sigmas at far weeks. We also make sure the new feature exists in both train/test and keep all paths and submission schema unchanged.'
- What this solution (achieved -12.69889) has done: 'We’re still far below the target (current -11.118 vs target -6.8685; higher is better), so the smallest likely win is to improve the metric by better calibrating `Confidence` without changing the model, folds, objective, or training loop. I keep your learned sigma but replace the current aggressive time-based floor (`70 + 2*abs_weeks`) with a gentler, more metric-friendly floor plus a light blend, reducing unnecessary penalty from overly large sigmas while still avoiding overconfidence for far weeks. I apply the same calibration consistently to both OOF scoring (for sanity checking) and test-time submission. Everything else (features, LightGBM params, custom loss, data paths, submission schema) stays the same.'
- What this solution (achieved -13.89398) has done: 'You’re currently below the target (higher is better), so we should improve score by making the smallest metric-aligned change: calibrate `Confidence` to better trade off error vs. over-penalization from too-large sigmas. We keep your model, folds, features, and loss exactly the same, but replace the current confidence blending with a gentler time-based floor and a light cap so sigma doesn’t grow unnecessarily for far weeks (which hurts the Laplace score via `-log(sigma)`). I apply the exact same calibration for OOF scoring and for test-time submission to keep evaluation semantics consistent. Everything else (data paths, schema, training loop) remains unchanged.'

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
    Minimal, score-relevant alignment fix (kept):
    - Anchor "current_*" at baseline (Week==0 else earliest) to match test semantics.
    - passed_Weeks is (pred_Weeks - current_Week) consistently.
    - sample_weight unchanged.

    Minimal metric-aligned improvement (kept):
    - Add abs_passed_Weeks as a feature so the model can learn time-distance effects.
    """
    dst_parts = []

    sex_map = {"Female": 0, "Male": 1}
    smoke_map = {"Currently smokes": 0, "Never smoked": 1, "Ex-smoker": 2}

    for patient, u_data in data.groupby("Patient"):
        u_data = u_data.copy()

        if (u_data["Weeks"] == 0).any():
            anchor = u_data.loc[u_data["Weeks"] == 0].iloc[[0]].copy()
        else:
            anchor = u_data.sort_values("Weeks").iloc[[0]].copy()

        label = pd.DataFrame(
            {
                "Patient": patient,
                "pred_Weeks": u_data["Weeks"].values,
                "FVC": u_data["FVC"].values,
            }
        )

        features = pd.DataFrame(
            {
                "current_FVC": np.repeat(anchor["FVC"].values[0], len(u_data)),
                "current_Percent": np.repeat(anchor["Percent"].values[0], len(u_data)),
                "current_Age": np.repeat(anchor["Age"].values[0], len(u_data)),
                "current_Week": np.repeat(anchor["Weeks"].values[0], len(u_data)),
                "Patient": np.repeat(patient, len(u_data)),
                "Sex": np.repeat(
                    anchor["Sex"].map(sex_map).fillna(-1).astype(int).values[0],
                    len(u_data),
                ),
                "SmokingStatus": np.repeat(
                    anchor["SmokingStatus"]
                    .map(smoke_map)
                    .fillna(-1)
                    .astype(int)
                    .values[0],
                    len(u_data),
                ),
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
            dst_u_data["pred_Weeks"] - dst_u_data["current_Week"]
        )

        dst_u_data["abs_passed_Weeks"] = np.abs(dst_u_data["passed_Weeks"]).astype(
            np.float32
        )

        abs_pw = dst_u_data["abs_passed_Weeks"].values.astype(np.float32)
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


def _osic_feval_adapter(preds: np.ndarray, data: lgb.Dataset):
    labels = data.get_label()
    weight = data.get_weight()
    n_example = len(labels)

    preds2 = np.asarray(preds).reshape(custom_loss.n_class, n_example).T  # (n, 2)
    sigma_clip = np.maximum(preds2[:, 1], 70)
    Delta = np.minimum(np.abs(preds2[:, 0] - labels), 1000)
    loss_by_sample = -np.sqrt(2) * Delta / sigma_clip - np.log(np.sqrt(2) * sigma_clip)
    loss = np.average(loss_by_sample, weights=weight)

    return custom_loss.name, loss, True


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
        "feval": _osic_feval_adapter,
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


oof_mu = oof[:, 0]
oof_sigma_model = np.log1p(np.exp(np.clip(oof[:, 1], -20.0, 20.0))).astype(np.float32)

oof_abs_pw = train["abs_passed_Weeks"].values.astype(np.float32)
sigma_floor = (70.0 + 0.55 * oof_abs_pw).astype(np.float32)  # gentler slope than 1.0
oof_sigma = np.maximum(oof_sigma_model, sigma_floor)
oof_sigma = (0.92 * oof_sigma + 0.08 * sigma_floor).astype(np.float32)  # lighter blend
oof_sigma = np.minimum(oof_sigma, 300.0).astype(
    np.float32
)  # light cap to avoid log-penalty blowup
oof_sigma = np.maximum(oof_sigma, 70.0)

score(train["FVC"].values, oof_mu, oof_sigma)



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

test["Sex"] = test["Sex"].map({"Female": 0, "Male": 1}).fillna(-1).astype(int)
test["SmokingStatus"] = (
    test["SmokingStatus"]
    .map({"Currently smokes": 0, "Never smoked": 1, "Ex-smoker": 2})
    .fillna(-1)
    .astype(int)
)

test = pd.merge(submission, test, how="left", on=["Patient"])

test["passed_Weeks"] = test["pred_Weeks"] - test["current_Week"]
test["abs_passed_Weeks"] = np.abs(test["passed_Weeks"]).astype(np.float32)



## === cell 10
test_idx = test["Patient_Week"].to_numpy().reshape(-1, 1)

missing_cols = [c for c in features if c not in test.columns]
for c in missing_cols:
    test[c] = 1.0

X_test = test[features]

pred = np.array([m.predict(X_test) for m in models]).mean(axis=0)

pred_mu = pred[:, 0]
pred_sigma_model = np.log1p(np.exp(np.clip(pred[:, 1], -20.0, 20.0))).astype(np.float32)

abs_pw_test = test["abs_passed_Weeks"].values.astype(np.float32)
sigma_floor = (70.0 + 0.55 * abs_pw_test).astype(np.float32)
pred_sigma = np.maximum(pred_sigma_model, sigma_floor)
pred_sigma = (0.92 * pred_sigma + 0.08 * sigma_floor).astype(np.float32)
pred_sigma = np.minimum(pred_sigma, 300.0).astype(np.float32)
pred_sigma = np.maximum(pred_sigma, 70.0)

pred_df = pd.DataFrame(
    np.concatenate(
        (test_idx, pred_mu.reshape(-1, 1), pred_sigma.reshape(-1, 1)), axis=1
    ),
    columns=["Patient_Week", "FVC", "Confidence"],
)



## === cell 11
submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

sub_df = submission.drop(columns=["FVC", "Confidence"])
sub_df = sub_df.merge(pred_df[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week")
sub_df.columns = submission.columns

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)

print(sub_df.shape)
print(f"Wrote: {sub_path}")
sub_df.head()
