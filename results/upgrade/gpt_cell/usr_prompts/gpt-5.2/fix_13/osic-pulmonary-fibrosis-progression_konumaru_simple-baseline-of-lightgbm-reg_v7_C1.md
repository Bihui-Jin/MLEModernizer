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

-8.60121

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -19.70138) has done: 'Diagnosis: LightGBM 4.6.0 removed the `fobj` keyword from `lgb.train()`, so passing `fobj` raises `TypeError: train() got an unexpected keyword argument 'fobj'`. The custom objective is still supported, but it must be passed via the `objective` argument (callable) instead. The current `train_params` dict includes `fobj`, and `LGBM_Wrapper.fit()` forwards it directly to `lgb.train()`, causing the crash.  
Patch summary: In cell 6 only, move the custom objective from `train_params["fobj"]` into `model_params["objective"]` and delete `fobj` from `train_params`, keeping the same callable (`custom_loss.grad_and_hess`) and leaving the evaluation callback (`feval`) unchanged. This preserves training semantics while matching the LightGBM 4.x API.  
Updated cells: only cell 6 is updated below.  
Compatibility notes for cell k+1: `models` remains a list of trained `LGBM_Wrapper` objects, and `model_importance()`/`feature_name()` behavior is unchanged, so cell 7 work as-is.  
Assumptions: LightGBM callable objectives are accepted via the `objective` parameter in this environment (standard for LightGBM 4.x), and no other cells rely on `params["train_params"]["fobj"]` existing.'
- What this solution (achieved -8.80859) has done: 'We make two minimal, score-relevant fixes: (1) enforce valid predictions for the metric by ensuring `Confidence` is positive and clipped to at least 70 at submission time (your current model can output negative/too-small confidence, which is heavily penalized), and (2) remove early stopping so training isn’t cut short (your stated constraints disallow it, and it can also depress performance vs your target). These keep the same model/feature/core training logic while improving metric alignment and stability. The rest of the pipeline (preprocessing, GroupKFold, LightGBM callable objective/feval, submission alignment) remains unchanged.'
- What this solution (achieved -inf) has done: 'To move your score up toward the target with minimal disruption, I keep the exact same LightGBM custom objective/feval and training loop, but fix one score-relevant issue: the confidence head is trained on *raw* sigma while the metric uses a clipped/positive sigma; aligning these by converting the model’s sigma output through the same softplus used in the loss (and then clipping at 70) improves metric consistency without changing the core approach. I apply the same transformation to OOF scoring (so it reflects true behavior) and to submission generation (so Confidence is always positive and consistent with training). I also make prediction use `num_iteration=None` (best available) to avoid any edge case around `best_iteration` when no early stopping is used. These are small, metric-aligned changes expected to increase performance from -8.81 toward -6.87.'
- What this solution (achieved -8.80859) has done: 'Your `-inf` score strongly suggests invalid values in the submitted `Confidence` (NaN/inf/zero/negative), which breaks the Laplace log-likelihood. I keep your model/training exactly as-is, and make minimal, score-relevant changes only in prediction/post-processing: compute `Confidence` with a numerically-stable softplus (prevents overflow to `inf`), then clip to at least 70 as required by the metric. I also harden the final submission by coercing non-finite predictions (FVC/Confidence) to safe finite values so Kaggle scoring never becomes `-inf`. These changes are purely metric-alignment/validity fixes and should move you upward toward the target.'
- What this solution (achieved -8.67129) has done: 'Your current score is below the target (gap = -8.80859 − (-6.8685) ≈ -1.94), so we should cautiously improve it without changing the core LightGBM/custom-loss setup. The smallest, metric-aligned improvement is to calibrate `Confidence` per patient using training residuals: the competition metric strongly depends on a reasonable sigma, and a single fixed post-processing (softplus + clip 70) often miscalibrates uncertainty. We keep the model outputs as-is, but learn a simple per-patient multiplicative scale for sigma (from OOF residuals on training), then apply it to that patient’s test predictions; this typically improves the Laplace log-likelihood without altering model architecture/training. We also compute OOF score using the same calibration to ensure consistency and avoid over/under-confidence penalties.'
- What this solution (achieved -8.67129) has done: 'Your current score (-8.67129) is worse than the target (-6.8685), so we should make a small, metric-aligned improvement without changing the model/training core. The biggest remaining mismatch is that the custom *evaluation* uses raw `preds[:,1]` (often <70 / negative) while your submission uses `softplus` + per-patient calibration + clip; this can lead the training process to select suboptimal iterations because it “thinks” the model is worse/better than it truly is under the real metric. I minimally fix `OSICLossForLGBM.__call__` to compute the metric using the same stable softplus transform as in inference (then clip at 70), keeping the same objective/grad-hess and training loop. I also make the OOF `patient_scale` estimation consistent by using the same clipped sigma as the metric (prevents extreme ratios from tiny sigma), which should gently improve confidence calibration toward the target.'
- What this solution (achieved -8.60802) has done: 'Your score is below the target (gap ≈ -1.80), so we should make a small, metric-aligned improvement without changing the LightGBM/custom-loss core. The main remaining lever is uncertainty calibration: your current per-patient sigma scaling uses only OOF residuals and can be noisy for patients with few observations, which can under/over-shoot the Laplace metric. I keep the same model and training, but (1) fit a tiny global affine calibration for sigma on OOF (scale + shift) to better match residual magnitudes under the competition clipping, and (2) shrink per-patient scales toward the global scale based on each patient’s OOF sample count for stability. This only changes post-processing of Confidence (and its OOF estimate), which is directly score-relevant for this metric and should move the score upward toward the target.'
- What this solution (achieved -8.67129) has done: 'I make a minimal, metric-aligned improvement to your uncertainty calibration only (no change to model architecture, loss, features, or training loop) because your gap to target is still sizable and the OSIC metric is very sensitive to Confidence. Specifically, I replace the least-squares affine fit (which optimizes squared error on |residual| and can miscalibrate Laplace NLL) with a robust, direct Laplace-optimal mapping: set the global affine to `sigma_cal = sigma_clip * median(|residual|/sigma_clip)` (i.e., only a single non-negative scale). I keep your per-patient scaling and shrinkage, but compute it from the *same calibrated/clipped sigma used by the metric* to reduce noise and prevent overconfident patients. This should move your score upward toward the target while preserving the core logic and keeping runtime within limits.'
- What this solution (achieved -8.60522) has done: 'Your score (-8.67129) is worse than the target (-6.8685), so we should make a small, metric-aligned improvement without changing the LightGBM model/loss or training loop. The OSIC metric is very sensitive to Confidence calibration, and your current per-patient sigma scaling is learned from OOF but then applied “as-is” to test patients whose residual distribution can differ; a tiny, safe improvement is to learn a **global bias term** for sigma (still keeping the same global scale + patient shrink) so the model is slightly less overconfident where needed. Concretely, we keep your global scale `a_hat` (median |resid|/sigma) but also fit a **small non-negative additive** `b_hat` from OOF to better match the clipped Laplace NLL, then reuse the same mapping at inference. This changes only Confidence post-processing (directly score-relevant), keeps runtime similar, and preserves all core modeling logic.'
- What this solution (achieved -8.60121) has done: 'Your score is below the target (gap ≈ -1.74), so we should make a small, metric-aligned improvement without changing the LightGBM model, features, or training loop. The OSIC metric is very sensitive to Confidence calibration; your current global sigma calibration uses a coarse grid over `b_hat`, which can easily miss a better bias value and leave you over/under-confident. I keep the same global scale `a_hat` (median ratio) and the same per-patient shrink logic, but refine `b_hat` with a short, fast 1D local search around the best grid value using the *actual competition metric* (OOF), then reuse that refined `b_hat` at inference. This is a minimal post-processing-only change, expected to move the score upward toward the target while staying within runtime limits.'

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




## === cell 1
SEED = 42

if os.path.exists("/kaggle/input"):
    DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression/"
else:
    DATA_DIR = "../data/raw/"




## === cell 2
def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    dst_data = pd.DataFrame()

    for patient, u_data in data.groupby("Patient"):
        label = pd.DataFrame(
            {"Patient": patient, "pred_Weeks": u_data["Weeks"], "FVC": u_data["FVC"]}
        )

        features = pd.DataFrame(
            {
                "Patient_Week": u_data["Patient"].astype(str)
                + "_"
                + u_data["Weeks"].astype(str),
                "current_FVC": u_data["FVC"],
                "current_Percent": u_data["Percent"],
                "current_Age": u_data["Age"],
                "current_Week": u_data["Weeks"],
                "Patient": u_data["Patient"],
                "Sex": u_data["Sex"].map({"Female": 0, "Male": 1}),
                "SmokingStatus": u_data["SmokingStatus"].map(
                    {"Currently smokes": 0, "Never smoked": 1, "Ex-smoker": 2}
                ),
            }
        )
        dst_u_data = pd.merge(label, features, how="outer", on="Patient")
        dst_u_data = dst_u_data.query("pred_Weeks!=current_Week")
        dst_u_data["passed_Weeks"] = (
            dst_u_data["current_Week"] - dst_u_data["pred_Weeks"]
        )

        dst_data = pd.concat([dst_data, dst_u_data])

    return dst_data


train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
train = preprocessing(train)




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
        """Initialize."""
        self.name = "osic_loss"
        self.n_class = 2  # FVC & Confidence
        self.epsilon = epsilon

    @staticmethod
    def _softplus_stable(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    def __call__(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> float:
        """Calc loss (competition metric)."""
        sigma = self._softplus_stable(preds[:, 1])
        sigma_clip = np.maximum(sigma, 70.0)
        Delta = np.minimum(np.abs(preds[:, 0] - labels), 1000.0)
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
        return self.model.predict(data, num_iteration=None)

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
        "callbacks": [],
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


def softplus_stable(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


oof_mu = oof[:, 0].astype(np.float64)
oof_sigma_raw = oof[:, 1].astype(np.float64)
oof_sigma = softplus_stable(oof_sigma_raw)

abs_resid = np.abs(train["FVC"].to_numpy(dtype=np.float64) - oof_mu)
abs_resid = np.minimum(abs_resid, 1000.0)

tmp_df = pd.DataFrame(
    {
        "Patient": train["Patient"].astype(str).to_numpy(),
        "abs_resid": abs_resid,
        "sigma": oof_sigma,
    }
)

sigma_clip0 = np.maximum(tmp_df["sigma"].to_numpy(dtype=np.float64), 70.0)
ratio0 = tmp_df["abs_resid"].to_numpy(dtype=np.float64) / np.maximum(sigma_clip0, 1e-6)
ratio0 = ratio0[np.isfinite(ratio0)]
global_scale = float(np.nanmedian(ratio0)) if ratio0.size else 1.0
if not np.isfinite(global_scale) or global_scale <= 0:
    global_scale = 1.0

a_hat = float(global_scale)

base_sigma = np.maximum(sigma_clip0 * a_hat, 70.0)
cand_b = np.array(
    [0.0, 10.0, 20.0, 35.0, 50.0, 70.0, 90.0, 120.0, 160.0, 220.0, 300.0],
    dtype=np.float64,
)

best_b = 0.0
best_sc = -1e18
y_true = train["FVC"].to_numpy(dtype=np.float64)

for b in cand_b:
    sigma_cal = np.maximum(base_sigma + b, 70.0)
    sc = score(y_true, oof_mu, sigma_cal)
    if np.isfinite(sc) and sc > best_sc:
        best_sc = sc
        best_b = float(b)


def _refine_b_local(
    base_sigma: np.ndarray, y_true: np.ndarray, mu: np.ndarray, b0: float
) -> float:
    b_best = float(max(0.0, b0))
    sc_best = score(y_true, mu, np.maximum(base_sigma + b_best, 70.0))

    for step in (10.0, 5.0, 2.0, 1.0):
        lo = max(0.0, b_best - 6.0 * step)
        hi = b_best + 6.0 * step
        grid = np.arange(lo, hi + 0.5 * step, step, dtype=np.float64)
        for b in grid:
            sc = score(y_true, mu, np.maximum(base_sigma + b, 70.0))
            if np.isfinite(sc) and sc > sc_best:
                sc_best = sc
                b_best = float(b)
    return float(b_best)


b_hat = _refine_b_local(base_sigma=base_sigma, y_true=y_true, mu=oof_mu, b0=best_b)

sigma_global_cal_clip = np.maximum(sigma_clip0 * a_hat + b_hat, 70.0)
ratio_patient = tmp_df["abs_resid"].to_numpy(dtype=np.float64) / np.maximum(
    sigma_global_cal_clip, 1e-6
)
tmp_df["ratio"] = ratio_patient

patient_scale_raw = (
    tmp_df.groupby("Patient")["ratio"]
    .median()
    .replace([np.inf, -np.inf], np.nan)
    .fillna(1.0)
    .clip(0.5, 2.0)
)

patient_counts = tmp_df.groupby("Patient").size().astype(np.float64)
k_shrink = 3.0  # keep same conservative shrink strength
shrink_w = patient_counts / (patient_counts + k_shrink)
patient_scale = (shrink_w * patient_scale_raw) + ((1.0 - shrink_w) * 1.0)
patient_scale = (
    patient_scale.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.5, 2.0)
)

oof_patient = train["Patient"].astype(str).to_numpy()
oof_scale = (
    pd.Series(oof_patient).map(patient_scale).fillna(1.0).to_numpy(dtype=np.float64)
)

oof_sigma_cal = (np.maximum(oof_sigma, 0.0) * a_hat + b_hat) * oof_scale
oof_sigma_cal = np.maximum(oof_sigma_cal, 70.0)

print(
    "OOF score (raw sigma):",
    score(train["FVC"].to_numpy(), oof_mu, np.maximum(oof_sigma, 70.0)),
)
print(
    "OOF score (calibrated sigma, global scale+bias + patient-shrink):",
    score(train["FVC"].to_numpy(), oof_mu, oof_sigma_cal),
)
print("Global sigma scale (a_hat) =", a_hat, ", global sigma bias (b_hat) =", b_hat)
print("Median patient scale (shrunk):", float(np.median(patient_scale.to_numpy())))




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
test["passed_Weeks"] = test["current_Week"] - test["pred_Weeks"]




## === cell 10
test_idx = test["Patient_Week"].to_numpy().reshape(-1, 1)
pred = np.array([m.predict(test[features]) for m in models]).mean(axis=0)

pred_df = pd.DataFrame(
    np.concatenate((test_idx, pred), axis=1),
    columns=["Patient_Week", "FVC", "Confidence"],
)

pred_df["FVC"] = pred_df["FVC"].astype(float)

pred_df["Confidence"] = pred_df["Confidence"].astype(float)
pred_df["Confidence"] = softplus_stable(pred_df["Confidence"].to_numpy())

pred_sigma = pred_df["Confidence"].to_numpy(dtype=np.float64)
pred_sigma = np.maximum(pred_sigma, 0.0) * a_hat + b_hat  # apply same global scale+bias

test_patients = test["Patient"].astype(str).to_numpy()
test_scale = (
    pd.Series(test_patients).map(patient_scale).fillna(1.0).to_numpy(dtype=np.float64)
)
pred_sigma = pred_sigma * test_scale
pred_sigma = np.maximum(pred_sigma, 70.0)
pred_df["Confidence"] = pred_sigma

pred_df["FVC"] = np.where(
    np.isfinite(pred_df["FVC"].to_numpy()), pred_df["FVC"].to_numpy(), 0.0
)
pred_df["Confidence"] = np.where(
    np.isfinite(pred_df["Confidence"].to_numpy()),
    pred_df["Confidence"].to_numpy(),
    70.0,
)




## === cell 11
submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

sub_df = submission.drop(columns=["FVC", "Confidence"])
sub_df = sub_df.merge(pred_df[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week")
sub_df.columns = submission.columns

sub_df["FVC"] = pd.to_numeric(sub_df["FVC"], errors="coerce").fillna(0.0).astype(float)
sub_df["Confidence"] = (
    pd.to_numeric(sub_df["Confidence"], errors="coerce")
    .fillna(70.0)
    .astype(float)
    .clip(lower=70.0)
)

if os.path.exists("/kaggle/input"):
    sub_df.to_csv("submission.csv", index=False)

print(sub_df.shape)
print(sub_df.head())
print("Any non-finite FVC?", (~np.isfinite(sub_df["FVC"].to_numpy())).any())
print(
    "Any non-finite Confidence?", (~np.isfinite(sub_df["Confidence"].to_numpy())).any()
)
print("Min Confidence:", sub_df["Confidence"].min())
print("Median Confidence:", float(np.median(sub_df["Confidence"].to_numpy())))
print(
    "Median patient sigma scale (train OOF, shrunk):",
    float(np.median(patient_scale.to_numpy())),
)
print("Global sigma scale (a_hat) =", a_hat, ", global sigma bias (b_hat) =", b_hat)
