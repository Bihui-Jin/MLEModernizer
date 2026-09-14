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

-10.00667

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -17.86777) has done: 'The crash is due to an API change in LightGBM 4.6.0: `lgb.train()` no longer accepts the legacy `fobj` keyword, so passing it via `**train_param` raises `TypeError`. In the failing cell, we keep the same custom objective and evaluation logic, but switch to LightGBM’s supported `objective` callback interface by setting `params["objective"]` to the custom `grad_and_hess` function and removing `fobj` from `train_params`. This preserves identical training semantics (same gradients/Hessians and same `feval`) while making the call compatible with the installed LightGBM version. No other logic (folding, features, model params, early stopping) is changed.'
- What this solution (achieved -17.86777) has done: 'Your current gap to the target is large (−17.8678 vs −7.4531; higher is better), so we should improve score with minimal, metric-aligned tweaks while keeping the same model, custom loss, and CV training loop. The biggest “free” improvement here is fixing the semantic mismatch between the model’s second output (trained as an unconstrained raw sigma passed through softplus in the loss) and how you use it for scoring/submission (currently using raw values directly, often too small/negative). I therefore apply the same softplus transform to the predicted Confidence at OOF scoring and for submission, and clip it to the competition’s minimum 70 to avoid over-penalization. Additionally, I disable feature-importance plotting inside the fold loop to keep runtime safely under 600s (it doesn’t change model predictions), improving stability.'
- What this solution (achieved -17.86777) has done: 'Your current score is far below the target (−17.87 vs −7.45; higher is better), so we should improve with the smallest changes that better match the competition’s evaluation while keeping your LightGBM custom objective and CV loop intact. The biggest low-risk gain is to align training targets with what you actually need to predict: in the competition, you must forecast future weeks relative to each patient’s baseline, so we add a single feature `Week_from_base = Weeks - base_Weeks` (no new model/loop changes). Additionally, we clip the predicted FVC to a realistic range derived from the training distribution to reduce extreme errors that get thresholded at 1000 but still hurt via the log-sigma term; this is only post-processing and keeps semantics. Everything else (custom loss, folds, parameters, softplus+clip for confidence, submission format/path) remains the same.'
- What this solution (achieved -inf) has done: 'Diagnosis: The crash happens when LightGBM constructs the `Dataset` and detects that `X_train` has `n` rows but `y_train` has `2n` elements. This mismatch is caused by flattening `y2` into a 1D array while `lgb.Dataset` still expects one label per row for standard training data; `num_class=2` does not change label length requirements in this setup. The intended approach here is to keep one label per row (the true FVC), while the custom objective/metric handles 2 outputs via the `preds` vector.

Patch summary: In cell 7, stop flattening the 2-column target into a 1D array. Instead, pass `train["FVC"]` as a 1D float32 label vector to LightGBM, while keeping the rest of the logic (custom objective, params, CV loop, and prediction reshaping) unchanged.

Updated cells: cell 7 only.

Compatibility notes for cell k+1: `oof` remains shape `(n_rows, 2)` and is filled from `pred_valid` exactly as before, so `cell 8` (which computes `oof_fvc` and `oof_sigma`) work unchanged.

Assumptions: The custom objective `OSICLossForLGBM.grad_and_hess` expects `data.get_label()` to be a 1D array of true FVC values (one per row), consistent with how the competition metric is computed later.'
- What this solution (achieved -21.41632) has done: 'Your current Kaggle score of `-inf` strongly suggests your submission contains invalid values (typically `Confidence <= 0`, `NaN`, or `inf`), which makes the metric undefined due to the `log(sigma)` term. I keep your model/training loop intact and only harden the prediction post-processing so `FVC` and `Confidence` are always finite, and `Confidence` is strictly positive and clipped to the competition minimum (70). I also enforce numeric dtypes and drop any accidental non-finite predictions before writing, so the generated `submission.csv` is always valid. These changes should move the score from `-inf` to a finite value and toward your target.'
- What this solution (achieved -10.00663) has done: 'I keep your exact LightGBM setup (same custom objective/feval, folds, and features) and make two minimal, metric-aligned fixes that typically yield a sizeable gain on OSIC without changing the modeling approach. First, I calibrate the predicted `Confidence` using an out-of-fold-derived global multiplier so sigma better matches the actual residual scale (reduces the Laplace-NLL penalty from under/over-confident sigmas). Second, I apply a per-patient linear correction using the known test baseline measurement (week 0 row in `test.csv`) so all future-week predictions are anchored to the only provided ground-truth FVC for that patient, which reduces systematic offsets while preserving your model’s learned week trend. Both changes are pure post-processing and should move your score upward toward the target with low risk.'
- What this solution (achieved -11.75764) has done: 'We’re still far below the target (current −10.0066 vs target −7.4531; higher is better), so we should improve score with minimal, metric-aligned post-processing rather than changing your LightGBM training/core loss. The largest remaining “cheap” gain is to predict the *patient-specific last-3-visit weeks* directly (those are the only scored rows) by building them from `test.csv` rather than blindly using `sample_submission.csv` weeks, while keeping your exact model/features and still writing a valid 1908-row submission. Then, for those last-3 rows we blend (anchor) predictions with a simple per-patient linear model fitted on that patient’s training trajectory (Weeks→FVC) and keep your baseline anchoring; this reduces bias and large deltas on scored weeks without altering training. Finally, we calibrate Confidence using OOF residuals more directly for the Laplace metric (still clipped at 70), improving the log-likelihood term without changing model semantics.'
- What this solution (achieved -11.75764) has done: 'I keep your LightGBM training, custom loss, folds, and feature set unchanged, and only adjust post-processing that directly impacts the competition metric. Specifically, I replace the current global Confidence scaling with an out-of-fold (OOF) grid search for a single sigma multiplier that maximizes the Laplace metric on OOF (this is metric-aligned calibration and usually yields a reliable lift). I also re-tune the strength of your per-patient linear “history blend” (alpha) using OOF evaluation on the same scored-weeks mask logic, so it nudges predictions in the right direction without changing the core model. Finally, I keep all safety clipping/finite checks and still write a valid `submission.csv` with the required columns and 1908 rows.'
- What this solution (achieved -11.75764) has done: 'Your current score (−11.7576) is still well below the target (−7.4531), so we should improve with minimal, metric-aligned changes without touching the LightGBM training/core loss. I (1) stop using the “train last-3 weeks” mask for test scoring rows (it’s mismatched: test patient IDs don’t exist in train), and instead infer the scored weeks directly from `sample_submission.csv` as the maximum week per patient (last 3 available weeks). Then (2) I re-tune the per-patient linear blend `alpha` using OOF but only on the true “last-3” weeks per patient in train to better match the evaluation rows. Finally (3) I keep your sigma softplus + scaling, but re-optimize the sigma scale on the same “last-3” OOF mask (not all weeks) so confidence calibration is aligned to what’s actually scored.'
- What this solution (achieved -10.00667) has done: 'I keep your LightGBM training/custom loss and fold loop unchanged and only adjust post-processing that directly impacts the Laplace metric. The biggest safe gain is to replace the current coarse sigma scale grid with a continuous, OOF-optimized single multiplier (maximize the exact competition metric on the same “last-3” rows), which improves confidence calibration without changing model predictions. Next, I tune the per-patient history blend strength (alpha) using the same continuous search instead of a small discrete set, again only on last-3 rows to match evaluation. Finally, I compute the per-patient linear history using `Week_from_base` (not absolute Weeks) to align with your feature definition and baseline anchoring, while keeping the same linear correction idea.'

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



## === cell 1
SEED = 42

if os.path.exists("/kaggle/input"):
    DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression/"
else:
    DATA_DIR = "../data/raw/"

np.random.seed(SEED)




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
y = train["FVC"].values.astype(np.float32)

groups = train["Patient"]

num_fold = 5
g_kfold = model_selection.GroupKFold(n_splits=num_fold)

models = []
oof = np.zeros((train.shape[0], 2), dtype=np.float32)

for fold, (train_idx, valid_idx) in enumerate(
    g_kfold.split(X, train["FVC"], groups), 1
):
    X_train, y_train = X.iloc[train_idx, :], y[train_idx]
    X_valid, y_valid = X.iloc[valid_idx, :], y[valid_idx]

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




## === cell 8
def score(fvc_true, fvc_pred, sigma):
    sigma_clip = np.maximum(sigma, 70)
    delta = np.minimum(np.abs(fvc_true - fvc_pred), 1000)
    metric = -(np.sqrt(2) * delta / sigma_clip) - np.log(np.sqrt(2) * sigma_clip)
    return np.mean(metric)


oof_fvc = oof[:, 0].astype(np.float64)
oof_sigma_raw = oof[:, 1].astype(np.float64)

oof_sigma_sp = np.log1p(np.exp(oof_sigma_raw))
oof_sigma_sp = np.where(np.isfinite(oof_sigma_sp), oof_sigma_sp, 70.0)
oof_sigma_sp = np.maximum(oof_sigma_sp, 70.0)

train_weeks = train["Weeks"].astype(int).values
train_patients = train["Patient"].astype(str).values
train_last3_map = (
    train.groupby("Patient")["Weeks"]
    .apply(lambda s: set(sorted(s.astype(int).unique())[-3:]))
    .to_dict()
)
is_scored_train = np.array(
    [
        train_weeks[i] in train_last3_map.get(train_patients[i], set())
        for i in range(train.shape[0])
    ],
    dtype=bool,
)

base_oof_score_all = score(
    train["FVC"].values.astype(np.float64), oof_fvc, oof_sigma_sp
)
base_oof_score_last3 = score(
    train["FVC"].values.astype(np.float64)[is_scored_train],
    oof_fvc[is_scored_train],
    oof_sigma_sp[is_scored_train],
)


def golden_section_maximize(f, a, b, tol=1e-4, max_iter=80):
    gr = (np.sqrt(5.0) + 1.0) / 2.0
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    fc = f(c)
    fd = f(d)
    it = 0
    while (b - a) > tol and it < max_iter:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / gr
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / gr
            fd = f(d)
        it += 1
    xbest = (a + b) / 2.0
    return float(xbest), float(f(xbest))


y_true_last3 = train["FVC"].values.astype(np.float64)[is_scored_train]
oof_fvc_last3 = oof_fvc[is_scored_train]
oof_sigma_last3 = oof_sigma_sp[is_scored_train]


def sigma_scale_objective(s):
    s = float(s)
    if not np.isfinite(s) or s <= 0:
        return -1e18
    return score(y_true_last3, oof_fvc_last3, np.maximum(oof_sigma_last3 * s, 70.0))


best_scale, best_score_last3 = golden_section_maximize(sigma_scale_objective, 0.4, 3.0)

print("OOF base score (all rows, no sigma scaling):", base_oof_score_all)
print("OOF base score (last3 rows, no sigma scaling):", base_oof_score_last3)
print(
    "OOF best score (last3 rows, sigma scaled - continuous):",
    best_score_last3,
    "best_scale:",
    best_scale,
)



## === cell 9
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
test



## === cell 10
submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["Weeks"] = (
    submission["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
)

test_raw = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

max_week_by_patient = (
    submission.groupby("Patient")["Weeks"].transform("max").astype(int)
)
is_scored = (submission["Weeks"].astype(int) >= (max_week_by_patient - 2)).to_numpy()

test_feat = test_raw.drop(columns=["Weeks"])
test_feat = submission.drop(columns=["FVC", "Confidence"]).merge(
    test_feat, how="left", on="Patient"
)
test_feat = preprocessing(test_feat)
test_feat = test_feat[train.columns.tolist()]

print(test_feat.shape)
test_feat.head()



## === cell 11
train.head()



## === cell 12
test_idx = test_feat["Patient_Week"].to_numpy().reshape(-1, 1)

pred_all = []
for m in models:
    p = m.predict(test_feat[features])
    p = np.asarray(p).reshape(2, test_feat.shape[0]).T
    pred_all.append(p)
pred = np.mean(np.stack(pred_all, axis=0), axis=0)

pred_fvc = pred[:, 0].astype(np.float64)
pred_sigma = np.log1p(np.exp(pred[:, 1].astype(np.float64)))

pred_fvc = np.where(np.isfinite(pred_fvc), pred_fvc, np.nan)
pred_sigma = np.where(np.isfinite(pred_sigma), pred_sigma, np.nan)

fvc_fill = float(np.nanmedian(train["FVC"].values))
pred_fvc = np.where(np.isnan(pred_fvc), fvc_fill, pred_fvc)

pred_sigma = np.where(np.isnan(pred_sigma), 70.0, pred_sigma)
pred_sigma = np.maximum(pred_sigma, 70.0)

pred_sigma = np.maximum(pred_sigma * best_scale, 70.0)

test_base = test_raw[["Patient", "FVC"]].copy()
base_fvc_map = dict(
    zip(test_base["Patient"].astype(str).values, test_base["FVC"].astype(float).values)
)

test_patients = test_feat["Patient"].astype(str).values
test_week_from_base = test_feat["Week_from_base"].astype(float).values

is_base_row = np.isclose(test_week_from_base, 0.0)
pred_base_by_patient = {}
for p in np.unique(test_patients[is_base_row]):
    idxs = np.where(is_base_row & (test_patients == p))[0]
    if idxs.size > 0:
        pred_base_by_patient[p] = float(np.median(pred_fvc[idxs]))

offsets = np.zeros_like(pred_fvc, dtype=np.float64)
for i, p in enumerate(test_patients):
    true_base = base_fvc_map.get(p, np.nan)
    pred_base = pred_base_by_patient.get(p, np.nan)
    if np.isfinite(true_base) and np.isfinite(pred_base):
        offsets[i] = true_base - pred_base
pred_fvc = pred_fvc + offsets

train_proc = preprocessing(train_raw.copy())
train_by_patient = train_proc.groupby("Patient")
weeks_full = submission["Weeks"].astype(int).values

train_week_from_base = train["Week_from_base"].astype(float).values

lin_pred_train = np.full(train.shape[0], np.nan, dtype=np.float64)
for p, hist in train_by_patient:
    hist = hist.sort_values("Weeks")
    x = hist["Week_from_base"].values.astype(np.float64)
    yv = hist["FVC"].values.astype(np.float64)
    if x.size < 2:
        continue
    x_mean = float(x.mean())
    y_mean = float(yv.mean())
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 1e-9:
        continue
    slope = float(np.sum((x - x_mean) * (yv - y_mean)) / denom)
    intercept = y_mean - slope * x_mean
    idxs = np.where(train_patients == str(p))[0]
    if idxs.size == 0:
        continue
    lin_pred_train[idxs] = intercept + slope * train_week_from_base[idxs].astype(
        np.float64
    )

valid_mask = is_scored_train & np.isfinite(lin_pred_train)

y_true_last3 = train["FVC"].values.astype(np.float64)[is_scored_train]
sigma_last3_scaled = np.maximum(oof_sigma_sp[is_scored_train] * best_scale, 70.0)


def alpha_objective(a):
    a = float(a)
    if not np.isfinite(a) or a < 0.0 or a > 0.8:
        return -1e18
    fvc_blend = oof_fvc.copy()
    fvc_blend[valid_mask] = (1.0 - a) * fvc_blend[valid_mask] + a * lin_pred_train[
        valid_mask
    ]
    return score(y_true_last3, fvc_blend[is_scored_train], sigma_last3_scaled)


if np.any(valid_mask):
    best_alpha, best_alpha_score = golden_section_maximize(alpha_objective, 0.0, 0.8)
else:
    best_alpha, best_alpha_score = 0.0, score(
        y_true_last3, oof_fvc[is_scored_train], sigma_last3_scaled
    )

print(
    "Chosen alpha (OOF-tuned on last3 rows, continuous):",
    best_alpha,
    "OOF last3 score with tuned alpha:",
    best_alpha_score,
)

for p in np.unique(test_patients[is_scored]):
    if p not in train_by_patient.groups:
        continue
    hist = train_by_patient.get_group(p).sort_values("Weeks")
    x = hist["Week_from_base"].values.astype(np.float64)
    yv = hist["FVC"].values.astype(np.float64)
    if x.size < 2:
        continue
    x_mean = float(x.mean())
    y_mean = float(yv.mean())
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 1e-9:
        continue
    slope = float(np.sum((x - x_mean) * (yv - y_mean)) / denom)
    intercept = y_mean - slope * x_mean

    idxs = np.where(is_scored & (test_patients == p))[0]
    if idxs.size == 0:
        continue
    fvc_lin = intercept + slope * test_week_from_base[idxs].astype(np.float64)
    pred_fvc[idxs] = (1.0 - best_alpha) * pred_fvc[idxs] + best_alpha * fvc_lin

fvc_low, fvc_high = np.percentile(train["FVC"].values, [0.5, 99.5])
pred_fvc = np.clip(pred_fvc, fvc_low, fvc_high)

pred_df = pd.DataFrame(
    {
        "Patient_Week": test_idx.reshape(-1),
        "FVC": pred_fvc.astype(np.float32),
        "Confidence": pred_sigma.astype(np.float32),
    }
)



## === cell 13
submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

sub_df = submission.drop(columns=["FVC", "Confidence"])
sub_df = sub_df.merge(pred_df[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week")
sub_df.columns = submission.columns

sub_df["FVC"] = pd.to_numeric(sub_df["FVC"], errors="coerce")
sub_df["Confidence"] = pd.to_numeric(sub_df["Confidence"], errors="coerce")
sub_df["FVC"] = sub_df["FVC"].fillna(float(np.nanmedian(train["FVC"].values)))
sub_df["Confidence"] = sub_df["Confidence"].fillna(70.0).clip(lower=70.0)

assert sub_df.shape[0] == submission.shape[0]
assert sub_df["FVC"].notna().all()
assert sub_df["Confidence"].notna().all()
assert np.isfinite(sub_df["FVC"].values).all()
assert np.isfinite(sub_df["Confidence"].values).all()
assert (sub_df["Confidence"].values >= 70.0).all()

sub_df.to_csv("submission.csv", index=False)

print(sub_df.shape)
sub_df.head()
