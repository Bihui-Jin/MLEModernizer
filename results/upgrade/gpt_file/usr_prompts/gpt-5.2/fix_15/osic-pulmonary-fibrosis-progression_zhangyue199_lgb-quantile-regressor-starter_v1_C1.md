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

-7.1368

# 6. Current score

-8.76005

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.65932) has done: 'I fix the pipeline-breaking pandas deprecation (`DataFrame.append`) by switching to `pd.concat`, which unblock creation of the combined `data` table and downstream feature engineering (`base_week`, `min_FVC`). I also fix indexing bugs in KFold training by using `.iloc` (since KFold yields positional indices), and remove LightGBM early-stopping arguments that can error depending on the installed LightGBM version while keeping the same training approach. Finally, I fix the syntax error in the metric print cell and ensure the submission writer outputs a correctly formatted `submission.csv` with required columns and valid confidence values (clipped to ≥70 as per metric). These are minimal correctness/stability fixes aimed at producing a valid end-to-end run and a reasonable score.'
- What this solution (achieved -8.39836) has done: 'I fix the LightGBM runtime error by removing the unsupported `verbose` argument from `LGBMRegressor.fit()` (some Kaggle environments use a version where it isn’t accepted). To keep training behavior stable while still quiet, I rely on LightGBM’s built-in settings without altering the model/loop/feature logic. I also add a small safety clamp to ensure predicted quantiles are ordered so `Confidence = q80 - q20` can’t go negative (this is consistent with the intended quantile-based uncertainty and should improve the metric toward your target). The rest of the pipeline and submission format stays unchanged and write `submission.csv`.'
- What this solution (achieved -9.79787) has done: 'To move your score up toward the target (higher-is-better), the smallest safe lever in this pipeline is the `Confidence` calibration, because the metric explicitly rewards well-calibrated (but clipped) uncertainty. I keep the same LightGBM quantile training loop and features, but I (1) compute out-of-fold quantile predictions for train so we can estimate a scale factor, and (2) rescale the predicted interval (`q80-q20`) to better match the observed absolute residuals of the median predictor, then clip to ≥70 as required. This typically improves the Laplace log-likelihood without changing the model architecture/approach. Submission format and the “fill baseline test measurement with confidence 70” behavior remain unchanged.'
- What this solution (achieved -9.07997) has done: 'Your current gap to the target is about 37% (−9.80 vs −7.14, higher-is-better), so the most direct minimal lever is better calibration of the predicted `Confidence` without changing the model or features. I keep the exact same LightGBM quantile training loop and predictions, but compute an out-of-fold optimal scalar for the interval width using the Laplace likelihood itself (rather than MAE/median heuristics), which is directly aligned with the competition metric. This only changes the post-processing scale factor applied to `(q80-q20)` and keeps the required clipping at 70. I also sort the OOF quantiles (as you already do for test) to ensure widths are valid before calibrating.'
- What this solution (achieved -9.08371) has done: 'We keep your LightGBM quantile training loop and features exactly as-is, and only adjust the post-processing that maps `(q80-q20)` to `Confidence`, because your current score gap to the target is large and the metric is highly sensitive to sigma calibration. Instead of selecting a scale using `abs(y - q50)` (which isn’t aligned with the metric’s clipped delta), we choose the confidence scale by directly maximizing the competition metric on OOF predictions using `delta = |y - q50|` (clipped at 1000) and `sigma = max(70, scale*(q80-q20))`. This is a minimal, metric-aligned change that typically improves the score without changing model logic. We also apply the same sorting/width floor as you already do, to ensure stability and valid (non-negative) confidence.'
- What this solution (achieved -9.0901) has done: 'We keep your exact LightGBM quantile CV training and features, and only make a minimal post-processing adjustment that is directly metric-aligned: calibrate a single global confidence scale using the *same clipped Laplace log-likelihood* but computed on out-of-fold predictions with both delta and sigma clipping applied (you already clip sigma at 70; we also optimize with delta clipped at 1000 exactly as the competition does). To avoid any hidden mismatch, we compute the metric using the same formula used by Kaggle (including the constant term) and pick the scale that maximizes it on OOF. Finally, we ensure the test-time `Confidence1` is strictly positive and clipped, without changing how FVC is filled for baseline rows.'
- What this solution (achieved -9.06127) has done: 'You’re already very close to the target (−9.0901 vs −7.1368; higher-is-better), so we should avoid any model/feature changes and only tune the single lever that most directly affects this metric: `Confidence` calibration. Your current calibration optimizes only a global *scale* for `(q80-q20)`, but the Laplace metric strongly depends on both scale and an additive floor (before the required ≥70 clip), so adding a minimal **affine** calibration `sigma = a*width + b` (with the same clipping) can move the score upward without changing training. We fit `(a,b)` on out-of-fold quantile predictions by directly maximizing the exact competition metric (with both delta and sigma clipping), then apply that to test widths. Everything else (LightGBM quantile CV, features, baseline-row overwrite, submission format/path) remains unchanged.'
- What this solution (achieved -8.6965) has done: 'I keep your LightGBM quantile CV and features unchanged, and only make minimal post-processing tweaks that are directly tied to the Laplace metric. Specifically, I calibrate `Confidence` with a slightly more expressive but still “single global” mapping by searching a small grid over `(a, b)` using the exact competition metric, but also allowing a tiny dependence on `base_week` (scale only) to better match the fact that uncertainty typically grows with time-from-baseline. This does not change any model training or predictions (FVC stays `q50`), it only adjusts sigma construction in a metric-aligned way. I also ensure `Patient_Week` alignment is preserved by ordering the submission exactly like `sample_submission.csv` before writing. The baseline test measurement overwrite (confidence=70) remains exactly as you had it.'
- What this solution (achieved -8.70336) has done: 'Your current score (−8.6965) is worse than the target (−7.1368), so we should cautiously improve while keeping the same LightGBM quantile-CV core. The most metric-aligned minimal lever is still `Confidence`: we (1) compute the OOF metric using the exact competition clipping rules and (2) calibrate a *single global* affine sigma `sigma = a*width + b` plus the existing mild `base_week` scaling `g`, but with a slightly better search that also accounts for the metric’s clipped delta (your current grid is a bit coarse and can miss better (a,b,g) combinations). We also ensure that the OOF calibration uses the same sorting/width floor logic as test, and we keep the baseline test measurement overwrite unchanged. Finally, we keep submission ordering exactly as `sample_submission.csv` and write `submission.csv`.'
- What this solution (achieved -8.69088) has done: 'We keep your LightGBM quantile-CV training and all features exactly the same, and only adjust the confidence calibration because that’s the single lever directly optimized by the Laplace log-likelihood metric. Your current calibration searches a coarse grid over `(a,b,g)`; we add a small, metric-aligned *local continuous refinement* around the best grid point using coordinate-descent with tiny steps, which typically recovers a bit of score without changing model semantics. We also ensure the calibration objective matches Kaggle exactly (delta clipped at 1000 and sigma clipped at 70) and keep the baseline test rows overwritten with `Confidence=70` exactly as you already do. Submission ordering and format remain identical and still write `submission.csv`.'
- What this solution (achieved -8.78737) has done: 'To move your score up toward the target (higher-is-better) while keeping the exact same LightGBM quantile-CV core, the safest lever remains the `Confidence` mapping. Your current calibration is already metric-aligned, but it only tunes an affine transform of `(q80-q20)` and can still be improved by (a) using the *correct Laplace scale factor* that maps an inter-quantile range to the Laplace `b` parameter (`sigma = sqrt(2)*b`) and (b) optionally blending in a small fraction of the model’s empirical residual scale so the confidence doesn’t collapse when width is uninformative. These are strictly post-processing changes (no feature/model/training loop changes) and they directly optimize the Kaggle metric with the same clipping rules. The submission writing/order and baseline overwrite behavior stay identical.'
- What this solution (achieved -8.77757) has done: 'We keep your LightGBM quantile-CV training and all features exactly the same, and only adjust the post-processing that turns (q20,q50,q80) into `Confidence`, since that’s the most metric-sensitive lever and your current score is still below target (needs to improve). Specifically, we (1) calibrate the confidence mapping on out-of-fold predictions with a slightly better metric-aligned search: optimize directly over the Laplace **b** parameter (then convert to sigma), which matches the competition’s likelihood form more naturally than scaling sigma directly. We also (2) remove the `mix` blending with a constant residual scale (set it to 1.0) because it can over-smooth patient/time-specific uncertainty and often hurts score when quantile widths are informative, while still keeping the same overall approach. Finally, we keep your baseline test-row overwrite and submission ordering unchanged to preserve evaluation semantics and ensure a valid `submission.csv`.'
- What this solution (achieved -8.75432) has done: 'We should move your score upward toward the target (higher-is-better) without touching the LightGBM quantile-CV core by only improving the **metric-aligned calibration of `Confidence`**. Right now calibration uses OOF quantiles but only via a coarse grid + small coordinate tweaks; we can usually gain a bit by switching to a deterministic, fast **two-stage refinement**: (1) coarse search as you have, then (2) a slightly denser local search around the best point with the *exact same Laplace metric* and clipping rules. I also apply the **same base_week scaling and width floor** consistently between OOF and test, and explicitly cast to float32/float to avoid tiny inconsistencies; submission writing/order and baseline overwrite remain identical. These are minimal post-processing changes directly tied to the competition metric and should improve from -8.78 toward your -7.14 target.'
- What this solution (achieved -8.76005) has done: 'We keep your LightGBM quantile-CV training and all features exactly unchanged, and only adjust the **confidence calibration** because that is the single lever directly optimized by the Laplace log-likelihood metric. Your current calibration optimizes on unclipped residuals; we make it match Kaggle exactly by using **delta clipped at 1000** during calibration (same as evaluation), which typically improves the score without changing FVC predictions. We also slightly expand and densify the **local refinement search** around the best (a,b,g) to reduce the chance we miss a better nearby setting, while staying fast enough for the time limit. Submission writing/order and the baseline overwrite (Confidence=70 at known test weeks) remain identical.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import pydicom
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor




## === cell 1
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(42)



## === cell 2
ROOT = "../input/osic-pulmonary-fibrosis-progression"



## === cell 3
tr = pd.read_csv(f"{ROOT}/train.csv")
tr.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])
chunk = pd.read_csv(f"{ROOT}/test.csv")

print("add infos")
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub = sub.merge(chunk.drop("Weeks", axis=1), on="Patient")



## === cell 4
tr["WHERE"] = "train"
chunk["WHERE"] = "val"
sub["WHERE"] = "test"
data = pd.concat([tr, chunk, sub], axis=0, ignore_index=True)



## === cell 5
data["min_week"] = data["Weeks"]
data.loc[data.WHERE == "test", "min_week"] = np.nan
data["min_week"] = data.groupby("Patient")["min_week"].transform("min")



## === cell 6
base = data.loc[data.Weeks == data.min_week]
base = base[["Patient", "FVC"]].copy()
base.columns = ["Patient", "min_FVC"]
base["nb"] = 1
base["nb"] = base.groupby("Patient")["nb"].transform("cumsum")
base = base[base.nb == 1]
base.drop("nb", axis=1, inplace=True)



## === cell 7
data = data.merge(base, on="Patient", how="left")
data["base_week"] = data["Weeks"] - data["min_week"]
del base



## === cell 8
for col in ["Sex", "SmokingStatus"]:
    data[col] = data[col].astype("category").cat.codes



## === cell 9
feature_list = ["Age", "Sex", "SmokingStatus", "Percent", "base_week", "min_FVC"]
cat_feat = ["Sex", "SmokingStatus"]



## === cell 10
tr = data.loc[data.WHERE == "train"].copy()
chunk = data.loc[data.WHERE == "val"].copy()
sub = data.loc[data.WHERE == "test"].copy()
del data

tr.shape, chunk.shape, sub.shape



## === cell 11
lgb_params = {
    "n_jobs": 1,
    "max_depth": 4,
    "min_data_in_leaf": 16,
    "subsample": 0.9,
    "n_estimators": 500,
    "learning_rate": 0.02,
    "colsample_bytree": 0.9,
    "boosting_type": "gbdt",
    "metric": ["quantile", "rmse"],
    "random_state": 42,
}



## === cell 12
for df in (tr, chunk, sub):
    for c in feature_list:
        if c not in df.columns:
            df[c] = np.nan
    df[feature_list] = df[feature_list].fillna(
        df[feature_list].median(numeric_only=True)
    )

y = tr["FVC"]
z = tr[feature_list]
ze = sub[feature_list]



## === cell 13
NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=42)



## === cell 14
pred = np.zeros((z.shape[0], 3), dtype=np.float32)
pe = np.zeros((ze.shape[0], 3), dtype=np.float32)

quantiles = [0.2, 0.5, 0.8]
cnt = 0
for tr_idx, val_idx in kf.split(z):
    cnt += 1
    X_tr, y_tr = z.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = z.iloc[val_idx], y.iloc[val_idx]
    for i, q in enumerate(quantiles):
        print(f"FOLD {cnt}, quantile {q}")
        lgb = LGBMRegressor(objective="quantile", alpha=q, **lgb_params)
        lgb.fit(
            X=X_tr,
            y=y_tr,
            eval_set=[(X_val, y_val)],
            categorical_feature=cat_feat,
        )
        pred[val_idx, i] = lgb.predict(X_val)
        pe[:, i] += lgb.predict(ze) / NFOLD

pe = np.sort(pe, axis=1)



## === cell 15
pred_sorted = np.sort(pred, axis=1)
oof_q20, oof_q50, oof_q80 = pred_sorted[:, 0], pred_sorted[:, 1], pred_sorted[:, 2]
y_true = y.values.astype(np.float32)

raw_width = np.maximum(oof_q80 - oof_q20, 1.0).astype(np.float32)

delta = np.abs(y_true - oof_q50).astype(np.float32)
delta_clipped = np.minimum(delta, 1000.0).astype(np.float32)

bw = tr["base_week"].values.astype(np.float32)
bw_factor = (1.0 + (np.abs(bw) / 50.0)).astype(np.float32)


def laplace_metric_mean(delta_arr_clipped, sigma_arr):
    sigma_clipped = np.maximum(sigma_arr, 70.0)
    return float(
        np.mean(
            -(np.sqrt(2.0) * delta_arr_clipped) / sigma_clipped
            - np.log(np.sqrt(2.0) * sigma_clipped)
        )
    )


LN4 = float(np.log(4.0))
B_FROM_WIDTH = (raw_width / (2.0 * LN4)).astype(np.float32)


def sigma_from_params(a_b, b_b, g_b):
    a_b = float(np.clip(a_b, 0.0005, 200.0))
    b_b = float(np.clip(b_b, 0.0, 2000.0))
    g_b = float(np.clip(g_b, 0.0, 1.5))
    bw_adj = np.power(bw_factor, g_b).astype(np.float32)
    b_param = (B_FROM_WIDTH * bw_adj * a_b + b_b).astype(np.float32)
    sigma = (np.sqrt(2.0) * b_param).astype(np.float32)
    return sigma


def score_params_b(a_b, b_b, g_b):
    sigma = sigma_from_params(a_b, b_b, g_b)
    return laplace_metric_mean(delta_clipped, sigma)


best_a_b, best_b_b, best_g_b = 1.0, 0.0, 0.0
best_score = score_params_b(best_a_b, best_b_b, best_g_b)

g_candidates = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25], dtype=np.float32)
a_candidates = np.linspace(0.3, 3.0, 181).astype(np.float32)
b_candidates = np.array(
    [0.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0], dtype=np.float32
)

for g in g_candidates:
    for a in a_candidates:
        for b in b_candidates:
            s = score_params_b(float(a), float(b), float(g))
            if s > best_score:
                best_score = s
                best_a_b, best_b_b, best_g_b = float(a), float(b), float(g)


def local_refine(a0, b0, g0, best_s):
    a0, b0, g0 = float(a0), float(b0), float(g0)

    a_grid = np.clip(np.linspace(a0 * 0.80, a0 * 1.20, 61), 0.0005, 200.0)
    b_grid = np.clip(np.linspace(b0 - 80.0, b0 + 80.0, 41), 0.0, 2000.0)
    g_grid = np.clip(np.linspace(g0 - 0.30, g0 + 0.30, 31), 0.0, 1.5)

    ba, bb, bg, bs = a0, b0, g0, float(best_s)
    for gg in g_grid:
        for aa in a_grid:
            for bb_cand in b_grid:
                s = score_params_b(float(aa), float(bb_cand), float(gg))
                if s > bs:
                    ba, bb, bg, bs = float(aa), float(bb_cand), float(gg), float(s)
    return ba, bb, bg, bs


best_a_b, best_b_b, best_g_b, best_score = local_refine(
    best_a_b, best_b_b, best_g_b, best_score
)

err = mean_absolute_error(y_true, oof_q50)
unc = float(np.mean(raw_width))
print("OOF MAE(q50):", err, "mean_raw_width:", unc)
print(
    "best_a_b:",
    best_a_b,
    "best_b_b:",
    best_b_b,
    "best_g_b:",
    best_g_b,
    "oof_metric(best):",
    best_score,
)

conf_a_b, conf_b_b, conf_g_b = best_a_b, best_b_b, best_g_b




## === cell 16
def get_submission(sub, pe, conf_a_b=1.0, conf_b_b=0.0, conf_g_b=0.0):
    sub = sub.copy()
    sub["FVC1"] = pe[:, 1]

    width = (pe[:, 2] - pe[:, 0]).astype(np.float32)
    width = np.maximum(width, 1.0)

    bw = sub["base_week"].values.astype(np.float32)
    bw_factor = (1.0 + (np.abs(bw) / 50.0)).astype(np.float32)
    bw_adj = np.power(bw_factor, float(conf_g_b)).astype(np.float32)

    LN4 = float(np.log(4.0))
    b_from_width = width / (2.0 * LN4)
    b_param = b_from_width * float(conf_a_b) * bw_adj + float(conf_b_b)
    sigma = (np.sqrt(2.0) * b_param).astype(np.float32)

    sub["Confidence1"] = sigma

    subm = sub[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()
    m = ~subm.FVC1.isnull()
    subm.loc[m, "FVC"] = subm.loc[m, "FVC1"]
    subm.loc[m, "Confidence"] = subm.loc[m, "Confidence1"]

    subm["Confidence"] = subm["Confidence"].astype(float).fillna(200.0)
    subm["Confidence"] = subm["Confidence"].clip(lower=70.0)

    print("fill in prediction that already exists")
    otest = pd.read_csv(f"{ROOT}/test.csv")
    for i in range(len(otest)):
        key = otest.Patient[i] + "_" + str(int(otest.Weeks[i]))
        subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])
        subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0

    sample = pd.read_csv(f"{ROOT}/sample_submission.csv")[["Patient_Week"]]
    subm = sample.merge(subm, on="Patient_Week", how="left")
    subm[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
    print("sub file saved: submission.csv")
    return subm[["Patient_Week", "FVC", "Confidence"]]


submission = get_submission(
    sub, pe, conf_a_b=conf_a_b, conf_b_b=conf_b_b, conf_g_b=conf_g_b
)
submission.head()
