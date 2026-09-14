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

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pydicom==3.0.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

-6.930800387972361

# 6. Current score

-13.2676

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -16.21033) has done: 'The crash happens because the expanded test feature frame never carries `Patient_Week`, so later alignment to `sample_submission` fails. I minimally propagate `Patient_Week` from the merged submission expansion into `data_test_features`, and ensure `make_eval_data()` preserves it in its output. I also fix the assignment of `final_sub[["FVC","Confidence"]]` (it currently creates a tuple of Series rather than a 2-column DataFrame), keeping the same prediction logic and confidence clipping so the pipeline runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved -8.78501) has done: 'Your current score is far below the target, so we should improve it with minimal, low-risk changes that don’t alter the model architecture or training loop. The biggest gain here is to make inference match the competition metric requirement: the Laplace metric uses a *clipped* sigma (≥70) and benefits from reasonable calibration, so we (1) enforce a valid, positive confidence everywhere and (2) apply a single global confidence scaling factor fitted on the training set (using out-of-fold-style per-row predictions without changing training) to better match typical residual sizes. We also fix a subtle preprocessing mismatch: `Healthy-FVC` in train uses `FVC` but test uses `base_FVC`; we align train to use `base_FVC` so the feature semantics match between train/test. These are small changes that typically move OSIC baselines several points upward without changing core logic.'
- What this solution (achieved -16.21033) has done: 'Your score (-8.785) is below the target (-6.93), so we should improve it with minimal, metric-aligned tweaks while keeping your model and training intact. The main low-risk gain is to make the model’s implied uncertainty behave like the competition’s clipped sigma by (1) enforcing a strictly positive confidence at inference and (2) fitting a single global confidence scaling factor using the competition metric (grid search on train predictions) instead of a median residual ratio. Additionally, we clip the inferred confidence to at least 70 inside `make_eval_data()` so both training-metric calibration and test submission use the same semantics. These changes do not alter architecture, features, loss, or the training loop; they only calibrate post-processing toward the evaluation metric.'
- What this solution (achieved -7.62814) has done: 'Your current score (-16.21) is far below the target (-6.93), so we should improve it with very small, metric-aligned fixes that don’t change your model, loss, or training loop. The biggest issue is a train/test feature mismatch: during training you treat each visit’s FVC as the “base_FVC”, but at test time “base_FVC” is the baseline-only measurement; this inconsistency can heavily hurt generalization, so we make training use a true per-patient baseline row (Week closest to 0) as `base_FVC/base_Weeks` for all that patient’s targets. Next, we ensure the confidence scaling search uses the same clipping semantics as the competition (sigma clipped at 70) and apply a final clip after scaling so calibration can’t produce invalid/too-small sigmas. These changes are minimal (preprocessing + calibration only), keep the model identical, and typically move OSIC baselines substantially upward toward your target band.'
- What this solution (achieved -24.65932) has done: 'Your current score (-7.62814) is below the target (-6.9308), so we should improve it slightly with minimal, low-risk changes that keep your model/training intact. The biggest issue in your current script is that the confidence scaling is accidentally undone (you multiply by `scale` and then immediately overwrite it), so the calibration you computed is not actually applied to the submission; fixing this should move the score toward the target. To keep the calibration meaningful and avoid optimistic in-sample scaling, we also pick the confidence scale using a simple patient-wise CV loop (same model/training approach, just repeated a few times with fewer epochs for speed), then train once on full data and apply that scale to test. All I/O paths stay the same and the submission format remains unchanged.'
- What this solution (achieved -13.2676) has done: 'Your current score (-24.66) is far below the target (-6.93), so we should improve it with the smallest, safest fixes that keep the same model, loss, and training loop. The biggest score-killer is that your training/inference `score()` doesn’t implement the competition’s clipped-sigma + capped-delta Laplace log-likelihood, so the model learns a different objective; we minimally change `score()` to match the metric while keeping the same 3-output head semantics. Next, we make the model’s sigma numerically safe during both training and inference by enforcing sigma ≥ 70 inside the loss/metric computation (as Kaggle does) and by preventing negative/near-zero sigma from exploding gradients. Finally, we keep your existing confidence scaling CV logic, but make its train-time evaluation consistent with the corrected metric (so the selected scale actually helps under the real scoring rule).'
- What this solution (achieved -13.2676) has done: 'Your current score (-13.2676) is well below the target (-6.9308), so we should improve it with minimal, low-risk changes that keep your model, loss, and training loop intact. The biggest issue is a train/test feature mismatch: at training time you use each row’s own FVC as `base_FVC`, while at test time `base_FVC` is the baseline measurement; we align training preprocessing to use the patient’s baseline FVC/Week (closest to week 0) as `base_FVC/base_Weeks` for all rows. Next, we make `make_eval_data()` fast and numerically consistent by batching inference (no semantic change) and ensuring confidence clipping is applied exactly once in one place. These changes typically yield a large score jump for OSIC baselines without altering the architecture or optimization scheme.'
- What this solution (achieved -13.2676) has done: 'We make two minimal, metric-aligned fixes that should improve the public score from -13.27 toward your target -6.93 without changing the model, features, architecture, or training loop. First, your training `score()` currently has the wrong sign versus the Kaggle metric (you’re minimizing the negative of what Kaggle maximizes), so we flip it to exactly match the competition Laplace log-likelihood (still using sigma-clipping at 70 and delta-capping at 1000). Second, your test-time overwrite for the baseline week checks `Week == base_Weeks`, but in your expanded test rows `Week` is the submission week while `base_Weeks` is the baseline week from test.csv (usually 0); we instead overwrite the row where `Week == 0` (baseline measurement) so FVC is anchored correctly and confidence set to 70 there. These are small, direct fixes that preserve your approach but make optimization and post-processing consistent with the evaluation and data semantics.'
- What this solution (achieved -13.2676) has done: 'Your current gap to the target is large (about -13.27 vs -6.93), so the safest improvement is to fix a core training bug that makes the model optimize the *opposite direction* of the Kaggle metric. I change `score()` to return the Kaggle Laplace log-likelihood mean directly (higher is better), and keep the existing training loop intact by minimizing `-score()` inside `quartile_loss()` (so the optimizer still minimizes a loss). This preserves the same model, features, and loop structure, but aligns optimization with the evaluation and should move the score substantially upward toward the target. I also make the sqrt(2) constant device/dtype-safe to avoid tiny numerical inconsistencies.'
- What this solution (achieved -13.2676) has done: 'I fix the crash in the test-expansion merge by sorting on the correct post-merge column names (`Weeks_x/Weeks_y`) and only then renaming them, so `data_test_expanded`/`data_test_features` are created and later cells can run. I also make the merge robust by merging on both `Patient` and the requested week so `Patient_Week` is preserved and aligned correctly to `sample_submission`. These are execution/blocking fixes and should be score-neutral (they don’t change the model, features, loss, or calibration logic). Finally, I add a tiny safety check to ensure all expected one-hot columns exist in test features (filled with 0) to prevent KeyErrors when a category is missing in test.'
- What this solution (achieved -13.2676) has done: 'Your current score (-13.2676) is still far below the target (-6.9308), so we should improve generalization with a minimal, metric-consistent fix rather than tuning for “best possible.” The biggest remaining issue is that the test expansion merge is currently done only on `Patient`, which duplicates baseline rows and can silently misalign `Patient_Week` ↔ `Week` associations; I change that merge to be on both `Patient` and `Weeks` (the requested submission week), preserving the same downstream feature logic but ensuring each `Patient_Week` gets the correct `Week`. I also make the baseline overwrite consistent with OSIC semantics by anchoring the row where `Week == base_Weeks` (not hard-coded to 0), which is a small but important correctness fix when baseline week isn’t zero. These changes are execution-safe, keep your model/training/loss unchanged, and should move the score upward toward the target by removing label/row misalignment noise.'
- What this solution (achieved -13.2676) has done: 'Your current score is far below the target, so we should improve it with very small, low-risk correctness fixes that don’t change the model, features, or training loop. The biggest issue is in the test expansion merge: because you merge on both `Patient` and `Weeks`, all non-baseline weeks have missing clinical fields, which then get filled with zeros and destroys inference quality; we minimally forward-fill each patient’s baseline clinical attributes (`Percent/Age/Sex/SmokingStatus/base_FVC`) across all requested weeks. Next, we fix the incorrect `rename()` in that same block (it currently references non-existent `Weeks_y`) to avoid silent column mishandling and ensure `Week/base_Weeks` are consistent. These changes keep the exact same architecture/loss/training, but make test features match train semantics much better, which should move the score upward toward your target band.'
- What this solution (achieved -13.2676) has done: 'Your current score (-13.27) is far below the target (-6.93), so we should improve generalization with minimal, correctness-focused changes that keep your model/training/loss intact. The biggest remaining issue is that test-time clinical fields (Percent/Age/Sex/SmokingStatus) are being forward/back-filled from `data_test` after a left-join that only matches baseline weeks; if any patient has missing baseline in that merge, you end up with zeros which badly distort features. I make the fill use the *patient’s known baseline row from `test.csv` explicitly* (one row per patient), then broadcast it to all requested weeks, ensuring test features match train semantics without changing feature definitions. I also ensure `Healthy-FVC` is computed safely (avoid divide-by-zero when Percent is missing/0) to prevent extreme values that destabilize predictions.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset




## === cell 1
def csv_preprocess(data):
    data = data.copy()

    COLS = ["Sex", "SmokingStatus"]
    FE = []
    for col in COLS:
        for mod in data[col].dropna().unique():
            FE.append(mod)
            data[mod] = (data[col] == mod).astype(int)

    data = data[["Patient", "Weeks", "FVC", "Age", "Percent"] + FE]
    data = data.sort_values(["Patient", "Weeks"], ascending=True).reset_index(drop=True)

    FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]

    def _pick_baseline_row(g):
        g = g.copy()
        base = g.sort_values(["Weeks"], key=lambda s: s.abs(), ascending=True).head(1)
        if len(base) == 0:
            return g
        base_weeks = float(base["Weeks"].iloc[0])
        base_fvc = float(base["FVC"].iloc[0])
        g["base_Weeks"] = base_weeks
        g["base_FVC"] = base_fvc
        g["Week"] = g["Weeks"].astype(float)
        g["actual_FVC"] = g["FVC"].astype(float)
        return g

    data = (
        data.groupby("Patient", group_keys=False)
        .apply(_pick_baseline_row)
        .reset_index(drop=True)
    )

    pct = pd.to_numeric(data["Percent"], errors="coerce").astype(float).values
    pct = np.where(np.isfinite(pct) & (pct > 0), pct, np.nan)
    healthy = np.round((data["base_FVC"].astype(float).values * 100.0) / pct)
    data["Healthy-FVC"] = pd.Series(healthy).fillna(0.0).astype(float)

    npData = data[
        ["Patient", "base_Weeks", "base_FVC", "Age"]
        + FE1
        + ["Week", "Healthy-FVC", "actual_FVC"]
    ].copy()

    npData.reset_index(inplace=True, drop=True)
    npData = npData.fillna(0)
    return npData




## === cell 2
class SIGMA(nn.Module):
    def __init__(self):
        super(SIGMA, self).__init__()
        self.data_net1 = nn.Sequential(
            nn.Linear(10, 42),
            nn.ReLU(),
            nn.Linear(42, 64),
            nn.ReLU(),
            nn.Linear(64, 118),
            nn.ReLU(),
        )
        self.data_net2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 502),
            nn.ReLU(),
        )
        self.data_net3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 118),
            nn.ReLU(),
        )
        self.data_net4 = nn.Sequential(
            nn.Linear(748, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.ReLU(),
        )

    def forward(self, data_i):
        out1 = self.data_net1(data_i)
        out2 = torch.cat((data_i, out1), dim=-1)
        out2 = self.data_net2(out2)
        out3 = torch.cat((data_i, out2), dim=-1)
        out3 = self.data_net3(out3)
        out4 = torch.cat((data_i, out1, out2, out3), dim=-1)
        out = self.data_net4(out4)
        return out




## === cell 3
def score(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    sigma = torch.clamp(sigma, min=70.0)

    fvc_pred = y_pred[:, 1]
    delta = (y_true[:, 0] - fvc_pred).abs()
    delta = torch.clamp(delta, max=1000.0)

    sq2 = torch.sqrt(torch.tensor(2.0, device=y_true.device, dtype=y_true.dtype))
    kaggle_metric = (-sq2 * delta / sigma) - torch.log(sq2 * sigma)
    return kaggle_metric.mean()


def closs(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    sigma = torch.clamp(sigma, min=70.0)
    e = torch.abs(y_true[:, 0] - y_pred[:, 1])
    loss = torch.abs(sigma - e)
    return loss


def quartile_loss(y_true, y_pred):
    return -score(y_true, y_pred)




## === cell 4
def make_eval_data(npEval, model, device="cuda", batch_size=1024):
    npEval = npEval.copy()

    x_features = npEval[
        [
            "base_Weeks",
            "base_FVC",
            "Age",
            "Male",
            "Female",
            "Ex-smoker",
            "Never smoked",
            "Currently smokes",
            "Week",
            "Healthy-FVC",
        ]
    ].astype(np.float32)
    x_features = torch.tensor(x_features.values, dtype=torch.float32)

    use_cuda = torch.cuda.is_available() and device == "cuda"
    dev = "cuda" if use_cuda else "cpu"
    model = model.to(dev)
    model.eval()

    dl = DataLoader(TensorDataset(x_features), batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(dev)
            p = model(xb).detach().to("cpu").numpy()
            preds.append(p)

    preds = np.concatenate(preds, axis=0)
    npEval["FVC"] = preds[:, 1]

    conf = (preds[:, 2] - preds[:, 0]).astype(np.float32)
    conf = np.where(np.isfinite(conf), conf, 70.0).astype(np.float32)
    conf = np.clip(conf, 70.0, None)
    npEval["Confidence"] = conf
    return npEval




## === cell 5
DATA_DIR = "../input/osic-pulmonary-fibrosis-progression"
data_train = pd.read_csv(f"{DATA_DIR}/train.csv")
data_test = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")



## === cell 6
submission = sample_submission.copy()
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["Weeks"] = submission["Patient_Week"].apply(lambda x: int(x.split("_")[1]))
submission_sorted = submission.sort_values(
    by=["Patient", "Weeks"], ascending=True
).reset_index(drop=True)



## === cell 7
test_base = data_test[
    ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
].copy()
test_base = test_base.rename(columns={"Weeks": "base_Weeks", "FVC": "base_FVC"})
test_base["base_Weeks"] = (
    pd.to_numeric(test_base["base_Weeks"], errors="coerce").fillna(0.0).astype(float)
)
test_base["base_FVC"] = (
    pd.to_numeric(test_base["base_FVC"], errors="coerce").fillna(0.0).astype(float)
)

merge = (
    pd.merge(
        submission_sorted[["Patient", "Weeks", "Patient_Week"]],
        test_base,
        on=["Patient"],
        how="left",
    )
    .sort_values(["Patient", "Weeks"])
    .reset_index(drop=True)
)

merge = merge.rename(columns={"Weeks": "Week"})

data_test_expanded = merge.loc[
    :,
    [
        "Patient_Week",
        "Patient",
        "base_Weeks",
        "base_FVC",
        "Percent",
        "Age",
        "Sex",
        "SmokingStatus",
        "Week",
    ],
].copy()

submission_out = merge.loc[:, ["Patient_Week", "base_FVC"]].rename(
    columns={"base_FVC": "FVC"}
)
submission_out["Confidence"] = 70.0



## === cell 8
data = data_test_expanded.copy()

pct = pd.to_numeric(data["Percent"], errors="coerce").astype(float).values
pct = np.where(np.isfinite(pct) & (pct > 0), pct, np.nan)
healthy = np.round(
    (
        pd.to_numeric(data["base_FVC"], errors="coerce")
        .fillna(0.0)
        .astype(float)
        .values
        * 100.0
    )
    / pct
)
data["Healthy-FVC"] = pd.Series(healthy).fillna(0.0).astype(float)

COLS = ["Sex", "SmokingStatus"]
for col in COLS:
    for mod in data[col].dropna().unique():
        data[mod] = (data[col] == mod).astype(int)

FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
for c in FE1:
    if c not in data.columns:
        data[c] = 0

npData = pd.DataFrame(
    columns=["Patient_Week", "Patient", "base_Weeks", "base_FVC", "Age", "Healthy-FVC"]
    + FE1
    + ["Week"]
)
npData = pd.concat([npData, data], ignore_index=True, sort=True)
npData = npData.fillna(0)

data_test_features = npData[
    [
        "Patient_Week",
        "Patient",
        "base_Weeks",
        "base_FVC",
        "Age",
        "Healthy-FVC",
        "Male",
        "Female",
        "Ex-smoker",
        "Never smoked",
        "Currently smokes",
        "Week",
    ]
].copy()




## === cell 9
def laplace_metric_np(fvc_true, fvc_pred, sigma):
    sigma = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(fvc_true - fvc_pred), 1000.0)
    return (-np.sqrt(2.0) * delta / sigma) - np.log(np.sqrt(2.0) * sigma)


def _set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_model_on_npTrain(
    npTrain, device, epochs=25, lr=1e-3, batch_size=256, seed=42
):
    _set_seeds(seed)
    x = (
        npTrain[
            [
                "base_Weeks",
                "base_FVC",
                "Age",
                "Male",
                "Female",
                "Ex-smoker",
                "Never smoked",
                "Currently smokes",
                "Week",
                "Healthy-FVC",
            ]
        ]
        .astype(np.float32)
        .values
    )
    y = npTrain[["actual_FVC"]].astype(np.float32).values

    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    ds = TensorDataset(x_t, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = SIGMA().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = quartile_loss(yb, pred).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model = model.to("cpu")
    return model


def _last3_rows_per_patient(np_df):
    np_df = np_df.copy()
    np_df["Patient"] = np_df["Patient"].astype(str)
    np_df["Week"] = (
        pd.to_numeric(np_df["Week"], errors="coerce").fillna(0.0).astype(float)
    )
    return (
        np_df.sort_values(["Patient", "Week"])
        .groupby("Patient", group_keys=False)
        .tail(3)
        .reset_index(drop=True)
    )


def fit_confidence_scale_cv(data_train_df, device, n_folds=4, epochs_cv=8, seed=42):
    npAll = csv_preprocess(data_train_df)
    patients = npAll["Patient"].astype(str).unique()
    patients = np.array(sorted(patients))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    folds = np.array_split(patients, n_folds)

    oof_fvc_true = []
    oof_fvc_pred = []
    oof_conf_raw = []

    for k in range(n_folds):
        val_pat = set(folds[k].tolist())
        trn_np = npAll[~npAll["Patient"].astype(str).isin(val_pat)].reset_index(
            drop=True
        )
        val_np = npAll[npAll["Patient"].astype(str).isin(val_pat)].reset_index(
            drop=True
        )

        model_k = _train_model_on_npTrain(
            trn_np, device=device, epochs=epochs_cv, seed=seed + k
        )

        val_pred = make_eval_data(val_np.copy(), model_k, device=device)

        val_pred_last3 = _last3_rows_per_patient(val_pred)

        oof_fvc_true.append(
            pd.to_numeric(val_pred_last3["actual_FVC"], errors="coerce")
            .astype(float)
            .values
        )
        oof_fvc_pred.append(
            pd.to_numeric(val_pred_last3["FVC"], errors="coerce").astype(float).values
        )
        oof_conf_raw.append(
            pd.to_numeric(val_pred_last3["Confidence"], errors="coerce")
            .fillna(70.0)
            .astype(float)
            .values
        )

    fvc_true = np.concatenate(oof_fvc_true)
    fvc_pred = np.concatenate(oof_fvc_pred)
    conf_raw = np.concatenate(oof_conf_raw)

    conf_raw = np.where(np.isfinite(conf_raw), conf_raw, 70.0)
    conf_raw = np.clip(conf_raw, 70.0, None)

    grid = np.linspace(0.6, 2.6, 41)
    best_scale = 1.0
    best_score = -1e18
    for s in grid:
        m = laplace_metric_np(fvc_true, fvc_pred, conf_raw * s).mean()
        if m > best_score:
            best_score = float(m)
            best_scale = float(s)

    return best_scale, best_score




## === cell 10
weights_path = (
    "../input/new01681/Epoch1_Score6.813718921799141_Acc0.9353082271685015.pth"
)

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"

model = SIGMA()
loaded = False
if os.path.exists(weights_path):
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    loaded = True

scale = 1.0
best_score = None

if not loaded:
    best_scale_cv, best_score_cv = fit_confidence_scale_cv(
        data_train, device=device, n_folds=4, epochs_cv=8, seed=42
    )
    scale = float(best_scale_cv)
    best_score = float(best_score_cv)

    npTrain = csv_preprocess(data_train)
    model = _train_model_on_npTrain(npTrain, device=device, epochs=25, seed=123)
else:
    npTrain_cal = csv_preprocess(data_train)
    train_pred = make_eval_data(npTrain_cal.copy(), model, device=device)

    train_pred_last3 = _last3_rows_per_patient(train_pred)

    fvc_true = (
        pd.to_numeric(train_pred_last3["actual_FVC"], errors="coerce")
        .astype(float)
        .values
    )
    fvc_pred = (
        pd.to_numeric(train_pred_last3["FVC"], errors="coerce").astype(float).values
    )
    conf_raw = (
        pd.to_numeric(train_pred_last3["Confidence"], errors="coerce")
        .fillna(70.0)
        .astype(float)
        .values
    )
    conf_raw = np.where(np.isfinite(conf_raw), conf_raw, 70.0)
    conf_raw = np.clip(conf_raw, 70.0, None)

    grid = np.linspace(0.6, 2.6, 41)
    best_scale = 1.0
    best_score_in = -1e18
    for s in grid:
        m = laplace_metric_np(fvc_true, fvc_pred, conf_raw * s).mean()
        if m > best_score_in:
            best_score_in = float(m)
            best_scale = float(s)
    scale = float(best_scale)
    best_score = float(best_score_in)



## === cell 11
test_pred = make_eval_data(data_test_features.copy(), model, device=device)

conf = (
    pd.to_numeric(test_pred["Confidence"], errors="coerce")
    .fillna(70.0)
    .astype(float)
    .values
)
conf = np.where(np.isfinite(conf), conf, 70.0)
conf = np.clip(conf, 70.0, None)
conf = conf * scale
conf = np.clip(conf, 70.0, None)
test_pred["Confidence"] = conf

test_pred["Week"] = (
    pd.to_numeric(test_pred["Week"], errors="coerce").fillna(0.0).astype(float)
)
test_pred["base_Weeks"] = (
    pd.to_numeric(test_pred["base_Weeks"], errors="coerce").fillna(0.0).astype(float)
)

for nid in test_pred.Patient.unique():
    base_w = float(test_pred.loc[test_pred.Patient == nid, "base_Weeks"].iloc[0])
    idx = test_pred[
        (test_pred.Patient == nid) & (test_pred.Week == base_w)
    ].index.values
    if len(idx) > 0:
        test_pred.iloc[idx[0], test_pred.columns.get_loc("FVC")] = test_pred.iloc[
            idx[0], test_pred.columns.get_loc("base_FVC")
        ]
        test_pred.iloc[idx[0], test_pred.columns.get_loc("Confidence")] = 70.0



## === cell 12
test_pred["Patient_Week"] = test_pred["Patient_Week"].astype(str)
test_pred = test_pred.drop_duplicates(
    subset=["Patient_Week"], keep="first"
).reset_index(drop=True)

pred_map = test_pred.set_index("Patient_Week")[["FVC", "Confidence"]]

final_sub = sample_submission.copy()
final_sub["FVC"] = final_sub["Patient_Week"].map(pred_map["FVC"])
final_sub["Confidence"] = final_sub["Patient_Week"].map(pred_map["Confidence"])

final_sub["FVC"] = pd.to_numeric(final_sub["FVC"], errors="coerce").fillna(
    final_sub["FVC"].median()
)
final_sub["Confidence"] = pd.to_numeric(
    final_sub["Confidence"], errors="coerce"
).fillna(70.0)
final_sub.loc[final_sub.Confidence < 70.0, "Confidence"] = 70.0

final_sub.to_csv("submission.csv", index=False)
print(final_sub.head())
print("Confidence scale used:", scale)
print("Train metric used for scale selection (mean, last-3-per-patient):", best_score)
print("Wrote submission.csv with shape:", final_sub.shape)
print("Columns:", list(final_sub.columns))
