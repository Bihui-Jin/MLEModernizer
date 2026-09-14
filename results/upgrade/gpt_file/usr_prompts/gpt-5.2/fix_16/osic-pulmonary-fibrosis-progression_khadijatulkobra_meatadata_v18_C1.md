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

-6.944239230412828

# 6. Current score

-8.16886

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.24227) has done: 'I fix the immediate runtime error by replacing the removed `DataFrame.append` with `pd.concat`, preserving the exact data flow and columns. Next, I fix the missing pretrained weight file by falling back to training the same `SIGMA` model on `train.csv` using the already-defined loss, so the notebook always runs end-to-end in this Kaggle environment. I also make the inference device handling safe (CPU/GPU) and ensure confidence is clipped to the competition minimum (70) and the base-week rows are forced to use the given baseline FVC with confidence 70 as your code intended. Finally, I write a valid `submission.csv` with the required columns and row order matching `sample_submission.csv`.'
- What this solution (achieved -8.24386) has done: 'I make two minimal changes that should improve the Laplace log-likelihood score without changing your model/training core: (1) compute `Confidence` using the competition’s required clipping logic more faithfully by applying the `min(|error|,1000)` and `max(conf,70)` behavior during post-processing, and (2) calibrate the predicted confidence with a single global scaling factor estimated from the training set residuals so sigma better matches typical error magnitude. This keeps the same network, features, optimizer, epochs, and loss; it only adjusts how `Confidence` is derived at inference (which directly affects the metric) and is fast enough for the time limit. The submission format and row order remain identical to `sample_submission.csv`, and we still force baseline-week rows to use the provided baseline FVC with confidence 70 as you intended.'
- What this solution (achieved -8.1547) has done: 'Your current gap to target is about -1.30 (worse than target), so we should cautiously improve score. The metric is very sensitive to the predicted Confidence, and your model’s raw sigma (`y2-y0`) can be non-positive or too small, which hurts the Laplace log-likelihood badly; we minimally enforce a valid sigma at inference and also calibrate sigma using a metric-consistent objective. Concretely, we (1) compute a global sigma scale factor by directly maximizing the competition metric on train residuals (instead of matching means), and (2) clip confidence to be positive and at least 70 everywhere, while keeping your model, training loop, and feature pipeline unchanged. This is a small post-processing change only, expected to move the score upward toward the target without altering core learning.'
- What this solution (achieved -8.1547) has done: 'We’re currently below the target (gap ≈ -1.21; higher is better), so we should make a small, low-risk improvement. The biggest controllable lever for this metric is Confidence calibration; your current approach fits a single global scale on in-sample residuals, which tends to be over-optimistic and can hurt the leaderboard score. I keep the same model/training loop/features, but estimate the sigma scale using out-of-fold (patient-grouped) predictions so the calibration better matches generalization, then use that scale at inference. This is a minimal change, stays within the same semantics, and still writes a valid `submission.csv`.'
- What this solution (achieved -8.1547) has done: 'We’re currently below the target (higher is better), so we want a small, low-risk score lift without changing your model/training core. The most direct lever for this metric is Confidence calibration; I make your calibration truly out-of-fold by generating patient-grouped OOF predictions (instead of using in-sample preds) and then fitting the global sigma scale on those OOF residuals. This keeps the same architecture, optimizer, epochs, features, and loss; it only changes how the single scalar `scale` is estimated, which should improve generalization and move the LB score toward the target. I also make the sigma extraction used for calibration consistent with inference by enforcing the same positivity floor before scaling/clipping.'
- What this solution (achieved -8.38981) has done: 'Your current solution is already training and predicting correctly; the biggest remaining lever (without changing your model/training core) is to make the confidence calibration truly out-of-fold. Right now you compute “OOF” predictions using the final model for every fold, which is effectively in-sample and tends to miscalibrate sigma, hurting the Laplace log-likelihood. I minimally change cell 10 to train one model per fold (same architecture, optimizer, epochs, loss, and features) and generate genuine patient-grouped OOF predictions, then fit the single global `scale` from those. Everything else (including the final model trained on full data for test predictions and submission formatting) stays the same, and runtime remains within the limit given this small dataset.'
- What this solution (achieved -8.28782) has done: 'We’re below the target (current -8.38981 vs target -6.9442; higher is better), so we want a small, low-risk score lift without changing your model/training core. The most impactful lever left is confidence calibration: your OOF folds are deterministic but not randomized/group-balanced, so sigma scaling can be mis-estimated and hurt the Laplace log-likelihood. I minimally switch to a patient-grouped KFold with shuffling (still 5 folds, same per-fold training loop/epochs/optimizer/model) and fit the global `scale` on those true OOF residuals. I also apply the exact competition clipping logic during scale fitting (sigma clipped at 70, delta clipped at 1000) to better align the calibration objective with the leaderboard metric.'
- What this solution (achieved -8.1608) has done: 'We’re currently below the target (current -8.28782 vs target -6.94424; higher is better), so we want a small, low-risk improvement without changing your model/training core. The biggest controllable lever for this metric is confidence calibration: instead of picking a single global sigma scale, we also estimate a per-patient additive sigma “floor” (in ml) from out-of-fold residuals, then use `Confidence = max(70, scale*sigma_pred + floor)` at inference. This keeps your exact architecture, loss, optimizer, epochs, and feature pipeline unchanged; it only adjusts post-processing of Confidence in a metric-consistent way, which typically improves Laplace log-likelihood. We keep the baseline-week override (FVC=base_FVC, Confidence=70) and still write a valid `submission.csv` in the required order.'
- What this solution (achieved -8.16797) has done: 'We’re below the target (current -8.1608 vs target -6.9442; higher is better), so we want a small, low-risk uplift without changing your model/training core. The most direct lever is Confidence calibration: your current calibration uses a per-patient floor estimated from mean absolute residuals, which can overfit and hurt generalization; I replace it with a single global additive floor learned out-of-fold alongside the existing scale to stabilize sigma on test. I keep the same OOF fold training loop, same grid-search calibration idea, and the same inference pipeline; only the confidence post-processing formula changes to `max(70, scale*sigma_pred + floor)`. This should improve the Laplace log-likelihood by avoiding patient-specific leakage-like calibration while still preventing overly small sigmas.'
- What this solution (achieved -8.16797) has done: 'Your current score (-8.16797) is below the target (-6.94424), so we should make a small, low-risk improvement without changing the model/training core. The biggest lever for this metric is Confidence calibration: right now you optimize scale/floor on a fold-averaged metric with a fold_id that is `-1` for any unassigned rows, and you don’t explicitly apply the competition’s sigma clipping (>=70) consistently at inference until after the baseline override. I (1) make fold assignment robust and compute the calibration objective as a plain mean over true OOF rows (no fold-weight quirks), (2) fit scale/floor to maximize the exact competition metric on OOF residuals (same grid-search idea), and (3) apply the same sigma clipping (>=70) immediately after scaling/flooring before the baseline-week override, to keep calibration/inference consistent. These are minimal post-processing/calibration adjustments only; the network, features, optimizer, epochs, and loss remain identical.'
- What this solution (achieved -8.16083) has done: 'I fix the test/baseline merge logic so it no longer assumes every `Patient_Week` exists in `test.csv` (only baseline weeks do), by merging on `Patient` only and propagating each patient’s baseline fields across all requested weeks. Then I ensure the engineered columns (`base_FVC`, `base_Weeks`, one-hot flags, `Healthy-FVC`) are created for every submission row, which resolves the downstream `KeyError` in feature selection and allows inference to run. These changes are score-neutral in intent (they restore the intended data flow); the model, training loop, loss, and confidence calibration logic are kept intact. Finally, I write a valid `submission.csv` with the exact required columns and row order.'
- What this solution (achieved -8.16687) has done: 'We keep your model/training/features exactly as-is and only adjust the confidence calibration, since the metric is extremely sensitive to `Confidence` and your current OOF calibration likely over-penalizes by choosing a suboptimal global scale/floor. Specifically, we (1) make the sigma used in calibration consistent with the competition clip by applying the `max(70, ...)` inside the calibration objective, and (2) switch the calibration from a coarse grid search to a small deterministic optimization (Nelder–Mead) that directly maximizes the exact competition metric on true OOF residuals, which should give a modest but reliable uplift. Inference stays identical except it uses the newly optimized `scale` and `floor`, still applies sigma>=70, and still overrides the baseline-week row to (base_FVC, 70). The code still runs end-to-end and writes a valid `submission.csv` in the required row order.'
- What this solution (achieved -8.16886) has done: 'We’re currently below the target (current -8.16687 vs target -6.94424; higher is better), so we want a small, low-risk uplift without changing your model/training core. The metric is highly sensitive to Confidence, and our current calibration still uses only a linear (scale+floor) mapping, which can remain miscalibrated across the error range. I keep the exact same model, features, epochs, optimizer, and loss, but make calibration slightly more flexible by adding a single global exponent on the predicted sigma (power transform) and fitting (scale, floor, power) on true OOF residuals by directly maximizing the exact competition metric. Inference remains identical except it uses `Confidence = max(70, scale*(sigma_pred**power) + floor)` (and still overrides baseline week to confidence 70), which should modestly improve score while staying within the same semantics and runtime constraints.'
- What this solution (achieved -8.16886) has done: 'We’re still below the target (current -8.16886 vs target -6.94424; higher is better), so we want a small, low-risk uplift without changing your model/training core. The most direct lever is metric-aligned confidence calibration: your current power/scale/floor are fit on OOF residuals, but the optimizer is unconstrained and can drift into degenerate regions; we replace it with a tiny bounded search in a safe range and then do a short local refine, which tends to yield a better (higher) Laplace log-likelihood without changing the model. We also ensure sigma used in calibration/inference is consistent by using the same “sigma base” (pred2-pred0 floored to 1) everywhere and applying clipping at 70 inside the calibration objective. Everything else (features, network, epochs, optimizer, loss, baseline-week override, submission order) stays the same and the notebook still writes a valid `submission.csv`.'

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

    data["Healthy-FVC"] = round((data["FVC"] * 100) / data["Percent"])
    FE = []
    FE.append("Healthy-FVC")

    COLS = ["Sex", "SmokingStatus"]
    for col in COLS:
        for mod in data[col].unique():
            FE.append(mod)
            data[mod] = (data[col] == mod).astype(int)

    data = data[["Patient", "Weeks", "FVC", "Age"] + FE]
    data = data.sort_values(["Patient", "Weeks"], ascending=True).reset_index(drop=True)

    FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
    rename_col = {"Weeks": "base_Weeks", "FVC": "base_FVC"}
    data = data.rename(columns=rename_col)

    npData = pd.DataFrame(
        columns=["Patient", "base_Weeks", "base_FVC", "Age"]
        + FE1
        + ["Week", "Healthy-FVC", "actual_FVC"]
    )

    for pid in data["Patient"].unique():
        weeks = data.loc[data["Patient"] == pid].base_Weeks
        fvc = data.loc[data["Patient"] == pid].base_FVC
        index = data.loc[data["Patient"] == pid].index
        weeks.reset_index(inplace=True, drop=True)
        fvc.reset_index(inplace=True, drop=True)
        for k in range(len(weeks)):
            npData = pd.concat([npData, data.loc[data.index == index[0]]], sort=False)
            npData.iloc[-1, npData.columns.get_loc("Week")] = weeks[k]
            npData.iloc[-1, npData.columns.get_loc("actual_FVC")] = fvc[k]
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
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 502), nn.ReLU()
        )
        self.data_net3 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 118), nn.ReLU()
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
C1, C2 = torch.tensor(70, dtype=torch.float32), torch.tensor(1000, dtype=torch.float32)


def score(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    delta = (y_true[:, 0] - fvc_pred).abs()

    sq2 = torch.tensor(2.0).sqrt()
    metric = (delta / sigma) * sq2 + (sigma * sq2).log()
    return (metric).mean()


def closs(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    e = torch.abs(y_true - y_pred[:, 1])
    loss = torch.abs(sigma - e)
    return loss


def quartile_loss(y_true, y_pred):  # 0.65
    loss = score(y_true, y_pred)  # 0.35
    return loss




## === cell 4
def make_eval_data(npEval, model, device="cuda"):
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
    ]
    x_features = torch.tensor(x_features.values).float()
    x_patientids_name = npEval[["Patient"]].values

    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        model = model.to("cuda")
    model.eval()

    predictions = []
    with torch.no_grad():
        for i, patientid in enumerate(x_patientids_name):
            x_feature = x_features[i].unsqueeze(0)
            if use_cuda:
                x_feature = x_feature.cuda()
            prediction = model(x_feature)
            predictions.append(prediction.to("cpu").numpy()[0])

    predictions = np.array(predictions)  # shape = [N, 3]
    npEval["FVC"] = predictions[:, 1]

    npEval["Confidence"] = np.maximum(
        predictions[:, 2] - predictions[:, 0], 1.0
    ).astype(np.float32)
    return npEval




## === cell 5
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

DATA_DIR = "../input/osic-pulmonary-fibrosis-progression"
data_train = pd.read_csv(f"{DATA_DIR}/train.csv")
data_test = pd.read_csv(f"{DATA_DIR}/test.csv")
submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")



## === cell 6
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["Weeks"] = (
    submission["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
)
submission = submission.reset_index(drop=True)



## === cell 7
baseline = data_test.copy()
baseline = baseline.rename(columns={"Weeks": "base_Weeks", "FVC": "base_FVC"})

merge = pd.merge(
    submission[["Patient_Week", "Patient", "Weeks"]],
    baseline,
    on=["Patient"],
    how="left",
    validate="many_to_one",
)

assert (
    merge.shape[0] == submission.shape[0]
), "Row count changed after merge; alignment broken."
req_cols = ["base_FVC", "base_Weeks", "Percent", "Age", "Sex", "SmokingStatus"]
if merge[req_cols].isna().any().any():
    missing_patients = merge.loc[merge[req_cols].isna().any(axis=1), "Patient"].unique()
    raise ValueError(f"Missing baseline fields for patients: {missing_patients}")

merge["Week"] = merge["Weeks"].astype(int)

data_test = merge.loc[
    :,
    [
        "Patient",
        "base_Weeks",
        "base_FVC",
        "Percent",
        "Age",
        "Sex",
        "SmokingStatus",
        "Week",
        "Patient_Week",
    ],
].copy()

submission = merge.loc[:, ["Patient_Week"]].copy()
submission["FVC"] = 0.0
submission["Confidence"] = 0.0



## === cell 8
data = data_test.copy()
data["Healthy-FVC"] = round((data["base_FVC"] * 100) / data["Percent"])
FE = []
FE.append("Healthy-FVC")

COLS = ["Sex", "SmokingStatus"]
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)

FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
npData = pd.DataFrame(
    columns=["Patient", "base_Weeks", "base_FVC", "Age", "Healthy-FVC"]
    + FE1
    + ["Week", "Patient_Week"]
)

npData = pd.concat([npData, data], ignore_index=True, sort=True)
npData = npData.fillna(0)

data_test = npData[
    [
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
        "Patient_Week",
    ]
].copy()
del npData, data



## === cell 9
train_np = csv_preprocess(data_train.copy())

x_train = (
    train_np[
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
y_train = train_np[["actual_FVC"]].astype(np.float32).values

x_train_t = torch.tensor(x_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

train_ds = TensorDataset(x_train_t, y_train_t)
train_loader = DataLoader(
    train_ds, batch_size=256, shuffle=True, num_workers=0, drop_last=False
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SIGMA().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(60):
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = closs(yb, pred).mean()
        loss.backward()
        optimizer.step()



## === cell 10
from sklearn.model_selection import GroupKFold, KFold
from scipy.optimize import minimize

groups = train_np["Patient"].values
n_folds = 5

n_unique_patients = len(pd.unique(groups))
use_group = n_unique_patients >= n_folds

if use_group:
    splitter = GroupKFold(n_splits=n_folds)
    split_iter = list(splitter.split(x_train, y_train.reshape(-1), groups=groups))
else:
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    split_iter = list(splitter.split(x_train, y_train.reshape(-1)))

fold_id = np.full(x_train.shape[0], -1, dtype=np.int32)
for k, (_, idx_val) in enumerate(split_iter):
    fold_id[idx_val] = k

pred_oof = np.zeros((x_train.shape[0], 3), dtype=np.float32)

for k, (idx_trn, idx_val) in enumerate(split_iter):
    if idx_val.size == 0 or idx_trn.size == 0:
        continue

    seed_everything(42 + k)

    fold_model = SIGMA().to(device)
    fold_optimizer = torch.optim.Adam(fold_model.parameters(), lr=1e-3)

    fold_ds = TensorDataset(x_train_t[idx_trn], y_train_t[idx_trn])
    fold_loader = DataLoader(
        fold_ds, batch_size=256, shuffle=True, num_workers=0, drop_last=False
    )

    fold_model.train()
    for epoch in range(60):
        for xb, yb in fold_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            fold_optimizer.zero_grad(set_to_none=True)
            pred = fold_model(xb)
            loss = closs(yb, pred).mean()
            loss.backward()
            fold_optimizer.step()

    fold_model.eval()
    with torch.no_grad():
        xb_val = x_train_t[idx_val].to(device)
        pred_k = fold_model(xb_val).detach().cpu().numpy().astype(np.float32)
        pred_oof[idx_val] = pred_k

fvc_pred_all = pred_oof[:, 1].astype(np.float32)
sigma_pred_all = (pred_oof[:, 2] - pred_oof[:, 0]).astype(np.float32)

y_true_all = y_train.reshape(-1).astype(np.float32)
delta_all = np.abs(y_true_all - fvc_pred_all).astype(np.float32)
delta_all = np.minimum(delta_all, 1000.0)

sigma_base_all = np.maximum(sigma_pred_all, 1.0).astype(np.float32)

oof_mask = fold_id >= 0
if not np.any(oof_mask):
    oof_mask = np.ones_like(delta_all, dtype=bool)

delta_oof = delta_all[oof_mask]
sigma_base_oof = sigma_base_all[oof_mask]

global_floor = float(np.mean(delta_oof))


def mean_metric_for_params_oof(scale: float, floor: float, power: float) -> float:
    p = np.float32(power)
    p = np.clip(p, np.float32(0.5), np.float32(2.0))
    sigma = (sigma_base_oof**p) * np.float32(scale) + np.float32(floor)
    sigma = np.maximum(sigma, 70.0)  # competition clip
    metric = -(np.sqrt(2.0) * delta_oof) / sigma - np.log(np.sqrt(2.0) * sigma)
    return float(np.mean(metric))


scale_grid = np.array([0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5], dtype=np.float32)
floor_grid = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 200.0], dtype=np.float32)
power_grid = np.array([0.7, 0.85, 1.0, 1.15, 1.3], dtype=np.float32)

best = (-1e18, 1.0, min(150.0, global_floor), 1.0)
for s in scale_grid:
    for fl in floor_grid:
        for pw in power_grid:
            m = mean_metric_for_params_oof(float(s), float(fl), float(pw))
            if m > best[0]:
                best = (m, float(s), float(fl), float(pw))


def objective(x):
    s = float(max(1e-6, x[0]))
    fl = float(max(0.0, x[1]))
    pw = float(x[2])
    return -mean_metric_for_params_oof(s, fl, pw)


x0 = np.array([best[1], best[2], best[3]], dtype=np.float64)
res = minimize(
    objective,
    x0=x0,
    method="Nelder-Mead",
    options={"maxiter": 250, "xatol": 1e-5, "fatol": 1e-7, "disp": False},
)

scale = float(max(1e-6, res.x[0]))
floor = float(max(0.0, res.x[1]))
power = float(np.clip(res.x[2], 0.5, 2.0))



## === cell 11
test = make_eval_data(data_test.copy(), model, device=device)

test_sigma = np.maximum(test["Confidence"].values.astype(np.float32), 1.0).astype(
    np.float32
)
test_sigma = test_sigma ** np.float32(power)

test["Confidence"] = test_sigma * np.float32(scale) + np.float32(floor)
test["Confidence"] = np.maximum(
    test["Confidence"].values.astype(np.float32), 70.0
).astype(np.float32)

for nid in test.Patient.unique():
    index = test[(test.Patient == nid) & (test.Week == test.base_Weeks)].index.values
    if len(index) > 0:
        test.iloc[index[0], test.columns.get_loc("FVC")] = test.iloc[
            index[0], test.columns.get_loc("base_FVC")
        ]
        test.iloc[index[0], test.columns.get_loc("Confidence")] = 70.0



## === cell 12
pred_df = test.loc[:, ["Patient_Week", "FVC", "Confidence"]].copy()
submission = submission.drop(columns=["FVC", "Confidence"]).merge(
    pred_df, on="Patient_Week", how="left", validate="one_to_one"
)

submission = submission[["Patient_Week", "FVC", "Confidence"]]
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(
    "Confidence calibration params used: scale =",
    scale,
    "floor =",
    floor,
    "power =",
    power,
)
print(
    "OOF metric (using calibrated sigma) =",
    mean_metric_for_params_oof(scale, floor, power),
)
print(submission.head())
