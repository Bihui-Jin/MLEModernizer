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

-7.072598860307702

# 6. Current score

-14.9683

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -14.9683) has done: 'I fix the pandas `DataFrame.append` deprecation by switching to `pd.concat`, which unblocks preprocessing on pandas 2.2+. I also fix the missing pretrained weights path by making the code robust: if the `.pth` file is not present in `/kaggle/input`, the script train the same `SIGMA` model on the provided `train.csv` (no CT features) using the existing loss function, then run inference. Finally, I ensure predictions are aligned to `sample_submission.csv` order and enforce `Confidence >= 70`, producing a valid `submission.csv` with the required columns.'
- What this solution (achieved -14.9683) has done: 'Your current score is far below the target (gap ≈ -7.90; higher is better), so we should make small, low-risk fixes that better match the competition metric without changing the model architecture or training loop structure. The biggest issue is that the loss/metric implementation is missing the required clipping (`sigma>=70`, `delta<=1000`) and is optimizing the wrong sign (you currently minimize something different from the competition’s negative log-likelihood). I minimally correct `score()/quartile_loss()` to exactly implement the competition’s formulation (as a minimization objective) and ensure inference uses the same `sigma` clipping. I also keep determinism settings and submission alignment intact, and still write a valid `submission.csv`.'
- What this solution (achieved -14.9683) has done: 'Your current score (-14.9683) is far below the target (-7.0726), so we should improve (increase) it with minimal, low-risk changes that preserve your model/training loop. The biggest issue is that the network is forced to output non-negative FVC due to a final `ReLU()`, and you also train/predict raw FVC without anchoring to the known baseline FVC at week 0, which is a strong signal in this competition. I keep the exact same architecture and training loop, but post-process predictions by predicting an FVC *delta* relative to `base_FVC` (i.e., `FVC_pred = base_FVC + delta`), and I train the same way by converting the target to a delta as well—this avoids changing the model while making the output scale easier and consistent with the data. I also ensure the output is aligned and `Confidence>=70` as before, producing a valid `submission.csv`.'
- What this solution (achieved -14.9683) has done: 'We should move the score up (less negative) toward the target by improving calibration against the Laplace log-likelihood while keeping your exact model and training loop intact. The smallest high-impact fix is to stop treating the model’s second output as an unconstrained “delta FVC” with a final ReLU (which can only increase FVC) and instead interpret it as a non-negative magnitude with a learned direction based on whether the target delta is negative or positive; this preserves architecture and loss but allows decreases in FVC as in the real data. Concretely, we train on `abs(delta)` and multiply predictions by `sign(delta)` at train/infer time, so the model can represent both decline and improvement without changing layers. Additionally, we use the same sigma clipping (>=70) consistently and compute `Healthy-FVC` robustly to avoid inf/NaNs that can destabilize training.'
- What this solution (achieved -14.9683) has done: 'Your current score (-14.9683) is much worse than the target (-7.0726), so we should increase it with minimal, low-risk fixes that better align training/inference with the evaluation metric while preserving your model and training loop. The biggest bug is that you train the model to predict `abs(delta)` but your inference applies a heuristic sign based on week direction; this mismatches the training target and systematically bias predictions. I keep the same architecture and loss, but remove the sign heuristic and instead train/predict the true signed delta (still letting the final ReLU enforce non-negative outputs by shifting the delta target to be non-negative and undoing the shift at inference). I also make the sigma floor consistent (use 70 everywhere, not 1e-3 during training), which improves metric-calibration without changing the core approach.'
- What this solution (achieved -14.9683) has done: 'We need to move the score up (less negative) toward the target; your current gap is about -7.90, so small fixes that better match the evaluation while preserving your model/loop are appropriate. The biggest low-risk improvement is to train the model on the same target you actually score: the Laplace log-likelihood expects an FVC prediction, but the current training uses a shifted delta target while the loss compares that shifted delta to the model’s second output as if it were an FVC prediction (target/units mismatch). I keep the exact architecture and training loop, but change the loss to compute the metric on reconstructed `FVC_pred = base_FVC + (delta_pred - shift)` and `FVC_true = base_FVC + (y_shifted - shift)`; this is purely a consistent reparameterization, not a new approach. I also clamp `Confidence` during training the same way as in inference and ensure tensors are on the correct device to avoid silent CPU/GPU mismatches.'
- What this solution (achieved -14.9683) has done: 'Your score is far below the target (gap ≈ -7.90, higher is better), so we should improve it with the smallest changes that directly better match the evaluation. The biggest issue is the model’s final `ReLU()` forces the second output (your shifted-delta head) to be non-negative, which cannot represent FVC decline; this is a fundamental mismatch with the task and heavily hurts score. Without changing the architecture, we can keep the same two-head output but reinterpret the second head as a *magnitude* and supply the *direction* (sign) from a per-patient linear trend fit on the training history, then apply that sign at inference (and also use it consistently for training targets). This preserves the training loop and loss structure while allowing negative deltas and typically moves the score substantially toward the target.'
- What this solution (achieved -14.9683) has done: 'I fix the immediate runtime error in `build_percent_to_slope_sign_model()` caused by using `groupby.agg()` with functions that return arrays (pandas now requires scalar aggregation), by rewriting it to iterate groups and compute slope/signs safely. This also unblock downstream variables (`percent_sign_model`, `pth_path`) so cells 10–12 run, train/load the model, and generate predictions. I additionally make the sign mapping robust for test patients (which don’t exist in train) by using the fitted Percent→sign model as intended, without changing the SIGMA architecture or the training loop structure. Finally, I ensure the submission is fully populated and written as `submission.csv` with the required columns and `Confidence >= 70`.'
- What this solution (achieved -14.9683) has done: 'We need to move the score up (less negative) toward the target, and the biggest low-risk issue in your current pipeline is that the model is still effectively constrained to predict only one global “direction” per patient (via a hard sign map), which mismatches reality because FVC trajectories can be non-linear and patient-specific over the requested weeks. Without changing the SIGMA architecture or the training loop structure, we can make the sign determination *week-relative* by learning a simple per-patient linear trend (slope) from the training history and using it to set the sign for each prediction week (relative to baseline) instead of a fixed sign per patient. This keeps the same “magnitude head + ReLU” core logic, but makes the reconstructed signed delta consistent with whether the requested week is before/after baseline and the patient’s learned slope, typically improving the Laplace log-likelihood noticeably. We also make the training target sign consistent with this week-relative rule (same tensors, same optimizer/epochs), and keep Confidence clipping and submission alignment unchanged.'
- What this solution (achieved -14.9683) has done: 'Your current score (-14.9683) is much worse than the target (-7.0726), so we should increase it with minimal, low-risk fixes that better align predictions to strong signals already present in the tabular data (without changing the SIGMA architecture or the training loop structure). The biggest issue is the “magnitude + externally-chosen sign” scheme: it can systematically choose the wrong direction per patient/week and it throws away the strongest per-patient signal (their observed FVC history / baseline anchor). I keep your exact SIGMA network and training loop, but change only the target parameterization to a strictly-nonnegative “distance from baseline” (absolute delta) and reconstruct final FVC at inference using a per-patient linear trend fitted from train (or Percent→trend for test), which is still the same approach you already started (slope_map) but used only for sign. Finally, I set Confidence using the empirical residual scale from training (still clipped to >=70) to improve calibration for the Laplace metric without altering model training.'
- What this solution (achieved -14.9683) has done: 'Your current score (-14.9683) is far below the target (-7.0726), so we should increase it with the smallest changes that directly align predictions with strong signals already available. The biggest low-risk gain is to use the per-patient linear trend we already compute (from Percent→slope) not only as a sign heuristic but as the actual FVC forecast (`base_FVC + slope*(week-base_week)`), and use the neural net mainly to provide a nonnegative uncertainty (sigma) rather than an extra magnitude term that can push FVC the wrong way. This preserves the same model, loss function, and training loop structure (we just change what we feed into the loss and what we output at inference), and keeps the required `Confidence>=70` clipping and submission alignment. As a safety measure, we also clip predicted FVC into a physiologically reasonable range to avoid extreme penalties (this affects only outliers and tends to improve Laplace LL).'
- What this solution (achieved -14.9683) has done: 'Your score is far below the target (gap ≈ -7.90; higher is better), so the smallest safe improvement is to fix the main modeling bug: the network is effectively ignored for FVC because training always uses a hardcoded `slope_const=-1.0`, and inference uses only the external slope map. I keep the same SIGMA architecture and training loop structure, but (1) train the model to predict the per-row residual around the external linear baseline (still nonnegative due to the existing final ReLU, so we predict absolute residual), and (2) add that learned residual back to the baseline at inference; this preserves the same feature set and loss form while actually using the model. I also compute the baseline slope per training row from the patient’s fitted slope map (instead of a constant), and keep `Confidence>=70` plus submission alignment unchanged so it still writes a valid `submission.csv`. These changes should materially move the score upward toward the target without changing the core approach (baseline linear model + SIGMA for uncertainty/adjustment).'
- What this solution (achieved -14.9683) has done: 'I remove the intentional `RuntimeError` that currently stops execution, because the actual slopes-aware training loop is already implemented in the next cell and the crash prevents any submission from being written. I keep the core SIGMA model, feature set, loss/metric, and training loop semantics unchanged, and simply make cell 10 a no-op when training is handled later (or when pretrained weights exist). I also ensure the inference/training use the same sigma floor (>=70) and that the final `submission.csv` is fully populated and aligned to `sample_submission.csv` ordering as your code already intends. These changes are score-positive only insofar as they restore the intended pipeline and avoid the crash.'
- What this solution (achieved -14.9683) has done: 'Your current score is far below the target (gap ≈ -7.90; higher is better), so we should make the smallest changes that better match the competition’s evaluation without changing your SIGMA model or training loop structure. The main issue is your residual head is trained on an absolute residual but then always *added* to the baseline, which systematically overpredicts FVC when the true residual is negative; we can preserve the non-negative residual head by applying a per-row sign derived from the per-patient slope and whether the week is before/after baseline. Second, your test-time slope from Percent can be extreme for out-of-range Percents; lightly clipping the predicted slopes to the empirical train slope range is a safe, minimal stabilization that typically improves Laplace LL. These two changes keep the same architecture, loss, optimizer, epochs, and features, but make predictions consistent with the intended “baseline linear trend + residual correction” semantics and should move the score upward toward the target band.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from tqdm.auto import tqdm




## === cell 1
def csv_preprocess(data):
    data = data.copy()

    percent = data["Percent"].replace(0, np.nan).astype(np.float32)
    data["Healthy-FVC"] = (
        np.rint((data["FVC"].astype(np.float32) * 100.0) / percent)
        .fillna(0)
        .astype(np.float32)
    )

    FE = ["Healthy-FVC"]

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
            npData = pd.concat(
                [npData, data.loc[data.index == index[0]]],
                ignore_index=True,
                sort=False,
            )
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
            nn.Linear(64, 2),
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
C1 = torch.tensor(70.0, dtype=torch.float32)
C2 = torch.tensor(1000.0, dtype=torch.float32)


def competition_metric(y_true_fvc, y_pred):
    """
    Returns the mean competition metric (higher is better, negative values).
    y_true_fvc: shape [N, 1] or [N]
    y_pred: shape [N, 2] where [:,0]=sigma, [:,1]=fvc_pred
    """
    if y_true_fvc.ndim == 2:
        y_true_fvc = y_true_fvc[:, 0]
    sigma = y_pred[:, 0].clamp_min(C1.to(y_pred.device))
    fvc_pred = y_pred[:, 1]
    delta = (y_true_fvc - fvc_pred).abs().clamp_max(C2.to(y_pred.device))
    sq2 = torch.sqrt(torch.tensor(2.0, device=y_pred.device))
    metric = -(sq2 * delta / sigma) - torch.log(sq2 * sigma)
    return metric.mean()


def quartile_loss(y_true_fvc, y_pred):
    return -competition_metric(y_true_fvc, y_pred)




## === cell 4
def make_eval_data(npEval, model, slope_map, device="cuda", conf_value=70.0):
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

    if torch.cuda.is_available() and device == "cuda":
        model.to("cuda")
    model.eval()

    preds_sigma = []
    preds_resid = []
    for i in range(len(npEval)):
        x_feature = x_features[i].unsqueeze(0)
        if torch.cuda.is_available() and device == "cuda":
            x_feature = x_feature.cuda()
        with torch.no_grad():
            prediction = model(x_feature)
            sigma = prediction[:, 0:1].clamp_min(C1.to(prediction.device))
            resid = prediction[:, 1:2]
        preds_sigma.append(float(sigma.detach().cpu().numpy()[0, 0]))
        preds_resid.append(float(resid.detach().cpu().numpy()[0, 0]))

    preds_sigma = np.asarray(preds_sigma, dtype=np.float32)
    preds_resid = np.asarray(preds_resid, dtype=np.float32)

    base_fvc = npEval["base_FVC"].values.astype(np.float32)
    base_w = npEval["base_Weeks"].values.astype(np.float32)
    wk = npEval["Week"].values.astype(np.float32)
    rel = wk - base_w

    slopes = np.asarray(
        [float(slope_map.get(p, -1.0)) for p in npEval["Patient"].values],
        dtype=np.float32,
    )

    sign = np.sign(slopes * rel).astype(np.float32)
    sign[sign == 0] = 1.0  # baseline week or flat slope -> do not flip residual
    resid_signed = preds_resid * sign

    fvc_pred = base_fvc + slopes * rel + resid_signed
    fvc_pred = np.clip(fvc_pred, 500.0, 6000.0)

    npEval["FVC"] = fvc_pred.astype(np.float32)
    npEval["Confidence"] = np.maximum(preds_sigma, np.float32(conf_value)).astype(
        np.float32
    )
    return npEval




## === cell 5
DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

data_train = pd.read_csv(train_path)
data_test = pd.read_csv(test_path)
submission = pd.read_csv(sub_path)



## === cell 6
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["Weeks"] = submission["Patient_Week"].apply(lambda x: int(x.split("_")[1]))
submission_sorted = submission.sort_values(
    by=["Patient", "Weeks"], ascending=True
).reset_index(drop=True)



## === cell 7
merge = (
    pd.merge(data_test, submission_sorted, on=["Patient"], how="left")
    .sort_values(["Patient", "Weeks_y"])
    .reset_index(drop=True)
)
merge = merge.drop(["FVC_y"], axis=1)
merge = merge.rename(
    columns={"FVC_x": "base_FVC", "Weeks_y": "Week", "Weeks_x": "base_Weeks"}
)

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
    ],
]

submission_out = merge.loc[:, ["Patient_Week", "base_FVC", "Confidence"]].rename(
    columns={"base_FVC": "FVC"}
)



## === cell 8
data = data_test.copy()

percent = data["Percent"].replace(0, np.nan).astype(np.float32)
data["Healthy-FVC"] = (
    np.rint((data["base_FVC"].astype(np.float32) * 100.0) / percent)
    .fillna(0)
    .astype(np.float32)
)

FE = ["Healthy-FVC"]

COLS = ["Sex", "SmokingStatus"]
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)

FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
npData = pd.DataFrame(
    columns=["Patient", "base_Weeks", "base_FVC", "Age", "Healthy-FVC"] + FE1 + ["Week"]
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
    ]
].copy()




## === cell 9
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def find_pretrained_pth():
    candidate = (
        "/kaggle/input/w-2-677/Epoch2_Score6.776965535641716_Acc0.9368517426624738.pth"
    )
    if os.path.exists(candidate):
        return candidate

    for root, _, files in os.walk("/kaggle/input"):
        for fn in files:
            if fn.endswith(".pth") and "Epoch2" in fn and "Score" in fn:
                return os.path.join(root, fn)
    return None


def build_slope_map_from_train(df_train_raw):
    slope_map = {}
    for pid, g in df_train_raw.groupby("Patient"):
        g = g.sort_values("Weeks")
        if len(g) < 2:
            slope_map[pid] = -1.0
            continue
        w = g["Weeks"].astype(np.float32).values
        f = g["FVC"].astype(np.float32).values
        w0 = w - w.mean()
        denom = float(np.dot(w0, w0))
        if denom <= 1e-6:
            slope_map[pid] = -1.0
            continue
        slope = float(np.dot(w0, f - f.mean()) / denom)  # ml per week
        slope_map[pid] = slope
    return slope_map


def build_percent_to_slope_model(df_train_raw):
    percents = []
    slopes = []

    for pid, g in df_train_raw.groupby("Patient"):
        g = g.sort_values("Weeks")
        if len(g) < 2:
            continue

        w = g["Weeks"].astype(np.float32).values
        f = g["FVC"].astype(np.float32).values

        w0 = w - w.mean()
        denom = float(np.dot(w0, w0))
        if denom <= 1e-6:
            continue

        slope = float(np.dot(w0, f - f.mean()) / denom)
        perc = float(g["Percent"].iloc[0]) if "Percent" in g.columns else np.nan
        if not np.isfinite(perc):
            continue

        percents.append(perc)
        slopes.append(slope)

    percents = np.asarray(percents, dtype=np.float32)
    slopes = np.asarray(slopes, dtype=np.float32)

    if len(percents) < 10:
        return {
            "mode": "const",
            "slope": float(np.median(slopes)) if len(slopes) else -1.0,
        }

    x = percents.reshape(-1, 1)
    y = slopes
    X = np.concatenate([np.ones((len(x), 1), dtype=np.float32), x], axis=1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {
        "mode": "linear",
        "beta0": float(beta[0]),
        "beta1": float(beta[1]),
        "fallback": -1.0,
    }


def predict_slope_from_percent(model_info, percent_value):
    if not np.isfinite(percent_value):
        return float(model_info.get("fallback", -1.0))
    if model_info["mode"] == "const":
        return float(model_info["slope"])
    return float(model_info["beta0"] + model_info["beta1"] * float(percent_value))


slope_map_train = build_slope_map_from_train(data_train)
percent_slope_model = build_percent_to_slope_model(data_train)

pth_path = find_pretrained_pth()
pth_path




## === cell 10
def build_train_tensors_residual_abs(df_train_raw, slope_map):
    df = csv_preprocess(df_train_raw)

    x = (
        df[
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

    base_fvc = df["base_FVC"].astype(np.float32).values
    base_w = df["base_Weeks"].astype(np.float32).values
    wk = df["Week"].astype(np.float32).values
    rel = wk - base_w

    slopes = np.asarray(
        [float(slope_map.get(p, -1.0)) for p in df["Patient"].values],
        dtype=np.float32,
    )

    fvc_true = df["actual_FVC"].astype(np.float32).values
    fvc_base = base_fvc + slopes * rel
    resid_abs = np.abs(fvc_true - fvc_base).astype(np.float32).reshape(-1, 1)

    return (
        torch.tensor(x),
        torch.tensor(resid_abs),
        torch.tensor(fvc_true.reshape(-1, 1)),
    )


def estimate_conf_from_train_residuals(df_train_raw, slope_map, min_conf=70.0):
    res = []
    for pid, g in df_train_raw.groupby("Patient"):
        g = g.sort_values("Weeks")
        if len(g) < 2:
            continue
        slope = float(slope_map.get(pid, -1.0))
        w = g["Weeks"].astype(np.float32).values
        f = g["FVC"].astype(np.float32).values
        w0 = float(w[0])
        f0 = float(f[0])
        pred = f0 + slope * (w - w0)
        res.append(np.abs(f - pred))
    if not res:
        return float(min_conf)
    res = np.concatenate([r.reshape(-1) for r in res]).astype(np.float32)
    return float(max(min_conf, float(np.median(res))))


device = "cuda" if torch.cuda.is_available() else "cpu"
model = SIGMA().to(device)

if pth_path is not None:
    state = torch.load(pth_path, map_location=device)
    model.load_state_dict(state)



## === cell 11
if pth_path is None:
    df_train_proc = csv_preprocess(data_train)
    x_train = (
        df_train_proc[
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

    base_fvc = df_train_proc["base_FVC"].astype(np.float32).values
    base_w = df_train_proc["base_Weeks"].astype(np.float32).values
    wk = df_train_proc["Week"].astype(np.float32).values
    rel = wk - base_w

    slopes = np.asarray(
        [float(slope_map_train.get(p, -1.0)) for p in df_train_proc["Patient"].values],
        dtype=np.float32,
    )

    fvc_true = df_train_proc["actual_FVC"].astype(np.float32).values
    fvc_base = base_fvc + slopes * rel
    y_resid_abs = np.abs(fvc_true - fvc_base).astype(np.float32)

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_resid_abs_t = torch.tensor(y_resid_abs, dtype=torch.float32).view(-1, 1)
    y_fvc_true_t = torch.tensor(fvc_true, dtype=torch.float32).view(-1, 1)
    slopes_t = torch.tensor(slopes, dtype=torch.float32).view(-1, 1)

    dataset = torch.utils.data.TensorDataset(
        x_train_t, y_resid_abs_t, y_fvc_true_t, slopes_t
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=0
    )

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(25):
        for xb, yb_resid_abs, yb_fvc, sb in loader:
            xb = xb.to(device)
            yb_resid_abs = yb_resid_abs.to(device).view(-1)
            yb_fvc = yb_fvc.to(device).view(-1)
            sb = sb.to(device).view(-1)

            pred = model(xb)
            sigma = pred[:, 0].clamp_min(C1.to(device))
            resid_pred = pred[:, 1]

            base_fvc_b = xb[:, 1]
            base_w_b = xb[:, 0]
            wk_b = xb[:, 8]
            rel_b = wk_b - base_w_b

            sign_b = torch.sign(sb * rel_b)
            sign_b = torch.where(sign_b == 0, torch.ones_like(sign_b), sign_b)

            fvc_base_b = base_fvc_b + sb * rel_b
            fvc_pred = fvc_base_b + resid_pred * sign_b

            loss = quartile_loss(yb_fvc, torch.stack([sigma, fvc_pred], dim=1))

            optim.zero_grad()
            loss.backward()
            optim.step()

test_base = pd.read_csv(test_path).copy()

train_slopes = np.asarray(list(slope_map_train.values()), dtype=np.float32)
lo, hi = float(np.percentile(train_slopes, 1)), float(np.percentile(train_slopes, 99))

slope_map_test = {}
for pid, g in test_base.groupby("Patient"):
    perc = float(g["Percent"].iloc[0])
    s = float(predict_slope_from_percent(percent_slope_model, perc))
    s = float(np.clip(s, lo, hi))
    slope_map_test[pid] = s

conf_const = estimate_conf_from_train_residuals(
    data_train, slope_map_train, min_conf=70.0
)

test_pred = make_eval_data(
    data_test.copy(),
    model,
    slope_map=slope_map_test,
    device=device,
    conf_value=conf_const,
)

for nid in test_pred.Patient.unique():
    idx = test_pred[
        (test_pred.Patient == nid) & (test_pred.Week == test_pred.base_Weeks)
    ].index.values
    if len(idx) > 0:
        test_pred.iloc[idx[0], test_pred.columns.get_loc("FVC")] = test_pred.iloc[
            idx[0], test_pred.columns.get_loc("base_FVC")
        ]
        test_pred.iloc[idx[0], test_pred.columns.get_loc("Confidence")] = 70

test_pred.loc[test_pred.Confidence < 70, "Confidence"] = 70

pred_map = test_pred.set_index(["Patient", "Week"])[["FVC", "Confidence"]]
sub_final = submission.copy()
sub_final["Patient"] = sub_final["Patient_Week"].apply(lambda x: x.split("_")[0])
sub_final["Weeks"] = sub_final["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

sub_final = sub_final.join(pred_map, on=["Patient", "Weeks"], rsuffix="_pred")

sub_final["FVC"] = sub_final["FVC"].fillna(2000)
sub_final["Confidence"] = sub_final["Confidence"].fillna(70)
sub_final.loc[sub_final.Confidence < 70, "Confidence"] = 70

sub_final = sub_final[["Patient_Week", "FVC", "Confidence"]]
sub_final.to_csv("submission.csv", index=False)

print(sub_final.head())
print("Wrote submission.csv with shape:", sub_final.shape)
print("Saved to:", os.path.abspath("submission.csv"))
