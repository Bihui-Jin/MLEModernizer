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

-7.053597774275135

# 6. Current score

-8.62096

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -9.19132) has done: 'I fix the pandas compatibility bug by replacing the removed `DataFrame.append` with `pd.concat`, which unblocks test feature construction. I also remove the hard dependency on a missing pretrained `.pth` file by adding a safe fallback that trains the same `SIGMA` model quickly on the provided tabular training data (same architecture/loss semantics) when the checkpoint is unavailable. I make inference deterministic and ensure confidence is positive and clipped to the competition minimum of 70, then write a valid `submission.csv` with the exact required columns. These changes are minimal, keep the core model intact, and ensure an end-to-end run that yields a valid Kaggle submission.'
- What this solution (achieved -19.09335) has done: 'Your current score (-9.19132) is worse than the target (-7.0536), so we should cautiously improve it with minimal risk while preserving your SIGMA architecture and training loop semantics. The biggest low-risk gain here is to fix a feature construction mismatch: train-time uses the “closest-to-0” visit as baseline, but test-time currently uses the single provided row (Week in test.csv) as baseline, which is inconsistent and hurts generalization. I rebuild the test expansion so that `base_Weeks/base_FVC/...` come from the original test row, and `Week` comes from `sample_submission` (instead of accidentally overwriting it during merge), keeping the same columns and inference method. I also make the sigma used in `make_eval_data` consistent with the training/inference constraints (positive and at least 70) without changing model outputs, just the post-processing used for the submission.'
- What this solution (achieved -19.09335) has done: 'We should move the score up (less negative) toward the target, and the biggest low-risk issue in your current pipeline is a subtle but severe column-mismatch in the test feature frame: you build `data_test` with `Week` at the end, but `make_eval_data()` expects `Week` before `Healthy-FVC`, so the model is effectively getting those two swapped at inference time. I fix this by reordering `data_test` columns to exactly match the feature order used in `make_eval_data()` (no change to architecture, training loop, or loss). I also enforce a consistent one-hot schema between train and test (ensure the five expected category columns exist), which prevents silent all-zero/shifted inputs when a category is missing in test. These are minimal, semantics-preserving fixes that typically yield a large score improvement without “optimizing” beyond correcting the input alignment.'
- What this solution (achieved -19.09335) has done: 'Your current score is far below the target, so we should improve it with minimal, semantics-preserving fixes. The biggest low-risk issue is that test-time one-hot encoding is built from only the test categories, which can silently differ from train and misalign feature meaning; we force test to use the same category set learned from train (same input size/order, no architecture change). Second, we compute `Healthy-FVC` and one-hots from the *baseline* row (base_FVC/base Percent/base Sex/base SmokingStatus) instead of the merged per-week row, which removes leakage/mismatch and matches how the model is trained (baseline-conditioned). These changes keep your SIGMA model, loss, and training loop intact, and only correct test feature construction to move the score upward toward the target.'
- What this solution (achieved -9.10906) has done: 'We keep your SIGMA model and training loop intact and focus on a small but high-impact correctness issue: the evaluation metric expects predictions only for the three final visits per patient, but your current pipeline predicts for *all* weeks in `sample_submission.csv`, which tends to drag the score down. We rebuild the test expansion to include only those final three `Weeks` per patient (learned from the distribution of final-3 offsets in train, then applied relative to each test patient’s baseline week), and output a submission with exactly 3 rows per patient. We also ensure the baseline row used for features is the single provided test row (as you already intended) and keep confidence clipping at 70. These changes are minimal, don’t alter the model architecture/loss, and are directly aimed at moving the score upward toward the target.'
- What this solution (achieved -9.16384) has done: 'Your current score (-9.10906) is worse than the target (-7.0536), so we should make small, low-risk improvements that keep your SIGMA model/training intact. The biggest correctness issue is that you’re submitting only 3 rows per patient, but this competition requires predictions for every `Patient_Week` in `sample_submission.csv` (weeks not scored are still required for a valid submission format). I revert to predicting for all `Patient_Week` rows (same features/model), and additionally make the feature construction consistent with training by using each patient’s baseline as the *closest-to-0 week* in test (not necessarily the single provided week), which typically improves generalization without changing architecture/loss. Finally, I set `FVC=base_FVC` and `Confidence=70` for the baseline week row per patient (Week==base_Weeks) to reduce unnecessary error and align with known baseline.'
- What this solution (achieved -9.19132) has done: 'We should move the score up (less negative) toward the target, so the safest gains come from fixing test-time feature semantics rather than changing your SIGMA architecture or training. Right now you hard-set every test patient’s `base_Weeks` to 0, which is not the actual baseline week in `test.csv` and breaks the meaning of the `base_Weeks` feature your model uses; we set `base_Weeks` to the patient’s provided `Weeks` and also keep `orig_Weeks` only for reference. Additionally, your current “baseline row fix” looks for `Week == base_Weeks`, but since `base_Weeks` was forced to 0 it often never triggers; after correcting `base_Weeks`, this correctly anchor the known baseline FVC at the provided week. These are minimal, semantics-preserving changes (no architecture/loss/loop changes) and are directly aimed at reducing error Δ for the scored weeks.'
- What this solution (achieved -9.79123) has done: 'We should move the score upward (less negative) toward the target, so the safest improvements are small correctness/calibration fixes at inference that don’t alter your SIGMA architecture or training loop. First, the `Healthy-FVC` feature is currently computed with `np.round`, while train uses Python `round` (banker’s rounding differences can shift this important feature); we match train exactly by using `round` on a float series. Second, your fallback training already uses the Laplace-like loss, but it does not clip `sigma` to the metric’s minimum (70) in the same way as evaluation; we incorporate `sigma_clipped = max(sigma, 70)` inside the loss (still the same metric form), which typically improves calibration and score without changing the model. Third, we clip predicted FVC to a reasonable physiological range based on train quantiles (a light post-process that reduces extreme errors which the metric still penalizes up to Δ=1000), while keeping confidence clipping at 70 and preserving submission formatting.'
- What this solution (achieved -8.0647) has done: 'I fix the crash in `calibrate_confidence_scale()` by ensuring the training feature frame includes a `Patient` column, since `make_eval_data()` expects it. I keep your model, loss, and inference semantics unchanged, only adjusting feature construction to add this missing identifier and to robustly handle the case where a checkpoint is loaded but confidence calibration still needs patient ids. This unblocks end-to-end execution and ensures `submission.csv` is written with the required columns and row alignment. No intentional score-degrading changes are made; this is primarily a correctness/runtime fix.'
- What this solution (achieved -8.39301) has done: 'Your current score (-8.0647) is worse than the target (-7.0536), so we should make a small, low-risk improvement that preserves your model/training exactly and only adjusts inference post-processing to better match the competition metric. The metric clips sigma at 70 and penalizes overly-large sigma via the log term, so your current confidence scaling (median-based) can easily inflate sigma and hurt score; we instead choose a confidence scale that directly maximizes the Laplace metric on the training set using your model’s own predictions (no retraining, no architecture change). Concretely, we grid-search a single global `conf_scale` on training predictions (fast) and apply it to test, keeping the existing baseline-week anchoring and sigma>=70 rule. This is minimal, deterministic, and aimed at moving the score upward toward the target.'
- What this solution (achieved -8.32974) has done: 'Your current score (-8.39301) is worse than the target (-7.0536), so we should make a small, low-risk improvement that keeps your model/training intact and only fixes a likely evaluation mismatch. The biggest correctness issue is that you’re predicting for the `Weeks` encoded in `sample_submission.csv` (1/2/3) instead of the required “final three visits” weeks relative to each test patient’s baseline week; this can strongly hurt the scored rows even if everything else is correct. I rebuild `Patient_Week` to use the three most common “final visit offsets” observed in the train set (applied as baseline_week + offset for each test patient), while still outputting a valid submission with exactly the required rows. I keep your confidence calibration and baseline-week anchoring logic unchanged, only applying it on the corrected week grid.'
- What this solution (achieved -8.39301) has done: 'Your current score (-8.32974) is below the target (-7.0536), so we should cautiously improve (increase) it with minimal changes. The biggest likely issue is that you output only 3 rows per patient, but Kaggle requires predictions for every `Patient_Week` in `sample_submission.csv`; invalid/incomplete coverage can severely hurt the score. I rebuild `data_test` to expand to *exactly* the `Patient_Week` rows in `sample_submission.csv` (using the weeks encoded there), while keeping your same feature set, model, and confidence calibration. I also keep your baseline-week anchoring, but adapt it to this full grid so the known baseline week is still set to `base_FVC` with `Confidence=70` when present.'
- What this solution (achieved -8.39301) has done: 'We keep your SIGMA model, training loop, and feature set unchanged, and focus on one minimal inference-time fix that typically improves this competition’s score: use only the baseline row’s Percent/Sex/SmokingStatus to build features (especially `Healthy-FVC` and one-hots) so they are constant per patient, matching the training construction where these come from the baseline visit. Right now, those features are computed from per-week merged rows, which can subtly drift/mismatch and degrade generalization. We also ensure the one-hot columns are created from the baseline columns explicitly, while still outputting predictions for every `Patient_Week` in `sample_submission.csv`. Everything else (confidence calibration grid-search, sigma clipping, baseline-week anchoring, file paths, submission schema) stays the same.'
- What this solution (achieved -8.62096) has done: 'Your current score (-8.39301) is below the target (-7.0536), so we should cautiously improve it with minimal, semantics-preserving fixes. The biggest likely drag is that at inference you still feed the raw `Week` value (1/2/3 from `sample_submission`) into the model, but the model was trained on real week numbers (often large positives/negatives), so this scale mismatch harms FVC prediction for the scored “final three” rows. I keep your same model/training and simply remap each test `Week` to plausible “final three visit” weeks by learning the most common last-three-week offsets from train and applying them per patient (still producing a full 1908-row submission). I also apply the same mapping during confidence-scale calibration so the chosen `conf_scale` matches the same week semantics used at test time (no retraining, just consistent post-processing).'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import pydicom  # kept (not used in this tabular-only pipeline, but preserve original imports)
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage import measure, morphology
from sklearn.preprocessing import normalize

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)




## === cell 1
def csv_preprocess(data):
    data = data.copy()

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
        for i, _patientid in enumerate(x_patientids_name):
            x_feature = x_features[i].unsqueeze(0)
            if use_cuda:
                x_feature = x_feature.cuda()
            prediction = model(x_feature)
            predictions.append(prediction.to("cpu").numpy()[0])

    predictions = np.array(predictions)
    npEval["FVC"] = predictions[:, 1]

    sigma = np.abs(predictions[:, 2] - predictions[:, 0]).astype(np.float32)
    npEval["Confidence"] = np.maximum(sigma, 70.0)

    return npEval




## === cell 5
data_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
data_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
submission = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)



## === cell 6
base = data_test.copy()
base = base.rename(columns={"Weeks": "base_Weeks", "FVC": "base_FVC"})
base = base[
    ["Patient", "base_Weeks", "base_FVC", "Percent", "Age", "Sex", "SmokingStatus"]
].copy()

sub = submission[["Patient_Week"]].copy()
sub["Patient"] = sub["Patient_Week"].str.split("_").str[0]
sub["Week"] = sub["Patient_Week"].str.split("_").str[1].astype(int)

test_expanded = sub.merge(base, on="Patient", how="left")

submission_out = test_expanded[["Patient_Week"]].copy()

data_test = test_expanded[
    [
        "Patient",
        "base_Weeks",
        "base_FVC",
        "Percent",
        "Age",
        "Sex",
        "SmokingStatus",
        "Week",
    ]
].copy()

del test_expanded, sub, base




## === cell 7
def build_category_schema_from_train(train_df: pd.DataFrame):
    schema = {}
    for col in ["Sex", "SmokingStatus"]:
        mods = list(pd.Series(train_df[col].dropna().unique()).astype(str))
        schema[col] = mods
    return schema


category_schema = build_category_schema_from_train(data_train)

data = data_test.copy()

_hfvc = (data["base_FVC"].astype(float) * 100.0) / data["Percent"].astype(float)
data["Healthy-FVC"] = _hfvc.map(lambda v: round(v) if np.isfinite(v) else 0.0).astype(
    float
)

for col in ["Sex", "SmokingStatus"]:
    mods = category_schema[col]
    for mod in mods:
        data[mod] = (data[col].astype(str) == str(mod)).astype(int)

FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
for c in FE1:
    if c not in data.columns:
        data[c] = 0

data = data.fillna(0)

data_test = data[
    [
        "Patient",
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
].copy()

del data, _hfvc




## === cell 8
def learn_common_last3_offsets(train_csv: pd.DataFrame) -> np.ndarray:
    df = train_csv.copy()
    last3 = []
    for pid, g in df.groupby("Patient", sort=False):
        g = g.sort_values("Weeks")
        base_w = float(g.loc[(g["Weeks"].abs()).idxmin(), "Weeks"])
        last_weeks = g["Weeks"].values.astype(float)[-3:]
        offsets = last_weeks - base_w
        for o in offsets:
            if np.isfinite(o):
                last3.append(float(o))
    last3 = np.array(last3, dtype=np.float32)
    if last3.size == 0:
        return np.array([0.0, 4.0, 8.0], dtype=np.float32)
    rounded = np.round(last3).astype(np.int32)
    vc = pd.Series(rounded).value_counts()
    offsets3 = np.array(sorted(vc.index[:3].tolist()), dtype=np.float32)
    if offsets3.size < 3:
        pad = [int(offsets3[-1] + 4)] if offsets3.size > 0 else [4]
        while offsets3.size < 3:
            offsets3 = np.append(offsets3, float(pad[-1]))
            pad.append(pad[-1] + 4)
    return offsets3[:3].astype(np.float32)


common_offsets = learn_common_last3_offsets(data_train)

idx_to_offset = {
    1: float(common_offsets[0]),
    2: float(common_offsets[1]),
    3: float(common_offsets[2]),
}
data_test["Week"] = data_test.apply(
    lambda r: float(r["base_Weeks"])
    + idx_to_offset.get(int(r["Week"]), float(r["base_Weeks"])),
    axis=1,
).astype(float)

print("Learned common last-3 offsets (weeks):", common_offsets.tolist())




## === cell 9
def build_train_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Healthy-FVC"] = round((df["FVC"] * 100) / df["Percent"])
    for col in ["Sex", "SmokingStatus"]:
        for mod in df[col].unique():
            df[mod] = (df[col] == mod).astype(int)

    for c in ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]:
        if c not in df.columns:
            df[c] = 0

    df = df.sort_values(["Patient", "Weeks"], ascending=True).reset_index(drop=True)

    out_rows = []
    for pid, g in df.groupby("Patient", sort=False):
        g = g.sort_values("Weeks")
        base_idx = (g["Weeks"].abs()).idxmin()
        base_row = g.loc[base_idx]

        for _, row in g.iterrows():
            out_rows.append(
                {
                    "Patient": pid,
                    "base_Weeks": float(base_row["Weeks"]),
                    "base_FVC": float(base_row["FVC"]),
                    "Age": float(base_row["Age"]),
                    "Male": int(base_row.get("Male", 0)),
                    "Female": int(base_row.get("Female", 0)),
                    "Ex-smoker": int(base_row.get("Ex-smoker", 0)),
                    "Never smoked": int(base_row.get("Never smoked", 0)),
                    "Currently smokes": int(base_row.get("Currently smokes", 0)),
                    "Week": float(row["Weeks"]),
                    "Healthy-FVC": float(base_row["Healthy-FVC"]),
                    "actual_FVC": float(row["FVC"]),
                }
            )
    return pd.DataFrame(out_rows)


def train_fallback_model(train_csv: pd.DataFrame, device: str = "cuda") -> SIGMA:
    train_df = build_train_features(train_csv)

    X = train_df[
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
    ].values.astype(np.float32)
    y = train_df[["actual_FVC"]].values.astype(np.float32)

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)

    model = SIGMA()
    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        model = model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _epoch in range(35):
        for xb, yb in dl:
            if use_cuda:
                xb = xb.cuda(non_blocking=True)
                yb = yb.cuda(non_blocking=True)

            pred = model(xb)

            sigma_raw = (pred[:, 2] - pred[:, 0]).abs()
            sigma = torch.clamp(sigma_raw, min=70.0)
            fvc_pred = pred[:, 1]
            delta = (yb[:, 0] - fvc_pred).abs().clamp(max=1000.0)

            sq2 = torch.sqrt(torch.tensor(2.0, device=xb.device))
            loss = (sq2 * delta / sigma + torch.log(sq2 * sigma)).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    return model


model = SIGMA()
ckpt_path = "../input/19-679/Epoch19_Score6.797810660608557_Acc0.93137617375677.pth"
loaded = False
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    loaded = True
else:
    model = train_fallback_model(data_train, device="cuda")
    loaded = False

test = make_eval_data(data_test.copy(), model, device="cuda")



## === cell 10
for nid in test.Patient.unique():
    index = test[(test.Patient == nid) & (test.Week == test.base_Weeks)].index.values
    if len(index) > 0:
        test.iloc[index[0], test.columns.get_loc("FVC")] = test.iloc[
            index[0], test.columns.get_loc("base_FVC")
        ]
        test.iloc[index[0], test.columns.get_loc("Confidence")] = 70




## === cell 11
def laplace_metric_np(
    fvc_true: np.ndarray, fvc_pred: np.ndarray, sigma: np.ndarray
) -> float:
    sigma_clipped = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(fvc_true - fvc_pred), 1000.0)
    return float(
        np.mean(
            -(np.sqrt(2.0) * delta) / sigma_clipped
            - np.log(np.sqrt(2.0) * sigma_clipped)
        )
    )


def build_train_final3_proxy(
    train_csv: pd.DataFrame, offsets: np.ndarray
) -> pd.DataFrame:
    df = train_csv.copy()

    base_rows = []
    for pid, g in df.groupby("Patient", sort=False):
        g = g.sort_values("Weeks")
        base_idx = (g["Weeks"].abs()).idxmin()
        base = g.loc[base_idx]
        base_rows.append(base)
    base_df = pd.DataFrame(base_rows).reset_index(drop=True)

    base_df["Healthy-FVC"] = round((base_df["FVC"] * 100) / base_df["Percent"])
    for col in ["Sex", "SmokingStatus"]:
        for mod in df[col].unique():
            base_df[mod] = (base_df[col] == mod).astype(int)
    for c in ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]:
        if c not in base_df.columns:
            base_df[c] = 0

    rows = []
    offsets = np.array(sorted(offsets.tolist()), dtype=np.float32)
    for _, r in base_df.iterrows():
        for k, off in enumerate(offsets, start=1):
            rows.append(
                {
                    "Patient": r["Patient"],
                    "base_Weeks": float(r["Weeks"]),
                    "base_FVC": float(r["FVC"]),
                    "Age": float(r["Age"]),
                    "Male": int(r.get("Male", 0)),
                    "Female": int(r.get("Female", 0)),
                    "Ex-smoker": int(r.get("Ex-smoker", 0)),
                    "Never smoked": int(r.get("Never smoked", 0)),
                    "Currently smokes": int(r.get("Currently smokes", 0)),
                    "Week": float(r["Weeks"]) + float(off),
                    "Healthy-FVC": float(r["Healthy-FVC"]),
                }
            )
    proxy = pd.DataFrame(rows)

    true_map = df[["Patient", "Weeks", "FVC"]].copy()
    true_map["Weeks"] = true_map["Weeks"].astype(float)
    true_map["FVC"] = true_map["FVC"].astype(float)

    actual = []
    for pid, g in true_map.groupby("Patient", sort=False):
        g = g.sort_values("Weeks")
        target = proxy.loc[proxy["Patient"] == pid, "Week"].values.astype(float)
        w = g["Weeks"].values.astype(float)
        f = g["FVC"].values.astype(float)
        for tw in target:
            j = int(np.argmin(np.abs(w - tw)))
            actual.append(float(f[j]))
    proxy = proxy.copy()
    proxy["actual_FVC"] = np.array(actual, dtype=np.float32)
    return proxy


def calibrate_confidence_scale(
    model: SIGMA, train_csv: pd.DataFrame, offsets: np.ndarray, device: str = "cuda"
) -> float:
    proxy_df = build_train_final3_proxy(train_csv, offsets=offsets)

    preds = make_eval_data(
        proxy_df[
            [
                "Patient",
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
                "actual_FVC",
            ]
        ].copy(),
        model,
        device=device,
    )

    fvc_true = preds["actual_FVC"].values.astype(np.float32)
    fvc_pred = preds["FVC"].values.astype(np.float32)
    sigma_base = np.maximum(preds["Confidence"].values.astype(np.float32), 70.0)

    grid = np.linspace(0.70, 1.30, 61).astype(np.float32)
    best_s = 1.0
    best_m = -1e18
    for s in grid:
        m = laplace_metric_np(fvc_true, fvc_pred, sigma_base * float(s))
        if m > best_m:
            best_m = m
            best_s = float(s)

    return float(best_s)


conf_scale = calibrate_confidence_scale(
    model, data_train, offsets=common_offsets, device="cuda"
)

test["Confidence"] = test["Confidence"].astype(float).abs()
test["Confidence"] = np.maximum(test["Confidence"] * conf_scale, 70.0)

test["FVC"] = test["FVC"].astype(float)

submission_out.loc[:, "FVC"] = test["FVC"].values
submission_out.loc[:, "Confidence"] = test["Confidence"].values
submission_out = submission_out[["Patient_Week", "FVC", "Confidence"]]
submission_out.to_csv("submission.csv", index=False)

print("Checkpoint loaded:", loaded)
print("Common offsets used:", common_offsets.tolist())
print("Confidence scale (metric-optimized on train proxy):", conf_scale)
print("Saved submission.csv with shape:", submission_out.shape)
print(submission_out.head())
