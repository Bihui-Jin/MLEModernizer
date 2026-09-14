# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -14.9683) has done: 'I fix the pandas `DataFrame.append` deprecation by switching to `pd.concat`, which unblocks preprocessing on pandas 2.2+. I also fix the missing pretrained weights path by making the code robust: if the `.pth` file is not present in `/kaggle/input`, the script train the same `SIGMA` model on the provided `train.csv` (no CT features) using the existing loss function, then run inference. Finally, I ensure predictions are aligned to `sample_submission.csv` order and enforce `Confidence >= 70`, producing a valid `submission.csv` with the required columns.'
- What this solution (achieved -14.9683) has done: 'Your current score is far below the target (gap ≈ -7.90; higher is better), so we should make small, low-risk fixes that better match the competition metric without changing the model architecture or training loop structure. The biggest issue is that the loss/metric implementation is missing the required clipping (`sigma>=70`, `delta<=1000`) and is optimizing the wrong sign (you currently minimize something different from the competition’s negative log-likelihood). I minimally correct `score()/quartile_loss()` to exactly implement the competition’s formulation (as a minimization objective) and ensure inference uses the same `sigma` clipping. I also keep determinism settings and submission alignment intact, and still write a valid `submission.csv`.'
- What this solution (achieved -14.9683) has done: 'Your current score (-14.9683) is far below the target (-7.0726), so we should improve (increase) it with minimal, low-risk changes that preserve your model/training loop. The biggest issue is that the network is forced to output non-negative FVC due to a final `ReLU()`, and you also train/predict raw FVC without anchoring to the known baseline FVC at week 0, which is a strong signal in this competition. I keep the exact same architecture and training loop, but post-process predictions by predicting an FVC *delta* relative to `base_FVC` (i.e., `FVC_pred = base_FVC + delta`), and I train the same way by converting the target to a delta as well—this avoids changing the model while making the output scale easier and consistent with the data. I also ensure the output is aligned and `Confidence>=70` as before, producing a valid `submission.csv`.'
- What this solution (achieved -14.9683) has done: 'We should move the score up (less negative) toward the target by improving calibration against the Laplace log-likelihood while keeping your exact model and training loop intact. The smallest high-impact fix is to stop treating the model’s second output as an unconstrained “delta FVC” with a final ReLU (which can only increase FVC) and instead interpret it as a non-negative magnitude with a learned direction based on whether the target delta is negative or positive; this preserves architecture and loss but allows decreases in FVC as in the real data. Concretely, we train on `abs(delta)` and multiply predictions by `sign(delta)` at train/infer time, so the model can represent both decline and improvement without changing layers. Additionally, we use the same sigma clipping (>=70) consistently and compute `Healthy-FVC` robustly to avoid inf/NaNs that can destabilize training.'
- What this solution (achieved -14.9683) has done: 'Your current score (-14.9683) is much worse than the target (-7.0726), so we should increase it with minimal, low-risk fixes that better align training/inference with the evaluation metric while preserving your model and training loop. The biggest bug is that you train the model to predict `abs(delta)` but your inference applies a heuristic sign based on week direction; this mismatches the training target and systematically bias predictions. I keep the same architecture and loss, but remove the sign heuristic and instead train/predict the true signed delta (still letting the final ReLU enforce non-negative outputs by shifting the delta target to be non-negative and undoing the shift at inference). I also make the sigma floor consistent (use 70 everywhere, not 1e-3 during training), which improves metric-calibration without changing the core approach.'
- What this solution (achieved -14.9683) has done: 'We need to move the score up (less negative) toward the target; your current gap is about -7.90, so small fixes that better match the evaluation while preserving your model/loop are appropriate. The biggest low-risk improvement is to train the model on the same target you actually score: the Laplace log-likelihood expects an FVC prediction, but the current training uses a shifted delta target while the loss compares that shifted delta to the model’s second output as if it were an FVC prediction (target/units mismatch). I keep the exact architecture and training loop, but change the loss to compute the metric on reconstructed `FVC_pred = base_FVC + (delta_pred - shift)` and `FVC_true = base_FVC + (y_shifted - shift)`; this is purely a consistent reparameterization, not a new approach. I also clamp `Confidence` during training the same way as in inference and ensure tensors are on the correct device to avoid silent CPU/GPU mismatches.'
- What this solution (achieved -14.9683) has done: 'Your score is far below the target (gap ≈ -7.90, higher is better), so we should improve it with the smallest changes that directly better match the evaluation. The biggest issue is the model’s final `ReLU()` forces the second output (your shifted-delta head) to be non-negative, which cannot represent FVC decline; this is a fundamental mismatch with the task and heavily hurts score. Without changing the architecture, we can keep the same two-head output but reinterpret the second head as a *magnitude* and supply the *direction* (sign) from a per-patient linear trend fit on the training history, then apply that sign at inference (and also use it consistently for training targets). This preserves the training loop and loss structure while allowing negative deltas and typically moves the score substantially toward the target.'

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


DELTA_SHIFT = 2000.0


def quartile_loss(y_true_fvc, y_pred):
    return -competition_metric(y_true_fvc, y_pred)


def quartile_loss_from_signed_shifted_delta(y_shifted_signed, base_fvc, y_pred):
    """
    Score-relevant fix while preserving architecture/training loop:
    the model's 2nd head is ReLU-constrained, so treat it as a magnitude and apply a sign
    to reconstruct signed delta before computing metric.
    """
    device = y_pred.device
    base_fvc = base_fvc.to(device).view(-1)
    y_shifted_signed = y_shifted_signed.to(device).view(-1)

    sigma = y_pred[:, 0].clamp_min(C1.to(device))
    delta_shifted_signed_pred = y_pred[:, 1]  # will already include sign+shift outside

    fvc_true = base_fvc + (y_shifted_signed - float(DELTA_SHIFT))
    fvc_pred = base_fvc + (delta_shifted_signed_pred - float(DELTA_SHIFT))

    y_pred_fvc = torch.stack([sigma, fvc_pred], dim=1)
    return quartile_loss(fvc_true, y_pred_fvc)




## === cell 4
def make_eval_data(npEval, model, sign_map, device="cuda"):
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

    if torch.cuda.is_available() and device == "cuda":
        model.to("cuda")
    model.eval()

    predictions = []
    for i, _ in enumerate(x_patientids_name):
        x_feature = x_features[i].unsqueeze(0)
        pid = npEval.iloc[i]["Patient"]
        sgn = float(sign_map.get(pid, -1.0))  # default to decline if unknown
        if torch.cuda.is_available() and device == "cuda":
            x_feature = x_feature.cuda()
        with torch.no_grad():
            prediction = model(x_feature)
            prediction = torch.cat(
                [
                    prediction[:, :1].clamp_min(C1.to(prediction.device)),
                    prediction[:, 1:],
                ],
                dim=1,
            )
        sigma = prediction[:, 0:1]
        mag_shifted = prediction[:, 1:2]  # >=0 due to ReLU

        signed_shifted = sgn * (mag_shifted - float(DELTA_SHIFT)) + float(DELTA_SHIFT)
        prediction = torch.cat([sigma, signed_shifted], dim=1)
        predictions.append(prediction.to("cpu").detach().numpy()[0])

    predictions = np.array(predictions)

    delta_pred = predictions[:, 1].astype(np.float32) - np.float32(DELTA_SHIFT)
    npEval["FVC"] = npEval["base_FVC"].values.astype(np.float32) + delta_pred
    npEval["Confidence"] = predictions[:, 0]
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


def build_sign_map_from_train(df_train_raw):
    """
    Estimate per-patient trend direction (sign of slope) from training history.
    """
    sign_map = {}
    for pid, g in df_train_raw.groupby("Patient"):
        g = g.sort_values("Weeks")
        if len(g) < 2:
            sign_map[pid] = -1.0
            continue
        w = g["Weeks"].astype(np.float32).values
        f = g["FVC"].astype(np.float32).values
        w0 = w - w.mean()
        denom = float(np.dot(w0, w0))
        if denom <= 1e-6:
            sign_map[pid] = -1.0
            continue
        slope = float(np.dot(w0, f - f.mean()) / denom)  # ml per week
        sign_map[pid] = 1.0 if slope >= 0 else -1.0
    return sign_map


def build_percent_to_slope_sign_model(df_train_raw):
    """
    Score-relevant minimal addition: test Patient IDs do not exist in train, so defaulting to -1
    forces all test trajectories to decline. Fit a simple mapping Percent -> slope_sign from train
    and use it to assign a sign for each test patient using their baseline Percent.
    """
    g = df_train_raw.groupby("Patient").agg(
        Percent=("Percent", "first"),
        Weeks=("Weeks", lambda x: x.astype(np.float32).values),
        FVC=("FVC", lambda x: x.astype(np.float32).values),
    )

    percents = []
    signs = []
    for _, row in g.iterrows():
        w = row["Weeks"]
        f = row["FVC"]
        if len(w) < 2:
            continue
        w0 = w - w.mean()
        denom = float(np.dot(w0, w0))
        if denom <= 1e-6:
            continue
        slope = float(np.dot(w0, f - f.mean()) / denom)
        perc = float(row["Percent"])
        if not np.isfinite(perc):
            continue
        percents.append(perc)
        signs.append(1.0 if slope >= 0 else -1.0)

    percents = np.asarray(percents, dtype=np.float32)
    signs = np.asarray(signs, dtype=np.float32)
    if len(percents) < 10:
        majority = 1.0 if float((signs > 0).mean()) >= 0.5 else -1.0
        return {"mode": "majority", "majority": majority}

    x = percents.reshape(-1, 1)
    y = signs
    X = np.concatenate([np.ones((len(x), 1), dtype=np.float32), x], axis=1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {
        "mode": "linear",
        "beta0": float(beta[0]),
        "beta1": float(beta[1]),
        "fallback": -1.0,
    }


def predict_sign_from_percent(model_info, percent_value):
    if not np.isfinite(percent_value):
        return -1.0
    if model_info["mode"] == "majority":
        return float(model_info["majority"])
    score = model_info["beta0"] + model_info["beta1"] * float(percent_value)
    return 1.0 if score >= 0 else -1.0


sign_map_train = build_sign_map_from_train(data_train)
percent_sign_model = build_percent_to_slope_sign_model(data_train)

pth_path = find_pretrained_pth()
pth_path




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2505828013.py in <cell line: 0>()
    109 
    110 sign_map_train = build_sign_map_from_train(data_train)
--> 111 percent_sign_model = build_percent_to_slope_sign_model(data_train)
    112 
    113 pth_path = find_pretrained_pth()

/tmp/ipykernel_55/2505828013.py in build_percent_to_slope_sign_model(df_train_raw)
     54     and use it to assign a sign for each test patient using their baseline Percent.
     55     """
---> 56     g = df_train_raw.groupby("Patient").agg(
     57         Percent=("Percent", "first"),
     58         Weeks=("Weeks", lambda x: x.astype(np.float32).values),

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py in aggregate(self, func, engine, engine_kwargs, *args, **kwargs)
   1430 
   1431         op = GroupByApply(self, func, args=args, kwargs=kwargs)
-> 1432         result = op.agg()
   1433         if not is_dict_like(func) and result is not None:
   1434             # GH #52849

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in agg(self)
    188 
    189         if is_dict_like(func):
--> 190             return self.agg_dict_like()
    191         elif is_list_like(func):
    192             # we require a list, but not a 'str'

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in agg_dict_like(self)
    421         Result of aggregation.
    422         """
--> 423         return self.agg_or_apply_dict_like(op_name="agg")
    424 
    425     def compute_dict_like(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in agg_or_apply_dict_like(self, op_name)
   1606             obj, "as_index", True, condition=hasattr(obj, "as_index")
   1607         ):
-> 1608             result_index, result_data = self.compute_dict_like(
   1609                 op_name, selected_obj, selection, kwargs
   1610             )

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in compute_dict_like(self, op_name, selected_obj, selection, kwargs)
    494         else:
    495             # key used for column selection and output
--> 496             results = [
    497                 getattr(obj._gotitem(key, ndim=1), op_name)(how, **kwargs)
    498                 for key, how in func.items()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in <listcomp>(.0)
    495             # key used for column selection and output
    496             results = [
--> 497                 getattr(obj._gotitem(key, ndim=1), op_name)(how, **kwargs)
    498                 for key, how in func.items()
    499             ]

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py in aggregate(self, func, engine, engine_kwargs, *args, **kwargs)
    255             kwargs["engine"] = engine
    256             kwargs["engine_kwargs"] = engine_kwargs
--> 257             ret = self._aggregate_multiple_funcs(func, *args, **kwargs)
    258             if relabeling:
    259                 # columns is not narrowed by mypy from relabeling flag

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py in _aggregate_multiple_funcs(self, arg, *args, **kwargs)
    360             for idx, (name, func) in enumerate(arg):
    361                 key = base.OutputKey(label=name, position=idx)
--> 362                 results[key] = self.aggregate(func, *args, **kwargs)
    363 
    364         if any(isinstance(x, DataFrame) for x in results.values()):

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py in aggregate(self, func, engine, engine_kwargs, *args, **kwargs)
    292 
    293             try:
--> 294                 return self._python_agg_general(func, *args, **kwargs)
    295             except KeyError:
    296                 # KeyError raised in test_groupby.test_basic is bc the func does

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py in _python_agg_general(self, func, *args, **kwargs)
    325 
    326         obj = self._obj_with_exclusions
--> 327         result = self._grouper.agg_series(obj, f)
    328         res = obj._constructor(result, name=obj.name)
    329         return self._wrap_aggregated_output(res)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in agg_series(self, obj, func, preserve_dtype)
    862             preserve_dtype = True
    863 
--> 864         result = self._aggregate_series_pure_python(obj, func)
    865 
    866         npvalues = lib.maybe_convert_objects(result, try_float=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in _aggregate_series_pure_python(self, obj, func)
    888             if not initialized:
    889                 # We only do this validation on the first iteration
--> 890                 check_result_array(res, group.dtype)
    891                 initialized = True
    892 

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in check_result_array(obj, dtype)
     86             # If it is object dtype, the function can be a reduction/aggregation
     87             #  and still return an ndarray e.g. test_agg_over_numpy_arrays
---> 88             raise ValueError("Must produce aggregated value")
     89 
     90 

ValueError: Must produce aggregated value

## === cell 10
def build_train_tensors(df_train_raw, sign_map):
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

    y_delta = (
        df["actual_FVC"].astype(np.float32).values
        - df["base_FVC"].astype(np.float32).values
    ).astype(np.float32)

    sgn = df["Patient"].map(lambda p: sign_map.get(p, -1.0)).astype(np.float32).values
    y_delta_signed = (sgn * np.abs(y_delta)).astype(np.float32)

    y_shifted_signed = (
        (y_delta_signed + np.float32(DELTA_SHIFT)).reshape(-1, 1).astype(np.float32)
    )

    base_fvc = df["base_FVC"].astype(np.float32).values.reshape(-1, 1)
    return torch.tensor(x), torch.tensor(y_shifted_signed), torch.tensor(base_fvc)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = SIGMA().to(device)

if pth_path is not None:
    state = torch.load(pth_path, map_location=device)
    model.load_state_dict(state)
else:
    x_train, y_train_shifted_signed, base_fvc_train = build_train_tensors(
        data_train, sign_map_train
    )

    dataset = torch.utils.data.TensorDataset(
        x_train, y_train_shifted_signed, base_fvc_train
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=0
    )

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(25):
        for xb, yb_shifted_signed, basefvc_b in loader:
            xb = xb.to(device)
            yb_shifted_signed = yb_shifted_signed.to(device)
            basefvc_b = basefvc_b.to(device)

            pred = model(xb)
            pred = torch.cat([pred[:, :1].clamp_min(C1.to(device)), pred[:, 1:]], dim=1)

            loss = quartile_loss_from_signed_shifted_delta(
                yb_shifted_signed, basefvc_b, pred
            )

            optim.zero_grad()
            loss.backward()
            optim.step()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1299542265.py in <cell line: 0>()
     40 model = SIGMA().to(device)
     41 
---> 42 if pth_path is not None:
     43     state = torch.load(pth_path, map_location=device)
     44     model.load_state_dict(state)

NameError: name 'pth_path' is not defined

## === cell 11
test_base = pd.read_csv(test_path).copy()
sign_map_test = {}
for pid, g in test_base.groupby("Patient"):
    perc = float(g["Percent"].iloc[0])
    sign_map_test[pid] = float(predict_sign_from_percent(percent_sign_model, perc))

test_pred = make_eval_data(
    data_test.copy(), model, sign_map=sign_map_test, device=device
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



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2624668336.py in <cell line: 0>()
      5 for pid, g in test_base.groupby("Patient"):
      6     perc = float(g["Percent"].iloc[0])
----> 7     sign_map_test[pid] = float(predict_sign_from_percent(percent_sign_model, perc))
      8 
      9 test_pred = make_eval_data(

NameError: name 'percent_sign_model' is not defined

## === cell 12
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

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1911083836.py in <cell line: 0>()
----> 1 pred_map = test_pred.set_index(["Patient", "Week"])[["FVC", "Confidence"]]
      2 sub_final = submission.copy()
      3 sub_final["Patient"] = sub_final["Patient_Week"].apply(lambda x: x.split("_")[0])
      4 sub_final["Weeks"] = sub_final["Patient_Week"].apply(lambda x: int(x.split("_")[1]))
      5 

NameError: name 'test_pred' is not defined
