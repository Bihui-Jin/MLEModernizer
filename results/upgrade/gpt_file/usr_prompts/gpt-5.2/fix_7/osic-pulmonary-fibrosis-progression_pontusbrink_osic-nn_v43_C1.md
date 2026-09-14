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
seaborn==0.12.2
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

-7.2458

# 6. Current score

-8.25766

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.22753) has done: 'I fix the runtime breakages caused by pandas 2.x deprecations and several indexing/shape bugs that prevent the notebook from reaching submission writing. Specifically: replace the removed `DataFrame.append` with `pd.concat`, ensure all one-hot encoded input columns exist in all splits (adding missing dummy columns with zeros), and correct the Dataset `__getitem__` logic so it supports KFold array indices (returning proper batched tensors). I also fix the training loop to actually train on minibatches from indices (instead of trying to index a Dataset with an array), and make the submission `FVC/Confidence` numeric numpy arrays with confidence clipped to a reasonable positive range to avoid invalid values. These changes preserve the same model architecture and loss setup, but make the pipeline run end-to-end and produce a valid `submission.csv`.'
- What this solution (achieved -8.22753) has done: 'Your current score is worse than the target (gap ≈ -0.98), so we should improve slightly without changing the core model or training loop. The biggest easy gain here is fixing a feature mismatch: `inputCols` uses `WeekIn` but you never created a `WeekIn` column (you created `data["WeekIn"]` but `inputCols` has `WeekIn` while the list contains `"WeekIn"`? actually it contains `"WeekIn"` but earlier you also used `"WeekIn"`—the real issue is you also have `"WeekIn"` but you compute `WeekIn` twice; the bigger issue is that you normalize `WeekIn` from `first_week` but for train/test rows the baseline week should be aligned consistently, and you should use the baseline (week=0) row per patient when building the submission feature rows). I keep your architecture/loss intact and only change submission-row construction: build the submission feature table by merging `sample_submission` with each patient’s baseline clinical row from `test.csv` (Week=0) and recompute `first_week/WeekIn` relative to that baseline, instead of inheriting potentially inconsistent `Weeks` rows from `test_raw.drop("Weeks")`. This is a minimal semantic fix that typically improves OSIC scores because each `Patient_Week` gets the correct week feature and baseline features.'
- What this solution (achieved -8.2092) has done: 'Your score is below the target (gap ≈ -0.98), so we should improve modestly without changing the model/loss/training loop. The biggest low-risk gain is fixing the week feature used at inference: right now `first_week/WeekIn` is computed from each patient’s minimum available week, but Kaggle expects weeks relative to the baseline CT (Week=0) in `test.csv`. I rebuild the submission feature rows by merging `sample_submission.csv` with each patient’s baseline (Week=0) clinical row from `test.csv`, and compute `first_week = Weeks - 0` for those rows, keeping the rest of your pipeline identical. This preserves core logic but makes the `WeekIn` signal consistent between train and submission rows, which typically improves OSIC scores.'
- What this solution (achieved -8.18947) has done: 'Your current score (-8.2092) is worse than the target (-7.2458), so we should improve it modestly without changing the model/loss/training loop. The most direct, low-risk gain is to make the `WeekIn` feature consistent between train and submission rows: compute `first_week` relative to each patient’s baseline week (Week=0 when available) for *all* splits instead of using “min week” for train and raw weeks for test/subm. This keeps your core logic intact but fixes a train/inference feature mismatch that typically hurts OSIC. I also clip the predicted quantiles to enforce `q20 <= q50 <= q80` before computing confidence/FVC, which is consistent with your intended quantile semantics and improves metric stability without changing architecture/training.'
- What this solution (achieved -8.25766) has done: 'We make one minimal, score-relevant fix: your model is trained to output *raw* FVC values, but at inference you’re feeding **min-max normalized** `base_FVC` and `min_FVC`, which causes a train/test feature mismatch and typically depresses OSIC scores. We keep the same model, loss, and training loop, and simply keep `base_FVC` (and `min_FVC`, though it’s unused) in milliliters (no normalization) while leaving the other normalized inputs unchanged. This preserves core logic/evaluation semantics but makes the learned mapping consistent between training and submission rows, which should move the score upward toward the target. The submission writing and quantile monotonicity + confidence clipping remain unchanged.'
- What this solution (achieved -8.25766) has done: 'Your score is below the target (gap ≈ -1.01), so we should improve modestly without changing the model or training semantics. The most direct, low-risk improvement for OSIC is to make the confidence prediction better calibrated to the Laplace metric: instead of using only the model’s predicted interval width, we blend it with an empirical per-patient residual scale learned from training (typical uncertainty for that patient) and then clip as usual. This keeps your architecture, loss, and training loop intact, but reduces overconfident/underconfident predictions which strongly affects the metric. I also ensure the submission row order matches `sample_submission.csv` exactly to avoid any accidental misalignment issues.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch as pt
from torch.utils.data import Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import seaborn as sns

pt.manual_seed(42)
np.random.seed(42)

trainImagesPath = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/"
dtype = pt.float32
use_cuda = pt.cuda.is_available()
device = pt.device("cuda:0" if use_cuda else "cpu")

inputCols = [
    "PercentIn",
    "AgeIn",
    "WeekIn",
    "base_FVC",
    "SmokingStatus_Currently smokes",
    "SmokingStatus_Ex-smoker",
    "SmokingStatus_Never smoked",
    "Sex_Male",
    "Sex_Female",
]

test_raw = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
train_raw = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
subms = pd.read_csv(
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)

subms["Patient"] = subms.Patient_Week.apply(lambda x: x.split("_")[0])
subms["Weeks"] = subms.Patient_Week.apply(lambda x: int(x.split("_")[-1]))

train_raw["Split"] = "train"
test_raw["Split"] = "test"

test_raw["PatientDir"] = test_raw.Patient.apply(
    lambda x: "/kaggle/input/osic-pulmonary-fibrosis-progression/test/" + x
)
train_raw["PatientDir"] = train_raw.Patient.apply(
    lambda x: "/kaggle/input/osic-pulmonary-fibrosis-progression/train/" + x
)


def _baseline_week_map(df: pd.DataFrame) -> pd.Series:
    tmp = df.copy()
    tmp["abs_to0"] = (tmp["Weeks"] - 0).abs()
    tmp = tmp.sort_values(["Patient", "abs_to0", "Weeks"]).drop_duplicates(
        "Patient", keep="first"
    )
    return tmp.set_index("Patient")["Weeks"]


baseline_week_train = _baseline_week_map(train_raw)
baseline_week_test = _baseline_week_map(test_raw)
baseline_week_all = pd.concat([baseline_week_train, baseline_week_test])
baseline_week_all = baseline_week_all[~baseline_week_all.index.duplicated(keep="first")]

test_base = test_raw.loc[test_raw["Weeks"] == 0].copy()
if test_base["Patient"].nunique() < test_raw["Patient"].nunique():
    fallback = (
        test_raw.sort_values(["Patient", "Weeks"])
        .drop_duplicates("Patient", keep="first")
        .copy()
    )
    test_base = pd.concat([test_base, fallback], ignore_index=True)
    test_base = test_base.sort_values(["Patient", "Weeks"]).drop_duplicates(
        "Patient", keep="first"
    )

test_base = test_base.drop(columns=["Weeks"])

subms["PatientDir"] = subms.Patient.apply(
    lambda x: "/kaggle/input/osic-pulmonary-fibrosis-progression/test/" + x
)
subms = subms[["Patient", "Weeks", "Patient_Week"]]
subms = subms.merge(test_base, on="Patient", how="left")
subms["Split"] = "subm"

data = pd.concat([test_raw, subms, train_raw], ignore_index=True)
data = data.sort_values(["Patient", "Weeks"], ascending=True)

data["baseline_week"] = data["Patient"].map(baseline_week_all).astype(np.float32)
missing_bw = data["baseline_week"].isna()
if missing_bw.any():
    data.loc[missing_bw, "baseline_week"] = (
        data.loc[missing_bw]
        .groupby("Patient")["Weeks"]
        .transform("min")
        .astype(np.float32)
    )

data["first_week"] = data["Weeks"].astype(np.float32) - data["baseline_week"].astype(
    np.float32
)

data["min_FVC"] = data.groupby("Patient")["FVC"].transform("min")

b = data[data["first_week"] == 0].copy()
if b["Patient"].nunique() < data["Patient"].nunique():
    fallback_b = data.copy()
    fallback_b["abs_fw"] = fallback_b["first_week"].abs()
    fallback_b = fallback_b.sort_values(["Patient", "abs_fw", "Weeks"]).drop_duplicates(
        "Patient", keep="first"
    )
    b = pd.concat([b, fallback_b], ignore_index=True).drop_duplicates(
        "Patient", keep="first"
    )

b["base_FVC"] = b.FVC
b = b[["Patient", "base_FVC"]].drop_duplicates("Patient")
data = data.merge(b, on="Patient", how="left")

eps = 1e-8
data["WeekIn"] = (data.first_week - data.first_week.min()) / (
    data.first_week.max() - data.first_week.min() + eps
)

data = pd.concat(
    [data, pd.get_dummies(data.SmokingStatus, prefix="SmokingStatus")], axis=1
)
data = pd.concat([data, pd.get_dummies(data.Sex, prefix="Sex")], axis=1)

data["Smoke"] = data.SmokingStatus.replace(
    {"Ex-smoker": 0.5, "Never smoked": 0, "Currently smokes": 1}
)
data["Gender"] = data.Sex.replace({"Male": 1, "Female": 0})

data["AgeIn"] = (data.Age - data.Age.min()) / (data.Age.max() - data.Age.min() + eps)
data["PercentIn"] = (data.Percent - data.Percent.min()) / (
    data.Percent.max() - data.Percent.min() + eps
)

for col in inputCols:
    if col not in data.columns:
        data[col] = 0.0

train = data.loc[data.Split == "train"].copy()
subms = data.loc[data.Split == "subm"].copy()
test = data.loc[data.Split == "test"].copy()

inputs = [data.columns.get_loc(c) for c in inputCols]
outputColI = data.columns.get_loc("FVC")

print("Prepared data shapes:", train.shape, test.shape, subms.shape)
print("Missing in subms:", [c for c in inputCols if c not in subms.columns])




## === cell 1
class OSICDataSet(Dataset):
    """
    BUGFIX: original __getitem__ assumed scalar idx; KFold provides arrays of indices.
    Support both scalar and array-like indices by returning batched tensors when idx is array-like.
    """

    def __init__(self, data, mode="train"):
        self.data = data.reset_index(drop=True)
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray, pd.Index)):
            idx = np.asarray(idx, dtype=int)
            other = self.data.iloc[idx, inputs].to_numpy(dtype=np.float32)
            otherData = pt.from_numpy(other)
            if self.mode == "train":
                y = self.data.iloc[idx, outputColI].to_numpy(dtype=np.float32)
                targets = pt.from_numpy(y)
                return otherData, targets
            return otherData

        otherData = pt.from_numpy(
            self.data.iloc[int(idx), inputs].to_numpy(dtype=np.float32)
        )
        if self.mode == "train":
            targets = pt.tensor(self.data.iloc[int(idx), outputColI], dtype=pt.float32)
            return otherData, targets
        return otherData  # TODO: Add image data


class Model(pt.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.start = pt.nn.Sequential(
            pt.nn.Linear(len(inputCols), 128),
            pt.nn.ReLU(),
            pt.nn.Linear(128, 256),
            pt.nn.ReLU(),
            pt.nn.Linear(256, 512),
            pt.nn.ReLU(),
            pt.nn.Linear(512, 1024),
            pt.nn.ReLU(),
            pt.nn.Linear(1024, 512),
            pt.nn.ReLU(),
            pt.nn.Linear(512, 256),
            pt.nn.ReLU(),
        )
        self.left = pt.nn.Sequential(pt.nn.Linear(256, 128))
        self.sigmoid = pt.nn.Sigmoid()
        self.right = pt.nn.Sequential(pt.nn.Linear(256, 128))
        self.last = pt.nn.Sequential(
            pt.nn.Linear(128, 3),
        )
        self.lastRelu = pt.nn.Sequential(pt.nn.Linear(128, 3), pt.nn.ReLU())

    def forward(self, x):
        h = self.start(x)
        l = self.left(h)
        r = self.right(h)
        h = l * self.sigmoid(r)
        p1 = self.last(h)
        p2 = self.lastRelu(h)
        out = p1 + pt.cumsum(p2, 1)
        return out


model = Model().to(device)

C1 = pt.tensor([70.0], device=device)
C2 = pt.tensor([1000.0], device=device)


def score(y_pred, y_true):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = pt.max(sigma, C1.expand_as(sigma))
    delta = pt.abs(y_true - fvc_pred)
    delta = pt.min(delta, C2)
    sq2 = pt.sqrt(pt.tensor([2.0], device=device))
    metric = (delta / sigma_clip) * sq2 + pt.log(sigma_clip * sq2)
    return pt.mean(metric)


def pinballLoss(pred, label, quant):
    err = label.unsqueeze(1) - pred
    m = pt.mean(pt.max(quant * err, (quant - 1) * err))
    return m


def loss(pred, label):
    quantiles = pt.tensor([0.2, 0.5, 0.8], device=device)
    return 0.8 * pinballLoss(pred, label, quantiles) + 0.2 * score(pred, label)


def earlyStop(useEarlyStopping, earlyStopping):
    if useEarlyStopping:
        return not earlyStopping
    return True




## === cell 2
trainDataSet = OSICDataSet(train, mode="train")

optimizer = optim.Adam(model.parameters(), lr=0.075, weight_decay=0.1, eps=0.001)

losses = []
valLosses = []
nSplits = 5
kf = KFold(n_splits=nSplits, shuffle=True, random_state=42)
c = 0
useEarlyStopping = True
earlyStopping = False
stopping = 0
i = 0

batch_size = 64

while i < 800 and earlyStop(useEarlyStopping, earlyStopping):
    for train_index, val_index in kf.split(np.arange(len(trainDataSet))):
        if len(train_index) == 0 or len(val_index) == 0:
            continue

        start = (i * batch_size) % len(train_index)
        end = min(start + batch_size, len(train_index))
        batch_idx = train_index[start:end]

        train_x, train_y = trainDataSet[batch_idx]
        val_x, val_y = trainDataSet[val_index]

        train_x = train_x.to(device)
        train_y = train_y.to(device)
        val_x = val_x.to(device)
        val_y = val_y.to(device)

        optimizer.zero_grad()
        pred = model(train_x)
        los = loss(pred, train_y)
        los.backward()

        if c % 10 == 0:
            with pt.no_grad():
                vpred = model(val_x)
                valLos = loss(vpred, val_y)
            losses.append(float(los.detach().cpu()))
            valLosses.append(float(valLos.detach().cpu()))

            if useEarlyStopping and (
                len(valLosses) > 1 and valLosses[-1] < valLosses[-2]
            ):
                stopping += 1
                if stopping > nSplits:
                    earlyStopping = True
                break

        c += 1
        i += 1
        stopping = 0
        optimizer.step()
        c += 1

print("Training finished. Steps:", i, "Recorded points:", len(losses))



## === cell 3
sns.set(style="white", palette="muted", color_codes=True)

plt.figure(figsize=(10, 4))
plt.plot(losses, label="Train loss")
plt.plot(valLosses, label="Val loss")
plt.legend()
plt.show()

with pt.no_grad():
    asd = pt.from_numpy(trainDataSet.data[inputCols].to_numpy(dtype=np.float32)).to(
        device
    )
    pred = model(asd).detach().cpu()

idxs = np.random.randint(0, len(trainDataSet), min(100, len(trainDataSet)))
plt.figure(figsize=(10, 4))
plt.plot(
    trainDataSet.data.iloc[idxs, outputColI].to_numpy(dtype=np.float32),
    label="ground truth",
)
plt.plot(pred[idxs, 0], label="q25")
plt.plot(pred[idxs, 1], label="q50")
plt.plot(pred[idxs, 2], label="q75")
plt.legend(loc="best")
plt.show()

c_conf = (pred[:, 2] - pred[:, 0]).numpy()
f_pred = pred[:, 1].numpy()

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
sns.histplot(c_conf, color="g", kde=False, ax=axes[0]).set_title(
    "Predicted Confidence on train set"
)
sns.histplot(f_pred, color="g", kde=False, ax=axes[1]).set_title(
    "Predicted FVC on train set"
)
plt.show()

with pt.no_grad():
    train_out = (
        model(
            pt.from_numpy(trainDataSet.data[inputCols].to_numpy(dtype=np.float32)).to(
                device
            )
        )
        .detach()
        .cpu()
        .numpy()
    )

tq0 = train_out[:, 0].astype(np.float32)
tq1 = train_out[:, 1].astype(np.float32)
tq2 = train_out[:, 2].astype(np.float32)
tq1 = np.maximum(tq1, tq0)
tq2 = np.maximum(tq2, tq1)

train_pred_fvc = tq1
train_true_fvc = trainDataSet.data["FVC"].to_numpy(dtype=np.float32)
train_abs_resid = np.abs(train_true_fvc - train_pred_fvc).astype(np.float32)

train_patients = trainDataSet.data["Patient"].astype(str).to_numpy()
resid_by_patient = (
    pd.DataFrame({"Patient": train_patients, "abs_resid": train_abs_resid})
    .groupby("Patient")["abs_resid"]
    .median()
)
global_resid_med = float(np.median(train_abs_resid))

submDataSet = OSICDataSet(subms, mode="submission")
with pt.no_grad():
    inp = pt.from_numpy(submDataSet.data[inputCols].to_numpy(dtype=np.float32)).to(
        device
    )
    out = model(inp).detach().cpu().numpy()

q0 = out[:, 0].astype(np.float32)
q1 = out[:, 1].astype(np.float32)
q2 = out[:, 2].astype(np.float32)
q1 = np.maximum(q1, q0)
q2 = np.maximum(q2, q1)

model_conf = (q2 - q0).astype(np.float32)
fvc = q1.astype(np.float32)

subm_patients = submDataSet.data["Patient"].astype(str).to_numpy()
pat_scale = (
    pd.Series(subm_patients)
    .map(resid_by_patient)
    .fillna(global_resid_med)
    .to_numpy(dtype=np.float32)
)

confidence = (0.6 * model_conf + 0.4 * (2.0 * pat_scale)).astype(np.float32)

confidence = np.clip(confidence, 70.0, 1000.0)

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
sns.histplot(confidence, color="g", kde=False, ax=axes[0]).set_title(
    "Confidence on submission set"
)
sns.histplot(fvc, color="g", kde=False, ax=axes[1]).set_title("FVC on submission set")
plt.show()

submission = pd.DataFrame(
    {
        "Patient_Week": submDataSet.data.Patient_Week.values,
        "FVC": fvc,
        "Confidence": confidence,
    }
)

otest = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
for j in range(len(otest)):
    key = otest.Patient[j] + "_" + str(int(otest.Weeks[j]))
    submission.loc[submission["Patient_Week"] == key, "FVC"] = float(otest.FVC[j])
    submission.loc[submission["Patient_Week"] == key, "Confidence"] = 70.0

submission = submission[["Patient_Week", "FVC", "Confidence"]]
submission["FVC"] = submission["FVC"].astype(np.float32)
submission["Confidence"] = submission["Confidence"].astype(np.float32)
submission = submission.dropna()

sample_sub = pd.read_csv(
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)[["Patient_Week"]]
submission = sample_sub.merge(submission, on="Patient_Week", how="left")
submission["FVC"] = submission["FVC"].astype(np.float32)
submission["Confidence"] = submission["Confidence"].astype(np.float32)
submission = submission.dropna()

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Rows:", len(submission), "Cols:", submission.columns.tolist())
print("Any nulls:", submission.isnull().any().to_dict())
