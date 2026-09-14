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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

-7.2289

# 6. Current score

-10.31834

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -9.75015) has done: 'Diagnosis: The crash happens in cell 3 when `plt.plot(losses, ...)` tries to convert elements of `losses`/`valLosses` (PyTorch tensors that still require gradients) into NumPy arrays internally. Matplotlib triggers `Tensor.__array__()`, which calls `.numpy()` and fails for tensors with `requires_grad=True`. The fix is to detach (and move to CPU) before plotting, without changing training or loss computation semantics.

Patch summary: In cell 3 only, convert `losses` and `valLosses` into plain Python floats (via `detach().cpu().item()`) right before plotting. This preserves all downstream variables and computations, and only changes the plotting inputs to avoid autograd-related NumPy conversion errors.

Updated cells:'
- What this solution (achieved -10.38005) has done: 'You’re currently below the target (−9.75015 vs −7.2289, higher is better), so we should improve score but with minimal changes. The biggest metric issue in your current pipeline is that `Confidence` is emitted as a raw (often too small/negative) `q80-q20`, while the competition clips sigma at 70; explicitly clipping to at least 70 in the submission is a small, metric-aligned post-processing change that typically improves the Laplace log-likelihood without changing the model or training. I also make the `score()` and loss constants device-safe and numerically consistent (so CPU/GPU runs don’t diverge or error) and set deterministic seeds for stability; these do not change the core approach. Everything else (model, loss form, training loops) stays the same, and the script still write a valid `submission.csv`.'
- What this solution (achieved -10.38005) has done: 'Your current score (−10.38005) is below the target (−7.2289), so we should improve it with the smallest metric-aligned changes. The biggest remaining issue is that you set `Confidence=0.1` for the known baseline test rows, but the metric clips sigma to ≥70 anyway; this unnecessarily worsens the log-likelihood term (it effectively uses 70 but still adds avoidable inconsistency), so we should instead set those rows’ confidence to 70. Also, your inference is done in training mode (dropout/BN aren’t present, but it still matters for determinism and correctness); switching to `model.eval()` and using `torch.no_grad()` is a minimal, semantics-preserving fix that typically improves stability and score. Everything else (data features, model architecture, loss, training loop) is kept identical.'
- What this solution (achieved -10.10195) has done: 'To move your score up toward the target with minimal risk, I make two metric-aligned post-processing tweaks that don’t change the model/training: (1) enforce a valid positive “sigma” by computing `Confidence` as an absolute quantile spread (so it can’t go negative) and clamping to ≥70 as the metric expects, and (2) slightly reduce overconfidence by inflating the clamped confidence by a small factor (1.1), which often improves Laplace log-likelihood when errors are larger than predicted uncertainty. I also ensure submission `FVC` stays within a realistic range based on training labels (simple clipping), which can reduce large-error penalties without altering core logic. Everything else (features, architecture, loss, training loop structure) is kept the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved -10.31834) has done: 'Your current score (−10.10195) is below the target (−7.2289), so we should improve it with minimal, metric-aligned changes without touching the model/training core. The biggest safe gain is to fix the per-row `Confidence` so it matches expected uncertainty: instead of using a global inflation factor, estimate sigma from training residuals binned by week-offset and use that as a calibrated confidence per `Patient_Week` (still clamped to ≥70 per metric). This only changes post-processing of `Confidence` (and keeps your FVC predictions intact), which typically increases Laplace log-likelihood when the model is miscalibrated. I also ensure the submission is aligned and numeric-stable, and keep the baseline-week override intact.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch as pt
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.optim as optim
import matplotlib.pyplot as plt
import pydicom
from sklearn.model_selection import KFold
import seaborn as sns

SEED = 42
np.random.seed(SEED)
pt.manual_seed(SEED)
if pt.cuda.is_available():
    pt.cuda.manual_seed_all(SEED)
pt.backends.cudnn.deterministic = True
pt.backends.cudnn.benchmark = False

trainImagesPath = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/"
dtype = pt.float
use_cuda = pt.cuda.is_available()
device = pt.device("cuda:0" if use_cuda else "cpu")

inputs = [
    "PercentIn",
    "AgeIn",
    "WeekIn",
    "min_FVC",
    "SmokingStatus_Currently smokes",
    "SmokingStatus_Ex-smoker",
    "SmokingStatus_Never smoked",
    "Sex_Male",
    "Sex_Female",
]

test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
subms = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
subms["Patient"] = subms.Patient_Week.apply(lambda x: x.split("_")[0])
subms["Weeks"] = subms.Patient_Week.apply(lambda x: int(x.split("_")[-1]))
train["Split"] = "train"
test["Split"] = "test"
subms = subms[["Patient", "Weeks", "Patient_Week"]]
subms = subms.merge(test.drop("Weeks", axis=1), on="Patient")
subms["Split"] = "subm"

data = pd.concat([train, subms, test], ignore_index=True)

data["first_week"] = data.Weeks
data["first_week"] = data.groupby("Patient")["first_week"].transform("min")
data["first_week"] = data.Weeks - data.first_week
data["min_FVC"] = data.groupby("Patient")["FVC"].transform("min")
print(len(data))
data["WeekIn"] = (data.first_week - data.first_week.min()) / (
    data.first_week.max() - data.first_week.min()
)
data = pd.concat(
    [data, pd.get_dummies(data.SmokingStatus, prefix="SmokingStatus")], axis=1
)
data = pd.concat([data, pd.get_dummies(data.Sex, prefix="Sex")], axis=1)

print(len(data))

train = data.loc[data.Split == "train"].copy()
subms = data.loc[data.Split == "subm"].copy()
test = data.loc[data.Split == "test"].copy()

f, axes = plt.subplots(1, 2, figsize=(7, 7), sharey=True, sharex=True)
sns.distplot(
    train.Age.values, color="g", kde=False, kde_kws={"shade": True}, ax=axes[0]
).set_title("Training age")
sns.distplot(
    subms.Age.values, color="g", kde=False, kde_kws={"shade": True}, ax=axes[1]
).set_title("Submission age")
plt.legend()
plt.show()




## === cell 1
class OSICDataSet(Dataset):
    def __init__(self, data, mode="train"):
        self.data = data
        self.data["Smoke"] = self.data.SmokingStatus.replace(
            {"Ex-smoker": 0.5, "Never smoked": 0, "Currently smokes": 1}
        )
        self.data["Gender"] = self.data.Sex.replace({"Male": 1, "Female": 0})
        self.data["min_FVC"] = (self.data.min_FVC - data.min_FVC.min()) / (
            data.min_FVC.max() - data.min_FVC.min()
        )
        self.data["WeekIn"] = (self.data.first_week - data.first_week.min()) / (
            data.first_week.max() - data.first_week.min()
        )
        self.data["AgeIn"] = (self.data.Age - data.Age.min()) / (
            data.Age.max() - data.Age.min()
        )
        self.data["PercentIn"] = (self.data.Percent - data.Percent.min()) / (
            data.Percent.max() - data.Percent.min()
        )
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        otherData = pt.from_numpy(self.data.loc[idx, inputs].values.astype(np.float32))
        if self.mode == "train":
            targets = pt.from_numpy(
                self.data.loc[idx, ["FVC"]].values.astype(np.float32)
            )
            return otherData, targets
        return otherData  # TODO: Add image data


class Model(pt.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.start = pt.nn.Sequential(
            pt.nn.Linear(len(inputs), 128),
            pt.nn.ReLU(),
            pt.nn.Linear(128, 256),
            pt.nn.ReLU(),
            pt.nn.Linear(256, 512),
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

C1 = pt.tensor([70.0], device=device, dtype=pt.float32)
C2 = pt.tensor([1000.0], device=device, dtype=pt.float32)
SQRT2 = pt.sqrt(pt.tensor([2.0], device=device, dtype=pt.float32))


def score(y_pred, y_true):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = pt.max(sigma, C1.expand_as(sigma))
    delta = pt.abs(y_true.squeeze(-1) - fvc_pred)
    delta = pt.min(delta, C2)
    metric = (delta / sigma_clip) * SQRT2 + pt.log(sigma_clip * SQRT2)
    return pt.mean(metric)


def pinballLoss(pred, label, quant):
    err = label - pred
    m = pt.mean(pt.max(quant * err, (quant - 1) * err))
    return m


def loss(pred, label):
    quantiles = pt.tensor([0.2, 0.5, 0.8], device=pred.device, dtype=pred.dtype)
    return 0.8 * pinballLoss(pred, label, quantiles) + 0.2 * score(pred, label)




## === cell 2
trainDataSet = OSICDataSet(train)
optimizer = optim.Adam(model.parameters(), lr=0.075, weight_decay=0.1, eps=0.001)
losses = []
valLosses = []
kf = KFold(n_splits=5)
c = 0
submDataSet = OSICDataSet(subms, "submission")
inp = pt.from_numpy(submDataSet.data[inputs].values.astype(np.float32))

for train_index, val_index in kf.split(trainDataSet):
    for i in range(200):
        train_x, train_y = trainDataSet[train_index]
        val_x, val_y = trainDataSet[val_index]
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        val_x = val_x.to(device)
        val_y = val_y.to(device)

        optimizer.zero_grad()
        pred = model(train_x)
        los = loss(pred, train_y)
        los.backward()
        optimizer.step()
        if c % 10 == 0:
            pred = model(val_x)
            valLos = loss(pred, val_y)
            losses.append(los)
            valLosses.append(valLos)
        c += 1



## === cell 3
sns.set(style="white", palette="muted", color_codes=True)

losses_plot = [
    l.detach().cpu().item() if isinstance(l, pt.Tensor) else float(l) for l in losses
]
valLosses_plot = [
    l.detach().cpu().item() if isinstance(l, pt.Tensor) else float(l) for l in valLosses
]

plt.plot(losses_plot, label="Train loss")
plt.plot(valLosses_plot, label="Val loss")
plt.legend()
plt.show()

model.eval()
with pt.no_grad():
    asd = pt.from_numpy(trainDataSet.data[inputs].values.astype(np.float32)).to(device)
    pred = model(asd).detach().cpu()

idxs = np.random.randint(0, len(trainDataSet), 100)
plt.plot(
    trainDataSet.data.loc[idxs, ["FVC"]].values.astype(np.float32), label="ground truth"
)
plt.plot(pred[idxs, 0], label="q25")
plt.plot(pred[idxs, 1], label="q50")
plt.plot(pred[idxs, 2], label="q75")
plt.legend(loc="best")
plt.show()
c_conf = pred[:, 2] - pred[:, 0]
f = pred[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
sns.distplot(
    c_conf, color="g", kde=False, kde_kws={"shade": True}, ax=axes[0]
).set_title("Predicted Confidence on train set")
sns.distplot(f, color="g", kde=False, kde_kws={"shade": True}, ax=axes[1]).set_title(
    "Predicted FVC on train set"
)
plt.show()

submDataSet = OSICDataSet(subms, "submission")
inp = pt.from_numpy(submDataSet.data[inputs].values.astype(np.float32)).to(device)

model.eval()
with pt.no_grad():
    out = model(inp).detach().cpu()

train_pred_q50 = pred[:, 1].numpy()
train_true_fvc = trainDataSet.data["FVC"].values.astype(np.float32)
train_abs_err = np.abs(train_true_fvc - train_pred_q50)

train_fw = trainDataSet.data["first_week"].values.astype(np.float32)
subm_fw = submDataSet.data["first_week"].values.astype(np.float32)

n_bins = 8
bin_edges = np.quantile(train_fw, np.linspace(0.0, 1.0, n_bins + 1))
bin_edges[0] = -1e9
bin_edges[-1] = 1e9

train_bin = np.digitize(train_fw, bin_edges[1:-1], right=False)
subm_bin = np.digitize(subm_fw, bin_edges[1:-1], right=False)

bin_sigma = np.zeros(n_bins, dtype=np.float32)
global_sigma = float(np.sqrt(2.0) * np.mean(train_abs_err))
for b in range(n_bins):
    mask = train_bin == b
    if mask.sum() >= 10:
        bin_sigma[b] = float(np.sqrt(2.0) * np.mean(train_abs_err[mask]))
    else:
        bin_sigma[b] = global_sigma

confidence_np = bin_sigma[subm_bin]
confidence = pt.from_numpy(confidence_np).float().clamp(min=70.0)

fvc_train_min = float(train["FVC"].min())
fvc_train_max = float(train["FVC"].max())
fvc = out[:, 1].clamp(min=fvc_train_min, max=fvc_train_max)

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
sns.distplot(
    confidence.numpy(), color="g", kde=False, kde_kws={"shade": True}, ax=axes[0]
).set_title("Confidence on submission set (calibrated by week-bin, clamped>=70)")
sns.distplot(fvc, color="g", kde=False, kde_kws={"shade": True}, ax=axes[1]).set_title(
    "FVC on submision set (clipped to train range)"
)
plt.show()

submDataSet.data["pFVC"] = fvc.numpy()
submDataSet.data["cConf"] = confidence.numpy()

submission = pd.DataFrame(
    {
        "Patient_Week": submDataSet.data.Patient_Week.values,
        "FVC": fvc.numpy(),
        "Confidence": confidence.numpy(),
    }
)

otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    submission.loc[submission["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])
    submission.loc[submission["Patient_Week"] == key, "Confidence"] = 70.0

submission["FVC"] = submission["FVC"].astype(np.float32)
submission["Confidence"] = submission["Confidence"].astype(np.float32)

submission.to_csv("submission.csv", index=False)
null_columns = submission.columns[submission.isnull().any()]
print(submission.head())
print(len(submission))
print("Null columns:", list(null_columns))
