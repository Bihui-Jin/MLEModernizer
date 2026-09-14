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

-7.2012

# 6. Current score

-8.71736

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -9.98171) has done: 'Diagnosis: The crash happens in cell 3 when `plt.plot(losses, ...)` tries to convert the `losses` list elements (PyTorch tensors that still require gradients) into NumPy arrays internally. Since `losses` and `valLosses` were appended during training without detaching, those tensors keep autograd history, and Matplotlib triggers `tensor.numpy()` which raises `RuntimeError: Can't call numpy() on Tensor that requires grad`.  
Patch summary: In cell 3 only, convert `losses` and `valLosses` to plain Python floats (or detached CPU tensors) before plotting, preserving identical plotting semantics while avoiding autograd-related NumPy conversion. No changes to training, model, or metric computations are made.  
Updated cells: Only cell 3 is modified.  
Compatibility notes for cell k+1: No new variables are introduced and all existing outputs (`submission.csv`, `submission`, `submDataSet`, etc.) are produced exactly as before; only the plotting inputs are detached for visualization.  
Assumptions: The intent of `losses`/`valLosses` is for plotting only, so converting them to scalars does not affect downstream model predictions or submission generation.'
- What this solution (achieved -24.65925) has done: 'Your current gap to the target is about -2.78 (you’re worse than target and higher is better), so we want a modest, low-risk score increase without changing the model or training loop structure. The biggest issue hurting the Laplace metric here is that you submit raw model “confidence” values that can be negative/small, but the metric clips at 70—so underestimating confidence is heavily penalized and you don’t benefit from it being smaller. I keep predictions identical but clamp the submission `Confidence` to be at least 70 (and also ensure it’s a plain float array), which typically improves this competition’s score substantially while preserving your core logic. I also move the score constants to the same device as the model inputs to avoid silent CPU/GPU mismatch if CUDA is enabled (no semantic change on CPU, avoids crashes on GPU).'
- What this solution (achieved -9.21799) has done: 'Diagnosis: The crash happens because `model` was moved to `device` (CUDA if available) in cell 2, but in cell 3 you create `asd` and `inp` on CPU and pass them into the CUDA model. PyTorch requires both the model weights and input tensors to be on the same device, so `model(asd)` and `model(inp)` raise the “found at least two devices” error. The rest of the logic is fine; only the tensor device placement needs to be aligned.  

Patch summary: In cell 3, move the inference input tensors (`asd` and `inp`) onto the same `device` as the model before calling `model(...)`. After inference, keep the existing `.cpu().numpy()` conversions for submission arrays unchanged to preserve output types/format.  

Updated cells:  

Compatibility notes for cell k+1: No interfaces or variable names change; `pred`, `out`, `confidence_subm`, `fvc_subm`, and `submission` remain identical in meaning and shape. This patch only ensures tensors are on the correct device, so any downstream cell expecting the same variables continue to work.  

Assumptions: `device` is defined earlier (cell 0) and `model` remains on `device` as set in cell 2; this patch aligns only the inputs to that existing device choice.'
- What this solution (achieved -8.4713) has done: 'Your score is worse than the target (higher is better), so we want a small, low-risk improvement without changing the model/training core. The biggest easy gain here is to make the predicted `FVC` consistent with how the Laplace metric uses `sigma`: currently you submit the median (`q50`) but use `q80-q20` as “sigma”, while for a Laplace distribution the location parameter is the median and the scale relates to quantile spread by a known constant. I keep the same quantile outputs but (1) convert your predicted spread into a Laplace-consistent `Confidence` (still clipped at 70), and (2) lightly blend the median toward the mean of the three quantiles to reduce extreme drift with essentially no structural change. I also ensure `Confidence` is strictly positive before the final clamp to avoid any corner cases, while keeping the rest of the pipeline identical and still writing a valid `submission.csv`.'
- What this solution (achieved -8.71736) has done: 'We’re currently below the target (−8.4713 vs −7.2012; higher is better), so we want a small, low-risk improvement without changing the model or training loop structure. The biggest metric-relevant issue left is calibration of `Confidence`: you already convert quantile spread to a Laplace-consistent sigma, but you never apply the metric’s required clipping (`sigma_clipped = max(sigma, 70)`) when evaluating, so the model is implicitly trained to push sigma below 70 (which only hurts the competition metric). I keep the exact architecture and training flow, but adjust only the `score()` loss term to use `sigma_clipped` (same as the Kaggle metric) so training aligns with evaluation. I also ensure `sigma` is positive before clipping to avoid any numerical edge cases; submission generation remains unchanged and still writes `submission.csv`.'

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

data = pd.concat([train, subms, test], ignore_index=True, sort=False)

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
            pt.nn.Linear(len(inputs), 100),
            pt.nn.ReLU(),
            pt.nn.Linear(100, 100),
            pt.nn.ReLU(),
        )
        self.left = pt.nn.Sequential(pt.nn.Linear(100, 100))
        self.sigmoid = pt.nn.Sigmoid()
        self.right = pt.nn.Sequential(pt.nn.Linear(100, 100))
        self.last = pt.nn.Sequential(
            pt.nn.Linear(100, 3),
        )
        self.lastRelu = pt.nn.Sequential(pt.nn.Linear(100, 3), pt.nn.ReLU())

    def forward(self, x):
        h = self.start(x)
        p1 = self.last(h)
        p2 = self.lastRelu(h)
        out = p1 + pt.cumsum(p2, 1)
        return out


model = Model()

C1, C2 = pt.FloatTensor([70]).to(device), pt.FloatTensor([1000]).to(device)


def score(y_pred, y_true):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    sigma = pt.abs(
        sigma
    )  # keep sigma non-negative; does not change semantics when sigma is already positive
    fvc_pred = y_pred[:, 1]

    sigma_clip = pt.max(sigma, C1.expand_as(sigma))
    delta = pt.abs(y_true - fvc_pred)
    delta = pt.min(delta, C2)
    sq2 = pt.sqrt(pt.FloatTensor([2]).to(y_pred.device))
    metric = (delta / sigma_clip) * sq2 + pt.log(sigma_clip * sq2)
    return pt.mean(metric)


def pinballLoss(pred, label, quant):
    err = label - pred
    m = pt.mean(pt.max(quant * err, (quant - 1) * err))
    return m


def loss(pred, label):
    quantiles = pt.FloatTensor([0.2, 0.5, 0.8]).to(pred.device)
    return 0.8 * pinballLoss(pred, label, quantiles) + 0.2 * score(pred, label)




## === cell 2
model = model.to(device)

trainDataSet = OSICDataSet(train)
optimizer = optim.Adam(model.parameters(), lr=0.075, weight_decay=0.1, eps=0.00)
losses = []
valLosses = []
kf = KFold(n_splits=5)
c = 0
submDataSet = OSICDataSet(subms, "submission")
inp = pt.from_numpy(submDataSet.data[inputs].values.astype(np.float32)).to(device)

for train_index, val_index in kf.split(trainDataSet):
    for i in range(100):
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

plot_losses = [float(l.detach().cpu()) if pt.is_tensor(l) else float(l) for l in losses]
plot_valLosses = [
    float(l.detach().cpu()) if pt.is_tensor(l) else float(l) for l in valLosses
]

plt.plot(plot_losses, label="Train loss")
plt.plot(plot_valLosses, label="Val loss")
plt.legend()
plt.show()

asd = pt.from_numpy(trainDataSet.data[inputs].values.astype(np.float32)).to(device)
pred = model(asd).detach()
idxs = np.random.randint(0, len(trainDataSet), 100)
plt.plot(
    trainDataSet.data.loc[idxs, ["FVC"]].values.astype(np.float32), label="ground truth"
)
plt.plot(pred[idxs, 0].detach().cpu(), label="q25")
plt.plot(pred[idxs, 1].detach().cpu(), label="q50")
plt.plot(pred[idxs, 2].detach().cpu(), label="q75")
plt.legend(loc="best")
plt.show()
c_pred = (pred[:, 2] - pred[:, 0]).detach().cpu()
f_pred = pred[:, 1].detach().cpu()

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
sns.distplot(
    c_pred, color="g", kde=False, kde_kws={"shade": True}, ax=axes[0]
).set_title("Predicted Confidence on train set")
sns.distplot(
    f_pred, color="g", kde=False, kde_kws={"shade": True}, ax=axes[1]
).set_title("Predicted FVC on train set")
plt.show()

submDataSet = OSICDataSet(subms, "submission")
inp = pt.from_numpy(submDataSet.data[inputs].values.astype(np.float32)).to(device)
out = model(inp).detach()

q20 = out[:, 0]
q50 = out[:, 1]
q80 = out[:, 2]

spread = q80 - q20
laplace_sigma = (
    pt.sqrt(pt.tensor(2.0, device=out.device))
    / (2.0 * pt.log(pt.tensor(1.6, device=out.device)))
) * spread

fvc = 0.85 * q50 + 0.15 * (q20 + q50 + q80) / 3.0

confidence = pt.clamp(laplace_sigma, min=1e-6)

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
sns.distplot(
    confidence.detach().cpu(),
    color="g",
    kde=False,
    kde_kws={"shade": True},
    ax=axes[0],
).set_title("Confidence on submission set")
sns.distplot(
    fvc.detach().cpu(), color="g", kde=False, kde_kws={"shade": True}, ax=axes[1]
).set_title("FVC on submision set")
plt.show()

submDataSet.data["pFVC"] = fvc.detach().cpu()
submDataSet.data["cConf"] = confidence.detach().cpu()

confidence_subm = (
    pt.clamp(confidence, min=70.0).detach().cpu().numpy().astype(np.float32)
)
fvc_subm = fvc.detach().cpu().numpy().astype(np.float32)

submission = pd.DataFrame(
    {
        "Patient_Week": submDataSet.data.Patient_Week.values,
        "FVC": fvc_subm,
        "Confidence": confidence_subm,
    }
)

otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    submission.loc[
        submission["Patient_Week"] == otest.Patient[i] + "_" + str(otest.Weeks[i]),
        "FVC",
    ] = otest.FVC[i]
    submission.loc[
        submission["Patient_Week"] == otest.Patient[i] + "_" + str(otest.Weeks[i]),
        "Confidence",
    ] = 70.0  # keep consistent with metric clipping floor

submission.to_csv("submission.csv", index=False)
null_columns = submission.columns[submission.isnull().any()]
print(submission.head())
print(len(submission))
