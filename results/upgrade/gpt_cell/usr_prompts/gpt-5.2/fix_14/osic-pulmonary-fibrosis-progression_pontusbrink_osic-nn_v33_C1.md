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

-7.2137

# 6. Current score

-8.10138

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -9.56325) has done: 'Diagnosis: Cell 3 crashes when plotting `losses`/`valLosses` because they are lists of PyTorch tensors that still require gradients, and Matplotlib tries to convert them to NumPy via `tensor.numpy()`, which PyTorch disallows for grad-tracking tensors. This happens because in cell 2 you append `los` and `valLos` without detaching. The minimal fix is to detach (or convert to Python floats) at plot time in cell 3, without changing the training loop or loss computation.

Patch summary: In cell 3, convert `losses` and `valLosses` to plain float lists using `detach().cpu().item()` before calling `plt.plot`. Also detach `c`, `f`, and `confidence`/`fvc` where they are passed to seaborn/matplotlib to avoid the same NumPy conversion issue. No model/training logic is changed.

Updated cells: Only cell 3 is modified.

Compatibility notes for cell k+1: No interface/variable contract changes; `submission.csv` generation and downstream variables (`submission`, `submDataSet`, etc.) remain the same types/shapes, with only plotting inputs converted to NumPy/float as needed.

Assumptions: Tensors in `losses`/`valLosses` are scalar tensors; converting via `.item()` is valid. CPU conversion is safe regardless of CUDA availability.'
- What this solution (achieved -24.65921) has done: 'Your current score is well below the target (gap = -9.56325 − (-7.2137) = -2.34955), so we should cautiously improve performance without changing the model/loss/training-loop core logic. The biggest low-risk issue is that the model is trained in full-batch mode with an extremely high learning rate and very large weight decay and `eps=0`, which tends to destabilize training; switching to standard, numerically-stable Adam settings usually improves this OSIC baseline materially while keeping the exact same training approach. I make only minimal optimizer hyperparameter adjustments (lr/weight_decay/eps) and ensure all tensors/constants are on the correct device for stable computation. Submission format, row alignment, and the rule that baseline test rows are overwritten with the known FVC be preserved exactly.'
- What this solution (achieved -9.52424) has done: 'Diagnosis: The crash happens in cell 3 when passing CUDA tensors directly to `matplotlib.pyplot.plot`, which internally tries to convert them to NumPy arrays. PyTorch disallows converting a CUDA tensor to NumPy without moving it to CPU first, hence the `TypeError`. The rest of the cell already uses `.detach().cpu().numpy()` elsewhere, so the fix is to do the same for the specific `pred[idxs, ...]` tensors being plotted. This keeps the model/training logic identical and only changes the plotting inputs to be CPU NumPy arrays.

Patch summary: In cell 3, convert `pred[idxs, 0/1/2]` to CPU NumPy arrays before plotting. No changes to model, loss, training, inference, or submission generation.

Updated cells: Only cell 3 is updated.

Compatibility notes for cell k+1: No variables, types, or interfaces used later are changed; `pred`, `out`, `submission`, etc. remain the same. Only the plotting calls now use CPU NumPy copies.

Assumptions: `pred` is on GPU in this environment (CUDA available), which triggers the error; moving only the plotted slices to CPU is acceptable and does not affect subsequent computations.'
- What this solution (achieved -9.47567) has done: 'Your current score (-9.52424) is worse than the target (-7.2137), so we should make small, stability-oriented changes that tend to improve the LaplaceLL score without changing the model architecture, loss, or training loop structure. The main low-risk improvement is to ensure the predicted uncertainty used for the metric is always valid and numerically stable: enforce a minimum positive spread between q80 and q20 at inference time (so Confidence doesn’t go negative/near-zero and get clipped to 70 in a harmful way). In addition, we output `Confidence` exactly as used by the evaluation (i.e., max(predicted_sigma, 70)) to avoid giving the scorer a smaller sigma that only hurts you. These are purely post-processing/alignment changes for the competition metric and do not alter training semantics.'
- What this solution (achieved -9.54963) has done: 'Your current score (-9.47567) is worse than the target (-7.2137), so we should make a small, metric-aligned fix that improves the Laplace Log Likelihood without changing the model, loss, or training loop. The biggest issue in your current submission is that you overwrite the known baseline test row’s `Confidence` with `0.1`, which gets clipped to `70` anyway but also removes any chance of benefiting from a larger uncertainty when your baseline FVC isn’t matched by the model’s predicted quantiles; instead, we should keep the model-derived confidence for those rows (or at least clip it to 70 like the metric) while still using the known baseline `FVC`. Additionally, we ensure the final `Confidence` written to the CSV is always `>= 70` everywhere (including the overwritten baseline rows), exactly matching the scorer’s clipping behavior. These are pure post-processing fixes and keep your training/inference semantics intact.'
- What this solution (achieved -11.92646) has done: 'You’re below the target (current −9.54963 vs target −7.2137), so we should make a small, metric-aligned improvement without changing the model, loss, or training loop. The biggest low-risk gain is to calibrate `Confidence` better: your current `sigma = q80 - q20` is typically too small and gets clipped to 70, which hurts the LaplaceLL; scaling sigma upward is a pure post-processing change that often improves this metric. We estimate a global scaling factor from out-of-fold residuals on the training set (same features, same trained model) and apply it to submission sigma, then still clip at 70 exactly like the metric. Everything else (data prep, model, training) stays identical and the script still writes `submission.csv`.'
- What this solution (achieved -12.03998) has done: 'Your score is worse than the target (gap = -11.92646 − (-7.2137) = -4.71276), so we should make the smallest metric-aligned change that typically improves LaplaceLL without changing the model/training core. The biggest issue is that your current `sigma_scale` is derived from in-sample residuals (optimistic) and only uses a median ratio; switching to simple out-of-fold (OOF) residual-based calibration makes the confidence closer to what the metric rewards and usually increases the score. I keep the same trained model and only add an OOF prediction pass (no retraining, no architecture change) to estimate a robust global `sigma_scale`, then apply it exactly as before and still clip Confidence to `>=70`. Submission generation, baseline FVC overwrite, and CSV schema remain unchanged.'
- What this solution (achieved -11.14378) has done: 'Your current score (-12.03998) is well below the target (-7.2137), so we should make a small, metric-aligned improvement without changing the model, loss, or training loop structure. The biggest issue is that the “OOF” sigma calibration is not truly out-of-fold because it uses the same single trained model for every fold, so it doesn’t reflect generalization and can miscalibrate Confidence. I keep the exact same architecture/loss and still use 5-fold training, but I store each fold’s validation predictions during the existing loop and use those true fold-held-out predictions to estimate `sigma_scale`. Then I apply that calibrated `sigma_scale` to the submission Confidence (still clipped to >=70 exactly like the metric), which typically improves LaplaceLL toward your target.'
- What this solution (achieved -10.59048) has done: 'Your current score (-11.14378) is well below the target (-7.2137), so we should make the smallest metric-aligned change that reliably improves LaplaceLL without changing the model, loss, or training loop. The biggest remaining issue is that `sigma_scale` is estimated from medians, which often underestimates the optimal uncertainty; for this metric, slightly larger (better-calibrated) Confidence frequently improves score when FVC errors are non-trivial. I keep the same fold-held-out OOF predictions you already collect, but compute `sigma_scale` by directly maximizing the mean Laplace Log Likelihood on OOF data (a simple 1D grid search), then apply that single global scale to test Confidence and still clip at 70 exactly like the metric. This is only a post-processing calibration change (no architecture/training changes) and should move the score upward toward your target.'
- What this solution (achieved -8.10138) has done: 'Your score is substantially below the target (current -10.59048 vs target -7.2137), so we should make a small, metric-aligned improvement without changing your model, loss, or training-loop structure. The safest gain here is to fix the cross-validation training bug: right now you reuse the same model/optimizer across folds, so later folds are not true folds and OOF calibration becomes inconsistent, hurting both FVC and Confidence quality. I minimally reset the model and optimizer at the start of each fold (same architecture, same optimizer settings, same 100-epoch loop) and keep collecting fold-held-out OOF predictions. Then we keep your existing 1D grid-search calibration for `sigma_scale` and submission generation unchanged.'

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
data = pd.concat([train, subms, test], axis=0, ignore_index=True)
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
        l = self.left(h)
        r = self.right(h)
        h = l * self.sigmoid(r)
        p1 = self.last(h)
        p2 = self.lastRelu(h)
        out = p1 + pt.cumsum(p2, 1)
        return out


model = Model()

C1, C2 = pt.tensor([70.0], dtype=pt.float32, device=device), pt.tensor(
    [1000.0], dtype=pt.float32, device=device
)


def score(y_pred, y_true):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = pt.max(sigma, C1.expand_as(sigma))
    delta = pt.abs(y_true - fvc_pred)
    delta = pt.min(delta, C2)
    sq2 = pt.sqrt(pt.tensor([2.0], dtype=pt.float32, device=y_pred.device))
    metric = (delta / sigma_clip) * sq2 + pt.log(sigma_clip * sq2)
    return pt.mean(metric)


def pinballLoss(pred, label, quant):
    err = label - pred
    m = pt.mean(pt.max(quant * err, (quant - 1) * err))
    return m


def loss(pred, label):
    quantiles = pt.tensor([0.2, 0.5, 0.8], dtype=pred.dtype, device=pred.device)
    return 0.8 * pinballLoss(pred, label, quantiles) + 0.2 * score(pred, label)




## === cell 2
trainDataSet = OSICDataSet(train)

losses = []
valLosses = []
kf = KFold(n_splits=5)

submDataSet = OSICDataSet(subms, "submission")
inp = pt.from_numpy(submDataSet.data[inputs].values.astype(np.float32)).to(device)

oof_fvc = pt.empty((len(trainDataSet),), dtype=pt.float32, device=device)
oof_sigma = pt.empty((len(trainDataSet),), dtype=pt.float32, device=device)
oof_y = (
    pt.from_numpy(trainDataSet.data[["FVC"]].values.astype(np.float32))
    .to(device)
    .view(-1)
)

c = 0
for fold, (train_index, val_index) in enumerate(kf.split(trainDataSet)):
    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5, eps=1e-8)

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
            pred_val = model(val_x)
            valLos = loss(pred_val, val_y)
            losses.append(los)
            valLosses.append(valLos)
        c += 1

    with pt.no_grad():
        pred_va = model(val_x).detach()
        sigma_va = (pred_va[:, 2] - pred_va[:, 0]).clamp_min(1e-3)
        fvc_va = pred_va[:, 1]
        oof_fvc[val_index] = fvc_va
        oof_sigma[val_index] = sigma_va





## === cell 3
sns.set(style="white", palette="muted", color_codes=True)

losses_plot = [
    float(x.detach().cpu().item()) if pt.is_tensor(x) else float(x) for x in losses
]
valLosses_plot = [
    float(x.detach().cpu().item()) if pt.is_tensor(x) else float(x) for x in valLosses
]

plt.plot(losses_plot, label="Train loss")
plt.plot(valLosses_plot, label="Val loss")
plt.legend()
plt.show()

asd = pt.from_numpy(trainDataSet.data[inputs].values.astype(np.float32)).to(device)
pred = model(asd).detach()
idxs = np.random.randint(0, len(trainDataSet), 100)
plt.plot(
    trainDataSet.data.loc[idxs, ["FVC"]].values.astype(np.float32), label="ground truth"
)

plt.plot(pred[idxs, 0].detach().cpu().numpy(), label="q25")
plt.plot(pred[idxs, 1].detach().cpu().numpy(), label="q50")
plt.plot(pred[idxs, 2].detach().cpu().numpy(), label="q75")
plt.legend(loc="best")
plt.show()

c_arr = (pred[:, 2] - pred[:, 0]).detach().cpu().numpy()
f_arr = pred[:, 1].detach().cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
sns.distplot(
    c_arr, color="g", kde=False, kde_kws={"shade": True}, ax=axes[0]
).set_title("Predicted Confidence on train set")
sns.distplot(
    f_arr, color="g", kde=False, kde_kws={"shade": True}, ax=axes[1]
).set_title("Predicted FVC on train set")
plt.show()

with pt.no_grad():
    resid_oof = (oof_y - oof_fvc).abs().clamp_max(1000.0)  # Delta in the metric
    base_sigma = oof_sigma.clamp_min(1e-3)

    scales = pt.linspace(0.5, 5.0, steps=91, device=device)  # step=0.05
    sq2 = pt.sqrt(pt.tensor([2.0], dtype=pt.float32, device=device))
    sigma_scaled = base_sigma.unsqueeze(0) * scales.unsqueeze(1)
    sigma_clip = pt.maximum(
        sigma_scaled, pt.tensor([70.0], dtype=pt.float32, device=device)
    )
    delta = resid_oof.unsqueeze(0)

    metric = -(sq2 * delta) / sigma_clip - pt.log(sq2 * sigma_clip)
    mean_metric = metric.mean(dim=1)

    best_idx = int(pt.argmax(mean_metric).detach().cpu().item())
    sigma_scale = float(scales[best_idx].detach().cpu().item())

submDataSet = OSICDataSet(subms, "submission")
inp = pt.from_numpy(submDataSet.data[inputs].values.astype(np.float32)).to(device)

out = model(inp).detach()

out[:, 2] = pt.max(out[:, 2], out[:, 0] + 1e-3)

sigma = (out[:, 2] - out[:, 0]).detach().cpu().numpy()
sigma = (sigma * sigma_scale).astype(np.float32)

fvc = out[:, 1].detach().cpu().numpy()

confidence = np.maximum(sigma, 70.0).astype(np.float32)

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
sns.distplot(
    confidence, color="g", kde=False, kde_kws={"shade": True}, ax=axes[0]
).set_title("Confidence on submission set")
sns.distplot(fvc, color="g", kde=False, kde_kws={"shade": True}, ax=axes[1]).set_title(
    "FVC on submision set"
)
plt.show()

submDataSet.data["pFVC"] = fvc
submDataSet.data["cConf"] = confidence

submission = pd.DataFrame(
    {
        "Patient_Week": submDataSet.data.Patient_Week,
        "FVC": fvc.astype(np.float32),
        "Confidence": confidence,
    }
)

otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")

for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    submission.loc[submission["Patient_Week"] == key, "FVC"] = otest.FVC[i]

submission["Confidence"] = submission["Confidence"].astype(np.float32)
submission["Confidence"] = np.maximum(submission["Confidence"].values, 70.0).astype(
    np.float32
)

submission.to_csv("submission.csv", index=False)
null_columns = submission.columns[submission.isnull().any()]
print("sigma_scale_used:", sigma_scale)
print(submission.head())
print(len(submission))
