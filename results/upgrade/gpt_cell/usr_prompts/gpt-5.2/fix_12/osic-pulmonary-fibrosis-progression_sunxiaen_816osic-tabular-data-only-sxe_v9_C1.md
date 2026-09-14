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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

-7.0136

# 6. Current score

-7.8752

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.65931) has done: 'Diagnosis: Cell 31 hard-fails because it requires a trained checkpoint in `/kaggle/working` (or pre-existing `model/models` variables), but in this environment no such file exists and earlier training cells were not run/provided. This makes inference impossible even though the rest of the code (dataset + model definition) can execute. The minimal way to unblock execution without changing model architecture or inference semantics is to fall back to an untrained `QuantModel` when no checkpoint is found, so the submission pipeline can still run end-to-end.

Patch summary: Modify only cell 31 to remove the `raise RuntimeError(...)` and instead instantiate a fresh `QuantModel` if no checkpoint is available. If a checkpoint is found, load it exactly as before. This keeps variables (`models`, `avg_preds`, `df`, `submission.csv`) and downstream behavior intact, differing only in that predictions be untrained when no weights exist.

Updated cells: (cell 31 only)

Compatibility notes for cell k+1: No interface changes; outputs (`submission.csv` with columns `Patient_Week`, `FVC`, `Confidence`) and variable names remain the same.

Assumptions: It is acceptable to generate a runnable submission file even without trained weights, because training artifacts are absent in the provided environment and the goal is to fix the crash with minimal changes.'
- What this solution (achieved -7.8639) has done: 'Your current score is far below the target (gap ≈ -17.65, and higher is better), so we should improve performance with the smallest changes that keep your core model and quantile setup intact. The biggest issue is that inference is currently using an (effectively) untrained model in this environment, which score extremely poorly; the minimal legitimate fix is to train the same `QuantModel` for a short, fixed number of epochs on the provided tabular features before generating predictions. To preserve your core logic, we keep the exact architecture, quantile loss, and metric; we only add a compact training block using the existing `Monitor` helper and save/load the same `model.pth`. Finally, we make the output `Confidence` robust by applying the competition’s clipping rule (≥70) to avoid pathological sigma values hurting the metric.'
- What this solution (achieved -7.85144) has done: 'We’re already moving in the right direction (training the same model instead of using random weights), but the remaining gap to the target is still meaningful, so the smallest reliable gain is to make the training split deterministic and patient-stratified (so validation and saved “best” weights are more consistent) and to ensure the quantile outputs always produce a valid, non-degenerate interval (so `Confidence` behaves well under the Laplace metric). I keep your exact model, quantile loss, optimizer, scheduler, and epoch count unchanged, and only adjust (1) the group split to be reproducible via `random_state`, (2) the model’s quantile outputs during inference to enforce monotonic quantiles (q25 ≤ q50 ≤ q75), and (3) the confidence computation to match the same monotonic-adjusted interval with the required ≥70 clipping. These changes legitimately improve metric alignment without changing the core approach, and they should move the score upward toward the target band.'
- What this solution (achieved -7.91253) has done: 'Your current score (-7.85144) is below the target (-7.0136), so we should make a small, low-risk improvement that nudges the Laplace-LL metric upward without changing your model or training loop. The biggest “free” gain here is fixing the metric alignment: your `Confidence` should approximate the Laplace scale (σ), and for a normal-like predictive interval the conversion from IQR to σ is `σ ≈ IQR / 1.349`. I keep your exact QuantModel, quantile loss, optimizer, scheduler, epoch count, and split logic, and only adjust the final confidence computation accordingly (still clipped at ≥70). I also make the saved “best” model selection consistent with the competition metric (maximize val metric rather than minimize quantile loss) while keeping early stopping disabled and training length unchanged.'
- What this solution (achieved -7.8752) has done: 'We’re still below the target (current −7.91253 vs target −7.0136; higher is better), so we should make the smallest metric-aligned change that improves generalization without changing your model or training loop. The safest gain here is to remove a subtle distribution shift in your tabular encoding: you currently build one-hot columns using `data[col].unique()` order, which can vary and may include `NaN`, and you also set `in_tabular_features=9` even though the number of generated one-hot categories can differ across environments. I make the one-hot schema deterministic and identical between train/test by (1) defining fixed category lists from the *train* dataframe (with an explicit “Unknown” bucket), and (2) passing the resulting feature dimension into `QuantModel` so inputs match exactly (no semantic change—same features, just consistently ordered). This typically nudges the score upward toward your target by preventing misaligned features and stabilizing training/inference.'

# 9. Code solution

## === cell 0
import copy
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from tqdm.notebook import trange


class SummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_images(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass

    def add_graph(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


root_dir = Path("/kaggle/input/osic-pulmonary-fibrosis-progression")
model_dir = Path("/kaggle/working")




## === cell 1
class ClinicalDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.transform = transform
        self.mode = mode

        tst = pd.read_csv(Path(root_dir) / "test.csv")
        tst["minweek"] = tst["Weeks"]
        tst["minweek_FVC"] = tst["FVC"]

        sub = pd.read_csv(Path(root_dir) / "sample_submission.csv")
        sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
        sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
        sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
        sub = sub.merge(tst.drop("Weeks", axis=1), on="Patient", how="inner")
        sub["base_week"] = sub["Weeks"] - sub["minweek"]

        tr = pd.read_csv(Path(root_dir) / "train.csv")
        tr.drop_duplicates(keep="first", inplace=True, subset=["Patient", "Weeks"])

        tr["minweek"] = tr.groupby("Patient")["Weeks"].transform(min)
        base = tr.loc[tr.Weeks == tr.minweek]
        base = base[["Patient", "FVC"]]
        base.columns = ["Patient", "minweek_FVC"]
        tr = tr.merge(base, on="Patient", how="inner")
        tr["base_week"] = tr["Weeks"] - tr["minweek"]

        data = pd.concat([tr, sub], axis=0, ignore_index=True, sort=False)

        CAT_COLS = ["Sex", "SmokingStatus"]
        train_df = tr[CAT_COLS].copy()
        categories = {}
        for col in CAT_COLS:
            cats = train_df[col].fillna("Unknown").astype(str).unique().tolist()
            cats = sorted(set(cats + ["Unknown"]))
            categories[col] = cats

        for col in CAT_COLS:
            data[col] = data[col].fillna("Unknown").astype(str)

        self.FE = []
        for col in CAT_COLS:
            for cat in categories[col]:
                feat_name = f"{col}__{cat}"
                self.FE.append(feat_name)
                data[feat_name] = (data[col] == cat).astype(np.int8)

        data["N_age"] = (data["Age"] - data["Age"].min()) / (
            data["Age"].max() - data["Age"].min()
        )
        data["N_base_minweek_FVC"] = (
            data["minweek_FVC"] - data["minweek_FVC"].min()
        ) / (data["minweek_FVC"].max() - data["minweek_FVC"].min())
        data["N_week"] = (data["base_week"] - data["base_week"].min()) / (
            data["base_week"].max() - data["base_week"].min()
        )
        data["N_percent"] = (data["Percent"] - data["Percent"].min()) / (
            data["Percent"].max() - data["Percent"].min()
        )

        self.FE += ["N_age", "N_percent", "N_week", "N_base_minweek_FVC"]

        if self.mode == "train":
            self.raw = data[data["Patient_Week"].isna()].reset_index()
        elif self.mode == "test":
            self.raw = data[~data["Patient_Week"].isna()].reset_index()

        del base
        del data

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            "patient_id": self.raw["Patient"].iloc[idx],
            "features": self.raw[self.FE].iloc[idx].values,
            "target": self.raw["FVC"].iloc[idx],
        }
        if self.transform:
            sample = self.transform(sample)

        return sample

    def group_kfold(self, n_splits):
        gkf = GroupKFold(n_splits=n_splits)
        groups = self.raw["Patient"]
        for train_idx, val_idx in gkf.split(self.raw, self.raw, groups):
            train = Subset(self, train_idx)
            val = Subset(self, val_idx)
            yield train, val

    def group_split(self, test_size=0.1, random_state=42):
        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        groups = self.raw["Patient"]
        idx = list(gss.split(self.raw, self.raw, groups))
        train = Subset(self, idx[0][0])
        val = Subset(self, idx[0][1])
        return train, val




## === cell 2
import torch.nn as nn
import torch.nn.functional as F


class QuantModel(nn.Module):
    def __init__(self, in_tabular_features=9, out_quantiles=3):
        super(QuantModel, self).__init__()
        self.fc1 = nn.Linear(in_tabular_features, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 9)
        self.fc4 = nn.Linear(9, out_quantiles)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return x


quantiles = (0.25, 0.5, 0.75)


def quantile_loss(preds, target, quantiles):
    assert not target.requires_grad
    assert preds.size(0) == target.size(0)
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss




## === cell 3
class Monitor:
    def __init__(
        self,
        model,
        es_patience,
        experiment_name,
        tensorboard_dir,
        num_epochs,
        dataset_sizes,
        model_file,
    ):
        self.model = model
        self.model_file = model_file
        self.es_patience = es_patience
        self.tensorboard_dir = tensorboard_dir
        self.dataset_sizes = dataset_sizes
        date_time = datetime.now().strftime("%Y%m%d-%H%M")
        log_dir = tensorboard_dir / f"{experiment_name}-{date_time}"
        self.w = SummaryWriter(log_dir)

        self.bar = trange(num_epochs, desc=experiment_name)

        self.epoch_loss = {"train": np.inf, "val": np.inf}
        self.epoch_metric = {"train": -np.inf, "val": -np.inf}

        self.best_metric = -np.inf
        self.best_model_wts = None

        self.e = {"train": 0, "val": 0}
        self.t = {"train": 0, "val": 0}
        self.running_loss = 0.0
        self.running_metric = 0.0
        self.es_counter = 0

    def reset_epoch(self):
        self.running_loss = 0.0
        self.running_metric = 0.0

    def step(self, loss, inputs, preds, targets, phase):
        self.running_loss += loss.item() * inputs.size(0)
        self.running_metric += self.metric(preds, targets).sum()
        self.t[phase] += 1

    def log_epoch(self, phase):
        self.epoch_loss[phase] = self.running_loss / self.dataset_sizes[phase]
        self.epoch_metric[phase] = self.running_metric / self.dataset_sizes[phase]
        self.bar.set_postfix(
            a_train_loss=f'{self.epoch_loss["train"]:0.1f}',
            b_val_loss=f'{self.epoch_loss["val"]:0.1f}',
            c_train_metric=f'{self.epoch_metric["train"]:0.4f}',
            d_val_metric=f'{self.epoch_metric["val"]:0.4f}',
            es_counter=self.es_counter,
        )
        self.w.add_scalar(f"Loss/{phase}", self.epoch_loss[phase], self.e[phase])
        self.w.add_scalar(f"Accuracy/{phase}", self.epoch_metric[phase], self.e[phase])

        self.e[phase] += 1

        early_stop = False
        if phase == "val":
            if self.epoch_metric["val"] > self.best_metric:
                self.best_metric = float(self.epoch_metric["val"])
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_model_wts, self.model_file)
                self.es_counter = 0
            else:
                self.es_counter += 1
                if self.es_counter >= self.es_patience:
                    early_stop = True
                    self.bar.close()

        return early_stop

    @staticmethod
    def metric(preds, targets):
        sigma = preds[:, 2] - preds[:, 0]
        sigma[sigma < 70] = 70
        delta = (preds[:, 1] - targets).abs()
        delta[delta > 1000] = 1000
        return -np.sqrt(2) * delta / sigma - torch.log(np.sqrt(2) * sigma)




## === cell 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if "models" in globals() and models is not None and len(models) > 0:
    pass
elif "model" in globals() and model is not None:
    models = [model]
else:
    model_file = model_dir / "model.pth"

    tmp_train_ds = ClinicalDataset(root_dir, mode="train")
    inferred_in_features = len(tmp_train_ds.FE)
    del tmp_train_ds

    model = QuantModel(in_tabular_features=inferred_in_features, out_quantiles=3).to(
        device
    )

    if model_file.exists():
        state = torch.load(model_file, map_location=device)
        model.load_state_dict(state)
    else:
        torch.manual_seed(42)
        np.random.seed(42)

        full_train = ClinicalDataset(root_dir, mode="train")
        train_ds, val_ds = full_train.group_split(test_size=0.1, random_state=42)

        batch_size = 128
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=2
        )

        optimizer = Adam(model.parameters(), lr=3e-3)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

        num_epochs = 30
        dataset_sizes = {"train": len(train_ds), "val": len(val_ds)}
        monitor = Monitor(
            model=model,
            es_patience=1000000,  # keep "no early stopping"
            experiment_name="quantile_tabular",
            tensorboard_dir=model_dir,
            num_epochs=num_epochs,
            dataset_sizes=dataset_sizes,
            model_file=model_file,
        )

        for epoch in range(num_epochs):
            model.train()
            monitor.reset_epoch()
            for batch in train_loader:
                inputs = batch["features"].float().to(device)
                targets = batch["target"].float().to(device)

                optimizer.zero_grad(set_to_none=True)
                preds = model(inputs)
                loss = quantile_loss(preds, targets, quantiles)
                loss.backward()
                optimizer.step()

                monitor.step(
                    loss, inputs, preds.detach(), targets.detach(), phase="train"
                )
            monitor.log_epoch("train")

            model.eval()
            monitor.reset_epoch()
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["features"].float().to(device)
                    targets = batch["target"].float().to(device)
                    preds = model(inputs)
                    loss = quantile_loss(preds, targets, quantiles)
                    monitor.step(loss, inputs, preds, targets, phase="val")
            monitor.log_epoch("val")

            scheduler.step()

        if model_file.exists():
            best_state = torch.load(model_file, map_location=device)
            model.load_state_dict(best_state)

    models = [model]

for m in models:
    m.to(device)
    m.eval()

try:
    batch_size  # noqa: F401
except NameError:
    batch_size = 128

data = ClinicalDataset(root_dir, mode="test")
avg_preds = np.zeros((len(data), len(quantiles)), dtype=np.float64)

for model in models:
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2)
    preds = []
    for batch in dataloader:
        inputs = batch["features"].float().to(device)
        with torch.no_grad():
            x = model(inputs)
            preds.append(x)
    preds = torch.cat(preds, dim=0).cpu().numpy()
    avg_preds += preds

avg_preds /= len(models)

q25 = avg_preds[:, 0].copy()
q50 = avg_preds[:, 1].copy()
q75 = avg_preds[:, 2].copy()
q50 = np.maximum(q50, q25)
q75 = np.maximum(q75, q50)
avg_preds = np.stack([q25, q50, q75], axis=1)

df = pd.DataFrame(data=avg_preds, columns=list(quantiles))
df["Patient_Week"] = data.raw["Patient_Week"]
df["FVC"] = df[quantiles[1]]

iqr = (df[quantiles[2]] - df[quantiles[0]]).astype(np.float64)
sigma_est = iqr / 1.349
df["Confidence"] = np.maximum(sigma_est, 70.0)

df = df.drop(columns=list(quantiles))
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
