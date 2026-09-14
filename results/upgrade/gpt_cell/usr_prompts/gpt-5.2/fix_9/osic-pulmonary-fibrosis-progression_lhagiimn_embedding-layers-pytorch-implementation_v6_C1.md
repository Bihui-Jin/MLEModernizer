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

No external packages required in the script and installed.

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

-6.8327

# 6. Current score

-8.11962

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.19464) has done: 'Your notebook currently never reaches a prediction/submission stage and also redefines the model class inconsistently; I keep the same core quantile-regression NN and features, but make the pipeline run end-to-end and always write a valid `submission.csv`. I (1) remove the duplicate `model_nn` definition to avoid subtle embedding shape bugs, (2) fix the early-stopping call to monitor validation loss (not the metric) so checkpointing behaves as intended without changing the training approach, and (3) add a minimal training/prediction block that trains on train rows and predicts `FVC`/`Confidence` for the provided `sample_submission.csv` rows using the metric-compatible confidence derived from the predicted quantiles. These changes are directly aimed at producing a valid submission and improving score versus “no submission”, while keeping your model/loss/features intact.'
- What this solution (achieved -8.19464) has done: 'Your current solution is close to correct but loses score mainly because the predicted quantiles are unconstrained (so q20/q50/q80 can cross), which makes `|q80-q20|` an unreliable confidence and hurts the Laplace-LL metric. I keep the same model, features, training loop, and quantile loss, but enforce monotonic quantiles at inference by sorting the three outputs per row before computing `FVC=q50` and `Confidence=q80-q20`. I also clip confidence to the competition’s effective minimum (70) and use a small scaling factor on the width (1.0 by default) so confidence isn’t systematically too small; this typically improves score when your current score is below target. These are minimal, metric-aligned post-processing changes that don’t alter training semantics and should move the score toward the -6.8327 target.'
- What this solution (achieved -8.13547) has done: 'Your current gap to the target is large (−8.1946 vs −6.8327, higher is better), so the smallest safe gain is to make the model’s confidence better aligned with the Laplace-LL metric without changing training or architecture. I keep your quantile-regression NN, features, split, and training loop intact, but (1) prevent a subtle categorical-encoding bug by fitting one `OrdinalEncoder` across all categorical columns at once (instead of refitting per column), which can otherwise make category indices inconsistent and harm generalization. Then (2) apply a small calibration on `Confidence` using the validation set to pick a single scale factor for `(q80-q20)` that minimizes the competition metric proxy, improving score while keeping prediction semantics. Finally, I keep the monotonic quantile sorting and enforce the metric’s effective minimum confidence (70) as you already do, ensuring a valid `submission.csv`.'
- What this solution (achieved -8.1402) has done: 'Your current score is below the target, so we should nudge performance upward with the smallest, metric-aligned changes that don’t touch the model, loss, or training loop. The main safe gain here is to calibrate the predicted confidence more robustly: the metric is very sensitive to miscalibrated σ, and your current single-scale search can be noisy. I (1) select the confidence scale using a more direct metric-proxy on the validation set (same proxy you already use, but with a slightly finer grid and a safe “shrink toward 1.0” to avoid overfitting), and (2) add a tiny additive floor to the width before clipping to reduce the chance of under-confident predictions (still clipped at 70, so it doesn’t break semantics). This keeps your architecture/features/training identical, only adjusting inference-time confidence calibration to move the score toward the target.'
- What this solution (achieved -8.11982) has done: 'We keep your exact model, loss, features, and training loop, and only adjust the inference-time confidence calibration because your score is still below target and the metric is highly sensitive to σ. Specifically, we (1) fix the calibration proxy to use the predicted median (q50) rather than q20 (your current proxy unintentionally shifts the predicted FVC and misguides scale selection), (2) search a slightly wider but still small grid for the confidence scale to better match the metric without changing training, and (3) keep your monotonic-quantile sorting and the σ≥70 clipping unchanged. These are minimal, metric-aligned changes that should move the score upward toward the -6.8327 target.'
- What this solution (achieved -8.11962) has done: 'Your current score is still below the target (−8.1198 vs −6.8327; higher is better), so we should only make metric-aligned, inference-time adjustments that can safely improve without changing your model, loss, features, or training loop. The main remaining low-risk lever is better confidence calibration: the Laplace-LL is extremely sensitive to σ, and a single point-estimate scale can be noisy. I keep your existing scale search but (1) switch it to a slightly more robust “patient-level” proxy averaging (reduces overfitting to patients with many rows), and (2) add a tiny, validation-derived multiplicative correction for the additive floor (CONF_ADD) while keeping the same concept and constraints (still clipped at ≥70). These are minimal changes meant to move score upward toward the target band while preserving core semantics.'

# 9. Code solution

## === cell 0
import os
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import OrdinalEncoder

warnings.simplefilter("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
quantiles = [0.2, 0.5, 0.8]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)




## === cell 1
def make_X(dt, dense_cols, cat_feats):
    X = {"dense": dt[dense_cols].to_numpy()}
    for v in cat_feats:
        X[v] = dt[[v]].to_numpy()
    return X


class Loader:
    def __init__(self, X, y, shuffle=True, batch_size=64, cat_cols=[]):
        self.X_cont = X["dense"]
        self.X_cat = (
            np.concatenate([X[k] for k in cat_cols], axis=1)
            if len(cat_cols)
            else np.zeros((len(self.X_cont), 0))
        )
        self.y = y

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.len = self.X_cont.shape[0]

        n_batches, remainder = divmod(self.len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

        self.idxes = np.arange(self.len)

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            ridxes = self.idxes.copy()
            np.random.shuffle(ridxes)
            self.X_cat = self.X_cat[ridxes]
            self.X_cont = self.X_cont[ridxes]
            if self.y is not None:
                self.y = self.y[ridxes]
        return self

    def __next__(self):
        if self.i >= self.len:
            raise StopIteration

        if self.y is not None:
            y = torch.FloatTensor(
                self.y[self.i : self.i + self.batch_size].astype(np.float32)
            )
        else:
            y = None

        xcont = torch.FloatTensor(
            self.X_cont[self.i : self.i + self.batch_size].astype(np.float32)
        )
        xcat = torch.LongTensor(
            self.X_cat[self.i : self.i + self.batch_size].astype(np.int64)
        )

        batch = (xcont, xcat, y)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches




## === cell 2
class model_nn(nn.Module):
    """
    Keep a single consistent definition.
    NOTE: This matches embedding usage for (batch, n_cat) LongTensor input.
    """

    def __init__(self, hidden_dim, output_dim, emb_dims, n_cont):
        super().__init__()
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        n_embs = sum([y for _, y in emb_dims])
        inp_dim = n_embs + n_cont

        self.fc0 = nn.Linear(inp_dim, hidden_dim)
        self.relu0 = nn.ELU()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu1 = nn.ELU()

        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def encode_and_combine_data(self, cat_data):
        xcat = [emb(cat_data[:, k]) for k, emb in enumerate(self.emb_layers)]
        return torch.cat(xcat, dim=1)

    def forward(self, cont_data, cat_data):
        cont_data = cont_data.to(device)
        cat_data = cat_data.to(device)

        cat_emb = (
            self.encode_and_combine_data(cat_data)
            if len(self.emb_layers)
            else torch.zeros((cont_data.size(0), 0), device=cont_data.device)
        )
        x = torch.cat([cont_data, cat_emb], dim=1)

        hz = self.relu0(self.fc0(x))
        hz = self.relu1(self.fc1(hz))

        out1 = self.fc2(hz)
        out2 = self.fc3(hz)
        out3 = self.fc4(hz)
        return torch.cat([out1, out2, out3], dim=1)




## === cell 3
def quantile_loss(preds, target, quantiles):
    assert not target.requires_grad
    assert preds.size(0) == target.size(0)
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss


def metric(outputs, target):
    confidence = np.abs(outputs[:, 2] - outputs[:, 0])
    clip = np.where(confidence > 70, confidence, 70)
    delta = np.abs(outputs[:, 1] - target)
    delta = np.where(delta > 1000, 1000, delta)
    metrics = (delta * np.sqrt(2) / clip) + np.log(clip * np.sqrt(2))
    return np.mean(metrics)




## === cell 4
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def model_training(
    model,
    train_loader,
    val_loader,
    epochs,
    lr=0.001,
    patience=10,
    model_path="model.pth",
):
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.4, verbose=False
    )

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_cont, X_cat, y in train_loader:
            preds = model(X_cont, X_cat)
            loss = quantile_loss(preds, y.to(device), quantiles)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_preds = []
        true_y = []
        with torch.no_grad():
            for X_cont, X_cat, y in val_loader:
                preds = model(X_cont, X_cat)
                loss = quantile_loss(preds, y.to(device), quantiles)
                val_loss += loss.item() / len(val_loader)
                val_preds.append(preds.detach().cpu())
                true_y.append(y.detach().cpu())

        val_preds_np = torch.cat(val_preds, dim=0).numpy()
        true_y_np = torch.cat(true_y, dim=0).numpy()
        val_metric = metric(val_preds_np, true_y_np)

        print(
            f"[valid] Epoch: {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val metric(proxy): {val_metric:.4f}"
        )

        early_stopping(val_loss, model, path=model_path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler.step(val_loss)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model




## === cell 5
ROOT = "../input/osic-pulmonary-fibrosis-progression"

tr = pd.read_csv(f"{ROOT}/train.csv")
tr.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])
te = pd.read_csv(f"{ROOT}/test.csv")

sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub = sub.merge(te.drop("Weeks", axis=1), on="Patient")

tr["WHERE"] = "train"
te["WHERE"] = "val"
sub["WHERE"] = "test"

data = pd.concat([tr, te, sub], axis=0, ignore_index=False)

data["min_week"] = data["Weeks"]
data.loc[data.WHERE == "test", "min_week"] = np.nan
data["min_week"] = data.groupby("Patient")["min_week"].transform("min")

base = data.loc[data.Weeks == data.min_week]
base = base[["Patient", "FVC", "Percent"]].copy()
base.columns = ["Patient", "min_FVC", "min_percent"]
base["nb"] = 1
base["nb"] = base.groupby("Patient")["nb"].transform("cumsum")
base = base[base.nb == 1]
base.drop("nb", axis=1, inplace=True)

data = data.merge(base, on="Patient", how="left")
data["base_week"] = data["Weeks"] - data["min_week"]
del base

COLS = ["Sex", "SmokingStatus"]
FE = []
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)

data["age_week"] = data["Age"].values + data["base_week"].values / 53

data["age"] = (data["Age"] - data["Age"].min()) / (
    data["Age"].max() - data["Age"].min()
)
data["BASE"] = (data["min_FVC"] - data["min_FVC"].min()) / (
    data["min_FVC"].max() - data["min_FVC"].min()
)
data["week"] = (data["base_week"] - data["base_week"].min()) / (
    data["base_week"].max() - data["base_week"].min()
)
data["min_percent_norm"] = (data["min_percent"] - data["min_percent"].min()) / (
    data["min_percent"].max() - data["min_percent"].min()
)
data["age_week_norm"] = (data["age_week"] - data["age_week"].min()) / (
    data["age_week"].max() - data["age_week"].min()
)

cat_feat = ["Sex", "SmokingStatus"]

enc = OrdinalEncoder(dtype=np.int64)
data[cat_feat] = enc.fit_transform(data[cat_feat]).astype(np.int64)

uniques = [int(data[v].nunique()) for v in cat_feat]

tr = data.loc[data.WHERE == "train"].copy()
te = data.loc[data.WHERE == "val"].copy()
sub_df = data.loc[data.WHERE == "test"].copy()

FE += ["age", "week", "BASE", "min_percent_norm", "age_week_norm"]

dims = [1, 2]
emb_dims = [(x, y) for x, y in zip(uniques, dims)]
n_cont = len(FE)



## === cell 6
patients = tr["Patient"].unique()
rng = np.random.RandomState(SEED)
rng.shuffle(patients)
cut = int(0.8 * len(patients))
train_pats = set(patients[:cut])
valid_pats = set(patients[cut:])

tr_tr = tr[tr["Patient"].isin(train_pats)].copy()
tr_va = tr[tr["Patient"].isin(valid_pats)].copy()

dense_cols = FE
cat_cols = cat_feat

X_train = make_X(tr_tr, dense_cols, cat_cols)
y_train = tr_tr["FVC"].values.reshape(-1, 1)

X_valid = make_X(tr_va, dense_cols, cat_cols)
y_valid = tr_va["FVC"].values.reshape(-1, 1)

train_loader = Loader(X_train, y_train, shuffle=True, batch_size=64, cat_cols=cat_cols)
val_loader = Loader(X_valid, y_valid, shuffle=False, batch_size=256, cat_cols=cat_cols)

model = model_nn(hidden_dim=128, output_dim=1, emb_dims=emb_dims, n_cont=n_cont).to(
    device
)

model_path = "model.pth"
model = model_training(
    model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    lr=1e-3,
    patience=10,
    model_path=model_path,
)


def predict_quantiles(loader, model):
    model.eval()
    out = []
    ytrue = []
    with torch.no_grad():
        for X_cont, X_cat, y in loader:
            p = model(X_cont, X_cat).detach().cpu().numpy()
            out.append(p)
            if y is not None:
                ytrue.append(y.detach().cpu().numpy())
    out = np.vstack(out)
    ytrue = np.vstack(ytrue).reshape(-1) if len(ytrue) else None
    return out, ytrue


val_raw, val_y = predict_quantiles(val_loader, model)
val_sorted = np.sort(val_raw, axis=1)
val_q20, val_q50, val_q80 = val_sorted[:, 0], val_sorted[:, 1], val_sorted[:, 2]

val_patients = tr_va["Patient"].values


def metric_patient_mean(outputs, target, patients_arr):
    df = pd.DataFrame(
        {
            "Patient": patients_arr,
            "m": (
                np.abs(outputs[:, 1] - target).clip(0, 1000)
                * np.sqrt(2)
                / np.maximum(np.abs(outputs[:, 2] - outputs[:, 0]), 70.0)
            )
            + np.log(
                np.maximum(np.abs(outputs[:, 2] - outputs[:, 0]), 70.0) * np.sqrt(2)
            ),
        }
    )
    return df.groupby("Patient")["m"].mean().mean()


scales = np.array(
    [
        0.60,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
        1.05,
        1.10,
        1.20,
        1.30,
        1.50,
        1.80,
        2.20,
    ],
    dtype=np.float32,
)
best_scale = 1.0
best_proxy = np.inf

BASE_CONF_ADD = 10.0
add_muls = np.array([0.75, 1.0, 1.25, 1.5], dtype=np.float32)

val_width = (val_q80 - val_q20).astype(np.float32)

best_add_mul = 1.0
for s in scales:
    for am in add_muls:
        conf_add = float(BASE_CONF_ADD * am)
        conf = np.maximum(val_width * float(s) + conf_add, 70.0).astype(np.float32)
        proxy_outputs = np.stack(
            [val_q50 - conf / 2.0, val_q50, val_q50 + conf / 2.0], axis=1
        )
        proxy = metric_patient_mean(proxy_outputs, val_y, val_patients)
        if proxy < best_proxy:
            best_proxy = proxy
            best_scale = float(s)
            best_add_mul = float(am)

SHRINK = 0.25
best_scale = (1.0 - SHRINK) * best_scale + SHRINK * 1.0

CONF_ADD = float(BASE_CONF_ADD * best_add_mul)

print(
    f"Selected CONF_SCALE={best_scale:.3f} (shrunk), CONF_ADD={CONF_ADD:.1f} (val patient-mean proxy={best_proxy:.4f})"
)

X_test = make_X(sub_df, dense_cols, cat_cols)
test_loader = Loader(X_test, y=None, shuffle=False, batch_size=256, cat_cols=cat_cols)

preds_all, _ = predict_quantiles(test_loader, model)
preds_sorted = np.sort(preds_all, axis=1)
q20 = preds_sorted[:, 0]
q50 = preds_sorted[:, 1]
q80 = preds_sorted[:, 2]

fvc_pred = q50
conf_pred = np.maximum((q80 - q20) * best_scale + CONF_ADD, 70.0)

submission = pd.DataFrame(
    {
        "Patient_Week": sub_df["Patient_Week"].values,
        "FVC": fvc_pred.astype(float),
        "Confidence": conf_pred.astype(float),
    }
)

sample = pd.read_csv(f"{ROOT}/sample_submission.csv")[["Patient_Week"]]
submission = sample.merge(submission, on="Patient_Week", how="left")

if submission["FVC"].isna().any():
    fallback = sub_df.groupby("Patient")["min_FVC"].first().to_dict()
    pw_pat = submission["Patient_Week"].str.split("_").str[0]
    submission.loc[submission["FVC"].isna(), "FVC"] = pw_pat.map(fallback).astype(float)
    submission.loc[submission["Confidence"].isna(), "Confidence"] = 70.0

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
