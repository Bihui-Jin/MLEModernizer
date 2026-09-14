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

-9.7219

# 6. Current score

-8.71947

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.15394) has done: 'Your script currently can’t yield a Kaggle score because it never reaches a prediction/submission-writing stage (it ends mid-definition and also has a model save/load bug). I make the smallest changes needed to run end-to-end and produce a valid `submission.csv` with the required columns and correct row alignment to `sample_submission.csv`. To move the score upward toward the target, I keep your quantile model and training loop intact, but fix checkpointing to save/load `state_dict` correctly and ensure validation uses the intended criterion (your EarlyStopping is currently fed the metric but treats it like a loss). Finally, I generate `FVC` from the median quantile and `Confidence` from the quantile spread (clipped to Kaggle’s minimum 70) which matches your model’s semantics and the competition metric.'
- What this solution (achieved -8.23748) has done: 'Your current score (-8.15394) is better than the target (-9.7219), so to move closer to the target we should slightly reduce performance while keeping the same model/training logic. The smallest safe lever here is prediction calibration: inflate the predicted `Confidence` (σ) modestly above your quantile-spread estimate, which directly reduces the Laplace NLL score magnitude (more negative) without changing the model, loss, or training loop. I also make the confidence computation consistent with the metric’s σ-clipping by applying the scale before the `max(70)` clip, and keep submission alignment identical. This should move the score downward (worse) toward the target band with minimal risk and minimal code changes.'
- What this solution (achieved -8.48676) has done: 'Your current score (-8.23748) is better than the target (-9.7219), so we should *decrease* performance slightly to move closer to the target band with minimal risk. The smallest lever that changes the competition metric without touching the model, loss, or training loop is to modestly inflate the predicted `Confidence` (σ), which makes the Laplace log-likelihood more negative. I only adjust `CONF_SCALE` upward a bit and keep the same quantile-spread confidence computation and the same submission alignment logic. Everything else (data prep, model, training, prediction) remains identical.'
- What this solution (achieved -8.71947) has done: 'Your current score (-8.48676) is better than the target (-9.7219), so we should intentionally move the score downward (more negative) toward the target band with minimal risk. The smallest lever that affects the Laplace log-likelihood without changing your model, loss, or training loop is the predicted `Confidence` (σ): increasing σ makes the metric worse (more negative) in a controlled way. I only increase `CONF_SCALE` moderately (and keep the same quantile-spread confidence definition and σ≥70 clipping) so the submission remains valid and aligned, while nudging the score closer to -9.7219. Everything else (data prep, split, model, training, prediction) stays identical.'

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
from tqdm import tqdm

warnings.simplefilter("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)




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
            else np.zeros((len(self.X_cont), 0), dtype=np.int64)
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

        xcont = torch.FloatTensor(self.X_cont[self.i : self.i + self.batch_size])
        xcat = torch.LongTensor(self.X_cat[self.i : self.i + self.batch_size])
        self.i += self.batch_size
        return xcont, xcat, y

    def __len__(self):
        return self.n_batches




## === cell 2
class model_nn(nn.Module):
    def __init__(self, hidden_dim, output_dim, emb_dims, n_cont):
        super().__init__()
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        n_embs = sum([y for x, y in emb_dims])
        inp_dim = n_embs + n_cont

        self.fc0 = nn.Linear(inp_dim, hidden_dim)
        self.relu0 = nn.ReLU(True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu1 = nn.ReLU(True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def encode_and_combine_data(self, cat_data):
        if len(self.emb_layers) == 0:
            return torch.zeros((cat_data.shape[0], 0), device=cat_data.device)
        xcat = [emb(cat_data[:, k]) for k, emb in enumerate(self.emb_layers)]
        return torch.cat(xcat, 1)

    def forward(self, cont_data, cat_data):
        cont_data = cont_data.to(device)
        cat_data = cat_data.to(device)
        cat_emb = self.encode_and_combine_data(cat_data)
        x = torch.cat([cont_data, cat_emb], dim=1)

        hz = self.relu0(self.fc0(x))
        hz = self.relu1(self.fc1(hz))
        return self.fc2(hz)




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
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def model_training(
    model,
    train_loader,
    val_loader,
    epochs,
    batch_size=64,
    lr=0.001,
    patience=10,
    model_path="model.pth",
):
    if os.path.isfile(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.4, verbose=True
    )

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in tqdm(range(epochs)):
        train_loss = 0.0
        model.train()
        for X_cont, X_cat, y in tqdm(train_loader, leave=False):
            preds = model(X_cont, X_cat)
            loss = quantile_loss(preds, y.to(device), [0.2, 0.5, 0.8])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(train_loader)

        val_loss = 0.0
        val_preds = []
        true_y = []
        model.eval()
        with torch.no_grad():
            for X_cont, X_cat, y in val_loader:
                preds = model(X_cont, X_cat)
                val_preds.append(preds.detach().cpu())
                true_y.append(y.detach().cpu())
                loss = quantile_loss(preds, y.to(device), [0.2, 0.5, 0.8])
                val_loss += loss.item() / len(val_loader)

        val_preds_np = torch.cat(val_preds, dim=0).numpy()
        true_y_np = torch.cat(true_y, dim=0).numpy()
        score_nll = metric(val_preds_np, true_y_np)

        print(
            f"[valid] Epoch: {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val NLL: {score_nll:.4f}"
        )

        early_stopping(val_loss, model, path=model_path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler.step(val_loss)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model




## === cell 5
ROOT = "../input/osic-pulmonary-fibrosis-progression"
MODEL_PATH = "model.pth"

tr = pd.read_csv(f"{ROOT}/train.csv")
tr.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])
te = pd.read_csv(f"{ROOT}/test.csv")

sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub = sub.merge(te.drop("Weeks", axis=1), on="Patient")

tr["WHERE"] = "train"
te["WHERE"] = "val"  # keep your naming; we won't train on this
sub["WHERE"] = "test"

data = pd.concat([tr, te, sub], axis=0, ignore_index=True)

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

cat_feat = ["Sex", "SmokingStatus"]
uniques = []
for v in cat_feat:
    data[v] = OrdinalEncoder(dtype="int").fit_transform(data[[v]])
    uniques.append(int(data[v].nunique()))

train_df = data.loc[data.WHERE == "train"].copy()
test_full_df = data.loc[data.WHERE == "test"].copy()

FE += ["age", "week", "BASE"]
n_cont = len(FE)

dims = [1, 2]
emb_dims = [(x, y) for x, y in zip(uniques, dims)]

patients = train_df["Patient"].unique()
rng = np.random.RandomState(42)
rng.shuffle(patients)
n_val = max(1, int(0.2 * len(patients)))
val_patients = set(patients[:n_val])

val_df = train_df[train_df["Patient"].isin(val_patients)].copy()
trn_df = train_df[~train_df["Patient"].isin(val_patients)].copy()

y_trn = trn_df["FVC"].values.reshape(-1, 1)
y_val = val_df["FVC"].values.reshape(-1, 1)

X_trn = make_X(trn_df, FE, cat_feat)
X_val = make_X(val_df, FE, cat_feat)
X_tst = make_X(test_full_df, FE, cat_feat)

train_loader = Loader(X_trn, y_trn, shuffle=True, batch_size=64, cat_cols=cat_feat)
val_loader = Loader(X_val, y_val, shuffle=False, batch_size=256, cat_cols=cat_feat)
test_loader = Loader(X_tst, None, shuffle=False, batch_size=256, cat_cols=cat_feat)



## === cell 6
model = model_nn(hidden_dim=128, output_dim=3, emb_dims=emb_dims, n_cont=n_cont).to(
    device
)
model = model_training(
    model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    batch_size=64,
    lr=1e-3,
    patience=10,
    model_path=MODEL_PATH,
)

model.eval()
all_preds = []
with torch.no_grad():
    for X_cont, X_cat, _ in test_loader:
        preds = model(X_cont, X_cat).detach().cpu().numpy()
        all_preds.append(preds)
all_preds = np.vstack(all_preds)

pred_fvc = all_preds[:, 1]

CONF_SCALE = 3.10

pred_conf = np.abs(all_preds[:, 2] - all_preds[:, 0]) * CONF_SCALE
pred_conf = np.maximum(pred_conf, 70.0)

submission = pd.read_csv(f"{ROOT}/sample_submission.csv")[["Patient_Week"]].copy()
submission = submission.merge(
    test_full_df[["Patient_Week"]].assign(_row=np.arange(len(test_full_df))),
    on="Patient_Week",
    how="left",
)
idx = submission["_row"].values
submission_out = pd.DataFrame(
    {
        "Patient_Week": submission["Patient_Week"].values,
        "FVC": pred_fvc[idx].round().astype(int),
        "Confidence": pred_conf[idx].astype(float),
    }
)

if submission_out["FVC"].isna().any():
    submission_out = pd.DataFrame(
        {
            "Patient_Week": test_full_df["Patient_Week"].values,
            "FVC": pred_fvc.round().astype(int),
            "Confidence": pred_conf.astype(float),
        }
    )

submission_out.to_csv("submission.csv", index=False)
print(submission_out.head())
print("Wrote submission.csv with shape:", submission_out.shape)
