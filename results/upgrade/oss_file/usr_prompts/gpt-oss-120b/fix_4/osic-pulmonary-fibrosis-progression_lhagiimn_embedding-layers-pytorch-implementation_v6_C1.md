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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GroupKFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
quantiles = [0.2, 0.5, 0.8]


def make_X(dt, dense_cols, cat_feats):
    X = {"dense": dt[dense_cols].to_numpy()}
    for v in cat_feats:
        X[v] = dt[[v]].to_numpy()
    return X


class Loader:
    def __init__(self, X, y, shuffle=True, batch_size=64, cat_cols=[]):
        self.X_cont = X["dense"]
        self.X_cat = np.concatenate([X[k] for k in cat_cols], axis=1)
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_conts = self.X_cont.shape[1]
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
        end = self.i + self.batch_size
        xcont = torch.FloatTensor(self.X_cont[self.i : end])
        xcat = torch.LongTensor(self.X_cat[self.i : end])
        y = None
        if self.y is not None:
            y = torch.FloatTensor(self.y[self.i : end].astype(np.float32))
        self.i = end
        return xcont, xcat, y

    def __len__(self):
        return self.n_batches




## === cell 1
class model_nn(nn.Module):
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
        xcat = [el(cat_data[:, k]) for k, el in enumerate(self.emb_layers)]
        xcat = torch.cat(xcat, 1)
        return xcat

    def forward(self, cont_data, cat_data):
        cont_data = cont_data.to(device)
        cat_data = cat_data.to(device)
        cat_data = self.encode_and_combine_data(cat_data)
        x = torch.cat([cont_data, cat_data], dim=1)
        hz = self.fc0(x)
        hz = self.relu0(hz)
        hz = self.fc1(hz)
        hz = self.relu1(hz)
        out1 = self.fc2(hz)
        out2 = self.fc3(hz)
        out3 = self.fc4(hz)
        return torch.cat([out1, out2, out3], dim=1)




## === cell 2
def quantile_loss(preds, target, quantiles):
    assert not target.requires_grad
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




## === cell 3
class EarlyStopping:
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
        torch.save(model, path)
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
        return torch.load(model_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.4, verbose=True
    )
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in tqdm(range(epochs), desc="Training epochs"):
        train_loss = 0.0
        model.train()
        for X_cont, X_cat, y in train_loader:
            preds = model(X_cont, X_cat)
            loss = quantile_loss(preds, y.to(device), quantiles)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(train_loader)
        val_loss = 0.0
        val_preds, true_y = [], []
        model.eval()
        with torch.no_grad():
            for X_cont, X_cat, y in val_loader:
                preds = model(X_cont, X_cat)
                val_preds.append(preds)
                true_y.append(y)
                loss = quantile_loss(preds, y.to(device), quantiles)
                val_loss += loss.item() / len(val_loader)
        val_preds_np = torch.cat(val_preds).cpu().numpy()
        true_y_np = torch.cat(true_y).cpu().numpy()
        score = metric(val_preds_np, true_y_np)
        print(
            f"[valid] Epoch {epoch} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val score {score:.4f}"
        )
        early_stopping(val_loss, model, model_path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        scheduler.step(val_loss)
    return torch.load(model_path)




## === cell 4
ROOT = "../input/osic-pulmonary-fibrosis-progression"
Model_Root = "models"
os.makedirs(Model_Root, exist_ok=True)




## === cell 5
tr = pd.read_csv(f"{ROOT}/train.csv")
tr.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])
chunk = pd.read_csv(f"{ROOT}/test.csv")
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub = sub.merge(chunk.drop("Weeks", axis=1), on="Patient")

tr["WHERE"] = "train"
chunk["WHERE"] = "val"
sub["WHERE"] = "test"

data = pd.concat([tr, chunk, sub], ignore_index=True)

data["min_week"] = data["Weeks"]
data.loc[data.WHERE == "test", "min_week"] = np.nan
data["min_week"] = data.groupby("Patient")["min_week"].transform("min")

base = data.loc[data.Weeks == data.min_week][["Patient", "FVC", "Percent"]].copy()
base.columns = ["Patient", "min_FVC", "min_percent"]
base["nb"] = 1
base["nb"] = base.groupby("Patient")["nb"].transform("cumsum")
base = base[base.nb == 1].drop("nb", axis=1)

data = data.merge(base, on="Patient", how="left")
data["base_week"] = data["Weeks"] - data["min_week"]

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
uniques = []
for v in cat_feat:
    data[v] = OrdinalEncoder(dtype="int").fit_transform(data[[v]])
    uniques.append(len(data[v].unique()))

tr = data.loc[data.WHERE == "train"]
chunk = data.loc[data.WHERE == "val"]
sub = data.loc[data.WHERE == "test"]

FE += ["age", "week", "BASE", "min_percent_norm", "age_week_norm"]

dims = [1, 2]
emb_dims = [(x, y) for x, y in zip(uniques, dims)]
n_cont = len(FE)




## === cell 6
hidden_dim = 256
out_dim = 1
kfold = 6
groups = tr["Patient"].values
skf = GroupKFold(n_splits=kfold)

avg_preds = np.zeros((len(sub), len(quantiles)))
models = []

for fold, (train_idx, val_idx) in enumerate(
    skf.split(tr, tr["FVC"].values, groups=groups)
):
    print(f"[Fold {fold + 1}/{kfold}]")
    model_path = f"{Model_Root}/nn_model_{fold}.pth"

    if os.path.isfile(model_path):
        final_model = torch.load(model_path)
    else:
        X_train_df = tr.iloc[train_idx].reset_index(drop=True)
        X_valid_df = tr.iloc[val_idx].reset_index(drop=True)
        y_train = tr.iloc[train_idx]["FVC"].values
        y_valid = tr.iloc[val_idx]["FVC"].values

        X_train = make_X(X_train_df, FE, cat_feat)
        X_valid = make_X(X_valid_df, FE, cat_feat)

        train_loader = Loader(
            X_train, y_train, cat_cols=cat_feat, batch_size=16, shuffle=True
        )
        val_loader = Loader(
            X_valid, y_valid, cat_cols=cat_feat, batch_size=64, shuffle=False
        )

        model = model_nn(hidden_dim, out_dim, emb_dims, n_cont).to(device)
        final_model = model_training(
            model,
            train_loader,
            val_loader,
            epochs=1000,
            batch_size=64,
            lr=0.01,
            patience=20,
            model_path=model_path,
        )
        models.append(final_model)

    X_test = make_X(sub, FE, cat_feat)
    test_loader = Loader(X_test, None, cat_cols=cat_feat, batch_size=256, shuffle=False)

    preds_fold = []
    with torch.no_grad():
        for X_cont, X_cat, _ in tqdm(test_loader, desc="Predict fold"):
            out = final_model(X_cont, X_cat)
            preds_fold.append(out)
    preds_fold = torch.cat(preds_fold, dim=0).cpu().numpy()
    avg_preds += preds_fold

avg_preds = avg_preds / kfold

sub["FVC"] = avg_preds[:, 1]
sub["Confidence"] = avg_preds[:, 2] - avg_preds[:, 0]
sub[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
UnpicklingError                           Traceback (most recent call last)
/tmp/ipykernel_55/4085399285.py in <cell line: 0>()
     33 
     34         model = model_nn(hidden_dim, out_dim, emb_dims, n_cont).to(device)
---> 35         final_model = model_training(
     36             model,
     37             train_loader,

/tmp/ipykernel_55/3219428994.py in model_training(model, train_loader, val_loader, epochs, batch_size, lr, patience, model_path)
     81             break
     82         scheduler.step(val_loss)
---> 83     return torch.load(model_path)
     84 
     85 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1468                         )
   1469                     except pickle.UnpicklingError as e:
-> 1470                         raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
   1471                 return _load(
   1472                     opened_zipfile,

UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
	WeightsUnpickler error: Unsupported global: GLOBAL __main__.model_nn was not an allowed global by default. Please use `torch.serialization.add_safe_globals([model_nn])` or the `torch.serialization.safe_globals([model_nn])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
