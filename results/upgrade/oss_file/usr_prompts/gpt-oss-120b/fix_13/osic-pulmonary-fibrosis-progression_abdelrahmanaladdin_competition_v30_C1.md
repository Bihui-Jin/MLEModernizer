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

3.13

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
tqdm==4.67.1
xgboost==2.0.3

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

-7.412076655674882

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -10.98174) has done: 'I adjust the confidence calculation in the ensemble function to produce realistic σ values (clipped to a sensible range [70, 200]) instead of the previous overly large values, which should improve the Laplace‑Log‑Likelihood score while keeping the rest of the pipeline unchanged.'
- What this solution (achieved -10.7038) has done: 'The fix aligns test feature names with those used when fitting the scaler, restores the correct optimizer‑model pairing for the MLPs, and keeps the rest of the pipeline unchanged so the ensemble can generate a valid `submission.csv` file.'
- What this solution (achieved -10.06733) has done: 'I increase the upper confidence bound from 200 to 1000 both when the ensemble variance‑based confidence is first computed and when the final confidence is clipped, because a larger σ reduces the dominant Δ/σ term in the Laplace log‑likelihood and should raise the score toward the target.'
- What this solution (achieved -9.68807) has done: 'I lower the confidence upper bound back to 200 instead of 1000 so the log‑penalty in the Laplace metric is not excessive, and I apply the same clipping when the final confidences are computed. This small calibration change is expected to raise the negative score toward the target without altering the model architecture or training process.'
- What this solution (achieved -8.88177) has done: 'I compute validation MAEs for each model, create inverse‑MAE weights, and use these weights to form a calibrated ensemble for the test set. This modest re‑weighting should reduce prediction error and move the Laplace‑Log‑Likelihood score toward the target without changing the core model architecture or training process.'
- What this solution (achieved -8.6011) has done: 'I slightly raise the confidence clipping upper bound from 200 to 300 both in the ensemble confidence computation and the final submission clipping. This modest increase lets the σ term be larger, reducing the dominant Δ/σ penalty in the Laplace‑Log‑Likelihood while keeping the log‑penalty reasonable, which should raise the score toward the target without altering the core model logic.'
- What this solution (achieved -8.89214) has done: 'I raise the upper bound for the confidence values from 300 to 1000 in the two places where the ensemble confidence is clipped. This larger σ reduces the dominant Δ/σ penalty in the Laplace‑Log‑Likelihood metric while keeping the required lower bound of 70, so the validation score should move upward toward the target without altering any model architecture or training process.'

# 9. Code solution

## === cell 0
import os
import random
import math
import copy

import numpy as np
import pandas as pd
import pydicom  # added import for DICOM handling

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

base_path = "/kaggle/input/osic-pulmonary-fibrosis-progression/"

train = pd.read_csv(os.path.join(base_path, "train.csv"))
test = pd.read_csv(os.path.join(base_path, "test.csv"))
sample_submission = pd.read_csv(os.path.join(base_path, "sample_submission.csv"))

print("training head")
print(train.head())
print("-" * 90)
print("testing head")
print(test.head())
print("-" * 90)
print("sample submission head")
print(sample_submission.head())




## === cell 1
sns.histplot(train["FVC"], kde=True)
plt.title("Distribution of FVC")
plt.show()

sns.histplot(train["Age"], kde=True)
plt.title("Age distribution")
plt.show()

sns.countplot(data=train, x="Sex")
plt.title("Sex distribution")
plt.show()

sns.countplot(data=train, x="SmokingStatus")
plt.title("SmokingStatus")
plt.show()




## === cell 2
sample_patient = train[train["Patient"] == train["Patient"].iloc[0]]
plt.plot(sample_patient["Weeks"], sample_patient["FVC"])
plt.xlabel("Weeks")
plt.ylabel("FVC")
plt.title(f"Patient {sample_patient['Patient'].iloc[0]}")
plt.show()

sample_patient = train[train["Patient"] == train["Patient"].iloc[9]]
plt.plot(sample_patient["Weeks"], sample_patient["FVC"])
plt.xlabel("Weeks")
plt.ylabel("FVC")
plt.title(f"Patient {sample_patient['Patient'].iloc[0]}")
plt.show()




## === cell 3
example_patient = train["Patient"].iloc[0]
path = os.path.join(base_path, "train", example_patient)
files = sorted(os.listdir(path))

dcm_path = os.path.join(path, files[len(files) // 2])
dcm = pydicom.dcmread(dcm_path)
plt.imshow(dcm.pixel_array, cmap="gray")
plt.title(f"CT Slice of {example_patient}")
plt.axis("off")
plt.show()




## === cell 4
sns.set(style="whitegrid", palette="muted")
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.boxplot(data=train, x="Sex", y="FVC")
plt.title("FVC by Sex")

plt.subplot(1, 3, 2)
sns.scatterplot(data=train, x="Age", y="FVC", alpha=0.4)
plt.title("FVC vs Age")

plt.subplot(1, 3, 3)
sns.boxplot(data=train, x="SmokingStatus", y="FVC")
plt.xticks(rotation=20)
plt.title("FVC by Smoking Status")

plt.tight_layout()
plt.show()




## === cell 5
sns.set(style="whitegrid", palette="muted")
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.boxplot(data=train, x="Sex", y="Percent")
plt.title("Percent by Sex")

plt.subplot(1, 3, 2)
sns.scatterplot(data=train, x="Age", y="Percent", alpha=0.4)
plt.title("Percent vs Age")

plt.subplot(1, 3, 3)
sns.boxplot(data=train, x="SmokingStatus", y="Percent")
plt.xticks(rotation=20)
plt.title("Percent by Smoking Status")

plt.tight_layout()
plt.show()




## === cell 6
plt.figure(figsize=(6, 5))
sns.scatterplot(data=train, x="FVC", y="Percent", alpha=0.5)
sns.regplot(data=train, x="FVC", y="Percent", scatter=False, color="red", label="Trend")
plt.title("Relation between FVC and Percent")
plt.xlabel("FVC (ml)")
plt.ylabel("Percent (%)")
plt.legend()
plt.tight_layout()
plt.show()




## === cell 7
def generate_pairwise_fvc_dataset(df):
    df = df.dropna()
    df = df.sort_values(["Patient", "Weeks"])
    all_rows = []
    for patient_id, group in df.groupby("Patient"):
        group = group.reset_index(drop=True)
        n = len(group)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                row_i = group.iloc[i]
                row_j = group.iloc[j]
                all_rows.append(
                    {
                        "Patient_ID": row_i["Patient"],
                        "Reading_week": row_i["Weeks"],
                        "target_week": row_j["Weeks"],
                        "Weeks_diff": row_j["Weeks"] - row_i["Weeks"],
                        "Curr_FVC": row_i["FVC"],
                        "Target_FVC": row_j["FVC"],
                        "Percent": row_i["Percent"],
                        "Age": row_i["Age"],
                        "Sex": row_i["Sex"],
                        "SmokingStatus": row_i["SmokingStatus"],
                    }
                )
    return pd.DataFrame(all_rows)




## === cell 8
pairwise_df = generate_pairwise_fvc_dataset(train)

le_sex = LabelEncoder().fit(pairwise_df["Sex"])
le_smoke = LabelEncoder().fit(pairwise_df["SmokingStatus"])

pairwise_df["Sex"] = le_sex.transform(pairwise_df["Sex"])
pairwise_df["SmokingStatus"] = le_smoke.transform(pairwise_df["SmokingStatus"])

features = [
    "Reading_week",
    "target_week",
    "Weeks_diff",
    "Curr_FVC",
    "Percent",
    "Age",
    "Sex",
    "SmokingStatus",
]
target = "Target_FVC"

X = pairwise_df[features]
y = pairwise_df[target]

scaler = StandardScaler()
x_processed = scaler.fit_transform(X)




## === cell 9
def get_model(x, y):
    seed = random.randint(1, 1000)
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        random_state=seed,
        n_jobs=4,
        verbosity=0,
    )
    model.fit(x, y)
    return model, seed




## === cell 10
def adaBoost(k, x, y, thre):
    m = 0
    models = []
    weights = []
    maes = []
    seeds = []
    while m < k:
        split_seed = random.randint(1, 1000)
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=0.2, random_state=split_seed
        )
        model, model_seed = get_model(x_train, y_train)
        y_pred = model.predict(x_val)
        mae = mean_absolute_error(y_val, y_pred)
        if mae < thre:
            m += 1
            w = math.log((thre * 2 - mae) / mae)
            models.append(model)
            weights.append(w)
            maes.append(mae)
            seeds.append([split_seed, model_seed])
    return models, weights, maes, seeds




## === cell 11
models, raw_weights, maes, seeds = adaBoost(k=10, x=x_processed, y=y, thre=75)


def normalize_to_sum_one(values):
    total = sum(values)
    if total == 0:
        raise ValueError("Cannot normalize list with sum = 0.")
    return [v / total for v in values]


weights = normalize_to_sum_one(raw_weights)




## === cell 12
def weighted_model_prediction_with_confidence(x_values, models, model_weights):
    """
    Ensemble prediction where confidence is derived from the weighted variance
    of individual model predictions. Confidence is clipped to the range
    [70, 200] as required by the competition metric.
    """
    if len(models) != len(model_weights):
        raise ValueError("Number of models and weights must be the same.")
    if not abs(sum(model_weights) - 1.0) < 1e-6:
        raise ValueError("Model weights must sum to 1.")
    all_preds = np.array([m.predict(x_values) for m in models])  # (n_models, n_samples)
    weights_arr = np.array(model_weights).reshape(-1, 1)  # (n_models, 1)
    weighted_pred = np.sum(weights_arr * all_preds, axis=0)
    variance = np.sum(weights_arr * (all_preds - weighted_pred) ** 2, axis=0)
    std_dev = np.sqrt(variance)
    confidence = np.clip(std_dev, 70, 200)
    return weighted_pred.tolist(), confidence.tolist()




## === cell 13
x_train_val, x_val_val, y_train_val, y_val_val = train_test_split(
    x_processed, y, test_size=0.2, random_state=17
)

ada_boost_preds_val, conf_val = weighted_model_prediction_with_confidence(
    x_val_val, models, weights
)

print("AdaBoost validation MAE:", mean_absolute_error(y_val_val, ada_boost_preds_val))




## === cell 14
def compute_loss(y_pred, y_true, sigma):
    """
    Laplace Log‑Likelihood loss used for the competition.
    sigma is clipped to a minimum of 70 as required.
    """
    f = torch.sqrt(torch.tensor(2.0, device=y_pred.device))
    sigma = torch.clamp(sigma, min=70.0)
    delta = torch.abs(y_pred - y_true)
    delta = torch.clamp(delta, max=1000.0)
    loss = f * delta / sigma + torch.log(f * sigma)
    return loss.mean()




## === cell 15
x_mlp = x_processed
y_mlp = y.values

x_train_mlp, x_val_mlp, y_train_mlp, y_val_mlp = train_test_split(
    x_mlp, y_mlp, test_size=0.2, random_state=42
)

x_train_tensor = torch.tensor(x_train_mlp, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_mlp, dtype=torch.float32)

x_val_tensor = torch.tensor(x_val_mlp, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_mlp, dtype=torch.float32)




## === cell 16
class ImprovedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),  # outputs: mu, sigma
        )

    def forward(self, x):
        return self.model(x)




## === cell 17
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.model(x)




## === cell 18
def train_MLP(model, device, optimizer, train_loader, val_loader, num_epochs=30):
    best_model = copy.deepcopy(model)
    min_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            y_pred = pred[:, 0]
            sigma = pred[:, 1]
            loss = compute_loss(y_pred, yb, sigma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                y_pred, sigma = pred[:, 0], pred[:, 1]
                loss = compute_loss(y_pred, yb, sigma)
                val_loss += loss.item()
        if val_loss < min_loss:
            min_loss = val_loss
            best_model = copy.deepcopy(model)
    return best_model




## === cell 19
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

deep_model = ImprovedMLP().to(device)
deep_optimizer = optim.Adam(deep_model.parameters(), lr=1e-3, weight_decay=1e-5)
deep_mlp = train_MLP(
    model=deep_model,
    device=device,
    optimizer=deep_optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=30,
)

simple_model = SimpleMLP().to(device)
simple_optimizer = optim.Adam(simple_model.parameters(), lr=1e-3, weight_decay=1e-5)
simple_mlp = train_MLP(
    model=simple_model,
    device=device,
    optimizer=simple_optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=30,
)




## === cell 20
sub_sample = pd.read_csv(os.path.join(base_path, "sample_submission.csv"))
sub_sample[["Patient_ID", "Target_Week"]] = sub_sample["Patient_Week"].str.rsplit(
    "_", n=1, expand=True
)
sub_sample["Target_Week"] = sub_sample["Target_Week"].astype(int)

test_df = test.rename(columns={"Patient": "Patient_ID", "Weeks": "Reading_Week"})
merged = test_df.merge(sub_sample[["Patient_ID", "Target_Week"]], on="Patient_ID")
merged["Weeks_Diff"] = merged["Target_Week"] - merged["Reading_Week"]

final_df = merged[
    [
        "Patient_ID",
        "Reading_Week",
        "Target_Week",
        "Weeks_Diff",
        "FVC",  # Curr_FVC
        "Percent",
        "Age",
        "Sex",
        "SmokingStatus",
    ]
].rename(columns={"FVC": "Curr_FVC"})

final_df = final_df.rename(
    columns={
        "Reading_Week": "Reading_week",
        "Target_Week": "target_week",
        "Weeks_Diff": "Weeks_diff",
    }
)

final_df["Sex"] = le_sex.transform(final_df["Sex"])
final_df["SmokingStatus"] = le_smoke.transform(final_df["SmokingStatus"])

X_test = final_df[features]
X_scaled = scaler.transform(X_test)

x_test_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

simple_mlp.eval()
with torch.no_grad():
    simple_preds = simple_mlp(x_test_tensor).cpu().numpy()

deep_mlp.eval()
with torch.no_grad():
    deep_preds = deep_mlp(x_test_tensor).cpu().numpy()

x_val_tensor = torch.tensor(x_val_val, dtype=torch.float32).to(device)

simple_mlp.eval()
with torch.no_grad():
    simple_val_preds = simple_mlp(x_val_tensor).cpu().numpy()

deep_mlp.eval()
with torch.no_grad():
    deep_val_preds = deep_mlp(x_val_tensor).cpu().numpy()




## === cell 21
mae_ada = mean_absolute_error(y_val_val, ada_boost_preds_val)
mae_deep = mean_absolute_error(y_val_val, deep_val_preds[:, 0])
mae_simple = mean_absolute_error(y_val_val, simple_val_preds[:, 0])

inv_weights = np.array([1.0 / (mae_ada**2), 1.0 / (mae_deep**2), 1.0 / (mae_simple**2)])
inv_weights = inv_weights / inv_weights.sum()
w_ada, w_deep, w_simple = inv_weights.tolist()

print(
    f"Validation MAEs -> AdaBoost: {mae_ada:.4f}, DeepMLP: {mae_deep:.4f}, SimpleMLP: {mae_simple:.4f}"
)
print(
    f"Ensemble weights -> AdaBoost: {w_ada:.3f}, DeepMLP: {w_deep:.3f}, SimpleMLP: {w_simple:.3f}"
)

preds = (
    w_ada * np.array(ada_boost_preds_test)
    + w_deep * deep_preds[:, 0]
    + w_simple * simple_preds[:, 0]
)

raw_confs = (
    w_ada * np.array(conf_test)
    + w_deep * deep_preds[:, 1]
    + w_simple * simple_preds[:, 1]
)

final_conf = np.clip(raw_confs, 70, 200)

final_df["FVC"] = preds
final_df["Confidence"] = final_conf
final_df["Patient_Week"] = (
    final_df["Patient_ID"] + "_" + final_df["target_week"].astype(str)
)

submission = final_df[["Patient_Week", "FVC", "Confidence"]].copy()
submission["FVC"] = submission["FVC"].round(1)
submission["Confidence"] = submission["Confidence"].round(1)

print(submission.head())
print("Submission shape:", submission.shape)
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3175722676.py in <cell line: 0>()
     17 # Final ensemble predictions on test set
     18 preds = (
---> 19     w_ada * np.array(ada_boost_preds_test)
     20     + w_deep * deep_preds[:, 0]
     21     + w_simple * simple_preds[:, 0]

NameError: name 'ada_boost_preds_test' is not defined
