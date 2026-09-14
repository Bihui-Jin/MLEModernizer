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

-8.503854528803748

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import pydicom
import os
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import random
import math
import torch.optim as optim




## === cell 1
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 2
base_path = "/kaggle/input/osic-pulmonary-fibrosis-progression/"

train = pd.read_csv(base_path + "train.csv")
test = pd.read_csv(base_path + "test.csv")
sample_submission = pd.read_csv(base_path + "sample_submission.csv")

print("training")
print(train.head())
print("-" * 90)
print("testing")
print(test.head())
print("-" * 90)
print("sample submission")
print(sample_submission.head())



## === cell 3
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



## === cell 4
sample_patient = train[train["Patient"] == train["Patient"].iloc[0]]
print(sample_patient)
plt.plot(sample_patient["Weeks"], sample_patient["FVC"])
plt.xlabel("Weeks")
plt.ylabel("FVC")
plt.show()

sample_patient = train[train["Patient"] == train["Patient"].iloc[9]]
print(sample_patient)
plt.plot(sample_patient["Weeks"], sample_patient["FVC"])
plt.xlabel("Weeks")
plt.ylabel("FVC")
plt.show()



## === cell 5
example_patient = train["Patient"].iloc[0]
path = os.path.join(base_path + "train", example_patient)
files = sorted(os.listdir(path))

dcm = pydicom.dcmread(os.path.join(path, files[len(files) // 2]))
plt.imshow(dcm.pixel_array, cmap="gray")
plt.title(f"CT Scan of {example_patient}")



## === cell 6
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



## === cell 7
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



## === cell 8
plt.figure(figsize=(6, 5))
sns.scatterplot(data=train, x="FVC", y="Percent", alpha=0.5)
sns.regplot(data=train, x="FVC", y="Percent", scatter=False, color="red", label="Trend")
plt.title("Relation between FVC and Percent")
plt.xlabel("FVC (ml)")
plt.ylabel("Percent (%)")
plt.legend()
plt.tight_layout()
plt.show()




## === cell 9
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




## === cell 10
df = pd.read_csv(base_path + "train.csv")
pairwise_df = generate_pairwise_fvc_dataset(df)
print(pairwise_df.head())



## === cell 11
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

X = pairwise_df[features].copy()
y_processed = pairwise_df[target].astype(np.float32).values

scaler = StandardScaler()
x_processed = scaler.fit_transform(X).astype(np.float32)

print("x_processed shape:", x_processed.shape, "y shape:", y_processed.shape)




## === cell 12
def get_model(x, y):
    seed = random.randint(1, 1000)
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        max_depth=10,
        learning_rate=0.5,
        random_state=seed,
    )
    model.fit(x, y)
    return model, seed




## === cell 13
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




## === cell 14
x_data = x_processed
y_data = y_processed

models, weights, maes, seeds = adaBoost(10, x_data, y_data, 75)



## === cell 15
print(maes)



## === cell 16
print(seeds)



## === cell 17
print(weights)




## === cell 18
def normalize_to_sum_one(values):
    total = sum(values)
    if total == 0:
        raise ValueError("Cannot normalize list with sum = 0.")
    return [x / total for x in values]


weights = normalize_to_sum_one(weights)




## === cell 19
def weighted_model_prediction_with_confidence(x_values, models, model_weights):
    if len(models) != len(model_weights):
        raise ValueError("Number of models and weights must be the same.")
    if not abs(sum(model_weights) - 1.0) < 1e-6:
        raise ValueError("Model weights must sum to 1.")

    all_preds = [
        np.array(model.predict(x_values), dtype=np.float32) for model in models
    ]
    all_preds = np.array(all_preds, dtype=np.float32)  # (n_models, n_samples)

    w = np.array(model_weights, dtype=np.float32).reshape(-1, 1)
    weighted_mean = np.sum(w * all_preds, axis=0)

    variance = np.sum(w * (all_preds - weighted_mean) ** 2, axis=0)
    std_dev = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)

    confidence = 1 / (1 + std_dev) * 100
    confidence = np.clip(confidence, 420, 1000)

    return weighted_mean.tolist(), confidence.tolist()




## === cell 20
x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_data, test_size=0.2, random_state=17
)
ada_boost_preds, conf_tmp = weighted_model_prediction_with_confidence(
    x_val, models, weights
)
mae = mean_absolute_error(y_val, ada_boost_preds)
print("AdaBoost-like ensemble MAE (sanity check):", mae)
print("Ensemble pseudo-conf (first 10):", conf_tmp[:10])




## === cell 21
def compute_loss(y_pred, y_true, sigma):
    f = torch.sqrt(torch.tensor(2.0, device=y_pred.device))
    sigma = torch.clamp(sigma, min=70.0)  # element-wise clamp to metric's sigma_min

    delta = torch.abs(y_pred - y_true)
    delta = torch.clamp(delta, max=1000.0)

    loss = f * delta / sigma + torch.log(f * sigma)
    return loss.mean()




## === cell 22
x = x_processed
y = y_processed

x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)  # (N,)

print("x_tensor:", x_tensor.shape, "y_tensor:", y_tensor.shape)




## === cell 23
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)


dataset = TensorDataset(x_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        pred = model(xb)  # (bs, 2)
        mu = pred[:, 0]  # predicted FVC
        sigma_raw = pred[:, 1]
        sigma = F.softplus(sigma_raw) + 70.0

        loss = compute_loss(mu, yb, sigma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    if (epoch + 1) % 10 == 0 or epoch < 5:
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/28246784.py in <cell line: 0>()
     27         mu = pred[:, 0]  # predicted FVC
     28         sigma_raw = pred[:, 1]
---> 29         sigma = F.softplus(sigma_raw) + 70.0
     30 
     31         loss = compute_loss(mu, yb, sigma)

NameError: name 'F' is not defined

## === cell 24
test = pd.read_csv(base_path + "test.csv")
sub_sample = pd.read_csv(base_path + "sample_submission.csv")

print(test.head())
print(sub_sample.head())



## === cell 25
sub_sample[["Patient_ID", "Target_Week"]] = sub_sample["Patient_Week"].str.rsplit(
    "_", n=1, expand=True
)
sub_sample["Target_Week"] = sub_sample["Target_Week"].astype(int)

test = test.rename(columns={"Patient": "Patient_ID", "Weeks": "Reading_Week"})

merged = test.merge(
    sub_sample[["Patient_ID", "Target_Week"]], on="Patient_ID", how="left"
)

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

print(final_df.head())



## === cell 26
final_df["Sex"] = le_sex.transform(final_df["Sex"])
final_df["SmokingStatus"] = le_smoke.transform(final_df["SmokingStatus"])

features_test = [
    "Reading_Week",
    "Target_Week",
    "Weeks_Diff",
    "Curr_FVC",
    "Percent",
    "Age",
    "Sex",
    "SmokingStatus",
]
X_test = final_df[features_test].copy()

X_test = X_test.rename(
    columns={
        "Reading_Week": "Reading_week",
        "Target_Week": "target_week",
        "Weeks_Diff": "Weeks_diff",
    }
)

X_scaled = scaler.transform(X_test[features]).astype(np.float32)
print("X_scaled shape:", X_scaled.shape)



## === cell 27
model.eval()
with torch.no_grad():
    x_tensor_test = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    preds = model(x_tensor_test)
    mu = preds[:, 0].detach().cpu().numpy().astype(np.float32)
    sigma = (F.softplus(preds[:, 1]) + 70.0).detach().cpu().numpy().astype(np.float32)

final_df["FVC"] = mu
final_df["Confidence"] = np.clip(sigma, 70.0, 1000.0)

final_df["Patient_Week"] = (
    final_df["Patient_ID"] + "_" + final_df["Target_Week"].astype(str)
)

submission = final_df[["Patient_Week", "FVC", "Confidence"]].copy()
submission["FVC"] = submission["FVC"].round(1)
submission["Confidence"] = submission["Confidence"].round(1)

print(submission.head())
print("submission shape:", submission.shape)

submission = sample_submission[["Patient_Week"]].merge(
    submission, on="Patient_Week", how="left"
)
assert submission.shape[0] == sample_submission.shape[0]
assert submission["FVC"].isna().sum() == 0
assert submission["Confidence"].isna().sum() == 0

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv")

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3048875612.py in <cell line: 0>()
      5     preds = model(x_tensor_test)
      6     mu = preds[:, 0].detach().cpu().numpy().astype(np.float32)
----> 7     sigma = (F.softplus(preds[:, 1]) + 70.0).detach().cpu().numpy().astype(np.float32)
      8 
      9 final_df["FVC"] = mu

NameError: name 'F' is not defined
