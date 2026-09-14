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

# 8. Previous improvement plan

- What this solution (achieved -10.98174) has done: 'I adjust the confidence calculation in the ensemble function to produce realistic σ values (clipped to a sensible range [70, 200]) instead of the previous overly large values, which should improve the Laplace‑Log‑Likelihood score while keeping the rest of the pipeline unchanged.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import pydicom
import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import OrderedDict
from tqdm import tqdm
import pickle
from skimage.transform import resize
from types import SimpleNamespace
import torch.nn.functional as F
import random
import math
import torch.optim as optim



## === cell 1
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



## === cell 2
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



## === cell 3
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



## === cell 4
example_patient = train["Patient"].iloc[0]
path = os.path.join(base_path + "train", example_patient)
files = sorted(os.listdir(path))

dcm = pydicom.dcmread(os.path.join(path, files[len(files) // 2]))
plt.imshow(dcm.pixel_array, cmap="gray")
plt.title(f"CT Scan of {example_patient}")



## === cell 5
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



## === cell 6
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



## === cell 7
plt.figure(figsize=(6, 5))
sns.scatterplot(data=train, x="FVC", y="Percent", alpha=0.5)
sns.regplot(data=train, x="FVC", y="Percent", scatter=False, color="red", label="Trend")
plt.title("Relation between FVC and Percent")
plt.xlabel("FVC (ml)")
plt.ylabel("Percent (%)")
plt.legend()
plt.tight_layout()
plt.show()




## === cell 8
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




## === cell 9
df = pd.read_csv(base_path + "train.csv")
pairwise_df = generate_pairwise_fvc_dataset(df)
print(pairwise_df.head())



## === cell 10
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
y_processed = pairwise_df[target]

scaler = StandardScaler()
x_processed = scaler.fit_transform(X)




## === cell 11
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




## === cell 12
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




## === cell 13
x_data = x_processed
y_data = y_processed

models, weights, maes, seeds = adaBoost(10, x_data, y_data, 75)



## === cell 14
print("MAEs of selected models:", maes)




## === cell 15
def normalize_to_sum_one(values):
    total = sum(values)
    if total == 0:
        raise ValueError("Cannot normalize list with sum = 0.")
    return [x / total for x in values]


weights = normalize_to_sum_one(weights)




## === cell 16
def weighted_model_prediction_with_confidence(x_values, models, model_weights):
    if len(models) != len(model_weights):
        raise ValueError("Number of models and weights must be the same.")
    if not abs(sum(model_weights) - 1.0) < 1e-6:
        raise ValueError("Model weights must sum to 1.")
    all_preds = np.array(
        [model.predict(x_values) for model in models]
    )  # (n_models, n_samples)
    weights_arr = np.array(model_weights).reshape(-1, 1)  # (n_models, 1)
    weighted_pred = np.sum(weights_arr * all_preds, axis=0)  # (n_samples,)
    weighted_mean = weighted_pred
    variance = np.sum(weights_arr * (all_preds - weighted_mean) ** 2, axis=0)
    std_dev = np.sqrt(variance)
    confidence = np.clip(std_dev, 70, 200)
    return weighted_pred.tolist(), confidence.tolist()




## === cell 17
x_train_val, x_val_val, y_train_val, y_val_val = train_test_split(
    x_data, y_data, test_size=0.2, random_state=17
)
ada_boost_preds_val, conf_val = weighted_model_prediction_with_confidence(
    x_val_val, models, weights
)
print("Validation MAE:", mean_absolute_error(y_val_val, ada_boost_preds_val))




## === cell 18
def compute_loss(y_pred, y_true, sigma):
    f = torch.sqrt(torch.tensor(2.0, device=y_pred.device))
    sigma = torch.clamp(sigma, min=70.0)
    delta = torch.abs(y_pred - y_true)
    delta = torch.clamp(delta, max=1000.0)
    loss = f * delta / sigma + torch.log(f * sigma)
    return loss.mean()




## === cell 19
x = x_processed
y = y_processed

x_train_mlp, x_val_mlp, y_train_mlp, y_val_mlp = train_test_split(
    x, y, test_size=0.2, random_state=42
)
y_train_mlp = y_train_mlp.to_numpy()

x_train_tensor = torch.tensor(x_train_mlp, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_mlp, dtype=torch.float32)

x_val_tensor = torch.tensor(x_val_mlp, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_mlp, dtype=torch.float32)




## === cell 20
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
            nn.Linear(32, 2),  # mu, sigma
        )

    def forward(self, x):
        return self.model(x)




## === cell 21
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)




## === cell 22
def train_MLP(model, device, optimizer, train_loader, val_loader, num_epochs=50):
    best_model = model
    min_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            y_pred = pred[:, 0]
            sigma = pred[:, 1]
            loss = compute_loss(y_pred, yb, sigma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
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




## === cell 23
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

deep_mlp = train_MLP(
    model=ImprovedMLP().to(device),
    device=device,
    optimizer=optim.Adam(ImprovedMLP().parameters(), lr=1e-3, weight_decay=1e-5),
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
)

simple_mlp = train_MLP(
    model=SimpleMLP().to(device),
    device=device,
    optimizer=optim.Adam(SimpleMLP().parameters(), lr=1e-3, weight_decay=1e-5),
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
)



## === cell 24
sub_sample = pd.read_csv(base_path + "sample_submission.csv")
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

final_df["Sex"] = le_sex.transform(final_df["Sex"])
final_df["SmokingStatus"] = le_smoke.transform(final_df["SmokingStatus"])

features = [
    "Reading_Week",
    "Target_Week",
    "Weeks_Diff",
    "Curr_FVC",
    "Percent",
    "Age",
    "Sex",
    "SmokingStatus",
]

X_test = final_df[features]
X_scaled = scaler.transform(X_test)  # use scaler fitted on pairwise training data



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1646727482.py in <cell line: 0>()
     40 
     41 X_test = final_df[features]
---> 42 X_scaled = scaler.transform(X_test)  # use scaler fitted on pairwise training data
     43 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in transform(self, X, copy)
    990 
    991         copy = copy if copy is not None else self.copy
--> 992         X = self._validate_data(
    993             X,
    994             reset=False,

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    546             validated.
    547         """
--> 548         self._check_feature_names(X, reset=reset)
    549 
    550         if y is None and self._get_tags()["requires_y"]:

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _check_feature_names(self, X, reset)
    479                 )
    480 
--> 481             raise ValueError(message)
    482 
    483     def _validate_data(

ValueError: The feature names should match those that were passed during fit.
Feature names unseen at fit time:
- Reading_Week
- Target_Week
- Weeks_Diff
Feature names seen at fit time, yet now missing:
- Reading_week
- Weeks_diff
- target_week


## === cell 25
ada_boost_preds_test, conf_test = weighted_model_prediction_with_confidence(
    X_scaled, models, weights
)



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3364970069.py in <cell line: 0>()
      1 # AdaBoost ensemble predictions on test set
      2 ada_boost_preds_test, conf_test = weighted_model_prediction_with_confidence(
----> 3     X_scaled, models, weights
      4 )
      5 

NameError: name 'X_scaled' is not defined

## === cell 26
x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2543825928.py in <cell line: 0>()
      1 # Convert test features to torch tensor for MLP inference
----> 2 x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
      3 

NameError: name 'X_scaled' is not defined

## === cell 27
simple_mlp.eval()
with torch.no_grad():
    simple_mlp_preds = simple_mlp(x_tensor).cpu().numpy()



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2432848802.py in <cell line: 0>()
      1 simple_mlp.eval()
      2 with torch.no_grad():
----> 3     simple_mlp_preds = simple_mlp(x_tensor).cpu().numpy()
      4 

NameError: name 'x_tensor' is not defined

## === cell 28
deep_mlp.eval()
with torch.no_grad():
    deep_mlp_preds = deep_mlp(x_tensor).cpu().numpy()



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4193582460.py in <cell line: 0>()
      1 deep_mlp.eval()
      2 with torch.no_grad():
----> 3     deep_mlp_preds = deep_mlp(x_tensor).cpu().numpy()
      4 

NameError: name 'x_tensor' is not defined

## === cell 29
preds = np.mean(
    [ada_boost_preds_test, deep_mlp_preds[:, 0], simple_mlp_preds[:, 0]],
    axis=0,
)

raw_confs = np.mean(
    [conf_test, deep_mlp_preds[:, 1], simple_mlp_preds[:, 1]],
    axis=0,
)
confs = np.clip(raw_confs, 70, 200)



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1711571572.py in <cell line: 0>()
      1 # Combine predictions from all three models
      2 preds = np.mean(
----> 3     [ada_boost_preds_test, deep_mlp_preds[:, 0], simple_mlp_preds[:, 0]],
      4     axis=0,
      5 )

NameError: name 'ada_boost_preds_test' is not defined

## === cell 30
final_df["FVC"] = preds
final_df["Confidence"] = confs
final_df["Patient_Week"] = (
    final_df["Patient_ID"] + "_" + final_df["Target_Week"].astype(str)
)

submission = final_df[["Patient_Week", "FVC", "Confidence"]].copy()
submission["FVC"] = submission["FVC"].round(1)
submission["Confidence"] = submission["Confidence"].round(1)

print(submission.head())
print("Submission shape:", submission.shape)
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3693558344.py in <cell line: 0>()
      1 # Build final submission dataframe
----> 2 final_df["FVC"] = preds
      3 final_df["Confidence"] = confs
      4 final_df["Patient_Week"] = (
      5     final_df["Patient_ID"] + "_" + final_df["Target_Week"].astype(str)

NameError: name 'preds' is not defined
