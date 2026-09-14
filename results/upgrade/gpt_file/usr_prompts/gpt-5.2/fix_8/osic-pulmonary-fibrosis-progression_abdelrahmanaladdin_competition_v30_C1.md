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

-9.23153

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -9.23153) has done: 'I fix the end-to-end execution by making the test feature column names match exactly what the `StandardScaler` was fit on (case-sensitive), which is the root cause of the current crash. I also ensure the same feature list/order is reused consistently for training and inference, so both XGBoost and the MLPs receive the exact same 8 standardized inputs. Finally, I keep your core modeling and ensembling logic intact, only changing the inference dataframe column names and the scaler input to unblock prediction and write a valid `submission.csv` with the required columns.'
- What this solution (achieved -16.26539) has done: 'The main score gap here is coming from a mismatch between what the metric rewards (good per-patient trajectory across weeks + well-calibrated sigma) and what your current inference does (it uses only the single baseline row in `test.csv` to predict every week). Keeping your exact models, losses, and ensembling, I make the smallest semantic fix: build the test “reading” row from the true baseline measurement in `sample_submission` (the `_0` week row) for each patient, then use that same baseline to predict all target weeks. I also make label encoding robust to unseen categories in test (rare but can crash or mis-map), and keep confidence post-processing identical except for this corrected input row construction. This should move the score upward toward your target without changing core training logic.'
- What this solution (achieved -8.376) has done: 'Your current score is far below the target (gap ≈ -8.85), so we should improve accuracy without changing the model/training core. The biggest issue is that inference is currently using baseline FVC from `sample_submission` (which is not a feature source and can be misleading/incorrect); instead we should use the only true baseline FVC available in `test.csv` (Week=0 row) as `Curr_FVC` for each patient and then generate features for all requested target weeks. To keep semantics identical while improving calibration, we also compute the ensemble confidence as the per-model predictive std (across the three model FVC predictions) and clip it to the metric’s sigma rules—this is a minimal, metric-aligned post-processing change that typically improves Laplace log-likelihood over averaging unrelated sigmas. All paths remain the same and the script still writes a valid `submission.csv` with the required columns and row order matching `sample_submission.csv`.'
- What this solution (achieved -9.23153) has done: 'Your current score (-8.376) is below the target (-7.412), so we should improve (make it less negative) with minimal semantic changes. The biggest low-risk gain here is to make the test-time “reading” row match the competition setup: use the true baseline measurement from `test.csv` at `Weeks == 0` (not the earliest/first week, which can be negative and shifts the whole trajectory). Second, your MLPs output an unconstrained sigma; at inference we should convert each model’s sigma to a valid competition sigma via `exp` (to ensure positivity) and then clip to ≥70, and combine confidences in a metric-aligned way (average sigmas) while still keeping your ensemble mean FVC unchanged. These changes keep your core modeling/training intact, only fixing inference feature construction and confidence post-processing to better match the evaluation metric.'
- What this solution (achieved -9.23153) has done: 'Your score gap is about +1.82 (target is less negative), so we should improve accuracy slightly without changing your core models or training loops. The biggest low-risk fix is correcting a subtle inference bug: your baseline merge can create multiple baseline rows per patient (because of the outer merge), which duplicates predictions and then mismatches them to `sample_submission`—this hurts both FVC and confidence. I replace that merge with a single, deterministic “baseline per patient” selection (prefer `Weeks==0`, else earliest), ensuring exactly one row per patient before expanding to all target weeks. Everything else (features, scaler, XGBoost+MLPs, ensembling, sigma handling, submission formatting/ordering) stays the same.'
- What this solution (achieved -9.23153) has done: 'Your score is worse than the target (gap ≈ -1.82), so we should make a small, low-risk improvement without changing the core modeling/training. The biggest remaining issue is that the MLP’s sigma head is trained as an unconstrained raw value but the loss expects sigma in ml space; we should minimally enforce positivity in the loss by using `softplus` on sigma during training (and keep your existing `exp` at inference unchanged to preserve semantics as much as possible). This typically stabilizes training and yields better-calibrated confidences, which the Laplace log-likelihood metric directly rewards. I also make the baseline selection at test-time strictly “one row per patient preferring Weeks==0 else earliest” (no merge overwrite), which avoids any subtle multi-row/overwrite artifacts while keeping the same intended baseline logic. Everything else (features, scaler, XGBoost adaboost, both MLP architectures, training loops, ensembling, and submission formatting) is kept intact.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import pydicom
import os
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import random
import math
import torch.optim as optim




## === cell 1
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
plt.show()



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
def to_valid_sigma(x, min_sigma=70.0, max_sigma=1000.0):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=min_sigma, posinf=max_sigma, neginf=min_sigma)
    x = np.clip(x, min_sigma, max_sigma)
    return x




## === cell 10
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




## === cell 11
df = pd.read_csv(base_path + "train.csv")
pairwise_df = generate_pairwise_fvc_dataset(df)
print(pairwise_df.head())



## === cell 12
le_sex = LabelEncoder().fit(pairwise_df["Sex"])
le_smoke = LabelEncoder().fit(pairwise_df["SmokingStatus"])

pairwise_df["Sex"] = le_sex.transform(pairwise_df["Sex"])
pairwise_df["SmokingStatus"] = le_smoke.transform(pairwise_df["SmokingStatus"])

FEATURES_TRAIN = [
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

X = pairwise_df[FEATURES_TRAIN]
y_processed = pairwise_df[target]

scaler = StandardScaler()
x_processed = scaler.fit_transform(X)




## === cell 13
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




## === cell 14
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




## === cell 15
x_data = x_processed
y_data = y_processed

models, weights, maes, seeds = adaBoost(10, x_data, y_data, 75)



## === cell 16
print(maes)



## === cell 17
print(seeds)



## === cell 18
print(weights)




## === cell 19
def normalize_to_sum_one(values):
    total = sum(values)
    if total == 0:
        raise ValueError("Cannot normalize list with sum = 0.")
    return [x / total for x in values]




## === cell 20
weights = normalize_to_sum_one(weights)




## === cell 21
def weighted_model_prediction_with_confidence(x_values, models, model_weights):
    if len(models) != len(model_weights):
        raise ValueError("Number of models and weights must be the same.")
    if not abs(sum(model_weights) - 1.0) < 1e-6:
        raise ValueError("Model weights must sum to 1.")

    all_preds = [np.array(model.predict(x_values)) for model in models]
    all_preds = np.array(all_preds)

    weights_arr = np.array(model_weights).reshape(-1, 1)
    weighted_pred = np.sum(weights_arr * all_preds, axis=0)

    weighted_mean = weighted_pred
    variance = np.sum(weights_arr * (all_preds - weighted_mean) ** 2, axis=0)
    std_dev = np.sqrt(np.maximum(variance, 0.0))

    sigma = to_valid_sigma(std_dev, min_sigma=70.0, max_sigma=1000.0)

    return weighted_pred.tolist(), sigma.tolist()




## === cell 22
x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_data, test_size=0.2, random_state=17
)



## === cell 23
ada_boost_preds_val, conf_val = weighted_model_prediction_with_confidence(
    x_val, models, weights
)
mae = mean_absolute_error(y_val, ada_boost_preds_val)
print(mae)
print(conf_val[:10])




## === cell 24
def compute_loss(y_pred, y_true, sigma):
    f = torch.sqrt(torch.tensor(2.0, device=y_pred.device))
    sigma = torch.nn.functional.softplus(sigma)  # > 0, smooth
    sigma = torch.clamp(sigma, min=70.0)

    delta = torch.abs(y_pred - y_true)
    delta = torch.clamp(delta, max=1000.0)

    loss = f * delta / sigma + torch.log(f * sigma)
    return loss.mean()




## === cell 25
x = x_processed
y = y_processed

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)




## === cell 26
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
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.model(x)




## === cell 27
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




## === cell 28
def train_MLP(model, device, optimizer, train_loader, val_loader, num_epochs=100):
    best_state = copy.deepcopy(model.state_dict())
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
            best_state = copy.deepcopy(model.state_dict())
            print(f"new best model with loss {val_loss}")

        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

    model.load_state_dict(best_state)
    return model




## === cell 29
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedMLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

deep_mlp = train_MLP(
    model=model,
    device=device,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
)



## === cell 30
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = SimpleMLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

simple_mlp = train_MLP(
    model=model,
    device=device,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
)



## === cell 31
test = pd.read_csv(base_path + "test.csv")
sub_sample = pd.read_csv(base_path + "sample_submission.csv")

sub_sample[["Patient_ID", "Target_Week"]] = sub_sample["Patient_Week"].str.rsplit(
    "_", n=1, expand=True
)
sub_sample["Target_Week"] = sub_sample["Target_Week"].astype(int)

test = test.rename(columns={"Patient": "Patient_ID", "Weeks": "Reading_week"})

test_sorted = test.sort_values(["Patient_ID", "Reading_week"]).copy()
baseline_first = test_sorted.groupby("Patient_ID", as_index=False).first()
baseline_zero = (
    test_sorted.loc[test_sorted["Reading_week"] == 0]
    .sort_values(["Patient_ID", "Reading_week"])
    .groupby("Patient_ID", as_index=False)
    .first()
)

baseline_test = baseline_first.copy()
baseline_test["has_zero"] = baseline_test["Patient_ID"].isin(
    baseline_zero["Patient_ID"]
)

baseline_zero_idx = baseline_zero.set_index("Patient_ID")
for col in ["Reading_week", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]:
    baseline_test.loc[baseline_test["has_zero"], col] = baseline_test.loc[
        baseline_test["has_zero"], "Patient_ID"
    ].map(baseline_zero_idx[col])

baseline_test = baseline_test.drop(columns=["has_zero"]).copy()
baseline_test = baseline_test.rename(columns={"FVC": "Curr_FVC"})

merged = baseline_test.merge(
    sub_sample[["Patient_ID", "Target_Week"]], on="Patient_ID", how="inner"
)
merged["target_week"] = merged["Target_Week"].astype(int)
merged["Weeks_diff"] = merged["target_week"] - merged["Reading_week"]

final_df = merged[
    [
        "Patient_ID",
        "Reading_week",
        "target_week",
        "Weeks_diff",
        "Curr_FVC",
        "Percent",
        "Age",
        "Sex",
        "SmokingStatus",
    ]
].copy()

print(final_df.head())



## === cell 32
sex_map = {cls: i for i, cls in enumerate(le_sex.classes_)}
smoke_map = {cls: i for i, cls in enumerate(le_smoke.classes_)}

final_df["Sex"] = final_df["Sex"].map(sex_map).fillna(0).astype(int)
final_df["SmokingStatus"] = (
    final_df["SmokingStatus"].map(smoke_map).fillna(0).astype(int)
)

X_test = final_df[FEATURES_TRAIN]
X_scaled = scaler.transform(X_test)



## === cell 33
ada_boost_preds, ada_sigma = weighted_model_prediction_with_confidence(
    X_scaled, models, weights
)
print(ada_boost_preds[:10])
print(ada_sigma[:10])



## === cell 34
x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

simple_mlp.to(device)
simple_mlp.eval()
with torch.no_grad():
    simple_mlp_out = simple_mlp(x_tensor)

deep_mlp.to(device)
deep_mlp.eval()
with torch.no_grad():
    deep_mlp_out = deep_mlp(x_tensor)



## === cell 35
deep_mlp_out = deep_mlp_out.cpu().detach().numpy()
simple_mlp_out = simple_mlp_out.cpu().detach().numpy()

deep_sigma = to_valid_sigma(
    np.exp(deep_mlp_out[:, 1]), min_sigma=70.0, max_sigma=1000.0
)
simple_sigma = to_valid_sigma(
    np.exp(simple_mlp_out[:, 1]), min_sigma=70.0, max_sigma=1000.0
)

preds = np.mean(
    [np.array(ada_boost_preds), deep_mlp_out[:, 0], simple_mlp_out[:, 0]], axis=0
)

confs = to_valid_sigma(
    (np.array(ada_sigma) + deep_sigma + simple_sigma) / 3.0,
    min_sigma=70.0,
    max_sigma=1000.0,
)

print(preds[:10])
print(confs[:10])



## === cell 36
final_df["FVC"] = preds
final_df["Confidence"] = confs

final_df["Patient_Week"] = (
    final_df["Patient_ID"] + "_" + final_df["target_week"].astype(str)
)

submission = final_df[["Patient_Week", "FVC", "Confidence"]].copy()
submission["FVC"] = submission["FVC"].round(1)
submission["Confidence"] = submission["Confidence"].round(1)

submission = sub_sample[["Patient_Week"]].merge(
    submission, on="Patient_Week", how="left"
)
submission["FVC"] = submission["FVC"].fillna(sub_sample["FVC"])
submission["Confidence"] = submission["Confidence"].fillna(sub_sample["Confidence"])

print(submission.head())
print(submission.shape)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
