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

-8.559423149042042

# 6. Current score

-9.89339

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -9.89339) has done: 'I fix the runtime errors caused by inconsistent feature column names between training and inference by renaming the test-time columns to exactly match the names used when fitting the `StandardScaler` (and both the XGBoost and MLP pipelines). I also ensure the code runs end-to-end by removing the broken intermediate inference cells’ dependency on undefined variables and producing `submission.csv` with the required columns. These changes are score-neutral in intent (they preserve the same core model logic and learned parameters) but finally yield a valid submission file. Additionally, I add a small safety step to keep prediction/Confidence arrays aligned to `sample_submission` order before writing.'
- What this solution (achieved -9.89339) has done: 'Your current score is worse than the target (gap = -9.89339 − (-8.5594) ≈ -1.334), so we should cautiously improve it without changing the model/training logic. The biggest low-risk gain here is fixing the inference-time feature mismatch: at training you used `Curr_FVC` but at inference `FEATURES_TRAIN` still references `Curr_FVC` while `X_test_tab` was built from `final_df`—we explicitly ensure the same feature list is used and the columns are in the exact same order/dtypes before scaling. Second, the confidence handling in the XGB ensemble is currently inverted and clipped to an odd range (420–1000) which tends to harm the Laplace metric; we keep the ensemble unused for final predictions (to preserve core semantics), but we adjust only the *MLP inference post-processing* to output `Confidence` that matches the metric expectation (sigma, clipped at 70) by removing rounding and ensuring sigma is not artificially capped too low/high. These are minimal, metric-aligned changes that typically improve OSIC score while keeping the architecture/training intact and still writing a valid `submission.csv`.'
- What this solution (achieved -9.89339) has done: 'We should move the score upward (less negative) toward your target, and the most direct low-risk lever (without touching model/training core logic) is calibrating the predicted `Confidence` to match the Laplace metric’s behavior. Right now you cap sigma up to 1000, which typically over-penalizes the `-ln(sigma)` term; we keep the metric-required minimum (70) but cap the maximum to a much smaller, safer value (commonly ~300) to improve the average score. We also ensure the MLP’s predicted sigma isn’t destabilized by extreme `log_sigma` by clipping `log_sigma` before `exp` (this doesn’t change architecture or training, only inference post-processing). Finally, we keep your submission alignment logic intact and still write a valid `submission.csv`.'
- What this solution (achieved -9.89339) has done: 'We keep your training and model architecture exactly the same and only adjust inference-time confidence calibration to better match the Laplace log-likelihood tradeoff (your current fixed cap at 300 is often still too large, hurting the `-ln(sigma)` term). Specifically, we (1) stop clipping `log_sigma` by `log(70)` (which unnecessarily distorts what the model learned) and instead convert to `sigma` then clip to the metric minimum 70, and (2) cap the maximum sigma to a smaller value (200) which usually increases the public score when the model’s FVC errors are not extremely large. We also ensure the predicted `FVC` is cast to float and keep your existing submission alignment/merge logic unchanged so the file remains valid. These are minimal, metric-aligned post-processing changes intended to move the score upward toward your target without changing core learning behavior.'

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
from tqdm import tqdm
import torch.optim as optim
import random
import math




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
try:
    sns.histplot(train["FVC"], kde=True)
    plt.title("Distribution of FVC")
    plt.show()
except Exception as e:
    print("Skipping plot:", e)




## === cell 4
def safe_label_transform(le: LabelEncoder, series: pd.Series) -> np.ndarray:
    classes = set(le.classes_.tolist())
    vals = series.astype(str).tolist()
    mapped = [v if v in classes else le.classes_[0] for v in vals]
    return le.transform(mapped)




## === cell 5
def generate_pairwise_fvc_dataset(df: pd.DataFrame) -> pd.DataFrame:
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




## === cell 6
df = pd.read_csv(base_path + "train.csv")
pairwise_df = generate_pairwise_fvc_dataset(df)
print(pairwise_df.head())



## === cell 7
le_sex = LabelEncoder().fit(pairwise_df["Sex"].astype(str))
le_smoke = LabelEncoder().fit(pairwise_df["SmokingStatus"].astype(str))

pairwise_df["Sex"] = safe_label_transform(le_sex, pairwise_df["Sex"])
pairwise_df["SmokingStatus"] = safe_label_transform(
    le_smoke, pairwise_df["SmokingStatus"]
)

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

X_train_tab = pairwise_df[features]
y_processed = pairwise_df[target]

scaler = StandardScaler()
x_processed = scaler.fit_transform(X_train_tab)



## === cell 8
FEATURES_TRAIN = features




## === cell 9
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
x_data = x_processed
y_data = y_processed

models, weights, maes, seeds = adaBoost(10, x_data, y_data, 75)
print("Collected models:", len(models))
print("MAEs:", maes[:10])




## === cell 12
def normalize_to_sum_one(values):
    total = sum(values)
    if total == 0:
        raise ValueError("Cannot normalize list with sum = 0.")
    return [x / total for x in values]


weights = normalize_to_sum_one(weights)




## === cell 13
def weighted_model_prediction_with_confidence(x_values, models, model_weights):
    if len(models) != len(model_weights):
        raise ValueError("Number of models and weights must be the same.")
    if not abs(sum(model_weights) - 1.0) < 1e-6:
        raise ValueError("Model weights must sum to 1.")

    all_preds = [np.array(model.predict(x_values)) for model in models]
    all_preds = np.array(all_preds)  # (n_models, n_samples)

    weights_arr = np.array(model_weights).reshape(-1, 1)  # (n_models, 1)
    weighted_pred = np.sum(weights_arr * all_preds, axis=0)  # (n_samples,)

    weighted_mean = weighted_pred
    variance = np.sum(weights_arr * (all_preds - weighted_mean) ** 2, axis=0)
    std_dev = np.sqrt(variance)

    confidence = 1 / (1 + std_dev) * 100
    confidence = np.clip(confidence, 420, 1000)

    return weighted_pred.tolist(), confidence.tolist()




## === cell 14
x_tr, x_va, y_tr, y_va = train_test_split(
    x_data, y_data, test_size=0.2, random_state=17
)
ada_boost_preds, conf = weighted_model_prediction_with_confidence(x_va, models, weights)
mae = mean_absolute_error(y_va, ada_boost_preds)
print("Ensemble MAE:", mae)
print("Confidence sample:", conf[:5])




## === cell 15
def compute_loss(y_pred, y_true, sigma):
    f = torch.sqrt(torch.tensor(2.0, device=y_pred.device))
    sigma = torch.exp(sigma)  # ensures positivity
    sigma = torch.clamp(sigma, min=70.0)  # element-wise clamp

    delta = torch.abs(y_pred - y_true)
    delta = torch.clamp(delta, max=1000.0)

    loss = f * delta / sigma + torch.log(f * sigma)
    return loss.mean()




## === cell 16
x = x_processed
y = y_processed

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)




## === cell 17
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
            nn.Linear(32, 2),  # Output: mu and log(sigma)
        )

    def forward(self, x):
        return self.model(x)




## === cell 18
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedMLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

best_state = copy.deepcopy(model.state_dict())
min_loss = float("inf")

for epoch in range(250):
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
    with torch.no_grad():
        val_loss = 0.0
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            y_pred, sigma = pred[:, 0], pred[:, 1]
            loss = compute_loss(y_pred, yb, sigma)
            val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        if avg_val_loss < min_loss:
            min_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            print(f"new best model with avg val loss {avg_val_loss:.6f}")

        print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Epoch {epoch+1}: Loss = {total_loss / max(1, len(train_loader)):.4f}")

model.load_state_dict(best_state)



## === cell 19
test = pd.read_csv(base_path + "test.csv")
sub_sample = pd.read_csv(base_path + "sample_submission.csv")

sub_sample[["Patient_ID", "target_week"]] = sub_sample["Patient_Week"].str.rsplit(
    "_", n=1, expand=True
)
sub_sample["target_week"] = sub_sample["target_week"].astype(int)

test = test.rename(columns={"Patient": "Patient_ID", "Weeks": "Reading_week"})

merged = test.merge(sub_sample[["Patient_ID", "target_week"]], on="Patient_ID")
merged["Weeks_diff"] = merged["target_week"] - merged["Reading_week"]

final_df = merged[
    [
        "Patient_ID",
        "Reading_week",
        "target_week",
        "Weeks_diff",
        "FVC",  # Curr_FVC
        "Percent",
        "Age",
        "Sex",
        "SmokingStatus",
    ]
].rename(columns={"FVC": "Curr_FVC"})

final_df["Sex"] = safe_label_transform(le_sex, final_df["Sex"])
final_df["SmokingStatus"] = safe_label_transform(le_smoke, final_df["SmokingStatus"])

X_test_tab = final_df[FEATURES_TRAIN].copy()
for c in FEATURES_TRAIN:
    X_test_tab[c] = pd.to_numeric(X_test_tab[c], errors="coerce")
X_test_tab = X_test_tab.fillna(X_train_tab.median(numeric_only=True))

X_scaled = scaler.transform(X_test_tab)

print("Inference matrix shape:", X_scaled.shape)



## === cell 20
x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    preds = model(x_tensor)

preds_np = preds.detach().cpu().numpy()
mu = preds_np[:, 0].astype(np.float64)
log_sigma = preds_np[:, 1].astype(np.float64)

sigma = np.exp(log_sigma)
sigma = np.clip(sigma, 70.0, 200.0)

final_df["FVC"] = mu
final_df["Confidence"] = sigma

final_df["Patient_Week"] = (
    final_df["Patient_ID"] + "_" + final_df["target_week"].astype(str)
)

submission = final_df[["Patient_Week", "FVC", "Confidence"]].copy()

submission = sub_sample[["Patient_Week"]].merge(
    submission, on="Patient_Week", how="left", validate="1:1"
)

if submission[["FVC", "Confidence"]].isna().any().any():
    fallback = test.rename(columns={"Patient": "Patient_ID", "Weeks": "Reading_week"})
    fb = fallback.copy()
    fb["Patient_Week"] = fb["Patient_ID"] + "_" + fb["Reading_week"].astype(str)
    fb_map = fb.set_index("Patient_Week")["FVC"].to_dict()
    submission["FVC"] = (
        submission["FVC"]
        .fillna(submission["Patient_Week"].map(fb_map))
        .fillna(test["FVC"].median())
    )
    submission["Confidence"] = submission["Confidence"].fillna(200.0)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("submission shape:", submission.shape)
print("Wrote submission.csv")
