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

-9.66136

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -11.29571) has done: 'I adjust the post‑processing of the neural‑network predictions so that the submitted `Confidence` matches the metric’s expected σ (apply the exponential and clipping that the loss function uses). This small change aligns the output with the competition scoring formula and should raise the score toward the target.'
- What this solution (achieved -11.52299) has done: 'I cap the predicted confidence σ at a more realistic upper bound (200 ml) instead of 1000 ml.  
The model already outputs σ via an exponent and is clipped at a minimum of 70 ml; limiting the maximum reduces the overly‑large ‑ln σ penalty in the Laplace Log Likelihood, moving the score closer to the target while preserving all core logic.'
- What this solution (achieved -11.64548) has done: 'Implemented a tighter upper bound on the confidence (sigma) clipping—from 200 ml to 100 ml. This reduces the penalizing ‑ln(σ) term in the competition metric, moving the score closer to the target while preserving all core modeling logic.'
- What this solution (achieved -12.02358) has done: 'I lower the upper bound used when clipping the predicted confidence σ from 100 ml to 80 ml. This reduces the ‑ln σ penalty in the competition metric while keeping the minimum 70 ml, which should raise the score toward the target without altering the core model or training logic.'
- What this solution (achieved -11.68739) has done: 'The score can be improved by allowing the model‑predicted confidence (σ) to take larger values.  
The competition metric rewards larger σ when the error Δ is big, so clipping σ to a low upper bound (80 ml) makes the Δ/σ term too punitive.  
We raise the upper clipping limit from 80 ml to 2000 ml, keeping the minimum of 70 ml, which better aligns the predictions with the scoring formula and should move the metric toward the target.'
- What this solution (achieved -11.42538) has done: 'I keep the full pipeline unchanged but replace the final prediction step with the already‑computed AdaBoost ensemble outputs, which give more realistic confidence values (clipped 70‑100) and tend to improve the Laplace Log Likelihood score. This small change moves the metric toward the target without altering the core model architecture or training logic.'
- What this solution (achieved -11.58854) has done: 'I raise the upper clipping bound for the predicted confidence σ in the ensemble weighting function. By allowing σ to reach a much larger value (up to 2000 ml) the Laplace Log Likelihood penalty ‑ln σ is reduced, which should improve the score toward the target while keeping all other logic unchanged.'
- What this solution (achieved -11.27124) has done: 'I replace the ad‑boost‑based predictions with the neural‑network outputs that were trained with the same loss used for the competition, and compute the confidence directly from the network’s sigma (exp‑scaled and clipped to the required 70‑2000 range). This aligns the submission values with the metric’s formula and should raise the score toward the target without altering the overall pipeline.'
- What this solution (achieved -11.06689) has done: 'Implemented consistent feature scaling across training and test data.  
- In **cell 10** the scaler fitted on the pairwise training data is saved as `scaler_pairwise`.  
- In **cell 33** the test features are transformed using this original scaler instead of refitting a new one, preserving the training distribution and improving prediction accuracy, which moves the validation score closer to the target.'
- What this solution (achieved -9.48533) has done: 'Implemented a column‑name alignment before applying the previously‑fit scaler.  
The pairwise model was trained on lower‑case feature names (`Reading_week`, `target_week`, `Weeks_diff`).  
During test preparation the columns were capital‑cased, causing a `ValueError`.  
The fix renames these columns to match the scaler’s expected names, then transforms the data with `scaler_pairwise`. This restores the end‑to‑end pipeline and enables generation of a valid `submission.csv`, moving the score toward the target.'
- What this solution (achieved -9.51223) has done: 'Implemented a minor adjustment to the confidence handling during inference.  
The upper clipping of σ at 2000 ml was removed, keeping only the required lower bound of 70 ml.  
This aligns the confidence values more closely with the metric’s formulation and is expected to move the score toward the target without altering the core model or training logic.'
- What this solution (achieved -9.66136) has done: 'I import the missing `torch.nn` module, ensure the neural‑network classes are defined, keep the trained best model, and clip the predicted confidence to the competition’s allowed range (70 – 2000 ml). These fixes resolve the NameError issues, allow the model to be trained and used for inference, and produce a valid `submission.csv` file with correctly calibrated confidence values, moving the score toward the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
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

scaler_pairwise = scaler  # save for later use




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
print(maes)




## === cell 15
print(seeds)




## === cell 16
print(weights)




## === cell 17
def normalize_to_sum_one(values):
    total = sum(values)
    if total == 0:
        raise ValueError("Cannot normalize list with sum = 0.")
    return [x / total for x in values]




## === cell 18
weights = normalize_to_sum_one(weights)




## === cell 19
def weighted_model_prediction_with_confidence(x_values, models, model_weights):
    """
    Produce an ensemble prediction and a confidence (sigma) estimate.
    The confidence is derived from the ensemble variance and clipped
    to the range required by the competition (min 70, max 2000).
    """
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

    confidence = 1 / (1 + std_dev) * 2000
    confidence = np.clip(confidence, 70, 2000)  # lower bound 70, upper bound 2000
    return weighted_pred.tolist(), confidence.tolist()




## === cell 20
x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_data, test_size=0.2, random_state=17
)




## === cell 21
ada_boost_preds, conf = weighted_model_prediction_with_confidence(
    x_val, models, weights
)
mae = mean_absolute_error(y_val, ada_boost_preds)
print(mae)
print(conf[:10])




## === cell 22
test = pd.read_csv(base_path + "test.csv")
sub_sample = pd.read_csv(base_path + "sample_submission.csv")




## === cell 23
print(test.head())
print(sub_sample.head())




## === cell 24
sub_sample[["Patient_ID", "Target_Week"]] = sub_sample["Patient_Week"].str.rsplit(
    "_", n=1, expand=True
)
sub_sample["Target_Week"] = sub_sample["Target_Week"].astype(int)

test = test.rename(columns={"Patient": "Patient_ID", "Weeks": "Reading_Week"})

merged = test.merge(sub_sample[["Patient_ID", "Target_Week"]], on="Patient_ID")
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




## === cell 25
le_sex = LabelEncoder().fit(final_df["Sex"])
le_smoke = LabelEncoder().fit(final_df["SmokingStatus"])

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

X = final_df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(
    X
)  # this scaler is **not** used later; kept for completeness




## === cell 26
def compute_loss(y_pred, y_true, sigma):
    f = torch.sqrt(torch.tensor(2.0, device=y_pred.device))
    sigma = torch.exp(sigma)
    sigma = torch.clamp(sigma, min=70.0)

    delta = torch.abs(y_pred - y_true)
    delta = torch.clamp(delta, max=1000.0)

    loss = f * delta / sigma + torch.log(f * sigma)
    return loss.mean()




## === cell 27
x = x_processed
y = y_processed

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
y_train = y_train.to_numpy()

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)




## === cell 28
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
            nn.Linear(32, 2),  # mu, log(sigma)
        )

    def forward(self, x):
        return self.model(x)




## === cell 29
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)




## === cell 30
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedMLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

best_model = model
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
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            y_pred, sigma = pred[:, 0], pred[:, 1]
            loss = compute_loss(y_pred, yb, sigma)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    if avg_val_loss < min_loss:
        min_loss = avg_val_loss
        best_model = model
        print(f"new best model with loss {avg_val_loss:.4f}")

    print(
        f"Epoch {epoch+1}: train loss {total_loss/len(train_loader):.4f}, val loss {avg_val_loss:.4f}"
    )
model = best_model




## === cell 31
test = pd.read_csv(base_path + "test.csv")
sub_sample = pd.read_csv(base_path + "sample_submission.csv")




## === cell 32
sub_sample[["Patient_ID", "Target_Week"]] = sub_sample["Patient_Week"].str.rsplit(
    "_", n=1, expand=True
)
sub_sample["Target_Week"] = sub_sample["Target_Week"].astype(int)

test = test.rename(columns={"Patient": "Patient_ID", "Weeks": "Reading_Week"})

merged = test.merge(sub_sample[["Patient_ID", "Target_Week"]], on="Patient_ID")
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




## === cell 33
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

X = final_df[features]

X = X.rename(
    columns={
        "Reading_Week": "Reading_week",
        "Target_Week": "target_week",
        "Weeks_Diff": "Weeks_diff",
    }
)

X_scaled = scaler_pairwise.transform(X)




## === cell 34
x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    preds = model(x_tensor)
    mu = preds[:, 0].cpu().numpy()
    raw_log_sigma = preds[:, 1].cpu().numpy()
    sigma = np.exp(raw_log_sigma)
    sigma = np.clip(sigma, 70.0, 2000.0)  # clip to competition range




## === cell 35
final_df["FVC"] = np.maximum(mu, 0.0)
final_df["Confidence"] = sigma

final_df["Patient_Week"] = (
    final_df["Patient_ID"] + "_" + final_df["Target_Week"].astype(str)
)

submission = final_df[["Patient_Week", "FVC", "Confidence"]].copy()
submission["FVC"] = submission["FVC"].round(1)
submission["Confidence"] = submission["Confidence"].round(1)

print(submission.head())
print(submission.shape)
submission.to_csv("submission.csv", index=False)
