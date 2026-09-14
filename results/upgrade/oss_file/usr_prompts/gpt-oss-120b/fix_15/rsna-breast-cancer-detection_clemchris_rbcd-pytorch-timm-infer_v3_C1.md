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
Detect breast cancer in mammograms.

## Metric
[Probabilistic F1 score](https://aclanthology.org/2020.eval4nlp-1.9.pdf) (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. 

With pX as the probabilistic version of X:

$$
pF_1 = 2 \frac{pPrecision \cdot pRecall}{pPrecision + pRecall}
$$

where:

$$
pPrecision = \frac{pTP}{pTP + pFP}
$$

$$
pRecall = \frac{pTP}{TP + FN}
$$

## Submission Format
For each `prediction_id`, you should predict the likelihood of cancer in the corresponding `cancer` column. The submission file should have the following format:

```
prediction_id,cancer
0-L,0
0-R,0.5
0-R,0.5
1-L,1
...
# Dataset

**[train/test]_images/[patient_id]/[image_id].dcm** The mammograms, in dicom format. You can expect roughly 8,000 patients in the hidden test set. There are usually but not always 4 images per patient. Note that many of the images use the jpeg 2000 format which may you may need special libraries to load.

**sample_submission.csv** A valid sample submission.

**[train/test].csv** Metadata for each patient and image. Only the first few rows of the test set are available for download.

- `site_id` - ID code for the source hospital.
- `patient_id` - ID code for the patient.
- `image_id` - ID code for the image.
- `laterality` - Whether the image is of the left or right breast.
- `view` - The orientation of the image. The default for a screening exam is to capture two views per breast.
- `age` - The patient's age in years.
- `implant` - Whether or not the patient had breast implants. Site 1 only provides breast implant information at the patient level, not at the breast level.
- `density` - A rating for how dense the breast tissue is, with A being the least dense and D being the most dense. Extremely dense tissue can make diagnosis more difficult. Only provided for train.
- `machine_id` - An ID code for the imaging device.
- `cancer` - Whether or not the breast was positive for malignant cancer. The target value. Only provided for train.
- `biopsy` - Whether or not a follow-up biopsy was performed on the breast. Only provided for train.
- `invasive` - If the breast is positive for cancer, whether or not the cancer proved to be invasive. Only provided for train.
- `BIRADS` - 0 if the breast required follow-up, 1 if the breast was rated as negative for cancer, and 2 if the breast was rated as normal. Only provided for train.
- `prediction_id` - The ID for the matching submission row. Multiple images will share the same prediction ID. Test only.
- `difficult_negative_case` - True if the case was unusually difficult. Only provided for train.

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
joblib==1.5.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
seaborn==0.12.2
sklearn-pandas==2.2.0
timm==1.0.19
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
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        input/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        working/
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
```

-> data/rsna-breast-cancer-detection/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/rsna-breast-cancer-detection/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/rsna-breast-cancer-detection/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> data/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> (stopped after 10 files for performance)

# 5. Target score

0.214194933415826

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from pathlib import Path
import torch  # added for fallback model

try:
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer

    SKLEARN_PRESENT = True
except Exception:
    SKLEARN_PRESENT = False




## === cell 1
KAGGLE_ROOT = Path("/kaggle") if Path("/kaggle").exists() else Path.cwd()
INPUT_DIR = KAGGLE_ROOT / "input" / "rsna-breast-cancer-detection"
OUTPUT_DIR = KAGGLE_ROOT / "working"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV_PATH = INPUT_DIR / "train.csv"
TEST_CSV_PATH = INPUT_DIR / "test.csv"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"  # unified variable for writing




## === cell 2
train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)

y = train_df["cancer"].astype(float)

exclude_cols = ["cancer", "biopsy", "invasive", "BIRADS", "difficult_negative_case"]
X = train_df.drop(columns=exclude_cols)

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

if SKLEARN_PRESENT:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numerical_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", LogisticRegression(max_iter=300, n_jobs=5, solver="saga")),
        ]
    )
else:
    model = None  # placeholder; will use torch fallback

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

if SKLEARN_PRESENT:
    try:
        model.fit(X_train, y_train)
        val_raw = model.predict_proba(X_val)[:, 1]
    except Exception as e:
        print(f"Model training/prediction failed ({e}); falling back to global mean.")
        val_raw = np.full_like(y_val, y_train.mean())
else:
    test_features_raw = test_df.drop(columns=exclude_cols, errors="ignore")
    combined = pd.concat([X, test_features_raw], axis=0, ignore_index=True)
    combined = combined.drop(columns=["prediction_id"], errors="ignore")

    combined = pd.get_dummies(combined, columns=categorical_cols, dummy_na=False)

    X_encoded = combined.iloc[: len(X)].reset_index(drop=True)
    test_encoded = combined.iloc[len(X) :].reset_index(drop=True)

    for col in X_encoded.columns:
        if X_encoded[col].isnull().any():
            median_val = X_encoded[col].median()
            X_encoded[col].fillna(median_val, inplace=True)
            test_encoded[col].fillna(median_val, inplace=True)

    X_train_enc = X_encoded.iloc[X_train.index].values.astype(np.float32)
    X_val_enc = X_encoded.iloc[X_val.index].values.astype(np.float32)
    y_train_enc = y_train.values.astype(np.float32)
    y_val_enc = y_val.values.astype(np.float32)

    device = torch.device("cpu")
    torch.manual_seed(42)

    class SimpleLogisticModel(torch.nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.linear = torch.nn.Linear(n_features, 1)

        def forward(self, x):
            return self.linear(x).squeeze(1)

    n_features = X_train_enc.shape[1]
    torch_model = SimpleLogisticModel(n_features).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-3)

    epochs = 8
    batch_size = 1024
    torch_model.train()
    for epoch in range(epochs):
        perm = np.random.permutation(len(X_train_enc))
        X_train_enc = X_train_enc[perm]
        y_train_enc = y_train_enc[perm]
        for i in range(0, len(X_train_enc), batch_size):
            xb = torch.tensor(X_train_enc[i : i + batch_size], device=device)
            yb = torch.tensor(y_train_enc[i : i + batch_size], device=device)
            optimizer.zero_grad()
            logits = torch_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    torch_model.eval()
    with torch.no_grad():
        val_logits = torch_model(torch.tensor(X_val_enc, device=device))
        val_raw = torch.sigmoid(val_logits).cpu().numpy()

    with torch.no_grad():
        test_logits = torch_model(
            torch.tensor(test_encoded.values.astype(np.float32), device=device)
        )
        raw_probs_test = torch.sigmoid(test_logits).cpu().numpy()

global_mean = y_train.mean()
target_score = 0.214194933415826
best_alpha = 0.5
best_gap = float("inf")
best_score = None

for alpha in np.linspace(0, 1, 21):  # 0, 0.05, ..., 1
    blended = alpha * val_raw + (1 - alpha) * global_mean
    score = (
        lambda probs, truths: (
            lambda tp, fp, fn: 2
            * (tp / (tp + fp + 1e-15))
            * (tp / (tp + fn + 1e-15))
            / ((tp / (tp + fp + 1e-15)) + (tp / (tp + fn + 1e-15)) + 1e-15)
        )(
            np.sum(probs * truths),
            np.sum(probs * (1 - truths)),
            np.sum((1 - probs) * truths),
        )
    )(blended, y_val.values)
    gap = abs(score - target_score)
    if gap < best_gap:
        best_gap = gap
        best_alpha = alpha
        best_score = score

fine_alpha_grid = np.linspace(
    max(0.0, best_alpha - 0.05), min(1.0, best_alpha + 0.05), 21
)  # step ~0.005
for alpha in fine_alpha_grid:
    blended = alpha * val_raw + (1 - alpha) * global_mean
    score = (
        lambda probs, truths: (
            lambda tp, fp, fn: 2
            * (tp / (tp + fp + 1e-15))
            * (tp / (tp + fn + 1e-15))
            / ((tp / (tp + fp + 1e-15)) + (tp / (tp + fn + 1e-15)) + 1e-15)
        )(
            np.sum(probs * truths),
            np.sum(probs * (1 - truths)),
            np.sum((1 - probs) * truths),
        )
    )(blended, y_val.values)
    gap = abs(score - target_score)
    if gap < best_gap:
        best_gap = gap
        best_alpha = alpha
        best_score = score

best_scale = 1.0
best_gap_scale = abs(best_score - target_score)

for scale in np.linspace(0.5, 1.5, 21):  # coarse search
    scaled = (
        np.clip(best_alpha * val_raw + (1 - best_alpha) * global_mean, 0, 1) * scale
    )
    scaled = np.clip(scaled, 0, 1)
    score_scaled = (
        lambda probs, truths: (
            lambda tp, fp, fn: 2
            * (tp / (tp + fp + 1e-15))
            * (tp / (tp + fn + 1e-15))
            / ((tp / (tp + fp + 1e-15)) + (tp / (tp + fn + 1e-15)) + 1e-15)
        )(
            np.sum(probs * truths),
            np.sum(probs * (1 - truths)),
            np.sum((1 - probs) * truths),
        )
    )(scaled, y_val.values)
    gap_scaled = abs(score_scaled - target_score)
    if gap_scaled < best_gap_scale:
        best_gap_scale = gap_scaled
        best_scale = scale
        best_score = score_scaled

fine_scale_grid = np.linspace(
    max(0.5, best_scale - 0.1), min(1.5, best_scale + 0.1), 21
)  # step ~0.01
for scale in fine_scale_grid:
    scaled = (
        np.clip(best_alpha * val_raw + (1 - best_alpha) * global_mean, 0, 1) * scale
    )
    scaled = np.clip(scaled, 0, 1)
    score_scaled = (
        lambda probs, truths: (
            lambda tp, fp, fn: 2
            * (tp / (tp + fp + 1e-15))
            * (tp / (tp + fn + 1e-15))
            / ((tp / (tp + fp + 1e-15)) + (tp / (tp + fn + 1e-15)) + 1e-15)
        )(
            np.sum(probs * truths),
            np.sum(probs * (1 - truths)),
            np.sum((1 - probs) * truths),
        )
    )(scaled, y_val.values)
    gap_scaled = abs(score_scaled - target_score)
    if gap_scaled < best_gap_scale:
        best_gap_scale = gap_scaled
        best_scale = scale
        best_score = score_scaled

if best_score is None:
    best_score = (
        lambda probs, truths: (
            lambda tp, fp, fn: 2
            * (tp / (tp + fp + 1e-15))
            * (tp / (tp + fn + 1e-15))
            / ((tp / (tp + fp + 1e-15)) + (tp / (tp + fn + 1e-15)) + 1e-15)
        )(
            np.sum(probs * truths),
            np.sum(probs * (1 - truths)),
            np.sum((1 - probs) * truths),
        )
    )(global_mean * np.ones_like(y_val), y_val.values)

print(
    f"Selected α = {best_alpha:.4f}, scale = {best_scale:.4f} with validation pF1 = {best_score:.6f} (target {target_score})"
)

if SKLEARN_PRESENT and model is not None:
    try:
        model.fit(X, y)  # same pipeline, full data
    except Exception as e:
        print(f"Full‑data retraining failed ({e}); keeping earlier model.")
global_mean = y.mean()  # overall cancer prevalence for fallback/bias




## === cell 3
test_features = test_df.drop(columns=exclude_cols, errors="ignore").copy()
for col in X.columns:
    if col not in test_features.columns:
        test_features[col] = np.nan

if SKLEARN_PRESENT and model is not None:
    try:
        raw_probs = model.predict_proba(test_features)[:, 1]
    except Exception as e:
        print(f"Prediction failed ({e}); using global mean for all rows.")
        raw_probs = np.full(len(test_features), global_mean)
elif not SKLEARN_PRESENT:
    raw_probs = raw_probs_test
else:
    raw_probs = np.full(len(test_features), global_mean)

try:
    best_alpha_clamped = float(np.clip(best_alpha, 0.0, 1.0))
    best_scale_clamped = float(np.clip(best_scale, 0.0, 1.0))

    blended_probs = (
        best_alpha_clamped * raw_probs + (1 - best_alpha_clamped) * global_mean
    )
    final_probs = np.clip(blended_probs * best_scale_clamped, 0, 1)

    submission = pd.DataFrame(
        {"prediction_id": test_df["prediction_id"], "cancer": final_probs}
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission written to {SUBMISSION_PATH}")
except Exception as e:
    print(
        f"Error during blending/writing ({e}); writing fallback CSV with global mean."
    )
    fallback = pd.DataFrame(
        {
            "prediction_id": test_df["prediction_id"],
            "cancer": np.full(len(test_df), global_mean),
        }
    )
    fallback.to_csv(SUBMISSION_PATH, index=False)
    print(f"Fallback submission written to {SUBMISSION_PATH}")

print("First few rows of the submission:")
print(submission.head() if "submission" in locals() else fallback.head())
