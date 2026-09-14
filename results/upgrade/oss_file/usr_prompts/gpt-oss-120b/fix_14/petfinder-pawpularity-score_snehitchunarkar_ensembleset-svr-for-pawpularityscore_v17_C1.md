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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.12

# 3. Installed packages

cuml-cu12==25.2.1
geopandas==0.14.4
libcuml-cu12==25.2.1
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
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
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

17.65498124995247

# 6. Current score

20.96529

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 20.62508) has done: 'I fixed the column slicing so that only the true metadata columns are used as features (the target column is no longer included in X_train). This resolves the dimension mismatch when stacking train and test data for scaling and eliminates the subsequent NameError cascades. The script now runs end‑to‑end, creates a properly scaled feature set, trains the SVR model, predicts the test set, and writes a valid `submission.csv` file with the required columns.'
- What this solution (achieved 20.71892) has done: 'I scale the target variable before fitting the SVR and inverse‑transform the predictions back to the original range. This small change often improves regression performance while keeping the same model and feature handling. I also slightly increase the SVR regularisation (C) to give the model more flexibility, which should reduce the RMSE and move the score closer to the target.'
- What this solution (achieved 20.77004) has done: 'I keep the overall pipeline unchanged and only adjust the SVR hyper‑parameters to give the model more flexibility (increase C and reduce epsilon) while fixing a missing random seed for reproducibility. These small tweaks are expected to lower the RMSE and move the score toward the target without altering the core logic.'
- What this solution (achieved 20.71392) has done: 'I keep the overall pipeline unchanged but add a lightweight polynomial feature expansion after scaling the metadata and adjust the SVR hyper‑parameters slightly (reduce C and set a small epsilon). The extra interaction features give the model a bit more expressive power without altering the core architecture, and the modest hyper‑parameter change helps regularise the fit, which should lower the RMSE toward the target.'
- What this solution (achieved 20.75955) has done: 'I remove the unnecessary scaling of the target variable and train the SVR directly on the original Pawpularity scores, while also increasing the regularisation strength (C) and setting epsilon to zero. This small change respects the existing pipeline, keeps the same features and model type, and is expected to lower the RMSE, moving the score closer to the target.'
- What this solution (achieved 20.71831) has done: 'I add a lightweight validation step that tries a few sensible SVR hyper‑parameter combos and optionally includes or excludes the polynomial expansion. The best‑performing setting on the held‑out split is then used to retrain on the full data, keeping the original model type and feature handling but allowing a modest improvement in RMSE, moving the score closer to the target.'
- What this solution (achieved 20.71892) has done: 'I add a lightweight target‑scaling step (fit a StandardScaler on y, train the SVR on the scaled target, then inverse‑transform predictions) and expand the hyper‑parameter search with a few more sensible SVR settings (including a higher C and a small epsilon). This keeps the model type and feature handling unchanged, but better‑conditions the regression problem and gives the validation loop a chance to pick a slightly better configuration, which should lower the final RMSE toward the target score. I also clip the final predictions to the valid 0‑100 range before writing the submission.'
- What this solution (achieved 20.71892) has done: 'The changes cache the polynomial feature expansion (instead of recomputing it for every hyper‑parameter trial) and evaluate the 120 SVR configurations in parallel using joblib, which dramatically reduces the overall runtime while keeping the exact same models, data splits, and scoring logic. All other steps—including scaling, training the final model, and creating the submission—remain unchanged.'
- What this solution (achieved 20.96529) has done: 'The changes limit the hyper‑parameter search to a small random subset of the training data, which makes each SVR fit much faster while keeping the same grid‑search logic and the final model trained on the full dataset unchanged. A reproducible random state guarantees deterministic results, and all later steps (full‑training, prediction, ensembling) remain exactly the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

from joblib import Parallel, delayed

np.random.seed(42)




## === cell 1
device = "cuda" if torch.cuda.is_available() else "cpu"

directory = "/kaggle/input/petfinder-pawpularity-score"

train_df = pd.read_csv(os.path.join(directory, "train.csv"))
test_df = pd.read_csv(os.path.join(directory, "test.csv"))

print("Train samples:", len(train_df), "\nTest samples:", len(test_df), "\n")




## === cell 2
X_train_raw = train_df.iloc[:, 1:13].values
X_test_raw = test_df.iloc[:, 1:13].values
y_train = train_df["Pawpularity"].values

scaler = StandardScaler()
scaler.fit(np.vstack((X_train_raw, X_test_raw)))
X_train_scaled_all = scaler.transform(X_train_raw)
X_test_scaled_all = scaler.transform(X_test_raw)

print(
    "Scaled feature shape train:",
    X_train_scaled_all.shape,
    "test:",
    X_test_scaled_all.shape,
)




## === cell 3

search_size = min(2000, X_train_scaled_all.shape[0])  # cap at 2000 samples
rng = np.random.RandomState(42)
search_idx = rng.choice(X_train_scaled_all.shape[0], size=search_size, replace=False)

X_search = X_train_scaled_all[search_idx]
y_search = y_train[search_idx]

candidate_params = {
    "use_poly": [False, True],
    "kernel": ["rbf", "linear"],
    "C": [200, 500, 1000, 2000, 5000, 10000, 20000],
    "epsilon": [0.0, 0.001, 0.01, 0.1],
    "gamma": ["scale", "auto"],  # only used for rbf, ignored otherwise
}
candidates = []
for use_poly in candidate_params["use_poly"]:
    for kernel in candidate_params["kernel"]:
        for C in candidate_params["C"]:
            for epsilon in candidate_params["epsilon"]:
                for gamma in candidate_params["gamma"]:
                    cfg = {
                        "use_poly": use_poly,
                        "kernel": kernel,
                        "C": C,
                        "epsilon": epsilon,
                        "gamma": gamma,
                    }
                    candidates.append(cfg)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_search, y_search, test_size=0.2, random_state=42
)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_tr_poly = poly.fit_transform(X_tr)
X_val_poly = poly.transform(X_val)

y_scaler_val = StandardScaler()
y_tr_scaled = y_scaler_val.fit_transform(y_tr.reshape(-1, 1)).ravel()
y_val_scaled = y_scaler_val.transform(y_val.reshape(-1, 1)).ravel()


def evaluate_cfg(cfg):
    if cfg["use_poly"]:
        X_tr_feat = X_tr_poly
        X_val_feat = X_val_poly
    else:
        X_tr_feat = X_tr
        X_val_feat = X_val

    svr = SVR(
        C=cfg["C"],
        epsilon=cfg["epsilon"],
        kernel=cfg["kernel"],
        degree=3,
        gamma=cfg["gamma"] if cfg["kernel"] == "rbf" else "auto",
        max_iter=400000,
    )
    svr.fit(X_tr_feat, y_tr_scaled)

    val_pred_scaled = svr.predict(X_val_feat)
    val_pred = y_scaler_val.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()
    rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
    return cfg, rmse, svr


results = Parallel(n_jobs=-1, prefer="threads", batch_size=10)(
    delayed(evaluate_cfg)(cfg) for cfg in candidates
)

best_rmse = np.inf
best_cfg = None
second_best_cfg = None
best_rmse_second = np.inf

for cfg, rmse, _ in results:
    print(f"Cfg {cfg} -> validation RMSE: {rmse:.4f}")
    if rmse < best_rmse:
        second_best_cfg = best_cfg
        best_rmse_second = best_rmse
        best_rmse = rmse
        best_cfg = cfg
    elif rmse < best_rmse_second:
        second_best_cfg = cfg
        best_rmse_second = rmse

print("\nBest configuration:", best_cfg, "validation RMSE:", best_rmse)
print(
    "Second‑best configuration:", second_best_cfg, "validation RMSE:", best_rmse_second
)




## === cell 4
if best_cfg["use_poly"]:
    poly_full = PolynomialFeatures(degree=2, include_bias=False)
    X_train_final = poly_full.fit_transform(X_train_scaled_all)
    X_test_final = poly_full.transform(X_test_scaled_all)
else:
    X_train_final = X_train_scaled_all
    X_test_final = X_test_scaled_all

y_scaler_full = StandardScaler()
y_train_scaled_full = y_scaler_full.fit_transform(y_train.reshape(-1, 1)).ravel()

svr_best = SVR(
    C=best_cfg["C"],
    epsilon=best_cfg["epsilon"],
    kernel=best_cfg["kernel"],
    degree=3,
    gamma=best_cfg["gamma"] if best_cfg["kernel"] == "rbf" else "auto",
    max_iter=400000,
)
svr_best.fit(X_train_final, y_train_scaled_full)

if second_best_cfg is not None:
    if second_best_cfg["use_poly"]:
        poly_sec = PolynomialFeatures(degree=2, include_bias=False)
        X_train_sec = poly_sec.fit_transform(X_train_scaled_all)
        X_test_sec = poly_sec.transform(X_test_scaled_all)
    else:
        X_train_sec = X_train_scaled_all
        X_test_sec = X_test_scaled_all

    svr_sec = SVR(
        C=second_best_cfg["C"],
        epsilon=second_best_cfg["epsilon"],
        kernel=second_best_cfg["kernel"],
        degree=3,
        gamma=(
            second_best_cfg["gamma"] if second_best_cfg["kernel"] == "rbf" else "auto"
        ),
        max_iter=400000,
    )
    svr_sec.fit(X_train_sec, y_train_scaled_full)

train_pred_scaled = svr_best.predict(X_train_final)
train_pred = y_scaler_full.inverse_transform(train_pred_scaled.reshape(-1, 1)).ravel()
train_rmse_full = np.sqrt(np.mean((y_train - train_pred) ** 2))
print("Full‑training RMSE (best model):", train_rmse_full)

y_pred_scaled = svr_best.predict(X_test_final)
y_pred = y_scaler_full.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

if second_best_cfg is not None:
    y_pred_scaled_sec = svr_sec.predict(X_test_sec)
    y_pred_sec = y_scaler_full.inverse_transform(
        y_pred_scaled_sec.reshape(-1, 1)
    ).ravel()
    y_pred = (y_pred + y_pred_sec) / 2.0

bias = np.mean(y_train - train_pred)
y_pred = y_pred + bias

y_pred = np.clip(y_pred, 0, 100)
print("First 10 corrected predictions:", y_pred[:10])




## === cell 5
submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": y_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file saved to {submission_path}")




## === cell 6
print(submission.head())
