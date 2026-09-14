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

17.652304483843338

# 6. Current score

20.11653

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 20.62622) has done: 'The fix removes the GPU‑only `cuml` import, replaces it with a CPU‑based `sklearn` SVR, and streamlines the pipeline to use only the provided metadata features. This prevents runtime crashes, ensures a complete end‑to‑end run, and writes a proper `submission.csv` with the required columns.'
- What this solution (achieved 20.76381) has done: 'I add a small validation split to monitor out‑of‑sample RMSE and modestly adjust the SVR hyper‑parameters (increase C and remove epsilon) which usually reduces error on this type of metadata. The model is still an SVR on the same standardized features, preserving the core logic, and the script still writes a correct `submission.csv` file.'
- What this solution (achieved 20.68904) has done: 'I add a lightweight polynomial feature expansion (degree 2) to capture interactions between the metadata columns, then keep the same StandardScaler and SVR model but with a slightly lower C (30) and a small epsilon (0.1) to reduce over‑fitting. These minimal changes preserve the overall pipeline while likely lowering the validation RMSE, moving the score closer to the target.'
- What this solution (achieved 20.08411) has done: 'I keep the overall pipeline (polynomial features, scaling, SVR) but raise the SVR’s regularization (C = 100, epsilon = 0) and introduce a tiny blend with the overall training‑set mean. The blend weight is chosen on the validation split to lower the validation RMSE, and the same weight is applied to the test predictions, which should reduce over‑fitting and move the score closer to the target while preserving the core logic.'
- What this solution (achieved 20.08411) has done: 'I add a tiny grid‑search over a few SVR C and epsilon values (keeping the same polynomial‑expanded, scaled metadata) and pick the combination that gives the lowest blended validation RMSE. The best blending weight from that sweep is then used for the final model trained on all data, keeping the rest of the pipeline unchanged. This modest hyper‑parameter tweak is expected to lower the validation error and thus move the leaderboard RMSE closer to the target while preserving the original logic.'
- What this solution (achieved 20.08411) has done: 'I reduce data leakage by fitting the polynomial feature generator and the scaler only on the training data (instead of also on the test set). This makes the preprocessing more realistic and often improves out‑of‑sample RMSE. I also broaden the SVR hyper‑parameter search slightly (extra C and epsilon values) so the model can find a marginally better blend without changing the overall pipeline.'
- What this solution (achieved 20.08411) has done: 'The changes add a parallel grid‑search with `joblib` and replace the inner weight‑loop with a fully vectorized computation, so every SVR fit is done only once per hyper‑parameter set and the blend‑weight evaluation incurs virtually no Python overhead. The rest of the pipeline (feature engineering, scaling, final training, and submission) stays exactly the same, preserving the original model logic and results.'
- What this solution (achieved 20.08411) has done: 'I keep the original SVR‑based pipeline but improve the validation search to keep the top 3 configurations instead of only the single best one. Each of those configurations is trained on the full data, their predictions are averaged, and the blended weight `w` is also averaged. This small ensemble often lowers the validation RMSE without changing the core model type or feature engineering, moving the score closer to the target.'
- What this solution (achieved 20.12189) has done: 'I compute a single optimal blending weight `w` that minimizes the training RMSE (instead of averaging the individual weights from the top‑k configurations). This small calibration keeps the same SVR + polynomial pipeline while likely lowering the blended error, moving the score closer to the target. The change is limited to the blending step and does not alter model architecture or hyper‑parameter search.'
- What this solution (achieved 20.11682) has done: 'I add a degree‑1 candidate and a finer epsilon value to give the grid a slightly broader search, and I replace the simple un‑weighted averaging of the top‑k models with a weighted blend that gives more influence to configurations that performed better on the validation split. This keeps the overall SVR‑polynomial pipeline intact while nudging the predictions toward lower RMSE, moving the score closer to the target.'
- What this solution (achieved 20.11653) has done: 'The changes replace the CPU‑based `sklearn.svm.SVR` with GPU‑accelerated `cuml.svm.SVR`, moving data to CuPy arrays only once per model. All polynomial feature construction, scaling, and data splitting stay unchanged, preserving the exact training/validation logic. Converting predictions back to NumPy for RMSE calculations keeps the original evaluation semantics while gaining a large speed boost that fits the 600‑second limit.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cupy as cp
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from cuml.svm import (
    SVR as CumlSVR,
)  # GPU‑accelerated SVR, drop‑in replacement for sklearn's SVR




## === cell 1
directory = "/kaggle/input/petfinder-pawpularity-score"




## === cell 2
train_df = pd.read_csv(os.path.join(directory, "train.csv"))
test_df = pd.read_csv(os.path.join(directory, "test.csv"))
print("Train rows:", len(train_df), "Test rows:", len(test_df))




## === cell 3
meta_cols = [
    "Subject Focus",
    "Eyes",
    "Face",
    "Near",
    "Action",
    "Accessory",
    "Group",
    "Collage",
    "Human",
    "Occlusion",
    "Info",
    "Blur",
]
X_train_raw = train_df[meta_cols].values.astype(np.float32)
X_test_raw = test_df[meta_cols].values.astype(np.float32)
y_train = train_df["Pawpularity"].values.astype(np.float32)

global_mean = y_train.mean()

X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
    X_train_raw, y_train, test_size=0.2, random_state=42
)

degree_candidates = [1, 2, 3]
C_candidates = [80, 100, 120, 150, 200, 250, 300, 400, 500, 600, 700, 800]
epsilon_candidates = [0.0, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2]
gamma_candidates = ["scale", "auto"]

best_overall_rmse = np.inf
best_cfg = {}
all_results = []  # keep every evaluated configuration
top_k = 3  # number of configurations to ensemble
w_candidates = np.linspace(0, 1, 21)  # step 0.05

for degree in degree_candidates:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly.fit(X_tr_raw)
    X_tr = poly.transform(X_tr_raw)
    X_val = poly.transform(X_val_raw)

    scaler = StandardScaler()
    scaler.fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_val = scaler.transform(X_val)

    X_val_gpu = cp.asarray(X_val, dtype=cp.float32)
    y_val_gpu = cp.asarray(y_val, dtype=cp.float32)

    def evaluate_cfg(C, eps, gamma):
        X_tr_gpu = cp.asarray(X_tr, dtype=cp.float32)
        y_tr_gpu = cp.asarray(y_tr, dtype=cp.float32)

        model = CumlSVR(C=C, epsilon=eps, kernel="rbf", gamma=gamma)
        model.fit(X_tr_gpu, y_tr_gpu)

        preds_gpu = model.predict(X_val_gpu)  # (n_val,)
        preds = cp.asnumpy(preds_gpu)  # back to NumPy for rmse logic

        diff = y_val[:, None] - (
            w_candidates * preds[:, None] + (1 - w_candidates) * global_mean
        )
        mse = np.mean(diff**2, axis=0)
        rmse_arr = np.sqrt(mse)
        best_idx = np.argmin(rmse_arr)
        return {
            "rmse": rmse_arr[best_idx],
            "w": w_candidates[best_idx],
            "C": C,
            "epsilon": eps,
            "gamma": gamma,
            "degree": degree,
            "poly": poly,
            "scaler": scaler,
        }

    param_grid = [
        (C, eps, gamma)
        for C in C_candidates
        for eps in epsilon_candidates
        for gamma in gamma_candidates
    ]

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(evaluate_cfg)(C, eps, gamma) for (C, eps, gamma) in param_grid
    )

    for cfg in results:
        all_results.append(cfg)
        if cfg["rmse"] < best_overall_rmse:
            best_overall_rmse = cfg["rmse"]
            best_cfg = cfg

top_cfgs = sorted(all_results, key=lambda x: x["rmse"])[:top_k]

print(
    f"Best config -> degree:{best_cfg['degree']}, C:{best_cfg['C']}, "
    f"epsilon:{best_cfg['epsilon']}, gamma:{best_cfg['gamma']}, w:{best_cfg['w']:.2f}"
)
print(f"Validation RMSE (blended, best single): {best_overall_rmse:.4f}")
print(f"Ensembling top {top_k} configs for final predictions")




## === cell 4
train_preds_list = []
test_preds_list = []
w_values = []

for cfg in top_cfgs:
    poly = cfg["poly"]
    X_full = poly.transform(X_train_raw)
    scaler_full = StandardScaler()
    scaler_full.fit(X_full)
    X_full = scaler_full.transform(X_full)

    X_test = poly.transform(X_test_raw)
    X_test = scaler_full.transform(X_test)

    X_full_gpu = cp.asarray(X_full, dtype=cp.float32)
    y_train_gpu = cp.asarray(y_train, dtype=cp.float32)
    X_test_gpu = cp.asarray(X_test, dtype=cp.float32)

    model = CumlSVR(
        C=cfg["C"], epsilon=cfg["epsilon"], kernel="rbf", gamma=cfg["gamma"]
    )
    model.fit(X_full_gpu, y_train_gpu)

    train_pred_gpu = model.predict(X_full_gpu)
    test_pred_gpu = model.predict(X_test_gpu)

    train_pred = cp.asnumpy(train_pred_gpu)
    test_pred = cp.asnumpy(test_pred_gpu)

    train_preds_list.append(train_pred)
    test_preds_list.append(test_pred)
    w_values.append(cfg["w"])

val_rmses = np.array([cfg["rmse"] for cfg in top_cfgs])
inv_weights = 1.0 / (val_rmses + 1e-12)  # avoid divide‑by‑zero
norm_weights = inv_weights / inv_weights.sum()

train_pred_ensembled = np.dot(np.column_stack(train_preds_list), norm_weights)
test_pred_ensembled = np.dot(np.column_stack(test_preds_list), norm_weights)

delta = train_pred_ensembled - global_mean
numer = np.sum((y_train - global_mean) * delta)
denom = np.sum(delta**2) + 1e-12
optimal_w = np.clip(numer / denom, 0.0, 1.0)

train_blended = optimal_w * train_pred_ensembled + (1 - optimal_w) * global_mean
train_rmse = np.sqrt(np.mean((y_train - train_blended) ** 2))
print(f"Training RMSE (blended, ensemble with optimal w): {train_rmse:.4f}")




## === cell 5
test_pred = optimal_w * test_pred_ensembled + (1 - optimal_w) * global_mean




## === cell 6
submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": test_pred})
submission["Pawpularity"] = submission["Pawpularity"].clip(0, 100)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
