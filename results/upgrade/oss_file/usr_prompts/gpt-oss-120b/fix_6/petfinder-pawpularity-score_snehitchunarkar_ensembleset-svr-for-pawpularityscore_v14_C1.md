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

20.08411

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 20.62622) has done: 'The fix removes the GPU‑only `cuml` import, replaces it with a CPU‑based `sklearn` SVR, and streamlines the pipeline to use only the provided metadata features. This prevents runtime crashes, ensures a complete end‑to‑end run, and writes a proper `submission.csv` with the required columns.'
- What this solution (achieved 20.76381) has done: 'I add a small validation split to monitor out‑of‑sample RMSE and modestly adjust the SVR hyper‑parameters (increase C and remove epsilon) which usually reduces error on this type of metadata. The model is still an SVR on the same standardized features, preserving the core logic, and the script still writes a correct `submission.csv` file.'
- What this solution (achieved 20.68904) has done: 'I add a lightweight polynomial feature expansion (degree 2) to capture interactions between the metadata columns, then keep the same StandardScaler and SVR model but with a slightly lower C (30) and a small epsilon (0.1) to reduce over‑fitting. These minimal changes preserve the overall pipeline while likely lowering the validation RMSE, moving the score closer to the target.'
- What this solution (achieved 20.08411) has done: 'I keep the overall pipeline (polynomial features, scaling, SVR) but raise the SVR’s regularization (C = 100, epsilon = 0) and introduce a tiny blend with the overall training‑set mean. The blend weight is chosen on the validation split to lower the validation RMSE, and the same weight is applied to the test predictions, which should reduce over‑fitting and move the score closer to the target while preserving the core logic.'
- What this solution (achieved 20.08411) has done: 'I add a tiny grid‑search over a few SVR C and epsilon values (keeping the same polynomial‑expanded, scaled metadata) and pick the combination that gives the lowest blended validation RMSE. The best blending weight from that sweep is then used for the final model trained on all data, keeping the rest of the pipeline unchanged. This modest hyper‑parameter tweak is expected to lower the validation error and thus move the leaderboard RMSE closer to the target while preserving the original logic.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split




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

poly = PolynomialFeatures(degree=2, include_bias=False)
poly.fit(np.vstack((X_train_raw, X_test_raw)))
X_train = poly.transform(X_train_raw)
X_test = poly.transform(X_test_raw)

print("Feature shape after polynomial expansion:", X_train.shape)




## === cell 4
scaler = StandardScaler()
scaler.fit(np.vstack((X_train, X_test)))
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)




## === cell 5
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

global_mean = y_train.mean()

C_candidates = [30, 60, 100, 150]
epsilon_candidates = [0.0, 0.1, 0.2]

best_rmse = np.inf
best_params = None
best_w = 1.0

for C in C_candidates:
    for eps in epsilon_candidates:
        model = SVR(C=C, epsilon=eps, kernel="rbf", gamma="scale")
        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_val)

        w_candidates = np.linspace(0, 1, 11)
        for w in w_candidates:
            blended = w * val_pred + (1 - w) * global_mean
            rmse = np.sqrt(np.mean((y_val - blended) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = (C, eps)
                best_w = w

print(f"Selected SVR params: C={best_params[0]}, epsilon={best_params[1]}")
print(f"Best blending weight on validation: {best_w:.2f}")
print(f"Validation RMSE (blended): {best_rmse:.4f}")

svr = SVR(C=best_params[0], epsilon=best_params[1], kernel="rbf", gamma="scale")
svr.fit(X_train, y_train)

train_pred = svr.predict(X_train)
train_blended = best_w * train_pred + (1 - best_w) * global_mean
train_rmse = np.sqrt(np.mean((y_train - train_blended) ** 2))
print(f"Training RMSE (blended, full data): {train_rmse:.4f}")




## === cell 6
test_pred = svr.predict(X_test)
test_pred = best_w * test_pred + (1 - best_w) * global_mean




## === cell 7
submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": test_pred})
submission["Pawpularity"] = submission["Pawpularity"].clip(0, 100)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
