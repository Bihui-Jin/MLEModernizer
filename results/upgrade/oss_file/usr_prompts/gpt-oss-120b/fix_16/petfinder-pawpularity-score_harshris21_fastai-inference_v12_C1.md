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

3.10

# 3. Installed packages

albumentations==2.0.8
cuml-cu12==25.2.1
fastai==2.8.5
geopandas==0.14.4
libcuml-cu12==25.2.1
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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

17.026066695100802

# 6. Current score

20.08305

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 20.10138) has done: 'I fixed the import error caused by the unavailable cuml package and removed the heavy image‑model pipeline, which also required missing weight files and produced out‑of‑range predictions. Instead, I implemented a lightweight tabular regression using Ridge on the provided binary metadata—this runs on CPU, fits quickly, yields predictions within the required 1‑100 range, and writes a valid submission.csv file.'
- What this solution (achieved 20.10364) has done: 'I add interaction‑only polynomial features and standard‑scale them before the Ridge regression, keeping the same model type but giving it richer inputs that often improve RMSE on this binary metadata. This change is limited to preprocessing and retains the original training‑prediction flow, so it should move the validation error closer to the target lower score.'
- What this solution (achieved 20.09868) has done: 'I add a quick validation split to pick a better Ridge regularisation strength instead of the single α=1.0 used before.  
By trying a few α values on a held‑out 20 % of the training data and then refitting the model on the full set with the best α, we keep the same pipeline (polynomial interactions + scaler + Ridge) while likely lowering the RMSE toward the target. Only the training cell is changed; the rest of the workflow (prediction, clipping, and CSV output) stays untouched.'
- What this solution (achieved 20.08846) has done: 'I replace the manual α‑search with a proper cross‑validated RidgeCV to pick a better regularisation strength, keeping the same polynomial‑interaction + scaler pipeline. This small change should lower the validation RMSE and move the score nearer the target while preserving the core model logic.'
- What this solution (achieved 20.07306) has done: 'I keep the overall Ridge‑with‑polynomial‑features pipeline but improve the preprocessing and hyper‑parameter search so the validation RMSE moves closer to the target.  
Specifically, I (1) expand the alpha grid for RidgeCV, (2) let the polynomial transformer include both interaction and power terms by setting `interaction_only=False`, and (3) increase the polynomial degree to 3 to give the model richer features while still using the same linear‑regression core. These minimal tweaks should lower the score from ~20.09 toward the target 17.02 without changing the fundamental model or workflow.'
- What this solution (achieved 20.07306) has done: 'I add a target‑scaling step so the Ridge model trains on a standardized y value and then inverse‑transforms its predictions back to the original Pawpularity range. This small change keeps the exact same pipeline (polynomial features + Ridge) while improving numerical stability and usually lowers RMSE, moving the score closer to the target.'
- What this solution (achieved 20.0817) has done: 'I lower the model’s complexity and remove the unnecessary target‑scaling step, which tends to over‑fit with degree‑3 polynomial features. By using degree 2 polynomial expansion and training Ridge directly on the original Pawpularity values, the validation RMSE should move closer to the target while keeping the same overall pipeline and output format.'
- What this solution (achieved 20.08061) has done: 'I increase the polynomial feature degree to 3 to give the linear model richer interactions and expand the α‑grid for RidgeCV so the cross‑validation can select a stronger regularisation if needed. These small tweaks keep the exact same pipeline (polynomial + scaler + Ridge) while giving it more expressive power, which should lower the validation RMSE and move the score closer to the target.'
- What this solution (achieved 20.08061) has done: 'I add target‑value scaling (standard‑score) to train the Ridge model on a normalized y and then inverse‑transform the predictions back to the original Pawpularity range. This keeps the exact same polynomial‑feature + Ridge pipeline while improving numerical stability, which should lower the validation RMSE and move the score toward the target.'
- What this solution (achieved 20.08544) has done: 'We reduce the polynomial feature degree from 3 to 2, which lowers model complexity and helps prevent over‑fitting on the limited tabular metadata. This simple change keeps the same pipeline (scaling, RidgeCV, target scaling) while expected to decrease the validation RMSE, moving the score nearer the target lower value.'
- What this solution (achieved 20.08135) has done: 'We modestly expand the feature set by using a degree‑3 polynomial that keeps only interaction terms (avoiding high‑order powers that over‑fit) and broaden the Ridge‑alpha search up to 1e6. This keeps the same Ridge‑based pipeline while giving it a richer yet controlled representation, which should lower validation RMSE and move the score closer to the target.'
- What this solution (achieved 20.08438) has done: 'I remove the manual target‑scaling steps and let the model train directly on the original Pawpularity values. I also switch the polynomial transformer to degree 2 with full power terms (interaction_only = False) and keep the RidgeCV search for the regularisation strength inside the pipeline. This small preprocessing tweak keeps the core Ridge‑based pipeline while expected to lower the validation RMSE, moving the score closer to the target.'
- What this solution (achieved 20.08434) has done: 'I change the polynomial feature generation to keep only interaction terms (`interaction_only=True`). This reduces the feature space, lessening over‑fitting and is expected to lower the validation RMSE, moving the score closer to the target while keeping the same Ridge‑based pipeline.'
- What this solution (achieved 20.08444) has done: 'I increase the polynomial feature degree to 3 and allow full power terms (interaction_only=False) so the Ridge model can capture richer non‑linear relationships while keeping the same linear‑regression pipeline and regularisation search. This modest change adds expressive power without altering the core algorithm, and should lower the validation RMSE, moving the score closer to the target lower value.'
- What this solution (achieved 20.08305) has done: 'I keep the overall Ridge‑with‑polynomial pipeline but change the preprocessing order and reduce the polynomial degree to 2.  
Scaling the raw binary features first, then generating degree‑2 polynomial terms, and scaling them again usually gives a more stable feature set and can lower over‑fitting, which should move the RMSE closer to the target while preserving the core model logic.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split



## === cell 1
base_dir = "/kaggle/input/petfinder-pawpularity-score"
train_path = os.path.join(base_dir, "train.csv")
test_path = os.path.join(base_dir, "test.csv")
submission_path = "/kaggle/working/submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

feature_cols = [
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

X = train_df[feature_cols].astype(float)
y = train_df["Pawpularity"].astype(float)

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True
)

candidate_alphas = np.logspace(-4, 6, 30)  # 0.0001 … 1,000,000

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

model_cv = make_pipeline(
    StandardScaler(), poly, StandardScaler(), RidgeCV(alphas=candidate_alphas, cv=5)
)

model_cv.fit(X_tr, y_tr)

best_alpha = model_cv.named_steps["ridgecv"].alpha_

val_pred = model_cv.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Selected alpha via CV: {best_alpha} with validation RMSE: {val_rmse:.4f}")

model = make_pipeline(
    StandardScaler(), poly, StandardScaler(), Ridge(alpha=best_alpha, random_state=42)
)

model.fit(X, y)

train_pred = model.predict(X)
train_rmse = np.sqrt(mean_squared_error(y, train_pred))
print(f"Training RMSE (full data): {train_rmse:.4f}")



## === cell 2
X_test = test_df[feature_cols].astype(float)

test_pred = model.predict(X_test)

test_pred_clipped = np.clip(test_pred, 1, 100)

submission_df = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": test_pred_clipped})



## === cell 3
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
