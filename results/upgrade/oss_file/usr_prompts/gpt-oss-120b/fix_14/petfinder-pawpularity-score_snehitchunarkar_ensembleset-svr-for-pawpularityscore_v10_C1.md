# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVR, LinearSVR

from torch.utils.data import Dataset  # retained for compatibility

device = "cpu"  # force CPU to avoid CUDA issues



## === cell 1
directory = "/kaggle/input/petfinder-pawpularity-score"
train_df = pd.read_csv(os.path.join(directory, "train.csv"))
test_df = pd.read_csv(os.path.join(directory, "test.csv"))

print("Train samples:", len(train_df))
print("Test samples :", len(test_df))




## === cell 2
class ImageExtract(Dataset):
    """
    Returns a flattened grayscale image (128×128) as a NumPy array.
    This class is retained for interface compatibility but is not used in the
    optimized pipeline.
    """

    def __init__(self, df, root_dir, test=False):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        split = "test" if self.test else "train"
        filename = self.df.loc[idx, "Id"]
        path = os.path.join(self.root_dir, split, filename + ".jpg")
        img = Image.open(path).convert("L").resize((128, 128))
        arr = np.array(img).astype(np.float32).flatten()
        return arr




## === cell 3
def _load_one(img_id, split):
    path = os.path.join(directory, split, f"{img_id}.jpg")
    img = Image.open(path).convert("L").resize((128, 128))
    return np.array(img, dtype=np.float32).ravel()




## === cell 4
max_parallel = max(1, os.cpu_count())

train_ids = train_df["Id"].tolist()
train_images_list = Parallel(n_jobs=max_parallel, backend="threading")(
    delayed(_load_one)(img_id, "train")
    for img_id in tqdm(train_ids, desc="Loading train images")
)
train_images = np.stack(train_images_list).astype(np.float32)

test_ids = test_df["Id"].tolist()
test_images_list = Parallel(n_jobs=max_parallel, backend="threading")(
    delayed(_load_one)(img_id, "test")
    for img_id in tqdm(test_ids, desc="Loading test images")
)
test_images = np.stack(test_images_list).astype(np.float32)



## === cell 5
pca = PCA(n_components=500, random_state=42)  # increased from 200
X_img_train = pca.fit_transform(train_images)
X_img_test = pca.transform(test_images)

print("Image PCA train shape:", X_img_train.shape)
print("Image PCA test shape :", X_img_test.shape)



## === cell 6
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
x_meta_train = train_df[meta_cols].values
x_meta_test = test_df[meta_cols].values

print("Meta train shape:", x_meta_train.shape)
print("Meta test shape :", x_meta_test.shape)



## === cell 7
X_train = np.hstack((X_img_train, x_meta_train))
X_test = np.hstack((X_img_test, x_meta_test))

print("Combined train shape:", X_train.shape)
print("Combined test shape :", X_test.shape)



## === cell 8
y = train_df["Pawpularity"].values
print("Target vector shape:", y.shape)



## === cell 9
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)




## === cell 10
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42)

candidate_C = [200, 400, 800, 1200, 1600, 2500, 3000, 3500, 4000]
candidate_gamma = ["scale", "auto", 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
param_grid = []

for C in candidate_C:
    param_grid.append(("linear", C, None))

for C in candidate_C:
    for gamma in candidate_gamma:
        param_grid.append(("rbf", C, gamma))


def resolve_gamma(gamma, X):
    if gamma == "scale":
        return 1.0 / (X.shape[1] * X.var())
    if gamma == "auto":
        return 1.0 / X.shape[1]
    return float(gamma)


numeric_gammas = set()
for g in candidate_gamma:
    numeric_gammas.add(resolve_gamma(g, X_tr))

kernel_train_dict = {}
kernel_val_dict = {}
for gamma in numeric_gammas:
    K_train = rbf_kernel(X_tr, X_tr, gamma=gamma).astype(np.float32)
    K_val = rbf_kernel(X_val, X_tr, gamma=gamma).astype(np.float32)
    kernel_train_dict[gamma] = K_train
    kernel_val_dict[gamma] = K_val


def eval_params(params):
    kernel, C, gamma = params
    if kernel == "linear":
        model = LinearSVR(C=C, epsilon=0.1, max_iter=400000, random_state=42)
        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_val)
    else:
        numeric_gamma = resolve_gamma(gamma, X_tr)
        K_train = kernel_train_dict[numeric_gamma]
        K_val = kernel_val_dict[numeric_gamma]
        model = SVR(C=C, kernel="precomputed", epsilon=0.1, max_iter=400000)
        model.fit(K_train, y_tr)
        val_pred = model.predict(K_val)
    rmse = RMSE(y_val, val_pred)
    print(f"kernel={kernel}, C={C}, gamma={gamma}, Validation RMSE={rmse:.4f}")
    return (rmse, params)


max_parallel = max(1, min(os.cpu_count(), 4))
results = Parallel(n_jobs=max_parallel, backend="threading")(
    delayed(eval_params)(p) for p in param_grid
)

best_rmse, best_params_tuple = min(results, key=lambda x: x[0])
best_kernel, best_C, best_gamma = best_params_tuple
best_params = {"kernel": best_kernel, "C": best_C, "gamma": best_gamma}

print(
    f"Selected params: kernel={best_params['kernel']}, C={best_params['C']}, "
    f"gamma={best_params['gamma']} with Validation RMSE={best_rmse:.4f}"
)

if best_params["kernel"] == "linear":
    reg = LinearSVR(C=best_params["C"], epsilon=0.1, max_iter=400000, random_state=42)
    reg.fit(X_train, y)
else:
    numeric_gamma = resolve_gamma(best_params["gamma"], X_train)
    K_full = rbf_kernel(X_train, X_train, gamma=numeric_gamma).astype(np.float32)
    reg = SVR(C=best_params["C"], kernel="precomputed", epsilon=0.1, max_iter=400000)
    reg.fit(K_full, y)



## === cell 11
if best_params["kernel"] == "linear":
    train_pred = reg.predict(X_train)
else:
    K_full_train = rbf_kernel(
        X_train, X_train, gamma=resolve_gamma(best_params["gamma"], X_train)
    ).astype(np.float32)
    train_pred = reg.predict(K_full_train)
print("Training RMSE:", RMSE(y, train_pred))



## === cell 12
if best_params["kernel"] == "linear":
    y_pred = reg.predict(X_test)
else:
    K_test = rbf_kernel(
        X_test, X_train, gamma=resolve_gamma(best_params["gamma"], X_train)
    ).astype(np.float32)
    y_pred = reg.predict(K_test)



## === cell 13
submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": y_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
