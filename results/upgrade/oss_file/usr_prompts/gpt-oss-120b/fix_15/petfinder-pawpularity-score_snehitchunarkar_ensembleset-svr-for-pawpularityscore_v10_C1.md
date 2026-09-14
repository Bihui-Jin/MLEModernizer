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

18.2230167933914

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 21.47864) has done: 'I remove the GPU‑only `cuml` import and the CLIP/EfficientNet sections that crash, and replace them with a simple grayscale‑image feature extractor using PCA. The metadata columns are kept, scaled, and combined with the PCA image features. I train a CPU‑based `sklearn.svm.SVR` model, predict the test set, and write a proper `submission.csv`. All changes are minimal and keep the overall pipeline (feature extraction → scaling → SVR regression) intact.'
- What this solution (achieved 20.55842) has done: 'The changes increase the PCA image feature dimension to capture more visual information and add a quick validation split to pick the best SVR C‑parameter among a few options, then retrain on the full data. This modest tuning is expected to lower the RMSE from the current ~21.5 toward the target ~18.2 while keeping the original pipeline intact.'
- What this solution (achieved 20.08474) has done: 'I increase the image PCA dimensionality to retain more visual information and expand the SVR hyper‑parameter search (trying larger C values and several γ settings). These incremental tweaks stay within the original pipeline while expected to lower the validation RMSE, moving the final score closer to the target 18.22.'
- What this solution (achieved 20.08471) has done: 'I increase the image PCA dimensionality from 300 to 500 to capture more visual information and expand the SVR hyper‑parameter search (larger C values and finer‑grained small γ values). These minimal adjustments keep the original pipeline intact while giving the model more capacity to reduce the validation RMSE, moving the score closer to the target.'

# 9. Code solution

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
max_parallel = max(1, min(os.cpu_count(), 4))

train_ids = train_df["Id"].tolist()
train_images_list = Parallel(
    n_jobs=max_parallel, backend="threading", prefer="threads"
)(
    delayed(_load_one)(img_id, "train")
    for img_id in tqdm(train_ids, desc="Loading train images")
)
train_images = np.stack(train_images_list).astype(np.float32)

test_ids = test_df["Id"].tolist()
test_images_list = Parallel(n_jobs=max_parallel, backend="threading", prefer="threads")(
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
        model = LinearSVR(C=C, epsilon=0.1, max_iter=10000, random_state=42, dual=False)
        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_val)
    else:
        numeric_gamma = resolve_gamma(gamma, X_tr)
        K_train = kernel_train_dict[numeric_gamma]
        K_val = kernel_val_dict[numeric_gamma]
        model = SVR(
            C=C, kernel="precomputed", epsilon=0.1, max_iter=10000, shrinking=False
        )
        model.fit(K_train, y_tr)
        val_pred = model.predict(K_val)
    rmse = RMSE(y_val, val_pred)
    print(f"kernel={kernel}, C={C}, gamma={gamma}, Validation RMSE={rmse:.4f}")
    return (rmse, params)


max_parallel = max(1, min(os.cpu_count(), 4))
results = Parallel(n_jobs=max_parallel, backend="loky", prefer="processes")(
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
    reg = LinearSVR(
        C=best_params["C"], epsilon=0.1, max_iter=10000, random_state=42, dual=False
    )
    reg.fit(X_train, y)
else:
    numeric_gamma = resolve_gamma(best_params["gamma"], X_train)
    K_full = rbf_kernel(X_train, X_train, gamma=numeric_gamma).astype(np.float32)
    reg = SVR(
        C=best_params["C"],
        kernel="precomputed",
        epsilon=0.1,
        max_iter=10000,
        shrinking=False,
    )
    reg.fit(K_full, y)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 490, in _process_worker
    r = call_item()
        ^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 291, in __call__
    return self.fn(*self.args, **self.kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in __call__
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in <listcomp>
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
            ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/169353810.py", line 45, in eval_params
  File "/usr/local/lib/python3.11/dist-packages/sklearn/svm/_classes.py", line 518, in fit
    self.coef_, self.intercept_, n_iter_ = _fit_liblinear(
                                           ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/svm/_base.py", line 1223, in _fit_liblinear
    solver_type = _get_liblinear_solver_type(multi_class, penalty, loss, dual)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/svm/_base.py", line 1062, in _get_liblinear_solver_type
    raise ValueError(
ValueError: Unsupported set of arguments: The combination of penalty='l2' and loss='epsilon_insensitive' are not supported when dual=False, Parameters: penalty='l2', loss='epsilon_insensitive', dual=False
"""

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/169353810.py in <cell line: 0>()
     62 # Use real process‑based parallelism; limit workers to avoid memory pressure
     63 max_parallel = max(1, min(os.cpu_count(), 4))
---> 64 results = Parallel(n_jobs=max_parallel, backend="loky", prefer="processes")(
     65     delayed(eval_params)(p) for p in param_grid
     66 )

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   2070         next(output)
   2071 
-> 2072         return output if self.return_generator else list(output)
   2073 
   2074     def __repr__(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_outputs(self, iterator, pre_dispatch)
   1680 
   1681             with self._backend.retrieval_context():
-> 1682                 yield from self._retrieve()
   1683 
   1684         except GeneratorExit:

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _retrieve(self)
   1782             # worker traceback.
   1783             if self._aborting:
-> 1784                 self._raise_error_fast()
   1785                 break
   1786 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _raise_error_fast(self)
   1857         # called directly or if the generator is gc'ed.
   1858         if error_job is not None:
-> 1859             error_job.get_result(self.timeout)
   1860 
   1861     def _warn_exit_early(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in get_result(self, timeout)
    756             # callback thread, and is stored internally. It's just waiting to
    757             # be returned.
--> 758             return self._return_or_raise()
    759 
    760         # For other backends, the main thread needs to run the retrieval step.

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _return_or_raise(self)
    771         try:
    772             if self.status == TASK_ERROR:
--> 773                 raise self._result
    774             return self._result
    775         finally:

ValueError: Unsupported set of arguments: The combination of penalty='l2' and loss='epsilon_insensitive' are not supported when dual=False, Parameters: penalty='l2', loss='epsilon_insensitive', dual=False

## === cell 11
if best_params["kernel"] == "linear":
    train_pred = reg.predict(X_train)
else:
    K_full_train = rbf_kernel(
        X_train, X_train, gamma=resolve_gamma(best_params["gamma"], X_train)
    ).astype(np.float32)
    train_pred = reg.predict(K_full_train)
print("Training RMSE:", RMSE(y, train_pred))




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4260989775.py in <cell line: 0>()
----> 1 if best_params["kernel"] == "linear":
      2     train_pred = reg.predict(X_train)
      3 else:
      4     K_full_train = rbf_kernel(
      5         X_train, X_train, gamma=resolve_gamma(best_params["gamma"], X_train)

NameError: name 'best_params' is not defined

## === cell 12
if best_params["kernel"] == "linear":
    y_pred = reg.predict(X_test)
else:
    K_test = rbf_kernel(
        X_test, X_train, gamma=resolve_gamma(best_params["gamma"], X_train)
    ).astype(np.float32)
    y_pred = reg.predict(K_test)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/679407624.py in <cell line: 0>()
----> 1 if best_params["kernel"] == "linear":
      2     y_pred = reg.predict(X_test)
      3 else:
      4     K_test = rbf_kernel(
      5         X_test, X_train, gamma=resolve_gamma(best_params["gamma"], X_train)

NameError: name 'best_params' is not defined

## === cell 13
submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": y_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4096902657.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": y_pred})
      2 submission_path = "submission.csv"
      3 submission.to_csv(submission_path, index=False)
      4 print(f"Submission saved to {submission_path}")

NameError: name 'y_pred' is not defined
