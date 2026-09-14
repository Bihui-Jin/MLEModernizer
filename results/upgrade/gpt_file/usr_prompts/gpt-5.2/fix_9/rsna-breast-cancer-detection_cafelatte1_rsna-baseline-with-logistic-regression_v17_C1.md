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

catboost==1.2.8
geopandas==0.14.4
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1
xgboost==2.0.3

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

0.02

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.04746) has done: 'Your code doesn’t currently yield a Kaggle score likely because `CFG.debug=True` causes you to train only on a tiny subset of patients (very few positives), which makes the model output near-constant probabilities and produces an extremely low pF1. I keep the same model and training loop, but turn off debug mode so you train on the full provided `train.csv`, which should move the score upward toward your target. I also set `scaled_vars=["age"]` so `age` is consistently scaled fold-by-fold (still the same features and same model), which typically improves calibration slightly without changing the core logic. Finally, I keep submission generation identical but add a small safety check to ensure the OHE columns match between train/test.'

# 9. Code solution

## === cell 0
GLOBAL_SEED = 42

import os

os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)

import random as rnd
import gc
import pandas as pd
import numpy as np
from numpy import random as np_rnd
import pickle

import sklearn as skl
from sklearn import linear_model as lm
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn import metrics

import warnings

warnings.filterwarnings("ignore")




## === cell 1
def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    rnd.seed(seed)
    np_rnd.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def pickleIO(obj, src, op="w"):
    if op == "w":
        with open(src, op + "b") as f:
            pickle.dump(obj, f)
    elif op == "r":
        with open(src, op + "b") as f:
            tmp = pickle.load(f)
        return tmp
    else:
        print("unknown operation")
        return obj


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


def findIdx(data_x, col_names):
    return [int(i) for i, j in enumerate(data_x) if j in col_names]


def resolve_path(rel_path: str) -> str:
    candidates = [
        os.path.join("/kaggle/input/rsna-breast-cancer-detection", rel_path),
        os.path.join("/kaggle/data/rsna-breast-cancer-detection", rel_path),
        os.path.join("/kaggle/input", rel_path),
        os.path.join("/kaggle/data", rel_path),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]




## === cell 2
class CFG:
    debug = False
    common_vars = ["age", "implant"]
    target_list = ["cancer", "biopsy", "invasive", "BIRADS", "difficult_negative_case"]

    threshold = 0.5
    n_folds = 5




## === cell 3
df_full = pd.read_csv(resolve_path("train.csv"))
df_full.head()



## === cell 4
df_full = (
    df_full[["patient_id"] + CFG.common_vars + ["cancer"]].groupby("patient_id").max()
)

df_full["implant"] = df_full["implant"].fillna(0.0)
df_full = df_full.dropna().drop_duplicates().reset_index(drop=True)

df_full["age_discrete"] = pd.cut(
    df_full["age"], bins=[-np.inf] + list(range(20, 81, 5)) + [np.inf], right=False
).astype("object")

df_full = df_full.sample(frac=1, random_state=42).reset_index(drop=True)
df_full.head()



## === cell 5
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
ohe.fit(df_full[["age_discrete"]])

oh_cols = ["oh_" + str(j) for j in range(sum(len(i) for i in ohe.categories_))]

df_full[oh_cols] = ohe.transform(df_full[["age_discrete"]])
df_full = df_full.drop(["age_discrete"], axis=1)



## === cell 6
df_full["age"] /= 100



## === cell 7
df_full_x = df_full.drop("cancer", axis=1).astype("float32")
df_full_y = df_full["cancer"].astype("int32")
del df_full
gc.collect()

df_full_x.head()



## === cell 8
print(df_full_y.value_counts(True))



## === cell 9
scaled_vars = ["age"]



## === cell 10
valid_pred = np.zeros((len(df_full_x),), dtype=np.float64)
model_name_list = ["logistic"]
ft_list = {i: [] for i in model_name_list}
model_list = {i: [] for i in model_name_list}
score_list = {i: [] for i in model_name_list}
kfolds_spliter = skl.model_selection.StratifiedKFold(
    CFG.n_folds, shuffle=True, random_state=42
)

seed_everything()
for fold, (train_idx, valid_idx) in enumerate(
    kfolds_spliter.split(df_full_x, df_full_y)
):
    scaler = MinMaxScaler()

    df_train_x = df_full_x.iloc[train_idx].copy()
    df_train_x[scaled_vars] = (
        scaler.fit_transform(df_train_x[scaled_vars])
        if len(scaled_vars) > 0
        else df_train_x[scaled_vars]
    )
    df_train_y = df_full_y.iloc[train_idx]

    df_valid_x = df_full_x.iloc[valid_idx].copy()
    df_valid_x[scaled_vars] = (
        scaler.transform(df_valid_x[scaled_vars])
        if len(scaled_vars) > 0
        else df_valid_x[scaled_vars]
    )
    df_valid_y = df_full_y.iloc[valid_idx]

    model = lm.LogisticRegression(
        penalty="l2",
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(df_train_x, df_train_y)

    valid_pred[valid_idx] = model.predict_proba(df_valid_x)[:, 1]
    y_pred_class = np.array(
        [1 if i > CFG.threshold else 0 for i in valid_pred[valid_idx]], dtype="int32"
    )

    score_list["logistic"].append(
        {
            "logloss": metrics.log_loss(df_valid_y.values, valid_pred[valid_idx]),
            "roc_aud": metrics.roc_auc_score(df_valid_y.values, valid_pred[valid_idx]),
            "acc": metrics.accuracy_score(df_valid_y.values, y_pred_class),
            "f1": metrics.f1_score(df_valid_y.values, y_pred_class, average="binary"),
        }
    )
    model_list["logistic"].append(model)
    ft_list["logistic"].append({"scaler": scaler})



## === cell 11
df_score = pd.DataFrame(score_list["logistic"])
df_score.loc["average"] = df_score.iloc[: CFG.n_folds].mean(axis=0)
print(df_score)
df_score.to_csv("./df_score.csv", index=True)



## === cell 12
print("positive rate @0.5:", float((valid_pred > CFG.threshold).mean()))




## === cell 13
def f1_at_threshold(y_true, y_prob, thr):
    y_hat = (y_prob >= thr).astype(np.int32)
    return metrics.f1_score(y_true, y_hat, average="binary")


thr_grid = np.linspace(0.01, 0.99, 99)
f1_grid = np.array([f1_at_threshold(df_full_y.values, valid_pred, t) for t in thr_grid])
best_idx = int(np.argmax(f1_grid))
best_thr = float(thr_grid[best_idx])
print(
    "OOF best threshold (by F1 proxy):", best_thr, "OOF F1:", float(f1_grid[best_idx])
)

CFG.tuned_threshold = best_thr




## === cell 14
def pf1_score(y_true, y_prob, eps=1e-12):
    y_true = y_true.astype(np.float64)
    y_prob = np.clip(y_prob.astype(np.float64), 0.0, 1.0)

    pTP = np.sum(y_true * y_prob)
    pFP = np.sum((1.0 - y_true) * y_prob)
    pFN = np.sum(y_true * (1.0 - y_prob))

    pPrecision = pTP / (pTP + pFP + eps)
    pRecall = pTP / (pTP + pFN + eps)
    return float(2.0 * pPrecision * pRecall / (pPrecision + pRecall + eps))


def apply_sharpen(y_prob, thr, alpha):
    y_prob = np.clip(y_prob, 0.0, 1.0)
    return np.clip(
        np.where(y_prob >= thr, 1.0 - (1.0 - y_prob) ** alpha, y_prob**alpha), 0.0, 1.0
    )


CFG.tuned_alpha = 1.0
print("Chosen alpha (disabled sharpening):", float(CFG.tuned_alpha))
print("OOF pF1 (raw probs):", pf1_score(df_full_y.values, valid_pred))
print("positive rate @tuned_thr:", float((valid_pred >= CFG.tuned_threshold).mean()))



## === cell 15
df_test = pd.read_csv(resolve_path("test.csv"))
submission_idx = df_test["prediction_id"].copy()

df_test = df_test[CFG.common_vars].copy()
df_test["implant"] = df_test["implant"].fillna(0.0)

df_test["age_discrete"] = pd.cut(
    df_test["age"], bins=[-np.inf] + list(range(20, 81, 5)) + [np.inf], right=False
).astype("object")

df_test[oh_cols] = ohe.transform(df_test[["age_discrete"]])
df_test = df_test.drop(["age_discrete"], axis=1)

df_test["age"] = (df_test["age"] / 100).astype("float32")
df_test = df_test.astype("float32")

df_test = df_test.reindex(columns=df_full_x.columns, fill_value=0.0)

df_test.head()



## === cell 16
test_pred = np.zeros((len(df_test),), dtype="float64")
for model, ft in zip(model_list["logistic"], ft_list["logistic"]):
    scaler = ft["scaler"]
    df_test_x = df_test.copy()
    df_test_x[scaled_vars] = (
        scaler.transform(df_test_x[scaled_vars])
        if len(scaled_vars) > 0
        else df_test_x[scaled_vars]
    )
    test_pred += model.predict_proba(df_test_x)[:, 1] / len(model_list["logistic"])

test_pred = np.clip(test_pred, 0.0, 1.0)

t = CFG.tuned_threshold
alpha = float(getattr(CFG, "tuned_alpha", 1.0))
test_pred = apply_sharpen(test_pred, t, alpha)

tmp = pd.DataFrame(
    {"prediction_id": submission_idx.values, "cancer": test_pred.astype("float32")}
)
tmp = tmp.groupby("prediction_id", as_index=False)["cancer"].mean()



## === cell 17
submission = pd.read_csv(resolve_path("sample_submission.csv"))

submission = submission[["prediction_id"]].merge(tmp, on="prediction_id", how="left")
submission["cancer"] = submission["cancer"].fillna(0.0).astype("float32")

assert (
    submission.shape[0] == pd.read_csv(resolve_path("sample_submission.csv")).shape[0]
)
assert set(submission.columns) == {"prediction_id", "cancer"}

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("submission.csv saved with shape:", submission.shape)
print(
    "cancer min/max:",
    float(submission["cancer"].min()),
    float(submission["cancer"].max()),
)
print("n unique prediction_id in submission:", submission["prediction_id"].nunique())
print("file exists:", os.path.exists("submission.csv"))
