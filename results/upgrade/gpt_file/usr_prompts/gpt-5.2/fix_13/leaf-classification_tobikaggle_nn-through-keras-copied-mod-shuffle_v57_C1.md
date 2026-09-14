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
Use binary leaf images and extracted features to identify the species of plant.

## Metric
Multi-class log loss. 

The submitted probabilities for a given device are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum), but they need to be in the range of [0, 1]. In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).

## Submission Format
You must submit a csv file with the image id, all candidate species names, and a probability for each species. The order of the rows does not matter. The file must have a header and should look like the following:

id,Acer_Capillipes,Acer_Circinatum,Acer_Mono,...
2,0.1,0.5,0,0.2,...
5,0,0.3,0,0.4,...
6,0,0,0,0.7,...
etc.

## Dataset
The dataset consists of images of leaf specimens which have been converted to binary black leaves against white backgrounds. 

Three sets of features are also provided per image: a shape contiguous descriptor, an interior texture histogram, and a ﬁne-scale margin histogram. 

For each feature, a 64-attribute vector is given per leaf sample.

### File descriptions
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format
- **images/** - the image files (each image is named with its corresponding id)

### Data fields
- **id** - an anonymous id unique to an image
- **margin_1, margin_2, margin_3, ..., margin_64** - each of the 64 attribute vectors for the margin feature
- **shape_1, shape_2, shape_3, ..., shape_64** - each of the 64 attribute vectors for the shape feature
- **texture_1, texture_2, texture_3, ..., texture_64** - each of the 64 attribute vectors for the texture feature

# 2. Python version

3.6

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (70 lines)
            images.zip (22.0 MB)
            sample_submission.csv (100 lines)
            sample_submission.csv.zip (2.3 kB)
            test.csv (100 lines)
            test.csv.zip (39.3 kB)
            train.csv (892 lines)
            train.csv.zip (357.1 kB)
            images/
                42.jpg (32.6 kB)
                168.jpg (16.5 kB)
                ... and 988 other files
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
        input/
            description.md (70 lines)
            images.zip (22.0 MB)
            sample_submission.csv (100 lines)
            sample_submission.csv.zip (2.3 kB)
            test.csv (100 lines)
            test.csv.zip (39.3 kB)
            train.csv (892 lines)
            train.csv.zip (357.1 kB)
            images/
                42.jpg (32.6 kB)
                168.jpg (16.5 kB)
                ... and 988 other files
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
        working/
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
```

-> data/leaf-classification/sample_submission.csv has 99 rows and 100 columns.
The columns are: id, Acer_Capillipes, Acer_Circinatum, Acer_Mono, Acer_Opalus, Acer_Palmatum, Acer_Pictum, Acer_Platanoids, Acer_Rubrum, Acer_Rufinerve, Acer_Saccharinum, Alnus_Cordata, Alnus_Maximowiczii, Alnus_Rubra, Alnus_Sieboldiana... and 85 more columns

-> data/leaf-classification/test.csv has 99 rows and 193 columns.
The columns are: id, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13, margin14... and 178 more columns

-> data/leaf-classification/train.csv has 891 rows and 194 columns.
The columns are: id, species, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns

-> data/sample_submission.csv has 99 rows and 100 columns.
The columns are: id, Acer_Capillipes, Acer_Circinatum, Acer_Mono, Acer_Opalus, Acer_Palmatum, Acer_Pictum, Acer_Platanoids, Acer_Rubrum, Acer_Rufinerve, Acer_Saccharinum, Alnus_Cordata, Alnus_Maximowiczii, Alnus_Rubra, Alnus_Sieboldiana... and 85 more columns

-> data/test.csv has 99 rows and 193 columns.
The columns are: id, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13, margin14... and 178 more columns

-> data/train.csv has 891 rows and 194 columns.
The columns are: id, species, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.0092

# 6. Current score

0.0461

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.06073) has done: 'I fix the environment/import crash by switching from `tf_keras` to the already-installed `tensorflow.keras`, which avoids the protobuf `MessageFactory.GetPrototype` error and keeps the same Keras model/fit logic. Then I resolve the split failure by increasing the validation fraction so the validation set has at least one example per class (StratifiedShuffleSplit requires `n_val >= n_classes`). Finally, I ensure the submission columns *exactly* match `sample_submission.csv` (both names and order) and write a `.csv` file so Kaggle accepts it.'
- What this solution (achieved 0.02299) has done: 'I fix the TensorFlow/Keras import crash that currently prevents the notebook from running by switching to the already-installed `tf_keras` package (which is compatible with this environment) while keeping the exact same model/training logic. I also make the input path resolution robust to your provided `/kaggle/data/...` layout without changing data usage. Finally, I keep the submission formatting exactly aligned to `sample_submission.csv` (names + order) and ensure the output is a `.csv` written successfully. These changes are execution/stability fixes and should not materially change the modeling semantics beyond negligible floating-point differences.'
- What this solution (achieved 0.02936) has done: 'I fix the runtime crash by changing the Keras import stack to use the `keras` package (with `tf_keras` as a fallback) which avoids the protobuf `MessageFactory.GetPrototype` error while keeping the same Sequential/Dense/Dropout model and training loop. Then I adjust only the train/validation splitting so it becomes robust to the “at least one sample per class” requirement (this is score-improving because it restores a proper stratified validation split instead of failing or forcing an overly small/biased split). Finally, I keep prediction and submission formatting identical (same columns/order as `sample_submission.csv`) and ensure a valid `.csv` is always written.'
- What this solution (achieved 0.14522) has done: 'I fix the crash in the Keras import by avoiding the incompatible `keras`/`tf_keras` protobuf stack and using scikit-learn’s `MLPClassifier` instead, which preserves the same core “MLP on standardized tabular features” logic and outputs proper class probabilities for log-loss. I keep the same preprocessing (LabelEncoder + StandardScaler) and the same stratified train/validation split logic so evaluation semantics remain consistent. I also ensure predictions are clipped to `[1e-15, 1-1e-15]` and submission columns exactly match `sample_submission.csv` (names + order), writing a `.csv` file. This change should both unblock execution and typically improve log-loss vs. the broken run by producing a stable, well-calibrated probability model on this dataset.'
- What this solution (achieved 0.05602) has done: 'Your current score (0.14522, lower-is-better) is far from the target (0.0092), so we should legitimately improve the probability model while keeping the same core “standardize tabular features → MLP classifier → predict_proba → submission aligned to sample columns” pipeline. The biggest issue holding score back is that `MLPClassifier` is using `early_stopping=True`, which changes the training objective by carving out an internal validation split and stopping early—this often hurts final log-loss here; we disable early stopping and let it fully converge within your existing `max_iter`. Next, we add a minimal amount of L2 regularization (`alpha`) and switch to a slightly more stable constant learning rate schedule (still Adam), which typically improves multiclass log-loss without changing the overall approach. Finally, we train on the full training data after checking validation log-loss (same model/logic), then generate the submission with the exact same formatting.'
- What this solution (achieved 0.05593) has done: 'We keep your exact “standardize tabular features → scikit-learn MLPClassifier → predict_proba → align to sample_submission columns” pipeline, but make two minimal changes that typically improve multiclass log-loss substantially on this dataset: (1) fit the scaler only on the training fold (not the full data) when doing validation to avoid leakage, and (2) increase MLP stability/generalization by adding a tiny amount more L2 regularization and enabling Nesterov momentum with the same architecture/solver (this often improves probability calibration). We also ensure deterministic behavior by setting `random_state` everywhere relevant and by using `warm_start=False` explicitly. Finally, we keep the same submission formatting and clipping rules, writing a valid `.csv`.'
- What this solution (achieved 0.0481) has done: 'We’re far from the target (0.05593 vs 0.0092, lower-is-better), so we should make the smallest changes that typically improve multiclass log-loss without changing your core pipeline (StandardScaler → MLPClassifier → predict_proba → sample_submission-aligned CSV). The most likely issue is probability miscalibration/overconfidence from a single MLP; we can reduce this by (a) using a slightly stronger L2 (`alpha`) and (b) averaging probabilities across a few different random initializations (same architecture/training loop, just repeated fits) which is a standard, minimal-risk way to improve log-loss. I keep your split/validation logic intact, use the same hyperparameters otherwise, and still refit on full data before test prediction, now as a small ensemble whose averaged probabilities are usually better calibrated. The submission format, column order, and clipping remain unchanged.'
- What this solution (achieved 0.05093) has done: 'We keep your exact pipeline (StandardScaler → MLPClassifier → averaged predict_proba → sample_submission-aligned CSV) and make the smallest changes that typically reduce multiclass log-loss on this dataset. The main score lever without changing core logic is probability calibration: we apply a tiny amount of label-smoothing-style shrinkage (mixing predictions with a uniform distribution) to reduce overconfidence, which often improves log-loss. We also weight ensemble members by their validation log-loss (better models contribute slightly more) rather than a flat average, using the same trained models you already fit. Finally, we keep clipping and submission formatting identical so Kaggle scoring semantics are unchanged.'
- What this solution (achieved 0.0478) has done: 'We’re still far above the target (0.05093 vs 0.0092, lower-is-better), so we should make a minimal, legitimate improvement that typically reduces multiclass log-loss without changing your core pipeline. The biggest current mismatch is that you compute ensemble weights on a validation split, but then reuse those weights after refitting on the full dataset—this can misweight members and worsen log-loss; instead, we recompute ensemble weights via out-of-fold (OOF) predictions using the same seeds and hyperparameters, then train final models on full data and use those OOF-derived weights. We also tune only the post-hoc smoothing strength on the OOF predictions (small grid) to pick the best calibration for log-loss, and then apply that same smoothing to test predictions. Everything else (StandardScaler → MLPClassifier with same architecture/solver/training, probability averaging, clipping, submission formatting) stays the same.'
- What this solution (achieved 0.0461) has done: 'We keep your exact pipeline (StandardScaler → same MLPClassifier hyperparameters per seed → OOF-derived weights → smoothing → submission aligned to sample columns), but make two minimal changes that usually improve multiclass log-loss: (1) replace the inverse-logloss weighting with a softmax-over-negative-logloss weighting (less brittle and less likely to over-weight a slightly lucky seed), and (2) select the smoothing strength using the OOF predictions with the same scoring semantics as Kaggle (row-normalize before log-loss, since Kaggle rescales rows). These are post-processing/calibration changes only; the model architecture, training loops, and data usage remain the same. The submission formatting and clipping remain unchanged, and the code still writes a valid `.csv`.'
- What this solution (achieved 0.0461) has done: 'We’re still above the target (0.0461 vs 0.0092, lower-is-better), so the smallest legitimate lever is better probability calibration without changing the core “StandardScaler → MLPClassifier per seed → weighted ensemble → smoothing → sample_submission-aligned CSV” pipeline. I keep your model/training exactly the same, but change the post-processing to (1) apply Kaggle’s row-normalization to predictions before smoothing (since Kaggle rescales rows when scoring) and (2) re-select the smoothing strength on OOF predictions using the exact same post-processing order you use at test time. This is a minimal, metric-aligned correction that often improves multiclass log-loss while preserving your overall approach and producing the same valid submission format.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(12345)



## === cell 1
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss



## === cell 2
INPUT_DIR_CANDIDATES = [
    "/kaggle/input/leaf-classification",
    "/kaggle/input",
    "../input",  # legacy notebooks
    "/kaggle/data",  # provided path in this environment description
    "/kaggle/data/leaf-classification",
]
INPUT_DIR = None
for d in INPUT_DIR_CANDIDATES:
    if os.path.exists(d):
        if os.path.exists(os.path.join(d, "train.csv")) or os.path.exists(
            os.path.join(d, "leaf-classification", "train.csv")
        ):
            INPUT_DIR = d
            break

if INPUT_DIR is None:
    raise FileNotFoundError("Could not find Kaggle input directory in known locations.")


def resolve_path(filename):
    p1 = os.path.join(INPUT_DIR, filename)
    p2 = os.path.join(INPUT_DIR, "leaf-classification", filename)
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(
        f"Could not find {filename} under {INPUT_DIR} or {os.path.join(INPUT_DIR, 'leaf-classification')}"
    )


train_path = resolve_path("train.csv")
test_path = resolve_path("test.csv")
sample_path = resolve_path("sample_submission.csv")

train_df = pd.read_csv(train_path)
parent_data = train_df.copy()  # keep original copy as in the notebook
train_id = train_df.pop("id")

print("train_df:", train_df.shape)



## === cell 3
y = train_df.pop("species")
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("y_enc:", y_enc.shape, "n_classes:", len(le.classes_))



## === cell 4
n_classes = len(le.classes_)
n_samples = train_df.shape[0]
test_size = 0.2
n_val = int(np.ceil(n_samples * test_size))
if n_val < n_classes:
    test_size = min(0.5, (n_classes / n_samples) + 0.01)  # minimal bump, keep semantics
    n_val = int(np.ceil(n_samples * test_size))

sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=12345)
train_index, val_index = next(sss.split(train_df.values, y_enc))

x_train_raw = train_df.values[train_index]
x_val_raw = train_df.values[val_index]
y_train, y_val = y_enc[train_index], y_enc[val_index]

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train_raw)
x_val = scaler.transform(x_val_raw)

print(
    "Stratified split test_size:",
    test_size,
    "n_val:",
    x_val.shape[0],
    "n_classes:",
    n_classes,
)
print("x_train dim: ", x_train.shape)
print("x_val dim:   ", x_val.shape)



## === cell 5
ALPHA = 1e-4  # keep same regularization as current best run
ENSEMBLE_SEEDS = [12345, 12346, 12347]

val_proba_list = []
val_lls = []

for seed in ENSEMBLE_SEEDS:
    mlp = MLPClassifier(
        hidden_layer_sizes=(600, 300),
        activation="relu",
        solver="adam",
        alpha=ALPHA,
        batch_size=192,
        learning_rate="constant",
        learning_rate_init=1e-3,
        max_iter=1800,
        early_stopping=False,
        random_state=seed,
        verbose=False,
        warm_start=False,
    )
    mlp.fit(x_train, y_train)
    vp = mlp.predict_proba(x_val)
    vp = np.clip(vp, 1e-15, 1 - 1e-15)
    ll = float(log_loss(y_val, vp, labels=np.arange(n_classes)))
    val_lls.append(ll)
    val_proba_list.append(vp)

val_lls_arr = np.asarray(val_lls, dtype=np.float64)
tau = 0.01  # small temperature -> still prefers better seeds but avoids extreme weights
w_raw = np.exp(-(val_lls_arr - val_lls_arr.min()) / max(tau, 1e-12))
weights_holdout = w_raw / w_raw.sum()

val_proba_ens = np.zeros((x_val.shape[0], n_classes), dtype=np.float64)
for w, vp in zip(weights_holdout, val_proba_list):
    val_proba_ens += w * vp

val_proba_ens = np.clip(val_proba_ens, 1e-15, 1 - 1e-15)

SMOOTH_EPS = 0.003
val_proba_ens_sm = (1.0 - SMOOTH_EPS) * val_proba_ens + SMOOTH_EPS * (1.0 / n_classes)
val_proba_ens_sm = np.clip(val_proba_ens_sm, 1e-15, 1 - 1e-15)

print("val_log_loss (per seed):", [float(x) for x in val_lls])
print("ensemble weights (holdout-derived):", [float(w) for w in weights_holdout])
print(
    "val_log_loss (weighted+smoothed ensemble, holdout-derived):",
    float(log_loss(y_val, val_proba_ens_sm, labels=np.arange(n_classes))),
)
print(
    "val_accuracy (ensemble argmax, holdout-derived):",
    float((val_proba_ens_sm.argmax(axis=1) == y_val).mean()),
)



## === cell 6
train_df_full = pd.read_csv(train_path)
y_full = train_df_full.pop("species")
train_id_full = train_df_full.pop("id")
y_full_enc = le.transform(y_full)  # same label mapping

X_full_raw = train_df_full.values
n_full = X_full_raw.shape[0]

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=12345)

oof_pred_per_seed = {
    seed: np.zeros((n_full, n_classes), dtype=np.float64) for seed in ENSEMBLE_SEEDS
}

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full_raw, y_full_enc), start=1):
    X_tr_raw, X_va_raw = X_full_raw[tr_idx], X_full_raw[va_idx]
    y_tr, y_va = y_full_enc[tr_idx], y_full_enc[va_idx]

    scaler_fold = StandardScaler()
    X_tr = scaler_fold.fit_transform(X_tr_raw)
    X_va = scaler_fold.transform(X_va_raw)

    for seed in ENSEMBLE_SEEDS:
        m = MLPClassifier(
            hidden_layer_sizes=(600, 300),
            activation="relu",
            solver="adam",
            alpha=ALPHA,
            batch_size=192,
            learning_rate="constant",
            learning_rate_init=1e-3,
            max_iter=1800,
            early_stopping=False,
            random_state=seed,
            verbose=False,
            warm_start=False,
        )
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_va)
        oof_pred_per_seed[seed][va_idx] = np.clip(p, 1e-15, 1 - 1e-15)

oof_lls = []
for seed in ENSEMBLE_SEEDS:
    ll = float(
        log_loss(y_full_enc, oof_pred_per_seed[seed], labels=np.arange(n_classes))
    )
    oof_lls.append(ll)

oof_lls_arr = np.asarray(oof_lls, dtype=np.float64)
tau = 0.01
w_raw = np.exp(-(oof_lls_arr - oof_lls_arr.min()) / max(tau, 1e-12))
weights = w_raw / w_raw.sum()

oof_ens = np.zeros((n_full, n_classes), dtype=np.float64)
for w, seed in zip(weights, ENSEMBLE_SEEDS):
    oof_ens += w * oof_pred_per_seed[seed]
oof_ens = np.clip(oof_ens, 1e-15, 1 - 1e-15)


def kaggle_row_normalize(p):
    rs = p.sum(axis=1, keepdims=True)
    rs = np.maximum(rs, 1e-15)
    return p / rs


def kaggle_style_logloss_postproc(y_true, p, eps):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    p = kaggle_row_normalize(p)
    p = (1.0 - eps) * p + eps * (1.0 / n_classes)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    p = kaggle_row_normalize(p)
    return float(log_loss(y_true, p, labels=np.arange(n_classes)))


smooth_grid = [0.0, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02]
best_eps = None
best_ll = None
for eps in smooth_grid:
    ll = kaggle_style_logloss_postproc(y_full_enc, oof_ens, eps)
    if best_ll is None or ll < best_ll:
        best_ll = ll
        best_eps = eps

SMOOTH_EPS = float(best_eps)

print("OOF log_loss (per seed):", [float(x) for x in oof_lls])
print("Ensemble weights (OOF-derived):", [float(w) for w in weights])
print("Best SMOOTH_EPS from grid (Kaggle-style, postproc-matched):", SMOOTH_EPS)
print(
    "OOF log_loss (weighted ensemble, eps=0, postproc-matched):",
    kaggle_style_logloss_postproc(y_full_enc, oof_ens, 0.0),
)
print(
    "OOF log_loss (weighted ensemble, best eps, postproc-matched):",
    kaggle_style_logloss_postproc(y_full_enc, oof_ens, SMOOTH_EPS),
)



## === cell 7
scaler_full = StandardScaler()
X_full = scaler_full.fit_transform(X_full_raw)

mlp_full_models = []
for seed in ENSEMBLE_SEEDS:
    m = MLPClassifier(
        hidden_layer_sizes=(600, 300),
        activation="relu",
        solver="adam",
        alpha=ALPHA,
        batch_size=192,
        learning_rate="constant",
        learning_rate_init=1e-3,
        max_iter=1800,
        early_stopping=False,
        random_state=seed,
        verbose=False,
        warm_start=False,
    )
    m.fit(X_full, y_full_enc)
    mlp_full_models.append(m)

print(
    "Refit ensemble on full data. n_iter_:",
    [int(getattr(m, "n_iter_", -1)) for m in mlp_full_models],
)



## === cell 8
test_df = pd.read_csv(test_path)
test_id = test_df.pop("id")
X_test = scaler_full.transform(test_df.values)

y_pred = np.zeros((X_test.shape[0], n_classes), dtype=np.float64)
for w, m in zip(weights, mlp_full_models):
    yp = m.predict_proba(X_test)
    yp = np.clip(yp, 1e-15, 1 - 1e-15)
    y_pred += w * yp

y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
y_pred = kaggle_row_normalize(y_pred)
y_pred = (1.0 - SMOOTH_EPS) * y_pred + SMOOTH_EPS * (1.0 / n_classes)
y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
y_pred = kaggle_row_normalize(y_pred)

print("y_pred:", y_pred.shape)
print("Using SMOOTH_EPS:", SMOOTH_EPS)



## === cell 9
sample_sub = pd.read_csv(sample_path)
class_cols = [c for c in sample_sub.columns if c != "id"]

model_class_names = list(le.classes_)
name_to_col_idx = {name: i for i, name in enumerate(model_class_names)}

sub = pd.DataFrame({"id": test_id.values})
for cname in class_cols:
    if cname not in name_to_col_idx:
        raise ValueError(
            f"Class {cname} exists in sample_submission but not in LabelEncoder classes."
        )
    sub[cname] = y_pred[:, name_to_col_idx[cname]]

sub = sub[["id"] + class_cols]
sub[class_cols] = sub[class_cols].astype(np.float64).clip(1e-15, 1 - 1e-15)

print(sub.head())
print("Submission columns match sample:", list(sub.columns) == list(sample_sub.columns))



## === cell 10
out_path = "submission_nn_kernel.csv"
sub.to_csv(out_path, index=False)

print(f"Wrote submission to: {out_path}")
print("Submission shape:", sub.shape)
print("Columns match sample:", list(sub.columns) == list(sample_sub.columns))
print(
    "Min/Max prob:",
    float(sub[class_cols].min().min()),
    float(sub[class_cols].max().max()),
)
