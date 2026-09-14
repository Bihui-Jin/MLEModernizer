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

3.5

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

0.01543

# 6. Current score

0.03276

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0389) has done: 'I update deprecated scikit-learn imports and make the Keras code compatible with your installed Keras version by replacing removed arguments/APIs (`init`, `nb_epoch`, `predict_proba`, and old `to_categorical` import). I also fix data leakage/shape issues by fitting one `StandardScaler` on train features and reusing it for test, and ensure the class/probability column order matches the sample submission (critical for a valid multi-class logloss submission). Finally, I write the submission with an explicit `id` column and the exact required header so Kaggle accepts it.'
- What this solution (achieved 0.03589) has done: 'The crash happens before any training due to an incompatibility between `keras` 3.x importing `protobuf` (triggering `MessageFactory.GetPrototype`), so the minimal fix is to stop using `keras` and switch the exact same API calls to `tf_keras` (which is installed and stable in this environment). I keep the model, loss, optimizer, epochs, and preprocessing identical, only changing imports and adding a small, score-positive calibration fix by explicitly normalizing each prediction row to sum to 1 (the metric rescales anyway, but this prevents numerical drift and typically improves logloss). I also make the label→column alignment robust by mapping predictions into the exact `sample_submission.csv` column order using the `LabelEncoder` classes, so no class gets shifted/misaligned. Finally, the script always write a valid `.csv` submission to `/kaggle/working/`.'
- What this solution (achieved 0.02309) has done: 'The failure happens immediately on importing `tf_keras`, which in this environment is pulling in an incompatible protobuf API (`MessageFactory.GetPrototype`). The minimal fix is to stop importing any TensorFlow/Keras stack and instead keep the same core “MLP on scaled features” logic using scikit-learn’s `MLPClassifier` (same training approach: feed-forward neural net, cross-entropy via `predict_proba`). I also keep your critical correctness steps: single `StandardScaler` fit on train and reused for test, and strict column alignment to `sample_submission.csv` to avoid class-order bugs that ruin logloss. Finally, I ensure probabilities are finite, clipped to [0,1], row-normalized, and a valid `.csv` is always written to `/kaggle/working/`.'
- What this solution (achieved 0.03465) has done: 'We keep your exact “scaled tabular features → MLPClassifier → predict_proba → align to sample_submission” core pipeline, and only make small, score-relevant tweaks that typically reduce multiclass logloss. Specifically: (1) add a tiny amount of L2 regularization (`alpha`) to reduce overconfident probabilities (overconfidence hurts logloss), (2) average predictions across a few fixed random seeds (simple ensembling reduces variance without changing the model family), and (3) apply very light probability smoothing (mix with uniform) after ensembling to further reduce extreme probabilities. These are minimal changes that preserve evaluation semantics and should move logloss down toward your 0.01543 target while staying stable and within runtime.'
- What this solution (achieved 0.0405) has done: 'You’re already generating a valid submission and the core pipeline is sound, but your logloss gap suggests the model is still a bit overconfident/variable. To move the score down toward 0.01543 with minimal, score-relevant changes, I (1) increase the ensemble size slightly (more seeds) to reduce variance, (2) apply a tiny amount of temperature scaling (soften probabilities in a metric-aligned way) and (3) tune the smoothing `eps` downward so we don’t over-flatten. These keep the same “StandardScaler → MLPClassifier → predict_proba → align to sample_submission” logic and stay within runtime.'
- What this solution (achieved 0.05189) has done: 'We keep your exact “StandardScaler → MLPClassifier ensemble → predict_proba → align to sample submission” pipeline, and only make small calibration changes that are directly logloss-relevant. Since your current score (0.0405) is worse than the target (0.01543), the safest move is to further reduce overconfidence by slightly increasing temperature (softer probabilities) and slightly increasing uniform mixing (more smoothing), both of which typically lower multiclass logloss without changing the model family or training loop. I also ensure we apply temperature/smoothing only after ensembling and perform one final strict row-normalization and clipping for metric stability. No architecture, features, or training approach is changed; runtime stays well under the limit.'
- What this solution (achieved 0.03435) has done: 'Your current logloss (0.05189) is worse than the target (0.01543), so we should cautiously improve calibration and reduce overconfidence without changing the core pipeline (StandardScaler → MLPClassifier ensemble → predict_proba). The most score-relevant minimal fix is to replace the ad-hoc temperature/eps values with a principled, data-driven calibration chosen via out-of-fold (OOF) predictions from the same MLP ensemble (no new model family, no early stopping, same training). We tune only two post-processing knobs—temperature and uniform-mix epsilon—by minimizing multiclass logloss on OOF, then retrain on full data and apply the tuned calibration to test predictions. This typically yields a meaningful logloss drop while preserving architecture/training semantics and staying within runtime.'
- What this solution (achieved 0.03259) has done: 'We keep your exact pipeline (StandardScaler → 5-seed MLPClassifier ensemble → predict_proba → row-normalize → temperature+uniform smoothing) and only make calibration more metric-aligned by tuning temperature/eps on a finer grid using the already-computed OOF predictions. This is a minimal change that can reduce multiclass logloss meaningfully without altering model architecture or training loops. We also ensure the final submission rows are strictly normalized (after column reindexing) so Kaggle’s rescaling doesn’t interact with zeros from missing columns and slightly distort probabilities. Everything else (data paths, features, model params, ensemble, folds) stays the same and still writes a valid CSV.'
- What this solution (achieved 0.03259) has done: 'We keep your exact StandardScaler → 5-seed MLPClassifier ensemble pipeline and only adjust the *post-processing calibration search* to better match multiclass logloss. Specifically, we widen and slightly densify the temperature/epsilon grid (your current best may sit outside the narrow range), and add one additional logit-domain “power” option via temperature extremes without changing the model or training loop. We also make the OOF calibration search more robust by optimizing on the same row-normalized probabilities but evaluating after final clipping/renormalization exactly as used for test-time, so the tuned parameters transfer more faithfully. These are minimal, score-relevant tweaks intended to reduce your logloss toward 0.01543 without altering core modeling.'
- What this solution (achieved 0.03309) has done: 'We keep your exact pipeline (StandardScaler → 5-seed MLPClassifier ensemble → OOF-based calibration → submission alignment) and only make calibration more expressive in a metric-aligned way. Specifically, we extend the post-processing search to include a very small 1-parameter “class prior” mix (blend with the empirical class distribution), which often improves multiclass logloss compared to uniform smoothing while preserving semantics. We tune (temperature, eps, prior_mix) on the already-computed OOF predictions (no model/training changes), then apply the best calibration to test predictions. Finally, we keep the strict sample-submission column alignment and do one last clip+row-normalize for numerical stability.'
- What this solution (achieved 0.03392) has done: 'I keep your exact “StandardScaler → 5-seed MLPClassifier ensemble → OOF-based calibration → sample_submission column alignment” pipeline, and make only score-relevant calibration changes. The main adjustment is to tune calibration on *logit-space* (which matches how temperature scaling is usually defined) and to expand the search to include a tiny **Dirichlet prior (additive) smoothing** strength, which often improves multiclass logloss vs only mixing with uniform/prior. I also apply the exact same final post-processing (clip + renormalize) during OOF scoring as at test-time to reduce train/submit mismatch. These are minimal, low-risk changes that typically push logloss down toward your 0.01543 target without changing the model/training approach.'
- What this solution (achieved 0.03309) has done: 'I keep your exact pipeline (StandardScaler → 5-seed MLPClassifier ensemble → OOF-based calibration → sample_submission column alignment) and only make small, score-relevant calibration improvements. The main fix is to correct the temperature scaling math: your current `apply_temperature_and_smoothing` uses a binary logit, which is not the right transform for multiclass probabilities and can worsen logloss; we switch to standard multiclass temperature scaling in log-prob space (equivalent to softmax(log(p)/T)). Then, to better match Kaggle’s scoring (row-rescaling), we tune calibration using a row-rescaled (normalized) + clipped objective exactly like submission-time, to reduce train/submit mismatch. Finally, we slightly densify the calibration grid around 1.0–1.6 (common sweet spot) while keeping runtime safe.'
- What this solution (achieved 0.03276) has done: 'I keep your exact pipeline (StandardScaler → 5-seed MLPClassifier ensemble → OOF-based calibration → sample_submission column alignment), but make two minimal, score-relevant fixes to reduce multiclass logloss toward your 0.01543 target. First, I remove the double-smoothing interaction by tuning calibration in a strictly “one family” way: temperature scaling in log-prob space plus either (a) class-prior mixing or (b) additive Dirichlet-style smoothing—whichever scores best—rather than stacking multiple smoothing types at once (which often over-flattens and hurts). Second, I apply the exact same final “clip + row-normalize” transform during OOF scoring and test-time (including after reindexing to submission columns) to eliminate small train/submit mismatches that can degrade logloss. Everything else (features, model, seeds, folds, max_iter, etc.) remains unchanged and it still writes a valid CSV to `/kaggle/working/`.'
- What this solution (achieved 0.03276) has done: 'I keep your exact StandardScaler → 5-seed MLPClassifier ensemble → OOF-calibration → submission alignment pipeline, and only make two minimal, score-relevant fixes that typically reduce multiclass logloss. First, I fit the `StandardScaler` **inside each CV fold** (and separately on full train for test-time) to remove subtle fold leakage that can inflate OOF calibration quality and hurt real Kaggle performance. Second, I make the OOF-calibration objective match Kaggle’s scoring more closely by evaluating logloss on **Kaggle-rescaled** probabilities (row-normalize + clip), which better transfers to the leaderboard. Everything else (model family/params, seeds, folds, calibration families, output CSV schema/path) stays the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pylab import rcParams

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold

np.random.seed(1337)
rcParams["figure.figsize"] = (10, 10)



## === cell 1
BASE = "/kaggle/input/leaf-classification"
if not os.path.exists(os.path.join(BASE, "train.csv")):
    BASE = "/kaggle/input"

train_path = os.path.join(BASE, "train.csv")
test_path = os.path.join(BASE, "test.csv")
sample_path = os.path.join(BASE, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

print("train:", train_df.shape, "test:", test_df.shape, "sample:", sample_sub.shape)



## === cell 2
parent_data = train_df.copy()

train_id = train_df.pop("id")
y_raw = train_df.pop("species")

le = LabelEncoder()
y = le.fit_transform(y_raw.values)

print("X columns:", train_df.shape[1], "num classes:", len(le.classes_))



## === cell 3
X_raw = train_df.values

input_dim = X_raw.shape[1]
n_classes = len(le.classes_)
print("input_dim:", input_dim, "n_classes:", n_classes)




## === cell 4
def row_normalize(p):
    p = np.asarray(p, dtype=np.float64)
    p[~np.isfinite(p)] = 0.0
    p = np.clip(p, 0.0, 1.0)
    rs = p.sum(axis=1, keepdims=True)
    rs[rs == 0.0] = 1.0
    return p / rs


def apply_temperature(p, temperature=1.0):
    p = row_normalize(p)
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    logp = np.log(p) / float(temperature)
    logp = logp - logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return row_normalize(p)


def apply_uniform_mix(p, eps=0.0):
    if eps <= 0.0:
        return row_normalize(p)
    p = row_normalize(p)
    k = p.shape[1]
    p = (1.0 - float(eps)) * p + float(eps) * (1.0 / k)
    return row_normalize(np.clip(p, 0.0, 1.0))


def apply_prior_mix(p, prior, prior_mix=0.0):
    if prior is None or prior_mix <= 0.0:
        return row_normalize(p)
    p = row_normalize(p)
    prior = np.asarray(prior, dtype=np.float64).reshape(1, -1)
    prior = row_normalize(np.tile(prior, (p.shape[0], 1)))
    p = (1.0 - float(prior_mix)) * p + float(prior_mix) * prior
    return row_normalize(np.clip(p, 0.0, 1.0))


def apply_additive_smoothing(p, alpha_add=0.0):
    if alpha_add <= 0.0:
        return row_normalize(p)
    p = row_normalize(p)
    p = p + float(alpha_add)
    return row_normalize(np.clip(p, 0.0, 1.0))


def multiclass_logloss(y_true_int, p_pred):
    p_pred = np.clip(p_pred, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(np.log(p_pred[np.arange(len(y_true_int)), y_true_int])))


def final_submit_transform(p):
    p = row_normalize(np.asarray(p, dtype=np.float64))
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    p = row_normalize(p)
    return p


def apply_calibration_family(
    p,
    temperature=1.0,
    family="uniform",
    eps=0.0,
    prior=None,
    prior_mix=0.0,
    alpha_add=0.0,
):
    p = apply_temperature(p, temperature=temperature)

    if family == "none":
        p = p
    elif family == "uniform":
        p = apply_uniform_mix(p, eps=eps)
    elif family == "prior":
        p = apply_prior_mix(p, prior=prior, prior_mix=prior_mix)
    elif family == "additive":
        p = apply_additive_smoothing(p, alpha_add=alpha_add)
    else:
        raise ValueError("Unknown family: %r" % family)

    return final_submit_transform(p)


seeds = [1337, 1338, 1339, 1340, 1341]

mlp_params = dict(
    hidden_layer_sizes=(1024, 512),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=192,
    learning_rate="constant",
    learning_rate_init=0.001,
    max_iter=73,
    shuffle=True,
    early_stopping=False,
    n_iter_no_change=200,
    validation_fraction=0.1,
    verbose=False,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
oof_pred = np.zeros((X_raw.shape[0], n_classes), dtype=np.float64)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_raw, y), 1):
    X_tr_raw, y_tr = X_raw[tr_idx], y[tr_idx]
    X_va_raw, y_va = X_raw[va_idx], y[va_idx]

    fold_scaler = StandardScaler()
    X_tr = fold_scaler.fit_transform(X_tr_raw)
    X_va = fold_scaler.transform(X_va_raw)

    fold_pred = None
    for rs in seeds:
        mlp = MLPClassifier(random_state=rs, **mlp_params)
        mlp.fit(X_tr, y_tr)
        p = mlp.predict_proba(X_va)
        p = np.asarray(p, dtype=np.float64)
        fold_pred = p if fold_pred is None else (fold_pred + p)

    fold_pred /= float(len(seeds))
    fold_pred = final_submit_transform(fold_pred)

    oof_pred[va_idx] = fold_pred
    print(
        "Fold",
        fold,
        "raw OOF logloss:",
        multiclass_logloss(y_va, final_submit_transform(fold_pred)),
    )

base_oof_loss = multiclass_logloss(y, final_submit_transform(oof_pred))
print("Base OOF logloss (no calibration):", base_oof_loss)

prior = np.bincount(y, minlength=n_classes).astype(np.float64)
prior = prior / prior.sum()

temp_grid = np.unique(
    np.round(
        np.concatenate(
            [
                np.arange(0.85, 1.21, 0.02),
                np.arange(1.20, 1.61, 0.02),
                np.arange(1.60, 2.01, 0.03),
            ]
        ),
        2,
    )
)

eps_grid = np.array(
    [0.0, 0.0005, 0.0010, 0.0015, 0.0020, 0.0030, 0.0040, 0.0050], dtype=np.float64
)
prior_mix_grid = np.array(
    [0.0, 0.002, 0.005, 0.010, 0.015, 0.020, 0.030], dtype=np.float64
)
alpha_add_grid = np.array([0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4], dtype=np.float64)

oof_base = final_submit_transform(oof_pred)

best = (
    multiclass_logloss(y, oof_base),
    "none",
    1.0,
    0.0,
    0.0,
    0.0,
)

for t in temp_grid:
    p_cal = apply_calibration_family(oof_base, temperature=float(t), family="none")
    loss = multiclass_logloss(y, final_submit_transform(p_cal))
    if loss < best[0]:
        best = (loss, "none", float(t), 0.0, 0.0, 0.0)

    for eps in eps_grid:
        p_cal = apply_calibration_family(
            oof_base, temperature=float(t), family="uniform", eps=float(eps)
        )
        loss = multiclass_logloss(y, final_submit_transform(p_cal))
        if loss < best[0]:
            best = (loss, "uniform", float(t), float(eps), 0.0, 0.0)

    for pm in prior_mix_grid:
        p_cal = apply_calibration_family(
            oof_base,
            temperature=float(t),
            family="prior",
            prior=prior,
            prior_mix=float(pm),
        )
        loss = multiclass_logloss(y, final_submit_transform(p_cal))
        if loss < best[0]:
            best = (loss, "prior", float(t), 0.0, float(pm), 0.0)

    for aa in alpha_add_grid:
        p_cal = apply_calibration_family(
            oof_base, temperature=float(t), family="additive", alpha_add=float(aa)
        )
        loss = multiclass_logloss(y, final_submit_transform(p_cal))
        if loss < best[0]:
            best = (loss, "additive", float(t), 0.0, 0.0, float(aa))

best_loss, best_family, best_t, best_eps, best_pm, best_aa = best
print(
    "Best OOF logloss:",
    best_loss,
    "best_family:",
    best_family,
    "best_temperature:",
    best_t,
    "best_eps:",
    best_eps,
    "best_prior_mix:",
    best_pm,
    "best_alpha_add:",
    best_aa,
)



## === cell 5
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

models = []
for rs in seeds:
    mlp = MLPClassifier(random_state=rs, **mlp_params)
    mlp.fit(X, y)
    models.append(mlp)

print("Trained", len(models), "MLP models.")



## === cell 6
mlp0 = models[0]
if hasattr(mlp0, "loss_curve_") and len(mlp0.loss_curve_) > 0:
    print("Final training loss (model 0):", float(mlp0.loss_curve_[-1]))
    plt.plot(mlp0.loss_curve_, "o-")
    plt.xlabel("Iteration")
    plt.ylabel("Training loss")
    plt.title("Training loss vs Iteration (model 0)")
    plt.show()



## === cell 7
test_ids = test_df["id"].values
test_features = test_df.drop(columns=["id"]).values
test_scaled = scaler.transform(test_features)

y_pred = None
for m in models:
    p = m.predict_proba(test_scaled)
    p = np.asarray(p, dtype=np.float64)
    y_pred = p if y_pred is None else (y_pred + p)

y_pred /= float(len(models))
y_pred = final_submit_transform(y_pred)

y_pred = apply_calibration_family(
    y_pred,
    temperature=best_t,
    family=best_family,
    eps=best_eps,
    prior=prior,
    prior_mix=best_pm,
    alpha_add=best_aa,
)

print(
    "y_pred shape:", y_pred.shape, "min/max:", float(y_pred.min()), float(y_pred.max())
)



## === cell 8
class_cols = [c for c in sample_sub.columns if c != "id"]

encoded_to_species = {enc: le.classes_[enc] for enc in models[0].classes_}
pred_species_cols = [encoded_to_species[enc] for enc in models[0].classes_]

pred_df = pd.DataFrame(y_pred, columns=pred_species_cols)
pred_df.insert(0, "id", test_ids)

submission = pred_df.reindex(columns=["id"] + class_cols, fill_value=0.0)

for c in class_cols:
    submission[c] = submission[c].clip(0.0, 1.0)

submission[class_cols] = final_submit_transform(submission[class_cols].values)

print("submission:", submission.shape)
print(submission.head(2))



## === cell 9
out_path = "/kaggle/working/submission_nn_kernel.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(
    "Used calibration:",
    "family =",
    best_family,
    "temperature =",
    best_t,
    "eps =",
    best_eps,
    "prior_mix =",
    best_pm,
    "alpha_add =",
    best_aa,
)
