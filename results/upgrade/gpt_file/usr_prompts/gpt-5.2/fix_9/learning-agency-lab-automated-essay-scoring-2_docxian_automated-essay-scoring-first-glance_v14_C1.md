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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
h2o==3.46.0.8
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
seaborn==0.12.2
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.72994

# 6. Current score

None

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator




## === cell 1
def _resolve_input_dir(
    preferred="../input/learning-agency-lab-automated-essay-scoring-2/",
):
    candidates = [
        preferred,
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/",
        "/kaggle/input/",
        "/kaggle/data/learning-agency-lab-automated-essay-scoring-2/",
        "/kaggle/data/",
    ]
    for c in candidates:
        if os.path.isdir(c):
            if os.path.basename(os.path.normpath(c)) in ("input", "data"):
                comp = os.path.join(c, "learning-agency-lab-automated-essay-scoring-2")
                if os.path.isdir(comp):
                    return comp + "/"
            return c if c.endswith("/") else c + "/"
    return preferred


input_dir = _resolve_input_dir()
if os.path.isdir(input_dir):
    for fn in sorted(os.listdir(input_dir))[:50]:
        print(fn)
else:
    print(f"Input dir not found: {input_dir}")



## === cell 2
default_color_1 = "darkblue"
default_color_2 = "darkgreen"
default_color_3 = "darkred"

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

my_random_seed = 123



## === cell 3
train_path = os.path.join(input_dir, "train.csv")
test_path = os.path.join(input_dir, "test.csv")
sub_path = os.path.join(input_dir, "sample_submission.csv")

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_sub = pd.read_csv(sub_path)



## === cell 4
df_train.full_text = df_train.full_text.str.strip()
df_test.full_text = df_test.full_text.str.strip()



## === cell 5
df_train.head()



## === cell 6
df_train.score = df_train.score.astype(int)



## === cell 7
df_train.info()



## === cell 8
df_test



## === cell 9
plt.figure(figsize=(8, 4))
df_train.score.value_counts().sort_index().plot(kind="bar", color=default_color_3)
plt.grid()
plt.title("Score")
plt.show()



## === cell 10
df_train.score.describe()



## === cell 11
df_train["n_char"] = df_train.full_text.str.len()

df_train["n_word"] = df_train.full_text.str.split().map(
    lambda x: len(x) if isinstance(x, list) else 0
)
df_train["n_word"] = df_train["n_word"].clip(lower=1)
df_train["char_per_word"] = df_train.n_char / df_train.n_word

df_test["n_char"] = df_test.full_text.str.len()
df_test["n_word"] = df_test.full_text.str.split().map(
    lambda x: len(x) if isinstance(x, list) else 0
)
df_test["n_word"] = df_test["n_word"].clip(lower=1)
df_test["char_per_word"] = df_test.n_char / df_test.n_word

features_new = ["n_char", "n_word", "char_per_word"]



## === cell 12
df_train[features_new].describe()



## === cell 13
for f in features_new:
    plt.figure(figsize=(10, 3))
    df_train[f].plot(kind="hist", bins=50, color=default_color_1)
    plt.title(f)
    plt.grid()
    plt.show()



## === cell 14
for f in features_new:
    plt.figure(figsize=(10, 1))
    plt.boxplot(df_train[f], vert=False)
    plt.title(f)
    plt.grid()
    plt.show()



## === cell 15
eps = 1.0  # minimal offset; only affects pathological empty/zero-word texts

df_train["log_n_char"] = np.log10(np.maximum(df_train.n_char.astype(float), eps))
df_train["log_n_word"] = np.log10(np.maximum(df_train.n_word.astype(float), eps))
df_train["log_char_per_word"] = np.log10(
    np.maximum(df_train.char_per_word.astype(float), eps)
)

df_test["log_n_char"] = np.log10(np.maximum(df_test.n_char.astype(float), eps))
df_test["log_n_word"] = np.log10(np.maximum(df_test.n_word.astype(float), eps))
df_test["log_char_per_word"] = np.log10(
    np.maximum(df_test.char_per_word.astype(float), eps)
)

features_log = ["log_n_char", "log_n_word", "log_char_per_word"]



## === cell 16
for f in features_log:
    plt.figure(figsize=(10, 3))
    df_train[f].plot(kind="hist", bins=50, color=default_color_1)
    plt.title(f)
    plt.grid()
    plt.show()



## === cell 17
for f in features_log:
    plt.figure(figsize=(10, 1))
    plt.boxplot(df_train[f], vert=False)
    plt.title(f)
    plt.grid()
    plt.show()



## === cell 18
corr_pearson = df_train[["n_char", "n_word", "char_per_word", "score"]].corr(
    method="pearson"
)
fig = plt.figure(figsize=(5, 4))
sns.heatmap(
    corr_pearson,
    annot=True,
    cmap="RdYlGn",
    vmin=-1,
    vmax=+1,
    fmt=".3f",
    linecolor="black",
    linewidths=0.5,
)
plt.title("Pearson Correlation")
plt.show()



## === cell 19
corr_pearson = df_train[
    ["log_n_char", "log_n_word", "log_char_per_word", "score"]
].corr(method="pearson")
fig = plt.figure(figsize=(5, 4))
sns.heatmap(
    corr_pearson,
    annot=True,
    cmap="RdYlGn",
    vmin=-1,
    vmax=+1,
    fmt=".3f",
    linecolor="black",
    linewidths=0.5,
)
plt.title("Pearson Correlation")
plt.show()



## === cell 20
sns.jointplot(data=df_train, x="log_n_char", y="score", color=default_color_1)
plt.show()



## === cell 21
sns.jointplot(data=df_train, x="log_n_word", y="score", color=default_color_1)
plt.show()



## === cell 22
sns.jointplot(data=df_train, x="log_char_per_word", y="score", color=default_color_1)
plt.show()



## === cell 23
df_train.to_csv("training_data.csv", index=False)



## === cell 24
h2o.init(max_mem_size="8G", nthreads=4, name="aes_glm", enable_assertions=False)



## === cell 25
col4upload = ["essay_id"] + features_log
train_hex = h2o.H2OFrame(df_train[col4upload + ["score"]])
test_hex = h2o.H2OFrame(df_test[col4upload])



## === cell 26
train_hex["score"] = train_hex["score"].asfactor()
predictors = features_log



## === cell 27
glm_model = H2OGeneralizedLinearEstimator(
    family="multinomial",
    standardize=True,
    nfolds=5,
    keep_cross_validation_predictions=True,
    alpha=1,  # 0:Ridge (L2), 1:LASSO (L1)
    score_each_iteration=True,
    seed=my_random_seed,
)

glm_model.train(predictors, "score", training_frame=train_hex)



## === cell 28
glm_model.cross_validation_metrics_summary().as_data_frame()



## === cell 29
glm_model.varimp_plot()



## === cell 30
pred_train = glm_model.predict(train_hex).as_data_frame()
pred_train.head()



## === cell 31
print(pred_train.predict.value_counts().sort_index())
plt.figure(figsize=(8, 4))
pred_train.predict.value_counts().sort_index().plot(kind="bar", color=default_color_3)
plt.title("Predictions - Train")
plt.grid()
plt.show()



## === cell 32
conf_train = pd.crosstab(pred_train.predict, df_train["score"])
sns.heatmap(
    conf_train, annot=True, cmap="Reds", fmt=".0f", linecolor="black", linewidths=0.5
)
plt.title("Confusion Matrix - Training")
plt.show()



## === cell 33
pred_test = glm_model.predict(test_hex).as_data_frame()
pred_test.head()



## === cell 34
print(pred_test.predict.value_counts().sort_index())
pred_test.predict.value_counts().sort_index().plot(kind="bar", color=default_color_3)
plt.title("Predictions - Test")
plt.grid()
plt.show()




## === cell 35
def _quadratic_weighted_kappa(y_true, y_pred, min_rating=1, max_rating=6):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}"
        )

    n = max_rating - min_rating + 1
    O = np.zeros((n, n), dtype=float)
    for a, b in zip(y_true, y_pred):
        O[a - min_rating, b - min_rating] += 1.0

    act_hist = np.bincount(y_true - min_rating, minlength=n).astype(float)
    pred_hist = np.bincount(y_pred - min_rating, minlength=n).astype(float)

    E = np.outer(act_hist, pred_hist) / float(len(y_true))

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            W[i, j] = ((i - j) ** 2) / float((n - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - (num / den if den != 0 else 0.0)


def _expected_score_from_pred_df(pred_df: pd.DataFrame):
    prob_cols = [c for c in pred_df.columns if str(c).startswith("p")]
    cls = []
    for c in prob_cols:
        name = str(c)[1:]  # drop leading 'p'
        try:
            cls.append(int(float(name)))
        except Exception:
            return None

    if len(prob_cols) == 0:
        return None

    cls = np.array(cls, dtype=float)
    probs = pred_df[prob_cols].to_numpy(dtype=float)
    if probs.shape[1] != cls.shape[0]:
        return None
    return (probs * cls.reshape(1, -1)).sum(axis=1)


def _apply_cutpoints(exp_scores: np.ndarray, cutpoints: np.ndarray):
    scores = np.digitize(exp_scores, bins=cutpoints, right=True) + 1
    return np.clip(scores, 1, 6).astype(int)


def _optimize_cutpoints_qwk(exp_scores: np.ndarray, y_true: np.ndarray, verbose=True):
    exp_scores = np.asarray(exp_scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    qs = [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]
    cutpoints = np.quantile(exp_scores, qs).astype(float)
    cutpoints = np.maximum.accumulate(cutpoints)

    def score(cp):
        y_pred = _apply_cutpoints(exp_scores, cp)
        return _quadratic_weighted_kappa(y_true, y_pred)

    best = score(cutpoints)
    if verbose:
        print("Initial CV QWK:", best, "cutpoints:", cutpoints)

    for _ in range(4):
        improved = False
        for i in range(5):
            left = -np.inf if i == 0 else cutpoints[i - 1]
            right = np.inf if i == 4 else cutpoints[i + 1]

            span = np.std(exp_scores) * 0.25
            grid = np.linspace(cutpoints[i] - span, cutpoints[i] + span, 31)
            grid = grid[(grid > left + 1e-9) & (grid < right - 1e-9)]
            if grid.size == 0:
                continue

            local_best = best
            local_cp = cutpoints[i]
            for v in grid:
                cp_try = cutpoints.copy()
                cp_try[i] = float(v)
                cp_try = np.maximum.accumulate(cp_try)
                s = score(cp_try)
                if s > local_best + 1e-8:
                    local_best = s
                    local_cp = float(v)

            if local_best > best + 1e-8:
                cutpoints[i] = local_cp
                cutpoints = np.maximum.accumulate(cutpoints)
                best = local_best
                improved = True

        if verbose:
            print("Refined CV QWK:", best, "cutpoints:", cutpoints)
        if not improved:
            break

    return cutpoints, best


cv_hex = glm_model.cross_validation_holdout_predictions()
use_calibration = cv_hex is not None

exp_test = _expected_score_from_pred_df(pred_test)

cutpoints = None
if use_calibration and (exp_test is not None):
    cv_df = cv_hex.as_data_frame()
    if "essay_id" not in cv_df.columns:
        print("Warning: CV holdout predictions missing essay_id; skipping calibration.")
        use_calibration = False
    else:
        cv_df = cv_df.merge(
            df_train[["essay_id", "score"]],
            on="essay_id",
            how="inner",
            validate="one_to_one",
        )
        exp_cv = _expected_score_from_pred_df(cv_df)
        if exp_cv is None:
            print(
                "Warning: Probabilities not available in CV preds; skipping calibration."
            )
            use_calibration = False
        else:
            y_true_cv = cv_df["score"].to_numpy(dtype=int)
            cutpoints, cv_qwk = _optimize_cutpoints_qwk(exp_cv, y_true_cv, verbose=True)
            print("Calibration CV QWK (on aligned CV holdout):", cv_qwk)

if (cutpoints is not None) and (exp_test is not None):
    pred_scores = _apply_cutpoints(exp_test, cutpoints).astype(float)
else:
    print(
        "Warning: Using predicted class labels directly (no probability calibration)."
    )
    pred_scores = (
        pd.to_numeric(pred_test["predict"], errors="coerce")
        .astype("Int64")
        .to_numpy()
        .astype(float)
    )

pred_scores = np.clip(pred_scores, 1, 6)
if np.isnan(pred_scores).any():
    mode_score = int(df_train["score"].mode().iloc[0])
    pred_scores = np.where(np.isnan(pred_scores), mode_score, pred_scores)

pred_scores = pred_scores.astype(int)

sub = pd.DataFrame({"essay_id": df_test["essay_id"].values, "score": pred_scores})
sub["score"] = sub["score"].astype(int)

sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("Score distribution in submission:\n", sub["score"].value_counts().sort_index())
