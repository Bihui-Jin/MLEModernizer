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

0.72859

# 6. Current score

None

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.63129) has done: 'I make the script runnable in your environment by fixing the input paths (your files are under `/kaggle/input/...`, not `../input/...`) and by renumbering cells to start at 1 as required. To move QWK upward toward the target without changing the model family/architecture or training approach, I keep your H2O multinomial GLM and thresholding, but (a) compute threshold-fitting from the saved out-of-fold predictions you already enabled (instead of the holdout frame, which can be missing/unsupported), and (b) ensure we use the expected-score from class probabilities consistently. Finally, I make the submission alignment strictly follow `sample_submission.csv` ordering and assert row/ID integrity before writing `submission.csv`.'

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
data_dir = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"
if os.path.isdir(data_dir):
    print("Listing:", data_dir)
    for name in sorted(os.listdir(data_dir))[:50]:
        path = os.path.join(data_dir, name)
        print(f"{name}\t{os.path.getsize(path)} bytes")
else:
    print("Data directory not found:", data_dir)



## === cell 2
default_color_1 = "darkblue"
default_color_2 = "darkgreen"
default_color_3 = "darkred"

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

my_random_seed = 123



## === cell 3
df_train = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
df_test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
df_sub = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)



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
df_train["n_word"] = df_train.full_text.str.split().map(lambda x: len(x))
df_train["n_word"] = df_train["n_word"].clip(lower=1)
df_train["char_per_word"] = df_train.n_char / df_train.n_word

df_test["n_char"] = df_test.full_text.str.len()
df_test["n_word"] = df_test.full_text.str.split().map(lambda x: len(x))
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
eps = 1e-9
df_train["log_n_char"] = np.log10(
    np.maximum(df_train.n_char.to_numpy(dtype=float), eps)
)
df_train["log_n_word"] = np.log10(
    np.maximum(df_train.n_word.to_numpy(dtype=float), eps)
)
df_train["log_char_per_word"] = np.log10(
    np.maximum(df_train.char_per_word.to_numpy(dtype=float), eps)
)

df_test["log_n_char"] = np.log10(np.maximum(df_test.n_char.to_numpy(dtype=float), eps))
df_test["log_n_word"] = np.log10(np.maximum(df_test.n_word.to_numpy(dtype=float), eps))
df_test["log_char_per_word"] = np.log10(
    np.maximum(df_test.char_per_word.to_numpy(dtype=float), eps)
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
h2o.init(max_mem_size="8G", nthreads=4)
h2o.no_progress()
try:
    h2o.cluster().set_timezone("UTC")
except Exception:
    pass



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
    alpha=0.5,
    score_each_iteration=True,
    seed=my_random_seed,
    fold_assignment="Modulo",  # deterministic folds for stable CV/training behavior
    keep_cross_validation_predictions=True,  # needed for OOF preds / threshold fitting
)
glm_model.train(predictors, "score", training_frame=train_hex)



## === cell 28
glm_model.cross_validation_metrics_summary().as_data_frame()



## === cell 29
glm_model.coef()



## === cell 30
glm_model.varimp_plot()



## === cell 31
pred_train = glm_model.predict(train_hex)
pred_train = pred_train.as_data_frame()
pred_train.head()



## === cell 32
print(pred_train.predict.value_counts().sort_index())
plt.figure(figsize=(8, 4))
pred_train.predict.value_counts().sort_index().plot(kind="bar", color=default_color_3)
plt.title("Predictions - Train")
plt.grid()
plt.show()



## === cell 33
conf_train = pd.crosstab(pred_train.predict, df_train["score"])
sns.heatmap(
    conf_train, annot=True, cmap="Reds", fmt=".0f", linecolor="black", linewidths=0.5
)
plt.title("Confusion Matrix - Training")
plt.show()



## === cell 34
sns.pairplot(data=pred_train, hue="predict")
plt.show()



## === cell 35
pred_test = glm_model.predict(test_hex)
pred_test = pred_test.as_data_frame()
pred_test.head()



## === cell 36
print(pred_test.predict.value_counts().sort_index())
pred_test.predict.value_counts().sort_index().plot(kind="bar", color=default_color_3)
plt.title("Predictions - Test")
plt.grid()
plt.show()




## === cell 37
def _extract_probs(df_pred: pd.DataFrame) -> np.ndarray | None:
    cand_sets = [
        [f"p{k}" for k in range(1, 7)],
        [f"p.{k}" for k in range(1, 7)],
        [str(k) for k in range(1, 7)],
    ]
    for cols in cand_sets:
        if all(c in df_pred.columns for c in cols):
            return df_pred[cols].to_numpy(dtype=float)
    return None


def _qwk(y_true: np.ndarray, y_pred: np.ndarray, min_rating=1, max_rating=6) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = max_rating - min_rating + 1

    O = np.zeros((n, n), dtype=float)
    for a, b in zip(y_true, y_pred):
        O[a - min_rating, b - min_rating] += 1.0

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist) / max(O.sum(), 1.0)

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            W[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - (num / den if den > 0 else 0.0)


def _apply_thresholds(x: np.ndarray, thr: np.ndarray) -> np.ndarray:
    return (np.digitize(x, thr, right=False) + 1).astype(int)


def _fit_thresholds_greedy(x: np.ndarray, y: np.ndarray, iters: int = 3) -> np.ndarray:
    thr = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=float)
    best = _qwk(y, _apply_thresholds(x, thr))
    for _ in range(iters):
        for i in range(len(thr)):
            grid = np.linspace(thr[i] - 0.30, thr[i] + 0.30, 31)
            local_best = best
            local_thr_i = thr[i]
            for g in grid:
                cand = thr.copy()
                cand[i] = g
                if i > 0 and cand[i] <= cand[i - 1] + 1e-6:
                    continue
                if i < len(thr) - 1 and cand[i] >= cand[i + 1] - 1e-6:
                    continue
                score = _qwk(y, _apply_thresholds(x, cand))
                if score > local_best:
                    local_best = score
                    local_thr_i = g
            thr[i] = local_thr_i
            best = local_best
    return thr


def _expected_from_pred_df(df_pred: pd.DataFrame) -> np.ndarray:
    probs = _extract_probs(df_pred)
    if probs is not None:
        return probs @ np.arange(1, 7, dtype=float)
    if "predict" not in df_pred.columns:
        raise RuntimeError(
            f"Prediction frame missing 'predict' column; cols={list(df_pred.columns)}"
        )
    return df_pred["predict"].astype(float).to_numpy()


def _get_oof_pred_df(model, train_frame: "h2o.H2OFrame") -> pd.DataFrame:
    """
    Bugfix: H2O may return cross_validation_predictions() as a list of frames (one per fold).
    Prefer cross_val_predict() to get an aligned OOF frame with the same row order as training_frame.
    """
    try:
        oof_hex = model.cross_val_predict()
        if oof_hex is not None:
            df = h2o.as_list(oof_hex, use_pandas=True)
            return df
    except Exception:
        pass

    cv_pred_fr = model.cross_validation_predictions()
    if cv_pred_fr is None:
        raise RuntimeError(
            "No cross-validation predictions found. "
            "Ensure keep_cross_validation_predictions=True and nfolds>1."
        )

    if isinstance(cv_pred_fr, list):
        parts = [h2o.as_list(fr, use_pandas=True) for fr in cv_pred_fr]
        cv_df = pd.concat(parts, axis=0, ignore_index=False)

        idx_cols = [
            c for c in cv_df.columns if c.lower() in ("row_id", "rowid", "row", "index")
        ]
        if idx_cols:
            cv_df = cv_df.sort_values(idx_cols[0]).reset_index(drop=True)
        else:
            cv_df = cv_df.reset_index(drop=True)

        train_ids = (
            h2o.as_list(train_frame["essay_id"], use_pandas=True)["essay_id"]
            .astype(str)
            .to_numpy()
        )
        if len(train_ids) != len(cv_df):
            raise RuntimeError(
                f"OOF concat length mismatch: train {len(train_ids)} vs cv {len(cv_df)}"
            )
        cv_df.insert(0, "essay_id", train_ids)
        return cv_df

    cv_df = h2o.as_list(cv_pred_fr, use_pandas=True)
    if "essay_id" not in cv_df.columns:
        train_ids = (
            h2o.as_list(train_frame["essay_id"], use_pandas=True)["essay_id"]
            .astype(str)
            .to_numpy()
        )
        if len(train_ids) != len(cv_df):
            raise RuntimeError(
                f"OOF length mismatch: train {len(train_ids)} vs cv {len(cv_df)}"
            )
        cv_df.insert(0, "essay_id", train_ids)
    return cv_df


y_train = df_train["score"].to_numpy(dtype=int)

cv_pred_df = _get_oof_pred_df(glm_model, train_hex)

if "essay_id" not in cv_pred_df.columns:
    raise RuntimeError(
        "OOF prediction frame missing essay_id; cannot align OOF predictions safely. "
        f"Columns: {list(cv_pred_df.columns)}"
    )

cv_pred_df["essay_id"] = cv_pred_df["essay_id"].astype(str)
train_ids = df_train["essay_id"].astype(str)

cv_pred_df = cv_pred_df.drop_duplicates(submission=["essay_id"], keep="first")

cv_pred_df = train_ids.to_frame().merge(
    cv_pred_df, on="essay_id", how="left", sort=False
)

expected_oof = _expected_from_pred_df(cv_pred_df)

if len(expected_oof) != len(df_train):
    raise RuntimeError(
        f"OOF length mismatch: expected {len(df_train)}, got {len(expected_oof)}"
    )
if np.isnan(expected_oof).any():
    missing = int(np.isnan(expected_oof).sum())
    raise RuntimeError(
        f"OOF contains {missing} NaN predictions; cannot fit thresholds."
    )

thr = _fit_thresholds_greedy(expected_oof, y_train, iters=3)
pred_oof_thr = np.clip(_apply_thresholds(expected_oof, thr), 1, 6)

print(
    "Local OOF QWK (plain rint):",
    _qwk(y_train, np.clip(np.rint(expected_oof).astype(int), 1, 6)),
)
print("Local OOF QWK (thresholded):", _qwk(y_train, pred_oof_thr))
print("Learned thresholds (from OOF preds):", thr)

expected_test = _expected_from_pred_df(pred_test)
pred_score = np.clip(_apply_thresholds(expected_test, thr), 1, 6).astype(int)

pred_map = pd.DataFrame(
    {"essay_id": df_test["essay_id"].astype(str).values, "score": pred_score}
)
submission = df_sub[["essay_id"]].copy()
submission["essay_id"] = submission["essay_id"].astype(str)
submission = submission.merge(pred_map, on="essay_id", how="left", sort=False)

if submission["score"].isna().any():
    missing = int(submission["score"].isna().sum())
    raise RuntimeError(
        f"Submission has {missing} missing predictions after merge; check essay_id alignment."
    )

submission["score"] = submission["score"].clip(1, 6).astype(int)

print("submission shape:", submission.shape)
print("unique essay_id:", submission["essay_id"].nunique(), " / rows:", len(submission))
print("score value counts:\n", submission["score"].value_counts().sort_index())

assert list(submission.columns) == ["essay_id", "score"]
assert len(submission) == len(df_sub)
assert submission["essay_id"].equals(df_sub["essay_id"].astype(str))

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv")

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1106646500.py in <cell line: 0>()
    141 y_train = df_train["score"].to_numpy(dtype=int)
    142 
--> 143 cv_pred_df = _get_oof_pred_df(glm_model, train_hex)
    144 
    145 if "essay_id" not in cv_pred_df.columns:

/tmp/ipykernel_11/1106646500.py in _get_oof_pred_df(model, train_frame)
    116         )
    117         if len(train_ids) != len(cv_df):
--> 118             raise RuntimeError(
    119                 f"OOF concat length mismatch: train {len(train_ids)} vs cv {len(cv_df)}"
    120             )

RuntimeError: OOF concat length mismatch: train 15576 vs cv 77880
