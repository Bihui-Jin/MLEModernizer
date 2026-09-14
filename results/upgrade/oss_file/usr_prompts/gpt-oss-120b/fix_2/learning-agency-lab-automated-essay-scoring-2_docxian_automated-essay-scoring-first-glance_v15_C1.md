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

0.72859

# 6. Current score

0.64324

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.64324) has done: 'I replace the shell‑style `!ls` call with regular Python code so the script can run outside a notebook, and I feed a few more straightforward length‑based features to the GLM (the raw counts in addition to the log‑transformed ones). I also enable `lambda_search` in the GLM estimator to let H2O tune regularisation automatically. These small, targeted tweaks keep the overall modeling pipeline unchanged while giving the model a modest chance to move its quadratic weighted‑kappa score closer to the target. The revised script now runs end‑to‑end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator



## === cell 1
import os, pathlib, sys

input_path = pathlib.Path("../input/learning-agency-lab-automated-essay-scoring-2")
if input_path.is_dir():
    print("Contents of", input_path)
    for p in sorted(input_path.iterdir()):
        print(p.name)
else:
    print("Input directory not found:", input_path, file=sys.stderr)



## === cell 2
default_color_1 = "darkblue"
default_color_2 = "darkgreen"
default_color_3 = "darkred"

pd.set_option("display.max_columns", None)
pd.set_option(
    "display.max_colwidth", None
)  # columns can be as wide as necessary to show full content

my_random_seed = 123



## === cell 3
df_train = pd.read_csv(
    "../input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
df_test = pd.read_csv("../input/learning-agency-lab-automated-essay-scoring-2/test.csv")
df_sub = pd.read_csv(
    "../input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
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
df_train["char_per_word"] = df_train.n_char / df_train.n_word

df_test["n_char"] = df_test.full_text.str.len()
df_test["n_word"] = df_test.full_text.str.split().map(lambda x: len(x))
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
df_train["log_n_char"] = np.log10(df_train.n_char.replace(0, np.nan))
df_train["log_n_word"] = np.log10(df_train.n_word.replace(0, np.nan))
df_train["log_char_per_word"] = np.log10(df_train.char_per_word.replace(0, np.nan))

df_test["log_n_char"] = np.log10(df_test.n_char.replace(0, np.nan))
df_test["log_n_word"] = np.log10(df_test.n_word.replace(0, np.nan))
df_test["log_char_per_word"] = np.log10(df_test.char_per_word.replace(0, np.nan))

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
df_train.to_csv("training_data.csv")



## === cell 24
h2o.init(max_mem_size="8G", nthreads=4)  # Use maximum of 8 GB RAM and 4 cores



## === cell 25
col4upload = ["essay_id"] + features_log + features_new
train_hex = h2o.H2OFrame(df_train[col4upload + ["score"]])
test_hex = h2o.H2OFrame(df_test[col4upload])



## === cell 26
train_hex["score"] = train_hex["score"].asfactor()

predictors = features_log + features_new



## === cell 27
glm_model = H2OGeneralizedLinearEstimator(
    family="multinomial",
    standardize=True,
    nfolds=5,
    alpha=0.5,  # 0:Ridge (L2), 1:LASSO (L1)
    lambda_search=True,  # Let H2O automatically tune regularisation
    seed=my_random_seed,
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
df_sub["score"] = pred_test["predict"]
df_sub.to_csv("submission.csv", index=False)
