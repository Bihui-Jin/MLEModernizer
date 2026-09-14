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

0.72994

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.01323) has done: 'I make the script robust and slightly improve the model without changing its core logic.  
Key changes: replace the shell‑style `!ls` with a portable Python directory listing, include both raw and log‑transformed length features as predictors, soften the regularisation by setting `alpha=0.5` and enable automatic lambda search, and add a clean shutdown of the H2O cluster. These adjustments keep the overall workflow identical while expected to move the quadratic weighted kappa closer to the target score and ensure a valid `submission.csv` is written.'
- What this solution (achieved 0.01323) has done: 'I adjust the H2O initialization to use the current‑compatible `quiet` flag, enable keeping cross‑validation predictions in the GLM estimator, and make the QWK calculation robust to the case where those predictions are unavailable. These small fixes let the script run through to the end, produce a proper `submission.csv`, and modestly improve the score without altering the core modeling approach.'
- What this solution (achieved 0.6426) has done: 'I fixed the H2O initialization argument (removed the unsupported `quiet` flag), made the log‑features robust by using `np.log1p` (so zeros don’t produce `-inf`), and guarded the final shutdown to avoid errors when the cluster is already disconnected. These changes let the notebook run end‑to‑end, produce a proper `submission.csv`, and should improve the quadratic weighted kappa score toward the target.'
- What this solution (achieved 0.6426) has done: 'Fix the cross‑validation prediction handling: H2O returns a list of frames, causing an AttributeError. Replace the faulty block with a straightforward prediction on the training set and compute the quadratic weighted kappa on those predictions. This keeps the original modeling pipeline intact while providing a valid QWK estimate and ensures the notebook runs to completion, producing a correct `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h2o
from h2O.estimators.glm import H2OGeneralizedLinearEstimator
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
import re



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2887566842.py in <cell line: 0>()
      4 import seaborn as sns
      5 import h2o
----> 6 from h2O.estimators.glm import H2OGeneralizedLinearEstimator
      7 from pathlib import Path
      8 from sklearn.metrics import cohen_kappa_score

ModuleNotFoundError: No module named 'h2O'

## === cell 1
data_dir = Path("../input/learning-agency-lab-automated-essay-scoring-2")
for p in sorted(data_dir.iterdir()):
    print(p)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2347096719.py in <cell line: 0>()
----> 1 data_dir = Path("../input/learning-agency-lab-automated-essay-scoring-2")
      2 for p in sorted(data_dir.iterdir()):
      3     print(p)
      4 

NameError: name 'Path' is not defined

## === cell 2
my_random_seed = 123
default_color_1 = "darkblue"
default_color_2 = "darkgreen"
default_color_3 = "darkred"
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)  # show full text



## === cell 3
df_train = pd.read_csv(data_dir / "train.csv")
df_test = pd.read_csv(data_dir / "test.csv")
df_sub = pd.read_csv(data_dir / "sample_submission.csv")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/938541101.py in <cell line: 0>()
----> 1 df_train = pd.read_csv(data_dir / "train.csv")
      2 df_test = pd.read_csv(data_dir / "test.csv")
      3 df_sub = pd.read_csv(data_dir / "sample_submission.csv")
      4 

NameError: name 'data_dir' is not defined

## === cell 4
df_train.full_text = df_train.full_text.str.strip()
df_test.full_text = df_test.full_text.str.strip()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4114124218.py in <cell line: 0>()
----> 1 df_train.full_text = df_train.full_text.str.strip()
      2 df_test.full_text = df_test.full_text.str.strip()
      3 

NameError: name 'df_train' is not defined

## === cell 5
df_train["score"] = df_train["score"].astype(int)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2153082417.py in <cell line: 0>()
----> 1 df_train["score"] = df_train["score"].astype(int)
      2 

NameError: name 'df_train' is not defined

## === cell 6
df_train["n_char"] = df_train.full_text.str.len()
df_train["n_word"] = df_train.full_text.str.split().map(len)
df_train["char_per_word"] = df_train.n_char / df_train.n_word

df_test["n_char"] = df_test.full_text.str.len()
df_test["n_word"] = df_test.full_text.str.split().map(len)
df_test["char_per_word"] = df_test.n_char / df_test.n_word

df_train["n_sent"] = df_train.full_text.str.count(r"[.!?]")
df_test["n_sent"] = df_test.full_text.str.count(r"[.!?]")

df_train["avg_word_len"] = df_train.n_char / df_train.n_word
df_test["avg_word_len"] = df_test.n_char / df_test.n_word

df_train["punct_count"] = df_train.full_text.apply(
    lambda x: len(re.findall(r"[^\w\s]", x))
)
df_test["punct_count"] = df_test.full_text.apply(
    lambda x: len(re.findall(r"[^\w\s]", x))
)

features_new = [
    "n_char",
    "n_word",
    "char_per_word",
    "n_sent",
    "avg_word_len",
    "punct_count",
]



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/615445576.py in <cell line: 0>()
      1 # basic length based features
----> 2 df_train["n_char"] = df_train.full_text.str.len()
      3 df_train["n_word"] = df_train.full_text.str.split().map(len)
      4 df_train["char_per_word"] = df_train.n_char / df_train.n_word
      5 

NameError: name 'df_train' is not defined

## === cell 7
for col in features_new:
    df_train[f"log_{col}"] = np.log1p(df_train[col])
    df_test[f"log_{col}"] = np.log1p(df_test[col])

features_log = [f"log_{col}" for col in features_new]



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3297207546.py in <cell line: 0>()
      1 # log‑transformed versions (log1p to avoid -inf)
----> 2 for col in features_new:
      3     df_train[f"log_{col}"] = np.log1p(df_train[col])
      4     df_test[f"log_{col}"] = np.log1p(df_test[col])
      5 

NameError: name 'features_new' is not defined

## === cell 8
h2o.init(max_mem_size="8G", nthreads=4, verbose=False)



## === cell 9
col4upload = ["essay_id"] + features_new + features_log
train_hex = h2o.H2OFrame(df_train[col4upload + ["score"]])
test_hex = h2o.H2OFrame(df_test[col4upload])



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2401624216.py in <cell line: 0>()
----> 1 col4upload = ["essay_id"] + features_new + features_log
      2 train_hex = h2o.H2OFrame(df_train[col4upload + ["score"]])
      3 test_hex = h2o.H2OFrame(df_test[col4upload])
      4 

NameError: name 'features_new' is not defined

## === cell 10
train_hex["score"] = train_hex["score"].asfactor()
predictors = features_new + features_log



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1429816302.py in <cell line: 0>()
----> 1 train_hex["score"] = train_hex["score"].asfactor()
      2 predictors = features_new + features_log
      3 

NameError: name 'train_hex' is not defined

## === cell 11
glm_model = H2OGeneralizedLinearEstimator(
    family="multinomial",
    standardize=True,
    nfolds=5,
    alpha=0.0,  # pure ridge regularisation
    lambda_search=True,
    score_each_iteration=True,
    keep_cross_validation_predictions=True,
    seed=my_random_seed,
)

glm_model.train(x=predictors, y="score", training_frame=train_hex)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2690999117.py in <cell line: 0>()
----> 1 glm_model = H2OGeneralizedLinearEstimator(
      2     family="multinomial",
      3     standardize=True,
      4     nfolds=5,
      5     alpha=0.0,  # pure ridge regularisation

NameError: name 'H2OGeneralizedLinearEstimator' is not defined

## === cell 12
train_pred = glm_model.predict(train_hex).as_data_frame()
kappa = cohen_kappa_score(
    df_train["score"].astype(str),
    train_pred["predict"].astype(str),
    weights="quadratic",
)
print(f"Training QWK (approx.): {kappa:.5f}")



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1041953249.py in <cell line: 0>()
----> 1 train_pred = glm_model.predict(train_hex).as_data_frame()
      2 kappa = cohen_kappa_score(
      3     df_train["score"].astype(str),
      4     train_pred["predict"].astype(str),
      5     weights="quadratic",

NameError: name 'glm_model' is not defined

## === cell 13
pred_test = glm_model.predict(test_hex).as_data_frame()
df_sub["score"] = pred_test["predict"].astype(int)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2157873090.py in <cell line: 0>()
----> 1 pred_test = glm_model.predict(test_hex).as_data_frame()
      2 df_sub["score"] = pred_test["predict"].astype(int)
      3 

NameError: name 'glm_model' is not defined

## === cell 14
submission_path = Path("submission.csv")
df_sub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path.resolve()}")



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3462862210.py in <cell line: 0>()
----> 1 submission_path = Path("submission.csv")
      2 df_sub.to_csv(submission_path, index=False)
      3 print(f"Submission written to {submission_path.resolve()}")
      4 

NameError: name 'Path' is not defined

## === cell 15
try:
    h2o.shutdown(prompt=False)
except Exception:
    pass
