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

cudf-polars-cu12==25.6.0
geopandas==0.14.4
lightgbm==4.6.0
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
polars==1.25.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1

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

0.8129436990726766

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The fix adds the missing imports, data loading, simple yet effective text feature extraction, a LightGBM training routine with cross‑validation, and the generation of a proper `submission.csv`. It replaces undefined functions with concrete implementations, ensures all variables are defined, and writes the required output file while keeping the original workflow structure.'
- What this solution (achieved 0.0) has done: 'I fix the KFold iteration error by using its `.split()` method, which returns train/validation index arrays. This change enables model training to run, produces predictions, and writes a proper `submission.csv`, moving the solution from a failing state (score 0.0) toward the target metric without altering the core modeling logic.'
- What this solution (achieved 0.0) has done: 'I convert the combined sparse feature matrices to CSR format so they can be indexed by the K‑Fold split indices, and I slightly increase the LightGBM `num_leaves` to give the model a bit more capacity while keeping the original workflow intact. These fixes enable the training loop to run without errors and should move the quadratic weighted kappa score toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb


class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = None  # set a path to load pre‑trained models
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"


np.random.seed(CFG.SEED)




## === cell 1
train_path = os.path.join(CFG.BASE_PATH, "train.csv")
test_path = os.path.join(CFG.BASE_PATH, "test.csv")
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

assert (
    "essay_id" in df_train.columns
    and "full_text" in df_train.columns
    and "score" in df_train.columns
)
assert "essay_id" in df_test.columns and "full_text" in df_test.columns




## === cell 2
def count_misspelled_words(text):
    return 0  # placeholder – does not affect core model


tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words="english",
    dtype=np.float32,
)

X_train_text = tfidf.fit_transform(df_train["full_text"].fillna(""))
X_test_text = tfidf.transform(df_test["full_text"].fillna(""))

miss_train = df_train["full_text"].apply(count_misspelled_words).values.reshape(-1, 1)
miss_test = df_test["full_text"].apply(count_misspelled_words).values.reshape(-1, 1)

from scipy import sparse

X_train = sparse.hstack([X_train_text, miss_train])
X_test = sparse.hstack([X_test_text, miss_test])

X_train = X_train.tocsr()
X_test = X_test.tocsr()

y = df_train["score"].values




## === cell 3
if CFG.LOAD_MODELS_FROM is None:
    print("Training LightGBM models with K‑fold")
    folds = KFold(n_splits=5, shuffle=True, random_state=CFG.SEED)
    model_paths = []
    for fold_idx, (tr_idx, val_idx) in enumerate(folds.split(X_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        lgb_train = lgb.Dataset(X_tr, label=y_tr)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 127,  # increased capacity without changing core logic
            "seed": CFG.SEED,
            "verbosity": -1,
        }

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_train, lgb_val],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        model_path = f"LGB_v{CFG.VER}_f{fold_idx}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        model_paths.append(model_path)
        print(f"Fold {fold_idx+1} trained, best iteration {model.best_iteration}")

else:
    model_paths = [
        os.path.join(CFG.LOAD_MODELS_FROM, f"LGB_v{CFG.VER}_f{i}.pkl") for i in range(5)
    ]




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1363469386.py in <cell line: 0>()
     19         }
     20 
---> 21         model = lgb.train(
     22             params,
     23             lgb_train,

TypeError: train() got an unexpected keyword argument 'early_stopping_rounds'

## === cell 4
preds = []
for i, path in enumerate(model_paths):
    with open(path, "rb") as f:
        model = pickle.load(f)
    pred = model.predict(X_test, num_iteration=model.best_iteration)
    preds.append(pred)

pred1 = np.mean(preds, axis=0)




## === cell 5
submission = pd.DataFrame(
    {
        "essay_id": df_test["essay_id"].values,
        "score": np.clip(pred1, 1, 6).round().astype(int),
    }
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Submission saved to", submission_path)
print("Submission shape:", submission.shape)
submission.head()
