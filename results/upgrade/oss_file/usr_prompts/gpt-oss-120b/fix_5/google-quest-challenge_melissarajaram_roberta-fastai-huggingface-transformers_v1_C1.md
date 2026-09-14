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
Given questions and answers from various StackExchange properties, predict target values of 30 labels for each question-answer pair.

## Metric
Mean column-wise Spearman's correlation coefficient. The Spearman's rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.

## Submission Format
For each qa_id in the test set, you must predict a probability for each target variable. The predictions should be in the range [0,1]. The file should contain a header and have the following format:

```
qa_id,question_asker_intent_understanding,...,answer_well_written
6,0.0,...,0.5
8,0.5,...,0.1
18,1.0,...,0.0
etc.
```

## Dataset
The list of 30 target labels are the same as the column names in the `sample_submission.csv` file. Target labels with the prefix `question_` relate to the `question_title` and/or `question_body` features in the data. Target labels with the prefix `answer_` relate to the `answer` feature.

Target labels are aggregated from multiple raters, and can have continuous values in the range `[0,1]`. Therefore, predictions must also be in that range.

- **train.csv** - the training data (target labels are the last 30 columns)
- **test.csv** - the test set (you must predict 30 labels for each test set row)
- **sample_submission.csv** - a sample submission file in the correct format; column names are the 30 target labels

# 2. Python version

3.8

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scipy==1.15.3
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
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        input/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        working/
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
```

-> data/google-quest-challenge/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/google-quest-challenge/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/google-quest-challenge/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> data/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.2415212217454294

# 6. Current score

0.28592

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.31909) has done: 'The changes fix the missing torch import, avoid the failing sparse‑CG solver by using Ridge with the “lsqr” solver, and correctly build the submission file using the test qa_id rows rather than the short sample template. These fixes unblock the training, allow predictions, and produce a properly‑shaped submission.csv that meets the competition format.'
- What this solution (achieved 0.29896) has done: 'I slightly increase regularization and simplify the TF‑IDF features so the model is a bit less powerful, which should lower the validation Spearman score from ~0.319 toward the target ~0.242 while keeping the overall pipeline unchanged. The changes are limited to the vectorizer parameters and the Ridge alpha value.'
- What this solution (achieved 0.28592) has done: 'I slightly weaken the model by reducing the TF‑IDF vocabulary size, raising the minimum document frequency, and increasing the Ridge regularization strength. These small adjustments are expected to lower the validation Spearman correlation, moving the score from the current 0.299 toward the target ~0.242 while keeping the overall pipeline unchanged and still producing a correct `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from pathlib import Path
import random
import os
import torch  # added for deterministic seeding

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr




## === cell 1
def seed_all(seed_value: int = 42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False




## === cell 2
seed = 42
seed_all(seed)



## === cell 3
DATA_ROOT = Path("../input/google-quest-challenge/")
train = pd.read_csv(DATA_ROOT / "train.csv")
test = pd.read_csv(DATA_ROOT / "test.csv")
sample_sub = pd.read_csv(DATA_ROOT / "sample_submission.csv")
print(f"train shape: {train.shape}, test shape: {test.shape}")



## === cell 4
labels = list(sample_sub.columns[1:])
print(f"Number of labels: {len(labels)}")




## === cell 5
def combine_text(df):
    cols = ["question_title", "question_body", "answer"]
    for c in cols:
        df[c] = df[c].fillna("")
    return (
        df["question_title"] + " " + df["question_body"] + " " + df["answer"]
    ).values


train_texts = combine_text(train)
test_texts = combine_text(test)



## === cell 6
vectorizer = TfidfVectorizer(
    max_features=5000,  # fewer features
    ngram_range=(1, 1),  # only unigrams
    stop_words="english",
    min_df=5,  # ignore rarer terms
)

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

y = train[labels].values.astype(np.float32)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y, test_size=0.1, random_state=seed
)

base_model = Ridge(
    alpha=200.0, random_state=seed, solver="lsqr"
)  # stronger regularisation
model = MultiOutputRegressor(base_model, n_jobs=-1)
model.fit(X_tr, y_tr)

val_pred = model.predict(X_val)
spearman_scores = []
for i in range(y.shape[1]):
    corr, _ = spearmanr(y_val[:, i], val_pred[:, i])
    spearman_scores.append(corr)
print(f"Validation mean Spearman: {np.mean(spearman_scores):.5f}")

model.fit(X_train, y)



## === cell 7
test_preds = model.predict(X_test)
test_preds = np.clip(test_preds, 0.0, 1.0)



## === cell 8
submission = pd.DataFrame()
submission["qa_id"] = test["qa_id"].values
for idx, label in enumerate(labels):
    submission[label] = test_preds[:, idx]
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}, shape: {submission.shape}")



## === cell 9
print(submission.head())
