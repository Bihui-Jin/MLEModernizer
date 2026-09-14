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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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

0.3173781628771867

# 6. Current score

0.27247

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.25486) has done: 'I fixed the protobuf import error by setting the required environment variable before importing TensorFlow, and I corrected the extraction of target columns so that only the numeric label columns are used (avoiding the “cooking” string conversion error). These changes let the script run end‑to‑end and produce a valid `submission.csv` file.'
- What this solution (achieved 0.27347) has done: 'The fix adds the missing Scikit‑learn imports, replaces the TensorFlow model with a lightweight `MultiOutputRegressor` based on Ridge regression (avoiding the protobuf‑TensorFlow crash), and keeps the original feature engineering (URL netloc one‑hot, TF‑IDF + SVD). The cross‑validation loop now trains and predicts with this model, then writes a proper `submission.csv` matching the required format.'
- What this solution (achieved 0.27247) has done: 'I add a StandardScaler to the Ridge regression pipeline and lower the regularization strength (alpha = 0.5). Scaling the features can improve the correlation of the predictions, and a weaker ridge penalty often helps when the feature set already contains many informative dimensions. The rest of the pipeline and feature engineering stay unchanged.'

# 9. Code solution

## === cell 0
import os, re, gc, sys
import numpy as np, pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"




## === cell 1
DIR = "/kaggle/input/google-quest-challenge"




## === cell 2
train_df = pd.read_csv(os.path.join(DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DIR, "test.csv"))




## === cell 3
train_df["netloc"] = train_df.url.apply(
    lambda x: re.search(r"//.*?\.", x).group(0)[2:-1]
)
test_df["netloc"] = test_df.url.apply(lambda x: re.search(r"//.*?\.", x).group(0)[2:-1])

from sklearn.preprocessing import OneHotEncoder

cat_cols = ["netloc", "category"]
ohe = OneHotEncoder(handle_unknown="ignore")
ohe.fit(pd.concat([train_df[cat_cols], test_df[cat_cols]]))
features_train = ohe.transform(train_df[cat_cols]).toarray()
features_test = ohe.transform(test_df[cat_cols]).toarray()




## === cell 4
text_cols = ["question_title", "question_body", "answer"]
train_text = (
    train_df[text_cols[0]] + " " + train_df[text_cols[1]] + " " + train_df[text_cols[2]]
).fillna("")
test_text = (
    test_df[text_cols[0]] + " " + test_df[text_cols[1]] + " " + test_df[text_cols[2]]
).fillna("")

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
tfidf.fit(pd.concat([train_text, test_text]))
train_tfidf = tfidf.transform(train_text)
test_tfidf = tfidf.transform(test_text)

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=128, random_state=42)
svd.fit(train_tfidf)
train_svd = svd.transform(train_tfidf)
test_svd = svd.transform(test_tfidf)

import numpy as np

X_train = np.hstack([train_svd, features_train])
X_test = np.hstack([test_svd, features_test])




## === cell 5
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sample_submission_path = os.path.join(DIR, "sample_submission.csv")
target_cols = pd.read_csv(sample_submission_path).columns[1:]
Y_train = train_df[target_cols].values.astype(np.float32)




## === cell 6
n_folds = 5
kf = KFold(n_splits=n_folds, random_state=10, shuffle=True)

test_pred_accum = np.zeros((X_test.shape[0], Y_train.shape[1]), dtype=np.float32)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    ridge_pipe = make_pipeline(StandardScaler(), Ridge(alpha=0.5, random_state=fold))
    model = MultiOutputRegressor(ridge_pipe)
    model.fit(X_train[train_idx], Y_train[train_idx])
    test_pred_accum += model.predict(X_test) / n_folds
    print(f"\nFold {fold+1}/{n_folds} completed.")




## === cell 7
sample_submission = pd.read_csv(os.path.join(DIR, "sample_submission.csv"))
sample_submission.iloc[:, 1:] = test_pred_accum
sample_submission.to_csv("submission.csv", index=False)
print("submission.csv written, shape:", sample_submission.shape)
