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

0.2710967102177729

# 6. Current score

0.31487

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.31487) has done: 'Implemented two key speed‑ups while keeping the exact algorithmic intent:

* **Removed the redundant `MultiOutputRegressor` wrapper.** `RandomForestRegressor` natively supports multi‑output regression, so fitting a single forest (300 trees) instead of 30 separate forests (≈ 9 000 trees) cuts training time by roughly an order of magnitude.
* **Enabled full parallelism** by setting `n_jobs=-1` (all available cores) and kept the original hyper‑parameters and data split, so predictions and final submission remain unchanged.'

# 9. Code solution

## === cell 0
import os, re, gc
import numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import scipy.sparse as sp

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"



## === cell 1
DIR = "/kaggle/input/google-quest-challenge"
BATCH_SIZE = 64
RANDOM_STATE = 42



## === cell 2
train_df = pd.read_csv(os.path.join(DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DIR, "test.csv"))


def extract_netloc(url):
    try:
        return re.search(r"//.*?\.", url).group(0)[2:-1]
    except Exception:
        return "unknown"


train_df["netloc"] = train_df.url.apply(extract_netloc)
test_df["netloc"] = test_df.url.apply(extract_netloc)

categorical_cols = ["netloc", "category"]
all_cats = pd.concat([train_df[categorical_cols], test_df[categorical_cols]])
ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
ohe.fit(all_cats)

features_train = ohe.transform(train_df[categorical_cols])  # sparse matrix
features_test = ohe.transform(test_df[categorical_cols])  # sparse matrix




## === cell 3
def combine_text(df):
    return (
        df["question_title"].fillna("")
        + " "
        + df["question_body"].fillna("")
        + " "
        + df["answer"].fillna("")
    ).astype(str)


train_text = combine_text(train_df)
test_text = combine_text(test_df)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
tfidf.fit(train_text)

tfidf_train = tfidf.transform(train_text)  # sparse CSR matrix
tfidf_test = tfidf.transform(test_text)  # sparse CSR matrix



## === cell 4
X_train = sp.hstack([tfidf_train, features_train]).tocsr()
X_test = sp.hstack([tfidf_test, features_test]).tocsr()



## === cell 5
sample_sub_path = os.path.join(DIR, "sample_submission.csv")
sample_submission = pd.read_csv(sample_sub_path)
target_cols = sample_submission.columns[1:]  # all 30 label columns

Y_train = train_df[target_cols].values.astype(np.float32)



## === cell 6
base_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=RANDOM_STATE,
    n_jobs=-1,  # full parallelism
    min_samples_split=2,
    min_samples_leaf=1,
)



## === cell 7
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, Y_train, test_size=0.1, random_state=RANDOM_STATE
)

base_model.fit(X_tr, y_tr)



## === cell 8
pred_test = base_model.predict(X_test)
pred_test = np.clip(pred_test, 0.0, 1.0)  # ensure range [0,1]

sample_submission.iloc[:, 1:] = pred_test
output_path = "submission.csv"
sample_submission.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")
