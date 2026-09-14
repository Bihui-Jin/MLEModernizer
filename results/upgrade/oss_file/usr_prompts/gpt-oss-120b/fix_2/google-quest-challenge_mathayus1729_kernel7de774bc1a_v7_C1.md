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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
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
seaborn==0.12.2
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
tf_keras==2.18.0

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

0.20176

# 6. Current score

0.32177

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.32177) has done: 'I removed the TensorFlow import that caused a protobuf AttributeError and replaced the LSTM model with a lightweight TF‑IDF + TruncatedSVD feature extraction followed by a multi‑output Ridge regression. This avoids the failing TF dependency, still performs reasonable learning, clamps predictions to [0, 1], and writes a proper `submission.csv` matching the required format.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def plot_len(df, col_name, i):
    plt.figure(i)
    sns.histplot(df[col_name].str.len(), kde=False)
    plt.ylabel("length of string")
    plt.show()


def plot_cnt_words(df, col_name, i):
    plt.figure(i)
    vals = df[col_name].apply(lambda x: len(str(x).strip().split()))
    sns.histplot(vals, kde=False)
    plt.ylabel("count of words")
    plt.show()


def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", str(text))
    text = text.lower()
    return text




## === cell 1
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

df_train = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
df_test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")
df_submission = pd.read_csv(
    "/kaggle/input/google-quest-challenge/sample_submission.csv"
)


def train_and_predict(df_train, df_test, df_submission):
    target_cols = df_submission.columns[1:]  # exclude qa_id
    text_cols = ["question_title", "question_body", "answer"]

    for col in text_cols:
        df_train[col] = df_train[col].astype(str).apply(clean_text)
        df_test[col] = df_test[col].astype(str).apply(clean_text)

    df_train["combined"] = (
        df_train["question_title"]
        + " "
        + df_train["question_body"]
        + " "
        + df_train["answer"]
    )
    df_test["combined"] = (
        df_test["question_title"]
        + " "
        + df_test["question_body"]
        + " "
        + df_test["answer"]
    )

    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(df_train["combined"])
    X_test_tfidf = tfidf.transform(df_test["combined"])

    svd = TruncatedSVD(n_components=256, random_state=42)
    X_train = svd.fit_transform(X_train_tfidf)
    X_test = svd.transform(X_test_tfidf)

    model = MultiOutputRegressor(Ridge(alpha=1.0))
    model.fit(X_train, df_train[target_cols])

    preds = model.predict(X_test)
    preds = np.clip(preds, 0.0, 1.0)

    out_dict = {"qa_id": df_test["qa_id"]}
    for idx, col in enumerate(target_cols):
        out_dict[col] = preds[:, idx]
    submission = pd.DataFrame(out_dict)

    submission.to_csv("submission.csv", index=False)


train_and_predict(df_train, df_test, df_submission)
