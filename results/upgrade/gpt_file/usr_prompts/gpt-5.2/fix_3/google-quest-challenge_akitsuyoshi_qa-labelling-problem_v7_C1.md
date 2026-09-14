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

3.11

# 3. Installed packages

geopandas==0.14.4
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
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0

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

0.15083

# 6. Current score

0.3022

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.29735) has done: 'Your current score (0.28781) is substantially higher than the target (0.15083), so the goal is to *reduce* performance in a controlled, minimal way while keeping the same overall pipeline (TF-IDF → SVD → MultiOutput KNN). The smallest reliable lever is to make the KNN model more biased/smoother by increasing `n_neighbors`, which typically lowers rank-correlation without changing the approach. I also add a fixed `random_state` to the train/validation split to keep the measured effect stable run-to-run, and I clip predictions to `[0,1]` to match submission requirements (usually negligible but safer). The script still run end-to-end and write `submission.csv` with the correct columns.'
- What this solution (achieved 0.3022) has done: 'Your current public score (0.29735) is well above the target (0.15083), so to move *toward* the target we should gently reduce model performance without changing the pipeline (TF‑IDF → SVD → MultiOutput KNN). The smallest stable lever is to further increase `n_neighbors`, which makes KNN predictions smoother/more biased and typically lowers Spearman rank correlation. I’m keeping everything else the same (same features, same SVD size, same MultiOutputRegressor), and I’m keeping clipping to `[0,1]` and the existing `random_state` to keep behavior consistent. The script still runs end-to-end and writes a valid `submission.csv` with the correct columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re, string

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv", low_memory=True)
test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv", low_memory=True)
subm = pd.read_csv(
    "/kaggle/input/google-quest-challenge/sample_submission.csv", low_memory=True
)
train.shape, test.shape, subm.shape



## === cell 2
target_cols = subm.columns[1:]
feat_cols = test.columns[1:]
target_cols, feat_cols



## === cell 3
train.head(4)



## === cell 4
print(train["question_title"][1])
print("------")
print(train["question_body"][1])
print("------")
print(train["answer"][1])



## === cell 5
print(train["question_title"][100])
print("-----")
print(train["question_body"][100])
print("-----")
print(train["answer"][100])



## === cell 6
lens_quest = train.question_body.str.len()
sns.histplot(lens_quest)
plt.title("Question Body Length")
plt.show()
lens_quest.min(), lens_quest.mean(), lens_quest.max(), lens_quest.std()



## === cell 7
lens_ans = train.answer.str.len()
sns.histplot(lens_ans)
plt.title("Answer Body Length")
plt.show()
lens_ans.min(), lens_ans.mean(), lens_ans.max(), lens_ans.std()



## === cell 8
train[train.question_title.str.len() < 10]



## === cell 9
train[train.question_body.str.len() < 10]



## === cell 10
train[train.answer.str.len() < 30].answer



## === cell 11
train.describe()



## === cell 12
train.info()



## === cell 13
train[train.isna().sum(axis=1) == 1]



## === cell 14
colors = sns.color_palette("pastel")[0:5]
plt.pie(
    train.category.value_counts(),
    labels=train.category.value_counts().index,
    colors=colors,
    autopct="%.0f%%",
)
plt.show()



## === cell 15
train["host_type"] = train.host.apply(lambda x: x.split(".")[0])
test["host_type"] = test.host.apply(lambda x: x.split(".")[0])
print("Top Genre: ")
print(train.host_type.value_counts()[:5])
print("Lower Genre: ")
print(train.host_type.value_counts()[-5:])



## === cell 16
sns.kdeplot(train.answer_well_written)
sns.kdeplot(train.question_well_written)
plt.legend()
plt.show()



## === cell 17
from sklearn.decomposition import TruncatedSVD

re_tok = re.compile(f"([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])")


def tokenize(s):
    return re_tok.sub(r" \1 ", s).split()


vect = TfidfVectorizer(
    ngram_range=(1, 2),
    tokenizer=tokenize,
    min_df=round(len(train) * 0.001),
    max_df=0.9,
    sublinear_tf=True,
)

tsvd = TruncatedSVD(100)



## === cell 18
train["QA"] = (
    "xxhostxx"
    + train["host_type"]
    + "xxqtixx "
    + train["question_title"]
    + "xxqbodxx "
    + train["question_body"]
    + "xxcategoryxx "
    + train["category"]
    + "xxq_userxx "
    + train["question_user_name"]
    + "xxansxx "
    + train["answer"]
    + "xxans_userxx "
    + train["answer_user_name"]
)
test["QA"] = (
    "xxhostxx"
    + test["host_type"]
    + "xxqtixx "
    + test["question_title"]
    + "xxqbodxx "
    + test["question_body"]
    + "xxcategoryxx "
    + test["category"]
    + "xxq_userxx "
    + test["question_user_name"]
    + "xxansxx "
    + test["answer"]
    + "xxans_userxx "
    + test["answer_user_name"]
)

vected_train = vect.fit_transform(train["QA"])
vected_test = vect.transform(test["QA"])

vected_train = tsvd.fit_transform(vected_train)
vected_test = tsvd.transform(vected_test)



## === cell 19
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    vected_train, train[target_cols], test_size=0.3, random_state=42
)
X_train.shape, X_test.shape



## === cell 20
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor

estimator = KNeighborsRegressor(n_neighbors=800)
model = MultiOutputRegressor(estimator)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)



## === cell 21
from scipy.stats import spearmanr


def spearmancoff(y_pred, y_true):
    return np.mean(
        [spearmanr(y_pred[:, i], y_true.iloc[:, i])[0] for i in range(y_true.shape[1])]
    )


spearmancoff(y_pred, y_test)



## === cell 22
estimator = KNeighborsRegressor(n_neighbors=800)
model = MultiOutputRegressor(estimator)
model.fit(vected_train, train[target_cols])

y_pred = model.predict(vected_test)

y_pred = np.clip(y_pred, 0.0, 1.0)

submission = pd.concat(
    [
        pd.DataFrame({"qa_id": test["qa_id"]}),
        pd.DataFrame(y_pred, columns=target_cols),
    ],
    axis=1,
)
submission.to_csv("submission.csv", index=False)



## === cell 23
submission
