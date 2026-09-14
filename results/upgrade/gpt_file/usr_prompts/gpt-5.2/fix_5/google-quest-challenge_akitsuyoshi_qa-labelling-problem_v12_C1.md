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

0.22112

# 6. Current score

0.24986

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.29218) has done: 'Your current score (0.29032) is higher than the target (0.22112), so we should make the smallest, safest change that *reduces* performance toward the target band without breaking the pipeline. I keep the exact same feature engineering, vectorization, model type, and training procedure, but I make the KNN slightly more biased/smoother by increasing `n_neighbors`, which typically reduces rank-correlation performance in this setup. I also clip predictions into `[0,1]` to match submission requirements (this is metric-consistent and should not increase performance unexpectedly). Finally, I add a fixed `random_state` for the split used in CV to keep the reported CV score stable.'
- What this solution (achieved 0.26674) has done: 'Your current score (0.29218) is higher than the target (0.22112), so the smallest change to move toward the target is to deliberately smooth the KNN more (which typically reduces rank-correlation performance for this task). I keep the exact same preprocessing, vectorization, dimensionality reduction, model type (KNN via `MultiOutputRegressor`), and training procedure, and only increase `n_neighbors` to reduce model variance. I also keep the existing `[0,1]` clipping to remain submission-valid and avoid accidental out-of-range predictions. The script still trains on all training rows and writes `submission.csv` in the required format.'
- What this solution (achieved 0.2442) has done: 'Your current score (0.26674) is still higher than the target (0.22112), so we should make the smallest change that predictably *reduces* performance toward the target band without changing the pipeline’s core logic. Since this solution’s main “knob” is KNN smoothing, I further increase `n_neighbors`, which typically lowers Spearman performance by over-smoothing rankings. I keep the exact same features, vectorization, SVD, model type, and training-on-full-train behavior, and keep clipping to `[0,1]` to stay submission-valid. The script still run end-to-end and write `submission.csv`.'
- What this solution (achieved 0.24986) has done: 'Your current score (0.2442) is still higher than the target (0.22112), so we should make a very small, controlled change that is likely to *reduce* performance toward the target band without changing the pipeline’s core logic. The safest “knob” in this solution is KNN smoothing, so I further increase `n_neighbors` (more averaging typically degrades rank-based correlation on this task). I also make the TF-IDF `min_df` deterministic (avoid Python `round` ties) to keep behavior stable across runs while preserving the same feature extraction approach. Everything else (features, TF-IDF+SVD, MultiOutput KNN, training on full train, clipping, and submission format) remains the same and still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re, string

from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputRegressor

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
def spearmancoff(y_pred, y_true):
    return np.mean(
        [
            spearmanr(y_pred[:, i], y_true[:, i], nan_policy="omit")[0]
            for i in range(y_true.shape[1])
        ]
    )


scorer = make_scorer(spearmancoff, greater_is_better=True)

re_tok = re.compile(f"([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])")


def tokenize(s):
    return re_tok.sub(r" \1 ", s).split()




## === cell 2
train = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv", low_memory=True)
test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv", low_memory=True)
subm = pd.read_csv(
    "/kaggle/input/google-quest-challenge/sample_submission.csv", low_memory=True
)
train.shape, test.shape, subm.shape



## === cell 3
target_cols = subm.columns[1:]
feat_cols = test.columns[1:]
target_cols, feat_cols



## === cell 4
train.head(4)



## === cell 5
print(train[train.isna().sum(axis=1) == 1])
print(train.info())
train.describe()



## === cell 6
print(train["question_title"][1])
print("------")
print(train["question_body"][1])
print("------")
print(train["answer"][1])



## === cell 7
print(train["question_title"][100])
print("-----")
print(train["question_body"][100])
print("-----")
print(train["answer"][100])



## === cell 8
fig, axs = plt.subplots(6, 5, figsize=(16, 12))
axs = axs.ravel()

for i, col in enumerate(target_cols):
    sns.histplot(data=train, x=col, kde=True, ax=axs[i])
    axs[i].set_title(col)

plt.tight_layout()
plt.show()



## === cell 9
lens_quest = train.question_body.str.len()
sns.histplot(lens_quest)
plt.title("Question Body Length")
plt.show()
lens_quest.min(), lens_quest.mean(), lens_quest.max(), lens_quest.std()



## === cell 10
lens_ans = train.answer.str.len()
sns.histplot(lens_ans)
plt.title("Answer Body Length")
plt.show()
lens_ans.min(), lens_ans.mean(), lens_ans.max(), lens_ans.std()



## === cell 11
train[train.question_title.str.len() < 10]



## === cell 12
train[train.question_body.str.len() < 10]



## === cell 13
train[train.answer.str.len() < 30].answer



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
train["quest"] = (
    "xxtit "
    + train["question_title"]
    + " xxbody "
    + train["question_body"]
    + " xxusr "
    + train["question_user_name"]
)
train["ans"] = "xxans " + train["answer"] + " xxusr " + train["answer_user_name"]
test["quest"] = (
    "xxtit "
    + test["question_title"]
    + " xxbody "
    + test["question_body"]
    + " xxusr "
    + test["question_user_name"]
)
test["ans"] = "xxans " + test["answer"] + " xxusr " + test["answer_user_name"]

feat_cols = ["category", "quest", "ans"]

X_train, X_test, y_train, y_test = train_test_split(
    train[feat_cols], train[target_cols], test_size=0.4, random_state=42
)
X_train.shape, X_test.shape



## === cell 17
min_df_val = int(len(train) * 0.001)

tfidvect = TfidfVectorizer(
    ngram_range=(1, 2),
    tokenizer=tokenize,
    min_df=min_df_val,
    max_df=0.9,
    sublinear_tf=True,
)
tsvd = TruncatedSVD(100)

vect = make_pipeline(tfidvect, tsvd)

preprocess = ColumnTransformer(
    [
        ("category", OneHotEncoder(dtype="int"), ["category"]),
        ("quest", vect, "quest"),
        ("ans", vect, "ans"),
    ],
    remainder="drop",
)



## === cell 18
estimator = KNeighborsRegressor(n_neighbors=901)

model = make_pipeline(preprocess, MultiOutputRegressor(estimator))
model.fit(train[feat_cols], train[target_cols])

y_pred = model.predict(test[feat_cols])

y_pred = np.clip(y_pred, 0.0, 1.0)

submission = pd.concat(
    [pd.DataFrame({"qa_id": test["qa_id"]}), pd.DataFrame(y_pred, columns=target_cols)],
    axis=1,
)
submission.to_csv("submission.csv", index=False)



## === cell 19
submission
