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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0
wordcloud==1.9.4

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

0.21221

# 6. Current score

0.26209

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.26477) has done: 'I fix the early crash by removing the problematic seaborn/matplotlib/wordcloud imports and making NLTK stopwords loading robust (download if missing, otherwise fall back to sklearn’s built-in stopword list). I also fix the seaborn `barplot` API errors by either switching to keyword arguments or skipping the plotting cells entirely so the training/inference pipeline always runs. Finally, I fix the undefined `stats` reference in the Spearman scorer (using `scipy.stats` properly) and make sure a valid `submission.csv` with the exact sample-submission columns is always written. These changes are score-neutral in intent (your current score is already within the ±10% band of the target), focusing on stability and correctness.'
- What this solution (achieved 0.26209) has done: 'We fix the early crash in the very first cell (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by avoiding importing NLTK entirely and instead using scikit-learn’s built-in English stopword list, which keeps the feature logic intact while removing the protobuf-related dependency issue in this Kaggle image. We also make `LabelEncoder` robust to unseen categories in `test` by fitting on the concatenation of train+test for `category` (as is already done for `host`), preventing runtime errors on different splits/environments without changing model logic. Finally, we keep the training/inference/submission pipeline unchanged and ensure `submission.csv` is always written with the exact sample submission columns and predictions clipped to `[0,1]`.'

# 9. Code solution

## === cell 0
import os
import gc
import string
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import make_scorer
from sklearn.feature_extraction import text as sk_text

from keras.models import Sequential
from keras.layers import Dense, Activation

from scipy.stats import spearmanr
import scipy.stats as stats

eng_stopwords = set(sk_text.ENGLISH_STOP_WORDS)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        pass  # keep silent to reduce notebook output noise



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
sample_sub_df = pd.read_csv(
    "/kaggle/input/google-quest-challenge/sample_submission.csv"
)
test_df = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")



## === cell 2
pd.set_option("display.max_columns", None)
train_df.head()



## === cell 3
test_df.head()



## === cell 4
sample_sub_df.head()



## === cell 5
print(f"Sahpe of training set: {train_df.shape}")
print(f"Sahpe of testing set: {test_df.shape}")



## === cell 6
train_df.columns



## === cell 7
total = len(train_df)



## === cell 8
_ = total



## === cell 9
_ = total



## === cell 10
_ = total



## === cell 11
_ = total



## === cell 12
_ = total



## === cell 13
target_cols = sample_sub_df.drop(["qa_id"], axis=1).columns.values
target_cols



## === cell 14
X_train = train_df.drop(np.concatenate([target_cols, np.array(["qa_id"])]), axis=1)
Y_train = train_df[target_cols]



## === cell 15
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of Y_train: {Y_train.shape}")



## === cell 16
X_train.head()



## === cell 17
X_test = test_df
del test_df
gc.collect()



## === cell 18
X_train["answer_size"] = X_train["answer"].apply(lambda x: len(str(x).split()))
X_test["answer_size"] = X_test["answer"].apply(lambda x: len(str(x).split()))

X_train["question_body_size"] = X_train["question_body"].apply(
    lambda x: len(str(x).split())
)
X_test["question_body_size"] = X_test["question_body"].apply(
    lambda x: len(str(x).split())
)

X_train["question_title_size"] = X_train["question_title"].apply(
    lambda x: len(str(x).split())
)
X_test["question_title_size"] = X_test["question_title"].apply(
    lambda x: len(str(x).split())
)

X_train["answer_num_unique_words"] = X_train["answer"].apply(
    lambda x: len(set(str(x).split()))
)
X_test["answer_num_unique_words"] = X_test["answer"].apply(
    lambda x: len(set(str(x).split()))
)

X_train["question_body_num_unique_words"] = X_train["question_body"].apply(
    lambda x: len(set(str(x).split()))
)
X_test["question_body_num_unique_words"] = X_test["question_body"].apply(
    lambda x: len(set(str(x).split()))
)

X_train["answer_num_chars"] = X_train["answer"].apply(lambda x: len(str(x)))
X_test["answer_num_chars"] = X_test["answer"].apply(lambda x: len(str(x)))

X_train["question_body_num_chars"] = X_train["question_body"].apply(
    lambda x: len(str(x))
)
X_test["question_body_num_chars"] = X_test["question_body"].apply(lambda x: len(str(x)))

X_train["answer_num_stopwords"] = X_train["answer"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
)
X_test["answer_num_stopwords"] = X_test["answer"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
)

X_train["question_body_num_stopwords"] = X_train["question_body"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
)
X_test["question_body_num_stopwords"] = X_test["question_body"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
)

X_train["answer_num_punctuations"] = X_train["answer"].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)
X_test["answer_num_punctuations"] = X_test["answer"].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)

X_train["question_body_num_punctuations"] = X_train["question_body"].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)
X_test["question_body_num_punctuations"] = X_test["question_body"].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)

X_train["answer_num_words_upper"] = X_train["answer"].apply(
    lambda x: len([w for w in str(x).split() if w.isupper()])
)
X_test["answer_num_words_upper"] = X_test["answer"].apply(
    lambda x: len([w for w in str(x).split() if w.isupper()])
)

X_train["question_body_num_words_upper"] = X_train["question_body"].apply(
    lambda x: len([w for w in str(x).split() if w.isupper()])
)
X_test["question_body_num_words_upper"] = X_test["question_body"].apply(
    lambda x: len([w for w in str(x).split() if w.isupper()])
)

X_train["answer_num_words_title"] = X_train["answer"].apply(
    lambda x: len([w for w in str(x).split() if w.istitle()])
)
X_test["answer_num_words_title"] = X_test["answer"].apply(
    lambda x: len([w for w in str(x).split() if w.istitle()])
)

X_train["question_body_num_words_title"] = X_train["question_body"].apply(
    lambda x: len([w for w in str(x).split() if w.istitle()])
)
X_test["question_body_num_words_title"] = X_test["question_body"].apply(
    lambda x: len([w for w in str(x).split() if w.istitle()])
)



## === cell 19
X_train.head()



## === cell 20
X_train = X_train.drop(
    [
        "question_user_name",
        "question_user_page",
        "answer_user_name",
        "answer_user_page",
        "url",
    ],
    axis=1,
)
X_test = X_test.drop(
    [
        "question_user_name",
        "question_user_page",
        "answer_user_name",
        "answer_user_page",
        "url",
        "qa_id",
    ],
    axis=1,
)



## === cell 21
tfv = TfidfVectorizer(
    min_df=3,
    max_features=None,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    ngram_range=(1, 3),
    use_idf=1,
    smooth_idf=1,
    sublinear_tf=1,
    stop_words="english",
)
tsvd = TruncatedSVD(n_components=150, random_state=42)

question_title = tfv.fit_transform(X_train["question_title"].values)
question_title_test = tfv.transform(X_test["question_title"].values)
question_title = tsvd.fit_transform(question_title)
question_title_test = tsvd.transform(question_title_test)

question_body = tfv.fit_transform(X_train["question_body"].values)
question_body_test = tfv.transform(X_test["question_body"].values)
question_body = tsvd.fit_transform(question_body)
question_body_test = tsvd.transform(question_body_test)

answer = tfv.fit_transform(X_train["answer"].values)
answer_test = tfv.transform(X_test["answer"].values)
answer = tsvd.fit_transform(answer)
answer_test = tsvd.transform(answer_test)



## === cell 22
cat_le = LabelEncoder()
cat_le.fit(pd.concat([X_train["category"], X_test["category"]], ignore_index=True))
category = cat_le.transform(X_train["category"])
category_test = cat_le.transform(X_test["category"])



## === cell 23
host_le = LabelEncoder()
host_le.fit(pd.concat([X_train["host"], X_test["host"]], ignore_index=True))
host = host_le.transform(X_train["host"])
host_test = host_le.transform(X_test["host"])



## === cell 24
meta_features_train = X_train.drop(
    ["question_title", "question_body", "answer", "category", "host"], axis=1
).to_numpy()
meta_features_test = X_test.drop(
    ["question_title", "question_body", "answer", "category", "host"], axis=1
).to_numpy()



## === cell 25
X_train = np.concatenate([question_title, question_body, answer], axis=1)
X_test = np.concatenate([question_title_test, question_body_test, answer_test], axis=1)



## === cell 26
X_train = np.column_stack((X_train, category, host, meta_features_train))
X_test = np.column_stack((X_test, category_test, host_test, meta_features_test))



## === cell 27
print(X_train.shape)
print(X_test.shape)



## === cell 28
del (
    question_title,
    question_title_test,
    answer,
    answer_test,
    question_body,
    question_body_test,
)
gc.collect()




## === cell 29
def spearman_corr(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if np.ndim(y_pred) == 2:
        corr = np.mean(
            [
                stats.spearmanr(y_true[:, i], y_pred[:, i]).correlation
                for i in range(y_true.shape[1])
            ]
        )
    else:
        corr = stats.spearmanr(y_true, y_pred).correlation
    return corr


custom_scorer = make_scorer(spearman_corr, greater_is_better=True)



## === cell 30
np.isnan(X_train).any()



## === cell 31
Y_train.shape



## === cell 32
model = Sequential(
    [
        Dense(256, input_shape=(X_train.shape[1],)),
        Activation("relu"),
        Dense(128),
        Activation("relu"),
        Dense(128),
        Activation("relu"),
        Dense(len(target_cols)),
        Activation("sigmoid"),
    ]
)
model.compile(optimizer="adam", loss="binary_crossentropy")



## === cell 33
model.fit(X_train, Y_train, epochs=25, verbose=1)



## === cell 34
preds = model.predict(X_train, verbose=0)



## === cell 35
overall_score = 0.0
for col_index, col in enumerate(target_cols):
    overall_score += spearmanr(
        preds[:, col_index], Y_train[col].values
    ).correlation / len(target_cols)



## === cell 36
overall_score



## === cell 37
preds = model.predict(X_test, verbose=0)



## === cell 38
preds.shape



## === cell 39
preds = np.clip(preds, 0.0, 1.0)
for col_index, col in enumerate(target_cols):
    sample_sub_df[col] = preds[:, col_index]



## === cell 40
sample_sub_df = sample_sub_df[["qa_id"] + list(target_cols)]
sample_sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_sub_df.shape)
