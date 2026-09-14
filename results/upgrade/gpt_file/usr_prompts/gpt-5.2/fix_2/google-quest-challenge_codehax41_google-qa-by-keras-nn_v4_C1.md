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

gensim==4.4.0
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
tqdm==4.67.1

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

0.28727

# 6. Current score

0.35553

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.35553) has done: 'I fix the import/runtime failures by removing the problematic `keras`/protobuf stack that triggers `MessageFactory.GetPrototype` and by using `tf_keras` consistently for the same Sequential/Dense model you already defined. I also restore missing imports (`MaxAbsScaler`, `tqdm`, `gensim`) and ensure NLTK Brown is available without internet by falling back to the local NLTK corpus (or, if missing, a deterministic tiny corpus) so Word2Vec can be trained. Finally, I fix small data issues that can crash training (NaNs in engineered counts) and guarantee a valid `submission.csv` with predictions clipped to `[0,1]` and correct column order.'

# 9. Code solution

## === cell 0
import os
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.notebook_repr_html", True)

np.random.seed(42)



## === cell 1
import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Activation
from tf_keras.callbacks import EarlyStopping

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MaxAbsScaler
from scipy.stats import spearmanr

import tqdm
import gensim
from gensim.models import Word2Vec

import nltk

gc.collect()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 3
train = pd.read_csv("../input/google-quest-challenge/train.csv")
test = pd.read_csv("../input/google-quest-challenge/test.csv")
df = pd.concat([train, test], axis=0, ignore_index=True)
df.head()



## === cell 4
df.shape



## === cell 5
df.dtypes



## === cell 6
df.isnull().sum()



## === cell 7
target_cols = [
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]



## === cell 8
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
width = 0.4
df.host.value_counts().plot(kind="bar", color="blue", ax=ax, width=width, position=1)
ax.set_xlabel("Sites")
ax.set_ylabel("Question Counts")
plt.show()



## === cell 9
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111)
width = 0.2
df.category.value_counts().plot(
    kind="bar", color="green", ax=ax, width=width, position=1, legend=True
)
ax.set_xlabel("Sites")
ax.set_ylabel("Question Counts")
plt.show()




## === cell 10
def char_count(s):
    if pd.isna(s):
        return 0
    return len(s)


def word_count(s):
    if pd.isna(s):
        return 0
    return s.count(" ")


for _df in (train, test):
    _df["question_title_n_chars"] = _df["question_title"].apply(char_count)
    _df["question_title_n_words"] = _df["question_title"].apply(word_count)
    _df["question_body_n_chars"] = _df["question_body"].apply(char_count)
    _df["question_body_n_words"] = _df["question_body"].apply(word_count)
    _df["answer_n_chars"] = _df["answer"].apply(char_count)
    _df["answer_n_words"] = _df["answer"].apply(word_count)



## === cell 11
train["question_body_n_chars"].clip(0, 5000, inplace=True)
test["question_body_n_chars"].clip(0, 5000, inplace=True)
train["question_body_n_words"].clip(0, 1000, inplace=True)
test["question_body_n_words"].clip(0, 1000, inplace=True)

train["answer_n_chars"].clip(0, 5000, inplace=True)
test["answer_n_chars"].clip(0, 5000, inplace=True)
train["answer_n_words"].clip(0, 1000, inplace=True)
test["answer_n_words"].clip(0, 1000, inplace=True)



## === cell 12
num_question = train["question_user_name"].value_counts()
num_answer = train["answer_user_name"].value_counts()



## === cell 13
train["num_answer_user"] = train["answer_user_name"].map(num_answer)
train["num_question_user"] = train["question_user_name"].map(num_question)
test["num_answer_user"] = test["answer_user_name"].map(num_answer)
test["num_question_user"] = test["question_user_name"].map(num_question)

train["num_answer_user"].fillna(1, inplace=True)
train["num_question_user"].fillna(1, inplace=True)
test["num_answer_user"].fillna(1, inplace=True)
test["num_question_user"].fillna(1, inplace=True)

simple_feature_cols = [
    "question_title_n_chars",
    "question_title_n_words",
    "question_body_n_chars",
    "question_body_n_words",
    "answer_n_chars",
    "answer_n_words",
    "num_answer_user",
    "num_question_user",
]
simple_engineered_feature = train[simple_feature_cols].values
simple_engineered_feature_test = test[simple_feature_cols].values



## === cell 14
scaler = MaxAbsScaler()
simple_engineered_feature = scaler.fit_transform(simple_engineered_feature)
simple_engineered_feature_test = scaler.transform(simple_engineered_feature_test)




## === cell 15
def simple_prepro(s):
    if pd.isna(s):
        s = ""
    return [
        w
        for w in s.replace("\n", " ")
        .replace(",", " , ")
        .replace("(", " ( ")
        .replace(")", " ) ")
        .replace(".", " . ")
        .replace("?", " ? ")
        .replace(":", " : ")
        .replace("n't", " not")
        .replace("'ve", " have")
        .replace("'re", " are")
        .replace("'s", " is")
        .split(" ")
        if w != ""
    ]


def simple_prepro_tfidf(s):
    if pd.isna(s):
        s = ""
    return " ".join(
        [
            w
            for w in s.lower()
            .replace("\n", " ")
            .replace(",", " , ")
            .replace("(", " ( ")
            .replace(")", " ) ")
            .replace(".", " . ")
            .replace("?", " ? ")
            .replace(":", " : ")
            .replace("n't", " not")
            .replace("'ve", " have")
            .replace("'re", " are")
            .replace("'s", " is")
            .split(" ")
            if w != ""
        ]
    )




## === cell 16
qt_max = max([len(simple_prepro(l)) for l in list(train["question_title"].values)])
qb_max = max([len(simple_prepro(l)) for l in list(train["question_body"].values)])
an_max = max([len(simple_prepro(l)) for l in list(train["answer"].values)])
print("max lenght of question_title is", qt_max)
print("max lenght of question_body is", qb_max)
print("max lenght of question_answer is", an_max)



## === cell 17
try:
    nltk.data.find("corpora/brown")
except LookupError:
    try:
        nltk.download("brown", quiet=True)
    except Exception:
        pass

try:
    from nltk.corpus import brown

    brown_sents = list(brown.sents())
    if len(brown_sents) == 0:
        raise LookupError("Brown corpus empty")
except Exception:
    brown_sents = [
        ["this", "is", "a", "sentence"],
        ["another", "example", "sentence"],
        ["stackexchange", "question", "answer", "text"],
        ["machine", "learning", "model", "training"],
    ]

w2v_model = Word2Vec(
    sentences=brown_sents, vector_size=100, window=5, min_count=1, workers=1
)


def get_word_embeddings(text):
    np.random.seed(abs(hash(text)) % (10**8))
    words = simple_prepro(text)
    vectors = np.zeros((len(words), 100), dtype=np.float32)
    if len(words) == 0:
        vectors = np.zeros((1, 100), dtype=np.float32)
    for i, word in enumerate(words):
        try:
            vectors[i] = w2v_model.wv[word]
        except Exception:
            vectors[i] = np.random.uniform(-0.01, 0.01, 100).astype(np.float32)
    return np.max(np.array(vectors), axis=0)




## === cell 18
question_title = np.vstack(
    [get_word_embeddings(l) for l in tqdm.tqdm(train["question_title"].values)]
)
question_title_test = np.vstack(
    [get_word_embeddings(l) for l in tqdm.tqdm(test["question_title"].values)]
)

question_body = np.vstack(
    [get_word_embeddings(l) for l in tqdm.tqdm(train["question_body"].values)]
)
question_body_test = np.vstack(
    [get_word_embeddings(l) for l in tqdm.tqdm(test["question_body"].values)]
)

answer = np.vstack([get_word_embeddings(l) for l in tqdm.tqdm(train["answer"].values)])
answer_test = np.vstack(
    [get_word_embeddings(l) for l in tqdm.tqdm(test["answer"].values)]
)



## === cell 19
tfidf = TfidfVectorizer(ngram_range=(1, 3))
tsvd = TruncatedSVD(n_components=50, random_state=42)

tfidf_question_title = tfidf.fit_transform(
    [simple_prepro_tfidf(l) for l in tqdm.tqdm(train["question_title"].values)]
)
tfidf_question_title_test = tfidf.transform(
    [simple_prepro_tfidf(l) for l in tqdm.tqdm(test["question_title"].values)]
)
tfidf_question_title = tsvd.fit_transform(tfidf_question_title)
tfidf_question_title_test = tsvd.transform(tfidf_question_title_test)

tfidf_question_body = tfidf.fit_transform(
    [simple_prepro_tfidf(l) for l in tqdm.tqdm(train["question_body"].values)]
)
tfidf_question_body_test = tfidf.transform(
    [simple_prepro_tfidf(l) for l in tqdm.tqdm(test["question_body"].values)]
)
tfidf_question_body = tsvd.fit_transform(tfidf_question_body)
tfidf_question_body_test = tsvd.transform(tfidf_question_body_test)

tfidf_answer = tfidf.fit_transform(
    [simple_prepro_tfidf(l) for l in tqdm.tqdm(train["answer"].values)]
)
tfidf_answer_test = tfidf.transform(
    [simple_prepro_tfidf(l) for l in tqdm.tqdm(test["answer"].values)]
)
tfidf_answer = tsvd.fit_transform(tfidf_answer)
tfidf_answer_test = tsvd.transform(tfidf_answer_test)



## === cell 20
categories = sorted(train["category"].dropna().unique().tolist())
type2int = {t: i for i, t in enumerate(categories)}
n_cats = len(categories)


def onehot_category(series):
    idx = series.map(type2int).fillna(0).astype(int).values
    eye = np.identity(n_cats, dtype=np.float32)
    return eye[idx]


cate = onehot_category(train["category"])
cate_test = onehot_category(test["category"])

train_features = np.concatenate(
    [
        question_title,
        question_body,
        answer,
        tfidf_question_title,
        tfidf_question_body,
        tfidf_answer,
        cate,
        simple_engineered_feature,
    ],
    axis=1,
).astype(np.float32)

test_features = np.concatenate(
    [
        question_title_test,
        question_body_test,
        answer_test,
        tfidf_question_title_test,
        tfidf_question_body_test,
        tfidf_answer_test,
        cate_test,
        simple_engineered_feature_test,
    ],
    axis=1,
).astype(np.float32)

train_features.shape, test_features.shape



## === cell 21
num_folds = 10
fold_scores = []
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
test_preds = np.zeros((len(test_features), len(target_cols)), dtype=np.float32)

y_all = train[target_cols].astype(np.float32)

for fold, (train_index, val_index) in enumerate(kf.split(train_features), start=1):
    gc.collect()
    train_X = train_features[train_index, :]
    train_y = y_all.iloc[train_index].values

    val_X = train_features[val_index, :]
    val_y = y_all.iloc[val_index].values

    model = Sequential(
        [
            Dense(512, input_shape=(train_features.shape[1],)),
            Activation("relu"),
            Dense(256),
            Activation("relu"),
            Dense(len(target_cols)),
            Activation("sigmoid"),
        ]
    )

    es = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=10,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(
        train_X,
        train_y,
        epochs=100,
        validation_data=(val_X, val_y),
        callbacks=[es],
        verbose=0,
    )

    preds = model.predict(val_X, verbose=0)
    overall_score = 0.0
    for col_index, col in enumerate(target_cols):
        corr = spearmanr(preds[:, col_index], val_y[:, col_index]).correlation
        if corr is None or np.isnan(corr):
            corr = 0.0
        overall_score += corr / len(target_cols)
    fold_scores.append(overall_score)
    print(f"Fold {fold} mean Spearman: {overall_score:.6f}")

    test_preds += model.predict(test_features, verbose=0).astype(np.float32) / num_folds

print("Fold scores:", fold_scores)
print("CV mean:", float(np.mean(fold_scores)))



## === cell 22
sub = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
test_preds = np.clip(test_preds, 0.0, 1.0)

for col_index, col in enumerate(target_cols):
    sub[col] = test_preds[:, col_index]

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
