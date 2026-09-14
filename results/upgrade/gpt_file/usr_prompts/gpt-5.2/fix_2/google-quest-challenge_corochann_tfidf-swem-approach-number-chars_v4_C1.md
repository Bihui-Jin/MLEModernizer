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
lightgbm==4.6.0
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

0.29138

# 6. Current score

0.34983

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.34983) has done: 'I fix the import/runtime issues that currently prevent the notebook from running (notably the protobuf/keras import crash, missing `tqdm`, missing `TfidfVectorizer`, and missing `seaborn`). I also make the feature-building cells robust to missing NLTK Brown corpus by falling back to a lightweight local Word2Vec training on the competition text (this keeps the same embedding approach but prevents download errors). Finally, I correct the one-hot encoding of `category` to use the actual number of unique categories (it currently assumes 5 and can crash or silently mis-shape), and ensure the script always writes a valid `submission.csv` with the required columns and [0,1] predictions.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import gensim
from gensim.models import Word2Vec

import tqdm

from sklearn.model_selection import KFold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from scipy.stats import spearmanr

import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Activation
from tf_keras.callbacks import EarlyStopping

import seaborn as sns

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_csv("../input/google-quest-challenge/train.csv")
test = pd.read_csv("../input/google-quest-challenge/test.csv")



## === cell 2
sample_sub = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")



## === cell 3
sample_sub



## === cell 4
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



## === cell 5
train




## === cell 6
def char_count(s):
    s = "" if pd.isna(s) else str(s)
    return len(s)


def word_count(s):
    s = "" if pd.isna(s) else str(s)
    return s.count(" ")


train["question_title_n_chars"] = train["question_title"].apply(char_count)
train["question_title_n_words"] = train["question_title"].apply(word_count)
train["question_body_n_chars"] = train["question_body"].apply(char_count)
train["question_body_n_words"] = train["question_body"].apply(word_count)
train["answer_n_chars"] = train["answer"].apply(char_count)
train["answer_n_words"] = train["answer"].apply(word_count)

test["question_title_n_chars"] = test["question_title"].apply(char_count)
test["question_title_n_words"] = test["question_title"].apply(word_count)
test["question_body_n_chars"] = test["question_body"].apply(char_count)
test["question_body_n_words"] = test["question_body"].apply(word_count)
test["answer_n_chars"] = test["answer"].apply(char_count)
test["answer_n_words"] = test["answer"].apply(word_count)

train["question_body_n_chars"] = train["question_body_n_chars"].clip(0, 5000)
test["question_body_n_chars"] = test["question_body_n_chars"].clip(0, 5000)
train["question_body_n_words"] = train["question_body_n_words"].clip(0, 1000)
test["question_body_n_words"] = test["question_body_n_words"].clip(0, 1000)

train["answer_n_chars"] = train["answer_n_chars"].clip(0, 5000)
test["answer_n_chars"] = test["answer_n_chars"].clip(0, 5000)
train["answer_n_words"] = train["answer_n_words"].clip(0, 1000)
test["answer_n_words"] = test["answer_n_words"].clip(0, 1000)



## === cell 7
num_question = train["question_user_name"].value_counts()
num_answer = train["answer_user_name"].value_counts()

train["num_answer_user"] = train["answer_user_name"].map(num_answer)
train["num_question_user"] = train["question_user_name"].map(num_question)
test["num_answer_user"] = test["answer_user_name"].map(num_answer)
test["num_question_user"] = test["question_user_name"].map(num_question)

test["num_answer_user"] = test["num_answer_user"].fillna(1)
test["num_question_user"] = test["num_question_user"].fillna(1)

train["num_answer_user"] = train["num_answer_user"].fillna(1)
train["num_question_user"] = train["num_question_user"].fillna(1)



## === cell 8
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



## === cell 9
scaler = MaxAbsScaler()
simple_engineered_feature = scaler.fit_transform(simple_engineered_feature)
simple_engineered_feature_test = scaler.transform(simple_engineered_feature_test)




## === cell 10
def simple_prepro(s):
    s = "" if pd.isna(s) else str(s)
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




## === cell 11
def simple_prepro_tfidf(s):
    s = "" if pd.isna(s) else str(s)
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




## === cell 12
qt_max = max([len(simple_prepro(l)) for l in list(train["question_title"].values)])
qb_max = max([len(simple_prepro(l)) for l in list(train["question_body"].values)])
an_max = max([len(simple_prepro(l)) for l in list(train["answer"].values)])
print("max lenght of question_title is", qt_max)
print("max lenght of question_body is", qb_max)
print("max lenght of question_answer is", an_max)




## === cell 13
def _build_w2v_corpus(df):
    corpus = []
    for col in ["question_title", "question_body", "answer"]:
        corpus.extend([simple_prepro(x) for x in df[col].astype(str).values])
    corpus = [sent for sent in corpus if len(sent) > 0]
    return corpus


try:
    from nltk.corpus import brown

    _ = brown.sents()  # may raise LookupError
    w2v_model = Word2Vec(
        sentences=brown.sents(),
        vector_size=100,
        window=5,
        min_count=1,
        workers=2,
        seed=SEED,
        epochs=5,
    )
    print("Word2Vec trained on NLTK Brown corpus.")
except Exception as e:
    print(
        "NLTK Brown not available; training Word2Vec on competition text instead. Reason:",
        repr(e),
    )
    corpus = _build_w2v_corpus(train)
    w2v_model = Word2Vec(
        sentences=corpus,
        vector_size=100,
        window=5,
        min_count=1,
        workers=2,
        seed=SEED,
        epochs=5,
    )
    del corpus
    gc.collect()




## === cell 14
def get_word_embeddings(text):
    np.random.seed(abs(hash(str(text))) % (10**8))
    words = simple_prepro(text)
    vectors = np.zeros((len(words), 100), dtype=np.float32)
    if len(words) == 0:
        vectors = np.zeros((1, 100), dtype=np.float32)
    for i, word in enumerate(words):
        try:
            vectors[i] = w2v_model.wv[word]
        except Exception:
            vectors[i] = np.random.uniform(-0.01, 0.01, 100).astype(np.float32)
    return np.max(np.asarray(vectors), axis=0)




## === cell 15
question_title = [
    get_word_embeddings(l)
    for l in tqdm.tqdm(train["question_title"].values, desc="w2v qt")
]
question_title_test = [
    get_word_embeddings(l)
    for l in tqdm.tqdm(test["question_title"].values, desc="w2v qt test")
]

question_body = [
    get_word_embeddings(l)
    for l in tqdm.tqdm(train["question_body"].values, desc="w2v qb")
]
question_body_test = [
    get_word_embeddings(l)
    for l in tqdm.tqdm(test["question_body"].values, desc="w2v qb test")
]

answer = [
    get_word_embeddings(l) for l in tqdm.tqdm(train["answer"].values, desc="w2v ans")
]
answer_test = [
    get_word_embeddings(l)
    for l in tqdm.tqdm(test["answer"].values, desc="w2v ans test")
]

question_title = np.asarray(question_title)
question_title_test = np.asarray(question_title_test)
question_body = np.asarray(question_body)
question_body_test = np.asarray(question_body_test)
answer = np.asarray(answer)
answer_test = np.asarray(answer_test)



## === cell 16
gc.collect()

tfidf = TfidfVectorizer(ngram_range=(1, 3))
tsvd = TruncatedSVD(n_components=50, random_state=SEED)

tfidf_question_title = tfidf.fit_transform(
    [
        simple_prepro_tfidf(l)
        for l in tqdm.tqdm(train["question_title"].values, desc="tfidf qt")
    ]
)
tfidf_question_title_test = tfidf.transform(
    [
        simple_prepro_tfidf(l)
        for l in tqdm.tqdm(test["question_title"].values, desc="tfidf qt test")
    ]
)
tfidf_question_title = tsvd.fit_transform(tfidf_question_title)
tfidf_question_title_test = tsvd.transform(tfidf_question_title_test)

tfidf_question_body = tfidf.fit_transform(
    [
        simple_prepro_tfidf(l)
        for l in tqdm.tqdm(train["question_body"].values, desc="tfidf qb")
    ]
)
tfidf_question_body_test = tfidf.transform(
    [
        simple_prepro_tfidf(l)
        for l in tqdm.tqdm(test["question_body"].values, desc="tfidf qb test")
    ]
)
tfidf_question_body = tsvd.fit_transform(tfidf_question_body)
tfidf_question_body_test = tsvd.transform(tfidf_question_body_test)

tfidf_answer = tfidf.fit_transform(
    [
        simple_prepro_tfidf(l)
        for l in tqdm.tqdm(train["answer"].values, desc="tfidf ans")
    ]
)
tfidf_answer_test = tfidf.transform(
    [
        simple_prepro_tfidf(l)
        for l in tqdm.tqdm(test["answer"].values, desc="tfidf ans test")
    ]
)
tfidf_answer = tsvd.fit_transform(tfidf_answer)
tfidf_answer_test = tsvd.transform(tfidf_answer_test)



## === cell 17
cats = sorted(train["category"].astype(str).unique().tolist())
type2int = {t: i for i, t in enumerate(cats)}
n_cats = len(cats)

train_cat_idx = train["category"].astype(str).map(type2int).values
test_cat_idx = test["category"].astype(str).map(type2int).fillna(0).astype(int).values

cate = np.eye(n_cats, dtype=np.float32)[train_cat_idx.astype(int)]
cate_test = np.eye(n_cats, dtype=np.float32)[test_cat_idx.astype(int)]



## === cell 18
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
)
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
)

print("train_features shape:", train_features.shape)
print("test_features shape:", test_features.shape)



## === cell 19
num_folds = 10
fold_scores = []
kf = KFold(n_splits=num_folds, shuffle=True, random_state=SEED)

test_preds = np.zeros((len(test_features), len(target_cols)), dtype=np.float32)

for fold, (train_index, val_index) in enumerate(kf.split(train_features), start=1):
    gc.collect()
    train_X = train_features[train_index, :]
    train_y = train[target_cols].iloc[train_index].values.astype(np.float32)

    val_X = train_features[val_index, :]
    val_y = train[target_cols].iloc[val_index].values.astype(np.float32)

    model = Sequential(
        [
            Dense(512, input_shape=(train_features.shape[1],)),
            Activation("relu"),
            Dense(128),
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
    print(f"Fold {fold}/{num_folds} mean spearman:", overall_score)

    test_preds += model.predict(test_features, verbose=0).astype(np.float32) / num_folds

print("CV scores:", fold_scores)
print("CV mean:", float(np.mean(fold_scores)))



## === cell 20
sub = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")

test_preds = np.clip(test_preds, 0.0, 1.0)

for col_index, col in enumerate(target_cols):
    sub[col] = test_preds[:, col_index]

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)



## === cell 21
test_preds



## === cell 22
sub.isna().sum()




## === cell 23
def plot_label_distribution(df):
    fig, axes = plt.subplots(6, 5, figsize=(18, 15))
    axes = axes.ravel()
    bins = np.linspace(0, 1, 20)

    for i, col in enumerate(target_cols):
        ax = axes[i]
        sns.histplot(df[col], label=col, bins=bins, ax=ax, stat="count")
        ax.set_xlim([0, 1])
    plt.tight_layout()
    plt.show()
    plt.close()




## === cell 24
fig, axes = plt.subplots(6, 5, figsize=(18, 15))
axes = axes.ravel()
bins = np.linspace(0, 1, 20)

for i, col in enumerate(target_cols):
    ax = axes[i]
    sns.histplot(train[col], bins=bins, ax=ax, color="blue", stat="density", alpha=0.5)
    sns.histplot(sub[col], bins=bins, ax=ax, color="orange", stat="density", alpha=0.5)
    ax.set_xlim([0, 1])
plt.tight_layout()
plt.show()
plt.close()
