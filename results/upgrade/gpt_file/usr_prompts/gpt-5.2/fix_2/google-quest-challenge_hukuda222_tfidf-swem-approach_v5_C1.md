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

0.2938

# 6. Current score

0.36545

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.36545) has done: 'I fix the import/runtime issues that prevent the notebook from executing (the keras EarlyStopping import, `tqdm` usage, and missing `TfidfVectorizer/TruncatedSVD` due to the first-cell crash). I also make the NLP dependencies robust in Kaggle (Brown corpus may be missing), by falling back to training Word2Vec on the competition text itself if Brown isn’t available—this preserves the same feature approach (Word2Vec max-pooling + extra counts). Finally, I harden preprocessing against NaNs and ensure category one-hot encoding works even if train/test categories differ, and I guarantee a valid `submission.csv` is written with the exact sample submission columns and predictions clipped to [0,1].'

# 9. Code solution

## === cell 0
import os
import gc
import random
import numpy as np
import pandas as pd

import gensim
from gensim.models import Word2Vec

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from scipy.stats import spearmanr

from tqdm import tqdm

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping

try:
    from nltk.corpus import brown
except Exception:
    brown = None

np.random.seed(42)
random.seed(42)



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
def simple_prepro(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        s = ""
    s = str(s)
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




## === cell 7
def simple_prepro_tfidf(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        s = ""
    s = str(s)
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




## === cell 8
qt_max = max([len(simple_prepro(l)) for l in list(train["question_title"].values)])
qb_max = max([len(simple_prepro(l)) for l in list(train["question_body"].values)])
an_max = max([len(simple_prepro(l)) for l in list(train["answer"].values)])
print("max lenght of question_title is", qt_max)
print("max lenght of question_body is", qb_max)
print("max lenght of question_answer is", an_max)



## === cell 9
np.max([str(l).count("!") for l in list(train["answer"].values)])



## === cell 10
sentences = None
if brown is not None:
    try:
        sentences = list(brown.sents())
    except Exception:
        sentences = None

if sentences is None:
    all_text = pd.concat(
        [
            train[["question_title", "question_body", "answer"]],
            test[["question_title", "question_body", "answer"]],
        ],
        axis=0,
        ignore_index=True,
    )
    sentences = []
    for col in ["question_title", "question_body", "answer"]:
        sentences.extend([simple_prepro(x) for x in all_text[col].values])

w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,
    epochs=5,
    seed=42,
)




## === cell 11
def get_word_embeddings(text):
    if text is None or (isinstance(text, float) and np.isnan(text)):
        text = ""
    text = str(text)

    np.random.seed(abs(hash(text)) % (10**8))
    words = simple_prepro(text)

    vectors = np.zeros((len(words), 100), dtype=np.float32)
    if len(words) == 0:
        vectors = np.zeros((1, 100), dtype=np.float32)

    for i, word in enumerate(words):
        try:
            vectors[i] = w2v_model.wv[word]
        except Exception:
            vectors[i] = np.random.uniform(-0.01, 0.01, 100)

    return np.concatenate(
        [
            np.max(np.array(vectors), axis=0),
            np.array(
                [
                    min(len(text), 5000) / 5000,
                    min(len(words), 5000) / 5000,
                    min(text.count("\n"), 100) / 100,
                    min(text.count("?"), 20) / 20,
                    min(text.count("!"), 20) / 20,
                ],
                dtype=np.float32,
            ),
        ]
    )




## === cell 12
question_title = [
    get_word_embeddings(l)
    for l in tqdm(train["question_title"].values, desc="w2v title train")
]
question_title_test = [
    get_word_embeddings(l)
    for l in tqdm(test["question_title"].values, desc="w2v title test")
]

question_body = [
    get_word_embeddings(l)
    for l in tqdm(train["question_body"].values, desc="w2v body train")
]
question_body_test = [
    get_word_embeddings(l)
    for l in tqdm(test["question_body"].values, desc="w2v body test")
]

answer = [
    get_word_embeddings(l)
    for l in tqdm(train["answer"].values, desc="w2v answer train")
]
answer_test = [
    get_word_embeddings(l) for l in tqdm(test["answer"].values, desc="w2v answer test")
]

question_title = np.asarray(question_title, dtype=np.float32)
question_title_test = np.asarray(question_title_test, dtype=np.float32)
question_body = np.asarray(question_body, dtype=np.float32)
question_body_test = np.asarray(question_body_test, dtype=np.float32)
answer = np.asarray(answer, dtype=np.float32)
answer_test = np.asarray(answer_test, dtype=np.float32)



## === cell 13
gc.collect()


def tfidf_svd_fit_transform(train_texts, test_texts, n_components=50):
    tfidf = TfidfVectorizer(ngram_range=(1, 3))
    Xtr = tfidf.fit_transform([simple_prepro_tfidf(x) for x in train_texts])
    Xte = tfidf.transform([simple_prepro_tfidf(x) for x in test_texts])

    n_comp = min(n_components, max(2, Xtr.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Xtr_s = svd.fit_transform(Xtr).astype(np.float32)
    Xte_s = svd.transform(Xte).astype(np.float32)
    return Xtr_s, Xte_s


tfidf_question_title, tfidf_question_title_test = tfidf_svd_fit_transform(
    train["question_title"].values, test["question_title"].values, n_components=50
)
tfidf_question_body, tfidf_question_body_test = tfidf_svd_fit_transform(
    train["question_body"].values, test["question_body"].values, n_components=50
)
tfidf_answer, tfidf_answer_test = tfidf_svd_fit_transform(
    train["answer"].values, test["answer"].values, n_components=50
)



## === cell 14
all_cats = (
    pd.concat([train["category"], test["category"]], axis=0)
    .astype(str)
    .unique()
    .tolist()
)
type2int = {t: i for i, t in enumerate(all_cats)}
n_cats = len(all_cats)

cate = np.identity(n_cats, dtype=np.float32)[
    train["category"].astype(str).map(type2int).values
]
cate_test = np.identity(n_cats, dtype=np.float32)[
    test["category"].astype(str).map(type2int).values
]



## === cell 15
train_features = np.concatenate(
    [
        question_title,
        question_body,
        answer,
        tfidf_question_title,
        tfidf_question_body,
        tfidf_answer,
        cate,
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
    ],
    axis=1,
).astype(np.float32)

print("train_features shape:", train_features.shape)
print("test_features shape:", test_features.shape)



## === cell 16
num_folds = 10
fold_scores = []
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

test_preds = np.zeros((len(test_features), len(target_cols)), dtype=np.float32)

for fold, (train_index, val_index) in enumerate(kf.split(train_features), 1):
    gc.collect()

    train_X = train_features[train_index, :]
    train_y = train[target_cols].iloc[train_index].values.astype(np.float32)

    val_X = train_features[val_index, :]
    val_y_df = train[target_cols].iloc[val_index]
    val_y = val_y_df.values.astype(np.float32)

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
        corr = spearmanr(preds[:, col_index], val_y_df[col].values).correlation
        if corr is None or np.isnan(corr):
            corr = 0.0
        overall_score += corr / len(target_cols)
    fold_scores.append(overall_score)
    print(f"Fold {fold}/{num_folds} Spearman mean:", overall_score)

    test_preds += model.predict(test_features, verbose=0).astype(np.float32) / num_folds

print("CV fold scores:", fold_scores)
print("CV mean:", float(np.mean(fold_scores)))



## === cell 17
test_preds = np.clip(test_preds, 0.0, 1.0)

sub = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
for col_index, col in enumerate(target_cols):
    sub[col] = test_preds[:, col_index]

sub["qa_id"] = pd.read_csv("../input/google-quest-challenge/test.csv")["qa_id"].values

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
