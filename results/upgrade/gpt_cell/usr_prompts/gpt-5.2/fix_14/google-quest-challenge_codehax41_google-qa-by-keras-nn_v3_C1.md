# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from sklearn.preprocessing import MaxAbsScaler


import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.notebook_repr_html", True)


## === cell 1
import gc
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
gc.collect()


## === cell 2

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 3
train = pd.read_csv("../input/google-quest-challenge/train.csv")
test = pd.read_csv("../input/google-quest-challenge/test.csv")
df = pd.concat([train, test])
df.head()


## === cell 4
df.shape


## === cell 5
df.dtypes


## === cell 6
df.isnull().sum()


## === cell 7
target_cols = ['question_asker_intent_understanding',
       'question_body_critical', 'question_conversational',
       'question_expect_short_answer', 'question_fact_seeking',
       'question_has_commonly_accepted_answer',
       'question_interestingness_others', 'question_interestingness_self',
       'question_multi_intent', 'question_not_really_a_question',
       'question_opinion_seeking', 'question_type_choice',
       'question_type_compare', 'question_type_consequence',
       'question_type_definition', 'question_type_entity',
       'question_type_instructions', 'question_type_procedure',
       'question_type_reason_explanation', 'question_type_spelling',
       'question_well_written', 'answer_helpful',
       'answer_level_of_information', 'answer_plausible', 'answer_relevance',
       'answer_satisfaction', 'answer_type_instructions',
       'answer_type_procedure', 'answer_type_reason_explanation',
       'answer_well_written']


## === cell 8
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
width = 0.4
df.host.value_counts().plot(kind='bar', color='blue', ax=ax, width=width, position=1)
ax.set_xlabel('Sites')
ax.set_ylabel('Question Counts')


## === cell 9
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111)
width = 0.2
df.category.value_counts().plot(kind='bar', color='green', ax=ax, width=width, position=1, legend=True)
ax.set_xlabel('Sites')
ax.set_ylabel('Question Counts')


## === cell 10
def char_count(s):
    return len(s)

def word_count(s):
    return s.count(' ')

train['question_title_n_chars'] = train['question_title'].apply(char_count)
train['question_title_n_words'] = train['question_title'].apply(word_count)
train['question_body_n_chars'] = train['question_body'].apply(char_count)
train['question_body_n_words'] = train['question_body'].apply(word_count)
train['answer_n_chars'] = train['answer'].apply(char_count)
train['answer_n_words'] = train['answer'].apply(word_count)

test['question_title_n_chars'] = test['question_title'].apply(char_count)
test['question_title_n_words'] = test['question_title'].apply(word_count)
test['question_body_n_chars'] = test['question_body'].apply(char_count)
test['question_body_n_words'] = test['question_body'].apply(word_count)
test['answer_n_chars'] = test['answer'].apply(char_count)
test['answer_n_words'] = test['answer'].apply(word_count)


## === cell 11
train['question_body_n_chars'].clip(0, 5000, inplace=True)
test['question_body_n_chars'].clip(0, 5000, inplace=True)
train['question_body_n_words'].clip(0, 1000, inplace=True)
test['question_body_n_words'].clip(0, 1000, inplace=True)

train['answer_n_chars'].clip(0, 5000, inplace=True)
test['answer_n_chars'].clip(0, 5000, inplace=True)
train['answer_n_words'].clip(0, 1000, inplace=True)
test['answer_n_words'].clip(0, 1000, inplace=True)


## === cell 12
num_question = train['question_user_name'].value_counts()
num_answer = train['answer_user_name'].value_counts()


## === cell 13
train['num_answer_user'] = train['answer_user_name'].map(num_answer)
train['num_question_user'] = train['question_user_name'].map(num_question)
test['num_answer_user'] = test['answer_user_name'].map(num_answer)
test['num_question_user'] = test['question_user_name'].map(num_question)


## === cell 14
test['num_answer_user'].fillna(1, inplace=True)
test['num_question_user'].fillna(1, inplace=True)

simple_feature_cols = [
    'question_title_n_chars', 'question_title_n_words', 'question_body_n_chars', 'question_body_n_words',
    'answer_n_chars', 'answer_n_words', 'num_answer_user', 'num_question_user'
]
simple_engineered_feature = train[simple_feature_cols].values
simple_engineered_feature_test = test[simple_feature_cols].values


## === cell 15
scaler = MaxAbsScaler()
simple_engineered_feature = scaler.fit_transform(simple_engineered_feature)
simple_engineered_feature_test = scaler.transform(simple_engineered_feature_test)


## === cell 16
def simple_prepro(s):
    return [w for w in s.replace("\n"," ").replace(","," , ").replace("("," ( ").replace(")"," ) ").
            replace("."," . ").replace("?"," ? ").replace(":"," : ").replace("n't"," not").
            replace("'ve"," have").replace("'re"," are").replace("'s"," is").split(" ") if w != ""]
def simple_prepro_tfidf(s):
    return " ".join([w for w in s.lower().replace("\n"," ").replace(","," , ").replace("("," ( ").replace(")"," ) ").
            replace("."," . ").replace("?"," ? ").replace(":"," : ").replace("n't"," not").
            replace("'ve"," have").replace("'re"," are").replace("'s"," is").split(" ") if w != ""])


## === cell 17
qt_max = max([len(simple_prepro(l)) for l in list(train["question_title"].values)])
qb_max = max([len(simple_prepro(l))  for l in list(train["question_body"].values)])
an_max = max([len(simple_prepro(l))  for l in list(train["answer"].values)])
print("max lenght of question_title is",qt_max)
print("max lenght of question_body is",qb_max)
print("max lenght of question_answer is",an_max)


## === cell 18
import gensim
import tqdm
import nltk
from nltk.corpus import brown

try:
    _ = brown.sents()
except LookupError:
    nltk.download("brown")

w2v_model = gensim.models.Word2Vec(brown.sents(), vector_size=100)


def get_word_embeddings(text):
    np.random.seed(abs(hash(text)) % (10**8))
    words = simple_prepro(text)
    vectors = np.zeros((len(words), 100))
    if len(words) == 0:
        vectors = np.zeros((1, 100))
    for i, word in enumerate(simple_prepro(text)):
        try:
            vectors[i] = w2v_model.wv[word]
        except Exception:
            vectors[i] = np.random.uniform(-0.01, 0.01, 100)
    return np.max(np.array(vectors), axis=0)


## === cell 19
question_title = [get_word_embeddings(l) for l in tqdm.tqdm(train["question_title"].values)]
question_title_test = [get_word_embeddings(l) for l in tqdm.tqdm(test["question_title"].values)]

question_body = [get_word_embeddings(l) for l in tqdm.tqdm(train["question_body"].values)]
question_body_test = [get_word_embeddings(l) for l in tqdm.tqdm(test["question_body"].values)]

answer = [get_word_embeddings(l) for l in tqdm.tqdm(train["answer"].values)]
answer_test = [get_word_embeddings(l) for l in tqdm.tqdm(test["answer"].values)]


## === cell 20
tfidf = TfidfVectorizer(ngram_range=(1, 3)) 
tsvd = TruncatedSVD(n_components = 50)

tfidf_question_title = tfidf.fit_transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(train["question_title"].values)])
tfidf_question_title_test = tfidf.transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(test["question_title"].values)])
tfidf_question_title = tsvd.fit_transform(tfidf_question_title)
tfidf_question_title_test = tsvd.transform(tfidf_question_title_test)

tfidf_question_body = tfidf.fit_transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(train["question_body"].values)])
tfidf_question_body_test = tfidf.transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(test["question_body"].values)])
tfidf_question_body = tsvd.fit_transform(tfidf_question_body)
tfidf_question_body_test = tsvd.transform(tfidf_question_body_test)

tfidf_answer = tfidf.fit_transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(train["answer"].values)])
tfidf_answer_test = tfidf.transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(test["answer"].values)])
tfidf_answer = tsvd.fit_transform(tfidf_answer)
tfidf_answer_test = tsvd.transform(tfidf_answer_test)


## === cell 21
type2int = {type:i for i,type in enumerate(list(set(train["category"])))}
cate = np.identity(5)[np.array(train["category"].apply(lambda x:type2int[x]))].astype(np.float64)
cate_test = np.identity(5)[np.array(test["category"].apply(lambda x:type2int[x]))].astype(np.float64)

train_features = np.concatenate([question_title, question_body, answer,
                                 tfidf_question_title, tfidf_question_body, tfidf_answer, 
                                 cate, simple_engineered_feature
                                ], axis=1)
test_features = np.concatenate([question_title_test, question_body_test, answer_test, 
                               tfidf_question_title_test, tfidf_question_body_test, tfidf_answer_test,
                                cate_test, simple_engineered_feature_test
                                ], axis=1)


## === cell 22
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import gc
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

from tf_keras.models import Sequential
from tf_keras.layers import Dense, Activation
from tf_keras.callbacks import EarlyStopping

num_folds = 7
fold_scores = []
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
test_preds = np.zeros((len(test_features), len(target_cols)))

for train_index, val_index in kf.split(train_features):
    gc.collect()
    train_X = train_features[train_index, :]
    train_y = train[target_cols].iloc[train_index]

    val_X = train_features[val_index, :]
    val_y = train[target_cols].iloc[val_index]
    model = Sequential(
        [
            Dense(512, input_shape=(train_features.shape[1],)),
            Activation("relu"),
            Dense(50),  # prev 128
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
        train_X, train_y, epochs=100, validation_data=(val_X, val_y), callbacks=[es]
    )
    preds = model.predict(val_X)
    overall_score = 0
    for col_index, col in enumerate(target_cols):
        overall_score += spearmanr(
            preds[:, col_index], val_y[col].values
        ).correlation / len(target_cols)
        print(col, spearmanr(preds[:, col_index], val_y[col].values).correlation)
    fold_scores.append(overall_score)
    print(overall_score)

    test_preds += model.predict(test_features) / num_folds

print(fold_scores)


## --- ERROR in cell 22, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;31mAttributeError[0m: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 23
sub = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
for col_index, col in enumerate(target_cols):
    sub[col] = test_preds[:, col_index]
sub.to_csv("submission.csv", index = False)
