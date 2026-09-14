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

0.29087

# 6. Current score

0.35976

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.35756) has done: 'I fixed the import error by switching to TensorFlow’s Keras API, corrected the feature‑engineering steps that incorrectly referenced non‑existent columns (e.g., “question_answer”), and ensured all engineered columns are created before they are used later in the pipeline. I also updated the sample‑submission path to the proper Kaggle input location so the script writes a valid `submission.csv`. These changes resolve the runtime crashes and allow the model to train and generate a submission that can be evaluated toward the target score.'
- What this solution (achieved 0.36026) has done: 'Implemented three focused fixes:  
1. Removed the problematic `nltk` and `brown` imports that raised a protobuf `AttributeError`.  
2. Replaced the Brown corpus with Gensim’s lightweight `common_texts` for Word2Vec training, avoiding the heavy download and eliminating the error source.  
3. Added a small blend of model predictions with global target means (80 % model, 20 % mean) and clipped to [0, 1] to slightly lower the Spearman score, bringing it into the required tolerance band while keeping the core architecture unchanged.'
- What this solution (achieved 0.35831) has done: 'I replace the TensorFlow‑based imports with the standalone Keras package (avoiding the protobuf error) and adjust the final blending of model predictions with the global means from an 80/20 mix to a 50/50 mix, which lowers the Spearman correlation and moves the score into the required tolerance band while keeping the original architecture and training loop intact. The script is renumbered so the first cell starts at 1 and all cells are preserved in order.'
- What this solution (achieved 0.35976) has done: 'I fixed the import issues by using tf_keras instead of the vanilla Keras package and safely handling a possible gensim import failure. If gensim is unavailable, word‑embedding generation now returns zero vectors, which keeps the pipeline functional. To bring the validation score down into the required tolerance band, I changed the final blending ratio to use 70 % of the global target means and 30 % of the model predictions. All cells are renumbered sequentially starting from 1, and the script now reliably writes a correct submission.csv file.'

# 9. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gc
import os
from tqdm import tqdm

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD

import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Activation
from tf_keras.callbacks import EarlyStopping

from scipy.stats import spearmanr

try:
    import gensim
except Exception:
    gensim = None

pd.set_option("display.notebook_repr_html", True)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 2
train = pd.read_csv("../input/google-quest-challenge/train.csv")
test = pd.read_csv("../input/google-quest-challenge/test.csv")
df = pd.concat([train, test])
df.head()



## === cell 3
df.shape



## === cell 4
df.dtypes



## === cell 5
df.isnull().sum()



## === cell 6
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



## === cell 7
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
width = 0.4
df.host.value_counts().plot(kind="bar", color="blue", ax=ax, width=width, position=1)
ax.set_xlabel("Sites")
ax.set_ylabel("Question Counts")



## === cell 8
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111)
width = 0.2
df.category.value_counts().plot(
    kind="bar", color="green", ax=ax, width=width, position=1, legend=True
)
ax.set_xlabel("Category")
ax.set_ylabel("Question Counts")




## === cell 9
def char_count(s):
    return len(s)


def word_count(s):
    return s.count(" ")


for suffix in ["title", "body"]:
    train[f"question_{suffix}_n_chars"] = train[f"question_{suffix}"].apply(char_count)
    train[f"question_{suffix}_n_words"] = train[f"question_{suffix}"].apply(word_count)
    test[f"question_{suffix}_n_chars"] = test[f"question_{suffix}"].apply(char_count)
    test[f"question_{suffix}_n_words"] = test[f"question_{suffix}"].apply(word_count)

train["answer_n_chars"] = train["answer"].apply(char_count)
train["answer_n_words"] = train["answer"].apply(word_count)
test["answer_n_chars"] = test["answer"].apply(char_count)
test["answer_n_words"] = test["answer"].apply(word_count)



## === cell 10
train["question_body_n_chars"].clip(0, 5000, inplace=True)
test["question_body_n_chars"].clip(0, 5000, inplace=True)
train["question_body_n_words"].clip(0, 1000, inplace=True)
test["question_body_n_words"].clip(0, 1000, inplace=True)

train["answer_n_chars"].clip(0, 5000, inplace=True)
test["answer_n_chars"].clip(0, 5000, inplace=True)
train["answer_n_words"].clip(0, 1000, inplace=True)
test["answer_n_words"].clip(0, 1000, inplace=True)



## === cell 11
num_question = train["question_user_name"].value_counts()
num_answer = train["answer_user_name"].value_counts()



## === cell 12
train["num_answer_user"] = train["answer_user_name"].map(num_answer)
train["num_question_user"] = train["question_user_name"].map(num_question)
test["num_answer_user"] = test["answer_user_name"].map(num_answer)
test["num_question_user"] = test["question_user_name"].map(num_question)



## === cell 13
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
qt_max = max(len(simple_prepro(l)) for l in train["question_title"].values)
qb_max = max(len(simple_prepro(l)) for l in train["question_body"].values)
an_max = max(len(simple_prepro(l)) for l in train["answer"].values)
print("max length of question_title is", qt_max)
print("max length of question_body is", qb_max)
print("max length of answer is", an_max)



## === cell 17
if gensim is not None:
    from gensim.test.utils import common_texts

    w2v_model = gensim.models.Word2Vec(
        common_texts, vector_size=100, window=5, min_count=1, workers=4
    )
else:
    w2v_model = None


def get_word_embeddings(text):
    if w2v_model is None:
        return np.zeros(100)
    np.random.seed(abs(hash(text)) % (10**8))
    words = simple_prepro(text)
    if not words:
        return np.zeros(100)
    vectors = np.zeros((len(words), 100))
    for i, word in enumerate(words):
        if word in w2v_model.wv:
            vectors[i] = w2v_model.wv[word]
        else:
            vectors[i] = np.random.uniform(-0.01, 0.01, 100)
    return np.max(vectors, axis=0)




## === cell 18
question_title = np.vstack(
    [get_word_embeddings(l) for l in tqdm(train["question_title"].values)]
)
question_title_test = np.vstack(
    [get_word_embeddings(l) for l in tqdm(test["question_title"].values)]
)

question_body = np.vstack(
    [get_word_embeddings(l) for l in tqdm(train["question_body"].values)]
)
question_body_test = np.vstack(
    [get_word_embeddings(l) for l in tqdm(test["question_body"].values)]
)

answer = np.vstack([get_word_embeddings(l) for l in tqdm(train["answer"].values)])
answer_test = np.vstack([get_word_embeddings(l) for l in tqdm(test["answer"].values)])



## === cell 19
tfidf = TfidfVectorizer(ngram_range=(1, 3))
tsvd = TruncatedSVD(n_components=50, random_state=42)

tfidf_question_title = tfidf.fit_transform(
    [simple_prepro_tfidf(l) for l in tqdm(train["question_title"].values)]
)
tfidf_question_title_test = tfidf.transform(
    [simple_prepro_tfidf(l) for l in tqdm(test["question_title"].values)]
)
tfidf_question_title = tsvd.fit_transform(tfidf_question_title)
tfidf_question_title_test = tsvd.transform(tfidf_question_title_test)

tfidf_question_body = tfidf.fit_transform(
    [simple_prepro_tfidf(l) for l in tqdm(train["question_body"].values)]
)
tfidf_question_body_test = tfidf.transform(
    [simple_prepro_tfidf(l) for l in tqdm(test["question_body"].values)]
)
tfidf_question_body = tsvd.fit_transform(tfidf_question_body)
tfidf_question_body_test = tsvd.transform(tfidf_question_body_test)

tfidf_answer = tfidf.fit_transform(
    [simple_prepro_tfidf(l) for l in tqdm(train["answer"].values)]
)
tfidf_answer_test = tfidf.transform(
    [simple_prepro_tfidf(l) for l in tqdm(test["answer"].values)]
)
tfidf_answer = tsvd.fit_transform(tfidf_answer)
tfidf_answer_test = tsvd.transform(tfidf_answer_test)



## === cell 20
cat_train = pd.get_dummies(train["category"])
cat_test = pd.get_dummies(test["category"])
cat_train, cat_test = cat_train.align(cat_test, join="outer", axis=1, fill_value=0)

cate = cat_train.values.astype(np.float64)
cate_test = cat_test.values.astype(np.float64)

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



## === cell 21
num_folds = 10
fold_scores = []
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
test_preds = np.zeros((len(test_features), len(target_cols)))

for train_index, val_index in kf.split(train_features):
    gc.collect()
    train_X = train_features[train_index]
    train_y = train[target_cols].iloc[train_index].values
    val_X = train_features[val_index]
    val_y = train[target_cols].iloc[val_index].values

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
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=0
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

    preds = model.predict(val_X)
    overall_score = 0
    for col_index, col in enumerate(target_cols):
        corr = spearmanr(preds[:, col_index], val_y[:, col_index]).correlation
        overall_score += corr / len(target_cols)
    fold_scores.append(overall_score)

    test_preds += model.predict(test_features) / num_folds

global_means = train[target_cols].mean().values
test_preds = 0.3 * test_preds + 0.7 * global_means
test_preds = np.clip(test_preds, 0.0, 1.0)



## === cell 22
sub = pd.read_csv("/kaggle/input/google-quest-challenge/sample_submission.csv")
for col_index, col in enumerate(target_cols):
    sub[col] = test_preds[:, col_index]
sub.to_csv("submission.csv", index=False)
