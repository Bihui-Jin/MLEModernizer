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

0.3537

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.35725) has done: 'I fix the import/runtime issues that currently prevent the notebook from running (the protobuf/keras EarlyStopping import, missing imports due to the first cell failing, and missing NLTK corpus availability). I keep the same feature engineering and model architecture/training loop, but make the Word2Vec embedding step robust by falling back to a simple, deterministic random embedding when the Brown corpus isn’t available (common on Kaggle without downloads). I also make category one-hot encoding stable by using the true number of categories instead of hard-coding 5, and ensure the submission is written with the correct columns and shape. These changes are primarily correctness/stability fixes; they should yield a valid `submission.csv` and a reasonable baseline score without changing the core approach.'
- What this solution (achieved 0.35781) has done: 'I fix the protobuf/MessageFactory crash that occurs on import by removing the unnecessary `gensim/nltk` imports from the first cell and importing them lazily only where needed (the Word2Vec cell), with a safe fallback when Brown isn’t available. I also make the data paths robust for this environment by resolving them from the known `/kaggle/input/google-quest-challenge/` location (while keeping the original relative path as a fallback). Since your current score (0.35725) is already above the target (0.29087) and within the ±10% tolerance band, I avoid any score-changing modeling/training changes and focus only on correctness/stability so the notebook runs end-to-end and writes a valid `submission.csv`. Finally, I keep the exact model, features, folds, and training loop intact to preserve evaluation semantics.'
- What this solution (achieved 0.35343) has done: 'I fix the runtime crash caused by the protobuf `MessageFactory` incompatibility that occurs when importing `tf_keras` in this Kaggle environment by switching those imports to the standard `tensorflow.keras` equivalents (same Keras API, same model/loss/training loop). I also make the early cells robust by importing heavy/optional libraries (seaborn/matplotlib) lazily so a plotting import can’t block training. Finally, I keep the exact feature engineering and model/training semantics intact and ensure the submission CSV is written with correct column order and aligned `qa_id`s; since your current score is already above the target band, I won’t introduce any score-improving changes.'
- What this solution (achieved 0.35793) has done: 'I fix the runtime crash that happens immediately on importing TensorFlow/Keras in this environment by avoiding the problematic `tensorflow` stack and instead using the already-installed `tf_keras` package (same Keras API) while keeping the exact same model, loss, and training loop. I also make the optional plotting cells non-blocking by switching Matplotlib to a non-interactive backend so they can’t crash headless Kaggle runs. Finally, I keep all feature engineering and CV logic unchanged, and ensure the submission is written as `submission.csv` with the correct column order and `qa_id` alignment. Since your current score (0.35343) is already above the target band, these changes are intended to be score-neutral and purely for end-to-end correctness.'
- What this solution (achieved 0.35443) has done: 'We fix the immediate runtime crash (`MessageFactory.GetPrototype`) by removing the `tf_keras` import path and using the standard `tensorflow.keras` API instead, which is compatible in Kaggle’s environment and preserves the same model/training semantics. Because your current score (0.35793) is already above the target (0.29087) and within the ±10% target band, we not make any score-improving changes—only stability/correctness fixes. We also ensure all needed imports are available even if earlier cells fail, and keep submission formatting/alignment intact so a valid `submission.csv` is always produced. All feature engineering, CV, model architecture, loss, and training loop remain unchanged.'
- What this solution (achieved 0.35714) has done: 'We fix the immediate runtime crash in the first cell caused by the protobuf/TensorFlow import incompatibility by switching the Keras imports to the already-installed `tf_keras` package (same API usage for `Sequential`, `Dense`, `Activation`, `EarlyStopping`) while keeping the model architecture, loss, and training loop identical. We also make the random seed setting compatible with this Keras distribution to keep behavior deterministic. Since your current score (0.35443) is already above the target (0.29087) and within the ±10% band, we not make any score-improving changes—only stability/correctness fixes so it runs end-to-end. Submission writing remains the same and always produce a valid `submission.csv` with correct columns/order and [0,1] clipping.'
- What this solution (achieved 0.35443) has done: 'I fix the immediate runtime crash happening on import (`MessageFactory.GetPrototype`) by avoiding the problematic `tf_keras` stack and using the standard `tensorflow.keras` API (same Keras model, same layers, same training loop). I also ensure all necessary imports occur in the first cell so later cells don’t fail due to the earlier crash. No modeling/training/feature changes are introduced because your current score (0.35714) is already above the target (0.29087) and within the ±10% band, so the intent is score-neutral stability. Finally, I keep submission formatting and `qa_id` alignment intact and always write a valid `submission.csv`.'
- What this solution (achieved 0.35855) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by avoiding the incompatible TensorFlow import stack in this environment and instead using the already-installed `tf_keras` package for the exact same Keras API calls (Sequential/Dense/Activation/EarlyStopping). I also make the seed-setting and imports resilient so that if one backend-specific function is missing it won’t crash the whole run. Because your current score (0.35443) is already above the target (0.29087) and within the ±10% tolerance band, I not change any feature engineering, CV, model architecture, loss, or training loop semantics (score should remain in the same ballpark). Finally, I keep the submission writing logic but ensure it always outputs `submission.csv` with the correct columns and aligned `qa_id`s.'
- What this solution (achieved 0.35372) has done: 'I fix the immediate runtime crash happening in the first cell by removing the incompatible `tf_keras` import path and switching to the standard `tensorflow.keras` API (same Keras model, layers, loss, and training loop semantics). I also add a small, safe import fallback so if `tensorflow` isn’t available for some reason in your environment, it still try `keras` as a backup without changing the core approach. Since your current score (0.35855) is already above the target (0.29087) and within the ±10% band, I won’t introduce any score-improving changes—only stability fixes to ensure the notebook runs end-to-end and writes a valid `submission.csv`. The rest of the pipeline (feature engineering, CV, model architecture, and submission formatting) is preserved.'
- What this solution (achieved 0.35939) has done: 'The crash happens before any training because importing `tensorflow.keras` triggers a protobuf `MessageFactory.GetPrototype` incompatibility in this Kaggle image. To make the notebook run end-to-end without changing the model/training semantics, I switch Keras imports to the standalone `tf_keras` package first (which is installed here), and only fall back to `tensorflow.keras`/`keras` if needed. I also renumber the provided cells to start at 1 (your cell 0 becomes cell 1) so the script format is valid, and I keep all feature engineering, CV, architecture, loss, and submission formatting unchanged (so score impact should be negligible). Finally, the code still writes a correct `submission.csv` with `qa_id` alignment and the required 30 target columns clipped to `[0,1]`.'
- What this solution (achieved 0.3537) has done: 'I fix the crash in the first cell caused by the protobuf `MessageFactory.GetPrototype` incompatibility by avoiding `tf_keras`/`tensorflow` imports entirely and using the already-installed standalone `keras==3.8.0` backend (same Sequential/Dense/Activation/EarlyStopping API and identical model/training loop semantics). I keep all feature engineering, CV, architecture, loss, and inference logic unchanged so the score impact should be minimal (your current score is already above the target band, so we avoid score-changing edits). I also make the data path resolution robust for both `/kaggle/input/google-quest-challenge` and the provided `../input` fallback without changing I/O destinations. Finally, I ensure the submission is always written as `submission.csv` with correct column order and `qa_id` alignment.'

# 9. Code solution

## === cell 0
import os
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.stats import spearmanr

import tqdm

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
pd.set_option("display.notebook_repr_html", True)
gc.collect()

np.random.seed(42)
try:
    keras.utils.set_random_seed(42)
except Exception:
    try:
        import random

        random.seed(42)
    except Exception:
        pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 2
base1 = Path("/kaggle/input/google-quest-challenge")
base2 = Path("../input/google-quest-challenge")
base3 = Path("/kaggle/data/google-quest-challenge")

base = base1 if base1.exists() else (base2 if base2.exists() else base3)
train_path = base / "train.csv"
test_path = base / "test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
df = pd.concat([train, test], axis=0, ignore_index=True)
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
ax.set_xlabel("Sites")
ax.set_ylabel("Question Counts")




## === cell 9
def char_count(s):
    return len(s)


def word_count(s):
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

train["num_answer_user"].fillna(1, inplace=True)
train["num_question_user"].fillna(1, inplace=True)

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
qt_max = max([len(simple_prepro(l)) for l in list(train["question_title"].values)])
qb_max = max([len(simple_prepro(l)) for l in list(train["question_body"].values)])
an_max = max([len(simple_prepro(l)) for l in list(train["answer"].values)])
print("max lenght of question_title is", qt_max)
print("max lenght of question_body is", qb_max)
print("max lenght of question_answer is", an_max)




## === cell 17
USE_W2V = True
try:
    import gensim
    from gensim.models import Word2Vec
    import nltk
    from nltk.corpus import brown

    try:
        _ = brown.sents()
    except Exception:
        USE_W2V = False
except Exception:
    USE_W2V = False

if USE_W2V:
    w2v_model = gensim.models.Word2Vec(
        sentences=brown.sents(), vector_size=100, window=5, min_count=1, workers=1
    )


def get_word_embeddings(text):
    np.random.seed(abs(hash(text)) % (10**8))
    words = simple_prepro(text)
    vectors = np.zeros((len(words), 100), dtype=np.float32)
    if len(words) == 0:
        vectors = np.zeros((1, 100), dtype=np.float32)

    if USE_W2V:
        for i, word in enumerate(words):
            try:
                vectors[i] = w2v_model.wv[word]
            except Exception:
                vectors[i] = np.random.uniform(-0.01, 0.01, 100).astype(np.float32)
    else:
        for i, _word in enumerate(words):
            vectors[i] = np.random.uniform(-0.01, 0.01, 100).astype(np.float32)

    return np.max(np.array(vectors), axis=0)




## === cell 18
question_title = [
    get_word_embeddings(l) for l in tqdm.tqdm(train["question_title"].values)
]
question_title_test = [
    get_word_embeddings(l) for l in tqdm.tqdm(test["question_title"].values)
]

question_body = [
    get_word_embeddings(l) for l in tqdm.tqdm(train["question_body"].values)
]
question_body_test = [
    get_word_embeddings(l) for l in tqdm.tqdm(test["question_body"].values)
]

answer = [get_word_embeddings(l) for l in tqdm.tqdm(train["answer"].values)]
answer_test = [get_word_embeddings(l) for l in tqdm.tqdm(test["answer"].values)]

question_title = np.vstack(question_title)
question_title_test = np.vstack(question_title_test)
question_body = np.vstack(question_body)
question_body_test = np.vstack(question_body_test)
answer = np.vstack(answer)
answer_test = np.vstack(answer_test)




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
type2int = {t: i for i, t in enumerate(sorted(train["category"].unique()))}
n_cats = len(type2int)

cate = np.identity(n_cats)[np.array(train["category"].map(type2int))].astype(np.float64)

test_cat_idx = test["category"].map(type2int).fillna(-1).astype(int).values
cate_test = np.zeros((len(test), n_cats), dtype=np.float64)
mask = test_cat_idx >= 0
cate_test[mask] = np.identity(n_cats)[test_cat_idx[mask]]

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

print("train_features:", train_features.shape, "test_features:", test_features.shape)




## === cell 21
num_folds = 10
fold_scores = []
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
test_preds = np.zeros((len(test_features), len(target_cols)), dtype=np.float32)

for fold, (train_index, val_index) in enumerate(kf.split(train_features), 1):
    gc.collect()
    train_X = train_features[train_index, :]
    train_y = train[target_cols].iloc[train_index].values

    val_X = train_features[val_index, :]
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
    print("fold", fold, "score", overall_score)

    test_preds += model.predict(test_features, verbose=0) / num_folds

print("fold_scores:", fold_scores, "mean:", float(np.mean(fold_scores)))




## === cell 22
sub_path = base / "sample_submission.csv"
sub = pd.read_csv(sub_path)

if "qa_id" in sub.columns and "qa_id" in test.columns and (len(sub) == len(test)):
    if not np.array_equal(sub["qa_id"].values, test["qa_id"].values):
        sub = sub.set_index("qa_id")
        tmp = pd.DataFrame({"qa_id": test["qa_id"].values}).set_index("qa_id")
        sub = sub.reindex(tmp.index).reset_index()

for col_index, col in enumerate(target_cols):
    sub[col] = np.clip(test_preds[:, col_index], 0.0, 1.0)

sub = sub[["qa_id"] + target_cols]
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
