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

0.26549

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.27186) has done: 'I fix the crash caused by an incompatibility between the installed `protobuf==6.x` and TensorFlow/Keras (which triggers `MessageFactory.GetPrototype` errors) by forcing the pure‑Python protobuf implementation before importing TensorFlow. I also make the NLTK stopwords/tokenizers robust in the Kaggle environment by downloading the required resources if missing and falling back safely when downloads are unavailable. These changes are score-neutral (same model/training/prediction logic) and ensure the notebook runs end-to-end to write a valid `submission.csv` with the correct columns. I keep all core modeling code and training settings unchanged.'
- What this solution (achieved 0.26141) has done: 'I fix the protobuf/TensorFlow crash by setting the protobuf pure-Python env var before importing TensorFlow/Keras and by importing `keras` from `tf_keras` (the Kaggle image includes both `keras==3` and `tf_keras==2`, and mixing them can trigger protobuf-related issues). I also adjust the Embedding layer’s `input_dim` to include the OOV index (currently you map unknown tokens to `vocab_size` but set `input_dim=vocab_size`, which can cause out-of-range indices at runtime). These are correctness/stability fixes and should be essentially score-neutral, keeping the same model, preprocessing, and training loop. The rest of the pipeline (data loading, cleaning, training, predicting, and writing `submission.csv`) remains unchanged and run end-to-end.'
- What this solution (achieved 0.26549) has done: 'I fix the `MessageFactory.GetPrototype` crash by forcing the pure‑Python protobuf runtime *before any TensorFlow/tf_keras import* (your current error happens because TF is imported in a later cell where the env var is no longer guaranteed to be early enough). I also make the notebook deterministic and stable by clearing the global `tokens` list at the start of model building, and by ensuring the target columns used for training are exactly the 30 label columns from `sample_submission.csv`. Finally, since your current score (0.26141) is already better than the target (0.20176), I avoid any model/training changes that would further increase performance; the edits are intended to be score-neutral while ensuring the notebook runs end-to-end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # linear algebra
import pandas as pd  # data processing

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import tensorflow as tf
import numpy as np
import nltk

import tf_keras as keras
import tf_keras.backend as K

from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string
from tf_keras.preprocessing.sequence import pad_sequences


def _ensure_nltk():
    for pkg in ["punkt", "stopwords"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


_ensure_nltk()

eng_stopwords = [
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "you're",
    "you've",
    "you'll",
    "you'd",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "she's",
    "her",
    "hers",
    "herself",
    "it",
    "it's",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "that'll",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "don't",
    "should",
    "should've",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "ve",
    "y",
    "ain",
    "aren",
    "aren't",
    "couldn",
    "couldn't",
    "didn",
    "didn't",
    "doesn",
    "doesn't",
    "hadn",
    "hadn't",
    "hasn",
    "hasn't",
    "haven",
    "haven't",
    "isn",
    "isn't",
    "ma",
    "mightn",
    "mightn't",
    "mustn",
    "mustn't",
    "needn",
    "needn't",
    "shan",
    "shan't",
    "shouldn",
    "shouldn't",
    "wasn",
    "wasn't",
    "weren",
    "weren't",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
]

import gc, pickle
from nltk import word_tokenize, sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def plot_len(df, col_name, i):
    plt.figure(i)
    sns.distplot(df[col_name].str.len())
    plt.ylabel("length of string")
    plt.show()


def plot_cnt_words(df, col_name, i):
    plt.figure(i)
    vals = df[col_name].apply(lambda x: len(x.strip().split()))
    sns.distplot(vals)
    plt.ylabel("count of words")
    plt.show()


puncts = [
    ",",
    ".",
    '"',
    ":",
    ")",
    "(",
    "-",
    "!",
    "?",
    "|",
    ";",
    "'",
    "$",
    "&",
    "/",
    "[",
    "]",
    ">",
    "%",
    "=",
    "#",
    "*",
    "+",
    "\\",
    "•",
    "~",
    "@",
    "£",
    "·",
    "_",
    "{",
    "}",
    "©",
    "^",
    "®",
    "`",
    "<",
    "→",
    "°",
    "€",
    "™",
    "›",
    "♥",
    "←",
    "×",
    "§",
    "″",
    "′",
    "Â",
    "█",
    "½",
    "à",
    "…",
    "\xa0",
    "\t",
    "“",
    "★",
    "”",
    "–",
    "●",
    "â",
    "►",
    "−",
    "¢",
    "²",
    "¬",
    "░",
    "¶",
    "↑",
    "±",
    "¿",
    "▾",
    "═",
    "¦",
    "║",
    "―",
    "¥",
    "▓",
    "—",
    "‹",
    "─",
    "\u3000",
    "\u202f",
    "▒",
    "：",
    "¼",
    "⊕",
    "▼",
    "▪",
    "†",
    "■",
    "’",
    "▀",
    "¨",
    "▄",
    "♫",
    "☆",
    "é",
    "¯",
    "♦",
    "¤",
    "▲",
    "è",
    "¸",
    "¾",
    "Ã",
    "⋅",
    "‘",
    "∞",
    "«",
    "∙",
    "）",
    "↓",
    "、",
    "│",
    "（",
    "»",
    "，",
    "♪",
    "╩",
    "╚",
    "³",
    "・",
    "╦",
    "╣",
    "╔",
    "╗",
    "▬",
    "❤",
    "ï",
    "Ø",
    "¹",
    "≤",
    "‡",
    "√",
]

mispell_dict = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "couldnt": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "doesnt": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "havent": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "I had",
    "i'll": "I will",
    "i'm": "I am",
    "isn't": "is not",
    "it's": "it is",
    "it'll": "it will",
    "i've": "I have",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "shouldnt": "should not",
    "that's": "that is",
    "thats": "that is",
    "there's": "there is",
    "theres": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "theyre": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "weren't": "were not",
    "we've": "we have",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll": " will",
    "tryin'": "trying",
}


def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", str(text))
    text = text.lower().split()
    try:
        stops = set(stopwords.words("english"))
    except LookupError:
        stops = set(eng_stopwords)
    text = [w for w in text if w not in stops]
    text = " ".join(text)
    return text


def _get_mispell(mispell_dict):
    mispell_re = re.compile("(%s)" % "|".join(mispell_dict.keys()))
    return mispell_dict, mispell_re


def replace_typical_misspell(text):
    mispellings, mispellings_re = _get_mispell(mispell_dict)

    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)


def clean_data(df, columns: list):
    for col in columns:
        df[col] = df[col].fillna("").astype(str)
        df[col] = df[col].apply(lambda x: clean_text(x.lower()))
        df[col] = df[col].apply(lambda x: replace_typical_misspell(x))
    return df


def plot_freq_dist(train_data):
    freq_dist = FreqDist(
        [
            word
            for text in train_data["question_body"].str.replace(
                "[^a-za-z0-9^,!.\/+-=]", " ", regex=True
            )
            for word in text.split()
        ]
    )
    plt.figure(figsize=(20, 7))
    plt.title("Word frequency on question title (Training Data)").set_fontsize(25)
    plt.xlabel("").set_fontsize(25)
    plt.ylabel("").set_fontsize(25)
    freq_dist.plot(60, cumulative=False)
    plt.show()


def get_tfidf_features(data, dims=256):
    tfidf = TfidfVectorizer(ngram_range=(1, 3))
    tsvd = TruncatedSVD(n_components=dims, n_iter=5)

    tfquestion_title = tfidf.fit_transform(data["question_title"].values)
    tfquestion_title = tsvd.fit_transform(tfquestion_title)

    tfquestion_body = tfidf.fit_transform(data["question_body"].values)
    tfquestion_body = tsvd.fit_transform(tfquestion_body)

    tfanswer = tfidf.fit_transform(data["answer"].values)
    tfanswer = tsvd.fit_transform(tfanswer)

    return tfquestion_title, tfquestion_body, tfanswer


def correlation(x, y):
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x - mx, y - my
    r_num = tf.math.reduce_mean(tf.multiply(xm, ym))
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return r_num / r_den




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
from tf_keras.layers import (
    Dense,
    Dropout,
    Embedding,
    LSTM,
    Bidirectional,
    Input,
    Concatenate,
)
from tf_keras.models import Model

df_train = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
df_test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")
df_submission = pd.read_csv(
    "/kaggle/input/google-quest-challenge/sample_submission.csv"
)

tokens = []


def get_words(col):
    global tokens
    toks = []
    try:
        sentences = sent_tokenize(col)
    except LookupError:
        sentences = [col]
    for x in sentences:
        try:
            wt = word_tokenize(x)
        except LookupError:
            wt = x.split()
        tokens += wt
        toks += wt
    return toks


def convert_to_indx(col, word2idx, vocab_size):
    return [word2idx[word] if word in word2idx else vocab_size for word in col]


def LSTM_model(df_train, df_test, df_submission):
    global tokens
    tokens = []

    columns = ["question_title", "question_body", "answer"]
    df_train = clean_data(df_train, columns)
    df_test = clean_data(df_test, columns)

    for col in columns:
        df_train[col] = df_train[col].apply(lambda x: get_words(x))
        df_test[col] = df_test[col].apply(lambda x: get_words(x))

    vocab = sorted(list(set(tokens)))
    vocab_size = len(vocab)

    word2idx = {}
    idx2word = {}
    for idx, word in enumerate(vocab):
        word2idx[word] = idx
        idx2word[idx] = word

    for col in columns:
        df_train[col] = df_train[col].apply(
            lambda x: convert_to_indx(x, word2idx, vocab_size)
        )
        df_test[col] = df_test[col].apply(
            lambda x: convert_to_indx(x, word2idx, vocab_size)
        )

    maxlen_qt = 50
    maxlen_qb = 200
    maxlen_an = 200

    X_train_question_title = pad_sequences(
        df_train["question_title"], maxlen=maxlen_qt, padding="post", value=0
    )
    X_train_question_body = pad_sequences(
        df_train["question_body"], maxlen=maxlen_qb, padding="post", value=0
    )
    X_train_answer = pad_sequences(
        df_train["answer"], maxlen=maxlen_an, padding="post", value=0
    )

    X_test_question_title = pad_sequences(
        df_test["question_title"], maxlen=maxlen_qt, padding="post", value=0
    )
    X_test_question_body = pad_sequences(
        df_test["question_body"], maxlen=maxlen_qb, padding="post", value=0
    )
    X_test_answer = pad_sequences(
        df_test["answer"], maxlen=maxlen_an, padding="post", value=0
    )

    target_columns = df_submission.columns[1:]
    y_train = df_train[target_columns]

    inpqt = Input(shape=(maxlen_qt,), name="inpqt")
    inpqb = Input(shape=(maxlen_qb,), name="inpqb")
    inpan = Input(shape=(maxlen_an,), name="inpan")

    Eqt = Embedding(vocab_size + 1, 200, input_length=maxlen_qt)(inpqt)
    Eqb = Embedding(vocab_size + 1, 200, input_length=maxlen_qb)(inpqb)
    Ean = Embedding(vocab_size + 1, 200, input_length=maxlen_an)(inpan)

    BLqt = Bidirectional(LSTM(64))(Eqt)
    BLqb = Bidirectional(LSTM(64))(Eqb)
    BLan = Bidirectional(LSTM(64))(Ean)

    Dqt = Dropout(0.2)(BLqt)
    Dqb = Dropout(0.2)(BLqb)
    Dan = Dropout(0.2)(BLan)

    Concatenated = Concatenate()([Dqt, Dqb, Dan])
    Ds = Dense(60, activation="relu")(Concatenated)
    Dsf = Dense(30, activation="sigmoid")(Ds)

    model = Model(inputs=[inpqt, inpqb, inpan], outputs=Dsf)
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    model.fit(
        {
            "inpqt": X_train_question_title,
            "inpqb": X_train_question_body,
            "inpan": X_train_answer,
        },
        y_train,
        batch_size=32,
        epochs=2,
        validation_split=0.1,
        verbose=2,
    )

    y_test = model.predict(
        {
            "inpqt": X_test_question_title,
            "inpqb": X_test_question_body,
            "inpan": X_test_answer,
        },
        verbose=0,
    )

    df_submission_local = pd.read_csv(
        "/kaggle/input/google-quest-challenge/sample_submission.csv"
    )
    df_test_local = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")

    outp = {"qa_id": df_test_local["qa_id"].values}
    for i, col in enumerate(df_submission_local.columns[1:]):
        outp[col] = np.clip(y_test[:, i], 0.0, 1.0)

    my_submission = pd.DataFrame(outp, columns=df_submission_local.columns)
    my_submission.to_csv("submission.csv", index=False)
    return my_submission


_ = LSTM_model(df_train, df_test, df_submission)
print("Wrote submission.csv with shape:", _.shape)
print(_.head())
print("Saved to:", os.path.abspath("submission.csv"))
