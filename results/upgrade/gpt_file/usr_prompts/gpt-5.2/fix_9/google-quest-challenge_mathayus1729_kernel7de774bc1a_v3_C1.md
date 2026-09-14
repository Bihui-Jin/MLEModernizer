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

0.18187

# 6. Current score

0.28939

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.27638) has done: 'I fix the runtime crash caused by an incompatibility between `protobuf>=6` and the old `keras` import path by switching to `tf.keras` (which is already installed and stable here) and avoiding `keras.backend`. I also remove the dependency on NLTK corpora downloads (the `stopwords` and punkt tokenizers) by using your already-defined `eng_stopwords` list and a simple regex tokenizer, so the notebook runs offline reliably. To keep score changes minimal (your current score is already well above the target band), I preserve the same model and training loop, and only make tokenization/stopword handling deterministic and robust. Finally, I ensure the submission is written as `submission.csv` with the exact `sample_submission.csv` column order and `[0,1]` clipping.'
- What this solution (achieved 0.29679) has done: 'The crash happens before training because TensorFlow 2.18 can trip over an incompatible protobuf runtime, producing `MessageFactory.GetPrototype` errors at import time. Since we can’t change the Kaggle base environment, the safest minimal fix is to force TensorFlow to use the pure-Python protobuf implementation before importing `tensorflow`, which avoids that failing C++ path. I also keep everything else (tokenization, model, training loop, submission formatting) identical so the score behavior stays essentially the same (you’re already well above the target band). Finally, I keep the submission written as `submission.csv` with the exact column order from `sample_submission.csv`.'
- What this solution (achieved 0.28611) has done: 'To fix the import-time crash (`MessageFactory.GetPrototype`) reliably in this Kaggle environment, I keep the existing pure-Python protobuf workaround but make it take effect even earlier by forcing it before any TensorFlow-related import (including transitive ones) and by disabling TF’s C++ protobuf fast-path. I also reset the global `tokens` list inside `LSTM_model()` so repeated runs don’t silently change the vocabulary and score. Finally, I keep the model/training logic identical and ensure the submission is written as `submission.csv` with the exact `sample_submission.csv` column order and predictions clipped to `[0, 1]` (score-neutral formatting correctness).'
- What this solution (achieved 0.29154) has done: 'I fix the import-time protobuf crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation and disabling TensorFlow’s C++ protobuf fast-path before *any* TensorFlow/Keras import, which is the root cause of your current runtime error. I keep your model, preprocessing, training loop, and submission formatting unchanged so the score behavior stays essentially the same (and since your current score is already above the target band, we avoid any score-tuning changes). I also add a tiny defensive fallback to locate the CSVs from either `/kaggle/input/...` or `/kaggle/data/...` without changing the expected Kaggle paths. Finally, I ensure `submission.csv` is always produced with the exact column order from `sample_submission.csv` and predictions clipped to `[0,1]`.'
- What this solution (achieved 0.29206) has done: 'The current failure happens at TensorFlow import time due to an incompatibility between `protobuf>=6` and TensorFlow’s compiled protobuf path, so I make the existing workaround take effect even earlier by forcing the pure-Python protobuf backend and disabling the C++ implementation via environment variables before *any* TensorFlow import. I also keep the modeling/training/inference logic identical (so score behavior remains essentially unchanged and still well above your target band), only adding a safe “retry import” fallback if the first TF import still triggers the protobuf crash. Finally, I ensure the submission is always written as `submission.csv` with the exact `sample_submission.csv` column order and `[0,1]` clipping.'
- What this solution (achieved 0.27647) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation even earlier, *before* any TensorFlow-related module can be imported, and by ensuring no stale `google.protobuf` / `tensorflow` modules remain in `sys.modules` when retrying. This is a runtime-stability fix and should be score-neutral (your current score is already well above the target, so we avoid any modeling/training changes). I also keep the same data paths, preprocessing, model, training loop, and submission formatting, only adding a small defensive check to guarantee the submission column order exactly matches `sample_submission.csv` and the file is written as `submission.csv`. The core logic and evaluation semantics remain unchanged.'
- What this solution (achieved 0.29883) has done: 'Your notebook is currently crashing at TensorFlow import with the protobuf `MessageFactory.GetPrototype` error; the minimal reliable fix in this Kaggle environment is to force the pure-Python protobuf runtime *before any protobuf/TensorFlow import* and to also set the newer `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus `TF_PROTOBUF_IMPLEMENTATION=python`. I keep your preprocessing, model architecture, training loop, and submission formatting identical so the score behavior stays essentially the same (and since your current score is already above the target band, we avoid any score-tuning changes). I also make the TF import retry catch broader exceptions (not just `AttributeError`), because this crash is often raised as a different exception type depending on import order. Finally, I ensure `submission.csv` is always written with the exact `sample_submission.csv` column order and predictions clipped to `[0,1]`.'
- What this solution (achieved 0.28939) has done: 'We fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *and* pre-importing `google.protobuf.message_factory` before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` path that breaks in this environment. The rest of the pipeline (cleaning/tokenization, LSTM architecture, training loop, and submission formatting) be kept identical to preserve score behavior (your current score is already well above the target, so we avoid score-changing tuning). We also make the TF import retry more robust and ensure the submission is always written as `submission.csv` with the exact `sample_submission.csv` column order and predictions clipped to `[0,1]`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CEXT", "1")
os.environ.setdefault("TF_PROTOBUF_IMPLEMENTATION", "python")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_PROTOBUF_USE_CLOUD_FAST_CPP", "0")

import re
import gc
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf") or k.startswith("tensorflow"):
        sys.modules.pop(k, None)

try:
    import google.protobuf.message_factory  # noqa: F401
except Exception:
    pass

try:
    import tensorflow as tf
except Exception:
    for k in list(sys.modules.keys()):
        if k.startswith("google.protobuf") or k.startswith("tensorflow"):
            sys.modules.pop(k, None)
    gc.collect()
    import google.protobuf.message_factory  # noqa: F401
    import tensorflow as tf  # noqa: F401

from tensorflow.keras.preprocessing.sequence import pad_sequences

try:
    import nltk  # noqa: F401
except Exception:
    nltk = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def plot_len(df, col_name, i):
    plt.figure(i)
    sns.histplot(df[col_name].str.len().dropna(), kde=True)
    plt.ylabel("length of string")
    plt.show()


def plot_cnt_words(df, col_name, i):
    plt.figure(i)
    vals = df[col_name].fillna("").apply(lambda x: len(str(x).strip().split()))
    sns.histplot(vals, kde=True)
    plt.ylabel("count of words")
    plt.show()


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
    "i'd": "I would",
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
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = set(eng_stopwords)
    text = [w for w in text if w not in stops]
    return " ".join(text)


def _get_mispell(mispell_dict):
    keys = sorted(mispell_dict.keys(), key=len, reverse=True)
    mispell_re = re.compile("(" + "|".join(map(re.escape, keys)) + ")")
    return mispell_dict, mispell_re


def replace_typical_misspell(text):
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text)
    mispellings, mispellings_re = _get_mispell(mispell_dict)

    def replace(match):
        return mispellings.get(match.group(0), match.group(0))

    return mispellings_re.sub(replace, text)


def clean_data(df, columns: list):
    for col in columns:
        df[col] = (
            df[col].fillna("").apply(lambda x: replace_typical_misspell(clean_text(x)))
        )
    return df


def get_tfidf_features(data, dims=256):
    tfidf = TfidfVectorizer(ngram_range=(1, 3))
    tsvd = TruncatedSVD(n_components=dims, n_iter=5, random_state=42)
    tfquestion_title = tsvd.fit_transform(
        tfidf.fit_transform(data["question_title"].values)
    )
    tfquestion_body = tsvd.fit_transform(
        tfidf.fit_transform(data["question_body"].values)
    )
    tfanswer = tsvd.fit_transform(tfidf.fit_transform(data["answer"].values))
    return tfquestion_title, tfquestion_body, tfanswer


def correlation(x, y):
    mx = tf.reduce_mean(x)
    my = tf.reduce_mean(y)
    xm, ym = x - mx, y - my
    r_num = tf.reduce_mean(xm * ym)
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return r_num / (r_den + tf.keras.backend.epsilon())


_word_re = re.compile(r"[A-Za-z0-9']+")


def simple_tokenize(text: str):
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    return _word_re.findall(str(text).lower())




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    LSTM,
    Bidirectional,
    Input,
    Concatenate,
)
from tensorflow.keras.models import Model

np.random.seed(42)
tf.random.set_seed(42)

BASE_CANDIDATES = [
    "/kaggle/input/google-quest-challenge",
    "/kaggle/data/google-quest-challenge",
    "/kaggle/input",
    "/kaggle/data",
]


def _find_file(filename: str) -> str:
    for base in BASE_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    return os.path.join("/kaggle/input/google-quest-challenge", filename)


train_path = _find_file("train.csv")
test_path = _find_file("test.csv")
sample_path = _find_file("sample_submission.csv")

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_submission = pd.read_csv(sample_path)

tokens = []


def get_words(col):
    global tokens
    toks = simple_tokenize(col)
    tokens.extend(toks)
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
        df_train[col] = df_train[col].apply(get_words)
        df_test[col] = df_test[col].apply(get_words)

    vocab = sorted(list(set(tokens)))
    vocab_size = len(vocab)

    word2idx = {word: idx for idx, word in enumerate(vocab)}

    for col in columns:
        df_train[col] = df_train[col].apply(
            lambda x: convert_to_indx(x, word2idx, vocab_size)
        )
        df_test[col] = df_test[col].apply(
            lambda x: convert_to_indx(x, word2idx, vocab_size)
        )

    maxlen = 50

    X_train_question_title = pad_sequences(
        df_train["question_title"], maxlen=maxlen, padding="post", value=0
    )
    X_train_question_body = pad_sequences(
        df_train["question_body"], maxlen=maxlen, padding="post", value=0
    )
    X_train_answer = pad_sequences(
        df_train["answer"], maxlen=maxlen, padding="post", value=0
    )

    X_test_question_title = pad_sequences(
        df_test["question_title"], maxlen=maxlen, padding="post", value=0
    )
    X_test_question_body = pad_sequences(
        df_test["question_body"], maxlen=maxlen, padding="post", value=0
    )
    X_test_answer = pad_sequences(
        df_test["answer"], maxlen=maxlen, padding="post", value=0
    )

    target_columns = df_submission.columns[1:]
    y_train = df_train[target_columns].astype(np.float32)

    inpqt = Input(shape=(maxlen,), name="inpqt")
    inpqb = Input(shape=(maxlen,), name="inpqb")
    inpan = Input(shape=(maxlen,), name="inpan")

    Eqt = Embedding(vocab_size + 1, 200, input_length=maxlen)(inpqt)
    Eqb = Embedding(vocab_size + 1, 200, input_length=maxlen)(inpqb)
    Ean = Embedding(vocab_size + 1, 200, input_length=maxlen)(inpan)

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
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(
        {
            "inpqt": X_train_question_title,
            "inpqb": X_train_question_body,
            "inpan": X_train_answer,
        },
        y_train,
        batch_size=128,
        epochs=10,
        validation_split=0.2,
        verbose=2,
    )

    y_test = model.predict(
        {
            "inpqt": X_test_question_title,
            "inpqb": X_test_question_body,
            "inpan": X_test_answer,
        },
        batch_size=256,
        verbose=1,
    )

    y_test = np.clip(y_test, 0.0, 1.0)

    sample = pd.read_csv(sample_path)
    target_cols = list(sample.columns[1:])

    out = pd.DataFrame({"qa_id": df_test["qa_id"].values})
    for i, col in enumerate(target_cols):
        out[col] = y_test[:, i]

    out = out[sample.columns]
    out.to_csv("submission.csv", index=False)


LSTM_model(df_train, df_test, df_submission)
