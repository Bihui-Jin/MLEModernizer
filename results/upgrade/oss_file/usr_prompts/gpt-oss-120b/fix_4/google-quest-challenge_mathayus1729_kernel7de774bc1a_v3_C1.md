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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.24084) has done: 'I fix the import error by removing the TensorFlow import that conflicts with the protobuf version, add the necessary NLTK downloads, replace the TensorFlow‑based correlation with a NumPy version, and lower the training epochs from 10 to 3 to modestly reduce the model’s performance and bring the score closer to the target. All changes keep the original modeling pipeline intact and ensure a valid submission.csv is written.'

# 9. Code solution

## === cell 0
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import nltk
import gc, pickle
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

nltk.download("stopwords")
nltk.download("punkt")

eng_stopwords = set(stopwords.words("english"))

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
    "didn't": "did not",
    "tryin'": "trying",
}


def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    text = [w for w in text if w not in eng_stopwords]
    return " ".join(text)


def _get_mispell(mispell_dict):
    mispell_re = re.compile("s)" % "|".join(mispell_dict.keys()))
    return mispell_dict, mispell_re


def replace_typical_misspell(text):
    mispellings, mispellings_re = _get_mispell(mispell_dict)
    return mispellings_re.sub(lambda match: mispellings[match.group(0)], text)


def clean_data(df, columns: list):
    for col in columns:
        df[col] = df[col].astype(str).apply(lambda x: clean_text(x))
        df[col] = df[col].apply(replace_typical_misspell)
    return df


def get_tfidf_features(data, dims=256):
    tfidf = TfidfVectorizer(ngram_range=(1, 3))
    tsvd = TruncatedSVD(n_components=dims, n_iter=5)
    tfquestion_title = tfidf.fit_transform(data["question_title"].values)
    tfquestion_title = tsvd.fit_transform(tfquestion_title)

    tfquestion_body = tfidf_transform(data["question_body"].values)
    tfquestion_body = tsvd.fit_transform(tfquestion_body)

    tfanswer = tfidf.fit_transform(data["answer"].values)
    tfanswer = tsvd.fit_transform(tfanswer)

    return tfquestion_title, tfquestion_body, tfanswer


def correlation(x, y):
    """NumPy based Pearson correlation (used only for reference)."""
    x = np.asarray(x)
    y = np.asarray(y)
    mx = x.mean()
    my = y.mean()
    xm, ym = x - mx, y - my
    r_num = np.mean(xm * ym)
    r_den = xm.std() * ym.std()
    return r_num / r_den




## === cell 2
from keras.layers import (
    Dense,
    Dropout,
    Embedding,
    LSTM,
    Bidirectional,
    Input,
    Concatenate,
)
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

df_train = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
df_test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")
df_submission = pd.read_csv(
    "/kaggle/input/google-quest-challenge/sample_submission.csv"
)

tokens = []


def get_words(col):
    global tokens
    toks = []
    for sent in sent_tokenize(col):
        words = word_tokenize(sent)
        tokens.extend(words)
        toks.extend(words)
    return toks


def convert_to_indx(col, word2idx, unknown_idx):
    return [word2idx.get(word, unknown_idx) for word in col]


def LSTM_model(df_train, df_test, df_submission):
    columns = ["question_title", "question_body", "answer"]
    df_train = clean_data(df_train, columns)
    df_test = clean_data(df_test, columns)

    for col in columns:
        df_train[col] = df_train[col].apply(lambda x: get_words(x))
        df_test[col] = df_test[col].apply(lambda x: get_words(x))

    vocab = sorted(set(tokens))
    vocab_size = len(vocab) + 1  # extra slot for unknown tokens
    unknown_idx = vocab_size - 1

    word2idx = {word: idx for idx, word in enumerate(vocab)}

    for col in columns:
        df_train[col] = df_train[col].apply(
            lambda x: convert_to_indx(x, word2idx, unknown_idx)
        )
        df_test[col] = df_test[col].apply(
            lambda x: convert_to_indx(x, word2idx, unknown_idx)
        )

    maxlen = 50

    X_train_question_title = pad_sequences(
        df_train["question_title"], maxlen=maxlen, padding="post"
    )
    X_train_question_body = pad_sequences(
        df_train["question_body"], maxlen=maxlen, padding="post"
    )
    X_train_answer = pad_sequences(df_train["answer"], maxlen=maxlen, padding="post")

    X_test_question_title = pad_sequences(
        df_test["question_title"], maxlen=maxlen, padding="post"
    )
    X_test_question_body = pad_sequences(
        df_test["question_body"], maxlen=maxlen, padding="post"
    )
    X_test_answer = pad_sequences(df_test["answer"], maxlen=maxlen, padding="post")

    target_columns = df_submission.columns[1:]
    y_train = df_train[target_columns]

    inpqt = Input(shape=(maxlen,), name="inpqt")
    inpqb = Input(shape=(maxlen,), name="inpqb")
    inpan = Input(shape=(maxlen,), name="inpan")
    Eqt = Embedding(vocab_size, 200, input_length=maxlen)(inpqt)
    Eqb = Embedding(vocab_size, 200, input_length=maxlen)(inpqb)
    Ean = Embedding(vocab_size, 200, input_length=maxlen)(inpan)

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
        epochs=1,
        validation_split=0.2,
        verbose=2,
    )

    y_test = model.predict(
        {
            "inpqt": X_test_question_title,
            "inpqb": X_test_question_body,
            "inpan": X_test_answer,
        },
        batch_size=128,
    )

    outp = {"qa_id": df_test["qa_id"]}
    for idx, col in enumerate(target_columns):
        outp[col] = y_test[:, idx]
    submission_df = pd.DataFrame(outp)
    submission_df.to_csv("submission.csv", index=False)


LSTM_model(df_train, df_test, df_submission)

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
