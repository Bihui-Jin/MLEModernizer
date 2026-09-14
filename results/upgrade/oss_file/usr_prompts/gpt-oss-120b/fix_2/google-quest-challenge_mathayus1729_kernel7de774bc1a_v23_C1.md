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

No external packages required in the script and installed.

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

0.2742426069204381

# 6. Current score

0.09504

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.09504) has done: 'I fixed the import error for tensorflow_hub, added a safe fallback for the missing embedding model, and replaced the BERT‑based feature extraction with a TF‑IDF + SVD pipeline that already exists in the notebook. This keeps the original architecture (dense layers on top of text features) while guaranteeing a valid `submission.csv` is written without crashing.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

try:
    import tensorflow_hub as hub
except Exception:
    hub = None  # fallback if tensorflow_hub cannot be imported
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import nltk
import keras.backend as K
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string
from keras.preprocessing.sequence import pad_sequences

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
import gc, os, pickle
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
    stops = set(stopwords.words("english"))
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
        df[col] = df[col].apply(lambda x: clean_text(x.lower()))
        df[col] = df[col].apply(lambda x: replace_typical_misspell(x))
    return df


def plot_freq_dist(train_data):
    freq_dist = FreqDist(
        [
            word
            for text in train_data["question_body"].str.replace(
                "[^a-za-z0-9^,!.\/+-=]", " "
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




## === cell 2
def bert_model(
    df_train, df_test, df_submission, batch_size=8, epochs=4, hidden_layers=[300, 90]
):
    """
    Original function attempted to load a TensorFlow Hub sentence‑embedding model.
    The required module is not present, causing an OSError. This version falls back
    to a TF‑IDF + SVD representation (via get_tfidf_features) when the hub model
    cannot be loaded, preserving the rest of the pipeline unchanged.
    """
    use_hub = False
    module_url = "/kaggle/input/sent-embed-model"
    if hub is not None and tf.io.gfile.exists(module_url):
        try:
            model = hub.load(module_url)
            use_hub = True
        except Exception:
            use_hub = False

    if use_hub:

        def embed(input):
            return model(input)

        qt_train = np.array(embed(df_train["question_title"]))
        qb_train = np.array(embed(df_train["question_body"]))
        an_train = np.array(embed(df_train["answer"]))
        qt_test = np.array(embed(df_test["question_title"]))
        qb_test = np.array(embed(df_test["question_body"]))
        an_test = np.array(embed(df_test["answer"]))
    else:
        tf_qt, tf_qb, tf_an = get_tfidf_features(df_train, dims=256)
        X_train = np.concatenate([tf_qt, tf_qb, tf_an], axis=1)

        tf_qt_t, tf_qb_t, tf_an_t = get_tfidf_features(df_test, dims=256)
        X_test = np.concatenate([tf_qt_t, tf_qb_t, tf_an_t], axis=1)

    target_columns = df_submission.columns[1:]
    y_train = df_train[target_columns].values

    model_dense = tf.keras.models.Sequential()
    model_dense.add(tf.keras.layers.Dense(hidden_layers[0], activation="selu"))
    model_dense.add(tf.keras.layers.Dropout(0.2))
    for h in hidden_layers[1:]:
        model_dense.add(tf.keras.layers.Dense(h, activation="selu"))
        model_dense.add(tf.keras.layers.Dropout(0.2))
    model_dense.add(tf.keras.layers.Dense(30, activation="sigmoid"))

    model_dense.compile("adam", "binary_crossentropy", metrics=["accuracy"])

    if use_hub:
        X_train = np.concatenate([qt_train, qb_train, an_train], axis=1)
        X_test = np.concatenate([qt_test, qb_test, an_test], axis=1)

    model_dense.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=2,
    )

    y_test = model_dense.predict(X_test)

    outp = {"qa_id": df_test["qa_id"].values}
    for i, col in enumerate(target_columns):
        outp[col] = y_test[:, i]
    my_submission = pd.DataFrame(outp)
    my_submission.to_csv("submission.csv", index=False)


df_train = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
df_test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")
df_submission = pd.read_csv(
    "/kaggle/input/google-quest-challenge/sample_submission.csv"
)

bert_model(df_train, df_test, df_submission, epochs=10)
