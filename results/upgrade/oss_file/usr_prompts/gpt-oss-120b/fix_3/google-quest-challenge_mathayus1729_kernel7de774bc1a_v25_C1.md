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

0.2632660942213997

# 6. Current score

0.13329

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'I fixed the import error for tensorflow_hub, added a safe‑check for the missing embedding model, and provided a fallback that predicts the mean target value from the training set when the embedding cannot be loaded. This ensures the script runs end‑to‑end and always creates a valid submission.csv with the correct columns.'
- What this solution (achieved 0.13329) has done: 'I guard TensorFlow‑related imports so they don’t crash the notebook, replace the TensorFlow‑based file‑exist check with a pure‑Python one, and improve the “mean‑baseline” fallback by training a lightweight Ridge regression on TF‑IDF features. This keeps the original NN‑model path unchanged (used when an embedding is available) while providing a deterministic, fast fallback that yields a valid submission.csv and a reasonable score.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

try:
    import tensorflow as tf
except Exception:
    tf = None
try:
    import tensorflow_hub as hub
except Exception:
    hub = None

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
import gc, pickle
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge


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


def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if w not in stops]
    return " ".join(text)


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
        df[col] = df[col].astype(str).apply(lambda x: clean_text(x))
        df[col] = df[col].apply(lambda x: replace_typical_misspell(x))
    return df


def get_tfidf_features(data, dims=256):
    tfidf = TfidfVectorizer(ngram_range=(1, 3))
    tsvd = TruncatedSVD(n_components=dims, n_iter=5, random_state=42)

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
from keras.layers import (
    Dense,
    Dropout,
    Embedding,
    LSTM,
    Bidirectional,
    Input,
    Concatenate,
    GRU,
)
from keras.models import Model

df_train = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
df_test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")
df_submission = pd.read_csv(
    "/kaggle/input/google-quest-challenge/sample_submission.csv"
)

tokens = []


def get_words(col):
    global tokens
    toks = []
    for x in sent_tokenize(col):
        tokens += word_tokenize(x)
        toks += word_tokenize(x)
    return toks


def convert_to_indx(col, word2idx, vocab_size):
    return [word2idx[word] if word in word2idx else vocab_size for word in col]


def attention_3d_block_self(hidden_states, rnn_units=64):
    hidden_size = int(hidden_states.shape[2])
    score_first_part = Dense(hidden_size, use_bias=False)(hidden_states)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,))(hidden_states)
    score = Dot([2, 1])([score_first_part, h_t])
    attention_weights = Activation("softmax")(score)
    context_vector = Dot([1, 1])([hidden_states, attention_weights])
    pre_activation = Concatenate()([context_vector, h_t])
    attention_vector = Dense(rnn_units * 2, use_bias=False, activation="tanh")(
        pre_activation
    )
    return attention_vector




## === cell 3
def nnlm128Model(
    df_train, df_test, df_submission, batch_size=8, epochs=4, hidden_layers=[90]
):
    """
    Tries to load a pre‑trained embedding from /kaggle/input/nnlmmodel.
    If the model directory does not exist (common in the current environment),
    falls back to a lightweight Ridge‑regression model built on TF‑IDF features.
    This guarantees a valid CSV output while improving over a simple mean baseline.
    """
    if len(hidden_layers) < 1:
        print("Non‑Empty Hidden Layers List Required!")
        return

    embed = None
    embed_path = "/kaggle/input/nnlmmodel"
    if hub is not None and tf is not None and os.path.isdir(embed_path):
        try:
            embed = hub.load(embed_path)
        except Exception as e:
            print(f"Embedding load failed ({e}); falling back to TF‑IDF baseline.")
            embed = None
    else:
        print("Embedding path not found or TF/HUB unavailable; using TF‑IDF baseline.")

    target_columns = df_submission.columns[1:]  # exclude qa_id
    y_train = df_train[target_columns].values

    if embed is not None:
        qt_train = np.array(embed(df_train["question_title"]))
        qb_train = np.array(embed(df_train["question_body"]))
        an_train = np.array(embed(df_train["answer"]))
        X_train = np.concatenate([qt_train, qb_train, an_train], axis=1)

        qt_test = np.array(embed(df_test["question_title"]))
        qb_test = np.array(embed(df_test["question_body"]))
        an_test = np.array(embed(df_test["answer"]))
        X_test = np.concatenate([qt_test, qb_test, an_test], axis=1)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(hidden_layers[0], activation="relu"))
        model.add(tf.keras.layers.Dropout(0.2))
        for h in hidden_layers[1:]:
            model.add(tf.keras.layers.Dense(h, activation="relu"))
            model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(30, activation="sigmoid"))
        model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            verbose=0,
        )
        y_test = model.predict(X_test)
    else:
        print("Training Ridge regression on TF‑IDF features...")
        qt_train, qb_train, an_train = get_tfidf_features(df_train, dims=128)
        X_train = np.concatenate([qt_train, qb_train, an_train], axis=1)

        qt_test, qb_test, an_test = get_tfidf_features(df_test, dims=128)
        X_test = np.concatenate([qt_test, qb_test, an_test], axis=1)

        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train, y_train)
        y_test = ridge.predict(X_test)

    outp = {"qa_id": df_test["qa_id"].values}
    for idx, col in enumerate(target_columns):
        outp[col] = y_test[:, idx]
    my_submission = pd.DataFrame(outp)
    my_submission.to_csv("submission.csv", index=False)
    print("submission.csv written with shape:", my_submission.shape)


nnlm128Model(df_train, df_test, df_submission, epochs=1)
