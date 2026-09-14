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
scipy==1.15.3
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

0.1726910959077161

# 6. Current score

0.20163

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.20163) has done: 'I fix the environment-breaking import error by removing unused `tensorflow-text`/protobuf-triggering imports and keeping only the libraries actually used by your pipeline. I also fix the missing GloVe file by switching to a locally-trained Word2Vec embedding (same embedding-lookup logic and LSTM model, just a different way to obtain `w2v_model` so the notebook can run end-to-end). Then I correct the target column selection to exactly match `sample_submission.csv` (30 labels), which is necessary for correct training/prediction shapes and a valid submission. Finally, I ensure folds actually run (your `fold in [1,3,5]` never triggers with 5 splits) and produce a `submission.csv` with the required columns and [0,1] clipping.'

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input/google-quest-challenge"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)

print(
    "train_df:",
    train_df.shape,
    "test_df:",
    test_df.shape,
    "sample_submission:",
    sample_submission.shape,
)



## === cell 1
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

from gensim.models import Word2Vec, KeyedVectors

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
stop_words = stopwords.words("english")

vector_dim = 100


def _iter_corpus_for_w2v(df):
    cols = ["question_title", "question_body", "answer"]
    for _, row in df[cols].fillna("").iterrows():
        text = " ".join(
            [row["question_title"], row["question_body"], row["answer"]]
        ).strip()
        if not text:
            continue
        yield text.split()


sentences = list(
    _iter_corpus_for_w2v(pd.concat([train_df, test_df], axis=0, ignore_index=True))
)

w2v = Word2Vec(
    sentences=sentences,
    vector_size=vector_dim,
    window=5,
    min_count=2,
    workers=os.cpu_count() or 2,
    sg=1,  # skip-gram
    negative=10,
    epochs=5,
    seed=42,
)
w2v_model: KeyedVectors = w2v.wv

print("Trained Word2Vec vocab size:", len(w2v_model), "vector_dim:", vector_dim)



## === cell 3
contractions = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}

rules = {"'t": " not", "'cause": " because", "'ve": " have", "'s": " is", "'d": " had"}




## === cell 4
def preprocess_string(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    if s == "":
        return ""
    s = re.sub(r"https{0,1}:\/\/([^\s*\n]*)?", " ", s)
    tokens = s.split()

    tokens = [contractions.get(tok, tok).lower() for tok in tokens]
    for i in range(len(tokens)):
        for key, item in rules.items():
            tokens[i] = tokens[i].replace(key.lower(), item)

    tokens = [re.sub("[^a-zA-Z\\s]+", " ", x) for x in tokens]
    s = " ".join(tokens).strip()
    s = re.sub("\\s+", " ", s)
    return s




## === cell 5
train_df["clean_title"] = train_df["question_title"].apply(preprocess_string)
train_df["clean_question_body"] = train_df["question_body"].apply(preprocess_string)
train_df["clean_answer"] = train_df["answer"].apply(preprocess_string)

test_df["clean_title"] = test_df["question_title"].apply(preprocess_string)
test_df["clean_question_body"] = test_df["question_body"].apply(preprocess_string)
test_df["clean_answer"] = test_df["answer"].apply(preprocess_string)




## === cell 6
def get_embed(input_series, time_steps, vector_dim):
    final_embed = []
    empty_array = np.zeros(shape=(vector_dim,), dtype=np.float32)

    for text in input_series.values:
        lst = str(text).split()
        tmp_embed = []
        for i in range(time_steps):
            if i < len(lst) and lst[i] in w2v_model:
                tmp_embed.append(w2v_model[lst[i]])
            else:
                tmp_embed.append(empty_array)
        tmp_embed = np.vstack(tmp_embed).astype(np.float32)
        final_embed.append(tmp_embed)

    return np.stack(final_embed, axis=0).astype(np.float32)


train_title_embed = get_embed(train_df.clean_title, 30, vector_dim)
train_question_body_embed = get_embed(train_df.clean_question_body, 133, vector_dim)
train_answer_embed = get_embed(train_df.clean_answer, 133, vector_dim)

test_title_embed = get_embed(test_df.clean_title, 30, vector_dim)
test_question_body_embed = get_embed(test_df.clean_question_body, 133, vector_dim)
test_answer_embed = get_embed(test_df.clean_answer, 133, vector_dim)

print(
    "Embeds:",
    train_title_embed.shape,
    train_question_body_embed.shape,
    train_answer_embed.shape,
    test_title_embed.shape,
    test_question_body_embed.shape,
    test_answer_embed.shape,
)




## === cell 7
def create_model():
    i1 = tf.keras.Input(shape=(None, vector_dim), dtype=tf.float32)
    i2 = tf.keras.Input(shape=(None, vector_dim), dtype=tf.float32)
    i3 = tf.keras.Input(shape=(None, vector_dim), dtype=tf.float32)
    input_concat = tf.keras.layers.Concatenate(axis=1)([i1, i2, i3])

    lstm = tf.keras.layers.LSTM(128)(input_concat)
    dense = tf.keras.layers.Dense(30, activation="sigmoid")(lstm)
    model = tf.keras.Model(inputs=[i1, i2, i3], outputs=[dense])
    return model




## === cell 8
def SpearmanCorrCoeff(A, B):
    overall_score = 0.0
    x1 = np.random.normal(loc=1e-8, scale=1e-12, size=A.shape[0])
    x2 = np.random.normal(loc=1e-8, scale=1e-12, size=B.shape[0])
    for index in range(30):
        overall_score += spearmanr(A[:, index] + x1, B[:, index] + x2).correlation
    return overall_score / 30.0


def tf_SpearmanCorrCoeff(A, B):
    result = tf.numpy_function(SpearmanCorrCoeff, [A, B], Tout=tf.double)
    return result




## === cell 9
target_cols = sample_submission.columns[1:].tolist()
final_outputs = train_df[target_cols].values.astype(np.float32)

print(
    "Number of targets:", len(target_cols), "final_outputs shape:", final_outputs.shape
)



## === cell 10
gkf = GroupKFold(n_splits=5).split(X=train_df.url, groups=train_df.url)

valid_preds = []
last_model = None

for fold, (train_idx, valid_idx) in enumerate(gkf):
    if fold in [1, 3]:
        tf.keras.backend.clear_session()
        model = create_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(loss="binary_crossentropy", optimizer=optimizer)

        train_inputs = [
            train_title_embed[train_idx],
            train_question_body_embed[train_idx],
            train_answer_embed[train_idx],
        ]
        val_inputs = [
            train_title_embed[valid_idx],
            train_question_body_embed[valid_idx],
            train_answer_embed[valid_idx],
        ]

        model.fit(
            train_inputs, final_outputs[train_idx], epochs=30, batch_size=16, verbose=2
        )

        fold_valid_pred = model.predict(val_inputs, batch_size=64, verbose=0)
        valid_preds.append(fold_valid_pred)

        print(
            "fold",
            fold,
            "validation score = ",
            SpearmanCorrCoeff(final_outputs[valid_idx], fold_valid_pred),
        )
        last_model = model

if last_model is None:
    tf.keras.backend.clear_session()
    last_model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    last_model.compile(loss="binary_crossentropy", optimizer=optimizer)
    last_model.fit(
        [train_title_embed, train_question_body_embed, train_answer_embed],
        final_outputs,
        epochs=30,
        batch_size=16,
        verbose=2,
    )



## === cell 11
test_preds = last_model.predict(
    [test_title_embed, test_question_body_embed, test_answer_embed],
    batch_size=64,
    verbose=0,
)
test_preds = np.clip(test_preds, 0.0, 1.0)

submission = pd.read_csv(sample_path)
submission[target_cols] = test_preds.astype(np.float32)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
