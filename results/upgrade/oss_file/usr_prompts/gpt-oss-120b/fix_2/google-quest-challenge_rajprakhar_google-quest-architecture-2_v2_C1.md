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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

0.0564005049205663

# 6. Current score

nan

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved nan) has done: 'I fixed the protobuf import error, removed the missing GloVe dependency by using a random embedding matrix, streamlined the tokenization/padding logic, and added a simple training loop (few epochs for quick execution) that correctly creates the model, trains it, and writes a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
import tensorflow as tf

print(tf.__version__)

import re
from tqdm import tqdm
from scipy.stats import spearmanr
import warnings

warnings.simplefilter("ignore")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
PATH = "../input/google-quest-challenge/"

df_train = pd.read_csv(PATH + "train.csv")
df_test = pd.read_csv(PATH + "test.csv")
df_sub = pd.read_csv(PATH + "sample_submission.csv")

output_categories = list(df_train.columns[11:])
print("\nOutput Categories:\n", output_categories)



## === cell 2
stopwords = [
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


def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def preprocess_text(text_series):
    processed = []
    for sent in tqdm(text_series, leave=False):
        sent = decontracted(str(sent))
        sent = sent.replace("\\r", " ").replace("\\n", " ").replace('\\"', " ")
        sent = re.sub("[^A-Za-z0-9]+", " ", sent)
        sent = " ".join(w for w in sent.split() if w.lower() not in stopwords)
        processed.append(sent.lower().strip())
    return processed


df_train["Preproc_Question_Title"] = preprocess_text(df_train["question_title"])
df_train["Preproc_Question_Body"] = preprocess_text(df_train["question_body"])
df_train["Preproc_Answer"] = preprocess_text(df_train["answer"])

df_test["Preproc_Question_Title"] = preprocess_text(df_test["question_title"])
df_test["Preproc_Question_Body"] = preprocess_text(df_test["question_body"])
df_test["Preproc_Answer"] = preprocess_text(df_test["answer"])



## === cell 3
all_train_text = pd.concat(
    [df_train["question_title"], df_train["question_body"], df_train["answer"]]
)
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(all_train_text.tolist())

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1


def seqs(series):
    return tokenizer.texts_to_sequences(series.tolist())


train_seq_qt = seqs(df_train["question_title"])
train_seq_qb = seqs(df_train["question_body"])
train_seq_ans = seqs(df_train["answer"])
test_seq_qt = seqs(df_test["question_title"])
test_seq_qb = seqs(df_test["question_body"])
test_seq_ans = seqs(df_test["answer"])

all_lengths = [len(s) for s in train_seq_qt + train_seq_qb + train_seq_ans]
MAX_SEQUENCE_LENGTH = int(np.percentile(all_lengths, 96))
print("MAX_SEQUENCE_LENGTH:", MAX_SEQUENCE_LENGTH)

train_pad_qt = tf.keras.preprocessing.sequence.pad_sequences(
    train_seq_qt, maxlen=MAX_SEQUENCE_LENGTH
)
train_pad_qb = tf.keras.preprocessing.sequence.pad_sequences(
    train_seq_qb, maxlen=MAX_SEQUENCE_LENGTH
)
train_pad_ans = tf.keras.preprocessing.sequence.pad_sequences(
    train_seq_ans, maxlen=MAX_SEQUENCE_LENGTH
)

test_pad_qt = tf.keras.preprocessing.sequence.pad_sequences(
    test_seq_qt, maxlen=MAX_SEQUENCE_LENGTH
)
test_pad_qb = tf.keras.preprocessing.sequence.pad_sequences(
    test_seq_qb, maxlen=MAX_SEQUENCE_LENGTH
)
test_pad_ans = tf.keras.preprocessing.sequence.pad_sequences(
    test_seq_ans, maxlen=MAX_SEQUENCE_LENGTH
)

embedding_matrix = np.random.normal(size=(vocab_size, 300)).astype(np.float32)



## === cell 4
embedding_layer_total_text = tf.keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=300,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGTH,
    name="Shared_Embedding_Layer",
    trainable=False,
)




## === cell 5
def create_model():
    inp_qt = tf.keras.layers.Input(
        shape=(MAX_SEQUENCE_LENGTH,), name="IP_Question_Title"
    )
    emb_qt = embedding_layer_total_text(inp_qt)

    inp_qb = tf.keras.layers.Input(
        shape=(MAX_SEQUENCE_LENGTH,), name="IP_Question_Body"
    )
    emb_qb = embedding_layer_total_text(inp_qb)

    inp_ans = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), name="IP_Answer")
    emb_ans = embedding_layer_total_text(inp_ans)

    tower_1 = tf.keras.layers.Conv1D(64, 5, activation="relu")(emb_qt)
    tower_2 = tf.keras.layers.Conv1D(64, 5, activation="relu")(emb_qb)
    tower_3 = tf.keras.layers.Conv1D(64, 5, activation="relu")(emb_ans)

    concat = tf.keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
    max_pool = tf.keras.layers.MaxPooling1D(9)(concat)

    t1 = tf.keras.layers.Conv1D(64, 5, activation="relu")(max_pool)
    t2 = tf.keras.layers.Conv1D(64, 7, activation="relu")(max_pool)
    t3 = tf.keras.layers.Conv1D(64, 9, activation="relu")(max_pool)

    concat2 = tf.keras.layers.concatenate([t1, t2, t3], axis=1)
    max_pool2 = tf.keras.layers.MaxPooling1D(9)(concat2)

    convP = tf.keras.layers.Conv1D(64, 9, activation="relu")(max_pool2)
    flat = tf.keras.layers.Flatten()(convP)
    drop = tf.keras.layers.Dropout(0.7)(flat)
    dense = tf.keras.layers.Dense(128, activation="relu")(drop)
    outputs = tf.keras.layers.Dense(30, activation="sigmoid", name="Output")(dense)

    model = tf.keras.models.Model(
        inputs=[inp_qt, inp_qb, inp_ans], outputs=outputs, name="Model_Google_QUEST"
    )
    return model


model_Google_QUEST = create_model()
model_Google_QUEST.summary()



## === cell 6
train_data = {
    "IP_Question_Title": train_pad_qt[:5000],
    "IP_Question_Body": train_pad_qb[:5000],
    "IP_Answer": train_pad_ans[:5000],
}
train_target = df_train[output_categories].values[:5000]

val_data = {
    "IP_Question_Title": train_pad_qt[5000:],
    "IP_Question_Body": train_pad_qb[5000:],
    "IP_Answer": train_pad_ans[5000:],
}
val_target = df_train[output_categories].values[5000:]

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model_Google_QUEST.compile(loss="mean_squared_error", optimizer=optimizer)

model_Google_QUEST.fit(
    train_data,
    train_target,
    validation_data=(val_data, val_target),
    epochs=5,
    batch_size=64,
    verbose=1,
)



## === cell 7
test_data = {
    "IP_Question_Title": test_pad_qt,
    "IP_Question_Body": test_pad_qb,
    "IP_Answer": test_pad_ans,
}
test_pred = model_Google_QUEST.predict(test_data, batch_size=64)

submission_df = pd.concat(
    [
        df_test[["qa_id"]].reset_index(drop=True),
        pd.DataFrame(test_pred, columns=output_categories),
    ],
    axis=1,
)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
