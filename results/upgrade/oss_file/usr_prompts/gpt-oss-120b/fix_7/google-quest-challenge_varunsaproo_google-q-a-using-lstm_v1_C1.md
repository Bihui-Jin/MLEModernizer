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

0.2593

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.26304) has done: 'I fix the import order, set the protobuf environment variable before TensorFlow is imported, and combine the initial data‑loading steps into a single cell so that `train_df`, `test_df`, and `TARGET_COLS` are defined before they are used. This resolves the NameError issues and the TensorFlow import error while keeping the core modeling logic unchanged.'
- What this solution (achieved 0.25986) has done: 'I import TensorFlow right after setting the protobuf environment variable (before Gensim is loaded) to avoid the protobuf‑related AttributeError, and remove the separate TensorFlow import in the later cell. This fixes the runtime error while keeping the original model architecture and training unchanged, so the score stays near the current (already above the target).'
- What this solution (achieved 0.2593) has done: 'The fix moves the protobuf environment variable setting to the very top, adds a safe fallback import for TensorFlow (using `tensorflow.compat.v1` if the standard import fails), and silences unnecessary TensorFlow logs. This resolves the `AttributeError` on import while keeping the original model, preprocessing, and training logic unchanged, ensuring a valid `submission.csv` is produced and the score remains above the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF warnings

try:
    import tensorflow as tf
except Exception as e:  # pragma: no cover
    import tensorflow.compat.v1 as tf

    tf.disable_eager_execution()

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
from nltk.stem import LancasterStemmer, PorterStemmer, WordNetLemmatizer
from gensim.models import KeyedVectors
import gensim.downloader as api
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")

train_df = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
test_df = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")
TARGET_COLS = train_df.columns[-30:].tolist()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
w2v_model = api.load("glove-wiki-gigaword-100")  # 100‑dim embeddings
vector_dim = 100
stop_words = stopwords.words("english")




## === cell 2
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
rules = {
    "'t": " not",
    "'cause": " because",
    "'ve": " have",
    "'t": " not",
    "'s": " is",
    "'d": " had",
}




## === cell 3
def preprocess_string(s):
    if s == "" or pd.isna(s):
        return ""
    s = re.sub(r"https{0,1}:\/\/([^\s*\n]*)?", " ", s)
    tokens = s.split()
    tokens = [contractions.get(tok, tok).lower() for tok in tokens]
    for i, tok in enumerate(tokens):
        for key, repl in rules.items():
            tokens[i] = tokens[i].replace(key, repl)
    tokens = [re.sub("[^a-zA-Z\s]+", " ", x) for x in tokens]
    s = " ".join(tokens).strip()
    s = re.sub("\s+", " ", s)
    return s


train_df["clean_title"] = train_df.question_title.apply(preprocess_string)
train_df["clean_question_body"] = train_df.question_body.apply(preprocess_string)
train_df["clean_answer"] = train_df.answer.apply(preprocess_string)

test_df["clean_title"] = test_df.question_title.apply(preprocess_string)
test_df["clean_question_body"] = test_df.question_body.apply(preprocess_string)
test_df["clean_answer"] = test_df.answer.apply(preprocess_string)




## === cell 4
def get_embed(series, time_steps, vector_dim):
    final_embed = []
    empty_vec = np.zeros(vector_dim, dtype=np.float32)
    for text in series:
        tokens = text.split()
        embed_seq = []
        for i in range(time_steps):
            if i < len(tokens):
                token = tokens[i]
                if token in w2v_model:
                    embed_seq.append(w2v_model[token])
                else:
                    embed_seq.append(empty_vec)
            else:
                embed_seq.append(empty_vec)
        embed_seq = np.vstack(embed_seq).astype(np.float32)
        final_embed.append(embed_seq)
    return np.stack(final_embed, axis=0)


train_title_embed = get_embed(train_df.clean_title, 30, vector_dim)
train_question_body_embed = get_embed(train_df.clean_question_body, 133, vector_dim)
train_answer_embed = get_embed(train_df.clean_answer, 133, vector_dim)

test_title_embed = get_embed(test_df.clean_title, 30, vector_dim)
test_question_body_embed = get_embed(test_df.clean_question_body, 133, vector_dim)
test_answer_embed = get_embed(test_df.clean_answer, 133, vector_dim)




## === cell 5
def create_model():
    i1 = tf.keras.Input(shape=(None, vector_dim), dtype=tf.float32)
    i2 = tf.keras.Input(shape=(None, vector_dim), dtype=tf.float32)
    i3 = tf.keras.Input(shape=(None, vector_dim), dtype=tf.float32)
    concat = tf.keras.layers.Concatenate(axis=1)([i1, i2, i3])
    lstm = tf.keras.layers.LSTM(128)(concat)
    out = tf.keras.layers.Dense(30, activation="sigmoid")(lstm)
    model = tf.keras.Model(inputs=[i1, i2, i3], outputs=out)
    return model




## === cell 6
def SpearmanCorrCoeff(A, B):
    overall = 0.0
    x1 = np.random.normal(loc=1e-8, scale=1e-12, size=A.shape[0])
    x2 = np.random.normal(loc=1e-8, scale=1e-12, size=B.shape[0])
    for idx in range(30):
        overall += spearmanr(A[:, idx] + x1, B[:, idx] + x2).correlation
    return overall / 30




## === cell 7
final_outputs = train_df[TARGET_COLS].astype(np.float32).values




## === cell 8
gkf = GroupKFold(n_splits=5)
valid_preds = []
for fold, (train_idx, valid_idx) in enumerate(gkf.split(train_df, groups=train_df.url)):
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
            train_inputs, final_outputs[train_idx], epochs=30, batch_size=16, verbose=0
        )

        preds = model.predict(val_inputs, verbose=0)
        valid_preds.append(preds)
        print(
            f"Fold {fold} validation score = {SpearmanCorrCoeff(final_outputs[valid_idx], preds):.5f}"
        )




## === cell 9
test_preds = model.predict(
    [test_title_embed, test_question_body_embed, test_answer_embed], verbose=0
)

label_names = TARGET_COLS
submission = pd.DataFrame(
    np.concatenate([test_df["qa_id"].values.reshape(-1, 1), test_preds], axis=1),
    columns=["qa_id"] + label_names,
)

submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
