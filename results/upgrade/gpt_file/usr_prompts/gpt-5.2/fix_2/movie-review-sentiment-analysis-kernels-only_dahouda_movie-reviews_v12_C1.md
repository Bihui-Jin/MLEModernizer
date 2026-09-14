# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the sentiment of phrases.

## Metric
Classification accuracy.

## Submission Format
For each phrase in the test set, predict a label for the sentiment. Your submission should have a header and look like the following:

```
PhraseId,Sentiment
156061,2
156062,2
156063,2
...
```

## Dataset
The dataset is comprised of tab-separated files with phrases. Each phrase has a PhraseId. Each sentence has a SentenceId.

The sentiment labels are:

0 - negative

1 - somewhat negative

2 - neutral

3 - somewhat positive

4 - positive

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        input/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        working/
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
```

-> data/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> data/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import re
import gc
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

pd.set_option("display.max_colwidth", None)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:50]:
        print(os.path.join(dirname, filename))



## === cell 1
DATA_DIR = "/kaggle/input/movie-review-sentiment-analysis-kernels-only"
train_file = os.path.join(DATA_DIR, "train.tsv.zip")
test_file = os.path.join(DATA_DIR, "test.tsv.zip")

df_train = pd.read_table(train_file)
df_test = pd.read_table(test_file)

sub_path = os.path.join(DATA_DIR, "sampleSubmission.csv")
sub = pd.read_csv(sub_path)

print(df_train.shape, df_test.shape, sub.shape)
print(df_train.columns)



## === cell 2
print(df_train.head())



## === cell 3
print(df_test.head())



## === cell 4
df_train["Phrase"] = df_train["Phrase"].astype(str)
df_test["Phrase"] = df_test["Phrase"].astype(str)



## === cell 5
df_train.Phrase = df_train.Phrase.str.replace("n't", "not", regex=False)
df_test.Phrase = df_test.Phrase.str.replace("n't", "not", regex=False)



## === cell 6
df_train.Phrase = df_train.Phrase.apply(lambda x: re.sub(r"[0-9]+", "0", x))
df_test.Phrase = df_test.Phrase.apply(lambda x: re.sub(r"[0-9]+", "0", x))



## === cell 7
seed = 101
np.random.seed(seed)
tf.random.set_seed(seed)

X = df_train["Phrase"]
temp = df_test["Phrase"]

num_classes = df_train["Sentiment"].nunique()
y = to_categorical(df_train["Sentiment"], num_classes=num_classes)

print("Number of classes:", num_classes)
print("X:", X.shape, "temp:", temp.shape, "y:", y.shape)



## === cell 8
y_labels = df_train["Sentiment"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y_labels, random_state=seed
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



## === cell 9
max_features = 15000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
temp_seq = tokenizer.texts_to_sequences(temp)

max_words = 50
X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=max_words)
X_test_pad = sequence.pad_sequences(X_test_seq, maxlen=max_words)
temp_pad = sequence.pad_sequences(temp_seq, maxlen=max_words)

print(X_train_pad.shape, X_test_pad.shape, temp_pad.shape)

del X_train_seq, X_test_seq, temp_seq
gc.collect()



## === cell 10
batch_size = 128
epochs = 10


def get_model(max_features, embed_dim, embedding_matrix, input_len, num_classes):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    K.clear_session()

    model = Sequential()
    model.add(
        Embedding(
            input_dim=max_features,
            output_dim=embed_dim,
            input_length=input_len,
            weights=[embedding_matrix],
            trainable=True,
        )
    )
    model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.summary()
    return model




## === cell 11
embed_dim = 200
word_index = tokenizer.word_index
num_words = min(max_features, len(word_index) + 1)

rng = np.random.default_rng(seed)
embedding_matrix = rng.normal(loc=0.0, scale=0.05, size=(num_words, embed_dim)).astype(
    np.float32
)
max_features_eff = embedding_matrix.shape[0]

print(
    "Using random embedding matrix:",
    embedding_matrix.shape,
    "max_features_eff:",
    max_features_eff,
)



## === cell 12
model = get_model(
    max_features=max_features_eff,
    embed_dim=embed_dim,
    embedding_matrix=embedding_matrix,
    input_len=X_train_pad.shape[1],
    num_classes=num_classes,
)

history = model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_test_pad, y_test),
    epochs=epochs,
    batch_size=batch_size,
    verbose=2,
)



## === cell 13
pred_proba = model.predict(temp_pad, batch_size=batch_size, verbose=0)
pred = np.argmax(pred_proba, axis=1).astype(int)

sub["Sentiment"] = pred

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print(sub.head())
print("Wrote:", out_path, "rows:", len(sub))
