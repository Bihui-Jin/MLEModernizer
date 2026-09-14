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

# 5. Target score

0.6511796295178905

# 6. Current score

0.0588

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0588) has done: 'The changes focus on eliminating unnecessary Python‑level overhead during training and freeing memory after loading the large GloVe file.  After building the embedding matrix we delete the original dictionary and run garbage collection.  Training now uses a `tf.data.Dataset` pipeline, which streams batches efficiently and matches the original shuffling behavior, keeping the model architecture, optimizer, epochs and other hyper‑parameters unchanged.  These tweaks speed up the fit step enough to stay under the 600‑second limit while preserving exact results.'

# 9. Code solution

## === cell 0
import os, re, gc, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", None)

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "/kaggle/input/movie-review-sentiment-analysis-kernels-only"
train_file = os.path.join(DATA_DIR, "train.tsv")
test_file = os.path.join(DATA_DIR, "test.tsv")
df_train = pd.read_csv(train_file, sep="\t")
df_test = pd.read_csv(test_file, sep="\t")
sub = pd.read_csv(os.path.join(DATA_DIR, "sampleSubmission.csv"))



## === cell 2
print(df_train.head())
print(df_test.head())



## === cell 3
df_train["Phrase"] = df_train["Phrase"].str.replace("n't", "not")
df_test["Phrase"] = df_test["Phrase"].str.replace("n't", "not")



## === cell 4
df_train["Phrase"] = df_train["Phrase"].apply(lambda x: re.sub(r"[0-9]+", "0", x))
df_test["Phrase"] = df_test["Phrase"].apply(lambda x: re.sub(r"[0-9]+", "0", x))



## === cell 5
seed = 101
np.random.seed(seed)
tf.random.set_seed(seed)

X = df_train["Phrase"].astype(str)
temp = df_test["Phrase"].astype(str)  # hold test phrases for final prediction
y = to_categorical(df_train["Sentiment"])
num_classes = df_train["Sentiment"].nunique()
print("Number of classes:", num_classes)



## === cell 6
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=seed
)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)



## === cell 7
max_features = 15000
tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
tokenizer.fit_on_texts(list(X_train))

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
temp_seq = tokenizer.texts_to_sequences(temp)

max_len = 50
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len)
temp_pad = pad_sequences(temp_seq, maxlen=max_len)
print("Padded shapes:", X_train_pad.shape, X_val_pad.shape)




## === cell 8
def get_model(max_features, embed_dim, embedding_matrix=None):
    K.clear_session()
    model = Sequential()
    if embedding_matrix is not None:
        model.add(
            Embedding(
                input_dim=max_features,
                output_dim=embed_dim,
                weights=[embedding_matrix],
                input_length=max_len,
                trainable=False,
            )
        )
    else:
        model.add(
            Embedding(
                input_dim=max_features, output_dim=embed_dim, input_length=max_len
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




## === cell 9
embed_dim = 200
embedding_file = "/kaggle/input/globalvectorsforwordrepresentation/glove.6B.200d.txt"

if os.path.exists(embedding_file):

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype="float32")

    embeddings_index = dict(
        get_coefs(*line.rstrip().rsplit(" "))
        for line in open(embedding_file, encoding="utf8")
    )
    print("Found %s word vectors." % len(embeddings_index))

    word_index = tokenizer.word_index
    num_words = min(max_features, len(word_index) + 1)
    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embedding_matrix = np.random.normal(
        emb_mean,
        emb_std,
        (num_words, embed_dim),
    )
    for word, i in word_index.items():
        if i >= max_features:
            continue
        vec = embeddings_index.get(word)
        if vec is not None:
            embedding_matrix[i] = vec
    max_features = embedding_matrix.shape[0]
    del embeddings_index, all_embs
    gc.collect()
else:
    print("Embedding file not found – using random initialization.")
    embedding_matrix = None  # random init inside model



## === cell 10
batch_size = 128
epochs = 10
model = get_model(max_features, embed_dim, embedding_matrix)

train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train_pad, y_train))
    .shuffle(buffer=len(X_train_pad), seed=seed)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((X_val_pad, y_val))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    verbose=2,
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4240980059.py in <cell line: 0>()
      6 train_dataset = (
      7     tf.data.Dataset.from_tensor_slices((X_train_pad, y_train))
----> 8     .shuffle(buffer=len(X_train_pad), seed=seed)
      9     .batch(batch_size)
     10     .prefetch(tf.data.AUTOTUNE)

TypeError: DatasetV2.shuffle() got an unexpected keyword argument 'buffer'

## === cell 11
pred_probs = model.predict(temp_pad, batch_size=batch_size, verbose=0)
pred_labels = np.argmax(pred_probs, axis=1)
sub["Sentiment"] = pred_labels
sub.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print(sub.head())
