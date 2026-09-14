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

3.7

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

0.6436523260725276

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os

print(os.listdir("../input"))



## === cell 1
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train = pd.read_csv(
    "../input/movie-review-sentiment-analysis-kernels-only/train.tsv", sep="\t"
)
test = pd.read_csv(
    "../input/movie-review-sentiment-analysis-kernels-only/test.tsv", sep="\t"
)
sub = pd.read_csv(
    "../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv",
    sep=",",
)



## === cell 3
x = train["Phrase"].astype(str)
y = train["Sentiment"].astype(int)



## === cell 4
train[["Phrase", "Sentiment"]].head()



## === cell 5
embedding_dim = 300
maxlen = 300
glove_path = "../input/gloveembeddings/glove.6B.300d.txt"

token = Tokenizer()
y = to_categorical(y, num_classes=5)



## === cell 6
token.fit_on_texts(x)



## === cell 7
seq_gen = token.texts_to_sequences_generator(x)
pad_seq = pad_sequences(seq_gen, maxlen=maxlen).astype(np.int32, copy=False)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3979367439.py in <cell line: 0>()
      2 # Correctness: identical tokenization and padding semantics as texts_to_sequences + pad_sequences.
      3 seq_gen = token.texts_to_sequences_generator(x)
----> 4 pad_seq = pad_sequences(seq_gen, maxlen=maxlen).astype(np.int32, copy=False)
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/sequence_utils.py in pad_sequences(sequences, maxlen, dtype, padding, truncating, value)
     76     """
     77     if not hasattr(sequences, "__len__"):
---> 78         raise ValueError("`sequences` must be iterable.")
     79     num_samples = len(sequences)
     80 

ValueError: `sequences` must be iterable.

## === cell 8
vocab_size = len(token.word_index) + 1
print(vocab_size)



## === cell 9
embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

word_index = token.word_index
needed_words = set(word_index.keys())

if os.path.exists(glove_path):
    found = 0
    with open(glove_path, "r", encoding="utf-8", newline="") as f:
        for line in f:
            parts = line.rstrip().split(" ", embedding_dim)
            if len(parts) != embedding_dim + 1:
                continue
            w = parts[0]
            if w not in needed_words:
                continue
            vec = np.fromstring(parts[1], sep=" ", dtype=np.float32)
            if vec.shape[0] != embedding_dim:
                continue
            embedding_matrix[word_index[w]] = vec
            found += 1
    print(f"Loaded {found} in-vocab GloVe vectors (filtered).")
else:
    print(f"GloVe file not found at: {glove_path}")
    print(
        "Proceeding with randomly initialized embedding matrix (trainable=False as in original)."
    )
    rng = np.random.RandomState(42)
    embedding_matrix[1:] = rng.normal(
        loc=0.0, scale=0.05, size=(vocab_size - 1, embedding_dim)
    ).astype(np.float32)



## === cell 10
x_train, x_test, y_train, y_test = train_test_split(
    pad_seq, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/295443994.py in <cell line: 0>()
      1 x_train, x_test, y_train, y_test = train_test_split(
----> 2     pad_seq, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
      3 )
      4 

NameError: name 'pad_seq' is not defined

## === cell 11
model = Sequential()



## === cell 12
model.add(
    Embedding(
        vocab_size,
        embedding_dim,
        input_length=maxlen,
        weights=[embedding_matrix],
        trainable=False,
    )
)



## === cell 13
model.add(LSTM(75))



## === cell 14
model.add(Dense(128, activation="relu"))



## === cell 15
model.add(Dense(5, activation="softmax"))



## === cell 16
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 17
batch_size = 32

data_opts = tf.data.Options()
data_opts.experimental_deterministic = True

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .with_options(data_opts)
    .shuffle(
        buffer_size=min(len(x_train), 100_000), seed=42, reshuffle_each_iteration=True
    )
    .batch(batch_size, drop_remainder=False)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .with_options(data_opts)
    .batch(batch_size, drop_remainder=False)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

history = model.fit(
    train_ds,
    epochs=4,
    validation_data=val_ds,
    verbose=2,
)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1253913745.py in <cell line: 0>()
      6 # Speed: cache after batching to reduce cache overhead; keep shuffle deterministic and semantics intact.
      7 train_ds = (
----> 8     tf.data.Dataset.from_tensor_slices((x_train, y_train))
      9     .with_options(data_opts)
     10     .shuffle(

NameError: name 'x_train' is not defined

## === cell 18
test.head()



## === cell 19
pass



## === cell 20
test.head()



## === cell 21
testing_phrase = test["Phrase"].astype(str)



## === cell 22
test_seq_gen = token.texts_to_sequences_generator(testing_phrase)



## === cell 23
pad_test_seq = pad_sequences(test_seq_gen, maxlen=maxlen).astype(np.int32, copy=False)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/891849214.py in <cell line: 0>()
----> 1 pad_test_seq = pad_sequences(test_seq_gen, maxlen=maxlen).astype(np.int32, copy=False)
      2 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/sequence_utils.py in pad_sequences(sequences, maxlen, dtype, padding, truncating, value)
     76     """
     77     if not hasattr(sequences, "__len__"):
---> 78         raise ValueError("`sequences` must be iterable.")
     79     num_samples = len(sequences)
     80 

ValueError: `sequences` must be iterable.

## === cell 24
pred_ds = (
    tf.data.Dataset.from_tensor_slices(pad_test_seq)
    .batch(1024)
    .prefetch(tf.data.AUTOTUNE)
)
proba = model.predict(pred_ds, verbose=0)
predict = np.argmax(proba, axis=1).astype(int)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/505370022.py in <cell line: 0>()
      1 # Speed: larger predict batch reduces Python/tf.data overhead; does not change outputs.
      2 pred_ds = (
----> 3     tf.data.Dataset.from_tensor_slices(pad_test_seq)
      4     .batch(1024)
      5     .prefetch(tf.data.AUTOTUNE)

NameError: name 'pad_test_seq' is not defined

## === cell 25
predict[0]



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1909701773.py in <cell line: 0>()
----> 1 predict[0]
      2 

NameError: name 'predict' is not defined

## === cell 26
test["Sentiment"] = predict



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3776737316.py in <cell line: 0>()
----> 1 test["Sentiment"] = predict
      2 

NameError: name 'predict' is not defined

## === cell 27
test.head()



## === cell 28
submission = test[["PhraseId", "Sentiment"]]



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3574667596.py in <cell line: 0>()
----> 1 submission = test[["PhraseId", "Sentiment"]]
      2 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['Sentiment'] not in index"

## === cell 29
submission.head()



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2023294942.py in <cell line: 0>()
----> 1 submission.head()
      2 

NameError: name 'submission' is not defined

## === cell 30
submission.to_csv("Submission.csv", index=False)
print("Wrote submission to Submission.csv with shape:", submission.shape)

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2038407195.py in <cell line: 0>()
----> 1 submission.to_csv("Submission.csv", index=False)
      2 print("Wrote submission to Submission.csv with shape:", submission.shape)

NameError: name 'submission' is not defined
