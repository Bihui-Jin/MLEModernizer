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
import os, numpy as np, pandas as pd

print(os.listdir("../input"))




## === cell 1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, CuDNNLSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm




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
    "../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv"
)




## === cell 3
df = train[["Phrase", "Sentiment"]]




## === cell 4
token = Tokenizer()
x_text = df["Phrase"].astype(str).tolist()
token.fit_on_texts(x_text)
seq = token.texts_to_sequences(x_text)
pad_seq = pad_sequences(seq, maxlen=300, dtype="int32")  # <- changed dtype

y = to_categorical(df["Sentiment"].values, num_classes=5)

vocab_size = len(token.word_index) + 1
print("Vocab size:", vocab_size)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/963763720.py in <cell line: 0>()
      1 # Use int32 for token indices to reduce memory bandwidth and speed up TF ops
----> 2 token = Tokenizer()
      3 x_text = df["Phrase"].astype(str).tolist()
      4 token.fit_on_texts(x_text)
      5 seq = token.texts_to_sequences(x_text)

NameError: name 'Tokenizer' is not defined

## === cell 5
embedding_dim = 300
embedding_layer = Embedding(
    input_dim=vocab_size, output_dim=embedding_dim, input_length=300, trainable=True
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2642941299.py in <cell line: 0>()
      1 embedding_dim = 300
      2 embedding_layer = Embedding(
----> 3     input_dim=vocab_size, output_dim=embedding_dim, input_length=300, trainable=True
      4 )
      5 

NameError: name 'vocab_size' is not defined

## === cell 6
x_train, x_val, y_train, y_val = train_test_split(
    pad_seq, y, test_size=0.3, random_state=42
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3317706730.py in <cell line: 0>()
----> 1 x_train, x_val, y_train, y_val = train_test_split(
      2     pad_seq, y, test_size=0.3, random_state=42
      3 )
      4 
      5 

NameError: name 'train_test_split' is not defined

## === cell 7
model = Sequential()
model.add(embedding_layer)

if tf.config.list_physical_devices("GPU"):
    model.add(CuDNNLSTM(75))
else:
    model.add(LSTM(75))

model.add(Dense(128, activation="relu"))
model.add(Dense(5, activation="softmax"))




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1291510040.py in <cell line: 0>()
      1 model = Sequential()
----> 2 model.add(embedding_layer)
      3 
      4 # Use CuDNNLSTM when a GPU is present; otherwise fall back to regular LSTM.
      5 if tf.config.list_physical_devices("GPU"):

NameError: name 'embedding_layer' is not defined

## === cell 8
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(buffer_size=1024).batch(32).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)

history = model.fit(train_ds, epochs=4, validation_data=val_ds, verbose=2)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3301539146.py in <cell line: 0>()
      2 
      3 # Build tf.data pipelines for fast input feeding
----> 4 train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
      5 train_ds = train_ds.shuffle(buffer_size=1024).batch(32).prefetch(tf.data.AUTOTUNE)
      6 

NameError: name 'tf' is not defined

## === cell 9
test_phrases = test["Phrase"].astype(str).tolist()
test_seq = token.texts_to_sequences(test_phrases)
pad_test_seq = pad_sequences(
    test_seq, maxlen=300, dtype="int32"
)  # keep dtype consistent

pred_probs = model.predict(pad_test_seq, batch_size=32, verbose=0)
predict = np.argmax(pred_probs, axis=1)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/136478850.py in <cell line: 0>()
      1 test_phrases = test["Phrase"].astype(str).tolist()
----> 2 test_seq = token.texts_to_sequences(test_phrases)
      3 pad_test_seq = pad_sequences(
      4     test_seq, maxlen=300, dtype="int32"
      5 )  # keep dtype consistent

NameError: name 'token' is not defined

## === cell 10
submission = pd.DataFrame({"PhraseId": test["PhraseId"], "Sentiment": predict})
submission.to_csv("Submission.csv", index=False)
print("Submission saved to Submission.csv")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1967593307.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"PhraseId": test["PhraseId"], "Sentiment": predict})
      2 submission.to_csv("Submission.csv", index=False)
      3 print("Submission saved to Submission.csv")

NameError: name 'predict' is not defined
