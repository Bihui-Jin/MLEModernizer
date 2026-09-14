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
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.9757901422142014

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Dense,
    Input,
    GlobalMaxPooling1D,
    CuDNNGRU,
    Embedding,
    Bidirectional,
    Dropout,
    SpatialDropout1D,
)
from tensorflow.keras.models import Model

MAX_SEQUENCE_LENGTH = 200
MAX_VOCAB_SIZE = 20000
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 300
BATCH_SIZE = 1000
EPOCHS = 10

train_data_path = "../input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_data_path = "../input/jigsaw-toxic-comment-classification-challenge/test.csv"
sample_submission_path = (
    "../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)
glove_path = "../input/glove6b/glove.6B.{0}d.txt".format(EMBEDDING_DIM)

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

possible_labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
targets = train_data[possible_labels].values
sentences = train_data["comment_text"].fillna("DUMMY_VALUES").values



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
word2vec = {}
if os.path.exists(glove_path):
    print("Loading GloVe vectors...")
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            vals = line.split()
            word = vals[0]
            vec = np.asarray(vals[1:], dtype="float32")
            word2vec[word] = vec
    print("Loaded {} vectors".format(len(word2vec)))
else:
    print("GloVe file not found – will use random embeddings.")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/868563310.py in <cell line: 0>()
      1 # Load GloVe embeddings if the file exists; otherwise create random embeddings
      2 word2vec = {}
----> 3 if os.path.exists(glove_path):
      4     print("Loading GloVe vectors...")
      5     with open(glove_path, encoding="utf8") as f:

NameError: name 'glove_path' is not defined

## === cell 2
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
word_index = tokenizer.word_index
print("Found {} unique tokens.".format(len(word_index)))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print("Data shape:", data.shape)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/815408145.py in <cell line: 0>()
      1 # Tokenize texts
----> 2 tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
      3 tokenizer.fit_on_texts(sentences)
      4 sequences = tokenizer.texts_to_sequences(sentences)
      5 word_index = tokenizer.word_index

NameError: name 'MAX_VOCAB_SIZE' is not defined

## === cell 3
num_words = min(MAX_VOCAB_SIZE, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

if word2vec:
    for word, i in word_index.items():
        if i < num_words:
            vec = word2vec.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
else:
    embedding_matrix = np.random.uniform(-0.05, 0.05, (num_words, EMBEDDING_DIM))

embedding_layer = Embedding(
    input_dim=num_words,
    output_dim=EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=False,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2779964777.py in <cell line: 0>()
      1 # Prepare embedding matrix
----> 2 num_words = min(MAX_VOCAB_SIZE, len(word_index) + 1)
      3 embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
      4 
      5 if word2vec:

NameError: name 'MAX_VOCAB_SIZE' is not defined

## === cell 4
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = Bidirectional(CuDNNGRU(50, return_sequences=True))(x)
x = SpatialDropout1D(0.1)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
output = Dense(len(possible_labels), activation="sigmoid")(x)

model = Model(inputs=input_, outputs=output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("Model summary:")
model.summary()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1665836293.py in <cell line: 0>()
      1 # Build the model
----> 2 input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
      3 x = embedding_layer(input_)
      4 x = Bidirectional(CuDNNGRU(50, return_sequences=True))(x)
      5 x = SpatialDropout1D(0.1)(x)

NameError: name 'MAX_SEQUENCE_LENGTH' is not defined

## === cell 5
print("Starting training...")
model.fit(
    data,
    targets,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    verbose=2,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2318734637.py in <cell line: 0>()
      1 # Train the model
      2 print("Starting training...")
----> 3 model.fit(
      4     data,
      5     targets,

NameError: name 'model' is not defined

## === cell 6
test_sentences = test_data["comment_text"].fillna("DUMMY_VALUES").values
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_feed = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

print("Predicting on test data...")
preds = model.predict(test_feed, batch_size=BATCH_SIZE, verbose=1)

submission = pd.read_csv(sample_submission_path)
submission[possible_labels] = preds
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1497460366.py in <cell line: 0>()
      1 # Predict on test set and create submission
----> 2 test_sentences = test_data["comment_text"].fillna("DUMMY_VALUES").values
      3 test_sequences = tokenizer.texts_to_sequences(test_sentences)
      4 test_feed = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
      5 

NameError: name 'test_data' is not defined
