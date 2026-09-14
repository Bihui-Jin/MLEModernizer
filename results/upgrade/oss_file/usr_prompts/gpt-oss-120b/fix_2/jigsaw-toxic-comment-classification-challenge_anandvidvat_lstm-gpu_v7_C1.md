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

0.9706243556146232

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'We replace the broken `keras` imports with `tensorflow.keras`, ensure `os` is imported, guard the GloVe loading so a missing file does not crash, and adjust the GRU layer to the standard `GRU` (which works on CPU). Minor hyper‑parameter tweaks (e.g., fewer epochs) keep runtime short while preserving the original model architecture and logic. These fixes unblock the pipeline and produce a correct `submission.csv` file.'

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
    GRU,
    Embedding,
    Bidirectional,
)
from tensorflow.keras.layers import Dropout, SpatialDropout1D
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score

print("Available input directories:", os.listdir("../input"))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
MAX_SEQUENCE_LENGTH = 200
MAX_VOCAB_SIZE = 20000

VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 300
BATCH_SIZE = 1000
EPOCHS = 2  # reduced for faster execution while keeping core logic




## === cell 2
train_data_path = "../input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_data_path = "../input/jigsaw-toxic-comment-classification-challenge/test.csv"
glove_path = (
    f"../input/glove6b/glove.6B.{EMBEDDING_DIM}d.txt"  # may be missing; handled later
)




## === cell 3
print("loading word2vec...")
word2vec = {}
if os.path.exists(glove_path):
    with open(glove_path, encoding="utf8") as fs:
        for line in fs:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype="float32")
            word2vec[word] = vec
    print(f"number of vectors loaded: {len(word2vec)}")
else:
    print("GloVe file not found – proceeding with empty embeddings.")

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)




## === cell 4
sentences = train_data["comment_text"].fillna("DUMMY_VALUES").values
possible_labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
targets = train_data[possible_labels].values




## === cell 5
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)




## === cell 6
len_seq = [len(s) for s in sequences]
print("maximum sequence length :", max(len_seq))
print("minimum sequence length :", min(len_seq))
len_seq_sorted = sorted(len_seq)
median_len = len_seq_sorted[len(len_seq_sorted) // 2]
print("median sequence length :", median_len)




## === cell 7
word_index = tokenizer.word_index
print("Vocabulary size:", len(word_index))
print("type(word_index):", type(word_index))




## === cell 8
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print("shape of data:", data.shape)




## === cell 9
print("Filling pre-trained embeddings...")
num_words = min(MAX_VOCAB_SIZE, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

print("shape of embedding matrix:", embedding_matrix.shape)




## === cell 10
trainable_emb = len(word2vec) == 0
embedding_layer = Embedding(
    input_dim=num_words,
    output_dim=EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=trainable_emb,
)




## === cell 11
print("Building the Model...")
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = Bidirectional(GRU(20, return_sequences=True))(x)
x = SpatialDropout1D(0.1)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(len(possible_labels), activation="sigmoid")(x)

model = Model(inputs=input_, outputs=output)




## === cell 12
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
print(model.summary())




## === cell 13
print("Training Model...")
model.fit(
    data,
    targets,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    verbose=2,
)




## === cell 14
test_sentences = test_data["comment_text"].fillna("DUMMY_VALUES").values
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_feed = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
predict = model.predict(test_feed, batch_size=BATCH_SIZE)

sample_sub_path = (
    "../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)
submission = pd.read_csv(sample_sub_path)
submission[possible_labels] = predict
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
