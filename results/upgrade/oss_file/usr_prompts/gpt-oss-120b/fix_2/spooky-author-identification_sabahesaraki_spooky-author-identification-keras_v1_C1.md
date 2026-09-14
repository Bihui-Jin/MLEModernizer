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
Given some text, predict the author.

## Metric
Multi-class logarithmic loss. 

The submitted probabilities for a given sentences are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum).

In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).

## Submission Format
You must submit a csv file with the id, and a probability for each of the three classes. The order of the rows does not matter. The file must have a header and should look like the following:

```
id,EAP,HPL,MWS
id07943,0.33,0.33,0.33
...
```

## Dataset 
### File descriptions
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Data fields
- **id** - a unique identifier for each sentence
- **text** - some text written by one of the authors
- **author** - the author of the sentence (EAP: Edgar Allan Poe, HPL: HP Lovecraft; MWS: Mary Wollstonecraft Shelley)

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        input/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        working/
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
```

-> data/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/spooky-author-identification/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> data/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> input/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> (stopped after 10 files for performance)

# 5. Target score

0.35006

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

np.random.seed(7)
tf.random.set_seed(7)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "/kaggle/input/spooky-author-identification/train.csv"
test_path = "/kaggle/input/spooky-author-identification/test.csv"
sample_sub_path = "/kaggle/input/spooky-author-identification/sample_submission.csv"

df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

a2c = {"EAP": 0, "HPL": 1, "MWS": 2}
y_int = np.array([a2c[a] for a in df["author"]])
y = to_categorical(y_int, num_classes=3)




## === cell 2
def preprocess(text: str) -> str:
    """Insert spaces around punctuation to keep them as separate tokens."""
    text = text.replace("'", " ' ")
    signs = set(',.:;"?!')
    present = signs & set(text)
    for sign in present:
        text = text.replace(sign, f" {sign} ")
    return text


def create_docs(df: pd.DataFrame, n_gram_max: int = 2) -> list:
    """Create token strings with optional character n‑grams."""

    def add_ngram(tokens, n_gram_max):
        ngrams = []
        for n in range(2, n_gram_max + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append("--".join(tokens[i : i + n]))
        return tokens + ngrams

    docs = []
    for txt in df["text"]:
        tokens = preprocess(txt).split()
        docs.append(" ".join(add_ngram(tokens, n_gram_max)))
    return docs




## === cell 3
min_count = 2
train_docs = create_docs(df)

tokenizer = Tokenizer(lower=False, filters="", oov_token="<OOV>")
tokenizer.fit_on_texts(train_docs)

filtered_vocab = {
    word: idx
    for word, idx in tokenizer.word_index.items()
    if tokenizer.word_counts[word] >= min_count
}
filtered_tokenizer = Tokenizer(lower=False, filters="", oov_token="<OOV>")
filtered_tokenizer.word_index = filtered_vocab
filtered_tokenizer.word_counts = {
    w: c
    for w, c in tokenizer.word_counts.items()
    if tokenizer.word_counts[w] >= min_count
}
filtered_tokenizer.index_word = {idx: w for w, idx in filtered_vocab.items()}

train_seq = filtered_tokenizer.texts_to_sequences(train_docs)
maxlen = 256
train_seq = pad_sequences(train_seq, maxlen=maxlen, padding="post")

input_dim = (
    max(filtered_tokenizer.word_index.values()) + 2
)  # +1 for OOV, +1 for padding




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/165456713.py in <cell line: 0>()
      8 # Filter out very rare tokens (frequency < min_count)
      9 # Build a new tokenizer that only keeps tokens meeting the threshold
---> 10 filtered_vocab = {
     11     word: idx
     12     for word, idx in tokenizer.word_index.items()

/tmp/ipykernel_55/165456713.py in <dictcomp>(.0)
     11     word: idx
     12     for word, idx in tokenizer.word_index.items()
---> 13     if tokenizer.word_counts[word] >= min_count
     14 }
     15 filtered_tokenizer = Tokenizer(lower=False, filters="", oov_token="<OOV>")

KeyError: '<OOV>'

## === cell 4
def create_model(embedding_dims: int = 64, optimizer: str = "adam"):
    model = Sequential(
        [
            Embedding(input_dim=input_dim, output_dim=embedding_dims, mask_zero=True),
            GlobalAveragePooling1D(),
            Dense(3, activation="softmax"),
        ]
    )
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model




## === cell 5
x_train, x_val, y_train, y_val = train_test_split(
    train_seq, y, test_size=0.2, random_state=7, stratify=y_int
)

model = create_model(embedding_dims=64)
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[
        EarlyStopping(patience=3, monitor="val_loss", restore_best_weights=True)
    ],
    verbose=2,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2399581241.py in <cell line: 0>()
      1 # Train/validation split
      2 x_train, x_val, y_train, y_val = train_test_split(
----> 3     train_seq, y, test_size=0.2, random_state=7, stratify=y_int
      4 )
      5 

NameError: name 'train_seq' is not defined

## === cell 6
test_docs = create_docs(test_df)
test_seq = filtered_tokenizer.texts_to_sequences(test_docs)
test_seq = pad_sequences(test_seq, maxlen=maxlen, padding="post")

preds = model.predict(test_seq, batch_size=32)

submission = pd.read_csv(sample_sub_path)
for author, idx in a2c.items():
    submission[author] = preds[:, idx]

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3840141749.py in <cell line: 0>()
      1 # Prepare test data using the same tokenizer
      2 test_docs = create_docs(test_df)
----> 3 test_seq = filtered_tokenizer.texts_to_sequences(test_docs)
      4 test_seq = pad_sequences(test_seq, maxlen=maxlen, padding="post")
      5 

NameError: name 'filtered_tokenizer' is not defined
