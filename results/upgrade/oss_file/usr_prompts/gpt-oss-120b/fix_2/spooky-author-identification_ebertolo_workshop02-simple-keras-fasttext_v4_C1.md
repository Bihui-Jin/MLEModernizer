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

3.7

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

0.35451

# 6. Current score

0.42829

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.42829) has done: 'The changes add the missing Keras/TensorFlow imports, correctly create the tokenizer and compute the vocabulary size, fix the label‑encoding import, ensure all variables (train‑test split, `input_dim`, `tokenizer`, etc.) are defined before they are used, and write the predictions to a properly‑named CSV file that matches the required submission format.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras import Sequential
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
df = pd.read_csv("./../input/train.csv")
a2c = {"EAP": 0, "HPL": 1, "MWS": 2}
y = np.array([a2c[a] for a in df.author])
y = to_categorical(y)




## === cell 2
def preprocess(text):
    text = text.replace("' ", " ' ")
    signs = set(',.:;"?!')
    prods = set(text) & signs
    if not prods:
        return text
    for sign in prods:
        text = text.replace(sign, f" {sign} ")
    return text


def create_docs(df, n_gram_max=2):
    def add_ngram(q, n_gram_max):
        ngrams = []
        for n in range(2, n_gram_max + 1):
            for w_index in range(len(q) - n + 1):
                ngrams.append("--".join(q[w_index : w_index + n]))
        return q + ngrams

    docs = []
    for doc in df.text:
        doc = preprocess(doc).split()
        docs.append(" ".join(add_ngram(doc, n_gram_max)))
    return docs




## === cell 3
min_count = 2
docs = create_docs(df)
tokenizer = Tokenizer(lower=False, filters="")
tokenizer.fit_on_texts(docs)
num_words = sum(1 for v in tokenizer.word_counts.values() if v >= min_count)

tokenizer = Tokenizer(num_words=num_words, lower=False, filters="")
tokenizer.fit_on_texts(docs)
seqs = tokenizer.texts_to_sequences(docs)

maxlen = 256
docs_padded = pad_sequences(sequences=seqs, maxlen=maxlen)

input_dim = max(tokenizer.word_index.values()) + 1



## === cell 4
x_train, x_val, y_train, y_val = train_test_split(
    docs_padded, y, test_size=0.20, random_state=42, stratify=y
)




## === cell 5
def create_model(embedding_dims=100, optimizer="adam"):
    model = Sequential()
    model.add(
        Embedding(input_dim=input_dim, output_dim=embedding_dims, mask_zero=False)
    )
    model.add(GlobalAveragePooling1D())
    model.add(Dense(3, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model




## === cell 6
model = create_model()
model.summary()



## === cell 7
epochs = 25
hist = model.fit(
    x_train,
    y_train,
    batch_size=16,
    validation_data=(x_val, y_val),
    epochs=epochs,
    callbacks=[
        EarlyStopping(patience=2, monitor="val_loss", restore_best_weights=True)
    ],
)



## === cell 8
test_df = pd.read_csv("../input/test.csv")
test_docs = create_docs(test_df)
test_seq = tokenizer.texts_to_sequences(test_docs)
test_seq_padded = pad_sequences(sequences=test_seq, maxlen=maxlen)

preds = model.predict(test_seq_padded, batch_size=16)



## === cell 9
submission = pd.read_csv("../input/sample_submission.csv")
for author, idx in a2c.items():
    submission[author] = preds[:, idx]
submission.to_csv("fastText_result_03.csv", index=False)
