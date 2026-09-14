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

0.3544

# 6. Current score

0.61597

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.61597) has done: 'The fix updates the imports to use TensorFlow‑Keras (avoiding the protobuf error), replaces the deprecated DataFrame.append with pd.concat, defines variables that were missing due to earlier failures, corrects padding length handling, uses model.predict instead of the removed predict_proba, and ensures the submission CSV is written correctly. These minimal changes let the notebook run end‑to‑end and produce a valid “fastText_result_05.csv” file while preserving the original modeling approach.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Embedding, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

np.random.seed(7)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv("./../input/train.csv")
df_test = pd.read_csv("./../input/test.csv")
df_full = pd.concat([df, df_test], sort=False)

a2c = {"EAP": 0, "HPL": 1, "MWS": 2}
y = np.array([a2c[a] for a in df.author])
y = to_categorical(y)




## === cell 2
counter = {name: defaultdict(int) for name in set(df.author)}
for text, author in zip(df.text, df.author):
    text = text.replace(" ", "")
    for c in text:
        counter[author][c] += 1

chars = set()
for v in counter.values():
    chars |= v.keys()

names = [author for author in counter.keys()]

print("c ", end="")
for n in names:
    print(n, end="   ")
print()
for c in chars:
    print(c, end=" ")
    for n in names:
        print(counter[n][c], end=" ")
    print()




## === cell 3
def preprocess(text):
    text = text.replace("' ", " ' ")
    signs = set(',.:;"?!')
    prods = set(text) & signs
    if not prods:
        return text

    for sign in prods:
        text = text.replace(sign, " {} ".format(sign))
    return text




## === cell 4
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




## === cell 5
min_count = 1

docs = create_docs(df)
docs_full = create_docs(df_full)

tokenizer = Tokenizer(lower=True, filters="")
tokenizer.fit_on_texts(docs_full)

num_words = sum(1 for _, v in tokenizer.word_counts.items() if v >= min_count)

tokenizer = Tokenizer(num_words=num_words, lower=True, filters="")
tokenizer.fit_on_texts(docs_full)

docs = tokenizer.texts_to_sequences(docs)
print("Samples Number:", len(docs))
print("Sample 1 tokens:", docs[0])
print("Sample 2 tokens:", docs[1])




## === cell 6
max_size = 0
for text in docs:
    max_size = len(text) if len(text) > max_size else max_size
print("Max number of words in a sample for full dataset:", max_size)




## === cell 7
maxlen = max_size  # use the computed maximum length
docs = pad_sequences(sequences=docs, maxlen=maxlen)
print("Shape after padding:", docs.shape)




## === cell 8
input_dim = int(np.max(docs)) + 1
embedding_dims = 50
print("input_dim:", input_dim)




## === cell 9
x_train, x_test, y_train, y_test = train_test_split(
    docs, y, test_size=0.20, random_state=42
)




## === cell 10
def create_model(embedding_dims=20, optimizer="adam"):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(3, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model




## === cell 11
model = create_model()
model.summary()




## === cell 12
epochs = 500
hist = model.fit(
    x_train,
    y_train,
    batch_size=100,
    validation_data=(x_test, y_test),
    epochs=epochs,
    callbacks=[EarlyStopping(patience=25, monitor="val_loss")],
    verbose=2,
)




## === cell 14
test_df = pd.read_csv("../input/test.csv")
docs_test = create_docs(test_df)
docs_test = tokenizer.texts_to_sequences(docs_test)
docs_test = pad_sequences(sequences=docs_test, maxlen=maxlen)
y_pred = model.predict(docs_test, verbose=0)

result = pd.read_csv("../input/sample_submission.csv")
for author, idx in a2c.items():
    result[author] = y_pred[:, idx]




## === cell 15
result.to_csv("fastText_result_05.csv", index=False)
