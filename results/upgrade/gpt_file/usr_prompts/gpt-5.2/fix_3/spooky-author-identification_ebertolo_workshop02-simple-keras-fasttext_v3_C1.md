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

0.4427

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.38534) has done: 'I fix the import/runtime issues caused by mixing `keras` v3 with legacy `keras.preprocessing` and `keras.utils`, by switching to the compatible `tf_keras` equivalents so Tokenizer/pad_sequences/to_categorical and model training work again. I also fix pathing to use the provided Kaggle-style `../input/*.csv` files, ensure `input_dim` is computed from the padded integer sequences (not from raw strings), and replace the deprecated `predict_proba()` call with `predict()` to generate class probabilities. Finally, I ensure the submission file is written with the required columns (`id,EAP,HPL,MWS`) and a `.csv` suffix.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from collections import defaultdict

import keras
from keras.layers import Dense, GlobalAveragePooling1D, Embedding
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

np.random.seed(7)

INPUT_DIR_CANDIDATES = [
    "../input",  # standard Kaggle notebooks
    "/kaggle/input",  # Kaggle docker
    "/kaggle/data",  # this environment shows /kaggle/data and /kaggle/input
    "/kaggle/data/spooky-author-identification",
    "/kaggle/input/spooky-author-identification",
]
INPUT_DIR = None
for d in INPUT_DIR_CANDIDATES:
    if os.path.exists(d):
        if os.path.exists(os.path.join(d, "train.csv")):
            INPUT_DIR = d
            break
if INPUT_DIR is None:
    for d in INPUT_DIR_CANDIDATES:
        if os.path.exists(d):
            INPUT_DIR = d
            break
if INPUT_DIR is None:
    raise FileNotFoundError(
        "Could not locate input directory with train.csv/test.csv/sample_submission.csv"
    )

TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

print("Using INPUT_DIR:", INPUT_DIR)
print("TRAIN_PATH exists:", os.path.exists(TRAIN_PATH))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv(TRAIN_PATH)

a2c = {"EAP": 0, "HPL": 1, "MWS": 2}
y = np.array([a2c[a] for a in df.author.values], dtype=np.int64)
y = to_categorical(y, num_classes=3)
y[:3]



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/333551141.py in <cell line: 0>()
----> 1 df = pd.read_csv(TRAIN_PATH)
      2 
      3 a2c = {"EAP": 0, "HPL": 1, "MWS": 2}
      4 y = np.array([a2c[a] for a in df.author.values], dtype=np.int64)
      5 y = to_categorical(y, num_classes=3)

NameError: name 'TRAIN_PATH' is not defined

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




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1615143128.py in <cell line: 0>()
----> 1 counter = {name: defaultdict(int) for name in set(df.author)}
      2 for text, author in zip(df.text, df.author):
      3     text = text.replace(" ", "")
      4     for c in text:
      5         counter[author][c] += 1

NameError: name 'df' is not defined

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
min_count = 2

docs = create_docs(df)
tokenizer = Tokenizer(lower=False, filters="")
tokenizer.fit_on_texts(docs)
num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])

tokenizer = Tokenizer(num_words=num_words, lower=False, filters="")
tokenizer.fit_on_texts(docs)
docs = tokenizer.texts_to_sequences(docs)
print("n_docs:", len(docs))

maxlen = 256
docs = pad_sequences(sequences=docs, maxlen=maxlen)
print("padded shape:", docs.shape, "dtype:", docs.dtype)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1638766031.py in <cell line: 0>()
      1 min_count = 2
      2 
----> 3 docs = create_docs(df)
      4 tokenizer = Tokenizer(lower=False, filters="")
      5 tokenizer.fit_on_texts(docs)

NameError: name 'df' is not defined

## === cell 6
docs_origin = create_docs(df)
docs_origin[0]



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2854780724.py in <cell line: 0>()
----> 1 docs_origin = create_docs(df)
      2 docs_origin[0]
      3 

NameError: name 'df' is not defined

## === cell 7
input_dim = int(np.max(docs)) + 1
embedding_dims = 20
input_dim




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/143719796.py in <cell line: 0>()
----> 1 input_dim = int(np.max(docs)) + 1
      2 embedding_dims = 20
      3 input_dim
      4 
      5 

NameError: name 'docs' is not defined

## === cell 8
def create_model(embedding_dims=20, optimizer="adam"):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(20, activation="tanh"))
    model.add(Dense(3, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model




## === cell 9
model = create_model()
model.summary()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1116701876.py in <cell line: 0>()
----> 1 model = create_model()
      2 model.summary()
      3 

/tmp/ipykernel_11/3216700920.py in create_model(embedding_dims, optimizer)
      1 def create_model(embedding_dims=20, optimizer="adam"):
      2     model = Sequential()
----> 3     model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
      4     model.add(GlobalAveragePooling1D())
      5     model.add(Dense(20, activation="tanh"))

NameError: name 'input_dim' is not defined

## === cell 10
epochs = 25
x_train, x_val, y_train, y_val = train_test_split(
    docs, y, test_size=0.25, random_state=7, stratify=y.argmax(1)
)

hist = model.fit(
    x_train,
    y_train,
    batch_size=32,
    validation_data=(x_val, y_val),
    epochs=epochs,
    callbacks=[
        EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True)
    ],
    verbose=2,
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1520633547.py in <cell line: 0>()
      1 epochs = 25
----> 2 x_train, x_val, y_train, y_val = train_test_split(
      3     docs, y, test_size=0.25, random_state=7, stratify=y.argmax(1)
      4 )
      5 

NameError: name 'train_test_split' is not defined

## === cell 11
docs = create_docs(df)
tokenizer = Tokenizer(lower=True, filters="")
tokenizer.fit_on_texts(docs)
num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])

tokenizer = Tokenizer(num_words=num_words, lower=True, filters="")
tokenizer.fit_on_texts(docs)
docs = tokenizer.texts_to_sequences(docs)

maxlen = 256
docs = pad_sequences(sequences=docs, maxlen=maxlen)

input_dim = int(np.max(docs)) + 1
input_dim



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3683682070.py in <cell line: 0>()
----> 1 docs = create_docs(df)
      2 tokenizer = Tokenizer(lower=True, filters="")
      3 tokenizer.fit_on_texts(docs)
      4 num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])
      5 

NameError: name 'df' is not defined

## === cell 12
model = create_model()
model.summary()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1116701876.py in <cell line: 0>()
----> 1 model = create_model()
      2 model.summary()
      3 

/tmp/ipykernel_11/3216700920.py in create_model(embedding_dims, optimizer)
      1 def create_model(embedding_dims=20, optimizer="adam"):
      2     model = Sequential()
----> 3     model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
      4     model.add(GlobalAveragePooling1D())
      5     model.add(Dense(20, activation="tanh"))

NameError: name 'input_dim' is not defined

## === cell 13
epochs = 25
x_train, x_val, y_train, y_val = train_test_split(
    docs, y, test_size=0.25, random_state=7, stratify=y.argmax(1)
)

hist = model.fit(
    x_train,
    y_train,
    batch_size=32,
    validation_data=(x_val, y_val),
    epochs=epochs,
    callbacks=[
        EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True)
    ],
    verbose=2,
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1520633547.py in <cell line: 0>()
      1 epochs = 25
----> 2 x_train, x_val, y_train, y_val = train_test_split(
      3     docs, y, test_size=0.25, random_state=7, stratify=y.argmax(1)
      4 )
      5 

NameError: name 'train_test_split' is not defined

## === cell 14
test_df = pd.read_csv(TEST_PATH)
test_docs = create_docs(test_df)
test_docs = tokenizer.texts_to_sequences(test_docs)
test_docs = pad_sequences(sequences=test_docs, maxlen=maxlen)

y_pred = model.predict(test_docs, batch_size=256, verbose=0)

result = pd.read_csv(SAMPLE_SUB_PATH)
for a, i in a2c.items():
    result[a] = y_pred[:, i]

eps = 1e-15
for col in ["EAP", "HPL", "MWS"]:
    result[col] = np.clip(result[col].astype(float), eps, 1.0 - eps)

result.head()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3917121791.py in <cell line: 0>()
----> 1 test_df = pd.read_csv(TEST_PATH)
      2 test_docs = create_docs(test_df)
      3 test_docs = tokenizer.texts_to_sequences(test_docs)
      4 test_docs = pad_sequences(sequences=test_docs, maxlen=maxlen)
      5 

NameError: name 'TEST_PATH' is not defined

## === cell 15
out_path = "fastText_result_02.csv"
result.to_csv(out_path, index=False)
print(
    "Wrote submission:",
    out_path,
    "shape:",
    result.shape,
    "columns:",
    list(result.columns),
)
print("Submission preview:")
print(result.head(3).to_string(index=False))

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2618239648.py in <cell line: 0>()
      1 out_path = "fastText_result_02.csv"
----> 2 result.to_csv(out_path, index=False)
      3 print(
      4     "Wrote submission:",
      5     out_path,

NameError: name 'result' is not defined
