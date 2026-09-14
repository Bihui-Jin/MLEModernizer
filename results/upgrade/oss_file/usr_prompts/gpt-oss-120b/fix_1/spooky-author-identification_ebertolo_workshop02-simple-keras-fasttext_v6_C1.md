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

import keras
import keras.backend as K
from keras.layers import Dense, GlobalAveragePooling1D, Embedding, Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
np.random.seed(7)


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv('./../input/train.csv')
df_test = pd.read_csv('./../input/test.csv')
df_full = df.append(df_test, sort=False)

a2c = {'EAP': 0, 'HPL' : 1, 'MWS' : 2}
y = np.array([a2c[a] for a in df.author])
y = to_categorical(y)
y


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2020934641.py in <cell line: 0>()
      1 df = pd.read_csv('./../input/train.csv')
      2 df_test = pd.read_csv('./../input/test.csv')
----> 3 df_full = df.append(df_test, sort=False)
      4 
      5 a2c = {'EAP': 0, 'HPL' : 1, 'MWS' : 2}

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 2
counter = {name : defaultdict(int) for name in set(df.author)}
for (text, author) in zip(df.text, df.author):
    text = text.replace(' ', '')
    for c in text:
        counter[author][c] += 1

chars = set()
for v in counter.values():
    chars |= v.keys()
    
names = [author for author in counter.keys()]

print('c ', end='')
for n in names:
    print(n, end='   ')
print()
for c in chars:    
    print(c, end=' ')
    for n in names:
        print(counter[n][c], end=' ')
    print()


## === cell 3
def preprocess(text):
    text = text.replace("' ", " ' ")
    signs = set(',.:;"?!')
    prods = set(text) & signs
    if not prods:
        return text

    for sign in prods:
        text = text.replace(sign, ' {} '.format(sign) )
    return text


## === cell 4
def create_docs(df, n_gram_max=2):
    def add_ngram(q, n_gram_max):
            ngrams = []
            for n in range(2, n_gram_max+1):
                for w_index in range(len(q)-n+1):
                    ngrams.append('--'.join(q[w_index:w_index+n]))
            return q + ngrams
        
    docs = []
    for doc in df.text:
        doc = preprocess(doc).split()
        docs.append(' '.join(add_ngram(doc, n_gram_max)))
    
    return docs


## === cell 5
min_count = 1

docs = create_docs(df)
docs_full = create_docs(df_full)
tokenizer = Tokenizer(lower=True, filters='')
tokenizer.fit_on_texts(docs_full)
num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])
num_words


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2980751641.py in <cell line: 0>()
      2 
      3 docs = create_docs(df)
----> 4 docs_full = create_docs(df_full)
      5 tokenizer = Tokenizer(lower=True, filters='')
      6 tokenizer.fit_on_texts(docs_full)

NameError: name 'df_full' is not defined

## === cell 6
tokenizer = Tokenizer(num_words=num_words, lower=True,  filters='')
tokenizer.fit_on_texts(docs_full)
docs = tokenizer.texts_to_sequences(docs)
print("Samples Number:", len(docs))
print("Sample 1:\n{}\nSample 1:\n{}".format(docs[0], docs[1]))


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1658785801.py in <cell line: 0>()
----> 1 tokenizer = Tokenizer(num_words=num_words, lower=True,  filters='')
      2 tokenizer.fit_on_texts(docs_full)
      3 docs = tokenizer.texts_to_sequences(docs)
      4 print("Samples Number:", len(docs))
      5 print("Sample 1:\n{}\nSample 1:\n{}".format(docs[0], docs[1]))

NameError: name 'Tokenizer' is not defined

## === cell 7
max_size = 0
for text in docs:
    max_size = len(text) if len(text) > max_size else max_size
print("Max number of words in a sample for full dataset:", max_size)


## === cell 8
maxlen = max_size
docs = pad_sequences(sequences=docs, maxlen=1749)
print(len(docs))


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3359887617.py in <cell line: 0>()
      1 maxlen = max_size
----> 2 docs = pad_sequences(sequences=docs, maxlen=1749)
      3 print(len(docs))

/usr/local/lib/python3.11/dist-packages/keras/src/utils/sequence_utils.py in pad_sequences(sequences, maxlen, dtype, padding, truncating, value)
    123 
    124         # check `trunc` has expected shape
--> 125         trunc = np.asarray(trunc, dtype=dtype)
    126         if trunc.shape[1:] != sample_shape:
    127             raise ValueError(

ValueError: invalid literal for int() with base 10: 'So I did not abandon the search until I had become fully satisfied that the thief is a more astute man than myself . So--I I--did did--not not--abandon abandon--the the--search search--until until--I

## === cell 9
docs_origin = create_docs(df)
type(docs_origin)
docs_origin[0]


## === cell 10
num_words
docs.shape


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2822163028.py in <cell line: 0>()
----> 1 num_words
      2 docs.shape

NameError: name 'num_words' is not defined

## === cell 11
input_dim = np.max(docs) + 1
embedding_dims = 50
input_dim


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
UFuncTypeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1700115397.py in <cell line: 0>()
----> 1 input_dim = np.max(docs) + 1
      2 embedding_dims = 50
      3 input_dim

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in max(a, axis, out, keepdims, initial, where)
   2808     5
   2809     """
-> 2810     return _wrapreduction(a, np.maximum, 'max', axis, None, out,
   2811                           keepdims=keepdims, initial=initial, where=where)
   2812 

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in _wrapreduction(obj, ufunc, method, axis, dtype, out, **kwargs)
     86                 return reduction(axis=axis, out=out, **passkwargs)
     87 
---> 88     return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
     89 
     90 

UFuncTypeError: ufunc 'maximum' did not contain a loop with signature matching types (dtype('<U14897'), dtype('<U14897')) -> None

## === cell 12
x_train, x_test, y_train, y_test = train_test_split(docs, y, test_size=0.20, random_state=42)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3642690571.py in <cell line: 0>()
----> 1 x_train, x_test, y_train, y_test = train_test_split(docs, y, test_size=0.20, random_state=42)

NameError: name 'train_test_split' is not defined

## === cell 13
def create_model(embedding_dims=20, optimizer='adam'):
    
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(3, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


## === cell 14
model = create_model()
model.summary()


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2792687876.py in <cell line: 0>()
----> 1 model = create_model()
      2 model.summary()

/tmp/ipykernel_11/3893485987.py in create_model(embedding_dims, optimizer)
      2 
      3     model = Sequential()
----> 4     model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
      5     model.add(GlobalAveragePooling1D())
      6     #model.add(Dense(15, activation='relu')) #tanh

NameError: name 'input_dim' is not defined

## === cell 15
epochs = 500
hist = model.fit(x_train, y_train,
                 batch_size=100,
                 validation_data=(x_test, y_test),
                 epochs=epochs,
                 callbacks=[EarlyStopping(patience=25, monitor='val_loss')])


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2974176601.py in <cell line: 0>()
      1 epochs = 500
----> 2 hist = model.fit(x_train, y_train,
      3                  batch_size=100,
      4                  validation_data=(x_test, y_test),
      5                  epochs=epochs,

NameError: name 'model' is not defined

## === cell 16
"""
docs = create_docs(df)
tokenizer = Tokenizer(lower=True, filters='')
tokenizer.fit_on_texts(docs)
num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])

tokenizer = Tokenizer(num_words=num_words, lower=True, filters='')
tokenizer.fit_on_texts(docs)
docs = tokenizer.texts_to_sequences(docs)

maxlen = 256

docs = pad_sequences(sequences=docs, maxlen=maxlen)

input_dim = np.max(docs) + 1
"""


## === cell 19
test_df = pd.read_csv('../input/test.csv')
docs = create_docs(test_df)
docs = tokenizer.texts_to_sequences(docs)
docs = pad_sequences(sequences=docs, maxlen=maxlen)
y = model.predict_proba(docs)

result = pd.read_csv('../input/sample_submission.csv')
for a, i in a2c.items():
    result[a] = y[:, i]


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3059722580.py in <cell line: 0>()
      1 test_df = pd.read_csv('../input/test.csv')
      2 docs = create_docs(test_df)
----> 3 docs = tokenizer.texts_to_sequences(docs)
      4 docs = pad_sequences(sequences=docs, maxlen=maxlen)
      5 y = model.predict_proba(docs)

NameError: name 'tokenizer' is not defined

## === cell 20
result.to_csv('fastText_result_05.csv', index=False)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/269649813.py in <cell line: 0>()
----> 1 result.to_csv('fastText_result_05.csv', index=False)

NameError: name 'result' is not defined
