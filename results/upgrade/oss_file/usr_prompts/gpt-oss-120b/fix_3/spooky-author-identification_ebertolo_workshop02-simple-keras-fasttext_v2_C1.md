# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 2 other files
        input/
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 2 other files
        working/
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 2 other files
```

-> data/spooky-author-identification/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/spooky-author-identification/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> input/spooky-author-identification/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> input/spooky-author-identification/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> input/spooky-author-identification/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> working/spooky-author-identification/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> (stopped after 10 files for performance)

# 5. Target score

0.34534

# 6. Current score

0.57138

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42858) has done: 'The fixes address the import errors (using tensorflow.keras instead of the incompatible keras package), add the missing to_categorical import, correctly restore the Tokenizer and train_test_split functions, compute input_dim after tokenization, use model.predict instead of the nonexistent predict_proba, and ensure the submission DataFrame is created and saved as a proper .csv file. No core modeling logic is changed, only the necessary bug fixes and clean‑up to produce a valid submission.'
- What this solution (achieved 0.57138) has done: 'The changes replace the fragile TensorFlow‑Keras import with a direct `tensorflow` import to fix the import error, simplify the tokenizer by using the full vocabulary (removing the `num_words` limit) and compute `input_dim` from the actual word index size, and modestly increase the embedding dimension to give the model a bit more capacity. These fixes enable the script to run end‑to‑end and should lower the log‑loss toward the target while keeping the core architecture unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

np.random.seed(7)
tf.random.set_seed(7)



## === cell 1
df = pd.read_csv("../input/train.csv")
a2c = {"EAP": 0, "HPL": 1, "MWS": 2}
y_vals = np.array([a2c[a] for a in df.author])
y = to_categorical(y_vals)




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




## === cell 3
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




## === cell 4
min_count = 2
docs = create_docs(df)

tmp_tokenizer = Tokenizer(lower=False, filters="")
tmp_tokenizer.fit_on_texts(docs)

vocab = [word for word, cnt in tmp_tokenizer.word_counts.items() if cnt >= min_count]

tokenizer = Tokenizer(lower=False, filters="")
tokenizer.fit_on_texts(vocab)

docs_seq = tokenizer.texts_to_sequences(docs)

maxlen = 256
docs_padded = pad_sequences(sequences=docs_seq, maxlen=maxlen)

input_dim = len(tokenizer.word_index) + 1




## === cell 5
def create_model(embedding_dims=50, optimizer="adam"):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(3, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model




## === cell 6
epochs = 25
x_train, x_val, y_train, y_val = train_test_split(
    docs_padded, y, test_size=0.2, random_state=7
)

model = create_model()
hist = model.fit(
    x_train,
    y_train,
    batch_size=16,
    validation_data=(x_val, y_val),
    epochs=epochs,
    callbacks=[EarlyStopping(patience=2, monitor="val_loss")],
)



## === cell 7
test_df = pd.read_csv("../input/test.csv")
test_docs = create_docs(test_df)
test_seq = tokenizer.texts_to_sequences(test_docs)
test_padded = pad_sequences(sequences=test_seq, maxlen=maxlen)

preds = model.predict(test_padded, batch_size=16)

submission = pd.read_csv("../input/sample_submission.csv")
for author, idx in a2c.items():
    submission[author] = preds[:, idx]



## === cell 8
submission.to_csv("fastText_result_01.csv", index=False)
