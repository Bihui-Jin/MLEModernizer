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

0.35451

# 6. Current score

0.39174

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.42829) has done: 'The changes add the missing Keras/TensorFlow imports, correctly create the tokenizer and compute the vocabulary size, fix the label‑encoding import, ensure all variables (train‑test split, `input_dim`, `tokenizer`, etc.) are defined before they are used, and write the predictions to a properly‑named CSV file that matches the required submission format.'
- What this solution (achieved 0.46956) has done: 'The fix replaces the failing `keras` imports with the compatible `tensorflow.keras` versions, restores the missing `to_categorical` import, ensures all variables (tokenizer, `docs_padded`, `input_dim`, etc.) are defined before they are used, and keeps the original modelling pipeline unchanged. This allows the notebook to run end‑to‑end and produce a correctly‑formatted `submission.csv` file, moving the solution toward the target score.'
- What this solution (achieved 0.39174) has done: 'Implemented fixes to resolve import errors, undefined variables, and ensure the pipeline runs end‑to‑end, producing a correctly formatted `submission.csv`. Replaced legacy `keras` imports with `tensorflow.keras`, consolidated code into sequential cells, and preserved the original modeling logic.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, set_random_seed
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(7)
random.seed(7)
set_random_seed(7)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "/kaggle/input/spooky-author-identification"


def get_path(filename: str) -> str:
    """Return absolute path for a file inside the competition folder."""
    p = os.path.join(BASE_PATH, filename)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Could not locate {filename} at {p}")
    return p


train_path = get_path("train.csv")
df = pd.read_csv(train_path)



## === cell 2
a2c = {"EAP": 0, "HPL": 1, "MWS": 2}
y_int = np.array([a2c[a] for a in df["author"]])
y = to_categorical(y_int)  # shape (n_samples, 3)




## === cell 3
def preprocess(text: str) -> str:
    text = text.replace("' ", " ' ")
    signs = set(',.:;"?!')
    prods = set(text) & signs
    if not prods:
        return text
    for sign in prods:
        text = text.replace(sign, f" {sign} ")
    return text


def create_docs(df_input: pd.DataFrame, n_gram_max: int = 2) -> list:
    def add_ngram(tokens, n_gram_max):
        ngrams = []
        for n in range(2, n_gram_max + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append("--".join(tokens[i : i + n]))
        return tokens + ngrams

    docs = []
    for doc in df_input["text"]:
        tokens = preprocess(doc).split()
        docs.append(" ".join(add_ngram(tokens, n_gram_max)))
    return docs




## === cell 4
min_count = 2
docs = create_docs(df)

temp_tokenizer = Tokenizer(lower=False, filters="", oov_token=None)
temp_tokenizer.fit_on_texts(docs)
num_words = sum(1 for cnt in temp_tokenizer.word_counts.values() if cnt >= min_count)

tokenizer = Tokenizer(num_words=num_words, lower=False, filters="", oov_token=None)
tokenizer.fit_on_texts(docs)

seqs = tokenizer.texts_to_sequences(docs)

maxlen = 256
docs_padded = pad_sequences(sequences=seqs, maxlen=maxlen)

input_dim = max(tokenizer.word_index.values()) + 1



## === cell 5
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(
    docs_padded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=np.argmax(y, axis=1),
)




## === cell 6
def create_model(embedding_dims: int = 300, optimizer: str = "adam"):
    """Build a simple text classification model."""
    model = Sequential()
    model.add(
        Embedding(input_dim=input_dim, output_dim=embedding_dims, mask_zero=False)
    )
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(3, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model




## === cell 7
model = create_model()
model.summary()



## === cell 8
epochs = 30
hist = model.fit(
    x_train,
    y_train,
    batch_size=16,
    validation_data=(x_val, y_val),
    epochs=epochs,
    callbacks=[
        EarlyStopping(patience=4, monitor="val_loss", restore_best_weights=True)
    ],
    verbose=2,
)



## === cell 9
test_path = get_path("test.csv")
test_df = pd.read_csv(test_path)

test_docs = create_docs(test_df)
test_seq = tokenizer.texts_to_sequences(test_docs)
test_seq_padded = pad_sequences(sequences=test_seq, maxlen=maxlen)

preds = model.predict(test_seq_padded, batch_size=16)



## === cell 10
sample_sub_path = get_path("sample_submission.csv")
submission = pd.read_csv(sample_sub_path)

for author, idx in a2c.items():
    submission[author] = preds[:, idx]

submission.to_csv("submission.csv", index=False)
