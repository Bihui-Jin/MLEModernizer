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

0.87428

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.61597) has done: 'The fix updates the imports to use TensorFlow‑Keras (avoiding the protobuf error), replaces the deprecated DataFrame.append with pd.concat, defines variables that were missing due to earlier failures, corrects padding length handling, uses model.predict instead of the removed predict_proba, and ensures the submission CSV is written correctly. These minimal changes let the notebook run end‑to‑end and produce a valid “fastText_result_05.csv” file while preserving the original modeling approach.'
- What this solution (achieved 0.87428) has done: 'Implemented fixes:
- Set protobuf implementation env variable before importing TensorFlow to avoid import error.
- Added robust data‑path resolution that searches common Kaggle locations, guaranteeing the CSV files are found.
- Adjusted imports and removed unused backend import.
- Ensured all variables are defined in the correct order and that the model prediction returns a NumPy array.
- Kept the original modeling approach unchanged while making the script executable end‑to‑end and producing a valid CSV submission.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Embedding, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

np.random.seed(7)
tf.random.set_seed(7)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
possible_dirs = [
    os.path.join("input", "spooky-author-identification"),
    os.path.join("/kaggle", "input", "spooky-author-identification"),
    os.path.join("data", "spooky-author-identification"),
    os.path.join("/kaggle", "working", "spooky-author-identification"),
    "spooky-author-identification",
]
base_dir = None
for d in possible_dirs:
    if os.path.isdir(d):
        base_dir = d
        break
if base_dir is None:
    raise FileNotFoundError(
        "Data directory for spooky-author-identification not found."
    )
train_path = os.path.join(base_dir, "train.csv")
test_path = os.path.join(base_dir, "test.csv")
sample_sub_path = os.path.join(base_dir, "sample_submission.csv")



## === cell 2
df = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_full = pd.concat([df, df_test], sort=False)

a2c = {"EAP": 0, "HPL": 1, "MWS": 2}
y = np.array([a2c[a] for a in df.author])
y = to_categorical(y)



## === cell 3
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




## === cell 4
def preprocess(text):
    text = text.replace("' ", " ' ")
    signs = set(',.:;"?!')
    prods = set(text) & signs
    if not prods:
        return text
    for sign in prods:
        text = text.replace(sign, f" {sign} ")
    return text


def create_docs(df, n_gram_max=3):  # increased n‑gram size slightly
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

docs_seq = tokenizer.texts_to_sequences(docs)
print("Samples Number:", len(docs_seq))
print("Sample 1 tokens:", docs_seq[0][:10])
print("Sample 2 tokens:", docs_seq[1][:10])



## === cell 6
maxlen = max(len(seq) for seq in docs_seq)
print("Max number of words in a sample for full dataset:", maxlen)

docs_padded = pad_sequences(sequences=docs_seq, maxlen=maxlen)
print("Shape after padding:", docs_padded.shape)



## === cell 7
input_dim = int(np.max(docs_padded)) + 1
embedding_dims = 100  # modestly larger embedding for better representation
print("input_dim:", input_dim, "embedding_dims:", embedding_dims)



## === cell 8
x_train, x_val, y_train, y_val = train_test_split(
    docs_padded, y, test_size=0.20, random_state=42, stratify=y
)




## === cell 9
def create_model(embedding_dims=100, optimizer="adam"):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(3, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model




## === cell 10
model = create_model()
model.summary()



## === cell 11
epochs = 500
hist = model.fit(
    x_train,
    y_train,
    batch_size=100,
    validation_data=(x_val, y_val),
    epochs=epochs,
    callbacks=[
        EarlyStopping(patience=25, monitor="val_loss", restore_best_weights=True)
    ],
    verbose=2,
)



## === cell 12
test_df = pd.read_csv(test_path)
docs_test = create_docs(test_df)
docs_test_seq = tokenizer.texts_to_sequences(docs_test)
docs_test_pad = pad_sequences(sequences=docs_test_seq, maxlen=maxlen)

y_pred = model.predict(docs_test_pad, verbose=0)  # returns numpy array

result = pd.read_csv(sample_sub_path)
for author, idx in a2c.items():
    result[author] = y_pred[:, idx]



## === cell 13
submission_path = "fastText_result_05.csv"
result.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
