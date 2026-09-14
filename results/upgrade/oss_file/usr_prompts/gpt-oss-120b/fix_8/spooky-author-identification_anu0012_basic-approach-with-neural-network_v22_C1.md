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

3.6

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
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
wordcloud==1.9.4
xgboost==2.0.3

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

0.3847

# 6. Current score

1.12309

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.42995) has done: 'I replace the outdated Keras imports with the current tensorflow‑keras API, fix the incorrect argument name for epochs, and remove unused imports that caused import errors. These minimal changes let the notebook run end‑to‑end, generate predictions, and write a properly formatted `result.csv` submission file.'
- What this solution (achieved 0.71049) has done: 'Implemented fixes:
- Replaced incompatible `keras` imports with `tensorflow.keras` equivalents.
- Corrected data paths to use the Kaggle `/kaggle/input` directory.
- Ensured all variables are defined before use by ordering code logically.
- Added a random seed for reproducibility.
- Slightly increased epochs to 60 for modest performance gain while keeping core model unchanged.
- Guaranteed the submission CSV is written as `result.csv` with required columns.'
- What this solution (achieved 1.10757) has done: 'Implemented two key fixes: (1) set the protobuf implementation environment variable before importing TensorFlow to eliminate the `MessageFactory` attribute error; (2) modestly boost model capacity and training duration (larger embedding size, longer input sequences, and more epochs) to improve the log‑loss toward the target while preserving the original architecture. All other logic remains unchanged, and the script now reliably writes a correctly formatted `result.csv`.'
- What this solution (achieved 1.52358) has done: 'I moved the protobuf environment setting to precede all TensorFlow imports and import TensorFlow itself first, then pull Keras objects from `tf.keras`. This resolves the `MessageFactory` attribute error while keeping the original model and workflow intact, allowing the notebook to run end‑to‑end and generate a correctly formatted `result.csv` submission.'
- What this solution (achieved 1.14629) has done: 'Implemented two key fixes:  
1. Set the protobuf environment variable **before any imports** to prevent the `MessageFactory` error.  
2. Slightly enhanced the neural network (added a hidden dense layer with dropout) and increased training epochs to improve log‑loss while preserving the original architecture. The script now runs end‑to‑end and writes a correctly formatted `result.csv`.'
- What this solution (achieved 1.12309) has done: 'I moved the protobuf‑environment setting to the very top and imported TensorFlow right after it so the “MessageFactory” error is avoided. I also modestly increased model capacity (larger embedding, an extra dense layer) and raised epochs to give the network more opportunity to learn, while keeping the same overall architecture and workflow. The script now runs end‑to‑end and writes a correctly formatted `result.csv` submission file.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf

import sys, subprocess, random, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(
    "Listing input directory:",
    subprocess.check_output(["ls", "/kaggle/input"]).decode(),
)

train_path = "/kaggle/input/spooky-author-identification/train.csv"
test_path = "/kaggle/input/spooky-author-identification/test.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train.head()



## === cell 2
mapping_target = {"EAP": 0, "HPL": 1, "MWS": 2}
train = train.replace({"author": mapping_target})



## === cell 3
test_id = test["id"].values
target = train["author"].values



## === cell 4
stops = [
    "the",
    "a",
    "an",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "what",
    "which",
    "this",
    "that",
    "these",
    "those",
    "then",
    "just",
    "so",
    "than",
    "such",
    "both",
    "through",
    "about",
    "for",
    "is",
    "of",
    "while",
    "during",
    "to",
    "What",
    "Which",
    "Is",
    "If",
    "While",
    "This",
]


def cleanData(
    text, lowercase=False, remove_stops=False, stemming=False, lemmatization=False
):
    txt = str(text)
    if lowercase:
        txt = txt.lower()
    if remove_stops:
        txt = " ".join([w for w in txt.split() if w not in stops])
    if stemming:
        st = PorterStemmer()
        txt = " ".join([st.stem(w) for w in txt.split()])
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        txt = " ".join([lemmatizer.lemmatize(w, pos="v") for w in txt.split()])
    return txt




## === cell 5
train["text"] = train["text"].apply(lambda x: cleanData(x, lowercase=True))
test["text"] = test["text"].apply(lambda x: cleanData(x, lowercase=True))



## === cell 6
MAX_SEQUENCE_LENGTH = 200
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 256  # increased embedding size
VALIDATION_SPLIT = 0.3

print("Processing text dataset")
texts_1 = train["text"].tolist()
labels = train["author"].values
print(f"Found {len(texts_1)} training texts.")
test_texts_1 = test["text"].tolist()
print(f"Found {len(test_texts_1)} test texts.")



## === cell 7
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts_1 + test_texts_1)

sequences_1 = tokenizer.texts_to_sequences(texts_1)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)

word_index = tokenizer.word_index
print(f"Found {len(word_index)} unique tokens.")

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)

labels = np.array(labels)
print("Shape of data tensor:", data_1.shape)
print("Shape of label tensor:", labels.shape)



## === cell 8
nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

model = Sequential()
model.add(
    Embedding(
        input_dim=nb_words, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH
    )
)
model.add(GlobalAveragePooling1D())
model.add(Dense(128, activation="relu"))  # extra dense layer
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(3, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()



## === cell 9
model.fit(
    data_1,
    to_categorical(labels),
    epochs=300,  # more epochs for better convergence
    batch_size=64,
    validation_split=VALIDATION_SPLIT,
    verbose=2,
)



## === cell 10
preds = model.predict(test_data_1, batch_size=64)



## === cell 11
result = pd.DataFrame(
    {"id": test_id, "EAP": preds[:, 0], "HPL": preds[:, 1], "MWS": preds[:, 2]}
)
result.to_csv("result.csv", index=False)
print("Submission file 'result.csv' created with shape:", result.shape)
