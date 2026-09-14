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

0.3876

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np

np.random.seed(666)

import pandas as pd

from sklearn.model_selection import train_test_split

DATA_DIR = "/kaggle/input"
if not os.path.exists(os.path.join(DATA_DIR, "train.csv")):
    alt = "/kaggle/data"
    if os.path.exists(os.path.join(alt, "train.csv")):
        DATA_DIR = alt
    else:
        for cand in [
            "/kaggle/input/spooky-author-identification",
            "/kaggle/data/spooky-author-identification",
            "/kaggle/working/spooky-author-identification",
        ]:
            if os.path.exists(os.path.join(cand, "train.csv")):
                DATA_DIR = cand
                break

print("Using DATA_DIR:", DATA_DIR)
print("Files:", sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])[:10])

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

train["EAP"] = (train.author == "EAP").astype(int)
train["HPL"] = (train.author == "HPL").astype(int)
train["MWS"] = (train.author == "MWS").astype(int)

train.drop(columns=["author"], inplace=True)

target_vars = ["EAP", "HPL", "MWS"]
train.head(2)



## === cell 1
import nltk
from nltk.corpus import stopwords

try:
    eng_stopwords = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    eng_stopwords = set(stopwords.words("english"))

import string

train["num_words"] = train["text"].apply(lambda x: len(str(x).split()))
test["num_words"] = test["text"].apply(lambda x: len(str(x).split()))

train["num_unique_words"] = train["text"].apply(lambda x: len(set(str(x).split())))
test["num_unique_words"] = test["text"].apply(lambda x: len(set(str(x).split())))

train["num_chars"] = train["text"].apply(lambda x: len(str(x)))
test["num_chars"] = test["text"].apply(lambda x: len(str(x)))

train["num_stopwords"] = train["text"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
)
test["num_stopwords"] = test["text"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
)

train["num_punctuations"] = train["text"].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)
test["num_punctuations"] = test["text"].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)

train["num_words_upper"] = train["text"].apply(
    lambda x: len([w for w in str(x).split() if w.isupper()])
)
test["num_words_upper"] = test["text"].apply(
    lambda x: len([w for w in str(x).split() if w.isupper()])
)

train["num_words_title"] = train["text"].apply(
    lambda x: len([w for w in str(x).split() if w.istitle()])
)
test["num_words_title"] = test["text"].apply(
    lambda x: len([w for w in str(x).split() if w.istitle()])
)

train["mean_word_len"] = train["text"].apply(
    lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) else 0.0
)
test["mean_word_len"] = test["text"].apply(
    lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) else 0.0
)

num_vars = [
    "mean_word_len",
    "num_words_title",
    "num_punctuations",
    "num_chars",
    "num_stopwords",
    "num_chars",
    "num_unique_words",
    "num_words",
]



## === cell 2
import nltk.stem as stm
import re

stemmer = stm.SnowballStemmer("english")
train["stem_text"] = train.text.apply(
    lambda x: (" ").join(
        [stemmer.stem(z) for z in re.sub("[^a-zA-Z0-9]", " ", str(x)).split(" ")]
    )
)
test["stem_text"] = test.text.apply(
    lambda x: (" ").join(
        [stemmer.stem(z) for z in re.sub("[^a-zA-Z0-9]", " ", str(x)).split(" ")]
    )
)

from keras.preprocessing.text import Tokenizer

tok_raw = Tokenizer()
tok_raw.fit_on_texts(train.text.str.lower())

tok_stem = Tokenizer()
tok_stem.fit_on_texts(train.stem_text)

train["seq_text_stem"] = tok_stem.texts_to_sequences(train.stem_text)
test["seq_text_stem"] = tok_stem.texts_to_sequences(test.stem_text)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
from tf_keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline


def get_keras_data(dataset, maxlen=20, scaler=None, tdfidf=None):
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(dataset[num_vars])

    if tdfidf is None:
        tdfidf = Pipeline(
            steps=[
                (
                    "tdfidf",
                    TfidfVectorizer(
                        analyzer="word",
                        binary=False,
                        ngram_range=(1, 4),
                        stop_words="english",
                    ),
                ),
                (
                    "svd",
                    TruncatedSVD(
                        algorithm="randomized",
                        n_components=20,
                        n_iter=10,
                        random_state=None,
                        tol=0.0,
                    ),
                ),
            ]
        )
        tdfidf.fit(dataset.text)

    X = {
        "stem_input": pad_sequences(dataset.seq_text_stem, maxlen=maxlen),
        "num_input": scaler.transform(dataset[num_vars]),
        "svd_vect": tdfidf.transform(dataset.text),
    }
    return X, scaler, tdfidf


maxlen = 60
dtrain, dvalid = train_test_split(train, random_state=123, train_size=0.9)

print("processing train...")
X_train, scaler, tdfidf = get_keras_data(dtrain, maxlen)
y_train = np.array(dtrain[target_vars])

print("processing valid...")
X_valid, _, _ = get_keras_data(dvalid, maxlen, scaler, tdfidf)
y_valid = np.array(dvalid[target_vars])

print("processing test...")
X_test, _, _ = get_keras_data(test, maxlen, scaler, tdfidf)

n_stem_seq = np.max([np.max(X_valid["stem_input"]), np.max(X_train["stem_input"])]) + 1
print("n_stem_seq:", n_stem_seq)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1757938119.py in <cell line: 0>()
     49 
     50 print("processing train...")
---> 51 X_train, scaler, tdfidf = get_keras_data(dtrain, maxlen)
     52 y_train = np.array(dtrain[target_vars])
     53 

/tmp/ipykernel_11/1757938119.py in get_keras_data(dataset, maxlen, scaler, tdfidf)
     38 
     39     X = {
---> 40         "stem_input": pad_sequences(dataset.seq_text_stem, maxlen=maxlen),
     41         "num_input": scaler.transform(dataset[num_vars]),
     42         "svd_vect": tdfidf.transform(dataset.text),

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'seq_text_stem'

## === cell 4
from tf_keras.layers import Dense, Dropout, Embedding
from tf_keras.layers import Flatten, Input, SpatialDropout1D, Concatenate
from tf_keras.models import Model
from tf_keras.optimizers import Adam
from tf_keras.callbacks import ModelCheckpoint, EarlyStopping


def get_callbacks(filepath, patience=2):
    es = EarlyStopping(monitor="val_loss", patience=patience, mode="min")
    msave = ModelCheckpoint(
        filepath,
        save_best_only=True,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
    )
    return [es, msave]


def get_model():
    embed_dim = 30
    dropout_rate = 0.9
    emb_dropout_rate = 0.9

    input_text = Input(shape=[maxlen], name="stem_input")
    input_num = Input(shape=[X_train["num_input"].shape[1]], name="num_input")
    input_svd = Input(shape=[X_train["svd_vect"].shape[1]], name="svd_vect")

    emb_lstm = SpatialDropout1D(emb_dropout_rate)(
        Embedding(n_stem_seq, embed_dim, input_length=maxlen)(input_text)
    )
    concatenate = Concatenate()([Flatten()(emb_lstm), input_num, input_svd])
    dense = Dropout(dropout_rate)(Dense(256)(concatenate))

    output = Dense(3, activation="softmax")(dense)
    model = Model([input_text, input_num, input_svd], output)

    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model


model = get_model()
model.summary()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1318105051.py in <cell line: 0>()
     44 
     45 
---> 46 model = get_model()
     47 model.summary()
     48 

/tmp/ipykernel_11/1318105051.py in get_model()
     24 
     25     input_text = Input(shape=[maxlen], name="stem_input")
---> 26     input_num = Input(shape=[X_train["num_input"].shape[1]], name="num_input")
     27     input_svd = Input(shape=[X_train["svd_vect"].shape[1]], name="svd_vect")
     28 

NameError: name 'X_train' is not defined

## === cell 5
file_path = "./model_weights.weights.h5"
callbacks = get_callbacks(filepath=file_path, patience=5)

model = get_model()
model.fit(
    X_train,
    y_train,
    epochs=150,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    callbacks=callbacks,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3211508672.py in <cell line: 0>()
      2 callbacks = get_callbacks(filepath=file_path, patience=5)
      3 
----> 4 model = get_model()
      5 model.fit(
      6     X_train,

/tmp/ipykernel_11/1318105051.py in get_model()
     24 
     25     input_text = Input(shape=[maxlen], name="stem_input")
---> 26     input_num = Input(shape=[X_train["num_input"].shape[1]], name="num_input")
     27     input_svd = Input(shape=[X_train["svd_vect"].shape[1]], name="svd_vect")
     28 

NameError: name 'X_train' is not defined

## === cell 6
from sklearn.metrics import log_loss

model = get_model()
model.load_weights(file_path)

preds_train = model.predict(X_train, verbose=0)
preds_valid = model.predict(X_valid, verbose=0)

print(log_loss(y_train, preds_train))
print(log_loss(y_valid, preds_valid))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3647186296.py in <cell line: 0>()
      1 from sklearn.metrics import log_loss
      2 
----> 3 model = get_model()
      4 model.load_weights(file_path)
      5 

/tmp/ipykernel_11/1318105051.py in get_model()
     24 
     25     input_text = Input(shape=[maxlen], name="stem_input")
---> 26     input_num = Input(shape=[X_train["num_input"].shape[1]], name="num_input")
     27     input_svd = Input(shape=[X_train["svd_vect"].shape[1]], name="svd_vect")
     28 

NameError: name 'X_train' is not defined

## === cell 7
model = get_model()
model.load_weights(file_path)

preds = pd.DataFrame(model.predict(X_test, verbose=0), columns=target_vars)
submission = pd.concat([test["id"], preds], axis=1)
submission = submission[["id", "EAP", "HPL", "MWS"]]

submission.to_csv("./submission.csv", index=False)

print(submission.head())
print("Wrote: ./submission.csv  rows:", len(submission))

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4073996504.py in <cell line: 0>()
      1 # PREDICTION + SUBMISSION (ensure model is defined and weights loaded)
----> 2 model = get_model()
      3 model.load_weights(file_path)
      4 
      5 preds = pd.DataFrame(model.predict(X_test, verbose=0), columns=target_vars)

/tmp/ipykernel_11/1318105051.py in get_model()
     24 
     25     input_text = Input(shape=[maxlen], name="stem_input")
---> 26     input_num = Input(shape=[X_train["num_input"].shape[1]], name="num_input")
     27     input_svd = Input(shape=[X_train["svd_vect"].shape[1]], name="svd_vect")
     28 

NameError: name 'X_train' is not defined
