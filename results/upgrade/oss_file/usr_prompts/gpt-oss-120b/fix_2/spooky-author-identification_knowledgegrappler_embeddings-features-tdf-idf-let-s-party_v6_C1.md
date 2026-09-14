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
import numpy as np, pandas as pd, os, string, re

np.random.seed(666)

from sklearn.model_selection import train_test_split

print("Files in ../input:", os.listdir("../input"))

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample = pd.read_csv("../input/sample_submission.csv")

train["EAP"] = (train.author == "EAP").astype(int)
train["HPL"] = (train.author == "HPL").astype(int)
train["MWS"] = (train.author == "MWS").astype(int)
train.drop("author", axis=1, inplace=True)

target_vars = ["EAP", "HPL", "MWS"]
print(train.head(2))




## === cell 1
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
eng_stopwords = set(stopwords.words("english"))

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
    lambda x: np.mean([len(w) for w in str(x).split()])
)
test["mean_word_len"] = test["text"].apply(
    lambda x: np.mean([len(w) for w in str(x).split()])
)

num_vars = [
    "mean_word_len",
    "num_words_title",
    "num_punctuations",
    "num_chars",
    "num_stopwords",
    "num_unique_words",
    "num_words",
]




## === cell 2
import nltk.stem as stm

stemmer = stm.SnowballStemmer("english")

train["stem_text"] = train["text"].apply(
    lambda x: " ".join(
        [stemmer.stem(z) for z in re.sub("[^a-zA-Z0-9]", " ", x).split()]
    )
)
test["stem_text"] = test["text"].apply(
    lambda x: " ".join(
        [stemmer.stem(z) for z in re.sub("[^a-zA-Z0-9]", " ", x).split()]
    )
)

from tensorflow.keras.preprocessing.text import Tokenizer

tok_raw = Tokenizer()
tok_raw.fit_on_texts(train["text"].str.lower())
tok_stem = Tokenizer()
tok_stem.fit_on_texts(train["stem_text"])

train["seq_text_stem"] = tok_stem.texts_to_sequences(train["stem_text"])
test["seq_text_stem"] = tok_stem.texts_to_sequences(test["stem_text"])




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from tensorflow.keras.preprocessing.sequence import pad_sequences


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
                        analyzer="word", ngram_range=(1, 4), stop_words="english"
                    ),
                ),
                (
                    "svd",
                    TruncatedSVD(
                        n_components=20,
                        algorithm="randomized",
                        n_iter=10,
                        random_state=42,
                    ),
                ),
            ]
        )
        tdfidf.fit(dataset["text"])
    X = {
        "stem_input": pad_sequences(dataset["seq_text_stem"], maxlen=maxlen),
        "num_input": scaler.transform(dataset[num_vars]),
        "svd_vect": tdfidf.transform(dataset["text"]),
    }
    return X, scaler, tdfidf


maxlen = 60
dtrain, dvalid = train_test_split(train, random_state=123, train_size=0.9)

print("Processing train...")
X_train, scaler, tdfidf = get_keras_data(dtrain, maxlen)
y_train = np.array(dtrain[target_vars])

print("Processing validation...")
X_valid, _, _ = get_keras_data(dvalid, maxlen, scaler, tdfidf)
y_valid = np.array(dvalid[target_vars])

print("Processing test...")
X_test, _, _ = get_keras_data(test, maxlen, scaler, tdfidf)

n_stem_seq = int(
    np.max([np.max(X_valid["stem_input"]), np.max(X_train["stem_input"])]) + 1
)




## === cell 4
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    SpatialDropout1D,
    Concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def get_callbacks(filepath, patience=2):
    es = EarlyStopping(monitor="val_loss", patience=patience, mode="min", verbose=1)
    msave = ModelCheckpoint(
        filepath,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )
    return [es, msave]


def get_model():
    embed_dim = 30
    dropout_rate = 0.9
    emb_dropout_rate = 0.9

    input_text = Input(shape=(maxlen,), name="stem_input")
    input_num = Input(shape=(X_train["num_input"].shape[1],), name="num_input")
    input_svd = Input(shape=(X_train["svd_vect"].shape[1],), name="svd_vect")

    embedding = Embedding(
        input_dim=n_stem_seq, output_dim=embed_dim, input_length=maxlen
    )(input_text)
    emb_lstm = SpatialDropout1D(emb_dropout_rate)(embedding)

    concatenated = Concatenate()([Flatten()(emb_lstm), input_num, input_svd])
    dense = Dropout(dropout_rate)(Dense(256, activation="relu")(concatenated))
    output = Dense(3, activation="softmax")(dense)

    model = Model(inputs=[input_text, input_num, input_svd], outputs=output)
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model




## === cell 5
file_path = "best_model.keras"
callbacks = get_callbacks(filepath=file_path, patience=5)

model = get_model()
model.summary()

model.fit(
    X_train,
    y_train,
    epochs=150,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    callbacks=callbacks,
    verbose=2,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2697595977.py in <cell line: 0>()
      1 file_path = "best_model.keras"
----> 2 callbacks = get_callbacks(filepath=file_path, patience=5)
      3 
      4 model = get_model()
      5 model.summary()

/tmp/ipykernel_55/1088665996.py in get_callbacks(filepath, patience)
     16     es = EarlyStopping(monitor="val_loss", patience=patience, mode="min", verbose=1)
     17     # Keras‑3 expects .keras extension for model files
---> 18     msave = ModelCheckpoint(
     19         filepath,
     20         save_weights_only=True,

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    182         if save_weights_only:
    183             if not self.filepath.endswith(".weights.h5"):
--> 184                 raise ValueError(
    185                     "When using `save_weights_only=True` in `ModelCheckpoint`"
    186                     ", the filepath provided must end in `.weights.h5` "

ValueError: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=best_model.keras

## === cell 6
from sklearn.metrics import log_loss

model.load_weights(file_path)

preds_train = model.predict(X_train)
preds_valid = model.predict(X_valid)

print("Log loss on training set:", log_loss(y_train, preds_train))
print("Log loss on validation set:", log_loss(y_valid, preds_valid))




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1391394201.py in <cell line: 0>()
      2 
      3 # Load the best weights saved during training
----> 4 model.load_weights(file_path)
      5 
      6 preds_train = model.predict(X_train)

NameError: name 'model' is not defined

## === cell 7
preds_test = pd.DataFrame(model.predict(X_test), columns=target_vars)
submission = pd.concat([test["id"], preds_test], axis=1)
submission_path = "./submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
submission.head()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2211877074.py in <cell line: 0>()
      1 # Generate predictions for the test set and create submission file
----> 2 preds_test = pd.DataFrame(model.predict(X_test), columns=target_vars)
      3 submission = pd.concat([test["id"], preds_test], axis=1)
      4 submission_path = "./submission.csv"
      5 submission.to_csv(submission_path, index=False)

NameError: name 'model' is not defined
