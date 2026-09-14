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
import numpy as np
np.random.seed(666)
import pandas as pd
from sklearn.cross_validation import train_test_split
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample = pd.read_csv("../input/sample_submission.csv")

train["EAP"] = (train.author=="EAP")*1
train["HPL"] = (train.author=="HPL")*1
train["MWS"] = (train.author=="MWS")*1
train.drop("author", 1, inplace=True)
target_vars = ["EAP", "HPL", "MWS"]
train.head(2)


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/196858053.py in <cell line: 0>()
      2 np.random.seed(666)
      3 import pandas as pd
----> 4 from sklearn.cross_validation import train_test_split
      5 from subprocess import check_output
      6 print(check_output(["ls", "../input"]).decode("utf8"))

ModuleNotFoundError: No module named 'sklearn.cross_validation'

## === cell 1
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))
import string
train["num_words"] = train["text"].apply(lambda x: len(str(x).split()))
test["num_words"] = test["text"].apply(lambda x: len(str(x).split()))

train["num_unique_words"] = train["text"].apply(lambda x: len(set(str(x).split())))
test["num_unique_words"] = test["text"].apply(lambda x: len(set(str(x).split())))

train["num_chars"] = train["text"].apply(lambda x: len(str(x)))
test["num_chars"] = test["text"].apply(lambda x: len(str(x)))

train["num_stopwords"] = train["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test["num_stopwords"] = test["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

train["num_punctuations"] =train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test["num_punctuations"] =test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

train["num_words_upper"] = train["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test["num_words_upper"] = test["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

train["num_words_title"] = train["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
test["num_words_title"] = test["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

train["mean_word_len"] = train["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test["mean_word_len"] = test["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

num_vars = ["mean_word_len", "num_words_title", "num_punctuations", "num_chars"
            , "num_stopwords", "num_chars", "num_unique_words", "num_words"]


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1650687505.py in <cell line: 0>()
      3 import string
      4 ## Number of words in the text ##
----> 5 train["num_words"] = train["text"].apply(lambda x: len(str(x).split()))
      6 test["num_words"] = test["text"].apply(lambda x: len(str(x).split()))
      7 

NameError: name 'train' is not defined

## === cell 2
import nltk.stem as stm
import re
stemmer = stm.SnowballStemmer("english")
train["stem_text"] = train.text.apply(lambda x: (" ").join([stemmer.stem(z) for z in re.sub("[^a-zA-Z0-9]"," ", x).split(" ")]))
test["stem_text"] = test.text.apply(lambda x: (" ").join([stemmer.stem(z) for z in re.sub("[^a-zA-Z0-9]"," ", x).split(" ")]))

from keras.preprocessing.text import Tokenizer
tok_raw = Tokenizer()
tok_raw.fit_on_texts(train.text.str.lower())
tok_stem = Tokenizer()
tok_stem.fit_on_texts(train.stem_text)
train["seq_text_stem"] = tok_stem.texts_to_sequences(train.stem_text)
test["seq_text_stem"] = tok_stem.texts_to_sequences(test.stem_text)


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2213389511.py in <cell line: 0>()
      3 import re
      4 stemmer = stm.SnowballStemmer("english")
----> 5 train["stem_text"] = train.text.apply(lambda x: (" ").join([stemmer.stem(z) for z in re.sub("[^a-zA-Z0-9]"," ", x).split(" ")]))
      6 test["stem_text"] = test.text.apply(lambda x: (" ").join([stemmer.stem(z) for z in re.sub("[^a-zA-Z0-9]"," ", x).split(" ")]))
      7 

NameError: name 'train' is not defined

## === cell 3
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

def get_keras_data(dataset, maxlen=20, scaler=None, tdfidf=None):
    if scaler==None:
        scaler = StandardScaler()
        scaler.fit(dataset[num_vars])
    if tdfidf==None:
        tdfidf = Pipeline(steps=[('tdfidf', TfidfVectorizer(analyzer='word', binary=False
                                , ngram_range=(1, 4), stop_words="english"))
                            , ('svd', TruncatedSVD(algorithm='randomized', n_components=20, n_iter=10,
                                       random_state=None, tol=0.0)
                                )])
        tdfidf.fit(dataset.text)
    X = {
        "stem_input": pad_sequences(dataset.seq_text_stem, maxlen=maxlen),
        "num_input": scaler.transform(dataset[num_vars]),
        "svd_vect": tdfidf.transform(dataset.text)
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

n_stem_seq = np.max( [np.max(X_valid["stem_input"]), np.max(X_train["stem_input"])])+1


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
from keras.layers import Dense, Dropout, Embedding
from keras.layers import Flatten, Input, SpatialDropout1D, Concatenate
from keras.models import Model
from keras.optimizers import Adam 
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]

def get_model():
    embed_dim = 30
    dropout_rate = 0.9
    emb_dropout_rate = 0.9
   
    input_text = Input(shape=[maxlen], name="stem_input")
    
    input_num = Input(shape=[X_train["num_input"].shape[1]], name="num_input")
    
    input_svd = Input(shape=[X_train["svd_vect"].shape[1]], name="svd_vect")
    
    emb_lstm = SpatialDropout1D(emb_dropout_rate) (Embedding(n_stem_seq, embed_dim
                                                ,input_length = maxlen
                                                               ) (input_text))
    concatenate = Concatenate()([(Flatten() (emb_lstm)), input_num, input_svd])
    dense = Dropout(dropout_rate) (Dense(256) (concatenate))
    
    output = Dense(3, activation="softmax")(dense)

    model = Model([input_text, input_num, input_svd], output)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

model = get_model()
model.summary()


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/993680614.py in <cell line: 0>()
     36     return model
     37 
---> 38 model = get_model()
     39 model.summary()

/tmp/ipykernel_11/993680614.py in get_model()
     18     input_text = Input(shape=[maxlen], name="stem_input")
     19 
---> 20     input_num = Input(shape=[X_train["num_input"].shape[1]], name="num_input")
     21 
     22     input_svd = Input(shape=[X_train["svd_vect"].shape[1]], name="svd_vect")

NameError: name 'X_train' is not defined

## === cell 5
file_path = ".model_weights.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=5)

model = get_model()
model.fit(X_train, y_train, epochs=150
          , validation_data=[X_valid, y_valid]
         , batch_size=512
         , callbacks = callbacks)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2524542997.py in <cell line: 0>()
      1 #TRAIN KERAS MODEL
      2 file_path = ".model_weights.hdf5"
----> 3 callbacks = get_callbacks(filepath=file_path, patience=5)
      4 
      5 model = get_model()

/tmp/ipykernel_11/993680614.py in get_callbacks(filepath, patience)
      8 def get_callbacks(filepath, patience=2):
      9     es = EarlyStopping('val_loss', patience=patience, mode="min")
---> 10     msave = ModelCheckpoint(filepath, save_best_only=True)
     11     return [es, msave]
     12 

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    192                 self.filepath.endswith(ext) for ext in (".keras", ".h5")
    193             ):
--> 194                 raise ValueError(
    195                     "The filepath provided must end in `.keras` "
    196                     "(Keras model format). Received: "

ValueError: The filepath provided must end in `.keras` (Keras model format). Received: filepath=.model_weights.hdf5

## === cell 6
from sklearn.metrics import log_loss

model = get_model()
model.load_weights(file_path)

preds_train = model.predict(X_train)
preds_valid = model.predict(X_valid)

print(log_loss(y_train, preds_train))
print(log_loss(y_valid, preds_valid))


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1871624735.py in <cell line: 0>()
      2 from sklearn.metrics import log_loss
      3 
----> 4 model = get_model()
      5 model.load_weights(file_path)
      6 

/tmp/ipykernel_11/993680614.py in get_model()
     18     input_text = Input(shape=[maxlen], name="stem_input")
     19 
---> 20     input_num = Input(shape=[X_train["num_input"].shape[1]], name="num_input")
     21 
     22     input_svd = Input(shape=[X_train["svd_vect"].shape[1]], name="svd_vect")

NameError: name 'X_train' is not defined

## === cell 7
preds = pd.DataFrame(model.predict(X_test), columns=target_vars)
submission = pd.concat([test["id"],preds], 1)
submission.to_csv("./submission.csv", index=False)
submission.head()


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3631687748.py in <cell line: 0>()
      1 #PREDICTION
----> 2 preds = pd.DataFrame(model.predict(X_test), columns=target_vars)
      3 submission = pd.concat([test["id"],preds], 1)
      4 submission.to_csv("./submission.csv", index=False)
      5 submission.head()

NameError: name 'model' is not defined
