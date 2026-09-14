# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.6

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import numpy as np

np.random.seed(666)
import pandas as pd

from sklearn.model_selection import train_test_split

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample = pd.read_csv("../input/sample_submission.csv")

train["EAP"] = (train.author == "EAP") * 1
train["HPL"] = (train.author == "HPL") * 1
train["MWS"] = (train.author == "MWS") * 1

train.drop("author", axis=1, inplace=True)

target_vars = ["EAP", "HPL", "MWS"]
train.head(2)


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
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;31mAttributeError[0m: 'MessageFactory' object has no attribute 'GetPrototype'

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
