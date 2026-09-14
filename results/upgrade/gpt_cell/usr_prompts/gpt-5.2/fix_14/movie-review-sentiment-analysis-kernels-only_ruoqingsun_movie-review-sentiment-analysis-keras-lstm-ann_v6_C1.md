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

3.7

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
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        input/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        working/
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
```

-> data/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> data/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))



## === cell 1
import numpy as np 
import pandas as pd 

train = pd.read_csv('../input/train.tsv',sep = '\t')
test = pd.read_csv('../input/test.tsv', sep = '\t')
print("Train set: {0}".format(train.shape))
print("Test set: {0}".format(test.shape))

df = pd.concat([train, test])
print("All df set: {0}".format(df.shape))

df.head()


## === cell 2
sub = pd.read_csv('../input/sampleSubmission.csv', sep = ',')
print("Submission: {0}".format(sub.shape))

sub.head()


## === cell 3
print("Training set distribution: ", train.groupby(['Sentiment']).size()/train.shape[0])


## === cell 4
import re
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


## === cell 5
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    review_stem=[]
    for word in text.split():
        word_stem = stemmer.stem(word)
        review_stem.append(word_stem)
    review_stem=' '.join(review_stem)
    return review_stem


## === cell 6
train['clean_phrase'] = train['Phrase'].apply(clean_text)
test['clean_phrase'] = test['Phrase'].apply(clean_text)
df['clean_phrase'] = df['Phrase'].apply(clean_text)


## === cell 7
train.head()


## === cell 8
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"


def to_categorical(y, num_classes=None, dtype="float32"):
    y = np.array(y, dtype="int64").ravel()
    if num_classes is None:
        num_classes = int(np.max(y)) + 1 if y.size else 0
    out = np.zeros((y.shape[0], int(num_classes)), dtype=dtype)
    if y.size:
        out[np.arange(y.shape[0]), y] = 1
    return out


from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk import FreqDist


## === cell 9
train_text=train.clean_phrase.values
test_text=test.clean_phrase.values
target=train.Sentiment.values
y=to_categorical(target)
print(train_text.shape,target.shape,y.shape)


## === cell 10
X_train_text,X_val_text,y_train,y_val=train_test_split(train_text,y,test_size=0.2,stratify=y,random_state=123)
print(X_train_text.shape,y_train.shape)
print(X_val_text.shape,y_val.shape)


## === cell 11
all_words = ' '.join(X_train_text)
word2count = {}
for word in all_words.split():
    if word not in word2count:
        word2count[word] = 1
    else:
        word2count[word] += 1
print("Number of unique words: ", len(word2count.keys()))


## === cell 12
df['length_review'] = df['clean_phrase'].apply(lambda x: len(x.split()))
print("Max phrase length: ", max(df['length_review']))


## === cell 13

import numpy as np
import re


class Tokenizer:
    def __init__(
        self,
        num_words=None,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
        split=" ",
        oov_token=None,
    ):
        self.num_words = num_words
        self.filters = filters
        self.lower = lower
        self.split = split
        self.oov_token = oov_token

        self.word_counts = {}
        self.word_index = {}
        self.index_word = {}

    def _text_to_word_sequence(self, text):
        if text is None:
            return []
        text = str(text)
        if self.lower:
            text = text.lower()
        translate_dict = {ord(c): self.split for c in self.filters}
        text = text.translate(translate_dict)
        return [w for w in text.split(self.split) if w]

    def fit_on_texts(self, texts):
        for t in texts:
            for w in self._text_to_word_sequence(t):
                self.word_counts[w] = self.word_counts.get(w, 0) + 1

        sorted_words = sorted(self.word_counts.items(), key=lambda kv: (-kv[1], kv[0]))

        index = 1
        if self.oov_token is not None:
            self.word_index[self.oov_token] = index
            index += 1

        for w, _ in sorted_words:
            if w in self.word_index:
                continue
            self.word_index[w] = index
            index += 1

        self.index_word = {i: w for w, i in self.word_index.items()}

    def texts_to_sequences(self, texts):
        sequences = []
        oov_index = (
            self.word_index.get(self.oov_token) if self.oov_token is not None else None
        )

        for t in texts:
            seq = []
            for w in self._text_to_word_sequence(t):
                i = self.word_index.get(w)
                if i is None:
                    if oov_index is not None:
                        i = oov_index
                    else:
                        continue
                if self.num_words is not None and i >= self.num_words:
                    if oov_index is not None and oov_index < self.num_words:
                        seq.append(oov_index)
                    continue
                seq.append(i)
            sequences.append(seq)
        return sequences


def pad_sequences(
    sequences, maxlen=None, dtype="int32", padding="pre", truncating="pre", value=0
):
    if maxlen is None:
        maxlen = max((len(s) for s in sequences), default=0)

    x = np.full((len(sequences), maxlen), value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if s is None:
            continue
        s = list(s)
        if len(s) > maxlen:
            if truncating == "pre":
                s = s[-maxlen:]
            elif truncating == "post":
                s = s[:maxlen]
            else:
                raise ValueError("truncating must be 'pre' or 'post'")
        if padding == "pre":
            x[idx, -len(s) :] = (
                np.asarray(s, dtype=dtype) if len(s) else np.asarray([], dtype=dtype)
            )
        elif padding == "post":
            x[idx, : len(s)] = (
                np.asarray(s, dtype=dtype) if len(s) else np.asarray([], dtype=dtype)
            )
        else:
            raise ValueError("padding must be 'pre' or 'post'")
    return x


## === cell 14
MAX_REVIEW_LENGTH = 49
FEATURE_LENGTH = 12011
BATCH_SIZE = 1000
EPOCHS = 100
NUM_CLASSES = 5


## === cell 15
tokenizer = Tokenizer(num_words = FEATURE_LENGTH)
tokenizer.fit_on_texts(list(np.concatenate((train_text, test_text), axis=0)))
X_train = tokenizer.texts_to_sequences(X_train_text)
X_val = tokenizer.texts_to_sequences(X_val_text)
X_test = tokenizer.texts_to_sequences(test_text)


## === cell 16
X_train = pad_sequences(X_train, maxlen=MAX_REVIEW_LENGTH)
X_val = pad_sequences(X_val, maxlen=MAX_REVIEW_LENGTH)
X_test= pad_sequences(X_test, maxlen=MAX_REVIEW_LENGTH)


## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2072031703.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mX_train[0m [0;34m=[0m [0mpad_sequences[0m[0;34m([0m[0mX_train[0m[0;34m,[0m [0mmaxlen[0m[0;34m=[0m[0mMAX_REVIEW_LENGTH[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mX_val[0m [0;34m=[0m [0mpad_sequences[0m[0;34m([0m[0mX_val[0m[0;34m,[0m [0mmaxlen[0m[0;34m=[0m[0mMAX_REVIEW_LENGTH[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0mX_test[0m[0;34m=[0m [0mpad_sequences[0m[0;34m([0m[0mX_test[0m[0;34m,[0m [0mmaxlen[0m[0;34m=[0m[0mMAX_REVIEW_LENGTH[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/4147571841.py[0m in [0;36mpad_sequences[0;34m(sequences, maxlen, dtype, padding, truncating, value)[0m
[1;32m    101[0m                 [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"truncating must be 'pre' or 'post'"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    102[0m         [0;32mif[0m [0mpadding[0m [0;34m==[0m [0;34m"pre"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 103[0;31m             x[idx, -len(s) :] = (
[0m[1;32m    104[0m                 [0mnp[0m[0;34m.[0m[0masarray[0m[0;34m([0m[0ms[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mdtype[0m[0;34m)[0m [0;32mif[0m [0mlen[0m[0;34m([0m[0ms[0m[0;34m)[0m [0;32melse[0m [0mnp[0m[0;34m.[0m[0masarray[0m[0;34m([0m[0;34m[[0m[0;34m][0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    105[0m             )

[0;31mValueError[0m: could not broadcast input array from shape (0,) into shape (49,)

## === cell 17
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
