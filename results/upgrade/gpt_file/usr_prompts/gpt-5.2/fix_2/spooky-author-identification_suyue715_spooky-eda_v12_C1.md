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

gensim==4.4.0
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
seaborn==0.12.2
sklearn-pandas==2.2.0
textblob==0.19.0
tf_keras==2.18.0
wordcloud==1.9.4

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

0.49321

# 6. Current score

1.09751

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.09751) has done: 'I fix the environment/runtime issues caused by old notebook/IPython syntax and outdated Keras APIs so the pipeline runs end-to-end with your current Kaggle packages. I keep the same core logic (NLTK tokenization + stemming → gensim dictionary → padded sequences → Embedding+LSTM+Softmax) while updating only the broken arguments/methods and preventing shape/dimension errors. I also ensure NLTK resources are available (download if missing) and replace unavailable `stop_words` with NLTK’s stopwords. Finally, I write a valid `submission.csv` with columns `id,EAP,HPL,MWS` in the required format.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM
from keras.utils import pad_sequences, to_categorical

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

from subprocess import check_output

np.random.seed(42)

print(check_output(["ls", "../input"]).decode("utf8"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SAMPLE_SUB_PATH = "../input/sample_submission.csv"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "../input/spooky-author-identification/train.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "../input/spooky-author-identification/test.csv"
if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = "../input/spooky-author-identification/sample_submission.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(train.shape, test.shape)
train.head()



## === cell 2
train.author.unique()



## === cell 3
plt.figure(figsize=(6, 4))
plt.hist(train.author, bins=3)
plt.title("Frequency of Authors Occurence", fontsize=15)
plt.xticks(np.arange(3), (["Edgar Allen Poe", "Mary Shelley", "HP Lovecraft"]))
plt.show()



## === cell 4
train_qs = pd.Series(train["text"].tolist()).astype(str)
dist_train = train_qs.apply(len)

plt.figure(figsize=(15, 6))
plt.hist(dist_train, bins=200, range=[0, 200], density=True, alpha=0.7)
plt.title("Normalized histogram of character count in text", fontsize=15)
plt.xlabel("Number of characters", fontsize=15)
plt.ylabel("Probability", fontsize=15)
plt.show()



## === cell 5
print(
    "mean num",
    dist_train.mean(),
    "\n",
    "std num",
    dist_train.std(),
    "\n",
    "min num",
    dist_train.min(),
    "\n",
    "max num",
    dist_train.max(),
)



## === cell 6
dist_train = train_qs.apply(lambda x: len(x.split(" ")))

plt.figure(figsize=(15, 6))
plt.hist(dist_train, bins=50, range=[0, 50], density=True, alpha=0.7)
plt.title("Normalised histogram of word count in texts", fontsize=15)
plt.xlabel("Number of words", fontsize=15)
plt.ylabel("Probability", fontsize=15)
plt.show()



## === cell 7
print(
    "mean num",
    dist_train.mean(),
    "\n",
    "std num",
    dist_train.std(),
    "\n",
    "min num",
    dist_train.min(),
    "\n",
    "max num",
    dist_train.max(),
)



## === cell 8
from wordcloud import WordCloud

for res in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{res}" if res == "punkt" else f"corpora/{res}")
    except LookupError:
        nltk.download(res, quiet=True)

stop_words_wc = set(stopwords.words("english"))
cloud = WordCloud(width=1440, height=1080, stopwords=stop_words_wc).generate(
    " ".join(train.text.astype(str))
)
plt.figure(figsize=(20, 12))
plt.imshow(cloud, interpolation="bilinear")
plt.axis("off")
plt.show()



## === cell 9
eap = train[train.author == "EAP"]["text"].values
hpl = train[train.author == "HPL"]["text"].values
mws = train[train.author == "MWS"]["text"].values

for title, arr in [("EAP", eap), ("HPL", hpl), ("MWS", mws)]:
    cloud = WordCloud(width=1440, height=1080, stopwords=stop_words_wc).generate(
        " ".join(arr.astype(str))
    )
    plt.figure(figsize=(20, 12))
    plt.imshow(cloud, interpolation="bilinear")
    plt.title(title)
    plt.axis("off")
    plt.show()



## === cell 10
train["author_num"] = train["author"].apply({"EAP": 0, "HPL": 1, "MWS": 2}.get)
train.head()



## === cell 11
raw_text_train = train["text"].astype(str).values
raw_text_test = test["text"].astype(str).values
author_train = train["author_num"].values
num_labels = len(np.unique(author_train))
num_labels



## === cell 12
stop_words = set(stopwords.words("english"))
stop_words.update(
    [".", ",", '"', "'", ":", ";", "(", ")", "[", "]", "{", "}", "``", "''", "--"]
)
stemmer = SnowballStemmer("english")



## === cell 13
print("pre-processing train docs...")
processed_train = []
for doc in raw_text_train:
    tokens = word_tokenize(doc)
    filtered = [w for w in tokens if w.lower() not in stop_words]
    stemmed = [stemmer.stem(w.lower()) for w in filtered]
    processed_train.append(stemmed)

print("pre-processing test docs...")
processed_test = []
for doc in raw_text_test:
    tokens = word_tokenize(doc)
    filtered = [w for w in tokens if w.lower() not in stop_words]
    stemmed = [stemmer.stem(w.lower()) for w in filtered]
    processed_test.append(stemmed)

len(processed_train), len(processed_test)



## === cell 14
from gensim import corpora

processed_docs_all = processed_train + processed_test
dictionary = corpora.Dictionary(processed_docs_all)
dictionary_size = len(dictionary.keys())
print("dictionary size:", dictionary_size)



## === cell 15
print("converting to token ids...")
word_id_train, word_id_len = [], []
for doc in processed_train:
    word_ids = [dictionary.token2id[w] for w in doc if w in dictionary.token2id]
    word_id_train.append(word_ids)
    word_id_len.append(len(word_ids))

word_id_test = []
for doc in processed_test:
    word_ids = [dictionary.token2id[w] for w in doc if w in dictionary.token2id]
    word_id_test.append(word_ids)
    word_id_len.append(len(word_ids))

if len(word_id_len) == 0:
    raise RuntimeError("No tokens were generated; check preprocessing/stopwords.")
seq_len = int(np.round(np.mean(word_id_len) + 2 * np.std(word_id_len)))
seq_len = max(seq_len, 1)
seq_len



## === cell 16
word_id_train = pad_sequences(word_id_train, maxlen=seq_len)
word_id_test = pad_sequences(word_id_test, maxlen=seq_len)
y_train_enc = to_categorical(author_train, num_labels=num_labels)

print(word_id_train.shape, word_id_test.shape, y_train_enc.shape)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3731926954.py in <cell line: 0>()
      2 word_id_train = pad_sequences(word_id_train, maxlen=seq_len)
      3 word_id_test = pad_sequences(word_id_test, maxlen=seq_len)
----> 4 y_train_enc = to_categorical(author_train, num_labels=num_labels)
      5 
      6 print(word_id_train.shape, word_id_test.shape, y_train_enc.shape)

TypeError: to_categorical() got an unexpected keyword argument 'num_labels'

## === cell 17
print("fitting LSTM ...")
model = Sequential()
model.add(Embedding(input_dim=dictionary_size, output_dim=128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_labels))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(word_id_train, y_train_enc, epochs=1, batch_size=250, verbose=1)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3029914631.py in <cell line: 0>()
     13 
     14 # Old arg nb_epoch -> epochs
---> 15 model.fit(word_id_train, y_train_enc, epochs=1, batch_size=250, verbose=1)
     16 

NameError: name 'y_train_enc' is not defined

## === cell 18
test_pred = model.predict(word_id_test, batch_size=512, verbose=0)
test_pred.shape



## === cell 19
prob = pd.DataFrame(test_pred, columns=["EAP", "HPL", "MWS"])
submit1 = pd.concat(
    [test[["id"]].reset_index(drop=True), prob.reset_index(drop=True)], axis=1
)

for c in ["EAP", "HPL", "MWS"]:
    submit1[c] = pd.to_numeric(submit1[c], errors="coerce").fillna(1.0 / 3.0)
submit1[["EAP", "HPL", "MWS"]] = np.clip(
    submit1[["EAP", "HPL", "MWS"]].values, 1e-15, 1 - 1e-15
)

submit1.head()



## === cell 20
SUB_PATH = "./submission.csv"
submit1.to_csv(SUB_PATH, index=False)
print("Wrote:", SUB_PATH)
print("Submission shape:", submit1.shape)

sample = pd.read_csv(SAMPLE_SUB_PATH)
print("Sample columns:", list(sample.columns))
print("Our columns:", list(submit1.columns))
