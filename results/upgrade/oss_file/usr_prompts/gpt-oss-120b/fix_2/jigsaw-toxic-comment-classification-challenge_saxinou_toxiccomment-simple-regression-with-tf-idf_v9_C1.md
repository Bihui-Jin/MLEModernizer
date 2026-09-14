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
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.6

# 3. Installed packages

geopandas==0.14.4
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
pillow==11.3.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
text-unidecode==1.3
wordcloud==1.9.4

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.97409

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import collections
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re  # for regex
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
color = sns.color_palette()
sns.set_style("dark")

stopword_list = set(stopwords.words("english"))
lem = WordNetLemmatizer()
tokenizer = TweetTokenizer()



## === cell 1
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("DIMENSION OF DATABASE : ")
print(">>> Dimension du train :", train.shape)
print(">>> Dimension du test :", test.shape)

print("\nMISSING VALUES : ")
print(">>> Check for missing values in Train dataset")
print(train.isnull().sum())
print(">>> Check for missing values in Test dataset")
print(test.isnull().sum())
print('>>> Filling NA with "unknown"')
train["comment_text"].fillna("unknown", inplace=True)
test["comment_text"].fillna("unknown", inplace=True)

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
for col in list_classes:
    print(
        "\nRépartition pour la variable ", col, " : \n", collections.Counter(train[col])
    )

rowsums = train.iloc[:, 2:].sum(axis=1)
train["total_toxicity"] = rowsums
train["clean"] = rowsums == 0

print("\nDistribution of Total Toxicity Labels (important for validation)")
print("On train set : ", pd.value_counts(train.total_toxicity))




## === cell 2
def indirect_features(df):
    df["count_sent"] = df["comment_text"].apply(
        lambda x: len(re.findall("\n", str(x))) + 1
    )
    df["count_word"] = df["comment_text"].apply(lambda x: len(str(x).split()))
    df["count_unique_word"] = df["comment_text"].apply(
        lambda x: len(set(str(x).split()))
    )
    df["count_letters"] = df["comment_text"].apply(lambda x: len(str(x)))
    df["count_words_upper"] = df["comment_text"].apply(
        lambda x: len([w for w in str(x).split() if w.isupper()])
    )
    df["count_words_title"] = df["comment_text"].apply(
        lambda x: len([w for w in str(x).split() if w.istitle()])
    )
    df["count_stopwords"] = df["comment_text"].apply(
        lambda x: len([w for w in str(x).lower().split() if w in stopword_list])
    )
    df["mean_word_len"] = df["comment_text"].apply(
        lambda x: np.mean([len(w) for w in str(x).split()])
    )
    df["total_length"] = df["comment_text"].apply(len)
    df["capitals"] = df["comment_text"].apply(
        lambda c: sum(1 for ch in c if ch.isupper())
    )
    df["caps_vs_length"] = df.apply(
        lambda row: float(row["capitals"]) / float(row["total_length"]), axis=1
    )
    df["count_punctuations"] = df["comment_text"].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation])
    )
    df["num_exclamation_marks"] = df["comment_text"].apply(lambda c: c.count("!"))
    df["num_question_marks"] = df["comment_text"].apply(lambda c: c.count("?"))
    df["num_symbols"] = df["comment_text"].apply(
        lambda c: sum(c.count(w) for w in "*&$%")
    )
    df["num_smilies"] = df["comment_text"].apply(
        lambda c: sum(c.count(w) for w in (":-)", ":)", ";-)", ";)"))
    )
    df["word_unique_percent"] = df["count_unique_word"] * 100 / df["count_word"]
    df["punct_percent"] = df["count_punctuations"] * 100 / df["count_word"]


indirect_features(train)
indirect_features(test)



## === cell 6
CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "you're": "you are",
    "you've": "you have",
}


def expand_contractions(sentence, contraction_mapping):
    contractions_pattern = re.compile(
        "({})".format("|".join(contraction_mapping.keys())),
        flags=re.IGNORECASE | re.DOTALL,
    )

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded = contraction_mapping.get(match) or contraction_mapping.get(
            match.lower()
        )
        expanded = first_char + expanded[1:]
        return expanded

    return contractions_pattern.sub(expand_match, sentence)





## === cell 7
from nltk.tokenize import sent_tokenize


def preprocessing_clean(comment):
    """
    Clean a comment and return a processed string.
    """
    comment = comment.lower()
    comment = re.sub("\\n", "", comment)
    comment = re.sub("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}", "", comment)
    comment = re.sub("\[\[.*\]", "", comment)
    words = tokenizer.tokenize(comment)
    words = [CONTRACTION_MAP.get(word, word) for word in words]
    words = [w for w in words if w not in stopword_list]
    words = [lem.lemmatize(word, "v") for word in words]
    return " ".join(words)


print(">>> Before cleaning")
print(train.comment_text.iloc[42])
print("\n>>> After cleaning")
print(preprocessing_clean(train.comment_text.iloc[42]))



## === cell 8
clean_corpus = train.comment_text.apply(preprocessing_clean)
print("Not cleaned : ", clean_corpus[42])
print("\nCleaned : ", clean_corpus[42])
print("FIN")



## === cell 11
def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi‑class logarithmic loss."""
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2
    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    return -np.sum(actual * np.log(clip)) / rows




## === cell 12
TARGET_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
lst_drop = TARGET_COLS + ["id", "comment_text", "total_toxicity", "clean"]



## === cell 13
train_x = train.drop(lst_drop, axis=1)
print("Feature columns used for indirect features:", list(train_x.columns))
target_y = train[TARGET_COLS]



## === cell 14
test_x = test.drop(["id", "comment_text"], axis=1)
print("Test feature columns:", list(test_x.columns))
test_x.fillna(0, inplace=True)



## === cell 16
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
subm = pd.read_csv("../input/sample_submission.csv")

df = pd.concat([train["comment_text"], test["comment_text"]], axis=0).fillna("unknown")
nrow_train = train.shape[0]

vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
data = vectorizer.fit_transform(df)

X = MaxAbsScaler().fit_transform(data.tocsr())

cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
preds = np.zeros((test.shape[0], len(cols)))
loss = []

for i, col in enumerate(cols):
    print("=== Fit", col)
    model = LogisticRegression(max_iter=1000, n_jobs=5)
    model.fit(X[:nrow_train], train[col])
    preds[:, i] = model.predict_proba(X[nrow_train:])[:, 1]
    pred_train = model.predict_proba(X[:nrow_train])[:, 1]
    loss_val = log_loss(train[col], pred_train)
    print("log loss:", loss_val)
    loss.append(loss_val)

print("Mean column‑wise log loss:", np.mean(loss))

submission = pd.DataFrame({"id": subm["id"]})
submission = pd.concat([submission, pd.DataFrame(preds, columns=cols)], axis=1)
submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created.")

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/990547363.py in <cell line: 0>()
     11 
     12 # Ensure CSR format for MaxAbsScaler compatibility
---> 13 X = MaxAbsScaler().fit_transform(data.tocsr())
     14 
     15 cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in fit_transform(self, X, y, **fit_params)
    876         if y is None:
    877             # fit method of arity 1 (unsupervised transformation)
--> 878             return self.fit(X, **fit_params).transform(X)
    879         else:
    880             # fit method of arity 2 (supervised transformation)

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in fit(self, X, y)
   1167         # Reset internal state before fitting
   1168         self._reset()
-> 1169         return self.partial_fit(X, y)
   1170 
   1171     def partial_fit(self, X, y=None):

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in partial_fit(self, X, y)
   1202 
   1203         if sparse.issparse(X):
-> 1204             mins, maxs = min_max_axis(X, axis=0, ignore_nan=True)
   1205             max_abs = np.maximum(np.abs(mins), np.abs(maxs))
   1206         else:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py in min_max_axis(X, axis, ignore_nan)
    504     if isinstance(X, (sp.csr_matrix, sp.csc_matrix)):
    505         if ignore_nan:
--> 506             return _sparse_nan_min_max(X, axis=axis)
    507         else:
    508             return _sparse_min_max(X, axis=axis)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py in _sparse_nan_min_max(X, axis)
    472 
    473 def _sparse_nan_min_max(X, axis):
--> 474     return (_sparse_min_or_max(X, axis, np.fmin), _sparse_min_or_max(X, axis, np.fmax))
    475 
    476 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py in _sparse_min_or_max(X, axis, min_or_max)
    459         axis += 2
    460     if (axis == 0) or (axis == 1):
--> 461         return _min_or_max_axis(X, axis, min_or_max)
    462     else:
    463         raise ValueError("invalid axis, use 0 for rows, or 1 for columns")

/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py in _min_or_max_axis(X, axis, min_or_max)
    442             (value, (major_index, np.zeros(len(value)))), dtype=X.dtype, shape=(M, 1)
    443         )
--> 444     return res.A.ravel()
    445 
    446 

AttributeError: 'coo_matrix' object has no attribute 'A'
