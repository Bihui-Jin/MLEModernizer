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
import os
import numpy as np
import pandas as pd
import collections
import warnings
import string
import re

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

sns.set_style("dark")
color = sns.color_palette()

try:
    stopword_list = set(stopwords.words("english"))
except Exception:
    stopword_list = set()

lem = WordNetLemmatizer()
tokenizer = TweetTokenizer()

TARGET_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

BASE1 = "../input/jigsaw-toxic-comment-classification-challenge"
BASE2 = "../input"
base_path = BASE1 if os.path.exists(os.path.join(BASE1, "train.csv")) else BASE2

train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")
sample_path = os.path.join(base_path, "sample_submission.csv")

print("Using data from:", base_path)
print(
    "Files exist:",
    os.path.exists(train_path),
    os.path.exists(test_path),
    os.path.exists(sample_path),
)



## === cell 1
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("DIMENSION OF DATABASE : ")
print(">>> Dimension du train :", train.shape)
print(">>> Dimension du test  :", test.shape)

g = train["id"].value_counts()
_ = g.where(g > 1).dropna()

g = test["id"].value_counts()
_ = g.where(g > 1).dropna()

print("\nMISSING VALUES : ")
print(">>> Check for missing values in Train dataset")
print(train.isnull().sum())
print(">>> Check for missing values in Test dataset")
print(test.isnull().sum())

print('>>> Filling NA with "unknown"')
train["comment_text"] = train["comment_text"].fillna("unknown")
test["comment_text"] = test["comment_text"].fillna("unknown")

for col in TARGET_COLS:
    print(
        "\nRépartition pour la variable ", col, " : \n", collections.Counter(train[col])
    )

rowsums = train[TARGET_COLS].sum(axis=1)
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

    if len(stopword_list) > 0:
        df["count_stopwords"] = df["comment_text"].apply(
            lambda x: len([w for w in str(x).lower().split() if w in stopword_list])
        )
    else:
        df["count_stopwords"] = 0

    df["mean_word_len"] = df["comment_text"].apply(
        lambda x: (
            float(np.mean([len(w) for w in str(x).split()]))
            if len(str(x).split()) > 0
            else 0.0
        )
    )

    df["total_length"] = df["comment_text"].apply(len)
    df["capitals"] = df["comment_text"].apply(
        lambda comment: sum(1 for c in str(comment) if c.isupper())
    )
    df["caps_vs_length"] = df.apply(
        lambda row: (
            float(row["capitals"]) / float(row["total_length"])
            if float(row["total_length"]) > 0
            else 0.0
        ),
        axis=1,
    )

    df["count_punctuations"] = df["comment_text"].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation])
    )
    df["num_exclamation_marks"] = df["comment_text"].apply(
        lambda comment: str(comment).count("!")
    )
    df["num_question_marks"] = df["comment_text"].apply(
        lambda comment: str(comment).count("?")
    )
    df["num_symbols"] = df["comment_text"].apply(
        lambda comment: sum(str(comment).count(w) for w in "*&$%")
    )
    df["num_smilies"] = df["comment_text"].apply(
        lambda comment: sum(str(comment).count(w) for w in (":-)", ":)", ";-)", ";)"))
    )

    df["word_unique_percent"] = df.apply(
        lambda r: (
            (r["count_unique_word"] * 100.0 / r["count_word"])
            if r["count_word"] > 0
            else 0.0
        ),
        axis=1,
    )
    df["punct_percent"] = df.apply(
        lambda r: (
            (r["count_punctuations"] * 100.0 / r["count_word"])
            if r["count_word"] > 0
            else 0.0
        ),
        axis=1,
    )


indirect_features(train)
indirect_features(test)



## === cell 3
features = [
    "count_sent",
    "count_word",
    "count_unique_word",
    "count_letters",
    "count_words_upper",
    "count_words_title",
    "count_stopwords",
    "mean_word_len",
    "total_length",
    "capitals",
    "caps_vs_length",
    "count_punctuations",
    "num_exclamation_marks",
    "num_question_marks",
    "num_symbols",
    "num_smilies",
    "word_unique_percent",
    "punct_percent",
]
columns = TARGET_COLS

rows = [{c: train[f].corr(train[c]) for c in columns} for f in features]
df_correlations = pd.DataFrame(rows, index=features)

try:
    ax = sns.heatmap(df_correlations, vmin=-0.2, vmax=0.2, center=0.0)
    plt.close()
except Exception:
    pass

df_correlations.head()



## === cell 4
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
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "this's": "this is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
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
        expanded_contraction = (
            contraction_mapping.get(match)
            if contraction_mapping.get(match)
            else contraction_mapping.get(match.lower())
        )
        if expanded_contraction is None:
            return match
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence


def remove_accent_before_tokens(sentences):
    return sentences


def remove_before_token(sentence, keep_apostrophe=False):
    sentence = sentence.strip()
    if keep_apostrophe:
        pattern = r"[?|$|&|*|%|@|(|)|~]"
        filtered_sentence = re.sub(pattern, r" ", sentence)
    else:
        pattern = r"[^a-zA-Z0-9]"
        filtered_sentence = re.sub(pattern, r" ", sentence)
    return filtered_sentence


print("Fin")




## === cell 5
def preprocessing_clean(comment):
    """
    Receives comment string and returns cleaned string.
    Kept consistent with original intent; not used for final TF-IDF model below.
    """
    comment = str(comment).lower()
    comment = re.sub("\\n", "", comment)
    comment = re.sub(r"\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}", "", comment)
    comment = re.sub(r"\[\[.*\]", "", comment)

    words = tokenizer.tokenize(comment)
    words = [CONTRACTION_MAP[w] if w in CONTRACTION_MAP else w for w in words]

    if len(stopword_list) > 0:
        words = [w for w in words if w not in stopword_list]

    words = [lem.lemmatize(w, "v") for w in words]
    clean_sent = " ".join(words)
    return clean_sent


print(">>> Before cleaning")
print(train.comment_text.iloc[42])
print("\n>>> After cleaning")
print(preprocessing_clean(train.comment_text.iloc[42]))



## === cell 6
clean_corpus = train.comment_text.apply(lambda x: preprocessing_clean(x))
print("Not cleaned : ", train.comment_text.iloc[42][:120])
print("\nCleaned : ", clean_corpus.iloc[42][:120])
print("FIN")



## === cell 7
df_clean_corpus = clean_corpus.reset_index(drop=True)
df_final = pd.concat(
    [train.drop("comment_text", axis=1), df_clean_corpus.rename("comment_text")], axis=1
)
df_final.head()




## === cell 8
def multiclass_logloss(actual, predicted, eps=1e-15):
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2
    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota




## === cell 9
lst_drop = list(TARGET_COLS) + ["id", "comment_text", "total_toxicity", "clean"]
train_x = train.drop(lst_drop, axis=1)
target_y = train[TARGET_COLS]

test_x = test.drop(["id", "comment_text"], axis=1)
test_x = test_x.fillna(0)

print("Indirect feature columns:", list(train_x.columns))



## === cell 10
from sklearn.model_selection import train_test_split

print("Using only Indirect features (sanity check; not final submission)")
X_train, X_valid, y_train, y_valid = train_test_split(
    train_x, target_y, test_size=0.25, random_state=42
)

_ = pd.read_csv(sample_path)
for j in TARGET_COLS:
    model_ind = LogisticRegression(C=3, max_iter=1000)
    model_ind.fit(X_train, y_train[j])

    preds_valid = model_ind.predict_proba(X_valid)[:, 1]
    preds_train = model_ind.predict_proba(X_train)[:, 1]

    print("Class:= " + j)
    print("Trainloss=log loss:", log_loss(y_train[j], preds_train))
    print("Validloss=log loss:", log_loss(y_valid[j], preds_valid))



## === cell 11
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
subm = pd.read_csv(sample_path)

df_all = pd.concat([train["comment_text"], test["comment_text"]], axis=0).fillna(
    "unknown"
)
nrow_train = train.shape[0]

vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
data = vectorizer.fit_transform(df_all)

data_csr = data.tocsr()
X = MaxAbsScaler().fit_transform(data_csr)

col = TARGET_COLS
preds = np.zeros((test.shape[0], len(col)), dtype=np.float64)

loss = []
for i, j in enumerate(col):
    print("===Fit " + j)
    model = LogisticRegression(max_iter=1000)
    model.fit(X[:nrow_train], train[j].values)

    preds[:, i] = model.predict_proba(X[nrow_train:])[:, 1]

    pred_train = model.predict_proba(X[:nrow_train])[:, 1]
    ll = log_loss(train[j].values, pred_train)
    print("log loss:", ll)
    loss.append(ll)

print("mean column-wise log loss:", float(np.mean(loss)))

submid = pd.DataFrame({"id": subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns=col)], axis=1)

submission = submission[["id"] + col]

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/695674105.py in <cell line: 0>()
     15 # Bug fix: ensure scaler sees CSR/CSC (not COO) to avoid attribute errors in this environment
     16 data_csr = data.tocsr()
---> 17 X = MaxAbsScaler().fit_transform(data_csr)
     18 
     19 col = TARGET_COLS

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
