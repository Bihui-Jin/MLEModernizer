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

beautifulsoup4==4.13.4
emoji==2.15.0
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
import numpy as np
import pandas as pd
import collections
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import matplotlib_venn as venn
from bs4 import BeautifulSoup
import string, re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from scipy import sparse

warnings.filterwarnings("ignore")
stopword_list = set(stopwords.words("english"))
lem = WordNetLemmatizer()
tokenizer = TweetTokenizer()

print("Listing input directory:")
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



## === cell 1
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("DIMENSION OF DATABASE :")
print(">>> Dimension du train :", train.shape)
print(">>> Dimension du test :", test.shape)

print("\nMISSING VALUES :")
print(">>> Train")
print(train.isnull().sum())
print(">>> Test")
print(test.isnull().sum())

train["comment_text"].fillna("unknown", inplace=True)
test["comment_text"].fillna("unknown", inplace=True)

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
for col in list_classes:
    print(f"\nRépartition pour la variable {col} :\n", collections.Counter(train[col]))

train["total_toxicity"] = train.iloc[:, 2:].sum(axis=1)
train["clean"] = train["total_toxicity"] == 0
print("\nDistribution of Total Toxicity Labels (important for validation)")
print(pd.value_counts(train.total_toxicity))



## === cell 2
import emoji


def extract_emojis(s):
    return " ".join(c for c in s if c in emoji.UNICODE_EMOJI)


emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\u2122"
    "\u260E"
    "]+",
    flags=re.UNICODE,
)


def pipeline_emoji(df):
    df["list_emoji"] = df["comment_text"].apply(lambda x: extract_emojis(x))
    df["count_emoji"] = df["comment_text"].apply(lambda x: len(extract_emojis(x)))
    df["comment_text"] = df["comment_text"].apply(lambda x: emoji_pattern.sub(r"", x))


def indirect_features(df):
    ip_pattern = re.compile("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}")
    df["ip"] = df["comment_text"].apply(
        lambda x: re.findall("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}", str(x))
    )
    df["count_ip"] = df["ip"].apply(len)
    df["comment_text"] = df["comment_text"].apply(lambda x: ip_pattern.sub(r"", x))

    df["complete_link"] = df["comment_text"].apply(
        lambda x: " ".join(
            re.findall(
                "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                str(x),
            )
        )
    )
    df["count_links"] = df["complete_link"].apply(len)

    time_pattern = re.compile("\d{1,2}:\d{1,2}")
    date_pattern = re.compile(
        r"\d\d\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
        r"January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s\d{4}",
        flags=re.UNICODE,
    )
    df["time"] = df["comment_text"].apply(
        lambda x: re.findall("\d{1,2}:\d{1,2}", str(x))
    )
    df["time_flag"] = df["time"].apply(len)
    df["comment_text"] = df["comment_text"].apply(lambda x: time_pattern.sub(r"", x))
    df["date"] = df["comment_text"].apply(
        lambda x: re.findall(
            r"\d\d\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
            r"January|February|March|April|May|June|July|August|September|"
            r"October|November|December)\s\d{4}",
            str(x),
        )
    )
    df["date_flag"] = df["date"].apply(len)
    df["comment_text"] = df["comment_text"].apply(lambda x: date_pattern.sub(r"", x))

    user_pattern = re.compile("\[\[User(.*)")
    df["username"] = df["comment_text"].apply(
        lambda x: re.findall("\[\[User(.*)", str(x))
    )
    df["count_usernames"] = df["username"].apply(len)
    df["comment_text"] = df["comment_text"].apply(lambda x: user_pattern.sub(r"", x))

    divers_pattern = re.compile("\(UTC\)|\(utc\)", flags=re.UNICODE)
    df["comment_text"] = df["comment_text"].apply(lambda x: divers_pattern.sub(r"", x))

    pipeline_emoji(df)

    df["comment_text"] = df["comment_text"].replace(r"^\s*$", "NAN", regex=True)
    df["count_sent"] = df["comment_text"].apply(
        lambda x: len(re.findall("\n", str(x))) + 1
    )
    df["count_word"] = df["comment_text"].apply(lambda x: len(str(x).split()))
    df["count_unique_word"] = df["comment_text"].apply(
        lambda x: len(set(str(x).split()))
    )
    df["count_letters"] = df["comment_text"].apply(len)
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
        lambda row: (
            float(row["capitals"]) / float(row["total_length"])
            if row["total_length"] > 0
            else 0
        ),
        axis=1,
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
    df["word_unique_percent"] = (
        df["count_unique_word"] * 100 / df["count_word"].replace(0, np.nan)
    )
    df["punct_percent"] = (
        df["count_punctuations"] * 100 / df["count_word"].replace(0, np.nan)
    )


indirect_features(train)
indirect_features(test)
print("Indirect feature extraction finished.")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1511068796.py in <cell line: 0>()
    131 
    132 
--> 133 indirect_features(train)
    134 indirect_features(test)
    135 print("Indirect feature extraction finished.")

/tmp/ipykernel_11/1511068796.py in indirect_features(df)
     77     df["comment_text"] = df["comment_text"].apply(lambda x: divers_pattern.sub(r"", x))
     78 
---> 79     pipeline_emoji(df)
     80 
     81     df["comment_text"] = df["comment_text"].replace(r"^\s*$", "NAN", regex=True)

/tmp/ipykernel_11/1511068796.py in pipeline_emoji(df)
     20 
     21 def pipeline_emoji(df):
---> 22     df["list_emoji"] = df["comment_text"].apply(lambda x: extract_emojis(x))
     23     df["count_emoji"] = df["comment_text"].apply(lambda x: len(extract_emojis(x)))
     24     df["comment_text"] = df["comment_text"].apply(lambda x: emoji_pattern.sub(r"", x))

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in apply(self, func, convert_dtype, args, by_row, **kwargs)
   4922             args=args,
   4923             kwargs=kwargs,
-> 4924         ).apply()
   4925 
   4926     def _reindex_indexer(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
   1425 
   1426         # self.func is Callable
-> 1427         return self.apply_standard()
   1428 
   1429     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1505         #  Categorical (GH51645).
   1506         action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
-> 1507         mapped = obj._map_values(
   1508             mapper=curried, na_action=action, convert=self.convert_dtype
   1509         )

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

/tmp/ipykernel_11/1511068796.py in <lambda>(x)
     20 
     21 def pipeline_emoji(df):
---> 22     df["list_emoji"] = df["comment_text"].apply(lambda x: extract_emojis(x))
     23     df["count_emoji"] = df["comment_text"].apply(lambda x: len(extract_emojis(x)))
     24     df["comment_text"] = df["comment_text"].apply(lambda x: emoji_pattern.sub(r"", x))

/tmp/ipykernel_11/1511068796.py in extract_emojis(s)
      3 
      4 def extract_emojis(s):
----> 5     return " ".join(c for c in s if c in emoji.UNICODE_EMOJI)
      6 
      7 

/tmp/ipykernel_11/1511068796.py in <genexpr>(.0)
      3 
      4 def extract_emojis(s):
----> 5     return " ".join(c for c in s if c in emoji.UNICODE_EMOJI)
      6 
      7 

AttributeError: module 'emoji' has no attribute 'UNICODE_EMOJI'

## === cell 3
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
    "he's": "he is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "might've": "might have",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "we'd": "we would",
    "we're": "we are",
    "weren't": "were not",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you would",
    "you're": "you are",
    "you've": "you have",
}


def expand_contractions(sentence, contraction_mapping):
    pattern = re.compile(
        "({})".format("|".join(contraction_mapping.keys())),
        flags=re.IGNORECASE | re.DOTALL,
    )

    def replace(match):
        word = match.group(0)
        first_char = word[0]
        expanded = contraction_mapping.get(word) or contraction_mapping.get(
            word.lower()
        )
        return first_char + expanded[1:] if expanded else word

    return pattern.sub(replace, sentence)


try:
    import unidecode
except ImportError:

    class _Dummy:
        @staticmethod
        def unidecode(s):
            return s

    unidecode = _Dummy()


def remove_accent_before_tokens(s):
    return unidecode.unidecode(s)


def remove_before_token(sentence, keep_apostrophe=False):
    sentence = sentence.strip()
    if keep_apostrophe:
        pattern = r"[?|$|&|*|%|@|(|)|~]"
    else:
        pattern = r"[^a-zA-Z0-9]"
    return re.sub(pattern, r" ", sentence)


print("Pre‑processing utilities loaded.")



## === cell 4
pass



## === cell 5
from bs4 import BeautifulSoup


def preprocessing_clean(comment):
    comment = BeautifulSoup(comment, "html.parser").get_text()
    comment = comment.lower()
    comment = re.sub("\\n", "", comment)
    comment = re.sub("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}", "", comment)
    comment = re.sub("\[\[.*\]", "", comment)
    words = tokenizer.tokenize(comment)
    words = [CONTRACTION_MAP.get(w, w) for w in words]
    words = [w for w in words if w not in stopword_list]
    return " ".join(words)


print("Sample before cleaning:", train.comment_text.iloc[23])
print("Sample after cleaning :", preprocessing_clean(train.comment_text.iloc[23]))



## === cell 6
clean_corpus = train.comment_text.apply(preprocessing_clean)
print("Example not cleaned :", clean_corpus.iloc[42])
print("Example cleaned     :", clean_corpus.iloc[42])
print("Cleaning completed.")



## === cell 7
print("Cleaned entry 23:", clean_corpus.iloc[23])



## === cell 8
pass



## === cell 9
tfidf_word = TfidfVectorizer()
X_tfidf_word = tfidf_word.fit_transform(clean_corpus)

tfidf_char = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=False)
X_tfidf_char = tfidf_char.fit_transform(clean_corpus)

X_tfidf = sparse.hstack([X_tfidf_word, X_tfidf_char])

features_tfidf_word = np.array(tfidf_word.get_feature_names_out())
print("Word‑level TFIDF features count:", len(features_tfidf_word))
features_tfidf_char = np.array(tfidf_char.get_feature_names_out())
print("Char‑level TFIDF features count:", len(features_tfidf_char))



## === cell 10
clean_corpus.head()



## === cell 11
pass



## === cell 12
X_tfidf_word = tfidf_word.transform(clean_corpus)
X_tfidf_char = tfidf_char.transform(clean_corpus)
X_tfidf = sparse.hstack([X_tfidf_word, X_tfidf_char])



## === cell 13
pass




## === cell 14
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




## === cell 15
TARGET_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
lst_drop = TARGET_COLS.copy()
lst_drop.extend(["id", "comment_text", "total_toxicity", "clean"])
print("Columns to drop for indirect‑feature model:", lst_drop)



## === cell 16
train_x = train.drop(lst_drop, axis=1)
target_y = train[TARGET_COLS]
test_x = test.drop(["id", "comment_text"], axis=1)
test_x.fillna(0, inplace=True)



## === cell 17
pass



## === cell 18
pass



## === cell 19
df = pd.concat([train["comment_text"], test["comment_text"]], axis=0).fillna("unknown")
vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
data = vectorizer.fit_transform(df)

X = MaxAbsScaler().fit_transform(data)

col = TARGET_COLS
n_train = train.shape[0]
preds = np.zeros((test.shape[0], len(col)))
losses = []

for i, label in enumerate(col):
    print(f"=== Training model for {label} ===")
    model = LogisticRegression(max_iter=1000, n_jobs=5)
    model.fit(X[:n_train], train[label])
    preds[:, i] = model.predict_proba(X[n_train:])[:, 1]
    pred_train = model.predict_proba(X[:n_train])[:, 1]
    loss = log_loss(train[label], pred_train)
    print(f"log loss for {label}: {loss:.5f}")
    losses.append(loss)

print("Mean column‑wise log loss:", np.mean(losses))

submission = pd.DataFrame({"id": test["id"]})
for i, label in enumerate(col):
    submission[label] = preds[:, i]

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1159946677.py in <cell line: 0>()
      4 
      5 # Fixed: pass the CSR matrix directly to MaxAbsScaler (no .tocsr() conversion needed)
----> 6 X = MaxAbsScaler().fit_transform(data)
      7 
      8 col = TARGET_COLS

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
