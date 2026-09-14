# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Build a model that recognizes toxicity and minimizes unintended bias with respect to mentions of identities.

## Metric
We combine several submetrics: An overall ROC-AUC for the full evaluation set, along with the ROC-AUCs on three specific subsets of the test set capturing different aspects of bias.

The final model score looks like:

$$
\text { score }=w_0 A U C_{\text {overall }}+\sum_{a=1}^A w_a M_p\left(m_{s, a}\right)
$$
where:
$A=$ number of submetrics $(3)$
$m_{s, a}=$ bias metric for identity subgroup $s$ using submetric $a$
$w_a=$ a weighting for the relative importance of each submetric; all four $w$ values set to 0.25

Overall AUC: This is the ROC-AUC for the full evaluation set.

### Bias AUCs
To measure unintended bias, we again calculate the ROC-AUC, this time on three specific subsets of the test set for each identity, each capturing a different aspect of unintended bias. 

**Subgroup AUC**: Here, we restrict the data set to only the examples that mention the specific identity subgroup. *A low value in this metric means the model does a poor job of distinguishing between toxic and non-toxic comments that mention the identity*.

**BPSN (Background Positive, Subgroup Negative) AUC**: Here, we restrict the test set to the non-toxic examples that mention the identity and the toxic examples that do not. *A low value in this metric means that the model confuses non-toxic examples that mention the identity with toxic examples that do not*, likely meaning that the model predicts higher toxicity scores than it should for non-toxic examples mentioning the identity.

**BNSP (Background Negative, Subgroup Positive) AUC**: Here, we restrict the test set to the toxic examples that mention the identity and the non-toxic examples that do not. *A low value here means that the model confuses toxic examples that mention the identity with non-toxic examples that do not*, likely meaning that the model predicts lower toxicity scores than it should for toxic examples mentioning the identity.

#### Generalized Mean of Bias AUCs
To combine the per-identity Bias AUCs into one overall measure, we calculate their generalized mean as defined below:

$$
M_p\left(m_s\right)=\left(\frac{1}{N} \sum_{s=1}^N m_s^p\right)^{\frac{1}{p}}
$$

where:
$M_p=$ the $p$ th power-mean function
$m_s=$ the bias metric $m$ calulated for subgroup $S$
$N=$ number of identity subgroups

For this competition, we use a $p$ value of -5 to encourage competitors to improve the model for the identity subgroups with the lowest model performance.

## Submission Format
```
id,prediction
7000000,0.0
7000001,0.0
etc.

```

## Dataset
The text of the individual comment is found in the `comment_text` column. Each comment in Train has a toxicity label (`target`), and models should predict the `target` toxicity for the Test data. This attribute (and all others) are fractional values which represent the fraction of human raters who believed the attribute applied to the given comment. For evaluation, test set examples with `target >= 0.5` will be considered to be in the positive class (toxic).

The data also has several additional toxicity subtype attributes. Models do not need to predict these attributes for the competition, they are included as an additional avenue for research. Subtype attributes are:

- severe_toxicity
- obscene
- threat
- insult
- identity_attack
- sexual_explicit

Additionally, a subset of comments have been labelled with a variety of identity attributes, representing the identities that are *mentioned* in the comment. The columns corresponding to identity attributes are listed below. Only identities shown below will be included in the evaluation calculation.

- **male**
- **female**
- **homosexual_gay_or_lesbian**
- **christian**
- **jewish**
- **muslim**
- **black**
- **white**
- **psychiatric_or_mental_illness**

### Files
- **train.csv** - the training set, which includes toxicity labels and subgroups
- **test.csv** - the test set, which does **not** include toxicity labels or subgroups
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

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
seaborn==0.12.2
sklearn-pandas==2.2.0
wordcloud==1.9.4
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (286 lines)
            sample_submission.csv (97321 lines)
            sample_submission.csv.zip (230.8 kB)
            test.csv (205781 lines)
            test.csv.zip (12.5 MB)
            train.csv (3820210 lines)
            train.csv.zip (285.9 MB)
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
        input/
            description.md (286 lines)
            sample_submission.csv (97321 lines)
            sample_submission.csv.zip (230.8 kB)
            test.csv (205781 lines)
            test.csv.zip (12.5 MB)
            train.csv (3820210 lines)
            train.csv.zip (285.9 MB)
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
        working/
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
```

-> data/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv has 97320 rows and 2 columns.
The columns are: id, prediction

-> data/jigsaw-unintended-bias-in-toxicity-classification/test.csv has 205780 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-unintended-bias-in-toxicity-classification/train.csv has 3820209 rows and 45 columns.
The columns are: id, target, comment_text, severe_toxicity, obscene, identity_attack, insult, threat, asian, atheist, bisexual, black, buddhist, christian, female... and 30 more columns

-> data/sample_submission.csv has 97320 rows and 2 columns.
The columns are: id, prediction

-> data/test.csv has 205780 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 3820209 rows and 45 columns.
The columns are: id, target, comment_text, severe_toxicity, obscene, identity_attack, insult, threat, asian, atheist, bisexual, black, buddhist, christian, female... and 30 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.79899

# 6. Current score

0.50194

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61079) has done: 'I fix the pipeline so it runs end-to-end by (1) removing the missing `all_data.csv` dependency and using the correct Kaggle input paths, (2) making NLTK stopwords work reliably in Kaggle by downloading them if absent, and (3) avoiding the extremely slow per-row test inference loop by vectorizing predictions in one shot (same LogisticRegression + TF-IDF core logic, just correct/efficient execution). I also remove notebook-only magics (`%matplotlib inline`) that crash in a `.py` run. Finally, I ensure the submission file is written as `submission.csv` with exactly `id,prediction` and the correct row order/length.'
- What this solution (achieved 0.50194) has done: 'You’re currently throwing away most of the training signal by (a) binarizing `target` at 0.7 and (b) training on only 10k rows, which severely limits ROC-AUC. To move the score upward toward 0.79899 without changing the core TF‑IDF + LogisticRegression approach, I (1) train on a larger but still time-safe slice of the original data, and (2) keep `target` as a continuous label in [0,1] and use `LogisticRegression` with sample weights to approximate a “soft-label” fit while still outputting proper probabilities for AUC. I also make the submission alignment robust by merging into `sample_submission` and filling any missing predictions (shouldn’t happen) with the mean prediction, ensuring a valid CSV with correct ordering/length. These are minimal changes that directly increase ROC-AUC while keeping the same model family, features, and inference semantics.'
- What this solution (achieved 0.50194) has done: 'I fix the crash in the model-comparison helper by ensuring the label passed to scikit-learn is a 1D array and by using `f1_score(y_true, y_pred)` in the correct argument order. I also stop the heavy helper from training very slow models on a huge sparse matrix by limiting that diagnostic function to LogisticRegression only (it’s not used for the submission model), which prevents timeouts while preserving the core TF‑IDF + LogisticRegression submission pipeline. Finally, I keep the actual training/inference path (cells 20–23) intact but make it a bit more robust (fill NaNs in labels/text, ensure submission length/order matches `sample_submission`) so it always writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import re
import string

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer



## === cell 1
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

_ = set(stopwords.words("english"))



## === cell 2
BASE = "../input/jigsaw-unintended-bias-in-toxicity-classification"
train_path = os.path.join(BASE, "train.csv")
test_path = os.path.join(BASE, "test.csv")
sub_path = os.path.join(BASE, "sample_submission.csv")

train = pd.read_csv(train_path, usecols=["id", "comment_text", "target"])
test = pd.read_csv(test_path, usecols=["id", "comment_text"])
sub = pd.read_csv(sub_path)



## === cell 3
train.info()



## === cell 4
train.target.value_counts(dropna=True).head()



## === cell 5
train.shape



## === cell 6
train["target"].isnull().sum()



## === cell 7
X = train[["id", "comment_text", "target"]].copy()
train.columns.values



## === cell 8
if "hindu" in train.columns:
    _hindu_head = train["hindu"].head()
else:
    _hindu_head = None

_hindu_head



## === cell 9
tox = int((X["target"] > 0.7).sum())
neut = int((X["target"] <= 0.7).sum())
no_of_rows = int(X.shape[0])



## === cell 10
print(f"{round((tox*100)/no_of_rows,3)}% data contains toxic comments")
print(f"{round((neut*100/no_of_rows),3)}% data contains neutral comments")



## === cell 11
alphanumeric = lambda x: re.sub(r"\w*\d\w*", " ", x)
punc_lower = lambda x: re.sub("[%s]" % re.escape(string.punctuation), " ", x.lower())
remove_n = lambda x: re.sub(r"\n", " ", x)
remove_non_ascii = lambda x: re.sub(r"[^\x00-\x7f]", r" ", x)


def _clean_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    return s.map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)


X["comment_text"] = _clean_series(X["comment_text"])



## === cell 12
test["comment_text"] = _clean_series(test["comment_text"])



## === cell 13
import wordcloud as _wordcloud_pkg  # noqa: F401
from wordcloud import WordCloud


def wordcloud(df, label):
    subset = df[df[label] > 0.7]
    text = subset.comment_text.values
    wc = WordCloud(background_color="white", max_words=4000)
    wc.generate(" ".join(text))

    plt.figure(figsize=(20, 20))
    plt.subplot(221)
    plt.axis("off")
    plt.title("Words frequented in {}".format(label), fontsize=20)
    plt.imshow(wc.recolor(colormap="gist_earth", random_state=244), alpha=0.98)




## === cell 14
pass



## === cell 15
N_TRAIN = 250_000  # chosen to remain time/memory-safe while improving beyond the 10k-row baseline
X_work = X.iloc[:N_TRAIN].copy()

X_work["target"] = (
    pd.to_numeric(X_work["target"], errors="coerce")
    .fillna(0.0)
    .astype(float)
    .clip(0.0, 1.0)
)
X_work.shape



## === cell 16
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier  # noqa: F401
from sklearn.naive_bayes import MultinomialNB, BernoulliNB  # noqa: F401
from sklearn.svm import LinearSVC  # noqa: F401
from sklearn.ensemble import RandomForestClassifier  # noqa: F401
from xgboost import XGBClassifier  # noqa: F401



## === cell 17
"""
df_done: data_tox_done, data_sev_done, ...
label: toxic, severe_toxic, ...
vectorizer values: CountVectorizer, TfidfVectorizer
gram_range values: (1,1) for unigram, (2,2) for bigram

Bugfixes:
- Ensure y is 1D (Series) not 2D DataFrame -> prevents sklearn ValueError.
- Use f1_score(y_true, y_pred) correct argument order.
- Keep this diagnostic helper lightweight to avoid timeouts on huge sparse matrices.
"""


def cv_tf_train_test(df_done, label, vectorizer, ngram):
    """Train/Test split"""
    Xtxt = df_done["comment_text"]
    y = df_done[label]
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = pd.Series(y).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        Xtxt, y, test_size=0.3, random_state=42, stratify=y
    )

    """ TF-IDF """
    cv1 = vectorizer(ngram_range=(ngram), stop_words="english")

    X_train_cv1 = cv1.fit_transform(X_train)
    X_test_cv1 = cv1.transform(X_test)

    lr = LogisticRegression(
        max_iter=2000, solver="liblinear", class_weight="balanced", random_state=42
    )
    lr.fit(X_train_cv1, y_train)
    y_pred = lr.predict(X_test_cv1)

    df_f1 = pd.DataFrame(
        {"F1 Score": [f1_score(y_test, y_pred)]}, index=["Log Regression"]
    )
    return df_f1




## === cell 18
balanced_train = X_work.copy()
balanced_train["target_bin"] = (balanced_train["target"] >= 0.5).astype(int)
balanced_train.head()



## === cell 19
import time

t0 = time.time()

df_tox_cv = cv_tf_train_test(
    balanced_train.rename(columns={"target_bin": "target"}),
    "target",
    TfidfVectorizer,
    (1, 1),
)
df_tox_cv.rename(columns={"F1 Score": "F1 Score(target_bin)"}, inplace=True)

t1 = time.time()

total = "Time taken: {} seconds".format(t1 - t0)
print(total)

df_tox_cv



## === cell 20
Xtxt = balanced_train.comment_text
y_soft = balanced_train["target"].values.astype(float)
y_bin = (y_soft >= 0.5).astype(int)

X_train, X_valid, y_train_bin, y_valid_bin, y_train_soft, y_valid_soft = (
    train_test_split(
        Xtxt, y_bin, y_soft, test_size=0.3, random_state=42, stratify=y_bin
    )
)

tfv = TfidfVectorizer(ngram_range=(1, 1), stop_words="english")

X_train_fit = tfv.fit_transform(X_train)
X_valid_fit = tfv.transform(X_valid)



## === cell 21
lr = LogisticRegression(
    max_iter=2000,
    solver="liblinear",
    class_weight="balanced",
    random_state=42,
)

sample_weight = (
    np.asarray(y_train_soft, dtype=float) * 0.999 + 0.0005
)  # avoid exact zeros
lr.fit(X_train_fit, y_train_bin, sample_weight=sample_weight)



## === cell 22
X_test_fit = tfv.transform(test["comment_text"])
pred = lr.predict_proba(X_test_fit)[:, 1].astype(float)

df_pred = pd.DataFrame({"id": test["id"].values, "prediction": pred})

df = sub[["id"]].merge(df_pred, on="id", how="left")
if df["prediction"].isnull().any():
    df["prediction"] = df["prediction"].fillna(float(df_pred["prediction"].mean()))

df["prediction"] = df["prediction"].astype(float).clip(0.0, 1.0)

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.columns.tolist())
print(
    "prediction stats:",
    float(df["prediction"].min()),
    float(df["prediction"].max()),
    float(df["prediction"].mean()),
)
