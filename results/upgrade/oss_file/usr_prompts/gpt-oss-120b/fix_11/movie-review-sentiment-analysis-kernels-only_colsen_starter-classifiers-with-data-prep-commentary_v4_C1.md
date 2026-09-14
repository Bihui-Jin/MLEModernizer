# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the sentiment of phrases.

## Metric
Classification accuracy.

## Submission Format
For each phrase in the test set, predict a label for the sentiment. Your submission should have a header and look like the following:

```
PhraseId,Sentiment
156061,2
156062,2
156063,2
...
```

## Dataset
The dataset is comprised of tab-separated files with phrases. Each phrase has a PhraseId. Each sentence has a SentenceId.

The sentiment labels are:

0 - negative

1 - somewhat negative

2 - neutral

3 - somewhat positive

4 - positive

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
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
xgboost==2.0.3

# 4. Data file paths

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

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import gc
from nltk.tokenize import TweetTokenizer
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion
from xgboost import XGBClassifier

np.random.seed(42)



## === cell 1
train = pd.read_csv("../input/train.tsv", sep="\t")
test = pd.read_csv("../input/test.tsv", sep="\t")
sub = pd.read_csv("../input/sampleSubmission.csv", sep=",")



## === cell 2
tfhash = [
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("hashing", HashingVectorizer(stop_words="english")),
]
vectorizer = FeatureUnion(tfhash)
train_vectorized = vectorizer.fit_transform(train.Phrase)  # no astype conversion
test_vectorized = vectorizer.transform(test.Phrase)  # no astype conversion



## === cell 3
xgb_cv = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    use_label_encoder=False,
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    n_jobs=1,  # single‑core per XGBoost instance
    tree_method="hist",
    random_state=42,
)

cv_scores = cross_val_score(
    xgb_cv,
    train_vectorized,
    train["Sentiment"],
    scoring="accuracy",
    n_jobs=1,  # sequential CV
    cv=3,
)
print(
    "Cross‑validation mean accuracy {0:.2f}%, std {1:.2f}%.".format(
        np.mean(cv_scores) * 100, np.std(cv_scores) * 100
    )
)

del xgb_cv, cv_scores
gc.collect()



## === cell 4
xgb = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    use_label_encoder=False,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,  # full parallelism for final fit
    tree_method="hist",
    random_state=42,
)
xgb.fit(train_vectorized, train["Sentiment"])

sub["Sentiment"] = xgb.predict(test_vectorized)
sub.to_csv("submission.csv", index=False)
