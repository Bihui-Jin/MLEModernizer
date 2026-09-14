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

# 5. Target score

0.5252

# 6. Current score

0.58661

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.58661) has done: 'The crashes come from converting sparse text features to `np.matrix` via `.todense()`, which scikit-learn no longer accepts; keeping the data as a CSR sparse matrix fixes both the Naive Bayes and SGD pipelines. I make the minimal changes needed to (1) avoid `.todense()`/`np.matrix`, (2) actually fit each model on the intended features, and (3) write a valid `submission.csv` with the required columns. To nudge accuracy toward your target without changing the core approach, I prefer the second (CountVectorizer + SGDClassifier) model for the final submission, since it’s typically stronger than MultinomialNB here while keeping the same architecture family and training semantics. All file paths remain unchanged and the script run end-to-end and produce a `.csv` submission.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os

print(os.listdir("../input"))



## === cell 1
import pandas as pd



## === cell 2
train = pd.read_csv("../input/train.tsv", sep="\t")
train.head()



## === cell 3
test = pd.read_csv("../input/test.tsv", sep="\t")
test.head()



## === cell 4
train["Sentiment"].unique()



## === cell 5
train.shape



## === cell 6
test.shape



## === cell 7
train.isnull().sum(axis=0)



## === cell 8
test.isnull().sum(axis=0)



## === cell 9
train["SentenceId"].value_counts()[0:5]



## === cell 10
test["SentenceId"].value_counts()[0:5]



## === cell 11
len(train["SentenceId"].unique()) + len(test["SentenceId"].unique())



## === cell 12
len(train["PhraseId"].unique()) + len(test["PhraseId"].unique())



## === cell 13
from sklearn.feature_extraction.text import TfidfVectorizer



## === cell 14
tfidf = TfidfVectorizer(min_df=0.01, max_df=0.9, norm=None)



## === cell 15
tfidf.fit(train["Phrase"])



## === cell 16
X_train = tfidf.transform(train["Phrase"])
Y_train = train["Sentiment"]



## === cell 17
from sklearn.naive_bayes import MultinomialNB



## === cell 18
NB = MultinomialNB()
NB.fit(X_train, Y_train)



## === cell 19
test_tfidf = tfidf.transform(test["Phrase"])



## === cell 20
y_pred_nb = NB.predict(test_tfidf)



## === cell 21
y_pred_nb_df = pd.DataFrame(y_pred_nb, columns=["Sentiment"])
sub_nb = pd.concat([test["PhraseId"].reset_index(drop=True), y_pred_nb_df], axis=1)
sub_nb.head()



## === cell 22
sub_nb.to_csv("submission_nb.csv", index=False)



## === cell 23
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer



## === cell 24
pattern = RegexpTokenizer(r"[a-zA-Z0-9]+")



## === cell 25
cv = CountVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 1),
    tokenizer=pattern.tokenize,
)



## === cell 26
cv.fit(train["Phrase"])



## === cell 27
train_cv = cv.transform(train["Phrase"])
test_cv = cv.transform(test["Phrase"])



## === cell 28
from sklearn.linear_model import SGDClassifier



## === cell 29
sv = SGDClassifier(max_iter=200, loss="log_loss", random_state=42)
sv.fit(train_cv, train["Sentiment"])



## === cell 30
y_pred_sgd = sv.predict(test_cv)



## === cell 31
y_pred_sgd_df = pd.DataFrame(y_pred_sgd, columns=["Sentiment"])
sub = pd.concat([test["PhraseId"].reset_index(drop=True), y_pred_sgd_df], axis=1)
sub.head()



## === cell 32
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
