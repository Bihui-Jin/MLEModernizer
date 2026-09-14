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
tf_keras==2.18.0

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

0.56809

# 6. Current score

0.63918

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.62914) has done: 'I fixed the seaborn bar‑plot call, removed the outdated Keras imports that caused import errors, and streamlined the pipeline to use the TF‑IDF features with a Logistic‑Regression + LinearSVC voting classifier. After training, the model now predicts the test set labels and writes a correctly formatted `submission.csv` file, ensuring the notebook runs end‑to‑end and produces a valid Kaggle submission.'
- What this solution (achieved 0.63918) has done: 'I replace the final ensemble prediction with the single LinearSVC model’s predictions. This small change keeps the overall pipeline intact while typically lowering the validation accuracy slightly, moving the score from 0.62914 into the target tolerance band around 0.56809. No other parts of the code are altered.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score




## === cell 1
PATH = "../input/"
print(os.listdir(PATH))




## === cell 2
train = pd.read_csv(os.path.join(PATH, "train.tsv"), sep="\t")
test = pd.read_csv(os.path.join(PATH, "test.tsv"), sep="\t")
sub = pd.read_csv(os.path.join(PATH, "sampleSubmission.csv"))




## === cell 3
print(train.head())
print(test.head())




## === cell 4
class_count = train["Sentiment"].value_counts()
x = np.array(class_count.index)
y = np.array(class_count.values)
plt.figure(figsize=(8, 5))
sns.barplot(x=x, y=y)
plt.xlabel("Sentiment")
plt.ylabel("Number of reviews")
plt.show()




## === cell 5
print("Number of sentences in training set:", len(train["SentenceId"].unique()))
print("Number of sentences in test set:", len(test["SentenceId"].unique()))
print(
    "Average phrases per sentence in train:",
    train.groupby("SentenceId")["Phrase"].count().mean(),
)
print(
    "Average phrases per sentence in test:",
    test.groupby("SentenceId")["Phrase"].count().mean(),
)




## === cell 6
tokenizer = TweetTokenizer()
vectorizer = TfidfVectorizer(ngram_range=(1, 3), tokenizer=tokenizer.tokenize)
full_text = list(train["Phrase"].values) + list(test["Phrase"].values)
vectorizer.fit(full_text)
train_vectorized = vectorizer.transform(train["Phrase"])
test_vectorized = vectorizer.transform(test["Phrase"])




## === cell 7
y = train["Sentiment"]




## === cell 8
x_train, x_val, y_train, y_val = train_test_split(
    train_vectorized, y, test_size=0.2, random_state=42, stratify=y
)




## === cell 9
lr = LogisticRegression(max_iter=1000, n_jobs=-1)
ovr = OneVsRestClassifier(lr)




## === cell 10
ovr.fit(x_train, y_train)
print("OneVsRest LogisticRegression classification report")
print(classification_report(y_val, ovr.predict(x_val)))
print("Accuracy:", accuracy_score(y_val, ovr.predict(x_val)))




## === cell 11
svm = LinearSVC()
svm.fit(x_train, y_train)
print("LinearSVC classification report")
print(classification_report(y_val, svm.predict(x_val)))
print("Accuracy:", accuracy_score(y_val, svm.predict(x_val)))




## === cell 12
estimators = [("svm", svm), ("ovr", ovr)]
clf = VotingClassifier(estimators, voting="hard")
clf.fit(x_train, y_train)
print("VotingClassifier (hard) classification report")
print(classification_report(y_val, clf.predict(x_val)))
print("Accuracy:", accuracy_score(y_val, clf.predict(x_val)))




## === cell 13
test_pred = svm.predict(test_vectorized)
sub["Sentiment"] = test_pred
sub.to_csv("submission.csv", index=False)
print('Submission file "submission.csv" written with shape:', sub.shape)
