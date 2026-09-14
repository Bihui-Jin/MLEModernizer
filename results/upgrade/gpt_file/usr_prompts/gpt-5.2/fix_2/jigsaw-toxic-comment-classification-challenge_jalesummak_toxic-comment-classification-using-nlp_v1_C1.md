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

3.14

# 3. Installed packages

geopandas==0.14.4
joblib==1.5.2
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

0.6818365439509163

# 6. Current score

0.97221

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.97221) has done: 'I fix the primary runtime failure by correcting the dataset paths to the ones that actually exist in your environment, and I remove the unused `test_labels` load (it’s not needed for submission and may not exist). I also make the text preprocessing apply consistently to both train and test (same lowercasing/cleanup) so inference doesn’t break and the model sees the same feature space. To match the competition’s ROC-AUC metric, I switch the submission predictions from hard class labels (`predict`) to probabilities (`predict_proba`) while keeping the same core TF‑IDF + OneVsRest LogisticRegression approach. Finally, I ensure the generated `submission.csv` has the exact required columns/order by using `sample_submission.csv` as the template.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:50]:
        print(os.path.join(dirname, filename))



## === cell 1
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



## === cell 2
BASE = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"

train = pd.read_csv(f"{BASE}/train.csv")
test = pd.read_csv(f"{BASE}/test.csv")
sample_sub = pd.read_csv(f"{BASE}/sample_submission.csv")

print(train.shape, test.shape, sample_sub.shape)



## === cell 3
train.head()



## === cell 4
train.info()



## === cell 5
labels = train.columns[2:]

train[labels].sum().sort_values(ascending=False).plot(kind="bar", figsize=(10, 5))

plt.title("Toxic Comment Label Distribution")
plt.ylabel("Number of Comments")
plt.xlabel("Toxic Categories")

plt.show()



## === cell 6
train["comment_length"] = train["comment_text"].astype(str).str.len()



## === cell 7
plt.figure(figsize=(10, 5))
sns.histplot(train["comment_length"], bins=50)
plt.title("Comment Length Distribution")
plt.show()



## === cell 8
plt.figure(figsize=(10, 5))
sns.boxplot(x=train["toxic"], y=train["comment_length"])
plt.title("Comment Length vs Toxicity")
plt.show()



## === cell 9
plt.figure(figsize=(8, 6))
sns.heatmap(train.iloc[:, 2:8].corr(), annot=True, cmap="coolwarm")
plt.title("Toxic Label Correlation")
plt.show()




## === cell 10
def basic_clean_text(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    s = s.str.lower()
    s = s.str.replace(r"[^\w\s]", "", regex=True)  # punctuation removal
    s = s.str.replace(r"\d+", "", regex=True)  # number removal
    s = s.str.replace("\n", " ", regex=False)
    s = s.str.replace("\r", " ", regex=False)
    return s


train["comment_text"] = basic_clean_text(train["comment_text"])
test["comment_text"] = basic_clean_text(test["comment_text"])



## === cell 11
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))  # stopwords removal



## === cell 12
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

train["comment_text"] = train["comment_text"].apply(
    lambda x: [word for word in word_tokenize(x) if word not in stop_words]
)

train["comment_text"].head()



## === cell 13
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
lemmatizer = WordNetLemmatizer()

train["comment_text"] = train["comment_text"].apply(
    lambda x: [lemmatizer.lemmatize(word) for word in x]
)



## === cell 14
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer



## === cell 15
test_tokens = test["comment_text"].apply(
    lambda x: [word for word in word_tokenize(x) if word not in stop_words]
)
test_tokens = test_tokens.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

train["comment_text"] = train["comment_text"].apply(
    lambda x: " ".join(x) if isinstance(x, list) else x
)
test["comment_text"] = test_tokens.apply(
    lambda x: " ".join(x) if isinstance(x, list) else x
)



## === cell 16
x = train["comment_text"]
y = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]



## === cell 17
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)



## === cell 18
vect = CountVectorizer(ngram_range=(1, 2))



## === cell 19
x_train_vec = vect.fit_transform(x_train)
x_test_vec = vect.transform(x_test)



## === cell 20
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(ngram_range=(1, 2))
x_sayisal = vect.fit_transform(train["comment_text"])
y = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]



## === cell 21
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression



## === cell 22
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))



## === cell 23
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x_sayisal, y, test_size=0.2, random_state=42
)



## === cell 24
model.fit(x_train, y_train)



## === cell 25
pred = model.predict(x_test)



## === cell 26
from sklearn.metrics import f1_score

f1_score(y_test, pred, average="micro")



## === cell 27
from sklearn.metrics import f1_score, precision_score, recall_score

label_scores = pd.DataFrame(
    {
        "Label": y.columns,
        "Precision": precision_score(y_test, pred, average=None, zero_division=0),
        "Recall": recall_score(y_test, pred, average=None, zero_division=0),
        "F1 Score": f1_score(y_test, pred, average=None, zero_division=0),
    }
)

label_scores.sort_values("F1 Score", ascending=False)



## === cell 28
plt.figure(figsize=(8, 4))
sns.barplot(
    data=label_scores.sort_values("F1 Score", ascending=False), x="Label", y="F1 Score"
)
plt.title("F1 Score by Toxicity Label")
plt.xlabel("Label")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.show()



## === cell 29
test_vec = vect.transform(test["comment_text"])

pred_test_proba = model.predict_proba(test_vec)

submission = sample_sub.copy()
submission.iloc[:, 1:] = pred_test_proba
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## === cell 30
import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

x = train["comment_text"]
y = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
x_vec = vectorizer.fit_transform(x)

model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(x_vec, y)

joblib.dump(model, "toxic_model.pkl")
joblib.dump(vectorizer, "toxic_vectorizer.pkl")
joblib.dump(y.columns.tolist(), "toxic_columns.pkl")
print("Saved model/vectorizer/columns artifacts.")
