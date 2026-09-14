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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
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

0.46668

# 6. Current score

0.55786

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.55786) has done: 'The changes fix three runtime errors that stopped the notebook from completing and ensure a proper CSV submission is written:
1. **Seaborn countplot** now receives the correct arguments.
2. **KFold** is configured with `shuffle=True` so the `random_state` argument is valid.
3. **pd.concat** now uses the `axis` parameter correctly.
These minimal fixes let the pipeline train, evaluate, and generate a valid `submission.csv` without altering the core modeling logic.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from string import punctuation
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from wordcloud import WordCloud, STOPWORDS

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix



## === cell 1
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")



## === cell 2
train.head()
print("--- Shape ---")
print(train.shape)
print("--- Missing values ---")
print(train.isnull().sum() * 100 / len(train))



## === cell 3
sns.countplot(x="author", data=train)




## === cell 4
def build_corpus(data):
    data = str(data)
    corpus = ""
    for sent in data:
        corpus += str(sent)
    return corpus




## === cell 5
eap = train[train.author == "EAP"]
hpl = train[train.author == "HPL"]
mws = train[train.author == "MWS"]



## === cell 6
plt.figure(figsize=(15, 10))
plt.subplot(331)
eap_wc = WordCloud(background_color="white", max_words=100, stopwords=STOPWORDS)
eap_wc.generate(build_corpus(eap.text))
plt.title("Edgar Allan Poe", fontsize=20)
plt.imshow(eap_wc, interpolation="bilinear")
plt.axis("off")

plt.subplot(332)
hpl_wc = WordCloud(background_color="white", max_words=100, stopwords=STOPWORDS)
hpl_wc.generate(build_corpus(hpl.text))
plt.title("HP Lovecraft", fontsize=20)
plt.imshow(hpl_wc, interpolation="bilinear")
plt.axis("off")

plt.subplot(333)
mws_wc = WordCloud(background_color="white", max_words=100, stopwords=STOPWORDS)
mws_wc.generate(build_corpus(mws.text))
plt.title("Marry Shelley", fontsize=20)
plt.imshow(mws_wc, interpolation="bilinear")
plt.axis("off")



## === cell 7
le = LabelEncoder()
author_encoded = le.fit_transform(train.author)



## === cell 8
seed = 12
X_train, X_test, y_train, y_test = train_test_split(
    train.text, author_encoded, test_size=0.3, random_state=seed
)
metric = "accuracy"
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)




## === cell 9
class ColumnExtractor(TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        Xcols = X[self.cols]
        return Xcols

    def fit(self, X, y=None):
        return self


class ModelTransformer(TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(self.model.predict(X))




## === cell 10
class CommaCountTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        return pd.DataFrame(X.apply(lambda x: x.count(",")))

    def fit(self, X, y=None, **fit_params):
        return self


class WordCountTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        return pd.DataFrame(X.apply(lambda x: len(str(x).split())))

    def fit(self, X, y=None, **fit_params):
        return self


class LengthTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        return pd.DataFrame(X.apply(len))

    def fit(self, X, y=None, **fit_params):
        return self




## === cell 11
pipeline = Pipeline(
    [
        (
            "features",
            FeatureUnion(
                [
                    ("comma_count", CommaCountTransformer()),
                    ("word_count", WordCountTransformer()),
                    ("text_length", LengthTransformer()),
                    ("count_vect", CountVectorizer(lowercase=False)),
                    ("tf_idf", TfidfVectorizer()),
                ]
            ),
        ),
        ("classifier", MultinomialNB()),
    ]
)



## === cell 12
clf_pipe = pipeline.fit(X_train, y_train)
score_pipe = cross_val_score(clf_pipe, X_train, y_train, cv=kfold, scoring=metric)
print(
    "Mean CV score = %.3f, Std deviation = %.3f"
    % (np.mean(score_pipe), np.std(score_pipe))
)
score_pipe_test = clf_pipe.score(X_test, y_test)
print("Test set score = %.3f" % (score_pipe_test))



## === cell 13
conf_mat = confusion_matrix(y_test, clf_pipe.predict(X_test))
sns.heatmap(conf_mat, annot=True, fmt="d")
plt.xticks(ticks=range(3), labels=("EAP", "HPL", "MWS"), rotation=0)
plt.yticks(ticks=range(3), labels=("EAP", "HPL", "MWS"), rotation=0)



## === cell 14
target_names = ["EAP", "HPL", "MWS"]
y_pred = pd.DataFrame(clf_pipe.predict_proba(test.text), columns=target_names)
submission = pd.concat([test["id"], y_pred], axis=1)
submission.to_csv("./submission.csv", index=False)
