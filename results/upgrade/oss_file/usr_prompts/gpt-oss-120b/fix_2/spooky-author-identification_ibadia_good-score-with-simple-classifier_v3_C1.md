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

0.37342

# 6. Current score

0.47997

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.47997) has done: 'I lower‑case the text, remove English stop‑words and add bi‑grams in the CountVectorizer (this usually yields richer features for both Naïve Bayes and Logistic Regression). I also give more voting weight to the LogisticRegression model, which tends to perform better on this type of text classification. These small adjustments are expected to reduce the log‑loss toward the target without changing the overall pipeline.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))




## === cell 1
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

train_df = pd.read_csv("../input/train.csv")
train_df.shape
train_df.head()
train_df["author_num"] = train_df.author.map({"EAP": 0, "HPL": 1, "MWS": 2})

train_df = shuffle(train_df)
train_df.head()

testing = pd.read_csv("../input/test.csv")
testing.head()




## === cell 2
X = train_df["text"]
y = train_df["author_num"]
print(X.shape)
print(y.shape)




## === cell 3
from nltk.corpus import stopwords

stopset = set(stopwords.words("english"))

vect = CountVectorizer(
    lowercase=True, stop_words="english", token_pattern=r"\w+|\,", ngram_range=(1, 2)
)
X_cv = vect.fit_transform(X)




## === cell 4
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier

models = [("MultiNB", MultinomialNB(alpha=0.01))]
models.append(("Logit", LogisticRegression(C=30, max_iter=1000)))
clf1 = VotingClassifier(models, voting="soft", weights=[1, 3])
clf1.fit(X_cv, y)
y_pred_test = clf1.predict_proba(X_cv)




## === cell 5
all_test = testing["text"]
all_test_v = vect.transform(all_test)
final_result = clf1.predict_proba(all_test_v)




## === cell 6
result = pd.DataFrame(final_result)
result.columns = ["EAP", "HPL", "MWS"]
result.insert(0, "id", testing["id"])
result.to_csv("resultfinal.csv", index=False)




## === cell 7
result.head()
