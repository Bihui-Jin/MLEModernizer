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
sklearn-pandas==2.2.0
tf_keras==2.18.0
wordcloud==1.9.4
xgboost==2.0.3

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

0.40609

# 6. Current score

0.52355

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.52355) has done: 'The error occurs because the logistic regression model was trained on numeric class labels (0, 1, 2), so `clf.classes_` contains numbers, not the original string labels “EAP”, “HPL”, “MWS”. When creating the probability DataFrame we therefore need to map those numeric columns back to the correct author names before re‑ordering. The fix builds the column list from the original `mapping_target` and uses it directly, eliminating the KeyError and producing a correctly formatted submission file.'

# 9. Code solution

## === cell 0
import os, sys, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

np.random.seed(25)

train_path = "../input/train.csv"
test_path = "../input/test.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)



## === cell 1
mapping_target = {"EAP": 0, "HPL": 1, "MWS": 2}
train = train.replace({"author": mapping_target})



## === cell 2
import re


def cleanData(text, lowercase=False):
    txt = str(text)
    txt = re.sub(r"[^A-Za-z\s]", " ", txt)  # keep only letters and spaces
    if lowercase:
        txt = txt.lower()
    return txt


train["text"] = train["text"].map(lambda x: cleanData(x, lowercase=True))
test["text"] = test["text"].map(lambda x: cleanData(x, lowercase=True))



## === cell 3
MAX_FEATURES = 10000
vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
X_train = vectorizer.fit_transform(train["text"])
y_train = train["author"].values
X_test = vectorizer.transform(test["text"])



## === cell 4
clf = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1,
    random_state=25,
)
clf.fit(X_train, y_train)



## === cell 5
class_order = ["EAP", "HPL", "MWS"]
numeric_to_name = {v: k for k, v in mapping_target.items()}
ordered_names = [numeric_to_name[i] for i in range(len(class_order))]
prob_df = pd.DataFrame(test_probs := clf.predict_proba(X_test), columns=ordered_names)
prob_df = prob_df[class_order]

submission = pd.DataFrame(
    {
        "id": test["id"],
        "EAP": prob_df["EAP"],
        "HPL": prob_df["HPL"],
        "MWS": prob_df["MWS"],
    }
)

submission_path = "result.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
