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

1.1111

# 6. Current score

0.61567

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.61567) has done: 'I adjust the data‑loading cell to correctly locate the training and test CSV files inside the nested Kaggle input folder, and keep the rest of the pipeline unchanged. This fixes the FileNotFoundError so the script can run end‑to‑end and produce a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, confusion_matrix



## === cell 1
possible_dirs = [
    os.path.join("input"),  # original attempt
    os.path.join("input", "spooky-author-identification"),
    os.path.join("/kaggle", "input", "spooky-author-identification"),
    os.path.join("/kaggle", "input"),  # fallback root
]


def find_file(filename):
    for d in possible_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not locate {filename} in any of the searched directories."
    )


train_path = find_file("train.csv")
test_path = find_file("test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)



## === cell 2
print("Class distribution:")
print(train["author"].value_counts())
print("\nTraining shape:", train.shape)
print("Test shape:", test.shape)



## === cell 3
le = LabelEncoder()
y = le.fit_transform(train["author"])
X = train["text"]



## === cell 4
seed = 12
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.30, random_state=seed, stratify=y
)



## === cell 5
pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(stop_words="english")),
        (
            "clf",
            LogisticRegression(
                multi_class="multinomial", solver="lbfgs", max_iter=1000, n_jobs=-1
            ),
        ),
    ]
)

pipeline.fit(X_train, y_train)

val_pred_proba = pipeline.predict_proba(X_val)
val_logloss = log_loss(y_val, val_pred_proba)
print(f"Validation Log‑Loss: {val_logloss:.5f}")



## === cell 6
test_pred_proba = pipeline.predict_proba(test["text"])

submission = pd.DataFrame(test_pred_proba, columns=le.classes_)
submission.insert(0, "id", test["id"])

submission_path = "./submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
