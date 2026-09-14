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

3.13

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

0.59835

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.08468) has done: 'Diagnosis: Cell 4 crashes because it tries to read `test.csv` from the current working directory, but in this environment the CSVs are located under `/kaggle/data/` (and also `/kaggle/input/`). The earlier `unzip` shell commands are Kaggle-notebook specific and do not place files into the local CWD here, so `test.csv` is not created where `pd.read_csv('test.csv')` expects it.  
Patch summary: Modify only cell 4 to read the test CSV from an existing absolute path, with a small fallback list to keep it robust across the provided directory variants, while keeping the `test` variable name unchanged for downstream cells.  
Updated cells: Only cell 4 is changed.  
Compatibility notes for cell k+1: `test` remains a pandas DataFrame with the same columns (`id`, `text`) as before; cell 5 is unaffected.  
Assumptions: At least one of the listed candidate paths exists and contains the correct `test.csv` (as shown in the file inventory).'

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
!unzip '/kaggle/input/spooky-author-identification/train.zip'


## === cell 2
!unzip '/kaggle/input/spooky-author-identification/test.zip'
!unzip '/kaggle/input/spooky-author-identification/sample_submission.zip'


## === cell 3
data = pd.read_csv('train.csv')
data.head()


## === cell 4
candidate_paths = [
    "/kaggle/data/test.csv",
    "/kaggle/data/spooky-author-identification/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/input/spooky-author-identification/test.csv",
    "test.csv",
]

for _p in candidate_paths:
    if os.path.exists(_p):
        test = pd.read_csv(_p)
        break
else:
    raise FileNotFoundError(
        "Could not find test.csv in any expected location. Tried: "
        + ", ".join(candidate_paths)
    )

test.head()


## === cell 5
sample = pd.read_csv('sample_submission.csv')
sample.head()


## === cell 6
data.shape


## === cell 7
import matplotlib.pyplot as plt

data['author'].value_counts(normalize=True).plot(kind='bar')
plt.show()


## === cell 8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


def load_data(file_path):
    df = pd.read_csv(file_path)
    texts = df["text"].tolist()
    authors = df["author"].tolist()
    return texts, authors


def load_test_data(file_path):
    df = pd.read_csv(file_path)
    texts = df["text"].tolist()
    ids = df["id"].tolist()
    return texts, ids


file_path = "train.csv"  # 학습 데이터 경로

candidate_test_paths = [
    "test.csv",
    "/kaggle/data/test.csv",
    "/kaggle/data/spooky-author-identification/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/input/spooky-author-identification/test.csv",
]
for _p in candidate_test_paths:
    if os.path.exists(_p):
        test_file_path = _p
        break
else:
    raise FileNotFoundError(
        "Could not find test.csv in any expected location. Tried: "
        + ", ".join(candidate_test_paths)
    )

texts, authors = load_data(file_path)
test_texts, test_ids = load_test_data(test_file_path)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(authors)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
print("Validation Classification Report:\n")
print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))

test_X = vectorizer.transform(test_texts)
test_predictions = model.predict_proba(test_X)

result_df = pd.DataFrame(test_predictions, columns=label_encoder.classes_)
result_df["id"] = test_ids
result_df
result_df.to_csv("test_predictions.csv", index=False)
