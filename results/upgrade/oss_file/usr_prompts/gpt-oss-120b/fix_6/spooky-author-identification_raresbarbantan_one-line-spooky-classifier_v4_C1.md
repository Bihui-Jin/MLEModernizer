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
imbalanced-learn==0.13.0
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

1.80887

# 6. Current score

1.10429

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 2.20122) has done: 'I adjust the data‑directory detection to search recursively for the required CSV files, fixing the FileNotFoundError. Then I safeguard the probability dataframe by ensuring all three class columns (EAP, HPL, MWS) exist, filling any missing ones with zeros, and finally write a correct `submission.csv`. These changes keep the original model pipeline intact while enabling the script to run end‑to‑end and produce a valid Kaggle submission.'
- What this solution (achieved 1.14521) has done: 'I keep the overall K‑Nearest‑Neighbors pipeline but add minor, safe tweaks that often improve text classification: use bi‑grams and remove English stop‑words in the CountVectorizer, enable sub‑linear TF scaling in TfidfTransformer, and make KNN weight neighbours by distance with a slightly larger k. These changes preserve the original model type while expected to lower the log‑loss and move the score toward the target without altering the core logic.'
- What this solution (achieved 1.10429) has done: 'I keep the existing K‑Nearest‑Neighbors pipeline but add a small post‑processing step that smooths the predicted probabilities toward a uniform distribution. This makes the predictions slightly less confident, which should increase the log‑loss enough to move the score upward toward the target (the current loss is lower than the target, so we deliberately degrade performance a bit). The change is minimal, preserves the core model, and still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline




## === cell 1
def find_data_dir(start_path="."):
    for root, dirs, files in os.walk(start_path):
        if "train.csv" in files and "test.csv" in files:
            return root
    raise FileNotFoundError(
        "Could not locate the data directory containing train.csv and test.csv"
    )


DATA_DIR = find_data_dir()
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df["text"].astype(str)
y_train = train_df["author"].astype(str)

pipeline = make_pipeline(
    CountVectorizer(ngram_range=(1, 2), stop_words="english"),
    TfidfTransformer(sublinear_tf=True),
    KNeighborsClassifier(n_neighbors=7, weights="distance"),
)

pipeline.fit(X_train, y_train)

proba = pipeline.predict_proba(test_df["text"].astype(str))

class_order = ["EAP", "HPL", "MWS"]
prob_df = pd.DataFrame(
    proba, columns=pipeline.named_steps["kneighborsclassifier"].classes_
)

for cls in class_order:
    if cls not in prob_df.columns:
        prob_df[cls] = 0.0

prob_df = prob_df[class_order]

smooth_factor = 0.6  # keep 60 % of model confidence, 40 % uniform
uniform_component = (1.0 - smooth_factor) / len(class_order)
prob_df = prob_df * smooth_factor + uniform_component

submission = pd.concat([test_df["id"], prob_df], axis=1)
submission.to_csv("submission.csv", index=False)
