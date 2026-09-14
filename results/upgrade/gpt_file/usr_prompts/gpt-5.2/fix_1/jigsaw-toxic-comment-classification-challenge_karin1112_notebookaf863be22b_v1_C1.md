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

3.13

# 3. Installed packages

geopandas==0.14.4
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

0.5

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import pandas as pd
import numpy as np
import re
import string
import gc
import zipfile
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

print("Loading and extracting data...")

data_base_path = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/'
extract_path = '/kaggle/working/'

def extract_zip_file(zip_file_path, extract_to_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)
    print(f"Extracted {zip_file_path.split('/')[-1]} to {extract_to_path}")

try:
    extract_zip_file(os.path.join(data_base_path, 'train.csv.zip'), extract_path)
    extract_zip_file(os.path.join(data_base_path, 'test.csv.zip'), extract_path)
    extract_zip_file(os.path.join(data_base_path, 'sample_submission.csv.zip'), extract_path)

    train_df = pd.read_csv(os.path.join(extract_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(extract_path, 'test.csv'))
    sample_submission = pd.read_csv(os.path.join(extract_path, 'sample_submission.csv'))

    print("Data loaded successfully after extraction.")

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print(">>> Please ensure the dataset is added to your Kaggle Notebook and the path is correct. <<<")
    exit()

except Exception as e:
    print(f"An unexpected error occurred during data loading or extraction: {e}")
    exit()

if train_df is None or test_df is None or sample_submission is None:
    print("Data loading failed. Exiting script.")
else:
    CATEGORIES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    print("Applying text cleaning to comments...")
    train_df['comment_text'] = train_df['comment_text'].apply(clean_text)
    test_df['comment_text'] = test_df['comment_text'].apply(clean_text)
    print("Text cleaning complete.")

    print("Concatenating all comments for TF-IDF vectorization...")
    all_comments = pd.concat([train_df['comment_text'], test_df['comment_text']], axis=0)

    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        min_df=3,
        max_df=0.9,
        ngram_range=(1, 2),
        stop_words='english',
        max_features=50000
    )
    X_all = vectorizer.fit_transform(all_comments)

    X_train = X_all[:len(train_df)]
    X_test = X_all[len(train_df):]
    print(f"TF-IDF Vectorization complete. Number of features: {X_train.shape[1]}")

    del all_comments
    gc.collect()

    classifier = OneVsRestClassifier(
        LogisticRegression(solver='sag', n_jobs=-1, max_iter=1000, random_state=42)
    )

    print("Training models for each category...")
    predictions = pd.DataFrame({'id': test_df['id']})

    for category in CATEGORIES:
        print(f"  Training model for category: {category}...")
        y_train_category = train_df[category]
        classifier.fit(X_train, y_train_category)
        predictions[category] = classifier.predict_proba(X_test)[:, 1]
        print(f"  Finished training for {category}.")

    print("All models trained and predictions generated.")

    predictions.to_csv('submission.csv', index=False)
    print("\n--- Submission file 'submission.csv' created successfully! ---")
    print("You can now download this file and submit it to Kaggle.")
