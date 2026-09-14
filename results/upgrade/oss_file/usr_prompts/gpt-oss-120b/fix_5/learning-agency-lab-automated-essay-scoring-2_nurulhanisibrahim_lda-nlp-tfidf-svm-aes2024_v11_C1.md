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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

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
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.71541

# 6. Current score

0.59529

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.53241) has done: 'The changes fix the pipeline error by replacing the MaxAbsScaler + SVC combo (which cannot handle the COO sparse matrix) with a LinearSVC that works directly on the TF‑IDF sparse features. Minor clean‑up steps ensure the same preprocessing, proper train‑validation split, metric calculation, and creation of a valid submission.csv file.'
- What this solution (achieved 0.61306) has done: 'I keep the overall pipeline unchanged but improve the feature extraction and classifier regularisation to boost the quadratic weighted kappa. The TF‑IDF vectoriser now use bi‑grams, a lower `min_df`, L2 normalisation and IDF weighting, which captures more context. The LinearSVC classifier gets a higher `C` and `class_weight='balanced'` to better handle the score imbalance. These modest tweaks are expected to raise the validation kappa toward the target while preserving the original workflow.'
- What this solution (achieved 0.60614) has done: 'I make two small, targeted adjustments that are expected to raise the quadratic weighted kappa while keeping the overall pipeline unchanged:  
1. Expand the TF‑IDF n‑gram range to include tri‑grams and lower `min_df` to 2, allowing the model to capture more informative phrases and rarer words.  
2. Increase the LinearSVC regularisation parameter `C` to 10.0 to give the classifier slightly more flexibility on this larger feature set.  
These tweaks stay within the original architecture and should move the validation score closer to the target without introducing any new dependencies or altering the submission format.'
- What this solution (achieved 0.59529) has done: 'I slightly broaden the token pattern to keep words of length 2 or more (capturing more useful information) and increase the LinearSVC regularisation parameter C to 20.0, giving the classifier a bit more flexibility on the richer feature set while preserving the overall pipeline and submission format. These minimal tweaks are expected to lift the quadratic weighted kappa toward the target score.'

# 9. Code solution

## === cell 0
import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from sklearn.svm import LinearSVC



## === cell 1
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 2
train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
train_df = pd.read_csv(train_path)



## === cell 3
print(train_df.dtypes)
print(train_df.head(3))
print(train_df["score"].value_counts())



## === cell 4
train_df["full_text"] = train_df["full_text"].map(lambda x: re.sub(r"\n", " ", str(x)))
train_df["full_text"] = train_df["full_text"].map(
    lambda x: re.sub(r"[^\w\s]", "", str(x))
)
train_df["full_text"] = train_df["full_text"].str.lower()
train_df["full_text"] = train_df["full_text"].str.replace(r"\d+", "", regex=True)



## === cell 5
test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
test_df = pd.read_csv(test_path)



## === cell 6
test_df["full_text"] = test_df["full_text"].map(lambda x: re.sub(r"\n", " ", str(x)))
test_df["full_text"] = test_df["full_text"].map(
    lambda x: re.sub(r"[^\w\s]", "", str(x))
)
test_df["full_text"] = test_df["full_text"].str.lower()
test_df["full_text"] = test_df["full_text"].str.replace(r"\d+", "", regex=True)



## === cell 7
X = train_df[["full_text", "essay_id"]]
y = train_df[["score"]]  # keep as DataFrame for later indexing



## === cell 8
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=123, stratify=y["score"]
)



## === cell 9
vectorizer = TfidfVectorizer(
    stop_words="english",
    sublinear_tf=True,
    strip_accents="ascii",
    analyzer="word",
    token_pattern=r"\w{2,}",  # keep words of length >=2 to capture more tokens
    ngram_range=(1, 3),  # capture up to 3‑grams
    norm="l2",
    use_idf=True,
    smooth_idf=True,
    max_features=None,
    min_df=2,  # include less frequent tokens
)

X_train_vec = vectorizer.fit_transform(X_train["full_text"])
X_val_vec = vectorizer.transform(X_val["full_text"])



## === cell 10
clf = LinearSVC(
    C=20.0,  # increased flexibility on richer features
    loss="squared_hinge",
    dual=True,
    random_state=123,
    max_iter=3000,
    class_weight="balanced",
)
clf.fit(X_train_vec, y_train["score"])
y_pred_val = clf.predict(X_val_vec)



## === cell 11
print("Confusion Matrix:")
print(confusion_matrix(y_val["score"], y_pred_val))
print("\nClassification Report:")
print(classification_report(y_val["score"], y_pred_val, zero_division=1))
kappa = cohen_kappa_score(y_val["score"], y_pred_val, weights="quadratic")
print(f"Cohen's quadratic kappa: {kappa:.5f}")



## === cell 12
test_vec = vectorizer.transform(test_df["full_text"])
test_predictions = clf.predict(test_vec)



## === cell 13
sample_sub_path = (
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
submission = pd.read_csv(sample_sub_path)
submission["score"] = test_predictions
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
