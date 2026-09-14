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
imbalanced-learn==0.13.0
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
ydata-profiling==4.17.0

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

0.54845

# 6. Current score

0.63733

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.63733) has done: 'The fix replaces the faulty emoji‑regex with a safe implementation that strips non‑ASCII characters, restoring the creation of the `text` column and allowing the train/validation split, model fitting, and final prediction to run. No core modeling logic is changed; the pipeline and evaluation remain identical, and a proper `submission.csv` is written.'

# 9. Code solution

## === cell 0
import nltk
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import re
import random
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
)
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    cohen_kappa_score,
)

nltk.download("wordnet", quiet=True)



## === cell 1
train = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
print(train.head())



## === cell 2
from ydata_profiling import ProfileReport

profile = ProfileReport(train, title="Pandas Profiling Report")
profile




## === cell 3
def remove_punctuations(data):
    punct_tag = re.compile(r"[^\w\s]")
    return punct_tag.sub(r"", data)


def remove_html(data):
    html_tag = re.compile(r"<.*?>")
    return html_tag.sub(r"", data)


def remove_url(data):
    url_clean = re.compile(r"https://\S+|www\.\S+")
    return url_clean.sub(r"", data)


def remove_emoji(data):
    """
    Remove emojis and other non‑ASCII symbols safely.
    """
    if isinstance(data, str):
        cleaned = data.encode("ascii", "ignore").decode("ascii")
        return cleaned.lower()
    return data


DESCRIPTION = train["full_text"]
DESCRIPTION = DESCRIPTION.apply(remove_punctuations)
DESCRIPTION = DESCRIPTION.apply(remove_html)
DESCRIPTION = DESCRIPTION.apply(remove_url)
DESCRIPTION = DESCRIPTION.apply(remove_emoji)
train["text"] = DESCRIPTION
train



## === cell 4
X = train["text"].astype(str)
y = train["score"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.33, random_state=0
)



## === cell 5
model = Pipeline(
    [
        (
            "Vectorizer",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                lowercase=True,
                max_features=150000,
            ),
        ),
        ("feature_selection", SelectFromModel(ExtraTreesClassifier(random_state=0))),
        ("clf", LinearSVC(class_weight="balanced")),
    ]
)

model.fit(X_train, y_train)

pred_train = model.predict(X_train)
f1_train = f1_score(y_train, pred_train, average="weighted")
kappa_train = cohen_kappa_score(y_train, pred_train)

print(f"F1 score (train): {f1_train:.4f}")
print(f"Cohen Kappa (train): {kappa_train:.4f}")



## === cell 6
pred_val = model.predict(X_test)

cm = confusion_matrix(y_test, pred_val, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()

f1_val = f1_score(y_test, pred_val, average="weighted")
kappa_val = cohen_kappa_score(y_test, pred_val)

print(f"F1 score (validation): {f1_val:.4f}")
print(f"Cohen Kappa (validation): {kappa_val:.4f}")



## === cell 7
test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)

DESCRIPTION = test["full_text"]
DESCRIPTION = DESCRIPTION.apply(remove_punctuations)
DESCRIPTION = DESCRIPTION.apply(remove_html)
DESCRIPTION = DESCRIPTION.apply(remove_url)
DESCRIPTION = DESCRIPTION.apply(remove_emoji)
test["text"] = DESCRIPTION

test_predictions = model.predict(test["text"])

submission = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
submission["score"] = test_predictions
submission.to_csv("submission.csv", index=False)

display(submission.head())
