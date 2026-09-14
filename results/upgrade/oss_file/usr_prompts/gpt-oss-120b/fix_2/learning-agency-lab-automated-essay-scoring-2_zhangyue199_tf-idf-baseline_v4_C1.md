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

0.49106

# 6. Current score

0.64581

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.64581) has done: 'I keep the overall pipeline unchanged but lower the model’s expressive power slightly so the validation kappa moves closer to the target (≈0.49). This is done by reducing the TF‑IDF n‑gram range from (1, 3) to (1, 2) and fixing the train‑test split random seed for reproducibility. These minimal tweaks preserve the core logic while expectedly decreasing the score into the acceptable band.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import cohen_kappa_score
import gc



## === cell 1
train = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)



## === cell 2
train_X, test_X, train_y, test_y = train_test_split(
    train[["full_text"]], train["score"], test_size=0.3, random_state=0
)

print(f"train shape: {train_X.shape}, test shape: {test_X.shape}")



## === cell 3
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    lowercase=False,
    sublinear_tf=True,
    analyzer="word",
    strip_accents="unicode",
    stop_words="english",
)

tf_train = vectorizer.fit_transform(train_X["full_text"])
tf_test = vectorizer.transform(test_X["full_text"])



## === cell 4
"""
loss{‘hinge’, ‘log_loss’, ‘modified_huber’, ‘squared_hinge’, ‘perceptron’, ‘squared_error’, ‘huber’, ‘epsilon_insensitive’, ‘squared_epsilon_insensitive’}, default=’hinge’
The loss function to be used.

‘hinge’ gives a linear SVM.

‘log_loss’ gives logistic regression, a probabilistic classifier.

‘modified_huber’ is another smooth loss that brings tolerance to outliers as well as probability estimates.

‘squared_hinge’ is like hinge but is quadratically penalized.

‘perceptron’ is the linear loss used by the perceptron algorithm.

The other losses, ‘squared_error’, ‘huber’, ‘epsilon_insensitive’ and ‘squared_epsilon_insensitive’ are designed for regression but can be useful in classification as well; see SGDRegressor for a description.

"""

from sklearn.calibration import CalibratedClassifierCV


def init_model():
    clf_sgd = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber")
    clf_sgd2 = CalibratedClassifierCV(
        SGDClassifier(max_iter=8000, tol=1e-4, loss="hinge"), cv=3, method="sigmoid"
    )
    clf_sgd3 = SGDClassifier(max_iter=8000, tol=2e-4, loss="log_loss")

    ensemble = VotingClassifier(
        estimators=[
            ("sgd", clf_sgd),  # 0.572
            ("sgd2", clf_sgd2),  # 0.576
            ("sgd3", clf_sgd3),  # 0.580
        ],
        voting="soft",
        weights=[0.4, 0.5, 0.1],
        n_jobs=-1,
    )
    return ensemble




## === cell 5
model = init_model()
model.fit(tf_train, train_y)
gc.collect()
preds = np.argmax(model.predict_proba(tf_test), axis=1) + 1

kappa_score = cohen_kappa_score(test_y, preds, weights="quadratic")
print("validation kappa score:", kappa_score)



## === cell 6
test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
sub = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)



## === cell 7
tf_train = vectorizer.fit_transform(train["full_text"])
tf_test = vectorizer.transform(test["full_text"])



## === cell 8
model = init_model()
model.fit(tf_train, train["score"])

print(gc.collect())
preds = np.argmax(model.predict_proba(tf_test), axis=1) + 1



## === cell 9
sub["score"] = preds
sub.to_csv("submission.csv", index=False)
display(sub.head())
