# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.13

# 3. Installed packages

geopandas==0.14.4
joblib==1.5.2
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
transformers==4.53.3
xgboost==2.0.3

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

0.7658101670359346

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import xgboost as xgb
import nltk
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import joblib

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("words")

stop_words = set(stopwords.words("english"))
english_words = set(w.lower() for w in words.words())



## === cell 1
train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)




## === cell 2
def remove_stop_words(text: str) -> str:
    tokens = word_tokenize(text)
    filtered = [t for t in tokens if t.lower() not in stop_words]
    return " ".join(filtered)


def spelling_score(text: str) -> int:
    tokens = word_tokenize(text)
    correct = [t for t in tokens if t.lower() in english_words]
    return len(correct)


train_df["clean_text"] = train_df["full_text"].apply(remove_stop_words)
test_df["clean_text"] = test_df["full_text"].apply(remove_stop_words)

train_df["spell_check"] = train_df["full_text"].apply(spelling_score)
test_df["spell_check"] = test_df["full_text"].apply(spelling_score)



## === cell 3
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(train_df["clean_text"])
X_test_tfidf = vectorizer.transform(test_df["clean_text"])

spell_train = train_df["spell_check"].values.reshape(-1, 1)
spell_test = test_df["spell_check"].values.reshape(-1, 1)

X_train = np.hstack([X_train_tfidf.toarray(), spell_train])
X_test = np.hstack([X_test_tfidf.toarray(), spell_test])

y_train = train_df["score"].astype(int)  # scores are 1‑6



## === cell 4
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

model = xgb.XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=6,
    use_label_encoder=False,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

model.fit(X_tr, y_tr)

val_pred = model.predict(X_val)
qwk = cohen_kappa_score(y_val, val_pred, weights="quadratic")
print(f"Validation QWK: {qwk:.5f}")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2605451438.py in <cell line: 0>()
     17 )
     18 
---> 19 model.fit(X_tr, y_tr)
     20 
     21 # Quick QWK check (optional)

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in fit(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)
   1469                 or not (classes == expected_classes).all()
   1470             ):
-> 1471                 raise ValueError(
   1472                     f"Invalid classes inferred from unique values of `y`.  "
   1473                     f"Expected: {expected_classes}, got {classes}"

ValueError: Invalid classes inferred from unique values of `y`.  Expected: [0 1 2 3 4 5], got [1 2 3 4 5 6]

## === cell 5
model.fit(X_train, y_train)

test_pred = model.predict(X_test)

submission = pd.DataFrame({"essay_id": test_df["essay_id"], "score": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2541599481.py in <cell line: 0>()
      1 # Train on the full training set
----> 2 model.fit(X_train, y_train)
      3 
      4 # Predict on test data
      5 test_pred = model.predict(X_test)

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in fit(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)
   1469                 or not (classes == expected_classes).all()
   1470             ):
-> 1471                 raise ValueError(
   1472                     f"Invalid classes inferred from unique values of `y`.  "
   1473                     f"Expected: {expected_classes}, got {classes}"

ValueError: Invalid classes inferred from unique values of `y`.  Expected: [0 1 2 3 4 5], got [1 2 3 4 5 6]
