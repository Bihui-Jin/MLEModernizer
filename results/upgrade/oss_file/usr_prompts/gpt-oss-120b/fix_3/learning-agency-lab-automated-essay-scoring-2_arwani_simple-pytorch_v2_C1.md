# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
tqdm==4.67.1

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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

stop_words = stopwords.words("english")



## === cell 2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import cohen_kappa_score



## === cell 3
df = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
df.head()



## === cell 4
df["full_text"] = df["full_text"].apply(lambda x: x.lower())
df.head()




## === cell 5
def get_data(df, mode="train"):
    df = df.copy()
    res = []
    stop_words_set = set(stop_words)
    punc = set(punctuation)
    for data in tqdm(df.values.tolist()):
        essay_id = data[0]
        text = data[1]
        score = 0 if mode != "train" else data[2]
        txt = word_tokenize(text)
        lt = len(txt)  # token count
        sw = len([word for word in txt if word in stop_words_set])  # stop‑word count
        pct = len([word for word in txt if word in punc])  # punctuation count
        char_len = len(text)  # character length
        avg_word_len = char_len / lt if lt > 0 else 0  # average characters per token
        sw_ratio = sw / lt if lt > 0 else 0  # stop‑word ratio
        pct_ratio = pct / lt if lt > 0 else 0  # punctuation ratio
        res.append(
            [essay_id, lt, sw, pct, char_len, avg_word_len, sw_ratio, pct_ratio, score]
        )
    ndf = pd.DataFrame(
        res,
        columns=[
            "essay_id",
            "lt",
            "sw",
            "pct",
            "char_len",
            "avg_word_len",
            "sw_ratio",
            "pct_ratio",
            "score",
        ],
    )
    ndf = ndf.set_index("essay_id")
    return ndf




## === cell 6
data = get_data(df)
data.head()



## === cell 7
data.corr()



## === cell 8
feature_cols = [
    "lt",
    "sw",
    "pct",
    "char_len",
    "avg_word_len",
    "sw_ratio",
    "pct_ratio",
]
X = data[feature_cols]
y = data["score"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



## === cell 9
pipeline = Pipeline([("algo", GradientBoostingClassifier())])



## === cell 10
pipeline.get_params()



## === cell 11
parameter = {
    "algo__tol": [0.0001, 0.001, 0.01],
    "algo__n_estimators": [100, 200, 300],
    "algo__max_depth": [3, 4, 5],
}
model = GridSearchCV(pipeline, parameter, cv=3, n_jobs=-1, verbose=1)
model.fit(X_train, y_train)



## === cell 12
train_acc = model.score(X_train, y_train)
val_acc = model.score(X_val, y_val)

classes = np.arange(1, 7)  # possible scores
proba_val = model.predict_proba(X_val)
expected_val = np.dot(proba_val, classes)
pred_val = np.rint(expected_val).astype(int)
pred_val = np.clip(pred_val, 1, 6)
val_qwk = cohen_kappa_score(y_val, pred_val, weights="quadratic")

print(
    "Train acc:",
    train_acc,
    "Val acc:",
    val_acc,
    "Best params:",
    model.best_params_,
    "Best CV score (accuracy):",
    model.best_score_,
    "Validation QWK:",
    val_qwk,
)



## === cell 13
df_test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
df_test.head()



## === cell 14
data_test = get_data(df_test, "test")
data_test.head()



## === cell 15
X_test_pred = data_test[feature_cols]



## === cell 16
proba_test = model.predict_proba(X_test_pred)
expected_test = np.dot(proba_test, classes)
pred_test = np.rint(expected_test).astype(int)
pred_test = np.clip(pred_test, 1, 6)



## === cell 17
data_test["score"] = pred_test
data_test.drop(columns=feature_cols).to_csv("submission.csv", index=True)
