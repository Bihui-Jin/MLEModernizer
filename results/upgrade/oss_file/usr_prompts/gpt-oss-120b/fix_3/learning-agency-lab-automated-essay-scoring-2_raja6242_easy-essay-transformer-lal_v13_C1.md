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
plotly==5.24.1
plotly-express==0.4.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
shap==0.44.1
shapely==2.1.2
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

0.62054

# 6. Current score

0.53881

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.53881) has done: 'I removed the non‑code narrative and stray markdown markers that caused syntax errors, eliminated the unnecessary pip‑install cell, and refined the TF‑IDF vectorizer (added bigrams and sublinear TF) along with a slightly broader alpha grid for the SGDRegressor. These minimal changes fix the runtime failures and should improve the validation Quadratic Weighted Kappa, moving the score toward the target while preserving the original modeling approach.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import make_scorer




## === cell 1
def quadratic_weighted_kappa(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    min_rating = np.min(y_true)
    max_rating = np.max(y_true)
    y_pred = np.clip(y_pred, min_rating, max_rating)

    hist_true = np.histogram(
        y_true, bins=max_rating - min_rating + 1, range=(min_rating, max_rating + 1)
    )[0]
    hist_pred = np.histogram(
        y_pred, bins=max_rating - min_rating + 1, range=(min_rating, max_rating + 1)
    )[0]

    O = np.zeros((max_rating - min_rating + 1, max_rating - min_rating + 1))
    for i in range(len(y_true)):
        O[int(y_true[i] - min_rating), int(y_pred[i] - min_rating)] += 1

    E = np.outer(hist_true, hist_pred) / len(y_true)
    W = np.zeros((max_rating - min_rating + 1, max_rating - min_rating + 1))
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = (i - j) ** 2.0 / (max_rating - min_rating) ** 2.0

    kappa = 1.0 - (np.sum(W * O) / np.sum(W * E))
    return kappa




## === cell 2
train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)



## === cell 3
plt.figure(figsize=(10, 6))
sns.histplot(train_df["score"], bins=20, kde=True, color="skyblue")
plt.title("Distribution of Scores in Training Data")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()



## === cell 4
vectorizer = TfidfVectorizer(
    stop_words="english", max_features=30000, ngram_range=(1, 2), sublinear_tf=True
)
X = vectorizer.fit_transform(train_df["full_text"])
y = train_df["score"]



## === cell 5
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



## === cell 6
model = SGDRegressor(max_iter=2000, tol=1e-3, random_state=42)
param_grid = {"alpha": [1e-5, 1e-4, 1e-3, 1e-2]}
scorer = make_scorer(quadratic_weighted_kappa, greater_is_better=True)

grid = GridSearchCV(
    estimator=model, param_grid=param_grid, scoring=scorer, cv=5, n_jobs=-1, verbose=0
)
grid.fit(X_train, y_train)



## === cell 7
val_pred = grid.predict(X_val).round().astype(int)
val_pred = np.clip(val_pred, y_val.min(), y_val.max())
val_kappa = quadratic_weighted_kappa(y_val, val_pred)
print(f"Quadratic Weighted Kappa on validation: {val_kappa:.4f}")



## === cell 8
X_test = vectorizer.transform(test_df["full_text"])
test_pred = grid.predict(X_test).round().astype(int)
test_pred = np.clip(test_pred, train_df["score"].min(), train_df["score"].max())
test_df["score"] = test_pred



## === cell 9
submission = pd.DataFrame({"essay_id": test_df["essay_id"], "score": test_df["score"]})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission saved to /kaggle/working/submission.csv")
