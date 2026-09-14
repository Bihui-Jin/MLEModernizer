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

0.75195

# 6. Current score

0.67472

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.67472) has done: 'I replace the label‑encoding step with the original integer scores and switch the model from a GradientBoosting **classifier** to a GradientBoosting **regressor** so the algorithm can better capture the ordinal nature of the 1‑6 scores. After predicting, I round the continuous outputs to the nearest integer (clipped to 1‑6) and evaluate with quadratic weighted kappa. This small change keeps the overall feature extraction and workflow intact while nudging the score upward toward the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



## === cell 1
train = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv",
    index_col="essay_id",
)
test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv",
    index_col="essay_id",
)



## === cell 2
train



## === cell 3
test



## === cell 4
y = train["score"].astype(int)



## === cell 5
y



## === cell 6
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))



## === cell 7
train["tokens"] = train["full_text"].apply(
    lambda x: [word for word in x.split() if word.isalpha()]
)
test["tokens"] = test["full_text"].apply(
    lambda x: [word for word in x.split() if word.isalpha()]
)



## === cell 8
stop_words



## === cell 9
train



## === cell 10
test




## === cell 11
class FeatureExtract:
    def __init__(self):
        pass

    def word_count(self, text):
        return len(text.split())

    def sen_count(self, text):
        return len(text.split("."))

    def ave_word_length(self, text):
        words = text.split()
        total_length = sum(len(word) for word in words)
        return total_length / len(words) if words else 0

    def total_stopwords(self, text):
        words = text.split()
        stopwords = [
            word
            for word in words
            if len(word) > 1 and word.isalpha() and word.lower() in stop_words
        ]
        return len(stopwords)

    def lexical_diversity(self, text):
        words = text.split()
        return len(set(words)) / len(words) if words else 0

    def sentiment(self, text):
        return sum(ord(c) for c in text) / len(text) if text else 0

    def extract_features(self, text):
        return {
            "word_count": self.word_count(text),
            "sen_count": self.sen_count(text),
            "ave_word_length": self.ave_word_length(text),
            "total_stopwords": self.total_stopwords(text),
            "lexical_diversity": self.lexical_diversity(text),
            "sentiment": self.sentiment(text),
        }




## === cell 12
def insert_features(df, text):
    extractor = FeatureExtract()
    for feature in [
        "word_count",
        "sen_count",
        "ave_word_length",
        "total_stopwords",
        "lexical_diversity",
        "sentiment",
    ]:
        df[feature] = df[text].apply(lambda x: getattr(extractor, feature)(x))


insert_features(train, "full_text")
insert_features(test, "full_text")



## === cell 13
train



## === cell 14
test



## === cell 15
features = [
    "word_count",
    "sen_count",
    "ave_word_length",
    "total_stopwords",
    "lexical_diversity",
    "sentiment",
]
X = train[features]
X_test = test[features]



## === cell 16
print("Value of X:")
print(X.head())



## === cell 17
print("\nValue of X_test:")
print(X_test.head())



## === cell 18
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 19
print("Value of X_train:")
print(X_train.head())



## === cell 20
print("\nValue of X_valid:")
print(X_valid.head())



## === cell 21
print("\nValue of y_train:")
print(y_train)



## === cell 22
print("\nValue of y_valid:")
print(y_valid)



## === cell 23
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42
)



## === cell 24
model.fit(X_train, y_train)



## === cell 25
y_pred_cont = model.predict(X_valid)



## === cell 26
y_pred = np.rint(y_pred_cont).astype(int)
y_pred = np.clip(y_pred, 1, 6)



## === cell 27
from sklearn.metrics import cohen_kappa_score, accuracy_score

kappa_score = cohen_kappa_score(y_valid, y_pred, weights="quadratic")
print("Cohen's Kappa Score:", kappa_score)



## === cell 28
accuracy = accuracy_score(y_valid, y_pred)
print("Accuracy:", accuracy)



## === cell 29
plt.figure(figsize=(8, 6))
plt.scatter(y_valid, y_pred, color="blue")
plt.plot(
    [min(y_valid), max(y_valid)],
    [min(y_valid), max(y_valid)],
    color="red",
    linestyle="--",
)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()



## === cell 30
plt.figure(figsize=(8, 6))
plt.hist(y_valid, bins=10, alpha=0.5, label="Actual", color="blue")
plt.hist(y_pred, bins=10, alpha=0.5, label="Predicted", color="orange")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Distribution of Actual and Predicted Scores")
plt.legend()
plt.grid(True)
plt.show()



## === cell 31
test_pred_cont = model.predict(X_test)



## === cell 32
test_pred = np.rint(test_pred_cont).astype(int)
test_pred = np.clip(test_pred, 1, 6)



## === cell 33
print("Predicted values on the test set (first 10):")
print(test_pred[:10])



## === cell 34
score_counts = pd.Series(test_pred).value_counts().sort_index()
plt.figure(figsize=(8, 6))
plt.pie(score_counts, labels=score_counts.index, autopct="%1.1f%%", startangle=140)
plt.title("Distribution of Predicted Scores")
plt.axis("equal")
plt.show()



## === cell 35
submission = pd.DataFrame({"essay_id": test.index, "score": test_pred})
submission.to_csv("submission.csv", index=False)
