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

0.75435

# 6. Current score

0.6729

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.6729) has done: 'I fix two issues that are directly harming your QWK: (1) your validation kappa currently compares encoded `y_valid` (0–5) to inverse-transformed predictions (1–6), which mis-measures performance and can lead you to trust the wrong configuration; and (2) your model is a classifier even though the metric is ordinal—without changing the core “GradientBoosting + handcrafted features” approach, switching to `GradientBoostingRegressor` and rounding/clipping predictions to 1–6 typically improves QWK noticeably. I also replace the extremely slow NLTK `words` dictionary spelling-error count with a lightweight proxy (token-level non-alphabetic/short/odd patterns) so feature extraction finishes within the 600s constraint while preserving the idea of a “spelling/noise” feature. Finally, I keep the same I/O paths and ensure a valid `submission.csv` is always written with `essay_id,score`.'

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
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(train["score"])
y_score = train[
    "score"
].to_numpy()  # <-- used for correct QWK evaluation and for regressor training



## === cell 5
y



## === cell 6
import nltk
from nltk.corpus import stopwords

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))



## === cell 7
train["tokens"] = train["full_text"].apply(
    lambda x: [word.lower() for word in x.split() if word.isalpha()]
)
test["tokens"] = test["full_text"].apply(
    lambda x: [word.lower() for word in x.split() if word.isalpha()]
)



## === cell 8
stop_words



## === cell 9
train



## === cell 10
test




## === cell 11
def count_errors(tokens):
    err = 0
    for t in tokens:
        if len(t) <= 2:
            err += 1
            continue
        if any(ch * 3 in t for ch in set(t)):
            err += 1
            continue
        if not t.isalpha():
            err += 1
    return err


train["count_spelling_errors"] = train["tokens"].apply(count_errors)
test["count_spelling_errors"] = test["tokens"].apply(count_errors)



## === cell 12
train



## === cell 13
test




## === cell 14
class FeatureExtract:
    def __init__(self):
        pass

    def word_count(self, text):
        return len(text.split())

    def sen_count(self, text):
        return len(text.split("."))

    def ave_word_length(self, text):
        words = text.split()
        total_length = 0
        for word in words:
            total_length += len(word)
        if len(words) == 0:
            return 0
        else:
            return total_length / len(words)

    def total_stopwords(self, text):
        words = text.split()
        stopwords_ = [
            word
            for word in words
            if len(word) > 1 and word.isalpha() and word.lower() in stop_words
        ]
        return len(stopwords_)

    def lexical_diversity(self, text):
        words = text.split()
        if len(words) == 0:
            return 0
        return len(set(words)) / len(words)

    def sentiment(self, text):
        if len(text) == 0:
            return 0.0
        return sum(ord(c) for c in text) / len(text)

    def extract_features(self, text):
        features = {
            "word_count": self.word_count(text),
            "sen_count": self.sen_count(text),
            "ave_word_length": self.ave_word_length(text),
            "total_stopwords": self.total_stopwords(text),
            "lexical_diversity": self.lexical_diversity(text),
            "sentiment": self.sentiment(text),
        }
        return features




## === cell 15
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



## === cell 16
train



## === cell 17
test



## === cell 18
features = [
    "count_spelling_errors",
    "word_count",
    "sen_count",
    "ave_word_length",
    "total_stopwords",
    "lexical_diversity",
    "sentiment",
]
X = train[features]
X_test = test[features]



## === cell 19
print("Value of X:")
print(X.head())



## === cell 20
print("\nValue of X_test:")
print(X_test.head())



## === cell 21
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y_score, test_size=0.2, random_state=42
)



## === cell 22
print("Value of X_train:")
print(X_train.head())



## === cell 23
print("\nValue of X_valid:")
print(X_valid.head())



## === cell 24
print("\nValue of y_train:")
print(y_train)



## === cell 25
print("\nValue of y_valid:")
print(y_valid)



## === cell 26
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(random_state=42)



## === cell 27
model.fit(X_train, y_train)



## === cell 28
y_pred_cont = model.predict(X_valid)
y_pred = np.clip(np.rint(y_pred_cont), 1, 6).astype(int)



## === cell 29
print("Predicted values on the validation set (post-processed to 1-6 integers):")
print(y_pred)



## === cell 30
from sklearn.metrics import cohen_kappa_score, accuracy_score

kappa_score = cohen_kappa_score(y_valid, y_pred, weights="quadratic")
print("Cohen's Kappa Score:", kappa_score)



## === cell 31
accuracy = accuracy_score(y_valid, y_pred)
print("Accuracy:", accuracy)



## === cell 32
plt.figure(figsize=(8, 6))
plt.scatter(y_valid, y_pred, color="blue", s=10, alpha=0.4)
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



## === cell 33
plt.figure(figsize=(8, 6))
plt.hist(
    y_valid, bins=np.arange(0.5, 6.6, 1.0), alpha=0.5, label="Actual", color="blue"
)
plt.hist(
    y_pred, bins=np.arange(0.5, 6.6, 1.0), alpha=0.5, label="Predicted", color="orange"
)
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Distribution of Actual and Predicted Scores")
plt.legend()
plt.grid(True)
plt.show()



## === cell 34
model.fit(X, y_score)



## === cell 35
predictions_cont = model.predict(X_test)
predictions = np.clip(np.rint(predictions_cont), 1, 6).astype(int)



## === cell 36
print("Predicted values on the test set (post-processed to 1-6 integers):")
print(predictions)



## === cell 37
predictions



## === cell 38
score_counts = pd.Series(predictions).value_counts().sort_index()
plt.figure(figsize=(8, 6))
plt.pie(
    score_counts,
    labels=score_counts.index.astype(str),
    autopct="%1.1f%%",
    startangle=140,
)
plt.title("Distribution of Predicted Scores")
plt.axis("equal")
plt.show()



## === cell 39
submission = pd.DataFrame({"essay_id": test.index, "score": predictions})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
