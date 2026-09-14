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

0.75978

# 6. Current score

0.66948

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.67963) has done: 'I remove the unavailable `spellchecker` dependency and replace the spelling-error feature with a lightweight, self-contained proxy (ratio of non-alphabetic tokens), keeping the rest of the feature pipeline and GradientBoostingClassifier approach intact. I also fix the feature extraction class (it referenced undefined methods), ensure NLTK tokenization works in the Kaggle offline environment by downloading the required tokenizer resource, and correct the label handling so quadratic weighted kappa is computed on the same label space. Finally, I ensure predictions are mapped/clipped to the required integer score range (1–6) and write a valid `submission.csv` with the correct columns.'
- What this solution (achieved 0.66948) has done: 'Your current pipeline is leaving a lot of signal unused because it treats an ordered 1–6 target as a multi-class label and uses hard class predictions; for quadratic weighted kappa this is usually improved by using the model’s class probabilities to form an expected score and then rounding/clipping. To move your score up toward 0.75978 with minimal disruption, I keep the same features, the same GradientBoostingClassifier, and the same train/valid split, but change only the prediction post-processing to probability-weighted scoring (both for validation evaluation and test submission). I also make the sentence-count feature slightly less noisy by counting common sentence terminators instead of splitting on “.” (still the same feature concept, just a more correct implementation). These are small, metric-aligned changes that typically increase QWK without changing the core modeling approach.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize

for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)



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
train.head()



## === cell 3
test.head()



## === cell 4
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(train["score"].astype(int))



## === cell 5
train["tokens"] = train["full_text"].astype(str).apply(word_tokenize)
test["tokens"] = test["full_text"].astype(str).apply(word_tokenize)



## === cell 6
train[["full_text", "score", "tokens"]].head()



## === cell 7
test[["full_text", "tokens"]].head()




## === cell 8
def count_spelling_errors_proxy(tokens):
    tokens = tokens if isinstance(tokens, (list, tuple)) else []
    alpha = sum(1 for t in tokens if isinstance(t, str) and t.isalpha())
    non_alpha = sum(1 for t in tokens if isinstance(t, str) and (not t.isalpha()))
    total = alpha + non_alpha
    if total == 0:
        return 0
    return int(round(10.0 * non_alpha / total))


train["count_spelling_errors"] = train["tokens"].apply(count_spelling_errors_proxy)
test["count_spelling_errors"] = test["tokens"].apply(count_spelling_errors_proxy)



## === cell 9
train[["count_spelling_errors"]].head()



## === cell 10
test[["count_spelling_errors"]].head()




## === cell 11
class FeatureExtract:
    def __init__(self):
        pass

    def word_count(self, text):
        return len(str(text).split())

    def sen_count(self, text):
        s = str(text)
        cnt = s.count(".") + s.count("!") + s.count("?")
        if len(s.strip()) == 0:
            return 0
        return max(1, cnt)

    def ave_word_length(self, text):
        words = str(text).split()
        total_length = 0
        for word in words:
            total_length += len(word)
        if len(words) == 0:
            return 0.0
        else:
            return total_length / len(words)

    def lexical_diversity(self, text):
        words = str(text).split()
        if len(words) == 0:
            return 0.0
        return len(set(words)) / len(words)




## === cell 12
def insert_features(df, text_col):
    extractor = FeatureExtract()
    for feature in ["word_count", "sen_count", "ave_word_length", "lexical_diversity"]:
        df[feature] = df[text_col].apply(lambda x: getattr(extractor, feature)(x))


insert_features(train, "full_text")
insert_features(test, "full_text")



## === cell 13
train[
    [
        "word_count",
        "sen_count",
        "ave_word_length",
        "lexical_diversity",
        "count_spelling_errors",
    ]
].head()



## === cell 14
test[
    [
        "word_count",
        "sen_count",
        "ave_word_length",
        "lexical_diversity",
        "count_spelling_errors",
    ]
].head()



## === cell 15
features = [
    "count_spelling_errors",
    "word_count",
    "sen_count",
    "ave_word_length",
    "lexical_diversity",
]
X = train[features].copy()
X_test = test[features].copy()

X = X.fillna(0)
X_test = X_test.fillna(0)



## === cell 16
print("Value of X:")
print(X.head())



## === cell 17
print("\nValue of X_test:")
print(X_test.head())



## === cell 18
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



## === cell 19
print("Value of X_train:")
print(X_train.head())



## === cell 20
print("\nValue of X_valid:")
print(X_valid.head())



## === cell 21
print("\nValue of y_train:")
print(y_train[:10])



## === cell 22
print("\nValue of y_valid:")
print(y_valid[:10])



## === cell 23
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(random_state=42)



## === cell 24
model.fit(X_train, y_train)



## === cell 25
from sklearn.metrics import cohen_kappa_score, accuracy_score

classes_enc = model.classes_
classes_score = le.inverse_transform(classes_enc).astype(float)

proba_valid = model.predict_proba(X_valid)
expected_valid_score = (proba_valid * classes_score.reshape(1, -1)).sum(axis=1)

y_pred_score_soft = np.rint(expected_valid_score).astype(int)
y_pred_score_soft = np.clip(y_pred_score_soft, 1, 6)
y_pred_enc_soft = le.transform(y_pred_score_soft)

print("Predicted encoded values on the validation set (soft/expected->rounded):")
print(y_pred_enc_soft[:20])

kappa_score = cohen_kappa_score(y_valid, y_pred_enc_soft, weights="quadratic")
print("Cohen's Kappa Score (encoded labels, soft/expected->rounded):", kappa_score)

accuracy = accuracy_score(y_valid, y_pred_enc_soft)
print("Accuracy (encoded labels, soft/expected->rounded):", accuracy)



## === cell 26
y_valid_score = le.inverse_transform(y_valid)
y_pred_score = y_pred_score_soft

plt.figure(figsize=(8, 6))
plt.scatter(y_valid_score, y_pred_score, color="blue", s=8, alpha=0.3)
plt.plot(
    [min(y_valid_score), max(y_valid_score)],
    [min(y_valid_score), max(y_valid_score)],
    color="red",
    linestyle="--",
)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()



## === cell 27
plt.figure(figsize=(8, 6))
plt.hist(y_valid_score, bins=6, alpha=0.5, label="Actual", color="blue")
plt.hist(y_pred_score, bins=6, alpha=0.5, label="Predicted", color="orange")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Distribution of Actual and Predicted Scores")
plt.legend()
plt.grid(True)
plt.show()



## === cell 28
proba_test = model.predict_proba(X_test)
expected_test_score = (proba_test * classes_score.reshape(1, -1)).sum(axis=1)

pred_score = np.rint(expected_test_score).astype(int)
pred_score = np.clip(pred_score, 1, 6)

print("Predicted values on the test set (first 20):")
print(pred_score[:20])



## === cell 29
score_counts = pd.Series(pred_score).value_counts().sort_index()
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



## === cell 30
submission = pd.DataFrame({"essay_id": test.index, "score": pred_score})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
