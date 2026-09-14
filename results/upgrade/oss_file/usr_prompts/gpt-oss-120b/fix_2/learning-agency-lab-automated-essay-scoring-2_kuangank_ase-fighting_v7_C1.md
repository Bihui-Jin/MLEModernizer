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

0.68085

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.68085) has done: 'The changes add a safe fallback for the missing `spellchecker` package, protect NLTK tokenization when required data isn’t present, and ensure the spelling‑error feature is created (set to 0) so all later columns exist. With these fixes the pipeline runs fully, trains a GradientBoosting model, predicts test scores, and writes a proper `submission.csv` file.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from spellchecker import SpellChecker
except ModuleNotFoundError:

    class SpellChecker:
        def __init__(self, *args, **kwargs):
            pass

        def unknown(self, words):
            return set()




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
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(train["score"])



## === cell 3
try:
    from nltk.tokenize import word_tokenize

    train["tokens"] = train["full_text"].apply(lambda x: word_tokenize(str(x)))
    test["tokens"] = test["full_text"].apply(lambda x: word_tokenize(str(x)))
except Exception:
    train["tokens"] = train["full_text"].astype(str).str.split()
    test["tokens"] = test["full_text"].astype(str).str.split()



## === cell 4
spell_checker = SpellChecker()


def count_errors(tokens):
    misspelled = spell_checker.unknown(token for token in tokens if token.isalpha())
    return len(misspelled)


train["count_spelling_errors"] = train["tokens"].apply(lambda x: count_errors(x))
test["count_spelling_errors"] = test["tokens"].apply(lambda x: count_errors(x))




## === cell 5
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

    def lexical_diversity(self, text):
        words = text.split()
        return len(set(words)) / len(words) if words else 0




## === cell 6
def insert_features(df, text):
    extractor = FeatureExtract()
    for feature in ["word_count", "sen_count", "ave_word_length", "lexical_diversity"]:
        df[feature] = df[text].apply(lambda x: getattr(extractor, feature)(x))


insert_features(train, "full_text")
insert_features(test, "full_text")



## === cell 7
features = [
    "count_spelling_errors",
    "word_count",
    "sen_count",
    "ave_word_length",
    "lexical_diversity",
]
X = train[features]
X_test = test[features]



## === cell 8
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 9
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()



## === cell 10
model.fit(X_train, y_train)



## === cell 11
y_pred = model.predict(X_valid)



## === cell 12
y_pred = le.inverse_transform(y_pred)



## === cell 13
from sklearn.metrics import cohen_kappa_score, accuracy_score

kappa_score = cohen_kappa_score(
    y_valid, le.inverse_transform(y_valid), weights="quadratic"
)
print("Cohen's Kappa Score (training split):", kappa_score)
accuracy = accuracy_score(le.inverse_transform(y_valid), y_pred)
print("Accuracy (training split):", accuracy)



## === cell 14
test_predictions = model.predict(X_test)



## === cell 15
test_predictions = le.inverse_transform(test_predictions)



## === cell 16
submission = pd.DataFrame({"essay_id": test.index, "score": test_predictions})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'.")
