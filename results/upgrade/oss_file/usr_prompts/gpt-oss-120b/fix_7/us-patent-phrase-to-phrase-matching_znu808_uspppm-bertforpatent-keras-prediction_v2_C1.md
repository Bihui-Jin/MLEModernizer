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
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

0.8046283730655472

# 6. Current score

0.61556

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.59036) has done: 'The fix changes the Ridge regression solver to one compatible with the current SciPy version (`solver='lsqr'`) so the model can be trained and predictions generated without errors. This also ensures the `test_pred` variable is defined for the submission step, allowing a proper CSV file to be written.'
- What this solution (achieved 0.61556) has done: 'I keep the overall pipeline unchanged but improve the model’s ability to capture the data by (1) removing the artificial feature‑size caps so the TF‑IDF vectors can use all available tokens, (2) lowering the Ridge regularisation strength (`alpha=0.01`) to let the model fit more closely, and (3) after the validation step, refitting the vectorizers and Ridge model on the **full** training set before predicting the test set. These minimal tweaks stay within the existing linear‑model framework and should raise the Pearson correlation toward the target while still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from scipy.sparse import hstack

BASE_DIR = "/kaggle/input/us-patent-phrase-to-phrase-matching"
train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")

print("Files available:")
for root, _, files in os.walk(BASE_DIR):
    for f in files:
        print(os.path.join(root, f))



## === cell 1
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print(train_df.head())




## === cell 2
def combine_fields(row):
    return f"{row['anchor']} {row['target']} {row['context']}"


train_df["text"] = train_df.apply(combine_fields, axis=1)
test_df["text"] = test_df.apply(combine_fields, axis=1)

train_text, val_text, train_y, val_y = train_test_split(
    train_df["text"], train_df["score"], test_size=0.2, random_state=42
)



## === cell 3
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b",
    sublinear_tf=True,
)  # no max_features

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    lowercase=True,
    sublinear_tf=True,
)  # no max_features

X_train_word = word_vectorizer.fit_transform(train_text)
X_val_word = word_vectorizer.transform(val_text)
X_test_word = word_vectorizer.transform(test_df["text"])

X_train_char = char_vectorizer.fit_transform(train_text)
X_val_char = char_vectorizer.transform(val_text)
X_test_char = char_vectorizer.transform(test_df["text"])

X_train = hstack([X_train_word, X_train_char])
X_val = hstack([X_val_word, X_val_char])
X_test = hstack([X_test_word, X_test_char])

print("Combined feature shape (train split):", X_train.shape)

model = Ridge(alpha=0.01, solver="lsqr", random_state=42)
model.fit(X_train, train_y)

val_pred = model.predict(X_val)
pearson, _ = pearsonr(val_y, val_pred)
print(f"Validation Pearson correlation (80% train): {pearson:.6f}")



## === cell 4
full_word_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b",
    sublinear_tf=True,
)

full_char_vec = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    lowercase=True,
    sublinear_tf=True,
)

X_full_word = full_word_vec.fit_transform(train_df["text"])
X_full_char = full_char_vec.fit_transform(train_df["text"])
X_full = hstack([X_full_word, X_full_char])

full_model = Ridge(alpha=0.01, solver="lsqr", random_state=42)
full_model.fit(X_full, train_df["score"])

X_test_full_word = full_word_vec.transform(test_df["text"])
X_test_full_char = full_char_vec.transform(test_df["text"])
X_test_full = hstack([X_test_full_word, X_test_full_char])

test_pred = full_model.predict(X_test_full)
test_pred = np.clip(test_pred, 0.0, 1.0)



## === cell 5
submission = pd.read_csv(sample_sub_path)
submission["score"] = test_pred
submission = submission[["id", "score"]]
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print(submission.head())
