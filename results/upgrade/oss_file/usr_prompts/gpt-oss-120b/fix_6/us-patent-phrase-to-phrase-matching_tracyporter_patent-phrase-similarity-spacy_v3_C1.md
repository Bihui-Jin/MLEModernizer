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
seaborn==0.12.2
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5

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

0.444

# 6. Current score

0.39053

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.37713) has done: 'The fix replaces the deprecated `DataFrame.append` with `pd.concat`, removes the failing spaCy model loading, and implements a lightweight TF‑IDF cosine similarity to generate scores. This restores the pipeline, creates a proper `submission.csv` with correct columns, and provides a reasonable baseline that should move the Pearson score toward the target.'
- What this solution (achieved 0.38701) has done: 'I added all missing imports, corrected the data loading, fixed the TF‑IDF similarity computation, mapped cosine values from \[-1, 1\] to \[0, 1\] to match the label range, and ensured the submission file is written with the correct columns and “.csv” extension. These changes resolve the NameError crashes and modestly improve the Pearson score toward the target while preserving the original model logic.'
- What this solution (achieved 0.39053) has done: 'I add a lightweight second similarity signal that compares the raw anchor and target texts (without context) using the same TF‑IDF space, then combine it with the existing anchor‑vs‑target‑with‑context similarity via a modest weighted average (70 % original, 30 % raw). This keeps the original TF‑IDF‑cosine logic while giving the model a bit more information, which should raise the Pearson correlation toward the 0.444 target without changing the overall pipeline.'
- What this solution (achieved 0.39053) has done: 'I add a tiny calibration step: after computing the same TF‑IDF‑based similarity for the training rows, I fit a simple linear regression that maps the combined similarity to the true scores. Then I apply this model to the test combined similarity, clipping the output to [0, 1]. This keeps the original TF‑IDF logic unchanged while adjusting predictions toward the target Pearson score. I also import the needed `LinearRegression` class.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression  # added for calibration

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv"
test_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv"
sample_sub_path = (
    "/kaggle/input/us-patent-phrase-to-phrase-matching/sample_submission.csv"
)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
submission = pd.read_csv(sample_sub_path)




## === cell 1
print(train.head())
print(test.head())
print(submission.head())




## === cell 2
sns.histplot(train["score"], kde=False, bins=5)
plt.title("Score Distribution")
plt.show()

plt.boxplot(train["score"])
plt.title("Score Boxplot")
plt.show()




## === cell 3
train_anchor_ctx = train["anchor"] + " " + train["context"]
train_target_ctx = train["target"] + " " + train["context"]
test_anchor_ctx = test["anchor"] + " " + test["context"]
test_target_ctx = test["target"] + " " + test["context"]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=1,
    stop_words="english",
).fit(pd.concat([train_anchor_ctx, train_target_ctx, test_anchor_ctx, test_target_ctx]))

anchor_vec_ctx_test = vectorizer.transform(test_anchor_ctx)
target_vec_ctx_test = vectorizer.transform(test_target_ctx)
cosine_vals_ctx_test = cosine_similarity(
    anchor_vec_ctx_test, target_vec_ctx_test
).diagonal()
similarity_ctx_test = (cosine_vals_ctx_test + 1.0) / 2.0  # map from [-1,1] to [0,1]

anchor_vec_ctx_train = vectorizer.transform(train_anchor_ctx)
target_vec_ctx_train = vectorizer.transform(train_target_ctx)
cosine_vals_ctx_train = cosine_similarity(
    anchor_vec_ctx_train, target_vec_ctx_train
).diagonal()
similarity_ctx_train = (cosine_vals_ctx_train + 1.0) / 2.0

anchor_vec_test = vectorizer.transform(test["anchor"])
target_vec_test = vectorizer.transform(test["target"])
cosine_vals_plain_test = cosine_similarity(anchor_vec_test, target_vec_test).diagonal()
similarity_plain_test = (cosine_vals_plain_test + 1.0) / 2.0

anchor_vec_train = vectorizer.transform(train["anchor"])
target_vec_train = vectorizer.transform(train["target"])
cosine_vals_plain_train = cosine_similarity(
    anchor_vec_train, target_vec_train
).diagonal()
similarity_plain_train = (cosine_vals_plain_train + 1.0) / 2.0

combined_similarity_test = 0.7 * similarity_ctx_test + 0.3 * similarity_plain_test
combined_similarity_train = 0.7 * similarity_ctx_train + 0.3 * similarity_plain_train

reg = LinearRegression()
reg.fit(combined_similarity_train.reshape(-1, 1), train["score"])

predicted_test = reg.predict(combined_similarity_test.reshape(-1, 1))
predicted_test = np.clip(predicted_test, 0.0, 1.0)




## === cell 4
submission["score"] = predicted_test
output_path = "/kaggle/working/submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")
print(submission.head())
