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

catboost==1.2.8
cuml-cu12==25.2.1
geopandas==0.14.4
libcuml-cu12==25.2.1
lightgbm==4.6.0
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
xgboost==2.0.3

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

0.2114

# 6. Current score

0.18704

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.23804) has done: 'I slightly weaken the model to bring the Pearson score closer to the target by (1) limiting the TF‑IDF to unigrams only and (2) increasing regularisation in the LogisticRegression (C=0.1). These are minimal, safe tweaks that keep the overall pipeline intact while reducing performance toward the desired range.'
- What this solution (achieved 0.06684) has done: 'I slightly increase regularisation and limit the TF‑IDF vocabulary to reduce model capacity a bit more, which should lower the Pearson score toward the target (from 0.238 → ≈0.21). The changes keep the same pipeline, model type, and output format.'
- What this solution (achieved nan) has done: 'I raise the model’s capacity slightly to improve the Pearson correlation toward the target.  
- In the TF‑IDF step I increase the n‑gram range to include bigrams and double the feature limit, and I also add the `context` column so the model can use that information.  
- I set the logistic regression regularisation parameter `C` to 0.05, which is between the overly‑regularised (C=0.01) and the previously higher‑scoring (C=0.1) settings, aiming for a score nearer the target without over‑fitting.'
- What this solution (achieved 0.1377) has done: 'Implemented fixes to resolve column mismatches, switched to CPU‑based sklearn LogisticRegression (compatible with sparse TF‑IDF output), reduced TF‑IDF to unigrams and tightened regularisation to move the Pearson score toward the target, and ensured the pipeline correctly creates a `submission.csv` with the required columns.'
- What this solution (achieved 0.17516) has done: 'I raise the LogisticRegression regularisation parameter `C` from 0.05 to 0.08 so the model is slightly less regularised, which should increase the Pearson correlation from 0.1377 toward the target 0.2114 without overshooting the allowed tolerance. No other parts of the pipeline are changed, preserving the original feature engineering and submission logic.'
- What this solution (achieved 0.18704) has done: 'I raise the regularisation strength slightly by setting `C` to 0.09 in the LogisticRegression model (between the previous 0.08 and 0.1). This modest increase should lift the Pearson correlation toward the target 0.2114 without risking a large overshoot, while keeping the rest of the pipeline unchanged and still writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression



## === cell 1
train = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/train.csv")
test = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/test.csv")

y = train["score"]
le = LabelEncoder()
y_enc = le.fit_transform(y).astype("int64")  # classification labels

X = train.drop(columns=["id", "score"])



## === cell 2
vectorizer = TfidfVectorizer(
    ngram_range=(1, 1), max_features=5000, stop_words="english"
)

transformer = make_column_transformer(
    (vectorizer, "anchor"),
    (vectorizer, "target"),
    (vectorizer, "context"),
    remainder="drop",
)

X_tf = transformer.fit_transform(X)  # sparse matrix



## === cell 3
model = LogisticRegression(
    C=0.09, max_iter=1000, multi_class="multinomial", solver="lbfgs"
)
model.fit(X_tf, y_enc)



## === cell 4
test_features = test[["anchor", "target", "context"]]
X_test_tf = transformer.transform(test_features)



## === cell 5
pred_enc = model.predict(X_test_tf)
pred_scores = le.inverse_transform(pred_enc)



## === cell 6
sample = pd.read_csv(
    "../input/us-patent-phrase-to-phrase-matching/sample_submission.csv"
)
sample["score"] = pred_scores
sample.to_csv("submission.csv", index=False)
