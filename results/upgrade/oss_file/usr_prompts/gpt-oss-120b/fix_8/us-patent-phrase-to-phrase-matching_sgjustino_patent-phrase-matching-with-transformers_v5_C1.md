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

3.12

# 3. Installed packages

datasets==4.4.1
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1
transformers==4.53.3
vega-datasets==0.9.0

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

0.7805410791891036

# 6. Current score

0.62305

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.43069) has done: 'I fix the runtime error by configuring the Ridge regression to use a solver that works with the current SciPy version (e.g., `solver='lsqr'`). This prevents the internal call to `scipy.sparse.linalg.cg` that fails due to a changed signature. No other logic is altered, so the model, feature engineering, and submission format remain unchanged, allowing the script to run end‑to‑end and produce a valid `submission.csv`.'
- What this solution (achieved 0.59003) has done: 'I lower the Ridge regularization strength (alpha = 0.1) and add a character‑level TF‑IDF alongside the word‑level TF‑IDF using a FeatureUnion, which usually captures extra similarity cues and improves Pearson correlation. I also drop the rounding step – keeping the raw predictions (only clipping to [0, 1]) preserves more variance, which is beneficial for the Pearson metric while still respecting the required score range. These minimal changes keep the overall pipeline structure intact.'
- What this solution (achieved 0.30112) has done: 'I slightly strengthen the text features (broader n‑gram ranges and a few more TF‑IDF terms) and reduce the Ridge regularisation (α = 0.01). I also insert a `StandardScaler(with_mean=False)` right after the combined TF‑IDF so the sparse features are scaled before regression, which often improves Pearson correlation without changing the overall pipeline logic. These minimal tweaks are expected to move the validation Pearson closer to the target score.'
- What this solution (achieved 0.60471) has done: 'I raise the Ridge regularisation strength back to `alpha=0.1` (the setting that previously yielded a much higher Pearson) and remove the unnecessary `StandardScaler` on the sparse TF‑IDF matrix, which can hurt linear models. These tiny adjustments keep the original pipeline structure while expectedly moving the validation Pearson much closer to the target score.'
- What this solution (achieved 0.62809) has done: 'I slightly expand the TF‑IDF vocabularies (set `min_df=1` and increase `max_features`) to capture more useful n‑grams and lower the Ridge regularisation (`alpha=0.05`). These small changes keep the original pipeline structure while giving the linear model a richer representation, which should raise the Pearson correlation toward the target score. The rest of the code—including the train/validation split, clipping, and submission writing—remains unchanged.'
- What this solution (achieved 0.62305) has done: 'We slightly expand the TF‑IDF vocabularies and reduce the Ridge regularisation (alpha = 0.03). These minimal adjustments keep the original pipeline structure while giving the linear model a richer representation and a bit more flexibility, which is expected to raise the Pearson correlation toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline, FeatureUnion
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)



## === cell 1
train_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv"
test_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)



## === cell 2
sep = " "
train_df["inputs"] = (
    train_df["context"] + sep + train_df["anchor"] + sep + train_df["target"]
)
test_df["inputs"] = (
    test_df["context"] + sep + test_df["anchor"] + sep + test_df["target"]
)



## === cell 3
train_split, val_split = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df["score"]
)



## === cell 4
word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 3), max_features=150000, min_df=1, sublinear_tf=True
)
char_vectorizer = TfidfVectorizer(
    analyzer="char", ngram_range=(2, 5), max_features=80000, min_df=1, sublinear_tf=True
)

combined_vectorizer = FeatureUnion(
    [("word", word_vectorizer), ("char", char_vectorizer)]
)

model = Ridge(alpha=0.03, solver="lsqr", random_state=42)

pipeline = make_pipeline(combined_vectorizer, model)



## === cell 5
pipeline.fit(train_split["inputs"], train_split["score"])



## === cell 6
val_pred = pipeline.predict(val_split["inputs"])
pearson = np.corrcoef(val_pred, val_split["score"])[0, 1]
print(f"Validation Pearson: {pearson:.6f}")



## === cell 7
pipeline.fit(train_df["inputs"], train_df["score"])



## === cell 8
test_pred = pipeline.predict(test_df["inputs"])



## === cell 9
test_pred = np.clip(test_pred, 0, 1)



## === cell 10
submission = pd.DataFrame({"id": test_df["id"], "score": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
