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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
transformers==4.53.3

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

0.7674398724521859

# 6. Current score

0.62996

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.46301) has done: 'Implemented a full end‑to‑end pipeline that avoids the broken local model loading, uses a publicly available Sentence‑Transformer to embed the texts, trains a lightweight regression model on the training data, and writes a correctly formatted `submission.csv`. The script now:
1. Loads the train and test CSV files.
2. Builds combined text strings (including context, anchor, target) and encodes them with `sentence‑transformers/all-MiniLM-L6-v2`.
3. Trains a `Ridge` regression model (with modest regularisation) on the embeddings.
4. Predicts similarity scores for the test set, clipping to the required \[0, 1\] range.
5. Saves the predictions with the proper `id,score` columns to `submission.csv`.'
- What this solution (achieved 0.63649) has done: 'The fix adds a protobuf compatibility setting before any imports to prevent the `MessageFactory` error, switches to a stronger pre‑trained Sentence‑Transformer model, and reduces the Ridge regularisation (alpha = 0.1) to improve correlation while keeping the overall pipeline unchanged. The script now runs end‑to‑end and writes a proper `submission.csv`.'
- What this solution (achieved 0.6363) has done: 'I add a small monkey‑patch for the protobuf MessageFactory before importing SentenceTransformer to stop the ImportError, and I set the Ridge regression regularisation to 0 (no L2) which usually raises the Pearson correlation for these embeddings. All other logic stays the same, and the script still write a proper `submission.csv`.'
- What this solution (achieved 0.44455) has done: 'I fixed the protobuf import issue by guarding the monkey‑patch with a safe try/except, switched to a modest regularisation (α = 0.1) to avoid over‑fitting, and enriched the feature set: instead of a single combined text embedding we now concatenate separate embeddings of *anchor*, *target* and *context* sentences. This keeps the original pipeline logic but provides more expressive inputs, which should raise the Pearson correlation toward the target score while still producing a correctly formatted `submission.csv`.'
- What this solution (achieved 0.68307) has done: 'I fixed the protobuf import error by correctly aliasing the missing `GetPrototype` method, switched to a plain `LinearRegression` (no L2 regularisation) and added richer features (absolute difference and element‑wise product between anchor and target embeddings) to give the model more expressive power, which should raise the Pearson correlation toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.68137) has done: 'I patch the protobuf monkey‑patch so it correctly adds a `GetPrototype` method to the `MessageFactory` class, and I enhance the feature builder by adding a cosine‑similarity feature between the anchor and target embeddings. This small, targeted change keeps the core pipeline unchanged while giving the linear model a more expressive input, which should raise the validation Pearson correlation toward the target score.'
- What this solution (achieved 0.68358) has done: 'I add a few inexpensive linear features (additional cosine similarities and Euclidean distance) and standard‑scale all features before training. The model stays a simple linear regressor (now Ridge with low regularisation) so the core logic is unchanged, but the richer feature set and scaling should lift the Pearson correlation toward the target.'
- What this solution (achieved 0.6792) has done: 'I add a few inexpensive but potentially useful linear features (squared differences and extra Euclidean distances) to give the regression more expressive power, and I reduce the Ridge regularisation (alpha = 0.001) so the model can better fit these richer features. These changes keep the overall pipeline identical while aiming to raise the Pearson correlation toward the target score.'
- What this solution (achieved 0.62996) has done: 'I add a PCA dimensionality‑reduction step after scaling the features. Reducing the very high‑dimensional embedding‑based vectors (≈ 2300 features) to a more manageable size often improves linear regression stability and Pearson correlation, moving the score closer to the target while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf.message_factory import MessageFactory

    if not hasattr(MessageFactory, "GetPrototype"):

        def _get_prototype(self, *args, **kwargs):
            return self.GetMessageClass(*args, **kwargs)

        MessageFactory.GetPrototype = _get_prototype
except Exception:
    pass  # If the patch fails, the import below will raise visibly

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # new import for dimensionality reduction
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer

TRAIN_PATH = "../input/us-patent-phrase-to-phrase-matching/train.csv"
TEST_PATH = "../input/us-patent-phrase-to-phrase-matching/test.csv"
SUBMISSION_PATH = "submission.csv"



## === cell 1
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_anchor = train_df["anchor"].tolist()
train_target = train_df["target"].tolist()
train_context = train_df["context"].tolist()

test_anchor = test_df["anchor"].tolist()
test_target = test_df["target"].tolist()
test_context = test_df["context"].tolist()



## === cell 2
embedder = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")

train_anchor_emb = embedder.encode(
    train_anchor, batch_size=64, show_progress_bar=False, convert_to_numpy=True
)
train_target_emb = embedder.encode(
    train_target, batch_size=64, show_progress_bar=False, convert_to_numpy=True
)
train_context_emb = embedder.encode(
    train_context, batch_size=64, show_progress_bar=False, convert_to_numpy=True
)

test_anchor_emb = embedder.encode(
    test_anchor, batch_size=64, show_progress_bar=False, convert_to_numpy=True
)
test_target_emb = embedder.encode(
    test_target, batch_size=64, show_progress_bar=False, convert_to_numpy=True
)
test_context_emb = embedder.encode(
    test_context, batch_size=64, show_progress_bar=False, convert_to_numpy=True
)


def build_features(a, t, c):
    """
    Construct a richer linear feature vector from the three embeddings.
    Added components:
      * raw difference (a‑t) and its squared values
      * absolute difference
      * element‑wise product
      * cosine similarities for each pair (a‑t, a‑c, t‑c)
      * Euclidean distances for each pair (a‑t, a‑c, t‑c)
    """
    diff = a - t
    absdiff = np.abs(diff)
    sqdiff = diff**2
    prod = a * t

    cos_at = np.dot(a, t) / (np.linalg.norm(a) * np.linalg.norm(t) + 1e-8)
    cos_ac = np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c) + 1e-8)
    cos_tc = np.dot(t, c) / (np.linalg.norm(t) * np.linalg.norm(c) + 1e-8)

    euclid_at = np.linalg.norm(diff)
    euclid_ac = np.linalg.norm(a - c)
    euclid_tc = np.linalg.norm(t - c)

    return np.hstack(
        [
            a,
            t,
            c,  # raw embeddings
            absdiff,
            prod,
            sqdiff,  # interaction terms
            [
                cos_at,
                cos_ac,
                cos_tc,
                euclid_at,
                euclid_ac,
                euclid_tc,
            ],  # similarity / distance features
        ]
    )


train_embeddings = np.vstack(
    [
        build_features(a, t, c)
        for a, t, c in zip(train_anchor_emb, train_target_emb, train_context_emb)
    ]
)
test_embeddings = np.vstack(
    [
        build_features(a, t, c)
        for a, t, c in zip(test_anchor_emb, test_target_emb, test_context_emb)
    ]
)



## === cell 3
X_train, X_val, y_train, y_val = train_test_split(
    train_embeddings, train_df["score"].values, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

pca = PCA(n_components=100, random_state=42)  # 100 components capture most variance
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)

model = Ridge(alpha=0.001, fit_intercept=True, random_state=42)
model.fit(X_train, y_train)

val_preds = model.predict(X_val)
val_preds = np.clip(val_preds, 0, 1)
pearson = pearsonr(y_val, val_preds)[0]
print(f"Validation Pearson correlation: {pearson:.5f}")



## === cell 4
test_embeddings_scaled = scaler.transform(test_embeddings)
test_embeddings_pca = pca.transform(test_embeddings_scaled)

test_preds = model.predict(test_embeddings_pca)
test_preds = np.clip(test_preds, 0, 1)



## === cell 5
submission = pd.DataFrame({"id": test_df["id"], "score": test_preds})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}")
