# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf.message_factory import MessageFactory

    if not hasattr(MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression  # monotonic calibration

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

try:
    model = SentenceTransformer(model_name, device=device)
except Exception:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

BATCH_SIZE = 1024  # increases throughput without altering results




## === cell 1
base_path = os.path.join("..", "input", "us-patent-phrase-to-phrase-matching")
train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)




## === cell 2
def build_sentences(df, col_anchor):
    return [f"{a} [SEP] {c}" for a, c in zip(df[col_anchor], df["context"])]


anchor_train_sentences = build_sentences(train_df, "anchor")
target_train_sentences = build_sentences(train_df, "target")

anchor_train_emb = model.encode(
    anchor_train_sentences,
    batch_size=BATCH_SIZE,
    convert_to_numpy=True,
    device=device,
    show_progress_bar=False,
)
target_train_emb = model.encode(
    target_train_sentences,
    batch_size=BATCH_SIZE,
    convert_to_numpy=True,
    device=device,
    show_progress_bar=False,
)

context_train_emb = model.encode(
    train_df["context"].tolist(),
    batch_size=BATCH_SIZE,
    convert_to_numpy=True,
    device=device,
    show_progress_bar=False,
)


def cosine_numpy(a, b):
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return dot / (norm_a * norm_b + 1e-8)


def euclidean_numpy(a, b):
    return np.linalg.norm(a - b, axis=1)


cos_at = cosine_numpy(anchor_train_emb, target_train_emb)  # anchor‑target
cos_ac = cosine_numpy(anchor_train_emb, context_train_emb)  # anchor‑context
cos_tc = cosine_numpy(target_train_emb, context_train_emb)  # target‑context

dist_at = euclidean_numpy(anchor_train_emb, target_train_emb)
dist_ac = euclidean_numpy(anchor_train_emb, context_train_emb)
dist_tc = euclidean_numpy(target_train_emb, context_train_emb)

X_train = np.column_stack([cos_at, cos_ac, cos_tc, dist_at, dist_ac, dist_tc])
y_train = train_df["score"].values

poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train)

gbr_reg = GradientBoostingRegressor(
    n_estimators=2500,  # unchanged hyper‑parameters
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    random_state=42,
)
gbr_reg.fit(X_train_poly, y_train)

train_pred_raw = gbr_reg.predict(X_train_poly)

iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
iso_reg.fit(train_pred_raw, y_train)




## === cell 3
anchor_test_sentences = build_sentences(test_df, "anchor")
target_test_sentences = build_sentences(test_df, "target")

anchor_test_emb = model.encode(
    anchor_test_sentences,
    batch_size=BATCH_SIZE,
    convert_to_numpy=True,
    device=device,
    show_progress_bar=False,
)
target_test_emb = model.encode(
    target_test_sentences,
    batch_size=BATCH_SIZE,
    convert_to_numpy=True,
    device=device,
    show_progress_bar=False,
)

context_test_emb = model.encode(
    test_df["context"].tolist(),
    batch_size=BATCH_SIZE,
    convert_to_numpy=True,
    device=device,
    show_progress_bar=False,
)

cos_at_test = cosine_numpy(anchor_test_emb, target_test_emb)
cos_ac_test = cosine_numpy(anchor_test_emb, context_test_emb)
cos_tc_test = cosine_numpy(target_test_emb, context_test_emb)

dist_at_test = euclidean_numpy(anchor_test_emb, target_test_emb)
dist_ac_test = euclidean_numpy(anchor_test_emb, context_test_emb)
dist_tc_test = euclidean_numpy(target_test_emb, context_test_emb)

X_test = np.column_stack(
    [cos_at_test, cos_ac_test, cos_tc_test, dist_at_test, dist_ac_test, dist_tc_test]
)
X_test_poly = poly.transform(X_test)

preds_raw = gbr_reg.predict(X_test_poly)

preds = iso_reg.predict(preds_raw)
preds = np.clip(preds, 0.0, 1.0)




## === cell 4
submission = pd.DataFrame({"id": test_df["id"], "score": preds})
submission.to_csv("submission.csv", index=False)
