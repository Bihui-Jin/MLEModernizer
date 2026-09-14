# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.7716525329500983

# 6. Current score

0.61385

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.56297) has done: 'I replace the failing HuggingFace imports with a lightweight SentenceTransformer model, compute cosine similarity between anchor and target sentences (scaled to 0‑1), and generate a proper `submission.csv`. This removes the protobuf error, fixes undefined variables, and ensures the script runs end‑to‑end while preserving the original idea of using a pretrained language model for similarity scoring.'
- What this solution (achieved 0.56297) has done: 'The fix removes the unused `util` import that caused the protobuf error, adds loading of the training data, and introduces a tiny linear regression calibration that maps raw cosine similarities to the true scores using the training set. This calibration usually raises the Pearson correlation without changing the core embedding‑based approach. Finally, the script writes a proper `submission.csv` with the required columns.'
- What this solution (achieved 0.59219) has done: 'The fix adds a protobuf‑compatibility setting before importing SentenceTransformer to stop the `MessageFactory` error, and improves the calibration step by using a quadratic feature transformation (PolynomialFeatures degree 2) before linear regression, which better maps raw cosine similarities to the 0‑1 score range, moving the Pearson correlation toward the target while keeping the original embedding‑based approach unchanged.'
- What this solution (achieved 0.56276) has done: 'I replace the failing SentenceTransformer import with a pure‑transformers embedding routine (avoiding the protobuf incompatibility) and enhance the calibration step by using a degree‑3 polynomial with Ridge regression, which normally captures the non‑linear relationship between cosine similarity and the 0‑1 score better and should raise the Pearson correlation toward the target. The rest of the workflow – loading data, computing cosine similarities, clipping predictions, and writing the CSV – stays unchanged.'
- What this solution (achieved 0.59224) has done: 'The fix replaces the failing `transformers` imports with `SentenceTransformer`, which avoids the protobuf error, and adds a simple context‑similarity feature to the calibration model. By using both anchor‑target and context‑context cosine similarities as inputs to a polynomial‑ridge regression, the Pearson correlation is expected to improve, moving the score closer to the target while keeping the original embedding‑based approach unchanged.'
- What this solution (achieved 0.59474) has done: 'I replace the failing SentenceTransformer import with a plain transformers based encoder, add useful context‑anchor and context‑target similarity features, and use a degree‑3 polynomial ridge regression to better map raw cosine similarities to the target scores. These changes fix the protobuf error, enrich the calibration features, and are expected to raise the Pearson correlation toward the target while keeping the original embedding‑based workflow.'
- What this solution (achieved 0.59474) has done: 'I replace the failing transformers imports with SentenceTransformer to avoid the protobuf error, and adjust the embedding function to use the SentenceTransformer `encode` method (which directly returns L2‑normalized tensors on the correct device). This fix eliminates the import crash while keeping the original calibration logic unchanged, so the model’s predictions and the final submission file are produced correctly.'
- What this solution (achieved 0.59721) has done: 'The fix removes the protobuf import error by replacing `SentenceTransformer` with a lightweight HuggingFace encoder built from `AutoTokenizer` and `AutoModel`. The encoder returns L2‑normalized mean‑pooled embeddings, preserving the original similarity‑based approach. The calibration now drops the constant context‑self similarity column, uses a 4‑degree polynomial feature expansion and a ridge regression without regularisation (α=0) to better fit the training scores, which should raise the Pearson correlation toward the target. The script finally writes a correct `submission.csv` with the required columns.'
- What this solution (achieved 0.59721) has done: 'The script failed because loading the HuggingFace model caused a protobuf incompatibility error. I switched to `SentenceTransformer`, which avoids that issue and provides L2‑normalized embeddings directly. The encoding function now returns NumPy arrays, and the cosine‑similarity calculations are updated accordingly, keeping the original calibration pipeline intact while fixing the runtime error and enabling a valid `submission.csv` to be created.'
- What this solution (achieved 0.61054) has done: 'The fix removes the unused `transformers` imports that trigger a protobuf incompatibility error, keeping only the necessary libraries. The embedding model is upgraded to a higher‑capacity “paraphrase‑mpnet‑base‑v2” model and the polynomial feature transformer now includes a bias term, both of which should raise the Pearson correlation toward the target while leaving the overall workflow unchanged. The script now runs end‑to‑end and writes a correct `submission.csv`.'
- What this solution (achieved 0.61385) has done: 'Implemented a robust fallback embedder to avoid protobuf crashes, added a GradientBoostingRegressor for stronger calibration (replacing the simple Ridge), and kept all existing feature engineering steps. The new model better captures non‑linear relationships, aiming to raise the Pearson score toward the target while preserving the original workflow and producing a correct `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor

try:
    from sentence_transformers import SentenceTransformer

    use_sentence_transformer = True
except Exception:
    use_sentence_transformer = False
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F

    class SimpleEmbedder:
        """Mean‑pooled, L2‑normalized embeddings using a HuggingFace AutoModel."""

        def __init__(self, model_name: str, device: str):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.device = device

        def encode(
            self,
            texts,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ):
            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    last_hidden = outputs.last_hidden_state  # (B, L, D)
                    mask = inputs["attention_mask"].unsqueeze(-1)  # (B, L, 1)
                    pooled = (last_hidden * mask).sum(1) / mask.sum(1)
                if normalize_embeddings:
                    pooled = F.normalize(pooled, p=2, dim=1)
                emb = pooled.cpu().numpy()
                all_embs.append(emb)
            return np.vstack(all_embs)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "../input/us-patent-phrase-to-phrase-matching/train.csv"
test_path = "../input/us-patent-phrase-to-phrase-matching/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)



## === cell 2
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "sentence-transformers/paraphrase-mpnet-base-v2"

if use_sentence_transformer:
    embedder = SentenceTransformer(model_name, device=device)
else:
    embedder = SimpleEmbedder(model_name, device)


def _batch_encode(texts, batch_size=64):
    """Encode a list of texts into L2‑normalized NumPy embeddings."""
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.array(embeddings)




## === cell 3
train_anchor_emb = _batch_encode(train_df["anchor"].astype(str).tolist())
train_target_emb = _batch_encode(train_df["target"].astype(str).tolist())
train_context_emb = _batch_encode(train_df["context"].astype(str).tolist())

train_cos = (train_anchor_emb * train_target_emb).sum(axis=1)
train_context_anchor_cos = (train_context_emb * train_anchor_emb).sum(axis=1)
train_context_target_cos = (train_context_emb * train_target_emb).sum(axis=1)

X_train = np.column_stack(
    [
        train_cos,
        train_context_anchor_cos,
        train_context_target_cos,
    ]
)

poly = PolynomialFeatures(degree=4, include_bias=True)
X_train_poly = poly.fit_transform(X_train)

train_scores = train_df["score"].values

reg = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)
reg.fit(X_train_poly, train_scores)



## === cell 4
test_anchor_emb = _batch_encode(test_df["anchor"].astype(str).tolist())
test_target_emb = _batch_encode(test_df["target"].astype(str).tolist())
test_context_emb = _batch_encode(test_df["context"].astype(str).tolist())

test_cos = (test_anchor_emb * test_target_emb).sum(axis=1)
test_context_anchor_cos = (test_context_emb * test_anchor_emb).sum(axis=1)
test_context_target_cos = (test_context_emb * test_target_emb).sum(axis=1)

X_test = np.column_stack(
    [
        test_cos,
        test_context_anchor_cos,
        test_context_target_cos,
    ]
)
X_test_poly = poly.transform(X_test)
preds = reg.predict(X_test_poly)

preds = np.clip(preds, 0.0, 1.0)



## === cell 5
submission = pd.DataFrame({"id": test_df["id"], "score": preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
