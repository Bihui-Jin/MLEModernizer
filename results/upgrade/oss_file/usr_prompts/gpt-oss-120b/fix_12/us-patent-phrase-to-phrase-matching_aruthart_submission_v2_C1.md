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

0.8039026604903498

# 6. Current score

0.59811

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.56297) has done: 'I fix the protobuf import issue by setting the environment variable before importing transformers, replace the missing model checkpoint with a publicly‑available sentence‑transformer embedding model, and simplify inference to compute cosine similarity between anchor and target embeddings (mapped to [0, 1]). This removes the failing dataset/DataLoader code, ensures a valid `submission.csv` is written, and provides a reasonable correlation score without altering the overall workflow.'
- What this solution (achieved 0.58403) has done: 'I fixed the protobuf import error by switching from the raw `transformers` model loading to the `sentence_transformers.SentenceTransformer` wrapper, which avoids the problematic protobuf call and also handles pooling internally. I upgraded the embedding model to a stronger pretrained checkpoint (“paraphrase‑mpnet‑base‑v2”) that generally yields higher cosine‑similarity correlation for semantic similarity tasks, moving the validation score closer to the target. The rest of the pipeline (reading the test file, computing cosine similarity, rescaling to [0, 1], and writing the submission) is kept unchanged but rewritten for clarity and robustness.'
- What this solution (achieved 0.56359) has done: 'I moved the protobuf environment setting to the very top, added a safe fallback model load, and incorporated the `context` field into the sentences that are encoded so the embeddings capture more information, which should improve the Pearson correlation while keeping the original cosine‑similarity–based prediction pipeline unchanged. The script now runs without the previous import error and writes a proper `submission.csv` file.'
- What this solution (achieved 0.56411) has done: 'The fix moves the protobuf‑environment setting to the very top, avoids the import‑time crash, adds a simple linear calibration using the training data to map raw cosine similarity to the target score range, and preserves the original embedding‑and‑cosine‑similarity pipeline while improving Pearson correlation toward the target.'
- What this solution (achieved 0.56411) has done: 'I added a compatibility patch for the protobuf `MessageFactory` before loading any transformer‑based libraries, which eliminates the `AttributeError` that stopped the script. The rest of the workflow (building sentences, encoding with a Sentence‑Transformer, linear calibration of cosine similarity, and writing the CSV) is unchanged, preserving the original logic while allowing the model to run and achieve a higher Pearson correlation nearer the target score.'
- What this solution (achieved 0.59557) has done: 'I keep the embedding‑and‑cosine‑similarity pipeline unchanged but replace the simple linear calibration with an isotonic regression fit on the training cosine scores. Isotonic regression preserves monotonicity and usually captures the non‑linear relationship between cosine similarity and the 0‑1 rating better, which should raise the Pearson correlation toward the target. I also add the required import and ensure predictions are clipped to the valid range.'
- What this solution (achieved 0.5641) has done: 'I replace the isotonic calibration with a simple three‑feature linear regression that uses cosine similarities between anchor‑target, anchor‑context, and target‑context embeddings. This keeps the embedding‑and‑cosine core unchanged while providing a stronger mapping to the target scores, which should raise the Pearson correlation toward the target. I also compute a separate embedding for the context field and use it in the new features. The rest of the pipeline—including data loading, sentence construction, and CSV output—remains the same.'
- What this solution (achieved 0.5702) has done: 'I keep the overall workflow (embedding, cosine features, regression, CSV output) but improve the calibration stage by (1) switching to a stronger similarity‑focused transformer (“multi‑qa‑mpnet‑base‑dot‑v1”) and (2) expanding the linear calibration to include second‑order polynomial features, which lets the model capture non‑linear relationships between the three cosine similarities and the target scores. This modest change is expected to raise the Pearson correlation toward the target while preserving the core logic and output format.'
- What this solution (achieved 0.60324) has done: 'I replace the simple Ridge regression with a GradientBoostingRegressor, which can model more complex non‑linear relationships between the cosine similarity features (and their polynomial expansions) and the target scores. This change keeps the embedding‑based feature extraction unchanged while providing a stronger calibration step, which should move the Pearson correlation closer to the target score.'
- What this solution (achieved 0.60085) has done: 'I increase the GradientBoostingRegressor capacity (more trees, slightly deeper, smaller learning‑rate) and add a monotonic isotonic regression calibration step that maps the raw GBR predictions to the 0‑1 score range. This keeps the embedding‑based feature extraction unchanged while providing a stronger, monotonic post‑processor, which should raise the Pearson correlation toward the target without altering the core pipeline.'
- What this solution (achieved 0.59811) has done: 'I increase the capacity of the GradientBoostingRegressor (more trees and a slightly deeper depth) so the model can capture more complex relations between the cosine‑similarity features, which is expected to raise the Pearson correlation toward the target score. The rest of the pipeline—including the sentence‑transformer embeddings, feature construction, polynomial expansion, isotonic calibration and CSV output—remains unchanged.'

# 9. Code solution

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
from sklearn.isotonic import IsotonicRegression  # new import for monotonic calibration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

try:
    model = SentenceTransformer(model_name, device=device)
except Exception:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)




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
    batch_size=64,
    convert_to_numpy=True,
    device=device,
)
target_train_emb = model.encode(
    target_train_sentences,
    batch_size=64,
    convert_to_numpy=True,
    device=device,
)

context_train_emb = model.encode(
    train_df["context"].tolist(),
    batch_size=64,
    convert_to_numpy=True,
    device=device,
)


def cosine_numpy(a, b):
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return dot / (norm_a * norm_b + 1e-8)


cos_at = cosine_numpy(anchor_train_emb, target_train_emb)  # anchor‑target
cos_ac = cosine_numpy(anchor_train_emb, context_train_emb)  # anchor‑context
cos_tc = cosine_numpy(target_train_emb, context_train_emb)  # target‑context

X_train = np.column_stack([cos_at, cos_ac, cos_tc])
y_train = train_df["score"].values

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)

gbr_reg = GradientBoostingRegressor(
    n_estimators=1500,
    learning_rate=0.02,
    max_depth=5,
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
    batch_size=64,
    convert_to_numpy=True,
    device=device,
)
target_test_emb = model.encode(
    target_test_sentences,
    batch_size=64,
    convert_to_numpy=True,
    device=device,
)

context_test_emb = model.encode(
    test_df["context"].tolist(),
    batch_size=64,
    convert_to_numpy=True,
    device=device,
)

cos_at_test = cosine_numpy(anchor_test_emb, target_test_emb)
cos_ac_test = cosine_numpy(anchor_test_emb, context_test_emb)
cos_tc_test = cosine_numpy(target_test_emb, context_test_emb)

X_test = np.column_stack([cos_at_test, cos_ac_test, cos_tc_test])
X_test_poly = poly.transform(X_test)

preds_raw = gbr_reg.predict(X_test_poly)

preds = iso_reg.predict(preds_raw)
preds = np.clip(preds, 0.0, 1.0)




## === cell 4
submission = pd.DataFrame({"id": test_df["id"], "score": preds})
submission.to_csv("submission.csv", index=False)
