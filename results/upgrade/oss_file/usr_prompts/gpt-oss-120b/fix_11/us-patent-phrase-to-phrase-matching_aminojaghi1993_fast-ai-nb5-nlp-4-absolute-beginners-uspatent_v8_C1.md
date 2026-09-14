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

3.12

# 3. Installed packages

datasets==4.4.1
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
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

0.8039326304000577

# 6. Current score

0.5415

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.46844) has done: 'I replace the failing transformer‑based classification setup with a lightweight embedding + linear‑regression pipeline that uses `sentence‑transformers` (which avoids the protobuf error). The code now loads a pretrained SentenceTransformer, encodes the combined text fields, trains a simple Ridge regression on the training split, evaluates Pearson on the validation split, and finally generates a correctly‑named `submission.csv` with the required columns. This fixes all runtime errors and produces a valid submission while keeping the overall modelling approach (sentence‑level embeddings → regression) unchanged.'
- What this solution (achieved 0.48334) has done: 'I fixed the protobuf error by replacing the failing `SentenceTransformer` with a lightweight custom embedder built from HuggingFace `AutoTokenizer`/`AutoModel`. I also made the data path robust and switched the regression model from Ridge to plain `LinearRegression`, which usually yields a higher Pearson correlation for these embeddings. The script now runs end‑to‑end, creates a proper `submission.csv`, and moves the validation score closer to the target.'
- What this solution (achieved 0.5415) has done: 'I make the embedder robust by catching model‑loading errors and falling back to a reliable Transformers model, while also switching to a stronger pretrained checkpoint (`paraphrase‑MiniLM‑L12‑v2`). Additionally, I replace the plain LinearRegression with a Ridge regression (small L2 regularisation) which generally improves Pearson on these embeddings. These minimal changes fix the protobuf crash and should raise the validation correlation toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.manual_seed(42)
np.random.seed(42)



## === cell 1
possible_paths = [
    Path("input/us-patent-phrase-to-phrase-matching"),
    Path("../input/us-patent-phrase-to-phrase-matching"),
]
base_path = next(p for p in possible_paths if p.exists())
train_path = base_path / "train.csv"
test_path = base_path / "test.csv"
sample_sub_path = base_path / "sample_submission.csv"



## === cell 2
from transformers import AutoTokenizer, AutoModel


class SimpleEmbedder:
    def __init__(
        self,
        model_name="sentence-transformers/paraphrase-MiniLM-L12-v2",
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
        except Exception as e:
            print(
                f"Primary model '{model_name}' failed to load ({e}). "
                "Falling back to 'distilbert-base-uncased'."
            )
            fallback_name = "distilbert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_name)
            self.model = AutoModel.from_pretrained(fallback_name).to(self.device)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # last hidden state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(
        self,
        sentences,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
    ):
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                model_output = self.model(**encoded)
                batch_embeddings = self._mean_pooling(
                    model_output, encoded["attention_mask"]
                )
                batch_embeddings = batch_embeddings.cpu()
                if convert_to_numpy:
                    batch_embeddings = batch_embeddings.numpy()
                all_embeddings.append(batch_embeddings)
        return np.vstack(all_embeddings)


embed_model = SimpleEmbedder()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


def build_input(df):
    return (
        "TEXT1: "
        + df["context"]
        + "; TEXT2: "
        + df["target"]
        + "; ANC1: "
        + df["anchor"]
    )


train_df["input"] = build_input(train_df)
test_df["input"] = build_input(test_df)



## === cell 4
train_split, val_split = train_test_split(train_df, test_size=0.2, random_state=42)

train_emb = embed_model.encode(
    train_split["input"].tolist(),
    batch_size=64,
    show_progress_bar=False,
    convert_to_numpy=True,
)
val_emb = embed_model.encode(
    val_split["input"].tolist(),
    batch_size=64,
    show_progress_bar=False,
    convert_to_numpy=True,
)
test_emb = embed_model.encode(
    test_df["input"].tolist(),
    batch_size=64,
    show_progress_bar=False,
    convert_to_numpy=True,
)

train_labels = train_split["score"].astype(np.float32).values
val_labels = val_split["score"].astype(np.float32).values



## === cell 5
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(train_emb, train_labels)

val_preds = ridge.predict(val_emb)
pearson, _ = pearsonr(val_preds, val_labels)
print(f"Validation Pearson correlation: {pearson:.6f}")



## === cell 6
test_preds = ridge.predict(test_emb)
test_preds = np.clip(test_preds, 0.0, 1.0)

submission = pd.DataFrame({"id": test_df["id"], "score": test_preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Saved {submission_path}")
print(submission.head())
