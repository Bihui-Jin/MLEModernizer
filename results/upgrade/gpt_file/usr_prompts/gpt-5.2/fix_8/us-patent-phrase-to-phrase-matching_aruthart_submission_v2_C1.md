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

0.58735

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.53375) has done: 'I fix the immediate runtime failure caused by forcing the Python protobuf implementation, which breaks with the installed protobuf version, by removing that environment override. Then I make the model/tokenizer loading robust to Kaggle’s offline environment by first trying local cache and, if missing, falling back to a locally available SentenceTransformer to generate cosine-similarity predictions (still producing valid 0–1 scores). I also fix the downstream NameErrors by ensuring `tokenizer`, `model`, and the dataloader are created in all branches. Finally, I always write a valid `submission.csv` with the required `id,score` columns.'
- What this solution (achieved 0.58735) has done: 'The protobuf/transformers import is failing before any model fallback logic can run, so the main fix is to avoid importing `transformers` at module import time and instead import it lazily only if we actually try to use the HF DeBERTa branch. Then, since the current score (0.53375) is far below the target (~0.804), we keep the same overall fallback idea but nudge score upward by switching to a stronger locally-cacheable SentenceTransformer (prioritizing `all-mpnet-base-v2`) and by using a simple calibration step that maps similarity to the discrete {0,0.25,0.5,0.75,1.0} levels (which matches label structure and typically improves Pearson). The core approach remains “encode two texts → cosine similarity → 0..1 score,” with only minimal robustness and calibration added. The script always write a valid `submission.csv` with the required `id,score` columns.'
- What this solution (achieved 0.58735) has done: 'I fix the crash happening before any fallback logic can run by preventing the `sentence_transformers` (and thus `transformers`) import from executing at module import time; instead, both libraries are imported lazily inside the branch that uses them. This avoids the protobuf-related `MessageFactory.GetPrototype` failure when `transformers` is imported too early in this environment. I keep the existing core approach (HF DeBERTa if locally available, otherwise SentenceTransformer cosine similarity + threshold quantization calibration) unchanged, but make the dataset class safe when `tokenizer` is `None` and ensure all required objects are defined only in the relevant branch. The script still run end-to-end and always write a valid `submission.csv` with `id,score`.'
- What this solution (achieved 0.58735) has done: 'I fix the crash by removing the eager `transformers` import that triggers the protobuf `MessageFactory.GetPrototype` error, and instead avoid the HF branch entirely (since it can’t run reliably here). Then I make the SentenceTransformer fallback import robust by disabling protobuf inside that import scope and by trying a small list of likely-local embedding models. Finally, I keep your existing cosine-similarity + threshold-quantization calibration logic intact and ensure we always produce a correctly formatted `submission.csv` with `id,score`.'
- What this solution (achieved 0.58735) has done: 'I fix the immediate runtime crash caused by the protobuf/transformers incompatibility that is triggered when importing `sentence_transformers`. The minimal robust workaround is to force the pure-Python protobuf implementation **before** any protobuf-dependent imports happen, so the SentenceTransformer fallback can load. I also make the `TextDataset` tokenizer requirement non-fatal since this script never uses the tokenizer branch, and add a tiny safety fallback to continuous predictions if threshold fitting ever degenerates (score-neutral/stability). Core logic (SentenceTransformer embeddings → cosine similarity → [0,1] → threshold quantization → submission.csv) is preserved.'
- What this solution (achieved 0.58735) has done: 'I fix the runtime crash caused by an incompatibility between `sentence_transformers/transformers` and the installed `protobuf` runtime (the `MessageFactory.GetPrototype` AttributeError). The minimal robust fix is to force-install a protobuf version that is compatible with these libraries at runtime (offline, from Kaggle’s prebuilt wheels cache), then import `sentence_transformers` only after that fix is applied. I keep your core approach unchanged (SentenceTransformer embeddings → cosine similarity → [0,1] → threshold quantization calibration) and ensure the script always writes a valid `submission.csv` with `id,score`. This should also allow loading the stronger local embedding model (e.g., `all-mpnet-base-v2`) again, which is expected to move the score upward toward the target.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver
    except Exception:
        pb_ver = None

    target = "protobuf==4.25.3"

    needs = True
    if pb_ver is not None:
        needs = pb_ver != "4.25.3"

    if needs:
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "--no-input",
                    "--no-warn-script-location",
                    "--no-index",
                    "--find-links",
                    "/kaggle/input",
                    target,
                ]
            )
        except Exception:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "--no-input",
                    "--no-warn-script-location",
                    target,
                ]
            )

        for k in list(sys.modules.keys()):
            if k.startswith("google.protobuf"):
                del sys.modules[k]


_ensure_compatible_protobuf()



## === cell 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

MODEL_NAME = "microsoft/deberta-v3-base"

tokenizer = None
model = None
use_st_fallback = True
st_model = None

try:
    from sentence_transformers import SentenceTransformer
except Exception as ie:
    raise RuntimeError(
        "Failed to import sentence_transformers; cannot run the fallback embedding model."
    ) from ie

st_candidates = [
    "sentence-transformers/all-mpnet-base-v2",
    "all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "all-MiniLM-L6-v2",
]
last_err = None
for cand in st_candidates:
    try:
        st_model = SentenceTransformer(cand, device=str(device))
        print(f"Loaded SentenceTransformer from local cache: {cand}")
        break
    except Exception as ee:
        last_err = ee
        st_model = None

if st_model is None:
    raise RuntimeError(
        "No locally available SentenceTransformer model found in the Kaggle environment cache."
    ) from last_err



## === cell 2
test_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv"
test_df = pd.read_csv(test_path)
print("Loaded test:", test_df.shape)
print(test_df.head())

train_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv"
train_df = pd.read_csv(train_path)
print("Loaded train:", train_df.shape)
print(train_df.head())




## === cell 3
class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.data = []
        if tokenizer is None:
            return
        texts = (
            "Category: "
            + df["context"].astype(str)
            + " Text 1: "
            + df["anchor"].astype(str)
            + " Text 2: "
            + df["target"].astype(str)
        ).tolist()
        self.data = [tokenizer(text, truncation=True) for text in texts]

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)




## === cell 4
def st_pair_texts(df: pd.DataFrame):
    texts_a = (
        "Category: "
        + df["context"].astype(str)
        + " Text 1: "
        + df["anchor"].astype(str)
    ).tolist()
    texts_b = (
        "Category: "
        + df["context"].astype(str)
        + " Text 2: "
        + df["target"].astype(str)
    ).tolist()
    return texts_a, texts_b


def cosine01_from_st(model, df: pd.DataFrame, batch_size: int = 64):
    texts_a, texts_b = st_pair_texts(df)
    emb_a = model.encode(
        texts_a,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    emb_b = model.encode(
        texts_b,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    cos = np.sum(emb_a * emb_b, axis=1)  # cosine similarity in [-1, 1]
    preds = (cos + 1.0) / 2.0  # map to [0, 1]
    return np.clip(preds, 0.0, 1.0)


def fit_quantize_thresholds(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Minimal calibration to better match the discrete label structure.
    Fits 4 thresholds to map continuous predictions into {0, .25, .5, .75, 1}.
    """
    levels = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    means = []
    for lv in levels:
        mask = np.isclose(y_true, lv)
        if mask.sum() == 0:
            means.append(np.nan)
        else:
            means.append(float(np.mean(y_pred[mask])))
    means = np.array(means, dtype=float)

    if np.isnan(means).any():
        idx = np.arange(len(means))
        good = ~np.isnan(means)
        means[~good] = np.interp(idx[~good], idx[good], means[good])

    means = np.maximum.accumulate(means)
    thr = (means[:-1] + means[1:]) / 2.0
    thr = np.clip(thr, 0.0, 1.0)

    eps = 1e-6
    for i in range(1, len(thr)):
        if thr[i] <= thr[i - 1]:
            thr[i] = min(1.0, thr[i - 1] + eps)

    if not np.all(np.diff(thr) > 0):
        thr = np.array([0.2, 0.4, 0.6, 0.8], dtype=float)
    return thr


def apply_quantize(y_pred: np.ndarray, thr: np.ndarray):
    levels = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    idx = np.digitize(y_pred, thr, right=False)
    return levels[idx]




## === cell 5
preds = None

raw_preds = cosine01_from_st(st_model, test_df, batch_size=64)

train_raw = cosine01_from_st(st_model, train_df, batch_size=64)
thr = fit_quantize_thresholds(
    train_df["score"].values.astype(float), train_raw.astype(float)
)
preds = apply_quantize(raw_preds.astype(float), thr).astype(float)

print("Preds:", preds.shape, "min/max:", float(preds.min()), float(preds.max()))
print("Thresholds:", thr)



## === cell 6
submission = pd.DataFrame(
    {"id": test_df["id"].astype(str).values, "score": preds.astype(float)}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
print("submission.csv exists:", os.path.exists("submission.csv"))
