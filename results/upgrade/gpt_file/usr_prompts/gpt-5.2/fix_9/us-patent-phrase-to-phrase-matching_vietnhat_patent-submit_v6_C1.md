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
numpy==1.26.4
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

0.8210748332613725

# 6. Current score

0.37415

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.45531) has done: 'I fix the root cause preventing execution: the notebook forces offline Hugging Face loading (`local_files_only=True` and `TRANSFORMERS_OFFLINE=1`) but the DeBERTa model isn’t available in the local cache, so tokenization/model creation fails and cascades into later NameErrors. To keep the core inference logic intact while ensuring the code runs end-to-end in Kaggle without internet, I add a safe offline fallback using `sentence-transformers` (which ships with cached models on Kaggle more often) to generate similarity scores, and I preserve your original 5-level scoring semantics by snapping predictions to the allowed {0,0.25,0.5,0.75,1.0} levels. I also make paths robust (try `/kaggle/input/...` then `/kaggle/data/...`) and ensure `submission.csv` is always written with the required columns and row alignment.'
- What this solution (achieved 0.5527) has done: 'I fix the runtime error coming from `sentence-transformers` by pinning a compatible `protobuf` implementation at runtime (via the pure-Python backend) before importing it, which avoids the `MessageFactory.GetPrototype` crash in Kaggle’s environment. I also make the fallback model choice deterministic and improve score toward your target by switching from hard “snap-to-levels” outputs (which tends to hurt Pearson correlation) to a calibrated continuous prediction mapped from cosine similarity into [0,1] and then lightly shrunk toward the dataset mean (calibration only; no architecture/training changes). The HF/DeBERTa offline path remains unchanged and still be used if it’s actually available in the local cache. Finally, I keep the submission-writing logic but ensure `pred` is always a float array aligned to `test_df` and the CSV is written as `submission.csv`.'
- What this solution (achieved 0.35916) has done: 'I fix the crash in the sentence-transformers fallback by avoiding the protobuf-dependent code path that triggers `MessageFactory.GetPrototype` in this Kaggle environment, while keeping your overall “HF offline if available else fallback” logic intact. Concretely, I replace the sentence-transformers fallback with a lightweight, fully offline TF‑IDF cosine similarity baseline (no new packages) and keep your same calibration step (shrink toward train mean) so predictions remain continuous in [0,1] for better Pearson correlation. I also add a small safety guard so `pred` is always created and aligned to `test_df`, ensuring `submission.csv` is written every time. The HF/DeBERTa offline path is preserved unchanged and still be used if the model is actually present in the local cache.'
- What this solution (achieved 0.3585) has done: 'Your current TF‑IDF fallback is likely underperforming because it (1) doesn’t L2-normalize vectors before taking the dot product (so it’s not true cosine similarity), and (2) fits the vectorizer on `test` text too, which can distort similarity structure and hurt generalization/Pearson. I keep the exact same overall fallback approach (TF‑IDF → similarity → shrink to train mean → clip) but compute *true cosine similarity* using `sklearn.metrics.pairwise.cosine_similarity` and fit the TF‑IDF vocabulary on `train` text (then transform test), which is a minimal, legitimate change expected to lift correlation toward your target. I also keep the HF-offline path untouched and only adjust the fallback branch. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.35624) has done: 'Your current gap to the target is large (0.3585 → 0.8211), so we need a real uplift while keeping the same overall fallback logic (TF‑IDF → cosine similarity → shrink to train mean → clip). The biggest low-risk improvement is to fit TF‑IDF on the *paired* text format you score (so the vocabulary/weights reflect anchor/context/target interactions) and to use a **single cosine similarity over the pair** (`[A]` vs `[B]`) rather than two independently vectorized sides whose diagonal cosine is often too weak. I keep the same calibration step (shrink toward train mean) but also add a tiny linear calibration learned on train (still within the same “calibration/post-processing” semantics) to better map cosine similarity to the 0–1 score range, which typically improves Pearson a lot for this competition. The HF offline branch remains unchanged; only the fallback branch is strengthened and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.37415) has done: 'Your current TF‑IDF fallback is leaving a lot of signal on the table because it only scores similarity between `[C]+[A]` and `[B]`, ignoring that the evaluation label is about the *pair* relation (anchor+context vs target) and benefits from cross-term overlap. To move the score up toward your target with minimal core-logic change, I keep the same fallback family (TF‑IDF → cosine → linear calibration → shrink → clip) but (1) compute an additional cosine on a “full-pair” view (`[C][A][B]` vs itself) and a “swapped” cross-view, and (2) combine these few cosines with a tiny linear regression (still just calibration/post-processing) learned on train. This is a small, legitimate adjustment that usually increases Pearson without changing the overall approach, and it stays fully offline and fast. The HF offline DeBERTa path remains untouched; only the fallback branch is strengthened, and the script still always writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
MODEL_NAME = "microsoft/deberta-v3-base"

tokenizer = None
model = None
use_fallback_st = False

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
except Exception as e:
    print(
        "HF offline load failed; will use offline fallback.\nReason:",
        repr(e),
    )
    use_fallback_st = True

use_fallback_st




## === cell 2
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]




## === cell 3
def _read_csv_robust(rel_path_under_competition: str) -> pd.DataFrame:
    candidates = [
        f"/kaggle/input/us-patent-phrase-to-phrase-matching/{rel_path_under_competition}",
        f"/kaggle/data/us-patent-phrase-to-phrase-matching/{rel_path_under_competition}",
        f"/kaggle/input/{rel_path_under_competition}",
        f"/kaggle/data/{rel_path_under_competition}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"Could not find {rel_path_under_competition} in known Kaggle paths: {candidates}"
    )


train_df = _read_csv_robust("train.csv")
test_df = _read_csv_robust("test.csv")
sample_sub = _read_csv_robust("sample_submission.csv")

train_df.shape, test_df.shape, sample_sub.shape




## === cell 4
def encode_row(row, test=False):
    text_a = row["context"][0] + " " + row["anchor"]
    text_b = row["target"]
    ret = tokenizer(text_a, text_b, truncation=True)
    if not test:
        ret["label"] = np.digitize(row["score"], bins=np.linspace(0, 1, 5)) - 1
    return ret


if not use_fallback_st:
    test_data = [encode_row(row, test=True) for _, row in test_df.iterrows()]
    testset = MyDataset(test_data)
    print("HF testset ready:", len(testset))
else:
    testset = None
    print("Skipping HF tokenization because fallback is enabled.")




## === cell 5
def collate_fn(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        if k == "label":
            continue
        out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
    return out


logits = None

if not use_fallback_st:
    loader = DataLoader(
        testset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            all_logits.append(out.logits.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    print("Logits shape:", logits.shape)
else:
    print("Skipping HF forward pass because fallback is enabled.")



## === cell 6
pred = None

levels = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1.0]

if logits is not None:
    x = logits - np.max(logits, axis=1, keepdims=True)
    prob = np.exp(x)
    prob = prob / np.sum(prob, axis=1, keepdims=True)
    pred = np.sum(prob * levels, axis=1).astype(np.float32)
else:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    def _ctx1(s: pd.Series) -> pd.Series:
        return s.astype(str).str[0].fillna("")

    def make_view_CA(df: pd.DataFrame) -> pd.Series:
        return (
            "[C] "
            + _ctx1(df["context"])
            + " [A] "
            + df["anchor"].astype(str).fillna("")
        )

    def make_view_B(df: pd.DataFrame) -> pd.Series:
        return "[B] " + df["target"].astype(str).fillna("")

    def make_view_full(df: pd.DataFrame) -> pd.Series:
        return (
            "[C] "
            + _ctx1(df["context"])
            + " [A] "
            + df["anchor"].astype(str).fillna("")
            + " [B] "
            + df["target"].astype(str).fillna("")
        )

    tr_CA = make_view_CA(train_df).tolist()
    tr_B = make_view_B(train_df).tolist()
    tr_full = make_view_full(train_df).tolist()

    te_CA = make_view_CA(test_df).tolist()
    te_B = make_view_B(test_df).tolist()
    te_full = make_view_full(test_df).tolist()

    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        sublinear_tf=True,
        norm="l2",
    )
    vec.fit((tr_CA + tr_B + tr_full))

    Tr_CA = vec.transform(tr_CA)
    Tr_B = vec.transform(tr_B)
    Tr_full = vec.transform(tr_full)

    Te_CA = vec.transform(te_CA)
    Te_B = vec.transform(te_B)
    Te_full = vec.transform(te_full)

    f1_tr = cosine_similarity(Tr_CA, Tr_B).diagonal().astype(np.float32)
    f1_te = cosine_similarity(Te_CA, Te_B).diagonal().astype(np.float32)

    f2_tr = cosine_similarity(Tr_full, Tr_CA).diagonal().astype(np.float32)
    f2_te = cosine_similarity(Te_full, Te_CA).diagonal().astype(np.float32)

    f3_tr = cosine_similarity(Tr_full, Tr_B).diagonal().astype(np.float32)
    f3_te = cosine_similarity(Te_full, Te_B).diagonal().astype(np.float32)

    f1_tr = np.clip(f1_tr, 0.0, 1.0)
    f2_tr = np.clip(f2_tr, 0.0, 1.0)
    f3_tr = np.clip(f3_tr, 0.0, 1.0)
    f1_te = np.clip(f1_te, 0.0, 1.0)
    f2_te = np.clip(f2_te, 0.0, 1.0)
    f3_te = np.clip(f3_te, 0.0, 1.0)

    y = train_df["score"].astype(np.float32).values
    X = np.stack([np.ones_like(f1_tr), f1_tr, f2_tr, f3_tr], axis=1).astype(np.float32)
    Xt = np.stack([np.ones_like(f1_te), f1_te, f2_te, f3_te], axis=1).astype(np.float32)

    ridge = 1e-3
    A = X.T @ X + ridge * np.eye(X.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, X.T @ y).astype(np.float32)

    raw_cal = (Xt @ w).astype(np.float32)
    raw_cal = np.clip(raw_cal, 0.0, 1.0)

    train_mean = float(train_df["score"].mean())
    alpha = 0.90
    pred = (alpha * raw_cal + (1.0 - alpha) * train_mean).astype(np.float32)
    pred = np.clip(pred, 0.0, 1.0)

if pred is None or len(pred) != len(test_df):
    raise RuntimeError(
        f"Prediction generation failed: pred is {type(pred)}, len={None if pred is None else len(pred)}"
    )

pred = np.nan_to_num(
    pred, nan=float(train_df["score"].mean()), posinf=1.0, neginf=0.0
).astype(np.float32)
pred[:5], pred.shape



## === cell 7
submit = pd.DataFrame({"id": test_df["id"].values, "score": pred.astype(float)})

if "id" in sample_sub.columns and len(sample_sub) == len(submit):
    submit = sample_sub[["id"]].merge(submit, on="id", how="left")
    if submit["score"].isna().any():
        raise ValueError(
            "Some test ids missing scores after merge; check id alignment."
        )

submit.to_csv("submission.csv", index=False)
submit.head(), submit.shape
