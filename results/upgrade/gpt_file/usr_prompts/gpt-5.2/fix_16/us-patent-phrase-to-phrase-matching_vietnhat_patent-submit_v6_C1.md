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

0.43781

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.45531) has done: 'I fix the root cause preventing execution: the notebook forces offline Hugging Face loading (`local_files_only=True` and `TRANSFORMERS_OFFLINE=1`) but the DeBERTa model isn’t available in the local cache, so tokenization/model creation fails and cascades into later NameErrors. To keep the core inference logic intact while ensuring the code runs end-to-end in Kaggle without internet, I add a safe offline fallback using `sentence-transformers` (which ships with cached models on Kaggle more often) to generate similarity scores, and I preserve your original 5-level scoring semantics by snapping predictions to the allowed {0,0.25,0.5,0.75,1.0} levels. I also make paths robust (try `/kaggle/input/...` then `/kaggle/data/...`) and ensure `submission.csv` is always written with the required columns and row alignment.'
- What this solution (achieved 0.5527) has done: 'I fix the runtime error coming from `sentence-transformers` by pinning a compatible `protobuf` implementation at runtime (via the pure-Python backend) before importing it, which avoids the `MessageFactory.GetPrototype` crash in Kaggle’s environment. I also make the fallback model choice deterministic and improve score toward your target by switching from hard “snap-to-levels” outputs (which tends to hurt Pearson correlation) to a calibrated continuous prediction mapped from cosine similarity into [0,1] and then lightly shrunk toward the dataset mean (calibration only; no architecture/training changes). The HF/DeBERTa offline path remains unchanged and still be used if it’s actually available in the local cache. Finally, I keep the submission-writing logic but ensure `pred` is always a float array aligned to `test_df` and the CSV is written as `submission.csv`.'
- What this solution (achieved 0.35916) has done: 'I fix the crash in the sentence-transformers fallback by avoiding the protobuf-dependent code path that triggers `MessageFactory.GetPrototype` in this Kaggle environment, while keeping your overall “HF offline if available else fallback” logic intact. Concretely, I replace the sentence-transformers fallback with a lightweight, fully offline TF‑IDF cosine similarity baseline (no new packages) and keep your same calibration step (shrink toward train mean) so predictions remain continuous in [0,1] for better Pearson correlation. I also add a small safety guard so `pred` is always created and aligned to `test_df`, ensuring `submission.csv` is written every time. The HF/DeBERTa offline path is preserved unchanged and still be used if the model is actually present in the local cache.'
- What this solution (achieved 0.3585) has done: 'Your current TF‑IDF fallback is likely underperforming because it (1) doesn’t L2-normalize vectors before taking the dot product (so it’s not true cosine similarity), and (2) fits the vectorizer on `test` text too, which can distort similarity structure and hurt generalization/Pearson. I keep the exact same overall fallback approach (TF‑IDF → similarity → shrink to train mean → clip) but compute *true cosine similarity* using `sklearn.metrics.pairwise.cosine_similarity` and fit the TF‑IDF vocabulary on `train` text (then transform test), which is a minimal, legitimate change expected to lift correlation toward your target. I also keep the HF-offline path untouched and only adjust the fallback branch. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.35624) has done: 'Your current gap to the target is large (0.3585 → 0.8211), so we need a real uplift while keeping the same overall fallback logic (TF‑IDF → cosine similarity → shrink to train mean → clip). The biggest low-risk improvement is to fit TF‑IDF on the *paired* text format you score (so the vocabulary/weights reflect anchor/context/target interactions) and to use a **single cosine similarity over the pair** (`[A]` vs `[B]`) rather than two independently vectorized sides whose diagonal cosine is often too weak. I keep the same calibration step (shrink toward train mean) but also add a tiny linear calibration learned on train (still within the same “calibration/post-processing” semantics) to better map cosine similarity to the 0–1 score range, which typically improves Pearson a lot for this competition. The HF offline branch remains unchanged; only the fallback branch is strengthened and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.37415) has done: 'Your current TF‑IDF fallback is leaving a lot of signal on the table because it only scores similarity between `[C]+[A]` and `[B]`, ignoring that the evaluation label is about the *pair* relation (anchor+context vs target) and benefits from cross-term overlap. To move the score up toward your target with minimal core-logic change, I keep the same fallback family (TF‑IDF → cosine → linear calibration → shrink → clip) but (1) compute an additional cosine on a “full-pair” view (`[C][A][B]` vs itself) and a “swapped” cross-view, and (2) combine these few cosines with a tiny linear regression (still just calibration/post-processing) learned on train. This is a small, legitimate adjustment that usually increases Pearson without changing the overall approach, and it stays fully offline and fast. The HF offline DeBERTa path remains untouched; only the fallback branch is strengthened, and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 0.37443) has done: 'Your current score (0.374) is far below the target (0.821), and since HF offline DeBERTa isn’t available your TF‑IDF fallback needs a small but meaningful uplift without changing the overall approach. I keep the same “TF‑IDF → cosine features → tiny linear (ridge) calibration → shrink to mean → clip” core logic, but fix two key mismatches: (1) learn the calibration on an out-of-fold (OOF) prediction instead of fitting and predicting on the same rows (reduces overfit, improves generalization/Pearson), and (2) add one minimal extra similarity feature that better matches the task: cosine between (anchor+context) and (target+context). The HF branch remains untouched, and the script still writes a valid `submission.csv` with correct alignment.'
- What this solution (achieved 0.43768) has done: 'I fix the TF‑IDF fallback crash by ensuring the word- and char‑TFIDF matrices have identical shapes before combining them (they currently have different vocab sizes, so sparse addition fails). The smallest safe change is to horizontally stack the two representations (`hstack`) instead of adding them, which preserves the same core “TF‑IDF → cosine similarity → linear calibration → shrink → clip” pipeline. I also fix a small logic bug where `f5_*` accidentally duplicates `f4_*` by computing `f5` from the intended `AC` view, improving signal without changing the modeling approach. Finally, I keep the submission-writing logic but make sure `pred` is always created so `submission.csv` is written successfully.'
- What this solution (achieved 0.43711) has done: 'Your current score (0.43768) is far below the target (0.82107), so we should improve the fallback branch (since HF offline DeBERTa likely isn’t available) with the smallest changes that legitimately increase Pearson. I keep your exact pipeline shape (TF‑IDF → a few cosine features → ridge linear calibration with OOF → affine calibrate → shrink to mean → clip), but make two minimal upgrades: (1) use GroupKFold by `context` (first CPC letter) to reduce leakage-like effects and improve generalization, and (2) add one extra similarity feature that’s very aligned with the task: cosine between (context+anchor) and (context+target). Everything else (paths, submission alignment, HF branch) stays the same, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.43205) has done: 'The pipeline fails because the normal-equation solve for the ridge calibration can hit a singular matrix in some folds, which stops prediction generation and leaves `pred=None`, causing the submission step to crash. I keep the exact TF‑IDF + cosine-feature + OOF ridge calibration logic, but replace `np.linalg.solve` with a numerically safe fallback that still solves the same ridge system (try Cholesky/solve; if it fails, use least-squares), and I add a tiny diagonal “jitter” only when needed to make the system invertible. I also ensure the selected ridge is never `None` (fallback to the first grid value if all attempts fail) so `pred` is always produced. No model/feature changes are introduced; this is a stability fix to guarantee an end-to-end run and a valid `submission.csv`.'
- What this solution (achieved 0.43781) has done: 'Your current score (0.432) is far below the target (0.821), and the main limitation is that the TF‑IDF fallback uses only a handful of cosine features and a ridge fit; we can legitimately lift Pearson without changing the overall approach by adding a few very cheap, high-signal cosine features that better capture anchor–target overlap and context conditioning. I keep the exact same pipeline shape (TF‑IDF → cosine features → GroupKFold OOF ridge selection → affine calibration → shrink to mean → clip), but (1) add 3 additional views/features (context-only match, anchor↔(context+target), target↔(context+anchor)), and (2) standardize features using train statistics before ridge (this is still linear calibration, but improves numerical conditioning and fit). Everything else (HF offline branch, data paths, training approach/loops, submission writing) remains unchanged, and it still run end-to-end and write a valid `submission.csv`.'

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
    from sklearn.model_selection import GroupKFold
    from scipy.sparse import hstack

    def _solve_ridge(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            jitter = 1e-6 * float(np.trace(A) / max(A.shape[0], 1))
            if not np.isfinite(jitter) or jitter <= 0:
                jitter = 1e-6
            A2 = A + jitter * np.eye(A.shape[0], dtype=A.dtype)
            try:
                return np.linalg.solve(A2, b)
            except np.linalg.LinAlgError:
                w, *_ = np.linalg.lstsq(A2, b, rcond=None)
                return w

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

    def make_view_CB(df: pd.DataFrame) -> pd.Series:
        return (
            "[C] "
            + _ctx1(df["context"])
            + " [B] "
            + df["target"].astype(str).fillna("")
        )

    def make_view_AC(df: pd.DataFrame) -> pd.Series:
        return (
            "[C] "
            + _ctx1(df["context"])
            + " [A] "
            + df["anchor"].astype(str).fillna("")
            + " [T] "
            + df["target"].astype(str).fillna("")
        )

    def make_view_CT(df: pd.DataFrame) -> pd.Series:
        return (
            "[C] "
            + _ctx1(df["context"])
            + " [T] "
            + df["target"].astype(str).fillna("")
        )

    def make_view_CTarget(df: pd.DataFrame) -> pd.Series:
        return (
            "[C] "
            + _ctx1(df["context"])
            + " [T] "
            + df["target"].astype(str).fillna("")
        )

    def make_view_Aonly(df: pd.DataFrame) -> pd.Series:
        return "[A] " + df["anchor"].astype(str).fillna("")

    def make_view_Tonly(df: pd.DataFrame) -> pd.Series:
        return "[T] " + df["target"].astype(str).fillna("")

    def make_view_Conly(df: pd.DataFrame) -> pd.Series:
        return "[C] " + _ctx1(df["context"])

    def make_view_AT(df: pd.DataFrame) -> pd.Series:
        return (
            "[A] "
            + df["anchor"].astype(str).fillna("")
            + " [T] "
            + df["target"].astype(str).fillna("")
        )

    tr_CA = make_view_CA(train_df).tolist()
    tr_B = make_view_B(train_df).tolist()
    tr_full = make_view_full(train_df).tolist()
    tr_CB = make_view_CB(train_df).tolist()
    tr_AC = make_view_AC(train_df).tolist()
    tr_CT = make_view_CT(train_df).tolist()
    tr_CTarget = make_view_CTarget(train_df).tolist()
    tr_Aonly = make_view_Aonly(train_df).tolist()
    tr_Tonly = make_view_Tonly(train_df).tolist()
    tr_Conly = make_view_Conly(train_df).tolist()
    tr_AT = make_view_AT(train_df).tolist()

    te_CA = make_view_CA(test_df).tolist()
    te_B = make_view_B(test_df).tolist()
    te_full = make_view_full(test_df).tolist()
    te_CB = make_view_CB(test_df).tolist()
    te_AC = make_view_AC(test_df).tolist()
    te_CT = make_view_CT(test_df).tolist()
    te_CTarget = make_view_CTarget(test_df).tolist()
    te_Aonly = make_view_Aonly(test_df).tolist()
    te_Tonly = make_view_Tonly(test_df).tolist()
    te_Conly = make_view_Conly(test_df).tolist()
    te_AT = make_view_AT(test_df).tolist()

    vec_word = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        sublinear_tf=True,
        norm="l2",
    )
    vec_char = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        sublinear_tf=True,
        norm="l2",
    )

    corpus = (
        tr_CA
        + tr_B
        + tr_full
        + tr_CB
        + tr_AC
        + tr_CT
        + tr_CTarget
        + tr_Aonly
        + tr_Tonly
        + tr_Conly
        + tr_AT
    )
    vec_word.fit(corpus)
    vec_char.fit(corpus)

    def _tfidf_mix(texts):
        Xw = vec_word.transform(texts)
        Xc = vec_char.transform(texts)
        return hstack([Xw, Xc], format="csr")

    Tr_CA = _tfidf_mix(tr_CA)
    Tr_B = _tfidf_mix(tr_B)
    Tr_full = _tfidf_mix(tr_full)
    Tr_CB = _tfidf_mix(tr_CB)
    Tr_AC = _tfidf_mix(tr_AC)
    Tr_CT = _tfidf_mix(tr_CT)
    Tr_CTarget = _tfidf_mix(tr_CTarget)
    Tr_Aonly = _tfidf_mix(tr_Aonly)
    Tr_Tonly = _tfidf_mix(tr_Tonly)
    Tr_Conly = _tfidf_mix(tr_Conly)
    Tr_AT = _tfidf_mix(tr_AT)

    Te_CA = _tfidf_mix(te_CA)
    Te_B = _tfidf_mix(te_B)
    Te_full = _tfidf_mix(te_full)
    Te_CB = _tfidf_mix(te_CB)
    Te_AC = _tfidf_mix(te_AC)
    Te_CT = _tfidf_mix(te_CT)
    Te_CTarget = _tfidf_mix(te_CTarget)
    Te_Aonly = _tfidf_mix(te_Aonly)
    Te_Tonly = _tfidf_mix(te_Tonly)
    Te_Conly = _tfidf_mix(te_Conly)
    Te_AT = _tfidf_mix(te_AT)

    f1_tr = cosine_similarity(Tr_CA, Tr_B).diagonal().astype(np.float32)
    f1_te = cosine_similarity(Te_CA, Te_B).diagonal().astype(np.float32)

    f2_tr = cosine_similarity(Tr_full, Tr_CA).diagonal().astype(np.float32)
    f2_te = cosine_similarity(Te_full, Te_CA).diagonal().astype(np.float32)

    f3_tr = cosine_similarity(Tr_full, Tr_B).diagonal().astype(np.float32)
    f3_te = cosine_similarity(Te_full, Te_B).diagonal().astype(np.float32)

    f4_tr = cosine_similarity(Tr_CA, Tr_CB).diagonal().astype(np.float32)
    f4_te = cosine_similarity(Te_CA, Te_CB).diagonal().astype(np.float32)

    f5_tr = cosine_similarity(Tr_AC, Tr_B).diagonal().astype(np.float32)
    f5_te = cosine_similarity(Te_AC, Te_B).diagonal().astype(np.float32)

    f6_tr = cosine_similarity(Tr_CA, Tr_CT).diagonal().astype(np.float32)
    f6_te = cosine_similarity(Te_CA, Te_CT).diagonal().astype(np.float32)

    f7_tr = cosine_similarity(Tr_CA, Tr_CTarget).diagonal().astype(np.float32)
    f7_te = cosine_similarity(Te_CA, Te_CTarget).diagonal().astype(np.float32)

    f8_tr = cosine_similarity(Tr_Aonly, Tr_Tonly).diagonal().astype(np.float32)
    f8_te = cosine_similarity(Te_Aonly, Te_Tonly).diagonal().astype(np.float32)

    f9_tr = (
        cosine_similarity(Tr_Conly, Tr_Conly).diagonal().astype(np.float32)
    )  # always 1, but kept for symmetry
    f9_te = cosine_similarity(Te_Conly, Te_Conly).diagonal().astype(np.float32)

    f10_tr = (
        cosine_similarity(Tr_Aonly, Tr_CB).diagonal().astype(np.float32)
    )  # anchor vs (context+target)
    f10_te = cosine_similarity(Te_Aonly, Te_CB).diagonal().astype(np.float32)

    f11_tr = (
        cosine_similarity(Tr_Tonly, Tr_CA).diagonal().astype(np.float32)
    )  # target vs (context+anchor)
    f11_te = cosine_similarity(Te_Tonly, Te_CA).diagonal().astype(np.float32)

    f12_tr = (
        cosine_similarity(Tr_AT, Tr_Conly).diagonal().astype(np.float32)
    )  # (anchor+target) vs context
    f12_te = cosine_similarity(Te_AT, Te_Conly).diagonal().astype(np.float32)

    for arr_name in [
        "f1_tr",
        "f2_tr",
        "f3_tr",
        "f4_tr",
        "f5_tr",
        "f6_tr",
        "f7_tr",
        "f8_tr",
        "f9_tr",
        "f10_tr",
        "f11_tr",
        "f12_tr",
        "f1_te",
        "f2_te",
        "f3_te",
        "f4_te",
        "f5_te",
        "f6_te",
        "f7_te",
        "f8_te",
        "f9_te",
        "f10_te",
        "f11_te",
        "f12_te",
    ]:
        locals()[arr_name] = np.clip(locals()[arr_name], 0.0, 1.0)

    y = train_df["score"].astype(np.float32).values

    X_all = np.stack(
        [
            np.ones_like(f1_tr),
            f1_tr,
            f2_tr,
            f3_tr,
            f4_tr,
            f5_tr,
            f6_tr,
            f7_tr,
            f8_tr,
            f10_tr,
            f11_tr,
            f12_tr,
        ],
        axis=1,
    ).astype(np.float32)
    Xt = np.stack(
        [
            np.ones_like(f1_te),
            f1_te,
            f2_te,
            f3_te,
            f4_te,
            f5_te,
            f6_te,
            f7_te,
            f8_te,
            f10_te,
            f11_te,
            f12_te,
        ],
        axis=1,
    ).astype(np.float32)

    mu = X_all[:, 1:].mean(axis=0, keepdims=True)
    sd = X_all[:, 1:].std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    X_all_std = X_all.copy()
    Xt_std = Xt.copy()
    X_all_std[:, 1:] = (X_all[:, 1:] - mu) / sd
    Xt_std[:, 1:] = (Xt[:, 1:] - mu) / sd

    groups = _ctx1(train_df["context"]).values
    gkf = GroupKFold(n_splits=5)

    ridge_grid = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    best_ridge = None
    best_mse = float("inf")
    best_oof = None

    for ridge in ridge_grid:
        oof_tmp = np.zeros(len(train_df), dtype=np.float32)
        ok = True
        for tr_idx, va_idx in gkf.split(X_all_std, y, groups=groups):
            X_tr, y_tr = X_all_std[tr_idx], y[tr_idx]
            X_va = X_all_std[va_idx]
            A = X_tr.T @ X_tr + float(ridge) * np.eye(X_tr.shape[1], dtype=np.float32)
            b = X_tr.T @ y_tr
            try:
                w_fold = _solve_ridge(A, b).astype(np.float32)
            except Exception:
                ok = False
                break
            oof_tmp[va_idx] = (X_va @ w_fold).astype(np.float32)

        if not ok:
            continue

        oof_tmp_clip = np.clip(oof_tmp, 0.0, 1.0)
        mse = float(np.mean((oof_tmp_clip - y) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_ridge = float(ridge)
            best_oof = oof_tmp.copy()

    if best_ridge is None or best_oof is None:
        best_ridge = float(ridge_grid[0])
        best_oof = np.zeros(len(train_df), dtype=np.float32)

    oof = best_oof
    ridge = best_ridge

    A_full = X_all_std.T @ X_all_std + ridge * np.eye(
        X_all_std.shape[1], dtype=np.float32
    )
    w = _solve_ridge(A_full, X_all_std.T @ y).astype(np.float32)
    raw_cal = (Xt_std @ w).astype(np.float32)

    oof_clip = np.clip(oof, 0.0, 1.0)
    A2 = np.stack([oof_clip, np.ones_like(oof_clip)], axis=1).astype(np.float32)
    ab, *_ = np.linalg.lstsq(A2, y, rcond=None)
    a, b = float(ab[0]), float(ab[1])

    raw_cal = (a * np.clip(raw_cal, 0.0, 1.0) + b).astype(np.float32)
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
