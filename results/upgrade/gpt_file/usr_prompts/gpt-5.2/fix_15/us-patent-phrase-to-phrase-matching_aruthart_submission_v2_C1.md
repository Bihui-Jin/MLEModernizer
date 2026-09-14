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

0.59361

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.53375) has done: 'I fix the immediate runtime failure caused by forcing the Python protobuf implementation, which breaks with the installed protobuf version, by removing that environment override. Then I make the model/tokenizer loading robust to Kaggle’s offline environment by first trying local cache and, if missing, falling back to a locally available SentenceTransformer to generate cosine-similarity predictions (still producing valid 0–1 scores). I also fix the downstream NameErrors by ensuring `tokenizer`, `model`, and the dataloader are created in all branches. Finally, I always write a valid `submission.csv` with the required `id,score` columns.'
- What this solution (achieved 0.58735) has done: 'The protobuf/transformers import is failing before any model fallback logic can run, so the main fix is to avoid importing `transformers` at module import time and instead import it lazily only if we actually try to use the HF DeBERTa branch. Then, since the current score (0.53375) is far below the target (~0.804), we keep the same overall fallback idea but nudge score upward by switching to a stronger locally-cacheable SentenceTransformer (prioritizing `all-mpnet-base-v2`) and by using a simple calibration step that maps similarity to the discrete {0,0.25,0.5,0.75,1.0} levels (which matches label structure and typically improves Pearson). The core approach remains “encode two texts → cosine similarity → 0..1 score,” with only minimal robustness and calibration added. The script always write a valid `submission.csv` with the required `id,score` columns.'
- What this solution (achieved 0.58735) has done: 'I fix the crash happening before any fallback logic can run by preventing the `sentence_transformers` (and thus `transformers`) import from executing at module import time; instead, both libraries are imported lazily inside the branch that uses them. This avoids the protobuf-related `MessageFactory.GetPrototype` failure when `transformers` is imported too early in this environment. I keep the existing core approach (HF DeBERTa if locally available, otherwise SentenceTransformer cosine similarity + threshold quantization calibration) unchanged, but make the dataset class safe when `tokenizer` is `None` and ensure all required objects are defined only in the relevant branch. The script still run end-to-end and always write a valid `submission.csv` with `id,score`.'
- What this solution (achieved 0.58735) has done: 'I fix the crash by removing the eager `transformers` import that triggers the protobuf `MessageFactory.GetPrototype` error, and instead avoid the HF branch entirely (since it can’t run reliably here). Then I make the SentenceTransformer fallback import robust by disabling protobuf inside that import scope and by trying a small list of likely-local embedding models. Finally, I keep your existing cosine-similarity + threshold-quantization calibration logic intact and ensure we always produce a correctly formatted `submission.csv` with `id,score`.'
- What this solution (achieved 0.58735) has done: 'I fix the immediate runtime crash caused by the protobuf/transformers incompatibility that is triggered when importing `sentence_transformers`. The minimal robust workaround is to force the pure-Python protobuf implementation **before** any protobuf-dependent imports happen, so the SentenceTransformer fallback can load. I also make the `TextDataset` tokenizer requirement non-fatal since this script never uses the tokenizer branch, and add a tiny safety fallback to continuous predictions if threshold fitting ever degenerates (score-neutral/stability). Core logic (SentenceTransformer embeddings → cosine similarity → [0,1] → threshold quantization → submission.csv) is preserved.'
- What this solution (achieved 0.58735) has done: 'I fix the runtime crash caused by an incompatibility between `sentence_transformers/transformers` and the installed `protobuf` runtime (the `MessageFactory.GetPrototype` AttributeError). The minimal robust fix is to force-install a protobuf version that is compatible with these libraries at runtime (offline, from Kaggle’s prebuilt wheels cache), then import `sentence_transformers` only after that fix is applied. I keep your core approach unchanged (SentenceTransformer embeddings → cosine similarity → [0,1] → threshold quantization calibration) and ensure the script always writes a valid `submission.csv` with `id,score`. This should also allow loading the stronger local embedding model (e.g., `all-mpnet-base-v2`) again, which is expected to move the score upward toward the target.'
- What this solution (achieved 0.58165) has done: 'Your current approach (SentenceTransformer cosine similarity + simple calibration) is fine, but the quantization step is likely hurting Pearson because it collapses predictions onto only 5 values; Pearson usually benefits from continuous outputs. I keep the same core embedding logic and calibration intent, but replace hard quantization with a tiny monotonic piecewise-linear calibrator learned from the training set (still aligned to the discrete label structure, but outputs remain continuous). I also add a context prefix normalization (`[CPC=...]`) and a stable post-scale (learned on train) to better match the 0–1 label range without changing the model. These are minimal changes that typically move correlation upward from the ~0.59 range toward your ~0.80 target while preserving evaluation semantics and producing the same `submission.csv` format.'
- What this solution (achieved 0.58165) has done: 'You’re far below the target (0.58165 vs 0.8039), so we should improve score with minimal, metric-aligned changes while keeping the same core logic (SentenceTransformer embeddings → cosine similarity → calibration → submission). The biggest likely issue is that the current calibrator is fit on *in-sample* train predictions (no CV), which often overfits the mapping and hurts generalization/Pearson on test; we fit the same piecewise-linear + affine calibration out-of-fold using KFold, then refit on full train and apply to test. We also make the affine step explicitly Pearson-optimal by standardizing to train mean/std (equivalent to best linear correlation alignment under clipping), still preserving evaluation semantics and keeping outputs in [0,1]. All I/O paths stay the same and we still write `submission.csv` with `id,score`.'
- What this solution (achieved 0.38808) has done: 'Your current score (0.58165) is far below the target (0.8039), so we should improve Pearson with the smallest change that keeps the same core “SentenceTransformer embeddings → cosine → calibration → submission” logic. The main weakness is encoding anchor and target separately; sentence-transformers models are typically trained/fine-tuned for pair scoring via *cross-encoder* or via *bi-encoder with pair prompts*, and a very effective minimal upgrade (still embeddings + cosine) is to encode a single combined text representation for the pair. We therefore switch to a “pair string” encoding (one encode call) and compare it to a fixed reference vector is not appropriate; instead we keep cosine by encoding *two* combined variants with shared information so the cosine captures the relation—specifically we encode `[CPC] anchor [SEP] target` and `[CPC] target [SEP] anchor` and average their cosine-similarity (symmetrization), which tends to increase correlation while preserving the same inference structure. Everything else (OOF piecewise-linear calibration + affine mean/std match, output format, paths) stays the same.'
- What this solution (achieved 0.58165) has done: 'Your current score (0.388) is far below the target (0.804), so we should increase correlation with the smallest safe changes that keep your core “SentenceTransformer encode → cosine → calibration → submission” pipeline intact. The biggest issue in your current feature is that you’re taking cosine between two *different pair-order encodings* (`[CPC] anchor [SEP] target` vs `[CPC] target [SEP] anchor`), which measures order-invariance rather than anchor-target similarity and can collapse signal; we switch to the standard bi-encoder cosine between separately encoded anchor and target (with the same context prefix) while keeping the same model and calibration stages. Additionally, since Pearson is invariant to affine transforms, we replace mean/std matching with a least-squares affine fit (with optional clipping) to better align predictions to labels without changing semantics. Everything else (offline robustness, model candidates, OOF piecewise-linear calibrator, output CSV format/path) remains the same.'
- What this solution (achieved 0.58287) has done: 'We keep your exact core pipeline (SentenceTransformer bi-encoder embeddings → cosine → piecewise-linear calibrator → affine fit → submission) but fix two likely score limiters with minimal risk: (1) build texts in a way that better matches typical ST training by separating context from the sentence pair using a `[SEP]` token-like delimiter, and (2) remove the unnecessary `[-1,1] → [0,1]` remap before calibration (cosine is already a good similarity proxy; the extra remap can compress useful variation and make calibration harder). We still clip final predictions to `[0,1]` and keep the same OOF calibration scheme and output format. These changes should increase Pearson from ~0.58 toward your ~0.80 target without changing the overall modeling approach.'
- What this solution (achieved 0.54027) has done: 'You’re far below the target (0.58287 vs 0.8039; higher is better), so we should increase Pearson with minimal, metric-aligned tweaks while keeping your exact “SentenceTransformer bi-encoder cosine → calibrate → affine → submission” pipeline. The biggest low-risk gain is to build the input strings in a way that better matches common ST training data: use a clean natural-language prefix with an explicit separator token and avoid repeating “context:” boilerplate that can dominate short phrases. Next, we keep your calibrators but fix the piecewise x-grid bounds to the *actual* cosine range observed in train (instead of hardcoding [-1,1]), which typically improves interpolation fidelity and correlation. Finally, we make the KFold calibration deterministic and slightly more stable by using StratifiedKFold on the discrete label levels (same CV idea, but better-balanced folds for this dataset’s discrete targets), without changing the modeling approach.'
- What this solution (achieved 0.59361) has done: 'Your current gap to target is large (0.54027 vs 0.80390; higher is better), so we should boost Pearson with minimal, metric-aligned tweaks while keeping your exact “SentenceTransformer bi-encoder cosine → piecewise-linear calibration → affine fit → submission” pipeline. The lowest-risk improvement is to strengthen the text construction so the embedding model sees the *same* CPC context and a clean, natural pair structure without extra “anchor:/target:” boilerplate that can drown short phrases. Concretely, we build two texts per row as `CPC <context> [SEP] <anchor>` and `CPC <context> [SEP] <target>` (same structure), which typically increases cosine signal quality. Everything else (model, cosine, StratifiedKFold OOF calibration, affine LS, clipping, and submission format/path) stays the same.'

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
def st_anchor_target_texts(df: pd.DataFrame):
    ctx = df["context"].astype(str)
    ctx_prefix = "CPC " + ctx + " [SEP] "

    anchor_texts = (ctx_prefix + df["anchor"].astype(str)).tolist()
    target_texts = (ctx_prefix + df["target"].astype(str)).tolist()
    return anchor_texts, target_texts


def cosine01_from_st(model, df: pd.DataFrame, batch_size: int = 64):
    anchor_texts, target_texts = st_anchor_target_texts(df)

    emb_a = model.encode(
        anchor_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    emb_t = model.encode(
        target_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    cos = np.sum(emb_a * emb_t, axis=1)  # cosine similarity
    return cos.astype(np.float32)


def fit_piecewise_linear_calibrator(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Piecewise monotonic mapping from prediction space to label levels.
    (kept same core calibration logic; only makes x-bounds data-driven for better interpolation)
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
    eps = 1e-6
    for i in range(1, len(means)):
        if means[i] <= means[i - 1]:
            means[i] = means[i - 1] + eps  # keep strict monotonicity in pred-space

    lo = float(np.min(y_pred))
    hi = float(np.max(y_pred))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = -1.0, 1.0

    x = np.concatenate(([lo], means, [hi])).astype(float)
    y = np.concatenate(([0.0], levels, [1.0])).astype(float)

    for i in range(1, len(x)):
        if x[i] < x[i - 1]:
            x[i] = x[i - 1]

    return x, y


def apply_piecewise_linear(y_pred: np.ndarray, x: np.ndarray, y: np.ndarray):
    y_cal = np.interp(y_pred.astype(float), x.astype(float), y.astype(float))
    return np.clip(y_cal, 0.0, 1.0)


def fit_affine_least_squares(y_true: np.ndarray, y_pred: np.ndarray):
    yt = y_true.astype(float)
    yp = y_pred.astype(float)
    x = yp - float(np.mean(yp))
    y = yt - float(np.mean(yt))
    denom = float(np.dot(x, x)) + 1e-12
    a = float(np.dot(x, y) / denom)
    b = float(np.mean(yt) - a * np.mean(yp))
    return a, b


def apply_affine(y_pred: np.ndarray, a: float, b: float):
    return np.clip(a * y_pred.astype(float) + b, 0.0, 1.0)




## === cell 5
from sklearn.model_selection import StratifiedKFold

preds = None

raw_test = cosine01_from_st(st_model, test_df, batch_size=64)
raw_train = cosine01_from_st(st_model, train_df, batch_size=64)
y_train = train_df["score"].values.astype(float)

y_strat = (y_train * 4).round().astype(int)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_cal = np.zeros_like(raw_train, dtype=float)

for fold, (tr_idx, va_idx) in enumerate(skf.split(raw_train, y_strat), start=1):
    x_cal, y_cal = fit_piecewise_linear_calibrator(y_train[tr_idx], raw_train[tr_idx])
    tr_cal = apply_piecewise_linear(raw_train[tr_idx], x_cal, y_cal)
    va_cal = apply_piecewise_linear(raw_train[va_idx], x_cal, y_cal)

    a, b = fit_affine_least_squares(y_train[tr_idx], tr_cal)
    oof_cal[va_idx] = apply_affine(va_cal, a, b)
    print(f"Fold {fold}: done")

print(
    "OOF cal stats:",
    float(oof_cal.min()),
    float(oof_cal.max()),
    "mean:",
    float(oof_cal.mean()),
)

x_full, y_full = fit_piecewise_linear_calibrator(y_train, raw_train)
train_cal_full = apply_piecewise_linear(raw_train, x_full, y_full)
test_cal_full = apply_piecewise_linear(raw_test, x_full, y_full)

a_full, b_full = fit_affine_least_squares(y_train, train_cal_full)
preds = apply_affine(test_cal_full, a_full, b_full).astype(float)

print("Preds:", preds.shape, "min/max:", float(preds.min()), float(preds.max()))
print("Calibrator x (pred-space):", x_full)
print("Calibrator y (label-space):", y_full)
print("Affine a,b:", a_full, b_full)



## === cell 6
submission = pd.DataFrame(
    {"id": test_df["id"].astype(str).values, "score": preds.astype(float)}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
print("submission.csv exists:", os.path.exists("submission.csv"))
