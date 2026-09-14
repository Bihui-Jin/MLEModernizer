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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

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
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.7871644009533957

# 6. Current score

0.56154

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker by replacing the missing local model path (`/kaggle/input/aes-model-vhhv-2`) with a robust fallback that loads a public Hugging Face checkpoint (`microsoft/deberta-v3-base`) already available in the installed `transformers` stack. Since that changes the pipeline from “inference with a provided fine-tuned classifier” to “inference without a fine-tuned classifier,” I keep the rest of your core inference logic intact but add a safe, deterministic, score-neutral fallback that outputs a valid 1–6 score for every test row. Finally, I ensure the submission length always matches the test set and the file is written as `submission.csv` with the required columns.'
- What this solution (achieved 0.56154) has done: 'I fix the runtime blocker by making model loading robust in this offline Kaggle environment: first try a local fine-tuned directory if present, otherwise use a local backbone if available, and if neither exists fall back to a deterministic, score-neutral CPU-only heuristic so the notebook always completes. I also fix the test file path to the correct competition location and ensure `tokenizer/backbone` are always defined (or inference is skipped safely) so later cells don’t crash. Finally, I guarantee the submission has exactly the required `essay_id,score` columns, matches the test row count, uses integer scores clipped to 1–6, and is written to `submission.csv`.'
- What this solution (achieved 0.56154) has done: 'Your current score is far below the target, and the main reason is that the Transformer path is effectively untrained (a zero-initialized linear head), so predictions are mostly driven by the length heuristic. To move the score toward the target with minimal changes and without altering the core architecture/training paradigm, I keep your exact backbone + CLS pooling + linear head, but fit that linear regression head on the provided training set embeddings (frozen backbone) using a closed-form ridge solution. Then I keep your existing length-based blending and quantile binning, but tune the blend weight slightly toward the now-informative model signal to improve QWK. All I/O paths and the submission format remain unchanged, and it still run offline and produce `submission.csv`.'
- What this solution (achieved 0.56154) has done: 'Your current gap to target is large (0.56154 → 0.78716), and the biggest limiter is that the linear ridge head is being fit on *raw* CLS embeddings while the final predictions are later z-scored and heavily dominated by the length heuristic. I keep the exact same backbone + CLS pooling + linear head + closed-form ridge training, but (1) standardize the embedding features before fitting ridge (and apply the same transform at test time) to make the regression signal materially stronger, and (2) slightly shift the blend weight toward the model signal so the improved regressor actually influences the final quantile binning. These are minimal, metric-relevant changes that preserve your core approach and keep the same submission semantics (still outputs integer 1–6 via quantile binning). The code still runs offline, keeps paths unchanged, and writes a valid `submission.csv`.'
- What this solution (achieved 0.56154) has done: 'Your score is far below the target, so we should increase it with minimal, metric-relevant changes while preserving your core pipeline (frozen Transformer embeddings → closed-form ridge head → blend with length → quantile binning to 1–6). The biggest issue is that your final quantile binning uses *test-only* quantiles, which doesn’t align predictions to the train score distribution (hurting QWK); we instead compute bin edges from the **training** combined signal and apply them to test. Additionally, we blend in a small “rank mapping” calibration learned on train (map combined rank → expected score), which keeps the same semantics (still outputs 1–6) but aligns ordering to the training labels better. These changes are lightweight, deterministic, offline-safe, and keep your architecture/training approach intact.'
- What this solution (achieved 0.56154) has done: 'Your current score (0.56154) is far below the target (0.78716), so we should make a small, metric-relevant change that increases QWK without changing your backbone/CLS/ridge/blend/binning core pipeline. The biggest weakness now is that you’re training the ridge head on the full training set and then using that same in-sample signal for calibration/bin edges; switching to out-of-fold (OOF) predictions for the training-side calibration makes the learned mapping and bins much better aligned to generalization, typically improving QWK. I keep the exact same feature extraction (CLS), ridge closed-form, length blending, rank-mapping calibration, and quantile binning—only changing how the train predictions used for calibration are produced (OOF instead of in-sample), and then fitting a final ridge on full train for test inference. This should move the score upward toward the target while remaining deterministic and offline-safe.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
DATA_DIR = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

df_test = pd.read_csv(TEST_PATH)
df_test.head()



## === cell 2
repository_id = "/kaggle/input/aes-model-vhhv-2"



## === cell 3
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 4
def find_hf_model_dir(root_dir: str) -> str:
    """
    Locate a local HuggingFace model directory (containing config.json).
    Returns absolute path if found, otherwise raises FileNotFoundError.
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(
            f"Model directory not found: {root_dir}. "
            f"Available dirs under /kaggle/input: {sorted(os.listdir('/kaggle/input'))[:50]}"
        )

    if os.path.isfile(os.path.join(root_dir, "config.json")):
        return os.path.abspath(root_dir)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "config.json" in filenames:
            return os.path.abspath(dirpath)

    raise FileNotFoundError(f"No config.json found under: {root_dir}")


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

local_backbone_dir = os.path.join(DATA_DIR, "deberta-v3-base")

tokenizer = None
backbone = None
backbone_config = None
hidden_size = 768
reg_head = None
USE_MODEL = False

try:
    model_dir = find_hf_model_dir(repository_id)
    print("Using local fine-tuned model_dir:", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    backbone_config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    backbone = AutoModel.from_pretrained(model_dir, local_files_only=True)
    USE_MODEL = True
except Exception as e1:
    print("WARNING (fine-tuned model not found):", str(e1))
    try:
        model_dir = find_hf_model_dir(local_backbone_dir)
        print("Using local backbone model_dir:", model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        backbone_config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        backbone = AutoModel.from_pretrained(model_dir, local_files_only=True)
        USE_MODEL = True
    except Exception as e2:
        print("WARNING (local backbone not found):", str(e2))
        print(
            "Falling back to deterministic heuristic predictions (no Transformer inference)."
        )
        USE_MODEL = False

if USE_MODEL:
    backbone.to(device)
    backbone.eval()

    hidden_size = getattr(backbone_config, "hidden_size", None)
    if hidden_size is None:
        dim_val = getattr(backbone_config, "dim", None)
        hidden_size = int(dim_val) if dim_val is not None else 768

    print("Backbone hidden_size:", hidden_size)

    reg_head = torch.nn.Linear(int(hidden_size), 1, bias=True).to(device)
    reg_head.eval()

    torch.manual_seed(0)
    with torch.no_grad():
        reg_head.weight.zero_()
        reg_head.bias.fill_(0.0)



## === cell 5
test_sentences = df_test["full_text"].astype(str).tolist()
len(test_sentences), test_sentences[0][:200]




## === cell 6
def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # [B, T, 1]
    summed = (last_hidden * mask).sum(dim=1)  # [B, H]
    denom = mask.sum(dim=1).clamp_min(1.0)  # [B, 1]
    return summed / denom


def extract_features(texts, batch_size=16, max_length=512, pool="cls"):
    feats = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_text = texts[start : start + batch_size]
            enc = tokenizer(
                batch_text,
                add_special_tokens=True,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = backbone(**enc)
            last_hidden = outputs.last_hidden_state  # [B, T, H]
            if pool == "cls":
                vec = last_hidden[:, 0, :]  # [B, H]
            elif pool == "mean":
                vec = _mean_pool(last_hidden, enc.get("attention_mask"))
            else:
                raise ValueError("pool must be 'cls' or 'mean'")
            feats.append(vec.detach().cpu())
    return torch.cat(feats, dim=0)  # [N, H]


def standardize_features_fit(X: torch.Tensor):
    Xn = X.numpy().astype(np.float32, copy=False)
    mu = Xn.mean(axis=0, keepdims=True)
    sd = Xn.std(axis=0, keepdims=True)
    sd = np.maximum(sd, 1e-6)
    Xs = (Xn - mu) / sd
    return (
        Xs.astype(np.float32, copy=False),
        mu.astype(np.float32),
        sd.astype(np.float32),
    )


def standardize_features_apply(X: torch.Tensor, mu: np.ndarray, sd: np.ndarray):
    Xn = X.numpy().astype(np.float32, copy=False)
    Xs = (Xn - mu) / sd
    return Xs.astype(np.float32, copy=False)


def fit_ridge_head(X_std_np: np.ndarray, y: np.ndarray, alpha: float = 10.0):
    """
    Closed-form ridge regression with intercept by augmenting X with ones.
    X_std_np: [N, H] float32 numpy (already standardized)
    y: [N] float numpy
    """
    Xn = X_std_np.astype(np.float64, copy=False)
    yn = y.astype(np.float64, copy=False).reshape(-1, 1)

    ones = np.ones((Xn.shape[0], 1), dtype=np.float64)
    Xa = np.concatenate([Xn, ones], axis=1)  # [N, H+1]

    d = Xa.shape[1]
    I = np.eye(d, dtype=np.float64)
    I[-1, -1] = 0.0  # do not regularize intercept
    A = Xa.T @ Xa + alpha * I
    w = np.linalg.solve(A, Xa.T @ yn).reshape(-1)  # [H+1]
    return w[:-1].astype(np.float32), np.float32(w[-1])


def zscore_np(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return (x - x.mean()) / (x.std() + eps)


def rank_to_unit_interval(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(len(x), dtype=np.int64)
    if len(x) <= 1:
        return np.zeros_like(x, dtype=np.float32)
    return (ranks.astype(np.float32)) / (len(x) - 1)


def build_rank_mapping(u_train: np.ndarray, y_train: np.ndarray, n_bins: int = 200):
    u = np.clip(u_train.astype(np.float32, copy=False), 0.0, 1.0)
    y = y_train.astype(np.float32, copy=False)
    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)

    idx = np.digitize(u, edges[1:-1], right=True)  # 0..n_bins-1
    sums = np.bincount(idx, weights=y, minlength=n_bins).astype(np.float32)
    cnts = np.bincount(idx, minlength=n_bins).astype(np.float32)

    global_mean = float(y.mean())
    means = np.where(cnts > 0, sums / np.maximum(cnts, 1.0), global_mean).astype(
        np.float32
    )

    nonempty = np.where(cnts > 0)[0]
    if nonempty.size > 0:
        for i in range(n_bins):
            if cnts[i] == 0:
                j = nonempty[np.argmin(np.abs(nonempty - i))]
                means[i] = means[j]
    return edges, means


def apply_rank_mapping(
    u: np.ndarray, edges: np.ndarray, means: np.ndarray
) -> np.ndarray:
    u = np.clip(u.astype(np.float32, copy=False), 0.0, 1.0)
    idx = np.digitize(u, edges[1:-1], right=True)
    return means[idx]


def make_stratified_folds(y_int: np.ndarray, n_splits: int = 5, seed: int = 0):
    """
    Minimal dependency-free stratified folds for classes 1..6.
    Deterministic: shuffles per-class with fixed seed.
    """
    rng = np.random.RandomState(seed)
    y_int = y_int.astype(int)
    fold_ids = -np.ones(len(y_int), dtype=np.int64)
    for cls in np.unique(y_int):
        idx = np.where(y_int == cls)[0]
        rng.shuffle(idx)
        parts = np.array_split(idx, n_splits)
        for f, part in enumerate(parts):
            fold_ids[part] = f
    if (fold_ids < 0).any():
        raise RuntimeError("Fold assignment failed.")
    return fold_ids


pred_scores_cont = []
train_pred_scores_cont = (
    None  # will store OOF train continuous predictions for calibration/bins
)
batch_size = 16  # keep as-is (safe default)

if USE_MODEL:
    df_train = pd.read_csv(TRAIN_PATH, usecols=["full_text", "score"])
    train_texts = df_train["full_text"].astype(str).tolist()
    y_train = df_train["score"].astype(np.float32).values
    y_train_int = df_train["score"].astype(int).values

    fit_batch = 24 if device.type == "cuda" else 8

    X_train = extract_features(
        train_texts, batch_size=fit_batch, max_length=256, pool="cls"
    )  # torch [N,H] on CPU

    n_folds = 5
    fold_ids = make_stratified_folds(y_train_int, n_splits=n_folds, seed=0)
    train_pred_scores_cont = np.zeros(len(train_texts), dtype=np.float32)

    for f in range(n_folds):
        tr_idx = np.where(fold_ids != f)[0]
        va_idx = np.where(fold_ids == f)[0]

        X_tr = X_train[tr_idx]
        X_va = X_train[va_idx]

        X_tr_std, x_mu_f, x_sd_f = standardize_features_fit(X_tr)
        w_f, b_f = fit_ridge_head(X_tr_std, y_train[tr_idx], alpha=20.0)

        X_va_std = standardize_features_apply(X_va, x_mu_f, x_sd_f)  # np float32
        train_pred_scores_cont[va_idx] = (X_va_std @ w_f + b_f).astype(np.float32)

    train_pred_scores_cont = train_pred_scores_cont.tolist()

    X_train_std_full, x_mu, x_sd = standardize_features_fit(X_train)
    w, b = fit_ridge_head(X_train_std_full, y_train, alpha=20.0)

    with torch.no_grad():
        reg_head.weight.copy_(torch.from_numpy(w).view(1, -1).to(device))
        reg_head.bias.copy_(torch.tensor([b], device=device, dtype=reg_head.bias.dtype))

    with torch.no_grad():
        for start in range(0, len(test_sentences), batch_size):
            batch_text = test_sentences[start : start + batch_size]
            enc = tokenizer(
                batch_text,
                add_special_tokens=True,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            outputs = backbone(**enc)
            last_hidden = outputs.last_hidden_state  # [B, T, H]
            if last_hidden.ndim != 3:
                last_hidden = last_hidden.view(
                    last_hidden.shape[0], -1, last_hidden.shape[-1]
                )

            cls_vec = last_hidden[:, 0, :]  # [B, H]

            cls_vec_cpu = cls_vec.detach().cpu()
            cls_std_np = standardize_features_apply(cls_vec_cpu, x_mu, x_sd)
            cls_std = torch.from_numpy(cls_std_np).to(device)

            cont = reg_head(cls_std).squeeze(-1)  # [B]
            pred_scores_cont.extend(cont.detach().cpu().numpy().tolist())
else:
    pred_scores_cont = [0.0] * len(df_test)
    train_pred_scores_cont = None

print("Preds:", len(pred_scores_cont), "Expected:", len(df_test))
if len(pred_scores_cont) != len(df_test):
    raise RuntimeError(
        f"Prediction length mismatch: got {len(pred_scores_cont)} preds for {len(df_test)} test rows."
    )

pred_scores_cont[:5]



## === cell 7
lengths_test = df_test["full_text"].astype(str).str.len().values.astype(np.float32)
lengths_test_z = zscore_np(lengths_test)

pred_cont_test = np.array(pred_scores_cont, dtype=np.float32)
pred_cont_test_z = zscore_np(pred_cont_test)

w_model = 0.60 if USE_MODEL else 0.0
w_len = 1.0 - w_model
combined_test = w_model * pred_cont_test_z + w_len * lengths_test_z

if USE_MODEL and train_pred_scores_cont is not None:
    df_train_for_cal = pd.read_csv(TRAIN_PATH, usecols=["full_text", "score"])
    y_train = df_train_for_cal["score"].astype(np.float32).values

    lengths_train = (
        df_train_for_cal["full_text"].astype(str).str.len().values.astype(np.float32)
    )
    lengths_train_z = zscore_np(lengths_train)

    pred_cont_train = np.array(train_pred_scores_cont, dtype=np.float32)
    pred_cont_train_z = zscore_np(pred_cont_train)

    combined_train = w_model * pred_cont_train_z + w_len * lengths_train_z

    u_train = rank_to_unit_interval(combined_train)
    edges_u, means_y = build_rank_mapping(u_train, y_train, n_bins=200)

    u_test = rank_to_unit_interval(combined_test)
    cal_test = apply_rank_mapping(
        u_test, edges_u, means_y
    )  # continuous in score space ~[1,6]

    cal_test_z = zscore_np(cal_test)
    combined_for_bins_test = 0.70 * combined_test + 0.30 * cal_test_z

    cal_train = apply_rank_mapping(u_train, edges_u, means_y)
    cal_train_z = zscore_np(cal_train)
    combined_for_bins_train = 0.70 * combined_train + 0.30 * cal_train_z

    bins = np.quantile(
        combined_for_bins_train, [0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0]
    )
    for i in range(1, len(bins)):
        if bins[i] <= bins[i - 1]:
            bins[i] = bins[i - 1] + 1e-6

    pred_scores = np.digitize(combined_for_bins_test, bins[1:-1], right=True) + 1
else:
    bins = np.quantile(combined_test, [0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0])
    for i in range(1, len(bins)):
        if bins[i] <= bins[i - 1]:
            bins[i] = bins[i - 1] + 1e-6
    pred_scores = np.digitize(combined_test, bins[1:-1], right=True) + 1

pred_scores = np.clip(pred_scores, 1, 6).astype(int)

df_submit = pd.DataFrame(
    {
        "essay_id": df_test["essay_id"].astype(str).values,
        "score": pred_scores,
    }
)

if df_submit.shape[0] != df_test.shape[0]:
    raise RuntimeError(
        f"Submission rows {df_submit.shape[0]} != test rows {df_test.shape[0]}"
    )

df_submit.head()



## === cell 8
out_path = "submission.csv"
df_submit.to_csv(out_path, index=False)

print("Wrote:", out_path)
print("Shape:", df_submit.shape)
print(df_submit.columns.tolist())
print("score range:", int(df_submit["score"].min()), int(df_submit["score"].max()))
print(df_submit.head())
