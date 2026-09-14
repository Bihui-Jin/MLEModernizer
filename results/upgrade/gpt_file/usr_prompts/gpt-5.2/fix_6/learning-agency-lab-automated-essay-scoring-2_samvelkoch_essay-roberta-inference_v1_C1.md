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

0.7756038481842737

# 6. Current score

0.63658

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker by making the Hugging Face loader treat the provided `/kaggle/input/...` paths as local directories (the current error happens because the hub validators interpret the absolute path as an invalid repo id). Then I add a safe fallback: if that local model directory doesn’t exist in your environment, the script still run end-to-end by producing a valid submission using a simple constant prediction (so you always get a `.csv`). Finally, I ensure a `score` column is always created, cast to int, clipped to the valid 1–6 range, and written with the exact required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from a modeling bug: you’re loading a causal LM and then applying a `Linear(1,1)` head to the last-token logits, which can’t produce a 6-class prediction (so it either errors or degenerates to constant/invalid behavior). To move your score toward the 0.775 target with minimal semantic change, I keep your overall “load local HF artifacts if available, else constant fallback” structure, but fix the head to be a proper 6-class classifier applied to a scalar feature derived from the model outputs (mean logit over vocab at last token). I also vectorize inference with a DataLoader-style batch loop so it finishes within the time limit and avoid per-row GPU cache clearing (which slows and can destabilize). The fallback constant submission remains intact so the notebook always produces a valid `submission.csv`.'
- What this solution (achieved 0.67285) has done: 'Your current 0.0 score is consistent with producing essentially uninformative/invalid predictions (constant `3` fallback or a head/model mismatch), so the smallest safe move toward the 0.7756 target is to make the script always produce a non-constant, label-aligned prediction while keeping your “local HF artifacts if available, else fallback” structure. I keep your existing causal-LM + simple head approach, but fix two likely score-killers: (1) ensure tokenization includes special tokens (so the model sees a sensible input format) and (2) replace the constant fallback with a deterministic, data-driven baseline (length-based mapping) that is legitimate and usually scores well above 0.0 on QWK. I also enforce that predictions match the required 1–6 integer scale and that the submission rows exactly align to `essay_id`. These are minimal changes that should increase score substantially without changing your core loading/inference design.'
- What this solution (achieved 0.67062) has done: 'To move your QWK from 0.67285 toward the 0.7756 target (higher is better), I keep your exact overall structure (local HF model+head if available, else length-based baseline), but make the fallback baseline slightly more label-aligned by using word counts (more robust than raw character counts) and by calibrating the binning with a train-fitted monotonic mapping from the continuous proxy to the 1–6 classes. These are small, legitimate changes that usually improve ranking/ordinal agreement without changing your “no training / no fine-tuning” approach. I also make the model-path branch more robust by ensuring a pad token is set (common for causal LMs) to avoid padding-related weirdness, which can silently hurt predictions if the local model exists. Submission writing, columns, clipping to 1–6, and file path remain unchanged.'
- What this solution (achieved 0.63658) has done: 'Your current score (0.67062) is below the target (0.77560), so we should make a small, legitimate improvement that better aligns the ordinal predictions with QWK without changing your overall “use local HF model if present, else proxy baseline” design. The safest lever is the proxy baseline: keep word-count as the single feature, but replace the median/quantile threshold heuristic with train-fitted ordinal thresholds that explicitly maximize Quadratic Weighted Kappa on the training distribution. This preserves your core logic (proxy → thresholds → digitize into 1–6) while directly optimizing for the competition metric, and it stays fast. The model branch is left intact except for keeping output handling unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch



## === cell 1
print("CUDA available:", torch.cuda.is_available())
print("CUDA device_count:", torch.cuda.device_count())



## === cell 2
DATA_DIR = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
test_path = os.path.join(DATA_DIR, "test.csv")
train_path = os.path.join(DATA_DIR, "train.csv")

test = pd.read_csv(test_path)
train = pd.read_csv(train_path)

print("Loaded test:", test.shape, test.columns.tolist())
print("Loaded train:", train.shape, train.columns.tolist())



## === cell 3
reverse_score_mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}



## === cell 4
from transformers import AutoModelForCausalLM, AutoTokenizer

local_model_path = "/kaggle/input/hf-proxy-roberta"
local_tokenizer_path = "/kaggle/input/hf-proxy-roberta"

have_local_model = os.path.isdir(local_model_path) and os.path.isdir(
    local_tokenizer_path
)
print("Local model dir exists:", have_local_model, "| path:", local_model_path)

tokenizer = None
model = None
head = None
use_model = False
device = "cuda" if torch.cuda.is_available() else "cpu"

if have_local_model:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            local_tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.sep_token_id is not None:
                tokenizer.pad_token = tokenizer.sep_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        model = (
            AutoModelForCausalLM.from_pretrained(
                local_model_path,
                torch_dtype=torch.float32,
                device_map=None,  # explicit device control
                trust_remote_code=True,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )

        head_weights_path = os.path.join(local_model_path, "classification_head.pth")

        if os.path.isfile(head_weights_path):
            head_weights = torch.load(head_weights_path, map_location=device)

            head = torch.nn.Linear(1, 6, bias=True).to(device).eval()

            if isinstance(head_weights, dict):
                if "weight" in head_weights and "bias" in head_weights:
                    head.weight.data = head_weights["weight"].to(device)
                    head.bias.data = head_weights["bias"].to(device)
                elif "state_dict" in head_weights:
                    head.load_state_dict(head_weights["state_dict"])
                else:
                    try:
                        head.load_state_dict(head_weights)
                    except Exception:
                        raise ValueError(
                            f"Unrecognized head weight dict keys: {list(head_weights.keys())[:10]}"
                        )
            elif torch.is_tensor(head_weights):
                w = head_weights.to(device)
                if w.numel() == 6:
                    head.weight.data = w.view(6, 1)
                    head.bias.data.zero_()
                elif w.shape == head.weight.data.shape:
                    head.weight.data = w
                    head.bias.data.zero_()
                else:
                    raise ValueError(
                        f"Unexpected tensor shape for head weights: {tuple(w.shape)}"
                    )
            else:
                raise ValueError(
                    f"Unexpected type for head weights: {type(head_weights)}"
                )

            use_model = True
            print("Hooray! Model/tokenizer/head loaded.")
        else:
            print(
                "WARNING: classification_head.pth not found; will fall back to baseline predictions."
            )
    except Exception as e:
        print(
            "WARNING: Failed to load local model/tokenizer/head; will fall back to baseline predictions."
        )
        print("Load error:", repr(e))
else:
    print(
        "WARNING: Local model directory not available; will fall back to baseline predictions."
    )



## === cell 5
if "score" not in test.columns:
    test["score"] = np.nan




## === cell 6
def _word_count(series: pd.Series) -> np.ndarray:
    s = series.astype("string").fillna("")
    return s.str.split().str.len().to_numpy(dtype=np.int32)


def qwk(
    y_true: np.ndarray, y_pred: np.ndarray, min_rating: int = 1, max_rating: int = 6
) -> float:
    """
    Fast quadratic weighted kappa for integer ratings in [min_rating, max_rating].
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    y_true = np.clip(y_true, min_rating, max_rating)
    y_pred = np.clip(y_pred, min_rating, max_rating)

    n = max_rating - min_rating + 1
    O = np.zeros((n, n), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        O[a - min_rating, b - min_rating] += 1.0

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E / E.sum() * O.sum()

    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            W[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - (num / den)


def predict_from_proxy(texts, thresholds: np.ndarray) -> np.ndarray:
    n_words = _word_count(pd.Series(texts, dtype="string"))
    return (
        np.digitize(n_words.astype(np.float32), thresholds, right=False) + 1
    ).astype(np.int32)


def fit_qwk_optimal_thresholds_from_train(
    train_df: pd.DataFrame,
    text_col: str = "full_text",
    y_col: str = "score",
    min_rating: int = 1,
    max_rating: int = 6,
) -> np.ndarray:
    tmp = train_df[[text_col, y_col]].copy()
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce").astype("Int64")
    tmp = tmp.dropna(subset=[y_col])
    tmp[text_col] = tmp[text_col].astype("string").fillna("")

    x = _word_count(tmp[text_col]).astype(np.float32)
    y = tmp[y_col].astype(int).to_numpy(dtype=np.int32)

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    n = len(y)

    K = max_rating - min_rating + 1  # 6

    y_idx = y - min_rating
    prefix = np.zeros((n + 1, K), dtype=np.int32)
    for i in range(n):
        prefix[i + 1] = prefix[i]
        prefix[i + 1, y_idx[i]] += 1

    def seg_hist(l: int, r: int) -> np.ndarray:
        return prefix[r] - prefix[l]

    min_seg = max(50, n // 2000)  # keep segments non-trivial; deterministic
    dp = -(10**18) * np.ones((K + 1, n + 1), dtype=np.int64)
    prev = -np.ones((K + 1, n + 1), dtype=np.int32)
    dp[0, 0] = 0

    for c in range(1, K + 1):
        label_idx = c - 1
        for r in range(c * min_seg, n + 1):
            best_val = -(10**18)
            best_k = -1
            k_min = (c - 1) * min_seg
            k_max = r - min_seg
            if k_max < k_min:
                continue
            target_k = int(r * (c - 1) / c)
            window = max(500, n // 200)  # deterministic
            a = max(k_min, target_k - window)
            b = min(k_max, target_k + window)
            candidates = list(range(a, b + 1))
            if a != k_min:
                candidates.append(k_min)
            if b != k_max:
                candidates.append(k_max)
            candidates = sorted(set(candidates))

            for k in candidates:
                if dp[c - 1, k] <= -(10**17):
                    continue
                reward = int(seg_hist(k, r)[label_idx])
                val = dp[c - 1, k] + reward
                if val > best_val:
                    best_val = val
                    best_k = k
            dp[c, r] = best_val
            prev[c, r] = best_k

    cuts = [n]
    r = n
    for c in range(K, 0, -1):
        k = int(prev[c, r])
        if k < 0:
            qs = [c / K for c in range(1, K)]
            return np.quantile(x, qs).astype(np.float32)
        cuts.append(k)
        r = k
    cuts = list(reversed(cuts))  # length K+1, from 0..n

    thresholds = []
    for c in range(1, K):
        left_end = cuts[c]
        if left_end <= 0:
            t = float(x[0])
        elif left_end >= n:
            t = float(x[-1])
        else:
            t = 0.5 * (float(x[left_end - 1]) + float(x[left_end]))
        thresholds.append(t)

    thresholds = np.array(thresholds, dtype=np.float32)

    for i in range(1, len(thresholds)):
        if thresholds[i] <= thresholds[i - 1]:
            thresholds[i] = thresholds[i - 1] + 1.0

    x_all = x  # sorted
    y_all = y

    def apply_thresholds_sorted(x_sorted: np.ndarray, thr: np.ndarray) -> np.ndarray:
        return (np.digitize(x_sorted, thr, right=False) + min_rating).astype(np.int32)

    base_pred = apply_thresholds_sorted(x_all, thresholds)
    base_qwk = qwk(y_all, base_pred, min_rating, max_rating)

    uniq_x = np.unique(x_all)
    if len(uniq_x) > 0:
        for i in range(len(thresholds)):
            cur = thresholds[i]
            pos = np.searchsorted(uniq_x, cur)
            idxs = np.clip(
                np.array([pos - 8, pos - 4, pos - 2, pos, pos + 2, pos + 4, pos + 8]),
                0,
                len(uniq_x) - 1,
            )
            cands = []
            for j in idxs:
                if j < len(uniq_x) - 1:
                    cands.append(0.5 * (float(uniq_x[j]) + float(uniq_x[j + 1])))
                else:
                    cands.append(float(uniq_x[j]))
            cands = sorted(set(cands))

            best_t = cur
            best_q = base_qwk
            for t in cands:
                thr2 = thresholds.copy()
                thr2[i] = float(t)
                for k in range(1, len(thr2)):
                    if thr2[k] <= thr2[k - 1]:
                        thr2[k] = thr2[k - 1] + 1.0
                for k in range(len(thr2) - 2, -1, -1):
                    if thr2[k] >= thr2[k + 1]:
                        thr2[k] = thr2[k + 1] - 1.0

                pred2 = apply_thresholds_sorted(x_all, thr2)
                q2 = qwk(y_all, pred2, min_rating, max_rating)
                if q2 > best_q:
                    best_q = q2
                    best_t = float(t)

            if best_q > base_qwk:
                thresholds[i] = best_t
                for k in range(1, len(thresholds)):
                    if thresholds[k] <= thresholds[k - 1]:
                        thresholds[k] = thresholds[k - 1] + 1.0
                base_qwk = best_q

    final_pred = apply_thresholds_sorted(x_all, thresholds)
    print(
        "Train proxy-threshold QWK (diagnostic):",
        float(qwk(y_all, final_pred, min_rating, max_rating)),
    )

    return thresholds


proxy_thresholds = fit_qwk_optimal_thresholds_from_train(train, "full_text", "score")
print("Proxy thresholds (word counts) between classes:", proxy_thresholds.tolist())



## === cell 7
max_length = 512

if use_model:
    batch_size = 8 if device == "cuda" else 2

    texts = test["full_text"].astype(str).tolist()
    pred_scores = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            out = model(**inputs).logits  # [B, T, V]
            last_token_logits = out[:, -1, :]  # [B, V]
            feat = last_token_logits.mean(dim=1, keepdim=True)  # [B, 1]

            cls_logits = head(feat)  # [B, 6]
            pred_class = torch.argmax(cls_logits, dim=1).detach().cpu().numpy()  # 0..5

            for c in pred_class.tolist():
                pred_scores.append(reverse_score_mapping.get(int(c), 3))

    test["score"] = pred_scores
else:
    test["score"] = predict_from_proxy(
        test["full_text"].astype(str).tolist(), proxy_thresholds
    )



## === cell 8
test["score"] = (
    pd.to_numeric(test["score"], errors="coerce").fillna(3).round().astype(int)
)
test["score"] = test["score"].clip(1, 6)

print(test[["essay_id", "score"]].head())



## === cell 9
sub = test[["essay_id", "score"]].copy()
sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print("Saved:", sub_path, "| shape:", sub.shape)
print(sub.head())
print("Score distribution:", sub["score"].value_counts().sort_index().to_dict())
