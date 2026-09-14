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

0.67285

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker by making the Hugging Face loader treat the provided `/kaggle/input/...` paths as local directories (the current error happens because the hub validators interpret the absolute path as an invalid repo id). Then I add a safe fallback: if that local model directory doesn’t exist in your environment, the script still run end-to-end by producing a valid submission using a simple constant prediction (so you always get a `.csv`). Finally, I ensure a `score` column is always created, cast to int, clipped to the valid 1–6 range, and written with the exact required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from a modeling bug: you’re loading a causal LM and then applying a `Linear(1,1)` head to the last-token logits, which can’t produce a 6-class prediction (so it either errors or degenerates to constant/invalid behavior). To move your score toward the 0.775 target with minimal semantic change, I keep your overall “load local HF artifacts if available, else constant fallback” structure, but fix the head to be a proper 6-class classifier applied to a scalar feature derived from the model outputs (mean logit over vocab at last token). I also vectorize inference with a DataLoader-style batch loop so it finishes within the time limit and avoid per-row GPU cache clearing (which slows and can destabilize). The fallback constant submission remains intact so the notebook always produces a valid `submission.csv`.'
- What this solution (achieved 0.67285) has done: 'Your current 0.0 score is consistent with producing essentially uninformative/invalid predictions (constant `3` fallback or a head/model mismatch), so the smallest safe move toward the 0.7756 target is to make the script always produce a non-constant, label-aligned prediction while keeping your “local HF artifacts if available, else fallback” structure. I keep your existing causal-LM + simple head approach, but fix two likely score-killers: (1) ensure tokenization includes special tokens (so the model sees a sensible input format) and (2) replace the constant fallback with a deterministic, data-driven baseline (length-based mapping) that is legitimate and usually scores well above 0.0 on QWK. I also enforce that predictions match the required 1–6 integer scale and that the submission rows exactly align to `essay_id`. These are minimal changes that should increase score substantially without changing your core loading/inference design.'

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
                "WARNING: classification_head.pth not found; will fall back to non-constant baseline predictions."
            )
    except Exception as e:
        print(
            "WARNING: Failed to load local model/tokenizer/head; will fall back to non-constant baseline predictions."
        )
        print("Load error:", repr(e))
else:
    print(
        "WARNING: Local model directory not available; will fall back to non-constant baseline predictions."
    )



## === cell 5
if "score" not in test.columns:
    test["score"] = np.nan




## === cell 6
def build_length_bins(
    train_df: pd.DataFrame, text_col: str = "full_text", y_col: str = "score"
):
    tmp = train_df[[text_col, y_col]].copy()
    tmp[text_col] = tmp[text_col].astype(str)
    tmp["n_chars"] = tmp[text_col].str.len().astype(np.int32)
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce").astype("Int64")
    tmp = tmp.dropna(subset=[y_col])

    med = tmp.groupby(y_col)["n_chars"].median().sort_index()

    full_idx = pd.Index(range(1, 7), dtype=int)
    med = med.reindex(full_idx)
    med = med.interpolate(limit_direction="both").ffill().bfill()

    thresholds = []
    for s in range(1, 6):
        thresholds.append(0.5 * (float(med.loc[s]) + float(med.loc[s + 1])))
    return np.array(thresholds, dtype=np.float32)


def predict_from_length(texts, thresholds: np.ndarray):
    n_chars = (
        pd.Series(texts, dtype="string").fillna("").str.len().to_numpy(dtype=np.int32)
    )
    return (
        np.digitize(n_chars.astype(np.float32), thresholds, right=False) + 1
    ).astype(np.int32)


length_thresholds = build_length_bins(train, "full_text", "score")
print("Length thresholds (chars) between classes:", length_thresholds.tolist())



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
    test["score"] = predict_from_length(
        test["full_text"].astype(str).tolist(), length_thresholds
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
