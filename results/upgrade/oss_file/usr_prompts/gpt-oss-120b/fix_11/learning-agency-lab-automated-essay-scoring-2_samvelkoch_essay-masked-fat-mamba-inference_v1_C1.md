# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import torch
import pandas as pd
import numpy as np

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:
    print(f"Transformers import failed ({e}); proceeding with fallback predictor.")
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print(f"SentenceTransformer import failed ({e}); embeddings will be disabled.")
    SentenceTransformer = None

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score




## === cell 1
train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)




## === cell 2
reverse_score_mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}




## === cell 3
local_model_path_part_1 = "/kaggle/input/hf-proxy-d-trick-part-1-masked-fat-mamba"
local_model_path_part_2 = "/kaggle/input/hf-proxy-d-trick-part-2-masked-fat-mamba"
local_tokenizer_path = "/kaggle/input/hf-proxy-d-trick-part-1-masked-fat-mamba"

combined_model_path = "/kaggle/working/combined_model"
os.makedirs(combined_model_path, exist_ok=True)

try:
    for file_name in os.listdir(local_model_path_part_1):
        if not file_name.endswith((".safetensors", ".json", ".bin", ".py")):
            continue
        full_file_name = os.path.join(local_model_path_part_1, file_name)
        symlink_name = os.path.join(combined_model_path, file_name)
        if os.path.isfile(full_file_name) and not os.path.exists(symlink_name):
            os.symlink(full_file_name, symlink_name)

    for file_name in os.listdir(local_model_path_part_2):
        if not file_name.endswith(".safetensors"):
            continue
        full_file_name = os.path.join(local_model_path_part_2, file_name)
        symlink_name = os.path.join(combined_model_path, file_name)
        if os.path.isfile(full_file_name) and not os.path.exists(symlink_name):
            os.symlink(full_file_name, symlink_name)

    tokenizer = AutoTokenizer.from_pretrained(
        local_tokenizer_path,
        trust_remote_code=True,
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            combined_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        .cuda()
        .eval()
    )

    head_weights_path = os.path.join(local_model_path_part_1, "classification_head.pth")
    head_weights = torch.load(head_weights_path, map_location="cuda")
    head = torch.nn.Linear(1, 1, bias=False).to("cuda")
    head.weight.data = head_weights
    print("Model loaded successfully.")
except Exception as e:
    print(f"Could not load custom model ({e}); using fallback predictor.")
    tokenizer = None
    model = None
    head = None




## === cell 4
fallback_mean_score = int(round(train["score"].mean()))
fallback_mean_score = max(1, min(6, fallback_mean_score))  # ensure within 1‑6




## === cell 5
embedder_device = "cuda" if torch.cuda.is_available() else "cpu"

if SentenceTransformer is not None and embedder_device == "cuda":
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=embedder_device)

    train_texts = train["full_text"].astype(str).tolist()
    train_embeddings = embedder.encode(
        train_texts,
        batch_size=256,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    train_word_counts = (
        train["full_text"].astype(str).str.split().str.len().values.reshape(-1, 1)
    )
    train_char_counts = train["full_text"].astype(str).str.len().values.reshape(-1, 1)

    train_X_full = np.hstack([train_embeddings, train_word_counts, train_char_counts])
    train_y = train["score"].values

    X_tr, X_val, y_tr, y_val = train_test_split(
        train_X_full, train_y, test_size=0.2, random_state=42, stratify=train_y
    )

    ridge_regressor = Ridge(alpha=1.0, random_state=42)
    ridge_regressor.fit(X_tr, y_tr)

    gbr_regressor = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    gbr_regressor.fit(X_tr, y_tr)

    def _qwk(model, X, y_true):
        preds = np.rint(model.predict(X)).astype(int)
        preds = np.clip(preds, 1, 6)
        return cohen_kappa_score(y_true, preds, weights="quadratic")

    ridge_qwk = _qwk(ridge_regressor, X_val, y_val)
    gbr_qwk = _qwk(gbr_regressor, X_val, y_val)
    print(f"Validation QWK – Ridge: {ridge_qwk:.4f}, GBR: {gbr_qwk:.4f}")

    best_regressor = ridge_regressor if ridge_qwk >= gbr_qwk else gbr_regressor
    best_regressor.fit(train_X_full, train_y)  # refit on full data
else:
    embedder = None
    best_regressor = None




## === cell 6
if tokenizer is not None and model is not None and head is not None:
    batch_size = 64  # fits comfortably in GPU memory for most essays
    test_indices = test.index.tolist()

    for start in range(0, len(test_indices), batch_size):
        batch_idx = test_indices[start : start + batch_size]
        texts = test.loc[batch_idx, "full_text"].astype(str).tolist()

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        ).to("cuda")

        with torch.no_grad():
            out = model(**inputs).logits  # (B, S, V)
            logits = head(out[:, -1])  # (B, 1)
            pred_idxs = torch.argmax(logits, dim=1).cpu().numpy()

        predicted_labels = [reverse_score_mapping[int(idx)] for idx in pred_idxs]

        del inputs, out, logits
        torch.cuda.empty_cache()

        test.loc[batch_idx, "score"] = predicted_labels

else:
    if embedder is not None and best_regressor is not None:
        test_texts = test["full_text"].astype(str).tolist()
        test_embeddings = embedder.encode(
            test_texts,
            batch_size=256,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        test_word_counts = (
            test["full_text"].astype(str).str.split().str.len().values.reshape(-1, 1)
        )
        test_char_counts = test["full_text"].astype(str).str.len().values.reshape(-1, 1)
        test_X_full = np.hstack([test_embeddings, test_word_counts, test_char_counts])

        pred_vals = best_regressor.predict(test_X_full)
        predicted_labels = np.clip(np.rint(pred_vals).astype(int), 1, 6)
        test["score"] = predicted_labels
    else:
        test["score"] = fallback_mean_score




## === cell 7
test["score"] = test["score"].astype(int)
submission_path = "submission.csv"
test[["essay_id", "score"]].to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
