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

0.8174961363497553

# 6. Current score

0.50617

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I wrap the model‑loading code in a try/except block so missing files no longer raise an error, and add a simple fallback that predicts the average training score when the model cannot be loaded. This ensures the script runs end‑to‑end, creates a “score” column in the test dataframe, and writes a valid `submission.csv` file.'
- What this solution (achieved 0.35881) has done: 'I add a lightweight sentence‑transformer embedding step and train a simple linear regression on the training essays, using it as the fallback predictor when the custom model cannot be loaded. This replaces the constant‑mean fallback with a learned mapping, which should substantially raise the quadratic weighted kappa toward the target while keeping the original model loading logic unchanged.'
- What this solution (achieved 0.46202) has done: 'I wrapped the transformers imports in a safe try/except to avoid the protobuf AttributeError that stopped the notebook, and I switched the fallback regressor from plain LinearRegression to a Ridge model (a modestly more powerful linear model) to improve the quadratic weighted kappa while keeping the original workflow unchanged. The script now runs end‑to‑end and writes a proper submission.csv.'
- What this solution (achieved 0.50617) has done: 'The changes move the sentence‑transformer embeddings onto the GPU (when available) and compute all test embeddings just once instead of repeatedly inside a loop. This removes the per‑batch encoding overhead while preserving the exact same ridge/MLP regression logic and fallback behavior. The language‑model branch is left unchanged, and all other code (paths, model loading, and final CSV writing) stays identical.'

# 9. Code solution

## === cell 0
import os
import torch
import pandas as pd
import numpy as np

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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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

if SentenceTransformer is not None:
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=embedder_device)
    train_texts = train["full_text"].astype(str).tolist()
    train_embeddings = embedder.encode(
        train_texts,
        batch_size=256,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    ridge_regressor = Ridge(alpha=1.0, random_state=42)
    ridge_regressor.fit(train_embeddings, train["score"])
    mlp_regressor = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=200,  # more iterations for better convergence
        random_state=42,
        early_stopping=False,
    )
    mlp_regressor.fit(train_embeddings, train["score"])
else:
    embedder = None
    ridge_regressor = None
    mlp_regressor = None  # will fall back to mean prediction




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
    if (
        embedder is not None
        and ridge_regressor is not None
        and mlp_regressor is not None
    ):
        test_texts = test["full_text"].astype(str).tolist()
        test_embeddings = embedder.encode(
            test_texts,
            batch_size=256,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        pred_ridge = ridge_regressor.predict(test_embeddings)
        pred_mlp = mlp_regressor.predict(test_embeddings)
        avg_pred = (pred_ridge + pred_mlp) / 2.0
        predicted_labels = np.clip(np.rint(avg_pred).astype(int), 1, 6)
        test["score"] = predicted_labels
    else:
        test["score"] = fallback_mean_score




## === cell 7
test["score"] = test["score"].astype(int)
submission_path = "submission.csv"
test[["essay_id", "score"]].to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
