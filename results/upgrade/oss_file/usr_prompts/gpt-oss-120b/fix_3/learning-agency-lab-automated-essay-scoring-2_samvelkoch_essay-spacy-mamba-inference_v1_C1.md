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

0.3970480505992886

# 6. Current score

0.53328

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I add error‑handling around the model loading so the script no longer crashes when the model files are missing, and fall back to a simple baseline prediction (the most common score from the training set). This ensures a `submission.csv` is always created with the required columns, letting the pipeline run end‑to‑end and produce a valid submission file.'
- What this solution (achieved 0.53328) has done: 'I keep the existing model‑loading code but add a cheap length‑based fallback that usually outperforms the single‑class baseline. After reading the data I compute each essay’s character length, map lengths to the average training score, and use these values (rounded‑to‑1‑6) whenever the large language model or its classification head cannot be used. This minimal change preserves the original pipeline while giving a better‑than‑zero Quadratic Weighted Kappa score, moving the result toward the target.'

# 9. Code solution

## === cell 0
import os
import torch
import pandas as pd
from pathlib import Path



## === cell 1
print("CUDA devices:", torch.cuda.device_count())



## === cell 2
train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df["len"] = train_df["full_text"].astype(str).str.len()
test_df["len"] = test_df["full_text"].astype(str).str.len()

print("Train shape:", train_df.shape, "Test shape:", test_df.shape)



## === cell 3
baseline_score = int(train_df["score"].mode()[0])
print("Baseline score (most frequent training label):", baseline_score)

len_score_map = train_df.groupby("len")["score"].mean()
overall_mean = train_df["score"].mean()

test_df["len_based_pred"] = test_df["len"].map(len_score_map)
test_df["len_based_pred"].fillna(overall_mean, inplace=True)
test_df["len_based_pred"] = test_df["len_based_pred"].round().astype(int).clip(1, 6)



## === cell 4
model = None
tokenizer = None
head = None
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_model_path_part_1 = "/kaggle/input/hf-proxy-part-1-spacy-mamba"
    local_model_path_part_2 = "/kaggle/input/hf-proxy-part-2-spacy-mamba"
    local_tokenizer_path = "/kaggle/input/hf-proxy-part-1-spacy-mamba"

    combined_model_path = "/kaggle/working/combined_model"
    os.makedirs(combined_model_path, exist_ok=True)

    for base_path, extensions in [
        (local_model_path_part_1, (".safetensors", ".json", ".bin", ".py")),
        (local_model_path_part_2, (".safetensors",)),
    ]:
        if Path(base_path).exists():
            for file_name in os.listdir(base_path):
                if not file_name.endswith(extensions):
                    continue
                src = os.path.join(base_path, file_name)
                dst = os.path.join(combined_model_path, file_name)
                if os.path.isfile(src) and not os.path.exists(dst):
                    os.symlink(src, dst)

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
    if Path(head_weights_path).exists():
        head_weights = torch.load(head_weights_path, map_location="cuda")
        head = torch.nn.Linear(1, 1, bias=False).to("cuda")
        head.weight.data = head_weights
    else:
        head = None

    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print("Model loading failed or files missing – using fallback predictions.")
    print("Error:", e)



## === cell 5
reverse_score_mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}

if model is not None and tokenizer is not None and head is not None:
    predictions = []
    for idx, row in test_df.iterrows():
        text = row["full_text"]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        ).to("cuda")

        with torch.no_grad():
            out = model(**inputs).logits
            logits = head(out[:, -1])
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_label = reverse_score_mapping.get(pred_idx, baseline_score)

        predictions.append(pred_label)

        del inputs, out
        torch.cuda.empty_cache()
    test_df["score"] = predictions
else:
    test_df["score"] = test_df["len_based_pred"]



## === cell 6
test_df["score"] = test_df["score"].astype(int)
print("Sample predictions:", test_df[["essay_id", "score"]].head())



## === cell 7
submission_path = "submission.csv"
test_df[["essay_id", "score"]].to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
