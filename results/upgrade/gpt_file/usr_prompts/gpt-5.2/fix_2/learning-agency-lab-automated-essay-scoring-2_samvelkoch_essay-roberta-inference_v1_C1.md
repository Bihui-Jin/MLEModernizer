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

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker by making the Hugging Face loader treat the provided `/kaggle/input/...` paths as local directories (the current error happens because the hub validators interpret the absolute path as an invalid repo id). Then I add a safe fallback: if that local model directory doesn’t exist in your environment, the script still run end-to-end by producing a valid submission using a simple constant prediction (so you always get a `.csv`). Finally, I ensure a `score` column is always created, cast to int, clipped to the valid 1–6 range, and written with the exact required columns.'

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
test = pd.read_csv(test_path)
print("Loaded test:", test.shape, test.columns.tolist())



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

if have_local_model:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            local_tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        ).eval()

        head_weights_path = os.path.join(local_model_path, "classification_head.pth")
        if os.path.isfile(head_weights_path):
            head_weights = torch.load(
                head_weights_path,
                map_location="cuda" if torch.cuda.is_available() else "cpu",
            )
            head = torch.nn.Linear(1, 1, bias=False)
            if torch.cuda.is_available():
                head = head.to("cuda")
            head.weight.data = head_weights
            use_model = True
            print("Hooray! Model/tokenizer/head loaded.")
        else:
            print(
                "WARNING: classification_head.pth not found; will fall back to constant predictions."
            )
    except Exception as e:
        print(
            "WARNING: Failed to load local model/tokenizer; will fall back to constant predictions."
        )
        print("Load error:", repr(e))
else:
    print(
        "WARNING: Local model directory not available; will fall back to constant predictions."
    )



## === cell 5
if "score" not in test.columns:
    test["score"] = np.nan



## === cell 6
max_length = 512

if use_model:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for idx, row in test.iterrows():
        text = row["full_text"]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )

        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs).logits
            logits = head(out[:, -1])
            predicted_score = torch.argmax(logits, dim=1).item()
            predicted_label = reverse_score_mapping.get(predicted_score, 3)
            test.at[idx, "score"] = predicted_label

        del inputs, out, logits
        if device == "cuda":
            torch.cuda.empty_cache()
else:
    test["score"] = 3



## === cell 7
test["score"] = (
    pd.to_numeric(test["score"], errors="coerce").fillna(3).round().astype(int)
)
test["score"] = test["score"].clip(1, 6)

print(test[["essay_id", "score"]].head())



## === cell 8
sub = test[["essay_id", "score"]].copy()
sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print("Saved:", sub_path, "| shape:", sub.shape)
print(sub.head())
