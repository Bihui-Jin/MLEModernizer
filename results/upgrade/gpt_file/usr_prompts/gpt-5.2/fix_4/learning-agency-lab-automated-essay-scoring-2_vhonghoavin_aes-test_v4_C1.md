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

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker by replacing the missing local model path (`/kaggle/input/aes-model-vhhv-2`) with a robust fallback that loads a public Hugging Face checkpoint (`microsoft/deberta-v3-base`) already available in the installed `transformers` stack. Since that changes the pipeline from “inference with a provided fine-tuned classifier” to “inference without a fine-tuned classifier,” I keep the rest of your core inference logic intact but add a safe, deterministic, score-neutral fallback that outputs a valid 1–6 score for every test row. Finally, I ensure the submission length always matches the test set and the file is written as `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
df_test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
df_test.head()



## === cell 2
repository_id = "/kaggle/input/aes-model-vhhv-2"



## === cell 3
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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


os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

fallback_model_id = "microsoft/deberta-v3-base"

try:
    model_dir = find_hf_model_dir(repository_id)
    print("Using local model_dir:", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    )
except FileNotFoundError as e:
    print("WARNING:", str(e))
    print(f"Falling back to online model_id: {fallback_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(fallback_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        fallback_model_id, num_labels=6
    )

model.to(device)
model.eval()

num_labels = getattr(model.config, "num_labels", None)
print("num_labels:", num_labels)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2584057011.py in <cell line: 0>()
     31 try:
---> 32     model_dir = find_hf_model_dir(repository_id)
     33     print("Using local model_dir:", model_dir)

/tmp/ipykernel_55/2584057011.py in find_hf_model_dir(root_dir)
      6     if not os.path.exists(root_dir):
----> 7         raise FileNotFoundError(
      8             f"Model directory not found: {root_dir}. "

FileNotFoundError: Model directory not found: /kaggle/input/aes-model-vhhv-2. Available dirs under /kaggle/input: ['description.md', 'learning-agency-lab-automated-essay-scoring-2', 'sample_submission.csv', 'sample_submission.csv.zip', 'test.csv', 'test.csv.zip', 'train.csv', 'train.csv.zip']

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
test_sentences = df_test["full_text"].astype(str).tolist()
len(test_sentences), test_sentences[0][:200]



## === cell 6
pred_classes = []
batch_size = 16  # keep as-is (safe default)

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
        outputs = model(**enc)

        logits = outputs.logits
        if logits.ndim != 2:
            logits = logits.view(logits.shape[0], -1)

        if logits.shape[1] == 1:
            batch_pred = [2] * logits.shape[0]
        else:
            probs = logits.softmax(dim=1)
            batch_pred = torch.argmax(probs, dim=1).detach().cpu().numpy().tolist()

        pred_classes.extend(batch_pred)

print("Preds:", len(pred_classes), "Expected:", len(df_test))
if len(pred_classes) != len(df_test):
    raise RuntimeError(
        f"Prediction length mismatch: got {len(pred_classes)} preds for {len(df_test)} test rows."
    )

pred_classes[:5]



## === cell 7
pred_scores = (np.array(pred_classes, dtype=np.int64) + 1).clip(1, 6)

df_submit = pd.DataFrame(
    {
        "essay_id": df_test["essay_id"].astype(str).values,
        "score": pred_scores.astype(int),
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
