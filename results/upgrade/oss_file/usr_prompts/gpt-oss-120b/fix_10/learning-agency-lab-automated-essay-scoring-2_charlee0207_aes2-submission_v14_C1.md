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

3.13

# 3. Installed packages

datasets==4.4.1
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
tensorflow-datasets==4.9.9
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
vega-datasets==0.9.0

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

0.781539931926555

# 6. Current score

-0.00285

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.00339) has done: 'I reduced the maximum sequence length to BERT’s limit (512 tokens) and removed the `Trainer`‑based inference, which caused protobuf errors. Instead, I perform manual batched forward passes with a simple `DataLoader`, collect logits, and build the submission CSV directly. This fixes the shape mismatch and eliminates the `MessageFactory` error while keeping the original model and tokenizer logic untouched.'
- What this solution (achieved -0.00528) has done: 'Implemented a robust model loading routine that avoids the protobuf `MessageFactory` error by always initializing the tokenizer and a base BERT model from the HuggingFace hub, then manually loading the fine‑tuned weights from the local checkpoint if they exist. This preserves the original architecture while fixing the crash. The rest of the pipeline (encoding, DataLoader, inference, and submission writing) remains unchanged, ensuring a valid `submission.csv` is produced and improving the model’s performance toward the target score.'
- What this solution (achieved -0.12036) has done: 'The fix loads the fine‑tuned checkpoint safely by using a CPU map location and allowing mismatched layer sizes, which avoids the protobuf `MessageFactory` error. The device handling is also corrected. All other logic stays unchanged, so the model now uses the intended weights and produces a proper `submission.csv` with predictions in the required format.'
- What this solution (achieved 4e-05) has done: 'I replace the manual checkpoint loading that triggers the protobuf `MessageFactory` error with a safer `from_pretrained` call that directly loads the fine‑tuned model if the checkpoint directory exists, otherwise it falls back to the base BERT model. This eliminates the AttributeError while keeping the original architecture unchanged, ensuring the script runs end‑to‑end and produces a valid `submission.csv`.'
- What this solution (achieved -0.00702) has done: 'I simplify the model loading to always use the base pretrained BERT model (avoiding the checkpoint that triggers a protobuf error) and generate the submission directly from the test dataframe rather than the small sample file, ensuring the output CSV has the correct number of rows and proper columns.'
- What this solution (achieved 0.04089) has done: 'The fix adds a protobuf compatibility setting before importing transformers and changes the model loading to first try the fine‑tuned checkpoint (MODEL_DIR); if it isn’t present it falls back to the base BERT model. This removes the `MessageFactory` error and allows the script to use the trained weights, which should raise the Quadratic Weighted Kappa toward the target while keeping the core logic unchanged.'
- What this solution (achieved -0.00285) has done: 'The fix adds a safe fallback when loading the fine‑tuned checkpoint: if the protobuf‑related load fails, the script now falls back to the base BERT model, preventing the `MessageFactory` error.  
Prediction is switched from a plain argmax to the expected class value (soft‑max weighted average) rounded to the 1‑6 range, which better matches the quadratic weighted kappa metric and should move the score toward the target while keeping the original model and training logic unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from pathlib import Path

MODEL_NAME = "bert-base-cased"
INPUT_DIR = "/kaggle/input/"
MODEL_DIR = "/kaggle/input/aes2-bertbase-5ep-results/aes2_bertbase_5ep_results/"
MAX_LENGTH = 512  # BERT’s positional embedding limit
RANDOM_SEED = 42

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

try:
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        num_labels=6,
        ignore_mismatched_sizes=True,
    )
    print(f"Loaded fine‑tuned model from: {MODEL_DIR}")
except Exception as e:
    print(f"Failed to load fine‑tuned model ({e}); loading base model instead.")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=6,
        ignore_mismatched_sizes=True,
    )
    print(f"Loaded base model from: {MODEL_NAME}")

model.to(device)
model.eval()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/aes2-bertbase-5ep-results/aes2_bertbase_5ep_results/'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_55/2671526995.py in <cell line: 0>()
     26 try:
---> 27     model = transformers.AutoModelForSequenceClassification.from_pretrained(
     28         MODEL_DIR,

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/auto_factory.py in from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
    507                 # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
--> 508                 resolved_config_file = cached_file(
    509                     pretrained_model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/aes2-bertbase-5ep-results/aes2_bertbase_5ep_results/'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv(
    os.path.join(INPUT_DIR, "learning-agency-lab-automated-essay-scoring-2/test.csv")
)
texts = df["full_text"].tolist()
print("Read test CSV successfully:", len(texts), "records")

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
print("First encoding example:", {k: v[0][:5].tolist() for k, v in encodings.items()})


class EssayDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


test_dataset = EssayDataset(encodings)
loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

all_logits = []
model.to(device)
with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits  # shape (batch, num_labels)
        all_logits.append(logits.cpu())

logits_tensor = torch.cat(all_logits, dim=0)  # keep as tensor for softmax
probs = F.softmax(logits_tensor, dim=1)
class_range = torch.arange(1, 7, dtype=torch.float32)  # 1‑6
expected_scores = (probs * class_range).sum(dim=1)
pred_ids = torch.clamp(torch.round(expected_scores), 1, 6).int().numpy()

print("Prediction completed, sample predictions:", pred_ids[:5])

submission_df = pd.DataFrame({"essay_id": df["essay_id"], "score": pred_ids})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")
print(submission_df.head())
