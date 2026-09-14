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

0.7600545627569122

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import torch

print(f"PyTorch version: {torch.__version__}")

import transformers

print(f"Hugging Face Transformers version: {transformers.__version__}")

import datasets

print(f"Hugging Face Datasets version: {datasets.__version__}")



## === cell 1
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_NAME = "bert-base-cased"
INPUT_DIR = "/kaggle/input/"
MODEL_DIR = "/kaggle/input/aes2-persuade-bertbase-2ep-results/"
CHECKPOINT_DIR = str(Path(MODEL_DIR) / "checkpoint-7000")
MAX_LENGTH = 1024
RANDOM_SEED = 42
EVAL_USE_PRETRAIN = 1
SUBMISSION = 1

has_cuda = torch.cuda.is_available()
device = "cuda:0" if has_cuda else "cpu"
print("device:", device)
if has_cuda:
    print("cuda current_device:", torch.cuda.current_device())
else:
    print("CUDA not available; running on CPU.")

print("Checkpoint dir:", CHECKPOINT_DIR)
print("Checkpoint exists:", Path(CHECKPOINT_DIR).exists())

if not Path(INPUT_DIR).exists():
    INPUT_DIR = "/kaggle/data/"
print("Using INPUT_DIR:", INPUT_DIR)



## === cell 2

from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

ckpt_path = Path(CHECKPOINT_DIR)


def _looks_like_hf_dir(p: Path) -> bool:
    if not p.exists() or not p.is_dir():
        return False
    expected_any = [
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "merges.txt",
        "special_tokens_map.json",
    ]
    return any((p / f).exists() for f in expected_any)


pretrained_source = ckpt_path if _looks_like_hf_dir(ckpt_path) else MODEL_NAME
print("Loading from:", pretrained_source)

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_source,
    local_files_only=True,
    use_fast=True,
)
print("Load tokenizer successfully")

model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_source,
    local_files_only=True,
)
model.to(device)
model.eval()
print("Load model successfully")

trainer = None
trainer_available = False
try:
    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir=".",
        per_device_eval_batch_size=32,
        report_to="none",
        fp16=bool(has_cuda),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
    )
    trainer_available = True
    print("Trainer initialized successfully")
except Exception as e:
    print("Trainer init failed; will use PyTorch forward fallback.")
    print("Trainer error:", repr(e))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
LocalEntryNotFoundError                   Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    113 
--> 114         return fn(*args, **kwargs)
    115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in hf_hub_download(repo_id, filename, subfolder, repo_type, revision, library_name, library_version, cache_dir, local_dir, user_agent, force_download, proxies, etag_timeout, token, local_files_only, headers, endpoint, resume_download, force_filename, local_dir_use_symlinks)
   1006     else:
-> 1007         return _hf_hub_download_to_cache_dir(
   1008             # Destination

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _hf_hub_download_to_cache_dir(cache_dir, repo_id, filename, repo_type, revision, endpoint, etag_timeout, headers, proxies, token, local_files_only, force_download)
   1113         # Otherwise, raise appropriate error
-> 1114         _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1645     if local_files_only:
-> 1646         raise LocalEntryNotFoundError(
   1647             "Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable"

LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/777301175.py in <cell line: 0>()
     30 print("Loading from:", pretrained_source)
     31 
---> 32 tokenizer = AutoTokenizer.from_pretrained(
     33     pretrained_source,
     34     local_files_only=True,

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
   1001                     config = AutoConfig.for_model(**config_dict)
   1002                 else:
-> 1003                     config = AutoConfig.from_pretrained(
   1004                         pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
   1005                     )

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/configuration_auto.py in from_pretrained(cls, pretrained_model_name_or_path, **kwargs)
   1195         code_revision = kwargs.pop("code_revision", None)
   1196 
-> 1197         config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
   1198         has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
   1199         has_local_code = "model_type" in config_dict and config_dict["model_type"] in CONFIG_MAPPING

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    606         original_kwargs = copy.deepcopy(kwargs)
    607         # Get config dict associated with the base config file
--> 608         config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
    609         if config_dict is None:
    610             return {}, kwargs

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    665             try:
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,
    669                     configuration_file,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    541             # even when `local_files_only` is True, in which case raising for connections errors only would not make sense)
    542             elif _raise_exceptions_for_missing_entries:
--> 543                 raise OSError(
    544                     f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load the files, and couldn't find them in the"
    545                     f" cached files.\nCheck your internet connection or see how to run the library in offline mode at"

OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

## === cell 3
test_path = os.path.join(
    INPUT_DIR, "learning-agency-lab-automated-essay-scoring-2", "test.csv"
)
sample_path = os.path.join(
    INPUT_DIR, "learning-agency-lab-automated-essay-scoring-2", "sample_submission.csv"
)

if not Path(test_path).exists():
    test_path = os.path.join(INPUT_DIR, "test.csv")
if not Path(sample_path).exists():
    sample_path = os.path.join(INPUT_DIR, "sample_submission.csv")

df = pd.read_csv(test_path)
texts = df["full_text"].astype(str).tolist()
print("Read test.csv successfully:", df.shape)

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
)

model_input_keys = {"input_ids", "attention_mask", "token_type_ids"}
encodings = {k: v for k, v in encodings.items() if k in model_input_keys}
print("Encode texts successfully. Keys:", list(encodings.keys()))

if trainer_available:
    dataset = datasets.Dataset.from_dict(encodings)
    print("Establish Dataset successfully:", dataset)
    pred_out = trainer.predict(dataset)
    submission_logits = pred_out.predictions
    submission_preds = (
        torch.argmax(torch.tensor(submission_logits), dim=1).cpu().numpy() + 1
    )
else:
    input_ids = torch.tensor(encodings["input_ids"], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        encodings["attention_mask"], dtype=torch.long, device=device
    )
    token_type_ids = None
    if "token_type_ids" in encodings:
        token_type_ids = torch.tensor(
            encodings["token_type_ids"], dtype=torch.long, device=device
        )

    batch_size = 32
    preds = []
    with torch.no_grad():
        for start in range(0, input_ids.size(0), batch_size):
            end = start + batch_size
            batch = {
                "input_ids": input_ids[start:end],
                "attention_mask": attention_mask[start:end],
            }
            if token_type_ids is not None:
                batch["token_type_ids"] = token_type_ids[start:end]
            outputs = model(**batch)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).detach().cpu().numpy() + 1
            preds.append(batch_preds)
    submission_preds = np.concatenate(preds, axis=0)

print("Predict successfully. Pred shape:", submission_preds.shape)

submission_csv = pd.DataFrame(
    {"essay_id": df["essay_id"].values, "score": submission_preds.astype(int)}
)
submission_csv = submission_csv[["essay_id", "score"]]
submission_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", submission_csv.shape)
print(submission_csv.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2174605089.py in <cell line: 0>()
     16 print("Read test.csv successfully:", df.shape)
     17 
---> 18 encodings = tokenizer(
     19     texts,
     20     truncation=True,

NameError: name 'tokenizer' is not defined
