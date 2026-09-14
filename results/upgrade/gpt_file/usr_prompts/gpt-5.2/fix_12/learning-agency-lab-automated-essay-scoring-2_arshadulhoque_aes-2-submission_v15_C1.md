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

datasets==4.4.1
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

0.741923748724992

# 6. Current score

-0.0074

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.03275) has done: 'The timeout is dominated by tokenizing all 15k essays up front into Python lists (very slow and memory-heavy) and by padding/batching overhead in the DataLoader. I keep the same model and inference logic, but switch to on-the-fly tokenization per batch using a Dataset that stores raw texts and a collator that tokenizes batches (fast tokenizer + Rust parallelism) to avoid building a huge `encodings` object. I also enable `model.eval()` + `torch.inference_mode()` as before, keep determinism settings, and use efficient DataLoader settings (workers, prefetch, pinned memory) while preserving identical truncation/max_length/padding semantics and argmax-to-score mapping.'
- What this solution (achieved -0.0074) has done: 'I fix two execution blockers: the offline HuggingFace model loading failure (by loading a compatible transformer checkpoint that is already available offline in the Kaggle environment) and the DataLoader collator error caused by passing `padding` twice. These changes keep the same inference-only, argmax-classification-to-1..6 mapping core logic, but make the notebook run end-to-end and reliably write `submission.csv`. Since your current score is extremely low, switching to a locally available pretrained sequence-classification checkpoint should also move the score substantially toward the target band (while still preserving the same evaluation semantics).'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.set_grad_enabled(False)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



## === cell 1
test_data = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
test_data



## === cell 2
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, local_files_only=True, use_fast=True
)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, local_files_only=True
)

if getattr(model.config, "num_labels", None) != 6:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=6,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )

model.eval()



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
/tmp/ipykernel_11/3415432924.py in <cell line: 0>()
      4 MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
      5 
----> 6 tokenizer = AutoTokenizer.from_pretrained(
      7     MODEL_NAME, local_files_only=True, use_fast=True
      8 )

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
MAX_LEN = 1536

tok_kwargs = dict(
    truncation=True,
    max_length=MAX_LEN,
    padding=False,  # collator will pad dynamically
    return_attention_mask=True,
    return_token_type_ids=False,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


class RawTextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


test_dataset = RawTextDataset(test_data["full_text"].tolist())


class TokenizeCollator:
    def __init__(self, tokenizer, tok_kwargs, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.tok_kwargs = {k: v for k, v in tok_kwargs.items() if k != "padding"}
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch_texts):
        enc = self.tokenizer(
            batch_texts,
            **self.tok_kwargs,
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}




## === cell 4
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)

if hasattr(model, "config"):
    try:
        model.config.use_cache = False
    except Exception:
        pass

if use_cuda and os.environ.get("DISABLE_TORCH_COMPILE", "0") != "1":
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception:
        pass

pad_to_multiple_of = 8 if use_cuda else None
collate_fn = TokenizeCollator(
    tokenizer=tokenizer, tok_kwargs=tok_kwargs, pad_to_multiple_of=pad_to_multiple_of
)

if use_cuda:
    num_workers = min(4, os.cpu_count() or 2)
    prefetch_factor = 4
    persistent_workers = True
else:
    num_workers = 0
    prefetch_factor = None
    persistent_workers = False

dl_kwargs = dict(
    dataset=test_dataset,
    batch_size=(64 if use_cuda else 8),
    shuffle=False,
    num_workers=num_workers,
    pin_memory=use_cuda,
    collate_fn=collate_fn,
    persistent_workers=persistent_workers,
)
if num_workers > 0:
    dl_kwargs["prefetch_factor"] = prefetch_factor

test_loader = DataLoader(**dl_kwargs)

if use_cuda:
    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception:
        pass



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2512762249.py in <cell line: 0>()
      1 use_cuda = torch.cuda.is_available()
      2 device = torch.device("cuda" if use_cuda else "cpu")
----> 3 model.to(device)
      4 
      5 if hasattr(model, "config"):

NameError: name 'model' is not defined

## === cell 5
pred_classes = np.empty(len(test_dataset), dtype=np.int64)

model.eval()
offset = 0

with torch.inference_mode():
    for batch in test_loader:
        if use_cuda:
            batch["input_ids"] = batch["input_ids"].to(device, non_blocking=True)
            batch["attention_mask"] = batch["attention_mask"].to(
                device, non_blocking=True
            )
        else:
            batch["input_ids"] = batch["input_ids"].to(device)
            batch["attention_mask"] = batch["attention_mask"].to(device)

        outputs = model(**batch)
        batch_pred = torch.argmax(outputs.logits, dim=-1)
        batch_pred_np = batch_pred.cpu().numpy().astype(np.int64, copy=False)

        bs = batch_pred_np.shape[0]
        pred_classes[offset : offset + bs] = batch_pred_np
        offset += bs

assert offset == len(test_dataset)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3355896025.py in <cell line: 0>()
      1 pred_classes = np.empty(len(test_dataset), dtype=np.int64)
      2 
----> 3 model.eval()
      4 offset = 0
      5 

NameError: name 'model' is not defined

## === cell 6
predicted_scores = (pred_classes + 1).clip(1, 6).astype("int32")



## === cell 7
submission = pd.DataFrame(
    {"essay_id": test_data["essay_id"].values, "score": predicted_scores}
)

submission.to_csv("submission.csv", index=False)
submission



## === cell 8
assert submission.shape[0] == test_data.shape[0]
assert list(submission.columns) == ["essay_id", "score"]
assert submission["score"].between(1, 6).all()
print("Wrote submission.csv with shape:", submission.shape)
