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
Predict which chatbot response a user will prefer in a competition between two chatbots.

## Metric
Log loss with "eps=auto"

## Submission Format
For each id in the test set, you must predict the probability for each target class. The file should contain a header and have the following format:

```
 id,winner_model_a,winner_model_b,winner_tie
 136060,0.33,0,33,0.33
 211333,0.33,0,33,0.33
 1233961,0.33,0,33,0.33
 etc
```

## Dataset
**train.csv**

- `id` - A unique identifier for the row.
- `model_[a/b]` - The identity of model_[a/b]. Included in train.csv but not test.csv.
- `prompt` - The prompt that was given as an input (to both models).
- `response_[a/b]` - The response from model_[a/b] to the given prompt.
- `winner_model_[a/b/tie]` - Binary columns marking the judge's selection. The ground truth target column.

**test.csv**

- `id`
- `prompt`
- `response_[a/b]`

**sample_submission.csv** A submission file in the correct format.

- `id`
- `winner_model_[a/b/tie]` - This is what is predicted from the test set.

# 2. Python version

3.14

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
peft==0.16.0
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
tqdm==4.67.1
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        input/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        working/
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
```

-> data/lmsys-chatbot-arena/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/lmsys-chatbot-arena/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/lmsys-chatbot-arena/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> data/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> (stopped after 10 files for performance)

# 5. Target score

1.3383791384225974

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()

TEST_PATH = "/kaggle/input/lmsys-chatbot-arena/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/lmsys-chatbot-arena/sample_submission.csv"
SUB_PATH = "submission.csv"

MODEL_ROOT = "/kaggle/input/qwen2.5"
MODEL_CANDIDATES = [
    "/kaggle/input/qwen2.5/transformers/3b-instruct",  # common Kaggle layout
    "/kaggle/input/qwen2.5/transformers/3b-instruct/1",  # sometimes exists but may break HF validators
]

USE_PEFT_ADAPTER = False
adapter_id = "/kaggle/input/qwen-lora/qwen25_arena_rm"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


def _has_any_weights(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        files = os.listdir(path)
    except Exception:
        return False
    return any(
        f in ("model.safetensors", "pytorch_model.bin")
        or f.endswith(".safetensors")
        or f.endswith(".bin")
        for f in files
    )


def _looks_like_hf_model_dir(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    has_config = os.path.isfile(os.path.join(path, "config.json"))
    return has_config and _has_any_weights(path)


def resolve_local_model_dir() -> str | None:
    for p in MODEL_CANDIDATES:
        if _looks_like_hf_model_dir(p):
            return p

    if os.path.isdir(MODEL_ROOT):
        for root, dirs, files in os.walk(MODEL_ROOT):
            if "config.json" in files and any(
                f.endswith(".safetensors") or f.endswith(".bin") for f in files
            ):
                return root

    return None


model_dir = resolve_local_model_dir()
print(f"Resolved local model directory: {model_dir}")

FALLBACK_MODEL_ID = "distilbert-base-uncased"

try:
    if model_dir is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=1,
            torch_dtype=dtype,
            device_map=None,
            trust_remote_code=True,
            local_files_only=True,
        ).to(device)
    else:
        raise FileNotFoundError(
            "No valid local model directory found under MODEL_ROOT/MODEL_CANDIDATES."
        )
except Exception as e:
    print(
        f"Local model load failed; falling back to {FALLBACK_MODEL_ID} (local cache if available). Error: {e}"
    )
    tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_ID, local_files_only=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        FALLBACK_MODEL_ID,
        num_labels=1,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)

base_model.eval()
model = base_model

if USE_PEFT_ADAPTER:
    try:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, adapter_id).to(device)
        model.eval()
    except Exception as e:
        print(f"PEFT adapter load failed; falling back to base model. Error: {e}")
        model = base_model

if not hasattr(model, "prepare_inputs_for_generation"):
    model.prepare_inputs_for_generation = lambda *args, **kwargs: None


@torch.no_grad()
def get_score(prompt: str, response: str) -> float:
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    return float(logits[0].item())


def calculate_probs(s_a, s_b, epsilon=0.6, temperature=0.7):
    """
    Difference-mapping with a tie component.
    Ensures outputs are finite, float64, and sum to 1.
    """
    diff = (s_a - s_b) / max(temperature, 1e-8)

    diff = float(np.clip(diff, -50.0, 50.0))
    p_a_raw = 1.0 / (1.0 + np.exp(-diff))
    p_b_raw = 1.0 - p_a_raw

    p_tie = float(np.exp(-abs(diff) / max(epsilon, 1e-8)) * 0.25)

    total = p_a_raw * (1.0 - p_tie) + p_b_raw * (1.0 - p_tie) + p_tie
    if not np.isfinite(total) or total <= 0:
        return 1 / 3, 1 / 3, 1 / 3

    winner_a = (p_a_raw * (1.0 - p_tie)) / total
    winner_b = (p_b_raw * (1.0 - p_tie)) / total
    winner_tie = p_tie / total

    probs = np.array([winner_a, winner_b, winner_tie], dtype=np.float64)
    probs = np.clip(probs, 1e-15, 1.0)
    probs = probs / probs.sum()
    return float(probs[0]), float(probs[1]), float(probs[2])


test_df = pd.read_csv(TEST_PATH)

results = []
for _, row in tqdm(test_df.iterrows(), total=len(test_df), disable=True):
    s_a = get_score(row["prompt"], row["response_a"])
    s_b = get_score(row["prompt"], row["response_b"])

    prob_a, prob_b, prob_tie = calculate_probs(s_a, s_b)

    results.append(
        {
            "id": row["id"],
            "winner_model_a": prob_a,
            "winner_model_b": prob_b,
            "winner_tie": prob_tie,
        }
    )

sub_df = pd.DataFrame(
    results, columns=["id", "winner_model_a", "winner_model_b", "winner_tie"]
)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sub_df["id"] = sub_df["id"].astype(np.int64)
sub_df = sample_sub[["id"]].merge(sub_df, on="id", how="left")

for c in ["winner_model_a", "winner_model_b", "winner_tie"]:
    sub_df[c] = pd.to_numeric(sub_df[c], errors="coerce")

missing = sub_df[["winner_model_a", "winner_model_b", "winner_tie"]].isna().any(axis=1)
if missing.any():
    sub_df.loc[missing, ["winner_model_a", "winner_model_b", "winner_tie"]] = 1.0 / 3.0

probs = sub_df[["winner_model_a", "winner_model_b", "winner_tie"]].to_numpy(
    dtype=np.float64
)
probs = np.nan_to_num(probs, nan=1.0 / 3.0, posinf=1.0 / 3.0, neginf=1.0 / 3.0)
probs = np.clip(probs, 1e-15, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
sub_df[["winner_model_a", "winner_model_b", "winner_tie"]] = probs

sub_df.to_csv(SUB_PATH, index=False)

print(f"finished; wrote {SUB_PATH} with shape {sub_df.shape}")
print(sub_df.head())
print("Columns:", list(sub_df.columns))

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/1642167595.py in <cell line: 0>()
     92     else:
---> 93         raise FileNotFoundError(
     94             "No valid local model directory found under MODEL_ROOT/MODEL_CANDIDATES."

FileNotFoundError: No valid local model directory found under MODEL_ROOT/MODEL_CANDIDATES.

During handling of the above exception, another exception occurred:

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
/tmp/ipykernel_56/1642167595.py in <cell line: 0>()
     98         f"Local model load failed; falling back to {FALLBACK_MODEL_ID} (local cache if available). Error: {e}"
     99     )
--> 100     tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_ID, local_files_only=True)
    101     base_model = AutoModelForSequenceClassification.from_pretrained(
    102         FALLBACK_MODEL_ID,

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
