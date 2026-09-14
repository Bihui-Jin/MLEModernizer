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

1.3105775659942334

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)
from transformers import logging as transformers_logging
from peft import PeftModel

transformers_logging.set_verbosity_error()

model_id = "/kaggle/input/qwen2.5/transformers/3b/1"
adapter_id = "/kaggle/input/qwen2-5-3b-lora/qwen25_arena_rm/qwen25_arena_rm"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_local_model_dir(path: str) -> str:
    """
    Ensure we pass an existing local directory. Some Kaggle datasets include nested folders.
    """
    if path and os.path.isdir(path):
        return path

    candidates = []
    if path:
        candidates.extend(
            [
                path,
                os.path.join(path, "transformers"),
                os.path.join(path, "model"),
                os.path.join(path, "model", "transformers"),
            ]
        )

    for c in candidates:
        if c and os.path.isdir(c):
            return c

    return path


model_id = _resolve_local_model_dir(model_id)
adapter_id = _resolve_local_model_dir(adapter_id)

print("DEVICE:", DEVICE)
print("Resolved model_id:", model_id, "exists:", os.path.isdir(model_id))
print("Resolved adapter_id:", adapter_id, "exists:", os.path.isdir(adapter_id))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
config = AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=True,
    local_files_only=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    config=config,
    trust_remote_code=True,
    local_files_only=True,
)

base_model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    config=config,
    num_labels=1,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True,
    local_files_only=True,
)

if not hasattr(base_model, "prepare_inputs_for_generation"):
    base_model.prepare_inputs_for_generation = lambda *args, **kwargs: None

model = PeftModel.from_pretrained(base_model, adapter_id, local_files_only=True)
model.eval()

if DEVICE != "cuda":
    model.to(DEVICE)




## --- ERROR in cell 1, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/qwen2.5/transformers/3b/1'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/qwen2.5/transformers/3b/1'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/1861931820.py in <cell line: 0>()
      1 # --- Fix 2: Bypass HF repo-id validation for absolute local paths by going through AutoConfig
      2 # This keeps core logic the same (same model, same adapter, same head), but avoids HFValidationError.
----> 3 config = AutoConfig.from_pretrained(
      4     model_id,
      5     trust_remote_code=True,

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
    688             except Exception:
    689                 # For any other exception, we throw a generic error.
--> 690                 raise OSError(
    691                     f"Can't load the configuration of '{pretrained_model_name_or_path}'. If you were trying to load it"
    692                     " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"

OSError: Can't load the configuration of '/kaggle/input/qwen2.5/transformers/3b/1'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '/kaggle/input/qwen2.5/transformers/3b/1' is the correct path to a directory containing a config.json file

## === cell 2
def calculate_probs(s_a, s_b, epsilon=0.6, temperature=0.7):
    """
    Difference-to-probability mapping with a tie component.
    Returns (winner_model_a, winner_model_b, winner_tie) that sum to 1.
    """
    diff = (s_a - s_b) / temperature

    p_a_raw = 1.0 / (1.0 + np.exp(-diff))
    p_b_raw = 1.0 - p_a_raw

    p_tie = np.exp(-np.abs(diff) / epsilon) * 0.25  # max tie probability at 0.25

    total = p_a_raw * (1.0 - p_tie) + p_b_raw * (1.0 - p_tie) + p_tie
    winner_a = (p_a_raw * (1.0 - p_tie)) / total
    winner_b = (p_b_raw * (1.0 - p_tie)) / total
    winner_tie = p_tie / total

    winner_a = float(np.clip(winner_a, 1e-8, 1.0))
    winner_b = float(np.clip(winner_b, 1e-8, 1.0))
    winner_tie = float(np.clip(winner_tie, 1e-8, 1.0))
    s = winner_a + winner_b + winner_tie
    winner_a, winner_b, winner_tie = winner_a / s, winner_b / s, winner_tie / s

    return winner_a, winner_b, winner_tie


@torch.no_grad()
def get_score(prompt, response, max_length=2048):
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    logits = model(**inputs).logits
    return float(logits[0].item())




## === cell 3
test_path = "/kaggle/input/lmsys-chatbot-arena/test.csv"
if not os.path.exists(test_path):
    test_path = "/kaggle/data/lmsys-chatbot-arena/test.csv"
if not os.path.exists(test_path):
    test_path = "/kaggle/data/test.csv"

test_df = pd.read_csv(test_path)

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

sub = pd.DataFrame(results)
sub = sub[["id", "winner_model_a", "winner_model_b", "winner_tie"]]

for c in ["winner_model_a", "winner_model_b", "winner_tie"]:
    sub[c] = sub[c].astype("float64")

row_sums = sub[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1).values
sub[["winner_model_a", "winner_model_b", "winner_tie"]] = sub[
    ["winner_model_a", "winner_model_b", "winner_tie"]
].div(row_sums, axis=0)

sub.to_csv("submission.csv", index=False)
print("finished, wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1546323691.py in <cell line: 0>()
     10 results = []
     11 for _, row in tqdm(test_df.iterrows(), total=len(test_df), disable=True):
---> 12     s_a = get_score(row["prompt"], row["response_a"])
     13     s_b = get_score(row["prompt"], row["response_b"])
     14 

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/3014382131.py in get_score(prompt, response, max_length)
     28 def get_score(prompt, response, max_length=2048):
     29     text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
---> 30     inputs = tokenizer(
     31         text,
     32         return_tensors="pt",

NameError: name 'tokenizer' is not defined
