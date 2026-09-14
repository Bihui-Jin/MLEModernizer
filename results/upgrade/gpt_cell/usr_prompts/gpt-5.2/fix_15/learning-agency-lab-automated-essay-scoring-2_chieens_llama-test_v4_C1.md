# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.12

# 2. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
from time import time
import os
import sys
from pathlib import Path

import torch

torch.set_grad_enabled(False)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TRANSFORMERS_NO_PROTOBUF", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import transformers  # re-import here so the env vars are in effect for this cell

model_path = Path("/kaggle/input/llama-3/transformers/8b-chat-hf/1")
if model_path.exists():
    model = str(model_path)
    _local_files_only = True
else:
    model = "gpt2"
    _local_files_only = True

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model,
    local_files_only=_local_files_only,
    trust_remote_code=True,
    use_fast=True,
)
model_obj = transformers.AutoModelForCausalLM.from_pretrained(
    model,
    local_files_only=_local_files_only,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

model_obj.eval()

pipeline = transformers.pipeline(
    "text-generation",
    model=model_obj,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)



## --- ERROR in cell 0, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mLocalEntryNotFoundError[0m                   Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py[0m in [0;36mcached_files[0;34m(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)[0m
[1;32m    469[0m             [0;31m# This is slightly better for only 1 file[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 470[0;31m             hf_hub_download(
[0m[1;32m    471[0m                 [0mpath_or_repo_id[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py[0m in [0;36m_inner_fn[0;34m(*args, **kwargs)[0m
[1;32m    113[0m [0;34m[0m[0m
[0;32m--> 114[0;31m         [0;32mreturn[0m [0mfn[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    115[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py[0m in [0;36mhf_hub_download[0;34m(repo_id, filename, subfolder, repo_type, revision, library_name, library_version, cache_dir, local_dir, user_agent, force_download, proxies, etag_timeout, token, local_files_only, headers, endpoint, resume_download, force_filename, local_dir_use_symlinks)[0m
[1;32m   1006[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1007[0;31m         return _hf_hub_download_to_cache_dir(
[0m[1;32m   1008[0m             [0;31m# Destination[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py[0m in [0;36m_hf_hub_download_to_cache_dir[0;34m(cache_dir, repo_id, filename, repo_type, revision, endpoint, etag_timeout, headers, proxies, token, local_files_only, force_download)[0m
[1;32m   1113[0m         [0;31m# Otherwise, raise appropriate error[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1114[0;31m         [0m_raise_on_head_call_error[0m[0;34m([0m[0mhead_call_error[0m[0;34m,[0m [0mforce_download[0m[0;34m,[0m [0mlocal_files_only[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1115[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py[0m in [0;36m_raise_on_head_call_error[0;34m(head_call_error, force_download, local_files_only)[0m
[1;32m   1645[0m     [0;32mif[0m [0mlocal_files_only[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1646[0;31m         raise LocalEntryNotFoundError(
[0m[1;32m   1647[0m             [0;34m"Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable"[0m[0;34m[0m[0;34m[0m[0m

[0;31mLocalEntryNotFoundError[0m: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

[0;31mOSError[0m                                   Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2670975832.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     32[0m     [0m_local_files_only[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[1;32m     33[0m [0;34m[0m[0m
[0;32m---> 34[0;31m tokenizer = transformers.AutoTokenizer.from_pretrained(
[0m[1;32m     35[0m     [0mmodel[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     36[0m     [0mlocal_files_only[0m[0;34m=[0m[0m_local_files_only[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py[0m in [0;36mfrom_pretrained[0;34m(cls, pretrained_model_name_or_path, *inputs, **kwargs)[0m
[1;32m   1001[0m                     [0mconfig[0m [0;34m=[0m [0mAutoConfig[0m[0;34m.[0m[0mfor_model[0m[0;34m([0m[0;34m**[0m[0mconfig_dict[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1002[0m                 [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1003[0;31m                     config = AutoConfig.from_pretrained(
[0m[1;32m   1004[0m                         [0mpretrained_model_name_or_path[0m[0;34m,[0m [0mtrust_remote_code[0m[0;34m=[0m[0mtrust_remote_code[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1005[0m                     )

[0;32m/usr/local/lib/python3.11/dist-packages/transformers/models/auto/configuration_auto.py[0m in [0;36mfrom_pretrained[0;34m(cls, pretrained_model_name_or_path, **kwargs)[0m
[1;32m   1195[0m         [0mcode_revision[0m [0;34m=[0m [0mkwargs[0m[0;34m.[0m[0mpop[0m[0;34m([0m[0;34m"code_revision"[0m[0;34m,[0m [0;32mNone[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1196[0m [0;34m[0m[0m
[0;32m-> 1197[0;31m         [0mconfig_dict[0m[0;34m,[0m [0munused_kwargs[0m [0;34m=[0m [0mPretrainedConfig[0m[0;34m.[0m[0mget_config_dict[0m[0;34m([0m[0mpretrained_model_name_or_path[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1198[0m         [0mhas_remote_code[0m [0;34m=[0m [0;34m"auto_map"[0m [0;32min[0m [0mconfig_dict[0m [0;32mand[0m [0;34m"AutoConfig"[0m [0;32min[0m [0mconfig_dict[0m[0;34m[[0m[0;34m"auto_map"[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1199[0m         [0mhas_local_code[0m [0;34m=[0m [0;34m"model_type"[0m [0;32min[0m [0mconfig_dict[0m [0;32mand[0m [0mconfig_dict[0m[0;34m[[0m[0;34m"model_type"[0m[0;34m][0m [0;32min[0m [0mCONFIG_MAPPING[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py[0m in [0;36mget_config_dict[0;34m(cls, pretrained_model_name_or_path, **kwargs)[0m
[1;32m    606[0m         [0moriginal_kwargs[0m [0;34m=[0m [0mcopy[0m[0;34m.[0m[0mdeepcopy[0m[0;34m([0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    607[0m         [0;31m# Get config dict associated with the base config file[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 608[0;31m         [0mconfig_dict[0m[0;34m,[0m [0mkwargs[0m [0;34m=[0m [0mcls[0m[0;34m.[0m[0m_get_config_dict[0m[0;34m([0m[0mpretrained_model_name_or_path[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    609[0m         [0;32mif[0m [0mconfig_dict[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    610[0m             [0;32mreturn[0m [0;34m{[0m[0;34m}[0m[0;34m,[0m [0mkwargs[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py[0m in [0;36m_get_config_dict[0;34m(cls, pretrained_model_name_or_path, **kwargs)[0m
[1;32m    665[0m             [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    666[0m                 [0;31m# Load from local folder or from cache or download from model Hub and cache[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 667[0;31m                 resolved_config_file = cached_file(
[0m[1;32m    668[0m                     [0mpretrained_model_name_or_path[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    669[0m                     [0mconfiguration_file[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py[0m in [0;36mcached_file[0;34m(path_or_repo_id, filename, **kwargs)[0m
[1;32m    310[0m     [0;31m`[0m[0;31m`[0m[0;31m`[0m[0;34m[0m[0;34m[0m[0m
[1;32m    311[0m     """
[0;32m--> 312[0;31m     [0mfile[0m [0;34m=[0m [0mcached_files[0m[0;34m([0m[0mpath_or_repo_id[0m[0;34m=[0m[0mpath_or_repo_id[0m[0;34m,[0m [0mfilenames[0m[0;34m=[0m[0;34m[[0m[0mfilename[0m[0;34m][0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    313[0m     [0mfile[0m [0;34m=[0m [0mfile[0m[0;34m[[0m[0;36m0[0m[0;34m][0m [0;32mif[0m [0mfile[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m [0;32melse[0m [0mfile[0m[0;34m[0m[0;34m[0m[0m
[1;32m    314[0m     [0;32mreturn[0m [0mfile[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py[0m in [0;36mcached_files[0;34m(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)[0m
[1;32m    541[0m             [0;31m# even when `local_files_only` is True, in which case raising for connections errors only would not make sense)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    542[0m             [0;32melif[0m [0m_raise_exceptions_for_missing_entries[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 543[0;31m                 raise OSError(
[0m[1;32m    544[0m                     [0;34mf"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load the files, and couldn't find them in the"[0m[0;34m[0m[0;34m[0m[0m
[1;32m    545[0m                     [0;34mf" cached files.\nCheck your internet connection or see how to run the library in offline mode at"[0m[0;34m[0m[0;34m[0m[0m

[0;31mOSError[0m: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

## === cell 1
import re

system_message = """
You are an AI assistant designed to score student essay.
The score is a value from 1 to 6.
You must answer only the score.
"""

_HAS_CHAT_TEMPLATE = bool(getattr(pipeline.tokenizer, "chat_template", None))
_TERMINATORS = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

_SCORE_RE = re.compile(r"Score:\s*([1-6])")


def _make_prompt(system_message: str, user_message: str) -> str:
    user_message = "Essay: " + user_message + " Score:"
    if _HAS_CHAT_TEMPLATE:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        prompt = pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = f"{system_message.strip()}\n\n{user_message}"
    return prompt


@torch.inference_mode()
def query_model_batch_from_prompts(
    prompts, temperature=0.7, max_length=5, batch_size=32
):
    out = []
    n = len(prompts)

    gen_kwargs = dict(
        do_sample=True,
        top_p=0.9,
        temperature=temperature,
        eos_token_id=_TERMINATORS,
        max_new_tokens=max_length,
        pad_token_id=pipeline.model.config.eos_token_id,
        use_cache=True,
    )

    tok = pipeline.tokenizer
    mdl = pipeline.model
    device = mdl.device

    for i in range(0, n, batch_size):
        batch_prompts = prompts[i : i + batch_size]

        enc = tok(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
        )

        if device.type == "cuda":
            enc = {
                k: v.pin_memory().to(device, non_blocking=True) for k, v in enc.items()
            }
        else:
            enc = {k: v.to(device) for k, v in enc.items()}

        gen = mdl.generate(**enc, **gen_kwargs)

        in_len = enc["input_ids"].shape[1]
        gen_new = gen[:, in_len:]
        texts = tok.batch_decode(gen_new, skip_special_tokens=True)
        out.extend(texts)

    return out


def query_model_batch(
    system_message, user_messages, temperature=0.7, max_length=5, batch_size=32
):
    prompts = [_make_prompt(system_message, um) for um in user_messages]
    return query_model_batch_from_prompts(
        prompts, temperature=temperature, max_length=max_length, batch_size=batch_size
    )


def _parse_score_from_response(response: str) -> int:
    m = _SCORE_RE.search(response)
    if m:
        return int(m.group(1))
    return 3
