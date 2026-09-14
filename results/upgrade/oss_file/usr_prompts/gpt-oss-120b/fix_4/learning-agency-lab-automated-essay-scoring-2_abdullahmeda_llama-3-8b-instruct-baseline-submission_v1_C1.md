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

No external packages required in the script and installed.

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

0.8030170953994958

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import gc
import sys
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace

try:
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM

    HAVE_MODEL = True
except Exception:
    HAVE_MODEL = False

if hasattr(torch.backends.cuda, "enable_flash_sdp"):
    torch.backends.cuda.enable_flash_sdp(False)
if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
    torch.backends.cuda.enable_mem_efficient_sdp(False)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def main(args):
    config = SimpleNamespace(
        data_dir="/kaggle/input/learning-agency-lab-automated-essay-scoring-2",
    )

    if HAVE_MODEL:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_pth,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, args.lora_pth)
        tokenizer = AutoTokenizer.from_pretrained(args.model_pth, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        model = None
        tokenizer = None

    def preprocess(
        sample,
        text=False,
        infer_mode=False,
        max_seq=args.max_length,
        return_tensors=None,
    ):
        sys_prompt = (
            "Please read the following essay and assign a score of 1,2,3,4,5,6 "
            "where 6 is the best. Output only a single number with no explanation.\n\n"
        )
        prompt = sample["full_text"]
        answer = "" if infer_mode else str(sample["score"])

        messages = [
            {"role": "user", "content": sys_prompt + prompt},
            {"role": "assistant", "content": f"\n\nThe score is: " + answer},
        ]
        formatted_sample = tokenizer.apply_chat_template(messages, tokenize=False)
        if infer_mode:
            formatted_sample = formatted_sample.replace("<|eot_id|>", "")

        tokenized_sample = tokenizer(
            formatted_sample,
            padding=True,
            return_tensors=return_tensors,
            truncation=True,
            add_special_tokens=False,
            max_length=max_seq,
        )

        if return_tensors == "pt":
            tokenized_sample["labels"] = tokenized_sample["input_ids"].clone()
        else:
            tokenized_sample["labels"] = tokenized_sample["input_ids"].copy()

        if text:
            return formatted_sample
        else:
            return tokenized_sample

    df_test = pd.read_csv(f"{config.data_dir}/test.csv")
    sub = pd.read_csv(f"{config.data_dir}/sample_submission.csv")

    test_preds = []

    if HAVE_MODEL:
        for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
            tokenized_sample = preprocess(
                row, infer_mode=True, max_seq=args.max_length, return_tensors="pt"
            )
            generated_ids = model.generate(
                **tokenized_sample,
                max_new_tokens=2,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            try:
                answer = decoded[0].rsplit("The score is: ", 1)[1]
                test_preds.append(int(answer))
            except Exception:
                test_preds.append(3)
    else:
        train_df = pd.read_csv(f"{config.data_dir}/train.csv")
        train_df["len"] = train_df["full_text"].str.split().str.len()

        train_df["len_bin"] = pd.qcut(train_df["len"], q=10, duplicates="drop")
        bin_means = train_df.groupby("len_bin")["score"].mean()

        for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
            essay_len = len(str(row["full_text"]).split())
            try:
                bin_label = pd.cut([essay_len], bins=bin_means.index.categories)[0]
                pred = bin_means.loc[bin_label]
            except Exception:
                pred = train_df["score"].mean()
            pred = int(round(pred))
            pred = max(1, min(6, pred))  # ensure within valid score range
            test_preds.append(pred)

    sub["score"] = test_preds
    sub["score"] = sub["score"].astype("int")
    sub.to_csv(args.sub_pth, index=False)

    if HAVE_MODEL:
        del model, tokenizer
        torch.cuda.empty_cache()
    gc.collect()




## === cell 3
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_pth", type=str, default="", help="Path to the pretrained model"
    )
    parser.add_argument(
        "--lora_pth", type=str, default="", help="Path to the PEFT LoRA adapter"
    )
    parser.add_argument(
        "--sub_pth",
        type=str,
        default="submission.csv",
        help="Path to save submission file",
    )
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Max length of input sequence"
    )
    args, _ = parser.parse_known_args()
    main(args)

## --- ERROR in cell 3, traceback:
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
    159     if not REPO_ID_REGEX.match(repo_id):
--> 160         raise HFValidationError(
    161             "Repo id must use alphanumeric chars, '-', '_' or '.'."

HFValidationError: Repo id must use alphanumeric chars, '-', '_' or '.'. The name cannot start or end with '-' or '.' and the maximum length is 96: ''.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_55/4184830200.py in <cell line: 0>()
     18     # parse_known_args ignores any unexpected notebook arguments (e.g., -f)
     19     args, _ = parser.parse_known_args()
---> 20     main(args)

/tmp/ipykernel_55/594845162.py in main(args)
      5 
      6     if HAVE_MODEL:
----> 7         model = AutoModelForCausalLM.from_pretrained(
      8             args.model_pth,
      9             torch_dtype=torch.bfloat16,

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/auto_factory.py in from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
    506             if not isinstance(config, PretrainedConfig):
    507                 # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
--> 508                 resolved_config_file = cached_file(
    509                     pretrained_model_name_or_path,
    510                     CONFIG_NAME,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    520 
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    521         # Now we try to recover if we can find all files correctly in the cache
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]
    525         if all(file is not None for file in resolved_files):

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    138 ):
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
    142         return resolved_file

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    104         ):
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 
    108             elif arg_name == "token" and arg_value is not None:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    158 
    159     if not REPO_ID_REGEX.match(repo_id):
--> 160         raise HFValidationError(
    161             "Repo id must use alphanumeric chars, '-', '_' or '.'."
    162             " The name cannot start or end with '-' or '.' and the maximum length is 96:"

HFValidationError: Repo id must use alphanumeric chars, '-', '_' or '.'. The name cannot start or end with '-' or '.' and the maximum length is 96: ''.
