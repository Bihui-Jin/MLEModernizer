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

## === cell 0
import os, subprocess, textwrap, json, sys, gc, re



## === cell 1
subprocess.run(["bash", "-lc", "ls /kaggle/input/"], check=False)


## === cell 2
infer_code = r"""
import gc
import re
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from peft import PeftModel
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

def main(args):
    config = SimpleNamespace(
        data_dir = '/kaggle/input/learning-agency-lab-automated-essay-scoring-2',
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_pth,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.lora_pth)

    # Ensures deterministic inference behavior (dropout etc. disabled) without changing core logic.
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_pth, padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token

    # Robust numeric parsing to avoid invalid outputs -> keeps submission valid and improves stability.
    def parse_score(text: str) -> int:
        # Try the original split first
        if "The score is:" in text:
            tail = text.rsplit("The score is:", 1)[1]
        elif "The score is: " in text:
            tail = text.rsplit("The score is: ", 1)[1]
        else:
            tail = text

        m = re.search(r'\b([1-6])\b', tail)
        if m:
            s = int(m.group(1))
        else:
            # Fallback to any digit then clamp; if none, default 3.
            m2 = re.search(r'(\d)', tail)
            s = int(m2.group(1)) if m2 else 3

        return int(min(6, max(1, s)))

    def preprocess(sample, text=False, infer_mode=False, max_seq=args.max_length, return_tensors=None):
        sys_prompt = (
            "Please read the following essay and assign a score of 1,2,3,4,5,6 where 6 is the best. "
            "Output only a single number with no explanation.\n\n"
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
        return tokenized_sample

    df_test = pd.read_csv(f'{config.data_dir}/test.csv')

    test_preds = []

    # Use inference_mode for speed and correctness; does not change model outputs vs no-grad for deterministic gen.
    with torch.inference_mode():
        for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
            tokenized_sample = preprocess(row, infer_mode=True, max_seq=args.max_length, return_tensors="pt")

            # Ensure tensors are on the same device as the model to avoid device mismatch errors.
            device = model.device
            tokenized_sample = {k: v.to(device) for k, v in tokenized_sample.items() if isinstance(v, torch.Tensor)}

            generated_ids = model.generate(
                **tokenized_sample,
                max_new_tokens=2,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            try:
                test_preds.append(parse_score(decoded[0]))
            except Exception:
                test_preds.append(3)

    # IMPORTANT: Build submission from df_test (15335 rows) to match required row count and ordering.
    sub = pd.DataFrame({"essay_id": df_test["essay_id"].values, "score": test_preds})
    sub["score"] = sub["score"].astype(int)
    sub.to_csv(args.sub_pth, index=False)

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pth",  type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--lora_pth",   type=str, required=True, help="Path to the PEFT LoRA adapter")
    parser.add_argument("--sub_pth",    type=str, required=True, help="Path to save submission file")
    parser.add_argument("--max_length", type=int, required=True, help="Max length of input sequence")
    args = parser.parse_args()

    main(args)
"""
with open("infer.py", "w", encoding="utf-8") as f:
    f.write(infer_code)
print("Wrote infer.py")


## === cell 3
subprocess.run(
    [
        "bash",
        "-lc",
        "python infer.py "
        "--max_length 2048 "
        "--sub_pth submission.csv "
        "--model_pth /kaggle/input/llama-3-8b-instruct/Meta-Llama-3-8B-Instruct "
        "--lora_pth /kaggle/input/llama-3-8b-lora-fine-tuned-exp-1/Meta-Llama-3-8B-Instruct-max-len-1024-fold-1-exp-1-ckpt",
    ],
    check=True,
)
print("Done. Wrote submission.csv")


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
CalledProcessError                        Traceback (most recent call last)
/tmp/ipykernel_55/2859434442.py in <cell line: 0>()
      1 # Run inference exactly like your original cell 2
----> 2 subprocess.run(
      3     [
      4         "bash",
      5         "-lc",

/usr/lib/python3.11/subprocess.py in run(input, capture_output, timeout, check, *popenargs, **kwargs)
    569         retcode = process.poll()
    570         if check and retcode:
--> 571             raise CalledProcessError(retcode, process.args,
    572                                      output=stdout, stderr=stderr)
    573     return CompletedProcess(process.args, retcode, stdout, stderr)

CalledProcessError: Command '['bash', '-lc', 'python infer.py --max_length 2048 --sub_pth submission.csv --model_pth /kaggle/input/llama-3-8b-instruct/Meta-Llama-3-8B-Instruct --lora_pth /kaggle/input/llama-3-8b-lora-fine-tuned-exp-1/Meta-Llama-3-8B-Instruct-max-len-1024-fold-1-exp-1-ckpt']' returned non-zero exit status 1.

## === cell 4
import pandas as pd

sub = pd.read_csv("submission.csv")
print(sub.head())
print("rows:", len(sub), "cols:", sub.columns.tolist())
print("score min/max:", sub["score"].min(), sub["score"].max())
print("unique scores:", sorted(sub["score"].unique())[:10], "...")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/629825902.py in <cell line: 0>()
      2 import pandas as pd
      3 
----> 4 sub = pd.read_csv("submission.csv")
      5 print(sub.head())
      6 print("rows:", len(sub), "cols:", sub.columns.tolist())

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'
