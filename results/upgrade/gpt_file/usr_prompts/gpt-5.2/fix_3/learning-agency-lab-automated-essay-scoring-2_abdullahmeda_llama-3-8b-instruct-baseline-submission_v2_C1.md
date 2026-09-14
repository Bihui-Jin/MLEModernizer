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
import os, sys, re, gc, argparse
import pandas as pd


def heuristic_fallback_submission(data_dir: str, sub_pth: str) -> None:
    """
    Minimal, dependency-free fallback that creates a valid submission using a simple
    text-length heuristic calibrated on train.csv. This is only used when transformers/peft
    and/or CUDA aren't available, so that a score is yielded.
    """
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    sample_path = os.path.join(data_dir, "sample_submission.csv")

    train = pd.read_csv(train_path, usecols=["full_text", "score"])
    test = pd.read_csv(test_path, usecols=["essay_id", "full_text"])
    sub = pd.read_csv(sample_path, usecols=["essay_id", "score"])

    tr_len = train["full_text"].fillna("").astype(str).str.len()
    te_len = test["full_text"].fillna("").astype(str).str.len()

    qs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    edges = tr_len.quantile(qs).to_numpy()
    edges = sorted(set(int(x) for x in edges))
    if len(edges) < 3:
        global_pred = int(round(float(train["score"].mean())))
        pred = pd.Series([global_pred] * len(test))
    else:
        edges[0] = max(edges[0], 0)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1

        tr_bins = pd.cut(tr_len, bins=edges, include_lowest=True, duplicates="drop")
        bin_mean = train.groupby(tr_bins, observed=False)["score"].mean()

        te_bins = pd.cut(te_len, bins=edges, include_lowest=True, duplicates="drop")
        pred = te_bins.map(bin_mean)

        pred = pred.fillna(train["score"].mean())
        pred = pred.round().astype(int)

    pred = pred.clip(1, 6)

    pred_df = pd.DataFrame({"essay_id": test["essay_id"].values, "score": pred.values})
    sub = sub[["essay_id"]].merge(pred_df, on="essay_id", how="left")
    sub["score"] = sub["score"].fillna(3).astype(int).clip(1, 6)

    sub.to_csv(sub_pth, index=False)


def main(args):
    data_dir = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"

    try:
        import torch
        from tqdm import tqdm
        from peft import PeftModel
        from types import SimpleNamespace
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)

        config = SimpleNamespace(data_dir=data_dir)

        model = AutoModelForCausalLM.from_pretrained(
            args.model_pth,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, args.lora_pth)
        tokenizer = AutoTokenizer.from_pretrained(args.model_pth, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token

        def preprocess(
            sample,
            text=False,
            infer_mode=False,
            max_seq=args.max_length,
            return_tensors=None,
        ):
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

            return formatted_sample if text else tokenized_sample

        df_test = pd.read_csv(f"{config.data_dir}/test.csv")
        sub = pd.read_csv(f"{config.data_dir}/sample_submission.csv")

        test_preds = []

        for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
            tokenized_sample = preprocess(
                row, infer_mode=True, max_seq=args.max_length, return_tensors="pt"
            )

            device = next(model.parameters()).device
            tokenized_sample = {k: v.to(device) for k, v in tokenized_sample.items()}

            generated_ids = model.generate(
                **tokenized_sample,
                max_new_tokens=2,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            pred = 3
            try:
                if "The score is:" in decoded:
                    tail = decoded.rsplit("The score is:", 1)[1]
                elif "The score is: " in decoded:
                    tail = decoded.rsplit("The score is: ", 1)[1]
                else:
                    tail = decoded

                m = re.search(r"\b([1-6])\b", tail)
                if m:
                    pred = int(m.group(1))
                else:
                    m2 = re.search(r"(\d)", tail)
                    if m2:
                        pred = int(m2.group(1))
            except Exception:
                pred = 3

            pred = max(1, min(6, int(pred)))
            test_preds.append(pred)

        sub["score"] = pd.Series(test_preds).astype(int).clip(1, 6)
        sub.to_csv(args.sub_pth, index=False)

        del model, tokenizer
        if "torch" in sys.modules and hasattr(sys.modules["torch"], "cuda"):
            try:
                sys.modules["torch"].cuda.empty_cache()
            except Exception:
                pass
        gc.collect()

    except Exception as e:
        print(
            f"[WARN] Falling back to heuristic submission due to: {type(e).__name__}: {e}"
        )
        heuristic_fallback_submission(data_dir=data_dir, sub_pth=args.sub_pth)


def _is_interactive():
    return (
        ("ipykernel" in sys.modules)
        or ("KAGGLE_KERNEL_RUN_TYPE" in os.environ)
        or ("__file__" not in globals())
    )


if __name__ == "__main__" and not _is_interactive():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_pth", type=str, required=True, help="Path to the pretrained model"
    )
    parser.add_argument(
        "--lora_pth", type=str, required=True, help="Path to the PEFT LoRA adapter"
    )
    parser.add_argument(
        "--sub_pth", type=str, required=True, help="Path to save submission file"
    )
    parser.add_argument(
        "--max_length", type=int, required=True, help="Max length of input sequence"
    )
    args = parser.parse_args()
    main(args)
else:

    class _Args:
        model_pth = "/kaggle/input/llama-3-8b-instruct/Meta-Llama-3-8B-Instruct"
        lora_pth = "/kaggle/input/llama-3-8b-lora-fine-tuned-exp-1/Meta-Llama-3-8B-Instruct-max-len-1024-fold-1-exp-1-ckpt"
        sub_pth = "submission.csv"
        max_length = 2048

    main(_Args())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from pathlib import Path
import subprocess, shlex

infer_py = r"""import os, sys, re, gc, argparse
import pandas as pd

def heuristic_fallback_submission(data_dir: str, sub_pth: str) -> None:
    train_path = os.path.join(data_dir, "train.csv")
    test_path  = os.path.join(data_dir, "test.csv")
    sample_path = os.path.join(data_dir, "sample_submission.csv")

    train = pd.read_csv(train_path, usecols=["full_text", "score"])
    test  = pd.read_csv(test_path, usecols=["essay_id", "full_text"])
    sub   = pd.read_csv(sample_path, usecols=["essay_id", "score"])

    tr_len = train["full_text"].fillna("").astype(str).str.len()
    te_len = test["full_text"].fillna("").astype(str).str.len()

    qs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    edges = tr_len.quantile(qs).to_numpy()
    edges = sorted(set(int(x) for x in edges))
    if len(edges) < 3:
        global_pred = int(round(float(train["score"].mean())))
        pred = pd.Series([global_pred] * len(test))
    else:
        edges[0] = max(edges[0], 0)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i-1]:
                edges[i] = edges[i-1] + 1

        tr_bins = pd.cut(tr_len, bins=edges, include_lowest=True, duplicates="drop")
        bin_mean = train.groupby(tr_bins, observed=False)["score"].mean()

        te_bins = pd.cut(te_len, bins=edges, include_lowest=True, duplicates="drop")
        pred = te_bins.map(bin_mean)
        pred = pred.fillna(train["score"].mean())
        pred = pred.round().astype(int)

    pred = pred.clip(1, 6)

    pred_df = pd.DataFrame({"essay_id": test["essay_id"].values, "score": pred.values})
    sub = sub[["essay_id"]].merge(pred_df, on="essay_id", how="left")
    sub["score"] = sub["score"].fillna(3).astype(int).clip(1, 6)
    sub.to_csv(sub_pth, index=False)


def main(args):
    data_dir = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"

    try:
        import torch
        from tqdm import tqdm
        from peft import PeftModel
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)

        model = AutoModelForCausalLM.from_pretrained(
            args.model_pth,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, args.lora_pth)
        tokenizer = AutoTokenizer.from_pretrained(args.model_pth, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token

        def preprocess(sample, text=False, infer_mode=False, max_seq=args.max_length, return_tensors=None):
            sys_prompt = (
                "Please read the following essay and assign a score of 1,2,3,4,5,6 where 6 is the best. "
                "Output only a single number with no explanation.\\n\\n"
            )
            prompt = sample["full_text"]
            answer = "" if infer_mode else str(sample["score"])

            messages = [
                {"role": "user", "content": sys_prompt + prompt},
                {"role": "assistant", "content": f"\\n\\nThe score is: " + answer},
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

            return formatted_sample if text else tokenized_sample

        df_test = pd.read_csv(f"{data_dir}/test.csv")
        sub = pd.read_csv(f"{data_dir}/sample_submission.csv")

        test_preds = []
        for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
            tokenized_sample = preprocess(row, infer_mode=True, max_seq=args.max_length, return_tensors="pt")
            device = next(model.parameters()).device
            tokenized_sample = {k: v.to(device) for k, v in tokenized_sample.items()}

            generated_ids = model.generate(
                **tokenized_sample,
                max_new_tokens=2,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            pred = 3
            try:
                if "The score is:" in decoded:
                    tail = decoded.rsplit("The score is:", 1)[1]
                elif "The score is: " in decoded:
                    tail = decoded.rsplit("The score is: ", 1)[1]
                else:
                    tail = decoded

                m = re.search(r"\\b([1-6])\\b", tail)
                if m:
                    pred = int(m.group(1))
                else:
                    m2 = re.search(r"(\\d)", tail)
                    if m2:
                        pred = int(m2.group(1))
            except Exception:
                pred = 3

            pred = max(1, min(6, int(pred)))
            test_preds.append(pred)

        sub["score"] = pd.Series(test_preds).astype(int).clip(1, 6)
        sub.to_csv(args.sub_pth, index=False)

        del model, tokenizer
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    except Exception as e:
        print(f"[WARN] Falling back to heuristic submission due to: {type(e).__name__}: {e}")
        heuristic_fallback_submission(data_dir=data_dir, sub_pth=args.sub_pth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pth", type=str, required=True)
    parser.add_argument("--lora_pth", type=str, required=True)
    parser.add_argument("--sub_pth", type=str, required=True)
    parser.add_argument("--max_length", type=int, required=True)
    args = parser.parse_args()
    main(args)
"""
Path("infer.py").write_text(infer_py)

cmd = """python infer.py \
    --max_length 2048 \
    --sub_pth submission.csv \
    --model_pth /kaggle/input/llama-3-8b-instruct/Meta-Llama-3-8B-Instruct \
    --lora_pth /kaggle/input/llama-3-8b-lora-fine-tuned-exp-1/Meta-Llama-3-8B-Instruct-max-len-1024-fold-1-exp-1-ckpt
"""

try:
    out = subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT).decode(
        "utf-8", errors="ignore"
    )
    print(out)
except subprocess.CalledProcessError as e:
    print("[WARN] infer.py failed; captured output:\n")
    print(e.output.decode("utf-8", errors="ignore"))

if not os.path.exists("submission.csv"):
    print("[WARN] submission.csv missing; writing heuristic fallback submission.")
    heuristic_fallback_submission(
        data_dir="/kaggle/input/learning-agency-lab-automated-essay-scoring-2",
        sub_pth="submission.csv",
    )

sub = pd.read_csv("submission.csv")
print(sub.head())
print(sub.shape)
print(sub.columns.tolist())
print(sub["score"].min(), sub["score"].max())
assert list(sub.columns) == ["essay_id", "score"]
assert (
    sub.shape[0]
    == pd.read_csv(
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
    ).shape[0]
)
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv")

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4077097846.py in <cell line: 0>()
    185 if not os.path.exists("submission.csv"):
    186     print("[WARN] submission.csv missing; writing heuristic fallback submission.")
--> 187     heuristic_fallback_submission(
    188         data_dir="/kaggle/input/learning-agency-lab-automated-essay-scoring-2",
    189         sub_pth="submission.csv",

/tmp/ipykernel_55/1866343007.py in heuristic_fallback_submission(data_dir, sub_pth)
     38         pred = te_bins.map(bin_mean)
     39 
---> 40         pred = pred.fillna(train["score"].mean())
     41         pred = pred.round().astype(int)
     42 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in fillna(self, value, method, axis, inplace, limit, downcast)
   7347                     )
   7348 
-> 7349                 new_data = self._mgr.fillna(
   7350                     value=value, limit=limit, inplace=inplace, downcast=downcast
   7351                 )

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/base.py in fillna(self, value, limit, inplace, downcast)
    184             limit = libalgos.validate_limit(None, limit=limit)
    185 
--> 186         return self.apply_with_block(
    187             "fillna",
    188             value=value,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in fillna(self, value, limit, inplace, downcast, using_cow, already_warned)
   2332                 # 3rd party EA that has not implemented copy keyword yet
   2333                 refs = None
-> 2334                 new_values = self.values.fillna(value=value, method=None, limit=limit)
   2335                 # issue the warning *after* retrying, in case the TypeError
   2336                 #  was caused by an invalid fill_value

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/_mixins.py in fillna(self, value, method, limit, copy)
    374             # We validate the fill_value even if there is nothing to fill
    375             if value is not None:
--> 376                 self._validate_setitem_value(value)
    377 
    378             if not copy:

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/categorical.py in _validate_setitem_value(self, value)
   1587             return self._validate_listlike(value)
   1588         else:
-> 1589             return self._validate_scalar(value)
   1590 
   1591     def _validate_scalar(self, fill_value):

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/categorical.py in _validate_scalar(self, fill_value)
   1612             fill_value = self._unbox_scalar(fill_value)
   1613         else:
-> 1614             raise TypeError(
   1615                 "Cannot setitem on a Categorical with a new "
   1616                 f"category ({fill_value}), set the categories first"

TypeError: Cannot setitem on a Categorical with a new category (2.950115562403698), set the categories first
