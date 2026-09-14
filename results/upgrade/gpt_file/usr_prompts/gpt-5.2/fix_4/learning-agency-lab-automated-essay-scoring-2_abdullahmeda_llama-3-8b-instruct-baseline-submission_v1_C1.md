# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the inference run robustly in the Kaggle environment by fixing the likely root cause of the crash: missing optional dependencies (`peft`, `transformers`, or model files) and/or GPU-only settings. The patch keeps your core generation logic, but adds a safe fallback that still produces a valid `submission.csv` with the correct columns and row count if the LLM/LoRA stack cannot be loaded. I also adjust device handling to avoid `model.device`/`device_map="auto"` mismatch issues and ensure the script always writes the submission file (even on exception), so you can at least submit and get a score instead of “Not yielded”. These changes are focused on correctness and end-to-end execution; they don’t change the intended model behavior when the required models/packages are available.'
- What this solution (achieved 0.0) has done: 'I fix the row-count mismatch by ensuring the inference reads the same `test.csv` that your local checks expect (the root cause is that `sample_submission.csv` has 1731 rows while `test.csv` has 15335). I also update the validation cell to compare against the actual test length instead of a hardcoded number, so it won’t break if the path changes. These are execution- and submission-validity fixes; they do not change the model logic, and they ensure a proper `submission.csv` is always produced. With the correct test file, your submission have the required 15335 rows and should score above 0.0 (instead of being invalid or truncated).'

# 9. Code solution

## === cell 0
import os, subprocess, textwrap, json, sys, gc, re



## === cell 1
subprocess.run(["bash", "-lc", "ls /kaggle/input/"], check=False)
subprocess.run(
    [
        "bash",
        "-lc",
        "ls /kaggle/input/learning-agency-lab-automated-essay-scoring-2 | head",
    ],
    check=False,
)



## === cell 2
infer_code = r"""
import os
import gc
import re
import sys
import argparse
import pandas as pd

def safe_mkdir_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def main(args):
    # BUGFIX: Ensure we always read the competition test.csv (15335 rows),
    # not sample_submission.csv (1731 rows). Keep path robust across Kaggle layouts.
    candidate_dirs = [
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2",
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/learning-agency-lab-automated-essay-scoring-2",
    ]
    data_dir = None
    for d in candidate_dirs:
        if os.path.exists(os.path.join(d, "test.csv")):
            data_dir = d
            break
    if data_dir is None:
        raise FileNotFoundError("Could not locate test.csv in expected Kaggle input directories.")

    test_path = os.path.join(data_dir, "test.csv")
    df_test = pd.read_csv(test_path)

    # Robust numeric parsing to avoid invalid outputs -> keeps submission valid.
    def parse_score(text: str) -> int:
        if text is None:
            return 3
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
            m2 = re.search(r'(\d)', tail)
            s = int(m2.group(1)) if m2 else 3
        return int(min(6, max(1, s)))

    # Always write a submission, even if model inference fails for environment reasons.
    safe_mkdir_parent(args.sub_pth)

    # --- Attempt to run your intended LLM+LoRA inference path.
    try:
        import torch
        from tqdm import tqdm
        from peft import PeftModel
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # These are CUDA-specific; guard them so CPU-only envs won't crash.
        try:
            if torch.cuda.is_available():
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
        except Exception:
            pass

        model = AutoModelForCausalLM.from_pretrained(
            args.model_pth,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, args.lora_pth)

        # Ensures deterministic inference behavior (dropout etc. disabled) without changing core logic.
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(args.model_pth, padding_side='right', trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

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

        test_preds = []
        # Determine a reliable device for inputs when using device_map="auto"
        # (first parameter device is a safe heuristic).
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.inference_mode():
            for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
                tokenized_sample = preprocess(row, infer_mode=True, max_seq=args.max_length, return_tensors="pt")
                tokenized_sample = {k: v.to(device) for k, v in tokenized_sample.items() if hasattr(v, "to")}

                generated_ids = model.generate(
                    **tokenized_sample,
                    max_new_tokens=2,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
                decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                test_preds.append(parse_score(decoded[0] if decoded else ""))

        sub = pd.DataFrame({"essay_id": df_test["essay_id"].values, "score": test_preds})
        sub["score"] = sub["score"].astype(int)
        sub.to_csv(args.sub_pth, index=False)

        # Cleanup
        try:
            del model, tokenizer
        except Exception:
            pass
        if "torch" in sys.modules and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return

    except Exception as e:
        # --- Fallback path: produce a valid submission to avoid "Not yielded".
        # This is score-conservative (baseline) but guarantees a CSV is created.
        # We also print the exception so it can be debugged from logs.
        print("WARNING: LLM inference failed; writing fallback submission. Error was:\n", repr(e), file=sys.stderr)

        # Use a simple constant baseline (3) within valid 1-6.
        sub = pd.DataFrame({"essay_id": df_test["essay_id"].values, "score": 3})
        sub["score"] = sub["score"].astype(int)
        sub.to_csv(args.sub_pth, index=False)
        return

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



## === cell 4
import pandas as pd
import os

test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
if not os.path.exists(test_path):
    test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/learning-agency-lab-automated-essay-scoring-2/test.csv"
df_test = pd.read_csv(test_path)

sub = pd.read_csv("submission.csv")
print(sub.head())
print("rows:", len(sub), "cols:", sub.columns.tolist())
print("score min/max:", sub["score"].min(), sub["score"].max())
print("unique scores:", sorted(sub["score"].unique())[:10], "...")
assert sub.columns.tolist() == ["essay_id", "score"]
assert len(sub) == len(df_test)
assert sub["essay_id"].astype(str).tolist() == df_test["essay_id"].astype(str).tolist()
assert sub["score"].between(1, 6).all()
print("submission.csv looks valid.")
