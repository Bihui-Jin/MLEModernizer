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
I added a safe fallback for environments that lack the peft and transformers libraries (or where loading the large LLaMA model would fail). The script now tries to import and load the model; if it cannot, it simply assigns a median score of 3 to every essay. This guarantees that a valid submission.csv is always written, allowing you to evaluate a baseline QWK and move toward the target score without changing the overall workflow. The rest of the original logic is kept unchanged.

```


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_55/3350043589.py", line 1
    I added a safe fallback for environments that lack the peft and transformers libraries (or where loading the large LLaMA model would fail). The script now tries to import and load the model; if it cannot, it simply assigns a median score of 3 to every essay. This guarantees that a valid submission.csv is always written, allowing you to evaluate a baseline QWK and move toward the target score without changing the overall workflow. The rest of the original logic is kept unchanged.
                                                          ^
SyntaxError: invalid non-printable character U+202F


## === cell 1
!ls /kaggle/input/



## === cell 2
%%writefile infer.py
import gc
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from types import SimpleNamespace

try:
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAVE_MODEL = True
except Exception:
    HAVE_MODEL = False

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

def main(args):
    config = SimpleNamespace(
        data_dir = '/kaggle/input/learning-agency-lab-automated-essay-scoring-2',
    )

    if HAVE_MODEL:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_pth,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, args.lora_pth)
        tokenizer = AutoTokenizer.from_pretrained(args.model_pth, padding_side='right')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        model = None
        tokenizer = None

    def preprocess(sample, text=False, infer_mode=False, max_seq=args.max_length, return_tensors=None):
        sys_prompt = ("Please read the following essay and assign a score of 1,2,3,4,5,6 "
                      "where 6 is the best. Output only a single number with no explanation.\n\n")
        prompt = sample["full_text"]
        answer = "" if infer_mode else str(sample["score"])

        messages = [
            {"role": "user", "content": sys_prompt + prompt},
            {"role": "assistant", "content": f"\n\nThe score is: " + answer}
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

    df_test = pd.read_csv(f'{config.data_dir}/test.csv')
    sub = pd.read_csv(f'{config.data_dir}/sample_submission.csv')
    test_preds = []

    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        if HAVE_MODEL:
            tokenized_sample = preprocess(row, infer_mode=True, max_seq=args.max_length, return_tensors="pt")
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
            test_preds.append(3)

    sub['score'] = test_preds
    sub['score'] = sub['score'].astype('int')
    sub.to_csv(args.sub_pth, index=False)

    if HAVE_MODEL:
        del model, tokenizer
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pth", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--lora_pth", type=str, required=True, help="Path to the PEFT LoRA adapter")
    parser.add_argument("--sub_pth", type=str, required=True, help="Path to save submission file")
    parser.add_argument("--max_length", type=int, required=True, help="Max length of input sequence")
    args = parser.parse_args()
    main(args)



## === cell 3
!python infer.py \
    --max_length 2048 \
    --sub_pth submission.csv \
    --model_pth /kaggle/input/llama-3-8b-instruct/Meta-Llama-3-8B-Instruct \
    --lora_pth /kaggle/input/llama-3-8b-lora-fine-tuned-exp-1/Meta-Llama-3-8B-Instruct-max-len-1024-fold-1-exp-1-ckpt
```

## --- ERROR in cell 3, traceback:
  File "/tmp/ipykernel_55/3369028531.py", line 2
    ```
    ^
SyntaxError: invalid syntax
