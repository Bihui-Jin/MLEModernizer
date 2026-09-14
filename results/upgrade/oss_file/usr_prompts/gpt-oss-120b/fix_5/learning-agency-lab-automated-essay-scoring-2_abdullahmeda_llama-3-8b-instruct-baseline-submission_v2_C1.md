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

- What this solution (achieved 0.58755) has done: 'Implemented a defensive check that falls back to the length‑based heuristic whenever the optional model libraries are unavailable **or** the supplied model/LORA paths do not exist. This prevents the HFValidationError caused by trying to load a non‑existent model and guarantees a valid `submission.csv` is written. The core fallback logic and overall workflow remain unchanged.'

# 9. Code solution

## === cell 0
import gc
import argparse
import pandas as pd
import numpy as np
import os

try:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM

    HAVE_MODEL = True
except Exception as e:
    print(f"Optional libraries not available ({e}); will use fallback heuristic.")
    HAVE_MODEL = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
except Exception as e:
    print(f"scikit‑learn not available ({e}); will use length‑based fallback.")
    SKLEARN_AVAILABLE = False
else:
    SKLEARN_AVAILABLE = True


def main(args):
    data_dir = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
    df_test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

    model_path_exists = os.path.isdir(args.model_pth) if args.model_pth else False
    lora_path_exists = os.path.isdir(args.lora_pth) if args.lora_pth else False

    if not HAVE_MODEL or not (model_path_exists and lora_path_exists):
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

        if SKLEARN_AVAILABLE:
            pipeline = make_pipeline(
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
                Ridge(alpha=1.0, random_state=42),
            )
            pipeline.fit(train_df["full_text"].astype(str), train_df["score"])
            raw_preds = pipeline.predict(df_test["full_text"].astype(str))
            test_preds = np.clip(np.rint(raw_preds), 1, 6).astype(int).tolist()
            print("Used TF‑IDF + Ridge fallback model for predictions.")
        else:
            train_df["len"] = train_df["full_text"].astype(str).str.len()
            length_bins = pd.qcut(train_df["len"], q=6, duplicates="drop")
            bin_score_means = train_df.groupby(length_bins)["score"].mean()

            def predict_from_length(text):
                l = len(str(text))
                try:
                    bin_idx = pd.cut([l], bins=length_bins.cat.categories)[0]
                    pred = bin_score_means.loc[bin_idx]
                except Exception:
                    pred = train_df["score"].mode()[0]
                pred_int = int(round(pred))
                return max(1, min(6, pred_int))

            test_preds = (
                df_test["full_text"].apply(predict_from_length).astype(int).tolist()
            )
            print("Used length‑based fallback heuristic for predictions.")

        sub["score"] = pd.Series(test_preds, dtype=int)
        sub.to_csv(args.sub_pth, index=False)
        print(f"Fallback predictions written to {args.sub_pth}.")
        return

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

    def preprocess(
        sample, infer_mode=False, max_seq=args.max_length, return_tensors=None
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
        formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        if infer_mode:
            formatted = formatted.replace("<|eot_id|>", "")

        tokenized = tokenizer(
            formatted,
            padding=True,
            return_tensors=return_tensors,
            truncation=True,
            add_special_tokens=False,
            max_length=max_seq,
        )
        if return_tensors == "pt":
            tokenized["labels"] = tokenized["input_ids"].clone()
        else:
            tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    test_preds = []
    for _, row in df_test.iterrows():
        tokenized = preprocess(
            row, infer_mode=True, max_seq=args.max_length, return_tensors="pt"
        )
        generated_ids = model.generate(
            **tokenized.to("cuda"),
            max_new_tokens=2,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        try:
            answer = decoded[0].rsplit("The score is: ", 1)[1]
            test_preds.append(int(answer))
        except Exception:
            test_preds.append(3)  # safe fallback

    sub["score"] = pd.Series(test_preds, dtype=int)
    sub.to_csv(args.sub_pth, index=False)
    print(f"Model predictions written to {args.sub_pth}.")

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_pth", type=str, default="", help="Path to the pretrained model"
    )
    parser.add_argument(
        "--lora_pth", type=str, default="", help="Path to the PEFT LoRA adapter"
    )
    parser.add_argument(
        "--sub_pth", type=str, required=True, help="Path to save submission file"
    )
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Max length of input sequence"
    )
    args = parser.parse_args()
    main(args)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import sys

sys.argv = [
    "infer.py",
    "--sub_pth",
    "submission.csv",
    "--model_pth",
    "/does/not/exist",
    "--lora_pth",
    "/does/not/exist",
    "--max_length",
    "2048",
]

if __name__ == "__main__":
    pass

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_pth", type=str, default="", help="Path to the pretrained model"
)
parser.add_argument(
    "--lora_pth", type=str, default="", help="Path to the PEFT LoRA adapter"
)
parser.add_argument(
    "--sub_pth", type=str, required=True, help="Path to save submission file"
)
parser.add_argument(
    "--max_length", type=int, default=2048, help="Max length of input sequence"
)
args = parser.parse_args()

main(args)
print("Submission file created:", os.path.abspath(args.sub_pth))

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    128             try:
--> 129                 coefs[i], info = sp_linalg.cg(
    130                     C, y_column, maxiter=max_iter, tol=tol, atol="legacy"

TypeError: cg() got an unexpected keyword argument 'tol'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/408312660.py in <cell line: 0>()
     32 args = parser.parse_args()
     33 
---> 34 main(args)
     35 print("Submission file created:", os.path.abspath(args.sub_pth))

/tmp/ipykernel_55/1181182219.py in main(args)
     54             )
     55             # Fit on the full training data.
---> 56             pipeline.fit(train_df["full_text"].astype(str), train_df["score"])
     57             # Predict on test data.
     58             raw_preds = pipeline.predict(df_test["full_text"].astype(str))

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in fit(self, X, y, **fit_params)
    403             if self._final_estimator != "passthrough":
    404                 fit_params_last_step = fit_params_steps[self.steps[-1][0]]
--> 405                 self._final_estimator.fit(Xt, y, **fit_params_last_step)
    406 
    407         return self

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in fit(self, X, y, sample_weight)
   1132             y_numeric=True,
   1133         )
-> 1134         return super().fit(X, y, sample_weight=sample_weight)
   1135 
   1136 

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in fit(self, X, y, sample_weight)
    898                 params = {}
    899 
--> 900             self.coef_, self.n_iter_ = _ridge_regression(
    901                 X,
    902                 y,

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _ridge_regression(X, y, alpha, sample_weight, solver, max_iter, tol, verbose, positive, random_state, return_n_iter, return_intercept, X_scale, X_offset, check_input, fit_intercept)
    669     n_iter = None
    670     if solver == "sparse_cg":
--> 671         coef = _solve_sparse_cg(
    672             X,
    673             y,

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    132             except TypeError:
    133                 # old scipy
--> 134                 coefs[i], info = sp_linalg.cg(C, y_column, maxiter=max_iter, tol=tol)
    135 
    136         if info < 0:

TypeError: cg() got an unexpected keyword argument 'tol'
