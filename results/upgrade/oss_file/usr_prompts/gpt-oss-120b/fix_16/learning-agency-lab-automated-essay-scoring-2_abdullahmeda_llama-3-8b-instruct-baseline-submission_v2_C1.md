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

0.62883

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.58755) has done: 'Implemented a defensive check that falls back to the length‑based heuristic whenever the optional model libraries are unavailable **or** the supplied model/LORA paths do not exist. This prevents the HFValidationError caused by trying to load a non‑existent model and guarantees a valid `submission.csv` is written. The core fallback logic and overall workflow remain unchanged.'
- What this solution (achieved 0.58755) has done: 'Implemented a fix for the `UnboundLocalError` by avoiding assignment to the module‑level `SKLEARN_AVAILABLE` inside `main`. A local flag `use_sklearn` now governs whether the TF‑IDF + Ridge fallback runs, preserving the original fallback logic while keeping variable scope correct. This resolves the runtime error and ensures a valid `submission.csv` is produced.'
- What this solution (achieved 0.58519) has done: 'Implemented a lightweight linear‑regression fallback on essay length to replace the coarse binning heuristic. This uses only NumPy (no sklearn) and predicts scores from essay length, then rounds and clips to the 1‑6 range. The change is confined to the fallback branch, preserving all original logic while providing a more informative model, expected to raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.59518) has done: 'Implemented a safe import strategy that avoids loading optional `peft` and `transformers` libraries, preventing the protobuf AttributeError. Replaced the simple length‑based linear regression fallback with a more expressive length‑binning approach that uses many quantile buckets to map essay length to average scores, which yields a higher quadratic weighted kappa. Added clear comments and retained all original functionality and paths, ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.59518) has done: 'Implemented a safe‑argument fix to stop the script from exiting when run in a notebook environment. The `--sub_pth` argument is now optional with a default filename, preventing the `SystemExit: 2` raised by argparse. This change lets the fallback heuristic execute and reliably creates `submission.csv` without altering core modeling logic or score‑related behavior.'
- What this solution (achieved 0.58519) has done: 'Implemented two key fixes:
1. **Enhanced fallback model:** When sklearn isn’t available, the script now fits a simple linear‑regression on essay length (using only NumPy). This replaces the coarse length‑binning heuristic and yields noticeably better quadratic weighted kappa scores while keeping the original fallback workflow intact.
2. **Argument parser correction:** Adjusted the secondary test cell’s parser so `sub_pth` is optional with a default, eliminating the `SystemExit: 2` error caused by a required argument that wasn’t supplied.

These changes ensure the script runs end‑to‑end, produces a valid `submission.csv`, and moves the validation score toward the target.'
- What this solution (achieved 0.62867) has done: 'The script now avoids crashes caused by unknown CLI arguments by using `parse_known_args`, and the fallback model has been upgraded: it fits a linear regression on both character length and word count (instead of just length) for more informative predictions. This modest enhancement is expected to raise the quadratic weighted kappa toward the target while preserving all original logic and ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 0.62883) has done: 'I strengthen the fallback prediction pipeline while keeping the overall structure unchanged.  
- Increase TF‑IDF capacity (max_features = 10000) and use a slightly smaller Ridge α (0.5) for a more expressive linear model when sklearn is available.  
- Enhance the pure‑numpy regression fallback by adding an average‑word‑length feature, giving the linear system three predictors (char length, word count, avg word length) instead of only two.  
These tweaks are lightweight, stay within the original logic, and are expected to raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.62883) has done: 'I enhance the fallback when scikit‑learn is available by expanding the TF‑IDF model (more features, sublinear term weighting) and blending its predictions with the simple length‑based regression. This keeps the original workflow while adding a lightweight, deterministic improvement expected to raise the quadratic weighted kappa closer to the target score.'

# 9. Code solution

## === cell 0
import gc
import argparse
import pandas as pd
import numpy as np
import os
import sys

try:
    import torch

    HAVE_TORCH = True
except Exception as e:
    print(f"torch not available ({e}); will use fallback heuristic.")
    HAVE_TORCH = False

HAVE_MODEL = HAVE_TORCH  # only true if torch imported; otherwise fallback.

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    SKLEARN_AVAILABLE = True
except Exception as e:
    print(f"scikit‑learn not available ({e}); will use length‑based fallback.")
    SKLEARN_AVAILABLE = False


def _length_regression_predictions(train_df, test_df):
    """
    Train a tiny linear regression on three simple text‑length features
    (char count, word count, avg word length) and return predictions
    for the test set.
    """
    train_df["char_len"] = train_df["full_text"].astype(str).str.len()
    train_df["word_cnt"] = train_df["full_text"].astype(str).str.split().apply(len)
    train_df["avg_word_len"] = train_df["char_len"] / train_df["word_cnt"].replace(
        0, np.nan
    )

    median_char_len = train_df["char_len"].median()
    train_df["avg_word_len"] = train_df["avg_word_len"].fillna(median_char_len)

    X = np.vstack(
        [
            np.ones_like(train_df["char_len"].values, dtype=np.float64),
            train_df["char_len"].values.astype(np.float64),
            train_df["word_cnt"].values.astype(np.float64),
            train_df["avg_word_len"].values.astype(np.float64),
        ]
    ).T
    y = train_df["score"].values.astype(np.float64)

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    test_char_len = test_df["full_text"].astype(str).str.len().values.astype(np.float64)
    test_word_cnt = (
        test_df["full_text"]
        .astype(str)
        .str.split()
        .apply(len)
        .values.astype(np.float64)
    )
    test_avg_word_len = test_char_len / np.where(
        test_word_cnt == 0, np.nan, test_word_cnt
    )
    test_avg_word_len = np.where(
        np.isnan(test_avg_word_len), median_char_len, test_avg_word_len
    )

    X_test = np.vstack(
        [
            np.ones_like(test_char_len),
            test_char_len,
            test_word_cnt,
            test_avg_word_len,
        ]
    ).T
    preds = X_test @ coeffs
    return np.clip(np.rint(preds), 1, 6).astype(int)


def main(args):
    data_dir = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
    df_test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

    model_path_exists = os.path.isdir(args.model_pth) if args.model_pth else False
    lora_path_exists = os.path.isdir(args.lora_pth) if args.lora_pth else False

    if not HAVE_MODEL or not (model_path_exists and lora_path_exists):
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

        if SKLEARN_AVAILABLE:
            try:
                pipeline = make_pipeline(
                    TfidfVectorizer(
                        max_features=20000,
                        ngram_range=(1, 3),
                        stop_words="english",
                        sublinear_tf=True,
                    ),
                    Ridge(alpha=0.5, random_state=42, solver="svd"),
                )
                pipeline.fit(train_df["full_text"].astype(str), train_df["score"])
                tfidf_raw_preds = pipeline.predict(df_test["full_text"].astype(str))

                len_preds = _length_regression_predictions(train_df, df_test)

                blended = np.rint(0.8 * tfidf_raw_preds + 0.2 * len_preds)
                test_preds = np.clip(blended, 1, 6).astype(int).tolist()

                print(
                    "Used TF‑IDF + Ridge blended with length regression for predictions."
                )
                sub["score"] = pd.Series(test_preds, dtype=int)
                sub.to_csv(args.sub_pth, index=False)
                print(f"Fallback predictions written to {args.sub_pth}.")
                return
            except Exception as e:
                print(
                    f"TF‑IDF + Ridge blend failed ({e}); falling back to length‑only model."
                )

        try:
            test_preds = _length_regression_predictions(train_df, df_test).tolist()
            print(
                "Used length + word‑count + avg‑word‑len regression fallback for predictions."
            )
            sub["score"] = pd.Series(test_preds, dtype=int)
            sub.to_csv(args.sub_pth, index=False)
            print(f"Fallback predictions written to {args.sub_pth}.")
            return
        except Exception as e:
            print(f"Length regression failed ({e}); using binning fallback.")

        train_df["len"] = train_df["full_text"].astype(str).str.len()
        n_bins = min(100, train_df["len"].nunique())
        length_bins = pd.qcut(train_df["len"], q=n_bins, duplicates="drop")
        bin_score_means = train_df.groupby(length_bins)["score"].mean()
        bin_edges = length_bins.cat.categories

        def predict_from_length(text):
            l = len(str(text))
            try:
                bin_idx = pd.cut([l], bins=bin_edges)[0]
                pred = bin_score_means.loc[bin_idx]
            except Exception:
                pred = train_df["score"].median()
            return int(np.clip(round(pred), 1, 6))

        test_preds = (
            df_test["full_text"].apply(predict_from_length).astype(int).tolist()
        )
        print("Used enhanced length‑based binning fallback for predictions.")
        sub["score"] = pd.Series(test_preds, dtype=int)
        sub.to_csv(args.sub_pth, index=False)
        print(f"Fallback predictions written to {args.sub_pth}.")
        return

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

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
        "--sub_pth",
        type=str,
        default="submission.csv",
        help="Path to save submission file (default: submission.csv)",
    )
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Max length of input sequence"
    )
    args, _ = parser.parse_known_args()
    main(args)



## === cell 1
import sys, os, argparse

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
        help="Path to save submission file (default: submission.csv)",
    )
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Max length of input sequence"
    )
    args, _ = parser.parse_known_args()
    main(args)
    print("Submission file created:", os.path.abspath(args.sub_pth))
