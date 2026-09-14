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

0.63204

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.58474) has done: 'I remove the imports of `peft` and `transformers` that cause protobuf errors and force `HAVE_MODEL` to `False`. This skips the model‑loading block, letting the lightweight length‑based baseline run and write a proper `submission.csv`. No other logic is changed, preserving the original prediction approach.'
- What this solution (achieved 0.60153) has done: 'I keep the overall structure and the optional language‑model branch unchanged, but replace the simple quantile‑bin length heuristic with a more precise length‑to‑score mapping that still relies on essay length only. By using the exact mean score for each observed word‑count in the training data (and falling back to the nearest available length when unseen), the predictions become finer‑grained while preserving the original length‑based logic, which should raise the quadratic weighted kappa toward the target value.'
- What this solution (achieved 0.62867) has done: 'I replace the simple length‑to‑mean heuristic with a tiny linear regression that uses both word count and character count as features. This keeps the overall “no‑model” branch unchanged in structure but provides a more expressive mapping from essay size to score, which should raise the quadratic weighted kappa toward the target while preserving the original workflow.'
- What this solution (achieved 0.62982) has done: 'I keep the overall workflow unchanged but enhance the “no‑model” branch by adding several simple, non‑linear length‑based features (squared terms, interaction, average word length, logarithms) and refit the linear regression on this richer feature set. This small change stays within the original lightweight approach, preserves the same code structure, and is expected to increase the quadratic weighted kappa toward the target without introducing heavy libraries or altering the core logic.'
- What this solution (achieved 0.61776) has done: 'I add a small ridge‐regularized linear model and blend its prediction with a simple word‑length‑to‑mean‑score lookup; this keeps the original lightweight logic while giving a modest, more stable estimate that should raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.62263) has done: 'I keep the overall workflow unchanged but soften the ridge regularisation and give the regression model a larger influence in the final blend.  Using a smaller `ridge_alpha` lets the linear model fit the training data more closely, and increasing the regression weight from 0.7 to 0.8 should improve the quadratic weighted kappa without altering any core logic or adding new libraries.'
- What this solution (achieved 0.62718) has done: 'I slightly adjust the lightweight “no‑model” branch: use a much weaker ridge regularization (α = 0.001) so the linear regression fits the training data more closely, and replace the length‑to‑score lookup with the median score for each word‑count (more robust than the mean). These tiny changes keep the overall workflow identical while giving the model a modest boost toward the target QWK.'
- What this solution (achieved 0.63204) has done: 'The fix escapes the “?” in the Pandas regex count to avoid the compilation error and slightly reduces the ridge regularization (α = 1e‑7) to let the linear model fit the data a bit better, which should modestly improve the quadratic weighted kappa while keeping the original lightweight logic intact.'
- What this solution (achieved 0.6262) has done: 'I make three small, safe adjustments to the “no‑model” fallback that keep the overall workflow identical:  
1. Set a modest ridge regularisation (`ridge_alpha = 1e‑4`) to improve generalisation.  
2. Replace the length‑median lookup with a length‑mean lookup (often smoother) and keep the overall‑mean fallback.  
3. Blend the regression and length‑based predictions with a slightly larger contribution from the length‑based term (0.85 × regression + 0.15 × length).  

These minimal tweaks are expected to raise the quadratic weighted kappa toward the target while preserving the original code structure and output format.'
- What this solution (achieved 0.63204) has done: 'I will reduce the ridge regularisation (‑ `ridge_alpha = 1e‑7`) so the linear model fits the data more closely, switch the length‑lookup from a mean to a median (more robust to outliers), and give the regression a slightly larger influence in the final blend (‑ `0.9 * pred_reg + 0.1 * pred_len`). These tiny, targeted tweaks keep the overall workflow unchanged while expected to raise the quadratic weighted kappa toward the target score.'

# 9. Code solution

## === cell 0
import gc
import sys
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace

HAVE_MODEL = False

if hasattr(torch.backends.cuda, "enable_flash_sdp"):
    torch.backends.cuda.enable_flash_sdp(False)
if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
    torch.backends.cuda.enable_mem_efficient_sdp(False)




## === cell 1
def main(args):
    config = SimpleNamespace(
        data_dir="/kaggle/input/learning-agency-lab-automated-essay-scoring-2",
    )

    if HAVE_MODEL:
        from transformers import AutoModelForCausalLM, AutoTokenizer
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

        train_df["word_len"] = train_df["full_text"].str.split().str.len().astype(float)
        train_df["char_len"] = train_df["full_text"].str.len().astype(float)

        train_df["word_len_sq"] = train_df["word_len"] ** 2
        train_df["char_len_sq"] = train_df["char_len"] ** 2
        train_df["word_char_prod"] = train_df["word_len"] * train_df["char_len"]
        train_df["avg_word_len"] = train_df["char_len"] / train_df["word_len"].replace(
            0, np.nan
        )
        train_df["log_word_len"] = np.log1p(train_df["word_len"])
        train_df["log_char_len"] = np.log1p(train_df["char_len"])

        train_df["sentence_cnt"] = (
            train_df["full_text"]
            .str.count(r"[.!?]")
            .replace(0, np.nan)  # avoid division by zero later
        )
        train_df["avg_sent_len"] = train_df["word_len"] / train_df["sentence_cnt"]
        train_df["comma_cnt"] = train_df["full_text"].str.count(",")
        train_df["semicolon_cnt"] = train_df["full_text"].str.count(";")
        train_df["exclamation_cnt"] = train_df["full_text"].str.count("!")
        train_df["question_cnt"] = train_df["full_text"].str.count(r"\?")

        feature_cols = [
            "word_len",
            "char_len",
            "word_len_sq",
            "char_len_sq",
            "word_char_prod",
            "avg_word_len",
            "log_word_len",
            "log_char_len",
            "sentence_cnt",
            "avg_sent_len",
            "comma_cnt",
            "semicolon_cnt",
            "exclamation_cnt",
            "question_cnt",
        ]

        X = np.column_stack(
            [np.ones(len(train_df))]
            + [train_df[col].fillna(0).values for col in feature_cols]
        )
        y = train_df["score"].values.astype(float)

        ridge_alpha = 1e-7
        XtX = X.T @ X
        ridge_term = ridge_alpha * np.eye(XtX.shape[0])
        coeffs = np.linalg.solve(XtX + ridge_term, X.T @ y)

        length_median_series = train_df.groupby(train_df["word_len"].astype(int))[
            "score"
        ].median()
        length_median_dict = {int(k): float(v) for k, v in length_median_series.items()}
        overall_mean_score = float(y.mean())

        for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
            essay_text = str(row["full_text"])
            w_len = float(len(essay_text.split()))
            c_len = float(len(essay_text))
            w_len_sq = w_len**2
            c_len_sq = c_len**2
            w_c_prod = w_len * c_len
            avg_w_len = c_len / (w_len if w_len != 0 else 1.0)
            log_w_len = np.log1p(w_len)
            log_c_len = np.log1p(c_len)

            sent_cnt = float(
                essay_text.count(".") + essay_text.count("!") + essay_text.count("?")
            )
            sent_cnt = sent_cnt if sent_cnt != 0 else 1.0
            avg_sent_len = w_len / sent_cnt
            comma_cnt = float(essay_text.count(","))
            semicolon_cnt = float(essay_text.count(";"))
            excl_cnt = float(essay_text.count("!"))
            ques_cnt = float(essay_text.count("?"))

            feats = np.array(
                [
                    1.0,
                    w_len,
                    c_len,
                    w_len_sq,
                    c_len_sq,
                    w_c_prod,
                    avg_w_len,
                    log_w_len,
                    log_c_len,
                    sent_cnt,
                    avg_sent_len,
                    comma_cnt,
                    semicolon_cnt,
                    excl_cnt,
                    ques_cnt,
                ]
            )
            pred_reg = feats @ coeffs

            w_len_int = int(round(w_len))
            pred_len = length_median_dict.get(w_len_int, overall_mean_score)

            pred_blend = 0.9 * pred_reg + 0.1 * pred_len
            pred = int(round(pred_blend))
            pred = max(1, min(6, pred))  # ensure valid range
            test_preds.append(pred)

    sub["score"] = test_preds
    sub["score"] = sub["score"].astype("int")
    sub.to_csv(args.sub_pth, index=False)

    if HAVE_MODEL:
        del model, tokenizer
        torch.cuda.empty_cache()
    gc.collect()




## === cell 2
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
