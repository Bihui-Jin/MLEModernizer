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

0.60074

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.60074) has done: 'I fix the heuristic fallback crash by ensuring the bin-mapped predictions are converted off the Categorical dtype before calling `fillna`, which currently raises the “new category” TypeError. I also make the inference loop robust to environments where `transformers/peft` (or their protobuf stack) fail to import by keeping the try/except fallback intact, so a valid `submission.csv` is always produced. Finally, I keep the core modeling logic unchanged and only adjust the postprocessing/parsing and fallback pipeline so it runs end-to-end and writes the required `essay_id,score` CSV.'
- What this solution (achieved 0.60074) has done: 'I fix the crash that prevents the LLM inference path from running by forcing the pure-Python protobuf implementation before importing `transformers/peft`, which avoids the `MessageFactory.GetPrototype` AttributeError in many Kaggle/Python 3.12 images. To move the score up toward your target while keeping the same overall “dependency-free fallback” concept, I minimally upgrade the heuristic fallback from length-binning to a small TF‑IDF + Ridge regression model (still just scikit‑learn; no deep learning or external downloads), and clip/round to the required 1–6 integer scores. I also make the fallback automatically choose the best available model based on local CV quadratic weighted kappa (QWK) using an internal QWK implementation, so it stays deterministic and metric-aligned without changing the LLM core logic. Finally, I ensure a valid `submission.csv` is always written with correct columns and row count.'
- What this solution (achieved 0.60074) has done: 'I fix the protobuf-related `MessageFactory.GetPrototype` crash by setting the pure-Python protobuf implementation *before* any possible `transformers/peft` import and by running the LLM inference in a clean subprocess with that environment enforced. I also remove a current runtime bug in the notebook driver cell where `heuristic_fallback_submission` is referenced but not defined in that scope, ensuring a valid `submission.csv` is always produced. To move the score toward your target while preserving the existing core approach, I keep the exact TF‑IDF+Ridge fallback but add a minimal QWK-aligned calibration step (fit a single scalar `a,b` mapping on out-of-fold predictions, then round/clip to 1–6), which usually improves QWK without changing the model class. All paths and submission format remain unchanged.'

# 9. Code solution

## === cell 0
import os, sys, re, gc, argparse
import pandas as pd


def heuristic_fallback_submission(data_dir: str, sub_pth: str) -> None:
    """
    Dependency-light fallback that creates a valid submission.

    Minimal score improvement (toward target) while preserving core fallback semantics:
    - Keep TF-IDF + Ridge regression.
    - Add a tiny QWK-aligned calibration: fit y ~= a*pred + b using OOF preds (least squares),
      then apply to test preds before round+clip to 1..6.
    - If sklearn is unavailable, fall back to length-binning heuristic.
    """
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    sample_path = os.path.join(data_dir, "sample_submission.csv")

    train = pd.read_csv(train_path, usecols=["essay_id", "full_text", "score"])
    test = pd.read_csv(test_path, usecols=["essay_id", "full_text"])
    sub = pd.read_csv(sample_path, usecols=["essay_id", "score"])

    train["full_text"] = train["full_text"].fillna("").astype(str)
    test["full_text"] = test["full_text"].fillna("").astype(str)
    y = train["score"].astype(int).to_numpy()

    def qwk(y_true, y_pred, min_rating=1, max_rating=6):
        import numpy as np

        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)

        y_true = np.clip(y_true, min_rating, max_rating)
        y_pred = np.clip(y_pred, min_rating, max_rating)

        n = max_rating - min_rating + 1
        O = np.zeros((n, n), dtype=np.float64)
        for a_, b_ in zip(y_true, y_pred):
            O[a_ - min_rating, b_ - min_rating] += 1.0

        act_hist = np.bincount(y_true - min_rating, minlength=n).astype(np.float64)
        pred_hist = np.bincount(y_pred - min_rating, minlength=n).astype(np.float64)
        E = np.outer(act_hist, pred_hist)
        if E.sum() > 0:
            E *= O.sum() / E.sum()

        W = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                W[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)

        denom = (W * E).sum()
        if denom == 0:
            return 1.0
        return 1.0 - (W * O).sum() / denom

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold
        import numpy as np

        X_text = train["full_text"].values
        X_test_text = test["full_text"].values

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=120000,
            strip_accents="unicode",
            lowercase=True,
            sublinear_tf=True,
        )

        X = vectorizer.fit_transform(X_text)
        X_test = vectorizer.transform(X_test_text)

        alphas = [0.5, 1.0, 2.0, 5.0]
        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        best_alpha = alphas[0]
        best_qwk = -1e9

        best_oof = None

        for a in alphas:
            fold_scores = []
            oof_pred = np.zeros(len(y), dtype=np.float64)
            for tr_idx, va_idx in kf.split(X):
                model = Ridge(alpha=a, random_state=42)
                model.fit(X[tr_idx], y[tr_idx])
                va_pred = model.predict(X[va_idx])
                oof_pred[va_idx] = va_pred

                va_pred_int = np.rint(va_pred).astype(int)
                va_pred_int = np.clip(va_pred_int, 1, 6)
                fold_scores.append(qwk(y[va_idx], va_pred_int))

            mean_score = float(np.mean(fold_scores))
            if mean_score > best_qwk:
                best_qwk = mean_score
                best_alpha = a
                best_oof = oof_pred

        final_model = Ridge(alpha=best_alpha, random_state=42)
        final_model.fit(X, y)

        a_cal, b_cal = 1.0, 0.0
        try:
            if best_oof is not None:
                A = np.vstack([best_oof, np.ones_like(best_oof)]).T
                sol, _, _, _ = np.linalg.lstsq(A, y.astype(np.float64), rcond=None)
                a_cal, b_cal = float(sol[0]), float(sol[1])
                a_cal = max(0.5, min(1.5, a_cal))
                b_cal = max(-1.0, min(1.0, b_cal))
        except Exception:
            a_cal, b_cal = 1.0, 0.0

        te_pred = final_model.predict(X_test)
        te_pred = a_cal * te_pred + b_cal
        te_pred = np.rint(te_pred).astype(int)
        te_pred = np.clip(te_pred, 1, 6)

        pred_df = pd.DataFrame({"essay_id": test["essay_id"].values, "score": te_pred})
        sub = sub[["essay_id"]].merge(pred_df, on="essay_id", how="left")
        sub["score"] = (
            sub["score"]
            .fillna(int(round(float(train["score"].mean()))))
            .astype(int)
            .clip(1, 6)
        )
        sub.to_csv(sub_pth, index=False)
        return

    except Exception as e:
        print(
            f"[WARN] sklearn TF-IDF fallback failed ({type(e).__name__}: {e}); using length-binning fallback."
        )

    tr_len = train["full_text"].astype(str).str.len()
    te_len = test["full_text"].astype(str).str.len()

    qs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    edges = tr_len.quantile(qs).to_numpy()
    edges = sorted(set(int(x) for x in edges))

    if len(edges) < 3:
        global_pred = int(round(float(train["score"].mean())))
        pred = pd.Series([global_pred] * len(test), index=test.index)
    else:
        edges[0] = max(edges[0], 0)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1

        tr_bins = pd.cut(tr_len, bins=edges, include_lowest=True, duplicates="drop")
        bin_mean = train.groupby(tr_bins, observed=False)["score"].mean()

        te_bins = pd.cut(te_len, bins=edges, include_lowest=True, duplicates="drop")
        pred = te_bins.map(bin_mean)

        pred = pd.Series(pred, index=test.index, dtype="float64")
        pred = pred.fillna(float(train["score"].mean()))
        pred = pred.round().astype(int)

    pred = pred.clip(1, 6)
    pred_df = pd.DataFrame({"essay_id": test["essay_id"].values, "score": pred.values})
    sub = sub[["essay_id"]].merge(pred_df, on="essay_id", how="left")
    sub["score"] = sub["score"].fillna(3).astype(int).clip(1, 6)
    sub.to_csv(sub_pth, index=False)


def main(args):
    data_dir = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"

    try:
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

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
import os
import pandas as pd


infer_py = r"""import os, sys, re, gc, argparse
import pandas as pd

# Ensure pure-Python protobuf before importing peft/transformers to avoid MessageFactory.GetPrototype crash
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

def heuristic_fallback_submission(data_dir: str, sub_pth: str) -> None:
    train_path = os.path.join(data_dir, "train.csv")
    test_path  = os.path.join(data_dir, "test.csv")
    sample_path = os.path.join(data_dir, "sample_submission.csv")

    train = pd.read_csv(train_path, usecols=["essay_id","full_text", "score"])
    test  = pd.read_csv(test_path, usecols=["essay_id", "full_text"])
    sub   = pd.read_csv(sample_path, usecols=["essay_id", "score"])

    train["full_text"] = train["full_text"].fillna("").astype(str)
    test["full_text"]  = test["full_text"].fillna("").astype(str)
    y = train["score"].astype(int).to_numpy()

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold
        import numpy as np

        def qwk(y_true, y_pred, min_rating=1, max_rating=6):
            y_true = np.asarray(y_true, dtype=int)
            y_pred = np.asarray(y_pred, dtype=int)
            y_true = np.clip(y_true, min_rating, max_rating)
            y_pred = np.clip(y_pred, min_rating, max_rating)

            n = max_rating - min_rating + 1
            O = np.zeros((n, n), dtype=np.float64)
            for a_, b_ in zip(y_true, y_pred):
                O[a_ - min_rating, b_ - min_rating] += 1.0

            act_hist = np.bincount(y_true - min_rating, minlength=n).astype(np.float64)
            pred_hist = np.bincount(y_pred - min_rating, minlength=n).astype(np.float64)
            E = np.outer(act_hist, pred_hist)
            if E.sum() > 0:
                E *= O.sum() / E.sum()

            W = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    W[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)

            denom = (W * E).sum()
            if denom == 0:
                return 1.0
            return 1.0 - (W * O).sum() / denom

        X_text = train["full_text"].values
        X_test_text = test["full_text"].values

        vectorizer = TfidfVectorizer(
            ngram_range=(1,2),
            min_df=2,
            max_features=120000,
            strip_accents="unicode",
            lowercase=True,
            sublinear_tf=True,
        )
        X = vectorizer.fit_transform(X_text)
        X_test = vectorizer.transform(X_test_text)

        alphas = [0.5, 1.0, 2.0, 5.0]
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        best_alpha = alphas[0]
        best_qwk = -1e9
        best_oof = None

        for a in alphas:
            fold_scores = []
            oof_pred = np.zeros(len(y), dtype=np.float64)
            for tr_idx, va_idx in kf.split(X):
                model = Ridge(alpha=a, random_state=42)
                model.fit(X[tr_idx], y[tr_idx])
                va_pred = model.predict(X[va_idx])
                oof_pred[va_idx] = va_pred

                va_pred_int = np.rint(va_pred).astype(int)
                va_pred_int = np.clip(va_pred_int, 1, 6)
                fold_scores.append(qwk(y[va_idx], va_pred_int))

            mean_score = float(np.mean(fold_scores))
            if mean_score > best_qwk:
                best_qwk = mean_score
                best_alpha = a
                best_oof = oof_pred

        final_model = Ridge(alpha=best_alpha, random_state=42)
        final_model.fit(X, y)

        a_cal, b_cal = 1.0, 0.0
        try:
            if best_oof is not None:
                A = np.vstack([best_oof, np.ones_like(best_oof)]).T
                sol, _, _, _ = np.linalg.lstsq(A, y.astype(np.float64), rcond=None)
                a_cal, b_cal = float(sol[0]), float(sol[1])
                a_cal = max(0.5, min(1.5, a_cal))
                b_cal = max(-1.0, min(1.0, b_cal))
        except Exception:
            a_cal, b_cal = 1.0, 0.0

        te_pred = final_model.predict(X_test)
        te_pred = a_cal * te_pred + b_cal
        te_pred = np.rint(te_pred).astype(int)
        te_pred = np.clip(te_pred, 1, 6)

        pred_df = pd.DataFrame({"essay_id": test["essay_id"].values, "score": te_pred})
        sub = sub[["essay_id"]].merge(pred_df, on="essay_id", how="left")
        sub["score"] = sub["score"].fillna(int(round(float(train["score"].mean())))).astype(int).clip(1, 6)
        sub.to_csv(sub_pth, index=False)
        return
    except Exception as e:
        print(f"[WARN] sklearn TF-IDF fallback failed ({type(e).__name__}: {e}); using length-binning fallback.")

    tr_len = train["full_text"].astype(str).str.len()
    te_len = test["full_text"].astype(str).str.len()

    qs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    edges = tr_len.quantile(qs).to_numpy()
    edges = sorted(set(int(x) for x in edges))

    if len(edges) < 3:
        global_pred = int(round(float(train["score"].mean())))
        pred = pd.Series([global_pred] * len(test), index=test.index)
    else:
        edges[0] = max(edges[0], 0)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i-1]:
                edges[i] = edges[i-1] + 1

        tr_bins = pd.cut(tr_len, bins=edges, include_lowest=True, duplicates="drop")
        bin_mean = train.groupby(tr_bins, observed=False)["score"].mean()

        te_bins = pd.cut(te_len, bins=edges, include_lowest=True, duplicates="drop")
        pred = te_bins.map(bin_mean)

        pred = pd.Series(pred, index=test.index, dtype="float64")
        pred = pred.fillna(float(train["score"].mean()))
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

env = os.environ.copy()
env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    out = subprocess.check_output(
        shlex.split(cmd), stderr=subprocess.STDOUT, env=env
    ).decode("utf-8", errors="ignore")
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
