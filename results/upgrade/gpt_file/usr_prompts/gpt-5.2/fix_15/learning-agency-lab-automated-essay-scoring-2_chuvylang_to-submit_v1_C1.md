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

datasets==4.4.1
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
tokenizers==0.21.2
transformers==4.53.3
vega-datasets==0.9.0

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

0.7993434084034929

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the environment-breaking import error by forcing the pure-Python protobuf implementation before importing `transformers/datasets`, which avoids the `MessageFactory.GetPrototype` crash. Then I make model loading robust by detecting when the provided `/kaggle/input/...` checkpoint path doesn’t exist in this notebook environment and falling back to a public base model with the same `AutoModelForSequenceClassification` core logic, so inference can run end-to-end. I update `TrainingArguments` to the current Transformers API (`eval_strategy` instead of `evaluation_strategy`) to remove the init error. Finally, I prevent NaNs/infs from reaching the submission by handling prediction shapes correctly (regression vs logits) and sanitizing non-finite values before casting to int, ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf crash by setting the pure-Python protobuf implementation *before* importing `transformers/datasets`, which prevents the `MessageFactory.GetPrototype` error in this environment. Then I fix the `Trainer` initialization error by disabling evaluation during pure inference (`eval_strategy="no"`), since no `eval_dataset` is provided and we’re not training here. Finally, I keep your core inference logic intact but make the “fold loop” load per-fold checkpoints when available (or just run once otherwise) so predictions aren’t needlessly duplicated, and ensure the submission is written as a valid `submission.csv` with aligned `essay_id` ordering.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf/transformers import crash by forcing the pure-Python protobuf implementation early (and restarting the import order), which removes the `MessageFactory.GetPrototype` failure. Then I correct a likely score-killing logic issue: the regression post-processing currently maps predictions to 1–6 using `+1` but clips to 0–5 first; I instead clip directly to 1–6 and round, which is consistent with the target label space. I also make the model checkpoint selection robust to missing `/kaggle/input/vuxvuxregression/...` by falling back to a known public model so inference always runs end-to-end. Finally, I ensure predictions align to `essay_id` order and always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the immediate crash by forcing the pure-Python protobuf implementation *before any* `transformers/datasets` import and by ensuring `google.protobuf` is imported only after the env var is set (this is the root cause of the `MessageFactory.GetPrototype` error). I also fix a second likely runtime issue in this Kaggle environment by disabling `fp16` automatically when CUDA isn’t available, preventing Trainer from failing on CPU-only runs. Finally, I keep your inference logic intact but make the submission alignment robust by building the submission from `test.csv` `essay_id` order (instead of assuming `sample_submission.csv` always matches), ensuring a valid `submission.csv` is always produced.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf import crash by forcing the pure-Python protobuf implementation and preventing `google.protobuf` from being imported before `transformers/datasets`. Then I make the checkpoint discovery logic robust when `MODEL_PATH` is a single checkpoint directory (so it doesn’t mistakenly look for nested `checkpoint-*` folders and skip). Finally, I keep your inference/post-processing intact but ensure the pipeline always runs end-to-end and writes a valid `submission.csv` with aligned `essay_id` ordering.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf crash by forcing the pure-Python protobuf implementation before any `transformers/datasets` import and by importing `google.protobuf` only after that env var is set, which resolves the `MessageFactory.GetPrototype` error in this environment. Then I make model loading robust if the provided Kaggle checkpoint path doesn’t exist by falling back to a public DeBERTa model so inference can complete and a submission is always produced. I also correct a tokenization bug (`batched=True` but treating inputs as scalars) that would otherwise break or tokenize incorrectly, while preserving the same overall inference approach. Finally, I keep the existing prediction post-processing but ensure the output is aligned to `test.csv` order and written to `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf-related `MessageFactory.GetPrototype` crash by forcing a compatible protobuf runtime configuration *before* importing `transformers/datasets`, and by ensuring we don’t import `google.protobuf` too early (which defeats the env var). Then I make the import path more robust by preferring the bundled Kaggle dataset checkpoint if it exists, otherwise falling back to a public base model so the notebook always runs end-to-end and produces a valid `submission.csv`. Finally, I keep your inference/post-processing logic intact but ensure tokenization and prediction shapes are handled safely so the submission is always numeric, finite, clipped to 1–6, and aligned to `test.csv` order.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf/transformers import crash that prevents the notebook from running by forcing the pure-Python protobuf backend *and* disabling the C++ protobuf implementation before any `transformers/datasets` import. Then I keep your existing inference-only approach intact but make it robust to environments without PyTorch by falling back to a lightweight baseline (mean train score) so a valid `submission.csv` is always produced instead of scoring 0.0 due to runtime failure. Finally, I keep your post-processing (clip/round to 1–6) but ensure it never breaks on shape mismatches and always aligns predictions to `test.csv` `essay_id` order.'
- What this solution (achieved 0.0) has done: 'I fix the root cause of the `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation early and also uninstalling the incompatible `google-protobuf` shim module from `sys.modules` before importing `transformers/datasets`. Then I keep your inference-only pipeline intact, but make it robust to missing checkpoints by reliably falling back to a public base model so it always produces a valid `submission.csv`. Finally, I ensure tokenization works correctly in batched mode and keep your existing 1–6 clipping/rounding so the output is valid and score improves above the current 0.0 (which is caused by the import-time crash).'
- What this solution (achieved 0.0) has done: 'I fix the protobuf/Transformers import crash that currently prevents any model inference, by forcing the pure-Python protobuf backend early and also preventing the incompatible `google.protobuf` C++ runtime from being used. Then I make the import guard actually work (right now the crash happens during the `try` import), so the script cleanly falls back only if imports truly fail. Finally, I keep your inference and post-processing logic the same, ensuring predictions are produced and a valid `submission.csv` is always written with `essay_id,score` and scores clipped to 1–6.'
- What this solution (achieved 0.0) has done: 'The immediate blocker is the protobuf/transformers crash (`MessageFactory.GetPrototype`) happening during import; I fix that by preventing the incompatible `google.protobuf` module from being imported and forcing the pure-Python protobuf backend before any transformers/datasets import. Then I make the transformers import truly optional: if it still fails, we cleanly fall back to the mean-score baseline (so we always generate a valid `submission.csv` instead of scoring 0.0 due to a crash). I not change your model/inference logic beyond the import guard; the tokenization, checkpoint discovery, prediction averaging, and 1–6 clipping/rounding remain the same. This should move the score up from 0.0 by allowing the pipeline to run end-to-end (and likely enabling transformer inference when available).'
- What this solution (achieved 0.0) has done: 'I fix the import-time protobuf crash that prevents `transformers/datasets` from loading (which is why you end up with a 0.0 score) by setting safer protobuf env vars early and removing the risky `sys.modules` deletion that can leave protobuf half-initialized. Then I make the inference path use `eval_dataset` (not `train_dataset`) to match the current `Trainer.predict` expectations without changing the overall inference approach. Finally, I keep your existing clipping/rounding to 1–6 and ensure `submission.csv` is always written from `test.csv` order.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf/Transformers import crash that currently prevents inference by forcing the pure-Python protobuf runtime early and explicitly downgrading the protobuf package to a compatible version at runtime (no internet needed) before importing `transformers/datasets`. This should allow the existing Transformer inference path to run, which raise the score from the current fallback-baseline behavior (0.0) toward your target. I also make the import guard robust so that if the environment still can’t load Transformers, the code cleanly falls back and still writes a valid `submission.csv`. Core model/inference logic, post-processing (clip/round to 1–6), and file paths remain unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from producing a “valid-looking” submission that is effectively uninformative (fallback mean-score baseline because transformer inference fails or uses an untrained base model). To move the score upward toward ~0.80 with minimal logic changes, I (1) ensure we actually load a compatible, task-finetuned checkpoint from the provided Kaggle dataset if it exists, and (2) keep your exact inference flow but fix the regression/classification mismatch by auto-detecting whether the loaded checkpoint is regression (1 output) or classification (6 outputs) and post-process accordingly. I also add a small, safe improvement for QWK: learn optimal rounding thresholds on a held-out split using the same model predictions (no architecture/training changes), then apply those thresholds to test predictions. This keeps your core approach (Transformer + Trainer.predict + averaging) intact while materially improving QWK versus naive rounding.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

LOAD_FROM = (
    "/kaggle/input/vuxvuxregression/transformers/fuckyou/1/output_v1/checkpoint-27692"
)



## === cell 1
import sys
import warnings
import subprocess
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")


def _ensure_compatible_protobuf():
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
    except Exception as e:
        print("WARNING: Could not pin protobuf version. Continuing. Error:", repr(e))


_ensure_compatible_protobuf()

TRANSFORMERS_OK = True
TRANSFORMERS_IMPORT_ERROR = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import TrainingArguments, Trainer
    from transformers import DataCollatorWithPadding
    from transformers import set_seed
    from datasets import Dataset
except Exception as e:
    TRANSFORMERS_OK = False
    TRANSFORMERS_IMPORT_ERROR = repr(e)
    print(
        "WARNING: transformers/datasets import failed; will use fallback baseline.\n",
        TRANSFORMERS_IMPORT_ERROR,
    )




## === cell 2
class PATHS:
    train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
    test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
    sub_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
    model_path = "/kaggle/input/vuxvuxregression/transformers/fuckyou/1/output_v1/checkpoint-27692"


USE_REGRESSION = True


def pick_model_path(
    preferred_path: str, fallback_repo: str = "microsoft/deberta-v3-small"
) -> str:
    if preferred_path and os.path.isdir(preferred_path):
        return preferred_path
    if LOAD_FROM and os.path.isdir(LOAD_FROM):
        return LOAD_FROM
    return fallback_repo


MODEL_PATH = pick_model_path(PATHS.model_path)
print("Using MODEL_PATH:", MODEL_PATH)



## === cell 3
if TRANSFORMERS_OK:

    class Tokenize(object):
        def __init__(self, train, valid, tokenizer):
            self.tokenizer = tokenizer
            self.train = train
            self.valid = valid

        def get_dataset(self, df):
            ds = Dataset.from_dict(
                {
                    "essay_id": [e for e in df["essay_id"]],
                    "full_text": [ft for ft in df["full_text"]],
                    "label": [s for s in df["label"]],
                }
            )
            return ds

        def tokenize_function(self, examples):
            tokenized_inputs = self.tokenizer(
                examples["full_text"], truncation=True, max_length=CFG.max_length
            )
            return tokenized_inputs

        def __call__(self):
            train_ds = self.get_dataset(self.train)
            valid_ds = self.get_dataset(self.valid)

            tokenized_train = train_ds.map(self.tokenize_function, batched=True)
            tokenized_valid = valid_ds.map(self.tokenize_function, batched=True)

            return tokenized_train, tokenized_valid, self.tokenizer




## === cell 4
class CFG:
    n_splits = 5
    seed = 42
    max_length = 4096
    lr = 1e-5
    train_batch_size = 2
    eval_batch_size = 2
    train_epochs = 30
    weight_decay = 0.01
    warmup_ratio = 0.0
    num_labels = 6




## === cell 5
if TRANSFORMERS_OK:
    try:
        import torch

        _use_fp16 = bool(torch.cuda.is_available())
    except Exception:
        _use_fp16 = False

    set_seed(CFG.seed)

    training_args = TrainingArguments(
        output_dir="output_v",
        fp16=_use_fp16,
        learning_rate=CFG.lr,
        per_device_train_batch_size=CFG.train_batch_size,
        per_device_eval_batch_size=CFG.eval_batch_size,
        num_train_epochs=CFG.train_epochs,
        weight_decay=CFG.weight_decay,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        warmup_ratio=CFG.warmup_ratio,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        logging_first_step=True,
    )



## === cell 6
test = pd.read_csv(PATHS.test_path)
print("Test shape:", test.shape)
test.head()



## === cell 7
test = test.copy()
test["label"] = 0.0




## === cell 8
def discover_checkpoints(base_path: str, n_splits: int):
    """
    MODEL_PATH may itself already be a checkpoint-* directory.
    In that case, return [MODEL_PATH] rather than scanning for nested checkpoints.
    """
    if not (base_path and os.path.isdir(base_path)):
        return []

    base_name = os.path.basename(os.path.normpath(base_path))
    if base_name.startswith("checkpoint-"):
        return [base_path]

    ckpts = []
    for name in os.listdir(base_path):
        p = os.path.join(base_path, name)
        if os.path.isdir(p) and name.startswith("checkpoint-"):
            ckpts.append(p)
    ckpts = sorted(ckpts)
    if len(ckpts) > n_splits:
        ckpts = ckpts[:n_splits]
    return ckpts


def _infer_head_type_from_model(model):
    try:
        out_dim = int(getattr(model.config, "num_labels", 0))
    except Exception:
        out_dim = 0
    is_reg = out_dim == 1
    is_cls = out_dim == 6
    return is_reg, is_cls, out_dim


def _pred_to_continuous(preds_arr, is_regression):
    preds_arr = np.asarray(preds_arr)
    if is_regression:
        if preds_arr.ndim == 2 and preds_arr.shape[1] == 1:
            return preds_arr[:, 0].astype(np.float32)
        return preds_arr.astype(np.float32)
    if preds_arr.ndim == 1:
        return preds_arr.astype(np.float32) + 1.0
    logits = preds_arr.astype(np.float32)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    classes = np.arange(1, 7, dtype=np.float32)[None, :]
    return (probs * classes).sum(axis=1)


def fit_rounding_thresholds(y_true_int, y_pred_cont, n_classes=6, n_iter=8):
    """
    Why: QWK is sensitive to rounding; optimizing thresholds on a held-out split
    usually increases score a lot vs naive round, without changing the model.
    Minimal coordinate-descent; fast on CPU.
    """
    from sklearn.metrics import cohen_kappa_score

    y_true_int = np.asarray(y_true_int, dtype=int)
    y_pred_cont = np.asarray(y_pred_cont, dtype=np.float32)

    thr = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)

    def apply_thr(x, t):
        bins = np.digitize(x, t, right=False) + 1
        return np.clip(bins, 1, n_classes)

    best = cohen_kappa_score(
        y_true_int, apply_thr(y_pred_cont, thr), weights="quadratic"
    )

    for _ in range(n_iter):
        improved = False
        for i in range(len(thr)):
            lo = 1.01 if i == 0 else float(thr[i - 1] + 1e-3)
            hi = 5.99 if i == len(thr) - 1 else float(thr[i + 1] - 1e-3)
            if lo >= hi:
                continue
            candidates = np.linspace(lo, hi, 41, dtype=np.float32)
            local_best = best
            local_t = thr[i]
            for c in candidates:
                t2 = thr.copy()
                t2[i] = c
                score = cohen_kappa_score(
                    y_true_int, apply_thr(y_pred_cont, t2), weights="quadratic"
                )
                if score > local_best:
                    local_best = score
                    local_t = c
            if local_best > best + 1e-10:
                thr[i] = local_t
                best = local_best
                improved = True
        if not improved:
            break

    return thr, best


def apply_rounding_thresholds(y_pred_cont, thr, n_classes=6):
    y_pred_cont = np.asarray(y_pred_cont, dtype=np.float32)
    bins = np.digitize(y_pred_cont, np.asarray(thr, dtype=np.float32), right=False) + 1
    return np.clip(bins, 1, n_classes).astype(np.int32)




## === cell 9
preds = None
preds_cont = (
    None  # continuous predictions before final discretization (for thresholding)
)
is_regression_head = True  # will be detected from checkpoint if possible

if TRANSFORMERS_OK:
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            ignore_mismatched_sizes=False,
        )
        is_reg, is_cls, out_dim = _infer_head_type_from_model(base_model)
        if is_reg or is_cls:
            is_regression_head = is_reg
        else:
            is_regression_head = bool(USE_REGRESSION)

        if is_regression_head:
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH,
                num_labels=1,
                problem_type="regression",
                ignore_mismatched_sizes=True,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH,
                num_labels=CFG.num_labels,
                problem_type="single_label_classification",
                ignore_mismatched_sizes=True,
            )

        all_pred = []

        tokenize = Tokenize(test, test, tokenizer)
        tokenized_test, _, _ = tokenize()

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        checkpoint_paths = discover_checkpoints(MODEL_PATH, CFG.n_splits)

        if len(checkpoint_paths) == 0:
            trainer = Trainer(
                model=model,
                args=training_args,
                eval_dataset=tokenized_test,
                data_collator=data_collator,
                tokenizer=tokenizer,
            )
            predictions = trainer.predict(tokenized_test).predictions
            all_pred.append(predictions)
        else:
            for ckpt in checkpoint_paths:
                if is_regression_head:
                    fold_model = AutoModelForSequenceClassification.from_pretrained(
                        ckpt,
                        num_labels=1,
                        problem_type="regression",
                        ignore_mismatched_sizes=True,
                    )
                else:
                    fold_model = AutoModelForSequenceClassification.from_pretrained(
                        ckpt,
                        num_labels=CFG.num_labels,
                        problem_type="single_label_classification",
                        ignore_mismatched_sizes=True,
                    )
                trainer = Trainer(
                    model=fold_model,
                    args=training_args,
                    eval_dataset=tokenized_test,
                    data_collator=data_collator,
                    tokenizer=tokenizer,
                )
                predictions = trainer.predict(tokenized_test).predictions
                all_pred.append(predictions)

        preds = np.mean(all_pred, axis=0)
        print("Raw predictions shape:", np.asarray(preds).shape)
        preds_cont = _pred_to_continuous(preds, is_regression_head)
        print(
            "Using head type:", "regression" if is_regression_head else "classification"
        )

    except Exception as e:
        print(
            "WARNING: transformer inference failed; will use fallback baseline.\n",
            repr(e),
        )
        preds = None
        preds_cont = None

if preds_cont is None:
    train_scores = pd.read_csv(PATHS.train_path, usecols=["score"])["score"].astype(
        float
    )
    mean_score = float(train_scores.mean())
    preds_cont = np.full(shape=(len(test),), fill_value=mean_score, dtype=np.float32)
    is_regression_head = True
    print("Fallback baseline used. mean_score:", mean_score)



## === cell 10
thr = None
if TRANSFORMERS_OK and preds is not None:
    try:
        from sklearn.model_selection import train_test_split

        train_df = pd.read_csv(PATHS.train_path, usecols=["full_text", "score"])
        train_df = train_df.dropna().reset_index(drop=True)
        train_df["label"] = train_df["score"].astype(int)
        trn, val = train_test_split(
            train_df,
            test_size=0.08,
            random_state=CFG.seed,
            stratify=train_df["label"],
        )

        tokenizer2 = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
        tok = Tokenize(
            trn[["full_text"]].assign(
                essay_id=np.arange(len(trn)), label=trn["label"].astype(float)
            ),
            val[["full_text"]].assign(
                essay_id=np.arange(len(val)), label=val["label"].astype(float)
            ),
            tokenizer2,
        )
        tokenized_trn, tokenized_val, _ = tok()

        if is_regression_head:
            calib_model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH,
                num_labels=1,
                problem_type="regression",
                ignore_mismatched_sizes=True,
            )
        else:
            calib_model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH,
                num_labels=CFG.num_labels,
                problem_type="single_label_classification",
                ignore_mismatched_sizes=True,
            )

        trainer2 = Trainer(
            model=calib_model,
            args=training_args,
            eval_dataset=tokenized_val,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer2),
            tokenizer=tokenizer2,
        )
        val_raw = trainer2.predict(tokenized_val).predictions
        val_cont = _pred_to_continuous(val_raw, is_regression_head)

        thr, best_kappa = fit_rounding_thresholds(
            val["label"].values, val_cont, n_classes=6, n_iter=8
        )
        print("Fitted thresholds:", thr, "holdout_QWK:", float(best_kappa))
    except Exception as e:
        print(
            "WARNING: threshold calibration failed; using naive rounding. Error:",
            repr(e),
        )
        thr = None



## === cell 11
sub = pd.DataFrame({"essay_id": test["essay_id"].values})

preds_cont = np.asarray(preds_cont, dtype=np.float32)
preds_cont = np.nan_to_num(preds_cont, nan=3.0, posinf=6.0, neginf=1.0)

if thr is None:
    sub["score"] = np.clip(np.rint(preds_cont), 1, 6).astype(np.int32)
else:
    sub["score"] = apply_rounding_thresholds(preds_cont, thr, n_classes=6)

sub = sub[["essay_id", "score"]].copy()
sub["score"] = sub["score"].clip(1, 6).astype(np.int32)

assert len(sub) == len(test), "Submission row count mismatch"

sub.to_csv("submission.csv", index=False)
print("Submission shape:", sub.shape)
sub.head()
