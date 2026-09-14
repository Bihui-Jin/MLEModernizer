# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
tokenizers==0.21.2
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
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

# 5. Code solution

## === cell 0
import os
import sys
import random
import warnings
from pathlib import Path

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.simplefilter("ignore")

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from tokenizers import AddedToken
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

IS_KAGGLE = True
IS_DEV = False

PERFORM_TRAINING = True

if IS_KAGGLE:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    p1 = Path("/kaggle/input/learning-agency-lab-automated-essay-scoring-2")
    p2 = Path(
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/learning-agency-lab-automated-essay-scoring-2"
    )
    base_data_dir = p1 if (p1 / "train.csv").exists() else p2

    class PATHS:
        train_path = base_data_dir / "train.csv"
        test_path = base_data_dir / "test.csv"
        sub_path = base_data_dir / "sample_submission.csv"
        model_path = "microsoft/deberta-v3-xsmall"
        output_dir = Path("model_1")

else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    base_data_dir = Path("../data/competition_data")

    class PATHS:
        train_path = base_data_dir / "train.csv"
        test_path = base_data_dir / "test.csv"
        sub_path = base_data_dir / "sample_submission.csv"
        model_path = "microsoft/deberta-v3-xsmall"
        output_dir = Path("for_stacking/model_1")


BASE_MODEL_NAME = PATHS.model_path.split("/")[-1]
USE_REGRESSION = True


def _is_hf_model_dir(p: Path) -> bool:
    """Minimal check that a directory is a HF checkpoint dir (improves checkpoint discovery correctness)."""
    if not p.exists() or not p.is_dir():
        return False
    has_config = (p / "config.json").exists()
    has_weights = (p / "pytorch_model.bin").exists() or (
        p / "model.safetensors"
    ).exists()
    return bool(has_config and has_weights)


def _find_fold_checkpoint_root(n_splits: int) -> Path | None:
    if not IS_KAGGLE:
        return None
    base = Path("/kaggle/input")
    if not base.exists():
        return None

    candidates: list[Path] = []

    for fold0 in base.rglob("fold_0"):
        parent = fold0.parent
        ok = True
        for k in range(n_splits):
            fk = parent / f"fold_{k}"
            if not fk.exists():
                ok = False
                break
            if not _is_hf_model_dir(fk):
                ok = False
                break
        if ok:
            candidates.append(parent)

    if not candidates:
        return None

    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except Exception:
            return 0.0

    candidates = sorted(candidates, key=lambda p: (-_mtime(p), len(str(p)), str(p)))
    return candidates[0]


if PERFORM_TRAINING:
    COMPUTE_CV = True
    LOAD_FROM = None
else:
    COMPUTE_CV = True
    candidate = (
        Path("/kaggle/input/model-collection-1/model_1")
        if IS_KAGGLE
        else PATHS.output_dir
    )
    LOAD_FROM = candidate if candidate.exists() else None
    if LOAD_FROM is None and IS_KAGGLE:
        LOAD_FROM = _find_fold_checkpoint_root(n_splits=5)


class CFG:
    n_splits = 5
    seed = 42
    max_length = 1024
    lr = 1e-5
    train_batch_size = 4
    eval_batch_size = 8
    train_epochs = 4
    weight_decay = 0.01
    warmup_ratio = 0.0
    num_labels = 6

    tokenization_num_proc = max(1, min(8, (os.cpu_count() or 2) // 2))


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(CFG.seed)


def compute_metrics_for_regression(eval_pred):
    predictions, labels = eval_pred
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    qwk = cohen_kappa_score(
        labels, predictions.clip(0, 5).round(0), weights="quadratic"
    )
    return {"qwk": qwk}


def compute_metrics_for_classification(eval_pred):
    predictions, labels = eval_pred
    qwk = cohen_kappa_score(labels, predictions.argmax(-1), weights="quadratic")
    return {"qwk": qwk}


def build_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.add_tokens([AddedToken("\n", normalized=False)])
    tok.add_tokens([AddedToken(" " * 2, normalized=False)])
    return tok


def make_hf_dataset_from_df(df: pd.DataFrame) -> Dataset:
    cols = ["essay_id", "full_text", "label"]
    return Dataset.from_pandas(
        df.loc[:, cols].reset_index(drop=True), preserve_index=False
    )


def tokenize_dataset(ds: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    def _tok(batch):
        return tokenizer(batch["full_text"], truncation=True, max_length=CFG.max_length)

    remove_cols = [c for c in ds.column_names if c == "full_text"]

    return ds.map(
        _tok,
        batched=True,
        remove_columns=remove_cols,
        load_from_cache_file=True,
        num_proc=CFG.tokenization_num_proc,
        desc="Tokenizing",
    )




## === cell 1
if IS_DEV:
    nrows = 2000
else:
    nrows = None

data = pd.read_csv(PATHS.train_path, nrows=nrows)

data["label"] = data["score"].to_numpy() - 1

if USE_REGRESSION:
    data["label"] = data["label"].astype("float32")
else:
    data["label"] = data["label"].astype("int32")

skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
data["fold"] = -1
for i, (_, val_index) in enumerate(skf.split(data, data["score"])):
    data.loc[val_index, "fold"] = i
data["fold"] = data["fold"].astype(int)

training_args = TrainingArguments(
    output_dir=f"tmp/{BASE_MODEL_NAME}",
    fp16=True,
    learning_rate=CFG.lr,
    per_device_train_batch_size=CFG.train_batch_size,
    per_device_eval_batch_size=CFG.eval_batch_size,
    num_train_epochs=CFG.train_epochs,
    weight_decay=CFG.weight_decay,
    eval_strategy="epoch",
    metric_for_best_model="qwk",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to="none",
    warmup_ratio=CFG.warmup_ratio,
    lr_scheduler_type="linear",
    optim="adamw_torch",
    logging_first_step=True,
    dataloader_num_workers=max(1, min(4, (os.cpu_count() or 2) // 2)),
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    group_by_length=True,
    length_column_name="length",
)

inference_args = TrainingArguments(
    output_dir=f"tmp/{BASE_MODEL_NAME}_infer",
    fp16=True,
    per_device_eval_batch_size=CFG.eval_batch_size,
    eval_strategy="no",
    report_to="none",
    dataloader_drop_last=False,
    dataloader_num_workers=max(1, min(4, (os.cpu_count() or 2) // 2)),
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
)

shared_tokenizer = build_tokenizer(PATHS.model_path)

full_train_ds = make_hf_dataset_from_df(data)

tokenized_full_train_ds = tokenize_dataset(full_train_ds, shared_tokenizer)


def _add_length(ex):
    ex["length"] = len(ex["input_ids"])
    return ex


tokenized_full_train_ds = tokenized_full_train_ds.map(
    _add_length, desc="Adding length column"
)

fold_arr = data["fold"].to_numpy(np.int32)
fold_indices = []
for fold in range(CFG.n_splits):
    tr_idx = np.where(fold_arr != fold)[0]
    va_idx = np.where(fold_arr == fold)[0]
    fold_indices.append((tr_idx, va_idx))




## === cell 2
PATHS.output_dir.mkdir(parents=True, exist_ok=True)

oof_frames = []

if PERFORM_TRAINING:
    for fold in range(CFG.n_splits):
        out_fold_dir = PATHS.output_dir / f"fold_{fold}"
        if _is_hf_model_dir(out_fold_dir):
            print(
                f"Found existing fold checkpoint at {out_fold_dir}, skipping training for this fold."
            )
            continue

        tr_idx, va_idx = fold_indices[fold]

        tokenized_train = tokenized_full_train_ds.select(tr_idx.tolist())
        tokenized_valid = tokenized_full_train_ds.select(va_idx.tolist())

        config = AutoConfig.from_pretrained(PATHS.model_path)
        if USE_REGRESSION:
            config.attention_probs_dropout_prob = 0.0
            config.hidden_dropout_prob = 0.0
            config.num_labels = 1
        else:
            config.num_labels = CFG.num_labels

        model = AutoModelForSequenceClassification.from_pretrained(
            PATHS.model_path, config=config
        )
        model.resize_token_embeddings(len(shared_tokenizer))

        data_collator = DataCollatorWithPadding(tokenizer=shared_tokenizer)
        compute_metrics = (
            compute_metrics_for_regression
            if USE_REGRESSION
            else compute_metrics_for_classification
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid,
            data_collator=data_collator,
            tokenizer=shared_tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        out_fold_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(out_fold_dir))
        shared_tokenizer.save_pretrained(str(out_fold_dir))

        valid_df = data.iloc[va_idx].copy()
        predictions0 = trainer.predict(tokenized_valid).predictions
        if USE_REGRESSION:
            valid_df["pred"] = np.asarray(predictions0).reshape(-1) + 1
        else:
            COLS = [f"p{x}" for x in range(CFG.num_labels)]
            valid_df[COLS] = predictions0

        oof_frames.append(valid_df)

    LOAD_FROM = PATHS.output_dir

if LOAD_FROM is not None:
    missing = []
    for fold in range(CFG.n_splits):
        if not (LOAD_FROM / f"fold_{fold}").exists():
            missing.append(fold)
    if len(missing) > 0:
        print(
            f"LOAD_FROM set but missing fold dirs: {missing}. Falling back to base model."
        )
        LOAD_FROM = None

if LOAD_FROM is not None:
    if not PERFORM_TRAINING:
        data_collator = DataCollatorWithPadding(tokenizer=shared_tokenizer)
        compute_metrics = (
            compute_metrics_for_regression
            if USE_REGRESSION
            else compute_metrics_for_classification
        )

        for fold in range(CFG.n_splits):
            _, va_idx = fold_indices[fold]
            tokenized_valid = tokenized_full_train_ds.select(va_idx.tolist())

            model = AutoModelForSequenceClassification.from_pretrained(
                LOAD_FROM / f"fold_{fold}"
            )
            model.resize_token_embeddings(len(shared_tokenizer))

            trainer = Trainer(
                model=model,
                args=inference_args,
                eval_dataset=tokenized_valid,
                data_collator=data_collator,
                tokenizer=shared_tokenizer,
                compute_metrics=compute_metrics,
            )

            predictions0 = trainer.predict(tokenized_valid).predictions
            valid_df = data.iloc[va_idx].copy()
            if USE_REGRESSION:
                valid_df["pred"] = np.asarray(predictions0).reshape(-1) + 1
            else:
                COLS = [f"p{x}" for x in range(CFG.num_labels)]
                valid_df[COLS] = predictions0

            oof_frames.append(valid_df)

    if COMPUTE_CV:
        if len(oof_frames) > 0:
            dfs = pd.concat(oof_frames, ignore_index=True)
            columns = ["essay_id", "score", "label", "fold", "pred"]
            if all(c in dfs.columns for c in columns):
                dfs[columns].to_csv("valid_df_m1.csv", index=False)

            print("Valid OOF shape:", dfs.shape)
            if USE_REGRESSION:
                for k in range(CFG.n_splits):
                    m = cohen_kappa_score(
                        dfs[dfs.fold == k].score.values,
                        dfs[dfs.fold == k].pred.values.clip(1, 6).round(0),
                        weights="quadratic",
                    )
                    print(f"Fold {k} QWK =", round(m, 5))
                m = cohen_kappa_score(
                    dfs.score.values,
                    dfs.pred.values.clip(1, 6).round(0),
                    weights="quadratic",
                )
            else:
                m = cohen_kappa_score(
                    dfs.score.values,
                    dfs.iloc[:, -6:].values.argmax(axis=1) + 1,
                    weights="quadratic",
                )
            print("Overall QWK CV =", round(m, 5))
else:
    print(
        "Pretrained fold checkpoints not found. Skipping CV/OOF evaluation and running direct inference from base model."
    )




## === cell 3
test = pd.read_csv(PATHS.test_path)
print("Test shape:", test.shape)

all_pred = []
test["label"] = 0.0

test_ds = make_hf_dataset_from_df(test)
tokenized_test = tokenize_dataset(test_ds, shared_tokenizer)
tokenized_test = tokenized_test.map(_add_length, desc="Adding test length")

data_collator = DataCollatorWithPadding(tokenizer=shared_tokenizer)

for fold in range(CFG.n_splits):
    if LOAD_FROM is None:
        config = AutoConfig.from_pretrained(PATHS.model_path)
        if USE_REGRESSION:
            config.num_labels = 1
        else:
            config.num_labels = CFG.num_labels

        model = AutoModelForSequenceClassification.from_pretrained(
            PATHS.model_path, config=config
        )
        model.resize_token_embeddings(len(shared_tokenizer))

        trainer = Trainer(
            model=model,
            args=inference_args,
            eval_dataset=tokenized_test,
            data_collator=data_collator,
            tokenizer=shared_tokenizer,
        )
        predictions = trainer.predict(tokenized_test).predictions
        all_pred.append(predictions)
        break

    model = AutoModelForSequenceClassification.from_pretrained(
        LOAD_FROM / f"fold_{fold}"
    )
    model.resize_token_embeddings(len(shared_tokenizer))

    trainer = Trainer(
        model=model,
        args=inference_args,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
        tokenizer=shared_tokenizer,
    )

    predictions = trainer.predict(tokenized_test).predictions
    all_pred.append(predictions)

preds = (
    np.mean(all_pred, axis=0)
    if len(all_pred) > 0
    else np.zeros((len(test), 1), dtype=np.float32)
)
print("Predictions shape:", np.asarray(preds).shape)




## === cell 4
sub = pd.read_csv(PATHS.sub_path)

if USE_REGRESSION:
    preds_1d = np.asarray(preds).reshape(-1)
    pred_scores = preds_1d.clip(0, 5).round(0) + 1
else:
    pred_scores = np.asarray(preds).argmax(axis=1) + 1

pred_df = pd.DataFrame({"essay_id": test["essay_id"].values, "score": pred_scores})
sub = sub[["essay_id"]].merge(pred_df, on="essay_id", how="left")

if sub["score"].isna().any():
    sub["score"] = sub["score"].fillna(3)

sub["score"] = sub["score"].astype("int32")

sub.to_csv("submission.csv", index=False)
print("Saved submission.csv")
print("Submission shape:", sub.shape)
print(sub.head())
