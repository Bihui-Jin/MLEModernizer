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
scipy==1.15.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
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

# 5. Target score

0.8030895862887903

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the environment/import crash in the first cell by forcing the pure-Python protobuf implementation (this avoids the `MessageFactory.GetPrototype` error that happens with some package combinations). Then I fix model/tokenizer loading so a local Kaggle dataset path like `/kaggle/input/...` is treated as a filesystem directory (not a Hub repo id), which resolves the `HFValidationError`. Finally, I make the inference-to-submission step correct for a sequence-classification model by converting logits to class predictions (1–6) rather than clipping raw logits; this preserves the intended evaluation semantics and produces a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I (1) fix the protobuf-related crash by setting the env var before any protobuf/transformers-related imports and forcing a safe protobuf runtime choice, (2) make the model path robust by automatically resolving your intended local model directory under `/kaggle/input` (and fall back to a simple baseline if it truly doesn’t exist, so a valid submission is always produced), and (3) keep inference semantics correct by converting sequence-classification logits to class labels 1–6. These changes are minimal and directly address the current failures that prevent any valid submission (hence score 0.0). The fallback baseline is only used if the model files are unavailable in the runtime; otherwise your original model inference is used unchanged.'
- What this solution (achieved 0.0) has done: 'I fix the runtime crash happening before any training/inference by forcing a compatible protobuf runtime early and, if needed, downgrading to a safe protobuf package version at notebook runtime (this is what triggers the `MessageFactory.GetPrototype` error). I also make sure the transformers/datasets imports only happen after that fix so the environment doesn’t crash on import. The rest of your pipeline (local model directory resolution, HF local loading, logits→class conversion, and submission writing/merging) be kept the same to preserve evaluation semantics and move the score up from 0.0 toward the target by enabling a valid model-based submission instead of failing at import. If the model directory truly isn’t present, the existing fallback baseline remains to guarantee a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission-format/alignment failure rather than a modeling failure, so I make the smallest changes that ensure we always write a valid `submission.csv` matching the competition’s `test.csv` row count and `essay_id` ordering. I remove the merge against `sample_submission` (its row count differs from `test.csv` in your environment) and instead validate uniqueness and align predictions directly to `df_test`. I also make the model-loading path more robust by catching any local loading errors and falling back cleanly, so you don’t silently produce an invalid file. These changes preserve your core inference logic (sequence-classification logits → argmax → 1–6) and should move the score up from 0.0 toward the target by ensuring Kaggle can score the submission.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with Kaggle being unable to score the file (usually wrong row count / missing ids / misalignment), and your environment notes show `sample_submission.csv` has fewer rows than `test.csv`, so I avoid using it entirely and instead hard-validate that `submission.csv` matches `test.csv` exactly. To move the score up from 0.0 toward the target (higher is better) without changing your model core logic, I keep the same tokenizer/model inference but make the prediction-to-score mapping robust to common Hugging Face label setups (`num_labels==1` regression vs `num_labels==6` classification) so we don’t accidentally produce degenerate constant predictions. Finally, I ensure the model directory resolution prefers exact matches and only falls back if the directory truly can’t be found, preventing an unnecessary baseline fallback that would score poorly.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from a “valid CSV but unscorable/degenerate predictions” issue rather than model quality, so I make the smallest changes that ensure we actually use your local model (if present) and that we map its outputs to 1–6 correctly for *both* common Hugging Face setups: regression (num_labels=1) and classification (num_labels=6), including handling `id2label` like `LABEL_0..LABEL_5`. I also remove the protobuf pip-install step (which can silently fail/offline) and rely on the already-set pure-Python protobuf env var to keep imports stable. Finally, I add a tiny sanity check that predictions are not all-constant; if they are, we fall back to rounding the train mean (still valid), but only in that degenerate case.'
- What this solution (achieved 0.0) has done: 'I fix the import-time protobuf crash that prevents any run by forcing a compatible protobuf runtime *before* importing `datasets/transformers`, and by removing the fragile dependency on compiled protobuf internals that triggers `MessageFactory.GetPrototype`. Then I keep your model loading/inference logic intact but make the local model path resolution and Hugging Face loading more robust (still local-only) so we don’t unnecessarily fall back. Finally, I ensure the submission always matches `test.csv` exactly (row count + ordering) and is written as `submission.csv` with valid integer scores 1–6, which should move the score up from 0.0 toward the target by producing a scorable file.'
- What this solution (achieved 0.0) has done: 'I fix the import-time protobuf crash by forcing the pure-Python protobuf runtime *and* applying a small compatibility monkey-patch for the missing `MessageFactory.GetPrototype` method before importing `datasets/transformers`. Then I keep your model loading/inference logic intact, but make the model path resolution faster/safer (bounded directory walk) so it doesn’t hang and can actually find the local model directory. Finally, I keep your logits→score mapping unchanged (classification argmax or regression rounding), and ensure the submission is written directly from `test.csv` ordering with valid integer scores 1–6.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf/MessageFactory crash by forcing the pure-Python protobuf implementation *and* patching the correct class method (`GetPrototype`) before importing `datasets/transformers`, using a safer patch that works whether `MessageFactory` is a class or instance. This unblocks the rest of your existing pipeline without changing your model/training/inference logic. I keep your local-model resolution, inference, and logits→score mapping intact, only ensuring imports happen after the protobuf fix so a valid `submission.csv` is always produced. This should move the score up from 0.0 (crash/unscorable) toward the target by producing a properly generated, scorable submission.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by applying the patch in a way that works for both `google.protobuf` and `google.protobuf.internal` implementations and by verifying the attribute after patching (so the import chain doesn’t die). Then I keep your model loading/inference and logits→score mapping unchanged, only making sure the environment variable is set before any protobuf/transformers imports. Finally, I keep the submission written directly from `test.csv` ordering (not `sample_submission`) and retain the existing validations so Kaggle can score it instead of returning 0.0.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by making the patch run earlier and patch the correct `MessageFactory` class used by the runtime, then verify it before importing `datasets/transformers`. This is a pure stability fix and doesn’t change your modeling logic. I also keep everything else intact, only adjusting the import order so the environment doesn’t error before inference and submission writing. With the crash removed, the notebook should run end-to-end and produce a scorable `submission.csv`, moving the score up from 0.0 toward your target.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _patch_protobuf_messagefactory_getprototype() -> None:
    """
    Some environments raise: AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
    during import of libraries that expect GetPrototype to exist.

    Minimal, targeted fix:
    - Patch google.protobuf.message_factory.MessageFactory (and internal variant if present)
      so that instances have GetPrototype, implemented via GetMessageClass when available.
    """
    try:
        modules_to_try = []

        try:
            from google.protobuf import message_factory as mf_mod  # type: ignore

            modules_to_try.append(mf_mod)
        except Exception:
            pass

        try:
            from google.protobuf.internal import message_factory as mf_mod_internal  # type: ignore

            modules_to_try.append(mf_mod_internal)
        except Exception:
            pass

        for mod in modules_to_try:
            MF = getattr(mod, "MessageFactory", None)
            if MF is None:
                continue

            mf_class = MF if isinstance(MF, type) else MF.__class__

            if not hasattr(mf_class, "GetPrototype") and hasattr(
                mf_class, "GetMessageClass"
            ):

                def _GetPrototype(self, descriptor):  # type: ignore
                    return self.GetMessageClass(descriptor)

                setattr(mf_class, "GetPrototype", _GetPrototype)

        try:
            from google.protobuf import message_factory as _mf_check  # type: ignore

            inst = _mf_check.MessageFactory()
            if (not hasattr(inst, "GetPrototype")) and hasattr(inst, "GetMessageClass"):

                def _GetPrototype2(self, descriptor):  # type: ignore
                    return self.GetMessageClass(descriptor)

                setattr(inst.__class__, "GetPrototype", _GetPrototype2)
        except Exception:
            pass

    except Exception:
        pass


_patch_protobuf_messagefactory_getprototype()

import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
MODEL_NAME = "/kaggle/input/f-tfm-small/funnel-small-ft_ver4"
MAX_LENGTH = 3072
BATCH_SIZE = 2
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



## === cell 2
df_train = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
df_test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)

assert {"essay_id", "full_text", "score"}.issubset(df_train.columns)
assert {"essay_id", "full_text"}.issubset(df_test.columns)

if df_test["essay_id"].isna().any():
    raise ValueError("Found NaN essay_id in test.csv.")
if df_test["essay_id"].duplicated().any():
    dups = df_test.loc[df_test["essay_id"].duplicated(), "essay_id"].head(5).tolist()
    raise ValueError(f"Duplicate essay_id in test.csv; examples: {dups}")




## === cell 3
def resolve_model_dir(model_path: str) -> str | None:
    """
    Keep core logic, but be stricter/safer about picking the right local directory:
    - Prefer exact given path if it exists.
    - Then try /kaggle/input/{model_path} if model_path was relative.
    - Then search by leaf directory name as last resort (bounded walk for speed).
    """
    candidates: list[str] = [model_path]

    if not model_path.startswith("/kaggle/"):
        candidates.append(os.path.join("/kaggle/input", model_path))

    leaf = os.path.basename(model_path.rstrip("/"))
    if leaf:
        max_dirs_to_scan = 5000
        scanned = 0
        for base in ["/kaggle/input", "/kaggle/data", "/kaggle/working"]:
            if not os.path.isdir(base):
                continue
            for root, dirs, _files in os.walk(base):
                scanned += 1
                if scanned > max_dirs_to_scan:
                    break
                if leaf in dirs:
                    candidates.append(os.path.join(root, leaf))
            if scanned > max_dirs_to_scan:
                break

    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    for c in uniq:
        if os.path.isdir(c):
            return c
    return None


model_dir = resolve_model_dir(MODEL_NAME)

use_fallback_baseline = model_dir is None
if use_fallback_baseline:
    print(f"WARNING: Could not find MODEL_NAME directory: {MODEL_NAME}")
    print("WARNING: Falling back to a simple baseline to produce a valid submission.")
else:
    print(f"Using local model directory: {model_dir}")



## === cell 4
tokenizer = None
model = None
test_ds = None

if not use_fallback_baseline:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True
        )

        model.eval()
        if torch.cuda.is_available():
            model.cuda()

        def tokenize(batch):
            return tokenizer(batch["full_text"], max_length=MAX_LENGTH, truncation=True)

        test_ds = (
            Dataset.from_pandas(df_test[["essay_id", "full_text"]])
            .map(tokenize, batched=True)
            .remove_columns(["essay_id", "full_text"])
        )
    except Exception as e:
        print(f"WARNING: Model/tokenizer pipeline failed: {repr(e)}")
        print(
            "WARNING: Falling back to a simple baseline to produce a valid submission."
        )
        use_fallback_baseline = True
        tokenizer = None
        model = None
        test_ds = None



## === cell 5
preds = None

if not use_fallback_baseline:
    args = TrainingArguments(
        output_dir=".",
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none",
        dataloader_drop_last=False,
        seed=SEED,
    )

    trainer = Trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    preds = trainer.predict(test_ds).predictions




## === cell 6
def _labels_from_id2label(id2label: dict) -> np.ndarray | None:
    """
    Minimal robustness: handle id2label like:
      {0: 'LABEL_0', 1: 'LABEL_1', ...} or {0:'1',1:'2',...}.
    Returns mapped score values for each class index, or None if not parseable.
    """
    try:
        keys = sorted(int(k) for k in id2label.keys())
        labels = [str(id2label[k]).strip() for k in keys]
        mapped = []
        for s in labels:
            if s.upper().startswith("LABEL_"):
                s = s.split("_", 1)[1]
            mapped.append(int(s))
        return np.asarray(mapped, dtype=int)
    except Exception:
        return None


if use_fallback_baseline:
    mean_score = float(df_train["score"].mean())
    const_pred = int(np.clip(np.rint(mean_score), 1, 6))
    df_test["score"] = const_pred
else:
    if preds is None:
        raise RuntimeError(
            "Internal error: preds is None despite model inference path."
        )

    preds_arr = np.asarray(preds)

    if preds_arr.ndim == 2 and preds_arr.shape[1] == 1:
        df_test["score"] = np.clip(np.rint(preds_arr[:, 0]), 1, 6).astype(int)
    elif preds_arr.ndim == 1:
        df_test["score"] = np.clip(np.rint(preds_arr), 1, 6).astype(int)
    elif preds_arr.ndim == 2:
        pred_idx = np.argmax(preds_arr, axis=1).astype(int)  # 0..C-1

        mapped = None
        if hasattr(model, "config") and getattr(model.config, "id2label", None):
            mapped = _labels_from_id2label(model.config.id2label)

        if mapped is not None and len(mapped) == preds_arr.shape[1]:
            df_test["score"] = mapped[pred_idx].astype(int)
        else:
            df_test["score"] = (pred_idx + 1).astype(int)
    else:
        df_test["score"] = int(np.clip(np.rint(float(df_train["score"].mean())), 1, 6))

    if df_test["score"].nunique() == 1:
        mean_score = float(df_train["score"].mean())
        const_pred = int(np.clip(np.rint(mean_score), 1, 6))
        print(
            "WARNING: Model predictions are constant; falling back to rounded train mean "
            f"({const_pred}) to avoid degenerate submission."
        )
        df_test["score"] = const_pred



## === cell 7
submission = df_test[["essay_id", "score"]].copy()

if submission.shape[0] != df_test.shape[0]:
    raise ValueError(
        f"Submission row count {submission.shape[0]} != test row count {df_test.shape[0]}"
    )
if submission["essay_id"].isna().any():
    raise ValueError("Found NaN essay_id in submission.")
if submission["essay_id"].duplicated().any():
    dups = (
        submission.loc[submission["essay_id"].duplicated(), "essay_id"].head(5).tolist()
    )
    raise ValueError(f"Duplicate essay_id in submission; examples: {dups}")

submission["score"] = submission["score"].astype(int).clip(1, 6)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print(f"Wrote submission.csv with shape: {submission.shape}")
print(f"Score value counts:\n{submission['score'].value_counts().sort_index()}")
