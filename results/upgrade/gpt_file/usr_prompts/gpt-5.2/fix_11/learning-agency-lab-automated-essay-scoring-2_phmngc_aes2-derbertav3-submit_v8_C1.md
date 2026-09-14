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

3.14

# 3. Installed packages

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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1
transformers==4.53.3

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

0.8019899540067545

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.12599) has done: 'I fix the two root causes preventing an end-to-end run: (1) the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation before importing `transformers`, and (2) the Hugging Face loader failing because the provided `MODEL_PATH` isn’t a valid local `from_pretrained` directory in this environment. To keep core logic intact while ensuring a submission is produced, I add a robust local-path resolver and a safe fallback to a public DeBERTa model if the Kaggle input model folder is missing/unloadable. I also make sure the prediction code always returns a correctly formatted `submission.csv` with `essay_id,score` aligned to the test set order. No changes are made to the chunking/prediction aggregation logic beyond making it run reliably.'
- What this solution (achieved 0.0) has done: 'You’re currently crashing at import time due to an incompatibility between `transformers` and the installed protobuf runtime (`MessageFactory.GetPrototype`), so the first fix is to force a protobuf version that works in Kaggle by uninstalling `protobuf`/`googleapis-common-protos` and installing a compatible protobuf (3.20.x) before importing `transformers`. Next, your predictions are likely badly calibrated for QWK because you’re using a 6-class classifier head with `argmax+1` on a base model that isn’t your intended fine-tuned checkpoint; to move the score toward the target, we keep the same inference logic but make the local model-path resolver actually find and load the Kaggle-provided checkpoint reliably (and avoid falling back to an unfine-tuned public model). Finally, we ensure the written file is exactly `submission.csv` with `essay_id,score` aligned to the test order and all scores clipped to 1–6.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with an “invalid submission” or a badly misaligned prediction scale, so the smallest safe improvement is to guarantee the model you load is the intended fine-tuned local checkpoint and that the written `submission.csv` exactly matches the required schema and row order. I keep your chunking + mean-aggregation + rounding core logic unchanged, but I (1) strengthen the local model directory resolver to correctly handle Kaggle dataset layouts (including `snapshots/...`), and (2) enforce `id2label/label2id`-aware mapping so we don’t accidentally shift labels with `+1` when the checkpoint already encodes labels 1–6. Finally, I add a lightweight sanity check that the submission has exactly `len(test)` rows, the right columns, and scores are in [1,6], so you don’t end up with a 0.0 due to formatting/alignment issues.'
- What this solution (achieved 0.0) has done: 'I fix the import-time protobuf crash by enforcing the pure-Python protobuf implementation *before* any `transformers` import, and by guarding against stale `google.protobuf` modules already loaded. Then I make model loading robust: it try to load your provided local checkpoint, but if that path truly doesn’t exist in this notebook’s inputs it fall back to a known public DeBERTa model so the pipeline still completes and writes a valid `submission.csv`. Finally, I keep your chunking, aggregation, and rounding logic unchanged, and add a small safety check to ensure the submission is aligned to `test.csv` order with valid score bounds (1–6). These changes are primarily to ensure an end-to-end run and a valid submission; score improvement depends on whether your fine-tuned checkpoint is actually available.'
- What this solution (achieved -0.12508) has done: 'We fix the import-time protobuf crash that stops the notebook at cell 1 by forcing the pure-Python protobuf implementation before any `transformers` import and by defensively patching `google.protobuf`’s `MessageFactory.GetPrototype` when the runtime lacks it. This is a correctness/stability fix that enables the rest of your pipeline (model load → chunking → inference → CSV write) to run end-to-end and produce a valid `submission.csv`. We keep your model inference/chunking/aggregation logic unchanged, and we also make the fallback model load “local-only first” to avoid internet dependency in Kaggle. Finally, we preserve the required submission schema and alignment checks so you don’t get another 0.0 from an invalid/misaligned file.'
- What this solution (achieved 0.0) has done: 'We fix the import-time protobuf crash by applying a safe, version-agnostic monkeypatch to `google.protobuf.message_factory.MessageFactory.GetPrototype` (and the module-level `GetPrototype`) before importing `transformers`, so the notebook runs end-to-end. We also make sure the script can always load a model locally (no internet) by searching the provided Kaggle input tree for a valid HF checkpoint if `MODEL_PATH` is missing, while keeping your inference/chunking/aggregation logic unchanged. Finally, we keep the submission formatting/alignment checks, ensuring a valid `submission.csv` is always written.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_DOWNLOADS", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

TRAIN_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
TEST_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
MODEL_PATH = "/kaggle/input/aes2-debertav3large/transformers/default/1"
OUTPUT_PATH = "/kaggle/working/"

MAX_LEN = 512
OVERLAP = 64
MIN_CHUNK_RATIO = 0.3  # Skip if this chunk < RATIO * MAX_LEN
BATCH_SIZE = 2
NUM_LABELS = 6

import sys

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

try:
    from google.protobuf import message_factory as _message_factory

    def _ensure_getprototype_on_factory_instance():
        if not hasattr(_message_factory.MessageFactory, "GetPrototype"):
            if hasattr(_message_factory.MessageFactory, "GetMessageClass"):

                def _GetPrototype(self, descriptor):
                    return self.GetMessageClass(descriptor)

                _message_factory.MessageFactory.GetPrototype = _GetPrototype
            else:

                def _GetPrototype(self, descriptor):
                    raise AttributeError(
                        "GetPrototype is not available in this protobuf runtime."
                    )

                _message_factory.MessageFactory.GetPrototype = _GetPrototype

        if not hasattr(_message_factory, "GetPrototype"):
            if hasattr(_message_factory, "GetMessageClass"):

                def _mf_GetPrototype(descriptor):
                    return _message_factory.GetMessageClass(descriptor)

                _message_factory.GetPrototype = _mf_GetPrototype

        gen = getattr(_message_factory, "_GENERATED_FACTORY", None)
        if gen is not None and not hasattr(gen, "GetPrototype"):
            try:
                gen.GetPrototype = _message_factory.MessageFactory.GetPrototype.__get__(
                    gen, gen.__class__
                )
            except Exception:
                pass

    _ensure_getprototype_on_factory_instance()
except Exception:
    pass

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")




## === cell 1
def _resolve_local_model_dir(model_path: str) -> str | None:
    """
    Ensure we reliably find the actual local fine-tuned HF checkpoint directory.
    Handles common Kaggle layouts including:
      - transformers/default/1
      - snapshots/<hash> (from HF cache-like exports)
    """
    if not model_path:
        return None

    candidates = []
    for p in [
        model_path,
        os.path.dirname(model_path),
        os.path.dirname(os.path.dirname(model_path)),
        os.path.dirname(os.path.dirname(os.path.dirname(model_path))),
    ]:
        if p and os.path.isdir(p):
            candidates.append(p)

    probes = [
        model_path,
        os.path.join(model_path, "transformers", "default", "1"),
        os.path.join(model_path, "transformers", "default"),
        os.path.join(model_path, "transformers"),
        os.path.join(model_path, "1"),
    ]
    for p in probes:
        if p and os.path.isdir(p):
            candidates.append(p)

    seen = set()
    uniq = []
    for c in candidates:
        c = os.path.abspath(c)
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    def looks_like_hf_dir(d: str) -> bool:
        if not os.path.isdir(d):
            return False
        has_config = os.path.isfile(os.path.join(d, "config.json"))
        has_weights = any(
            os.path.isfile(os.path.join(d, fn))
            for fn in ["pytorch_model.bin", "model.safetensors"]
        )
        return has_config and has_weights

    for c in uniq:
        if looks_like_hf_dir(c):
            return c

    for c in uniq:
        snap_root = os.path.join(c, "snapshots")
        if os.path.isdir(snap_root):
            try:
                for sub in sorted(os.listdir(snap_root)):
                    d = os.path.join(snap_root, sub)
                    if looks_like_hf_dir(d):
                        return d
            except Exception:
                pass

        try:
            for root, dirs, files in os.walk(c):
                rel_depth = os.path.relpath(root, c).count(os.sep)
                if rel_depth > 8:
                    dirs[:] = []
                    continue
                if "config.json" in files and (
                    "pytorch_model.bin" in files or "model.safetensors" in files
                ):
                    return root
        except Exception:
            continue

    return None


def _search_kaggle_input_for_checkpoint() -> str | None:
    """
    Search /kaggle/input for any local HF checkpoint folder (config.json + weights).
    """
    root = "/kaggle/input"
    if not os.path.isdir(root):
        return None

    def looks_like_hf_dir(d: str) -> bool:
        if not os.path.isdir(d):
            return False
        if not os.path.isfile(os.path.join(d, "config.json")):
            return False
        if not any(
            os.path.isfile(os.path.join(d, fn))
            for fn in ["pytorch_model.bin", "model.safetensors"]
        ):
            return False
        return True

    preferred_tokens = [
        "aes",
        "essay",
        "deberta",
        "automated",
        "scoring",
        "lab",
        "model",
        "checkpoint",
    ]
    try:
        for top in sorted(os.listdir(root)):
            top_path = os.path.join(root, top)
            if not os.path.isdir(top_path):
                continue
            low = top.lower()
            if not any(t in low for t in preferred_tokens):
                continue
            for dirpath, dirnames, filenames in os.walk(top_path):
                rel_depth = os.path.relpath(dirpath, top_path).count(os.sep)
                if rel_depth > 10:
                    dirnames[:] = []
                    continue
                if "config.json" in filenames and (
                    "pytorch_model.bin" in filenames or "model.safetensors" in filenames
                ):
                    if looks_like_hf_dir(dirpath):
                        return dirpath
    except Exception:
        pass

    try:
        for top in sorted(os.listdir(root)):
            top_path = os.path.join(root, top)
            if not os.path.isdir(top_path):
                continue
            for dirpath, dirnames, filenames in os.walk(top_path):
                rel_depth = os.path.relpath(dirpath, top_path).count(os.sep)
                if rel_depth > 8:
                    dirnames[:] = []
                    continue
                if "config.json" in filenames and (
                    "pytorch_model.bin" in filenames or "model.safetensors" in filenames
                ):
                    if looks_like_hf_dir(dirpath):
                        return dirpath
    except Exception:
        pass

    return None


def _load_tokenizer_and_model_strict_local(model_path: str):
    """
    Returns (tokenizer, model, resolved_dir) or (None, None, None) if no local checkpoint exists.
    """
    resolved = _resolve_local_model_dir(model_path)
    if resolved is None:
        alt = _search_kaggle_input_for_checkpoint()
        if alt is not None:
            print(
                f"[INFO] MODEL_PATH not usable; found alternate local checkpoint: {alt}"
            )
            resolved = alt

    if resolved is None:
        return None, None, None

    try:
        tok = AutoTokenizer.from_pretrained(
            resolved, local_files_only=True, use_fast=True
        )
        mdl = AutoModelForSequenceClassification.from_pretrained(
            resolved, local_files_only=True
        )
        print(f"Loaded local model via Auto* from: {resolved}")
        return tok, mdl, resolved
    except Exception as e_auto:
        print(f"[WARN] Local Auto* load failed ({resolved}): {repr(e_auto)}")

    try:
        tok = DebertaV2Tokenizer.from_pretrained(resolved, local_files_only=True)
        mdl = DebertaV2ForSequenceClassification.from_pretrained(
            resolved, local_files_only=True
        )
        print(f"Loaded local model via DebertaV2* from: {resolved}")
        return tok, mdl, resolved
    except Exception as e_deb:
        print(f"[WARN] Local DebertaV2* load failed ({resolved}): {repr(e_deb)}")

    return None, None, None


tokenizer, model, _resolved_dir = _load_tokenizer_and_model_strict_local(MODEL_PATH)
USE_TRANSFORMER = (tokenizer is not None) and (model is not None)

if (
    USE_TRANSFORMER
    and hasattr(model, "config")
    and getattr(model.config, "num_labels", None) is not None
):
    NUM_LABELS = int(model.config.num_labels)

print(f"USE_TRANSFORMER={USE_TRANSFORMER} | Using NUM_LABELS={NUM_LABELS}")




## === cell 2
def plot_confusion_matrix(preds, labels, title="Confusion Matrix"):
    preds = np.array(preds)
    labels = np.array(labels)

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greens", values_format="d")
    plt.title(title)
    plt.show()




## === cell 3
def compute_metrics(preds, labels, device="cuda"):
    if len(preds.shape) > 1:
        preds = np.argmax(preds, axis=-1)
    qwk_score = cohen_kappa_score(labels, preds, weights="quadratic")
    acc_score = accuracy_score(labels, preds)
    return {"quadratic_weighted_kappa": qwk_score, "accuracy": acc_score}




## === cell 4
def _infer_label_mapping(mdl, num_labels: int):
    """
    Avoid incorrect '+1' shifting when the checkpoint's id2label already maps
    indices to 1..6 or otherwise non-0-based labels. If mapping is unclear, fall back to +1.
    Returns a function: idx_array -> score_array (int).
    """
    cfg = getattr(mdl, "config", None)
    id2label = getattr(cfg, "id2label", None) if cfg is not None else None

    def default_map(idxs: np.ndarray) -> np.ndarray:
        return idxs.astype(int) + 1

    if not isinstance(id2label, dict) or len(id2label) != num_labels:
        return default_map

    try:
        keys = sorted(int(k) for k in id2label.keys())
        labels = [id2label[str(k)] if str(k) in id2label else id2label[k] for k in keys]

        numeric = []
        for lab in labels:
            if isinstance(lab, (int, np.integer)):
                numeric.append(int(lab))
            elif isinstance(lab, str) and lab.strip().isdigit():
                numeric.append(int(lab.strip()))
            else:
                numeric = []
                break

        if len(numeric) == num_labels and sorted(numeric) == list(
            range(1, num_labels + 1)
        ):
            mapping = {k: v for k, v in zip(keys, numeric)}

            def mapped(idxs: np.ndarray) -> np.ndarray:
                return np.vectorize(lambda z: mapping.get(int(z), int(z) + 1))(
                    idxs
                ).astype(int)

            print("[INFO] Using config.id2label-based mapping (no forced +1 shift).")
            return mapped
    except Exception:
        pass

    return default_map


def predict_essay_score(model, dataset, num_labels, batch_size, device):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.eval()

    map_fn = _infer_label_mapping(model, num_labels)

    preds, essay_ids_all = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            essay_ids = batch["essay_id"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            pred_class = torch.argmax(logits, axis=1).cpu().numpy()
            pred_labels = map_fn(pred_class)

            preds.extend(pred_labels)
            essay_ids_all.extend(essay_ids)

    preds = np.array(preds)
    essay_ids_all = np.array(essay_ids_all)

    chunk_results = pd.DataFrame({"essay_id": essay_ids_all, "raw_prediction": preds})

    aggregated_predictions = chunk_results.groupby("essay_id")["raw_prediction"].mean()
    y_pred_aggregated = np.rint(aggregated_predictions.values)
    final_scores = np.clip(y_pred_aggregated, 1, num_labels).astype(int)

    final_results_df = pd.DataFrame(
        {"essay_id": aggregated_predictions.index.values, "score": final_scores}
    )

    return final_results_df, preds




## === cell 5
class TestEssayDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        max_len: int = 512,
        overlap: int = 128,
        min_chunk_ratio: float = 0.3,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.overlap = overlap
        self.min_chunk_ratio = min_chunk_ratio

        num_special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        self.max_content_len = max_len - num_special_tokens

        for _, row in df.iterrows():
            essay_id = row["essay_id"]
            text = "[A] " + row["full_text"]
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]

            step = self.max_content_len - overlap
            start = 0

            while start < len(tokens):
                end = start + self.max_content_len
                chunk_content = tokens[start:end]
                if len(chunk_content) < self.max_content_len * self.min_chunk_ratio:
                    break

                processed_tokens = tokenizer.build_inputs_with_special_tokens(
                    chunk_content
                )
                padding_len = self.max_len - len(processed_tokens)
                if padding_len > 0:
                    processed_tokens += [tokenizer.pad_token_id] * padding_len

                self.samples.append((processed_tokens, essay_id))
                start += step

                if end >= len(tokens):
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, essay_id = self.samples[idx]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "essay_id": essay_id,
        }




## === cell 6

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

df_test = pd.read_csv(TEST_PATH)

if USE_TRANSFORMER:
    test_dataset = TestEssayDataset(
        df_test, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO
    )

    test_results, _ = predict_essay_score(
        dataset=test_dataset,
        model=model,
        num_labels=NUM_LABELS,
        batch_size=BATCH_SIZE,
        device=device,
    )
    test_results = df_test[["essay_id"]].merge(test_results, on="essay_id", how="left")
    test_results["score"] = test_results["score"].fillna(3).astype(int)
    test_results["score"] = test_results["score"].clip(1, NUM_LABELS)
else:
    print(
        "[WARN] No local HF checkpoint found. Using TF-IDF + Ridge fallback to generate submission."
    )
    df_train = pd.read_csv(TRAIN_PATH)

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        sublinear_tf=True,
    )
    X_train = vec.fit_transform(df_train["full_text"].astype(str).values)
    y_train = df_train["score"].astype(float).values
    X_test = vec.transform(df_test["full_text"].astype(str).values)

    reg = Ridge(alpha=1.0, random_state=42)
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    score = np.rint(pred).astype(int)
    score = np.clip(score, 1, 6)

    test_results = pd.DataFrame(
        {"essay_id": df_test["essay_id"].values, "score": score}
    )

test_results = test_results[["essay_id", "score"]]
assert len(test_results) == len(df_test), "Submission row count mismatch with test.csv"
assert list(test_results.columns) == [
    "essay_id",
    "score",
], "Submission columns incorrect"

print("Submission head:")
print(test_results.head())
print(f"Rows: {len(test_results)} (expected {len(df_test)})")
print("Score value counts:")
print(test_results["score"].value_counts().sort_index())



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    114             try:
--> 115                 coef, info = sp_linalg.cg(C, y_column, tol=tol, atol="legacy")
    116             except TypeError:

TypeError: cg() got an unexpected keyword argument 'tol'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_54/693524873.py in <cell line: 0>()
     42 
     43     reg = Ridge(alpha=1.0, random_state=42)
---> 44     reg.fit(X_train, y_train)
     45     pred = reg.predict(X_test)
     46 

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
    116             except TypeError:
    117                 # old scipy
--> 118                 coef, info = sp_linalg.cg(C, y_column, tol=tol)
    119             coefs[i] = X1.rmatvec(coef)
    120         else:

TypeError: cg() got an unexpected keyword argument 'tol'

## === cell 7
sub_path = os.path.join(OUTPUT_PATH, "submission.csv")
test_results.to_csv(sub_path, index=False)

check = pd.read_csv(sub_path)
assert len(check) == len(df_test), "Written CSV row count mismatch"
assert list(check.columns) == ["essay_id", "score"], "Written CSV columns mismatch"
assert check["score"].between(1, 6).all(), "Written CSV has out-of-range scores"

print(f"Wrote submission to: {sub_path}")
print(check.head())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/1039014934.py in <cell line: 0>()
      1 sub_path = os.path.join(OUTPUT_PATH, "submission.csv")
----> 2 test_results.to_csv(sub_path, index=False)
      3 
      4 check = pd.read_csv(sub_path)
      5 assert len(check) == len(df_test), "Written CSV row count mismatch"

NameError: name 'test_results' is not defined
