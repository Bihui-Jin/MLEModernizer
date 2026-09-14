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

0.7924246284831522

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the failure to locate the local HuggingFace model by (1) falling back to a compatible pretrained model that can be downloaded in Kaggle (so the notebook runs even when the attached model dataset is missing), and (2) making the resolver also search the competition dataset folders under `/kaggle/input/learning-agency-lab-automated-essay-scoring-2/`. I keep your chunking/overlap inference logic and aggregation exactly the same, only making the model loading robust so later cells can run. Finally, I ensure a valid `submission.csv` is always written with the required `essay_id,score` columns and correct row count/order.'
- What this solution (achieved 0.0) has done: 'I fix the `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by switching the protobuf runtime back to the C++/upb implementation (your current env var forces the pure-Python protobuf backend that triggers this Transformers import/model load crash in some Kaggle images). I also make the model-loading more robust in offline Kaggle runs by (a) preferring any attached local HF model and (b) only attempting an online fallback if explicitly possible; if not, we fail fast with a clear message instead of producing a broken 0-score submission. Finally, I keep your chunking/inference/aggregation exactly the same and ensure `submission.csv` is always written with correct columns and row order.'
- What this solution (achieved 0.00335) has done: 'I fix the Transformers import/model-load crash causing `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by forcing the safe protobuf Python implementation *before* importing `transformers`, which is the minimal change needed to unblock execution in this Kaggle image. I also make model loading deterministic/offline-safe by preferring the provided local `MODEL_PATH` and, if that fails, scanning `/kaggle/input` for any attached HF model directory; only then it try the remote fallback (which likely won’t work without internet). The chunking/overlap inference, aggregation, and submission formatting remain unchanged so the core logic and evaluation semantics are preserved. Finally, the script always write `/kaggle/working/submission.csv` with the required `essay_id,score` columns and correct row count/order.'
- What this solution (achieved 0.0) has done: 'I fix the Transformers/protobuf crash by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` *and* disabling the problematic `google.protobuf` C-descriptor path before importing `transformers`, then force a clean import so the environment variable actually takes effect. I also make model loading more robust and score-improving by preferring the competition’s commonly-attached local DeBERTa model dataset (if present) and by ensuring we load the fine-tuned checkpoint correctly (rather than accidentally constructing a fresh random head). Finally, I keep your chunking/overlap inference and aggregation unchanged, and ensure we always write `/kaggle/working/submission.csv` with the required `essay_id,score` columns and correct row order.'
- What this solution (achieved 0.0) has done: 'I fix the failure to load the HuggingFace model/tokenizer by (1) strictly preferring a valid local HF folder and (2) removing the broken “remote fallback but forced local_files_only=True” path that currently guarantees a crash in Kaggle’s offline environment. I also add a safe, score-neutral fallback baseline (predict score=3 for all) that only triggers if no local model is available, so the notebook always runs end-to-end and writes a valid `submission.csv`. Finally, I keep your chunking/overlap inference, aggregation, and submission formatting unchanged when the model loads successfully, preserving the core logic and evaluation semantics.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score strongly suggests the submission is invalid or severely misaligned (most commonly: wrong row count/order vs the sample, wrong IDs, or non-integer/out-of-range scores), rather than the model being intrinsically that bad. I make the smallest changes that improve validity and stability: (1) always enforce the exact `essay_id` order from `sample_submission.csv`, (2) aggregate chunk predictions using logits-probabilities (expected value) instead of averaging hard labels (same model, same inference loop; just a more metric-consistent aggregation), and (3) remove the FP16 cast at inference to avoid numerical instability on some GPUs. These changes preserve your core chunking/inference logic and move the score upward toward the target by producing a correctly aligned, better-calibrated submission without changing architecture or training. The script still write `/kaggle/working/submission.csv` with exactly 1731 rows and the required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score most likely comes from a still-invalid submission or extreme misalignment between the test IDs you predict for (15335) and the sample submission IDs (1731), causing most rows to be missing and default-filled to 3. I minimally fix this by ensuring we always predict for exactly the `essay_id`s in `sample_submission.csv` (inner-join/filter test to those IDs before chunking/inference), which preserves your model and chunking logic but prevents mass fallback-to-3. I also add a hard validation that the merged predictions have no missing rows when a model is available (to catch silent ID mismatches), while keeping your rounding/clipping and aggregation semantics intact. This should move the score sharply upward toward the target without changing architecture, loss, or inference design.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an invalid submission pipeline (often caused by failing model load and falling back to a constant baseline), not with a “bad but valid” model. The smallest score-improving change toward your target is to make sure we actually load the intended local DeBERTa checkpoint reliably in Kaggle’s offline environment, and fail over only to other local checkpoints (not a silent baseline). I also enforce a deterministic, correct label mapping by using the checkpoint’s `config.num_labels` (and only falling back to `NUM_LABELS=6` if missing), so we don’t accidentally create a fresh random head. Core chunking/overlap inference and aggregation stay the same; submission remains aligned to `sample_submission.csv` and always writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from an invalid or degenerate submission (most commonly: the model never actually loads and you fall back to constant “3”, or the loaded checkpoint has an uninitialized/random classification head). I make the smallest changes that keep your chunking + overlap inference and aggregation intact, but ensure we load a *valid fine-tuned checkpoint* by requiring `pytorch_model.bin`/`model.safetensors` to exist alongside `config.json`, and by scanning `/kaggle/input` for a real checkpoint if your `MODEL_PATH` points to a partial folder. I also remove the tokenizer/model resize mismatch risk by only resizing embeddings when needed (unchanged behavior, but avoids accidental weight changes when the checkpoint already matches). Finally, I keep your submission alignment to `sample_submission.csv` exactly as-is so the output remains valid and should move the score sharply upward toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because you’re clipping to `NUM_LABELS=6` even when the loaded checkpoint predicts more labels (many AES2 DeBERTa checkpoints are trained with 7 labels for scores 1–7). That forces all predicted 7s down to 6 and can severely hurt QWK. I keep your exact chunking/overlap inference and expected-value aggregation, but (1) make the output clipping use `EFFECTIVE_NUM_LABELS` (the checkpoint’s label space) and (2) ensure the expected-value class values match the model’s logits dimension to avoid silent mismatches. This is a minimal, metric-aligned fix that should move the score up toward your target without changing the modeling approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because the code falls back to the constant baseline (score=3) when it can’t find the local model at `MODEL_PATH`, so the minimal score-improving step is to reliably locate the attached DeBERTa checkpoint under `/kaggle/input` and actually run your existing chunking + expected-value aggregation inference. I make `find_any_hf_model_dir()` search recursively (not just one directory level) for likely model folders, prioritizing DeBERTa/AES paths, without changing your dataset chunking, model forward pass, or aggregation logic. I also make the score clipping consistently use the model’s effective label space to avoid harming QWK if the checkpoint’s `num_labels` differs. The output submission remain strictly aligned to `sample_submission.csv` with exactly the required columns and row order.'

# 9. Code solution

## === cell 0
TRAIN_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
TEST_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
MODEL_PATH = "/kaggle/input/aes2-debertav3-large/transformers/default/1"
OUTPUT_PATH = "/kaggle/working/"

MAX_LEN = 512
OVERLAP = 64
MIN_CHUNK_RATIO = 0.3  # Skip if this chunk < RATIO * MAX_LEN
BATCH_SIZE = 2
NUM_LABELS = 6



## === cell 1
import os
import sys
import warnings

for k in [
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION",
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION",
]:
    if k in os.environ:
        del os.environ[k]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

for m in [
    "transformers",
    "google.protobuf",
    "google.protobuf.descriptor_pool",
    "google.protobuf.message_factory",
]:
    if m in sys.modules:
        del sys.modules[m]

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore", category=UserWarning)




## === cell 2
def _has_hf_checkpoint_files(model_dir: str) -> bool:
    if not model_dir or not os.path.isdir(model_dir):
        return False
    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        return False
    weight_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    ]
    return any(os.path.isfile(os.path.join(model_dir, wf)) for wf in weight_files)


def resolve_local_model_dir(path: str) -> str:
    if path and os.path.isdir(path) and _has_hf_checkpoint_files(path):
        return path

    if path and os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            if "config.json" in files and _has_hf_checkpoint_files(root):
                return root

    raise FileNotFoundError(
        f"Could not find a local HuggingFace model folder with config.json + weights under: {path}"
    )


def find_any_hf_model_dir(search_roots) -> str:
    if isinstance(search_roots, str):
        search_roots = [search_roots]

    candidates = []
    visited = set()

    def add_candidate(p: str, bonus: int = 0):
        if not p or not os.path.isdir(p):
            return
        key = os.path.realpath(p)
        if key in visited:
            return
        visited.add(key)
        name = os.path.basename(p).lower()
        score = 0 + bonus
        for kw, w in [
            ("deberta", 50),
            ("debertav3", 50),
            ("aes", 30),
            ("essay", 30),
            ("scoring", 20),
            ("transformers", 10),
            ("checkpoint", 10),
            ("model", 5),
        ]:
            if kw in name:
                score += w
        candidates.append((score, p))

    for sr in search_roots:
        add_candidate(sr, bonus=100)

    for sr in search_roots:
        if os.path.isdir(sr):
            try:
                for d in os.listdir(sr):
                    add_candidate(os.path.join(sr, d), bonus=20)
            except Exception:
                pass

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    for _, sr in candidates:
        if not os.path.isdir(sr):
            continue
        for root, dirs, files in os.walk(sr):
            if "config.json" in files and _has_hf_checkpoint_files(root):
                return root

    raise FileNotFoundError(
        f"Could not find any local HuggingFace model folder with config.json + weights under: {search_roots}. "
        f"Check that the model dataset is attached."
    )


resolved_model_dir = None
have_local_model = False

try:
    resolved_model_dir = resolve_local_model_dir(MODEL_PATH)
    have_local_model = True
except FileNotFoundError as e:
    print("MODEL_PATH not usable:", e)
    try:
        resolved_model_dir = find_any_hf_model_dir(
            [
                "/kaggle/input/aes2-debertav3-large",
                "/kaggle/input",
                "/kaggle/input/learning-agency-lab-automated-essay-scoring-2",
                "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/learning-agency-lab-automated-essay-scoring-2",
            ]
        )
        have_local_model = True
    except FileNotFoundError as e2:
        print("No local HF model found:", e2)
        have_local_model = False

if have_local_model:
    print("Resolved local model dir:", resolved_model_dir)
    print(
        "Has config.json:",
        os.path.isfile(os.path.join(resolved_model_dir, "config.json")),
    )
    print("Has weights:", _has_hf_checkpoint_files(resolved_model_dir))
else:
    print("Proceeding without a local model; will use baseline submission (score=3).")



## === cell 3
tokenizer = None
model = None
EFFECTIVE_NUM_LABELS = NUM_LABELS

if have_local_model:
    model_source = resolved_model_dir
    local_only = True

    try:
        config = AutoConfig.from_pretrained(model_source, local_files_only=local_only)
    except Exception:
        config = None

    if (
        config is not None
        and hasattr(config, "num_labels")
        and config.num_labels is not None
    ):
        try:
            EFFECTIVE_NUM_LABELS = int(config.num_labels)
        except Exception:
            EFFECTIVE_NUM_LABELS = NUM_LABELS

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_source, local_files_only=local_only
        )
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            config=config,
            local_files_only=local_only,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to load the local HuggingFace model/tokenizer. "
            "Ensure the intended local model dataset is attached and MODEL_PATH points to a valid HF directory "
            "(must contain config.json and model weights).\n"
            f"Model source: {model_source}\n"
            f"local_files_only={local_only}\n"
            f"Error: {repr(e)}"
        )

    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    if (
        hasattr(model, "config")
        and getattr(model.config, "num_labels", None) is not None
    ):
        EFFECTIVE_NUM_LABELS = int(model.config.num_labels)

    print("Model num_labels (effective):", EFFECTIVE_NUM_LABELS)




## === cell 4
def plot_confusion_matrix(preds, labels, title="Confusion Matrix"):
    preds = np.array(preds)
    labels = np.array(labels)

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greens", values_format="d")
    plt.title(title)
    plt.show()




## === cell 5
def compute_metrics(preds, labels, device="cuda"):
    if len(preds.shape) > 1:
        preds = np.argmax(preds, axis=-1)
    qwk_score = cohen_kappa_score(labels, preds, weights="quadratic")
    acc_score = accuracy_score(labels, preds)
    return {"quadratic_weighted_kappa": qwk_score, "accuracy": acc_score}




## === cell 6
def predict_essay_score(model, dataset, num_labels, batch_size, device):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.eval()

    exp_scores_all, essay_ids_all = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            essay_ids = batch["essay_id"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            k = int(logits.shape[-1])
            class_vals = torch.arange(
                1, k + 1, device=device, dtype=torch.float32
            ).view(1, -1)

            probs = torch.softmax(logits.float(), dim=-1)
            exp_scores = (probs * class_vals).sum(dim=1).detach().cpu().numpy()

            exp_scores_all.extend(exp_scores.tolist())
            essay_ids_all.extend(essay_ids)

    exp_scores_all = np.array(exp_scores_all, dtype=np.float32)
    essay_ids_all = np.array(essay_ids_all)

    chunk_results = pd.DataFrame(
        {"essay_id": essay_ids_all, "exp_score": exp_scores_all}
    )

    aggregated = chunk_results.groupby("essay_id")["exp_score"].mean()
    y_pred_aggregated = np.rint(aggregated.values)

    final_scores = np.clip(y_pred_aggregated, 1, num_labels).astype(int)

    final_results_df = pd.DataFrame(
        {"essay_id": aggregated.index.values, "score": final_scores}
    )

    return final_results_df, exp_scores_all




## === cell 7
class TestEssayDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        max_len: int = 512,
        overlap: int = 64,
        min_chunk_ratio: float = 0.3,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.overlap = overlap
        self.min_chunk_ratio = min_chunk_ratio

        self.num_special = tokenizer.num_special_tokens_to_add(pair=False)

        tag_tokens = tokenizer("[A]", add_special_tokens=False)["input_ids"]
        len_tag = len(tag_tokens)

        self.max_content_len = max_len - self.num_special - len_tag
        if self.max_content_len <= 0:
            raise ValueError(
                f"max_content_len must be > 0, got {self.max_content_len}. "
                f"Check MAX_LEN/num_special/tag length."
            )

        for _, row in df.iterrows():
            essay_id = row["essay_id"]
            base_tokens = tokenizer(row["full_text"], add_special_tokens=False)[
                "input_ids"
            ]

            step = self.max_content_len - overlap
            if step <= 0:
                raise ValueError(
                    f"Step must be > 0, got {step}. Reduce OVERLAP or increase MAX_LEN."
                )

            start = 0
            while start < len(base_tokens):
                end = start + self.max_content_len
                chunk = base_tokens[start:end]

                if (
                    len(chunk) < self.max_content_len * self.min_chunk_ratio
                    and start != 0
                ):
                    break

                chunk_with_tag = tag_tokens + chunk
                processed = tokenizer.build_inputs_with_special_tokens(chunk_with_tag)

                if len(processed) > self.max_len:
                    processed = processed[: self.max_len]

                pad_len = self.max_len - len(processed)
                if pad_len > 0:
                    processed += [tokenizer.pad_token_id] * pad_len

                self.samples.append((processed, essay_id))

                start += step
                if end >= len(base_tokens):
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




## === cell 8
df_test = pd.read_csv(TEST_PATH)

sample_path = (
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
if not os.path.isfile(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"
df_sample = pd.read_csv(sample_path)

sample_ids = df_sample["essay_id"].astype(str)
df_test["essay_id"] = df_test["essay_id"].astype(str)

df_test_eval = df_test.merge(sample_ids.to_frame(), on="essay_id", how="inner")
df_test_eval = df_test_eval.drop_duplicates(subset=["essay_id"], keep="first")

if not have_local_model:
    test_results = df_sample[["essay_id"]].copy()
    test_results["score"] = 3
else:
    test_dataset = TestEssayDataset(
        df_test_eval, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_pred_df, _ = predict_essay_score(
        dataset=test_dataset,
        model=model,
        num_labels=EFFECTIVE_NUM_LABELS,
        batch_size=BATCH_SIZE,
        device=device,
    )

    test_pred_df["essay_id"] = test_pred_df["essay_id"].astype(str)

    test_results = df_sample[["essay_id"]].copy()
    test_results["essay_id"] = test_results["essay_id"].astype(str)
    test_results = test_results.merge(test_pred_df, on="essay_id", how="left")

    missing = int(test_results["score"].isna().sum())
    if missing > 0:
        raise RuntimeError(
            f"Missing predictions for {missing}/{len(test_results)} sample_submission rows. "
            f"This indicates essay_id mismatch between test.csv and sample_submission.csv."
        )

test_results = test_results[["essay_id", "score"]].copy()
test_results["score"] = (
    pd.to_numeric(test_results["score"], errors="coerce").fillna(3).round().astype(int)
)

test_results["score"] = test_results["score"].clip(1, EFFECTIVE_NUM_LABELS)

assert list(test_results.columns) == ["essay_id", "score"]
assert len(test_results) == len(df_sample)
assert test_results["essay_id"].tolist() == df_sample["essay_id"].astype(str).tolist()

print("Submission preview:")
print(test_results.head())
print("Rows:", len(test_results), "Cols:", list(test_results.columns))
print("Any missing scores:", test_results["score"].isna().sum())
print("Score value counts (head):")
print(test_results["score"].value_counts().head(10))



## === cell 9
submission_path = os.path.join(OUTPUT_PATH, "submission.csv")
test_results.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(pd.read_csv(submission_path).head())
print("File size (bytes):", os.path.getsize(submission_path))
