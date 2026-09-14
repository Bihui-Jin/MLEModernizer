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
Given questions and answers from various StackExchange properties, predict target values of 30 labels for each question-answer pair.

## Metric
Mean column-wise Spearman's correlation coefficient. The Spearman's rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.

## Submission Format
For each qa_id in the test set, you must predict a probability for each target variable. The predictions should be in the range [0,1]. The file should contain a header and have the following format:

```
qa_id,question_asker_intent_understanding,...,answer_well_written
6,0.0,...,0.5
8,0.5,...,0.1
18,1.0,...,0.0
etc.
```

## Dataset
The list of 30 target labels are the same as the column names in the `sample_submission.csv` file. Target labels with the prefix `question_` relate to the `question_title` and/or `question_body` features in the data. Target labels with the prefix `answer_` relate to the `answer` feature.

Target labels are aggregated from multiple raters, and can have continuous values in the range `[0,1]`. Therefore, predictions must also be in that range.

- **train.csv** - the training data (target labels are the last 30 columns)
- **test.csv** - the test set (you must predict 30 labels for each test set row)
- **sample_submission.csv** - a sample submission file in the correct format; column names are the 30 target labels

# 2. Python version

3.14

# 3. Installed packages

geopandas==0.14.4
huggingface-hub==0.36.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scipy==1.15.3
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
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        input/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        working/
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
```

-> data/google-quest-challenge/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/google-quest-challenge/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/google-quest-challenge/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> data/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.3649142456650344

# 6. Current score

0.04729

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.04729) has done: 'I fix the tokenizer/model loading by pointing `MODEL_NAME` to a valid local HuggingFace model directory (or falling back to a known public base model if that local path doesn’t exist), and by making checkpoint loading robust: if the `.ckpt` file is missing, we load the model weights directly from `MODEL_NAME` so inference can still run. I also correct the broken cell numbering and ensure the dataset/dataloader are created only after the tokenizer is successfully instantiated. Finally, I keep the same architecture and sigmoid post-processing, and guarantee a valid `submission.csv` with the exact `sample_submission.csv` column order and matching `qa_id` alignment.'
- What this solution (achieved 0.04729) has done: 'The runtime error comes from an old protobuf API incompatibility that `transformers` can hit when importing/using tokenizers/models in this environment; setting the protobuf implementation to Python fixes it without changing your model logic. I also make checkpoint loading more robust by stripping Lightning’s `"model."` prefix and by gracefully handling missing keys, which should materially improve score versus the current near-random fallback. Finally, I ensure `qa_id` stays aligned and always write a valid `submission.csv` with the exact `sample_submission.csv` column order.'
- What this solution (achieved 0.04729) has done: 'You’re hitting a known protobuf/runtime incompatibility triggered during model execution (`MessageFactory.GetPrototype`), not in your PyTorch code; the minimal fix is to pin protobuf’s implementation and version *before* importing `transformers`. I add a safe protobuf import/patch and (if available) a small on-the-fly downgrade to a compatible protobuf version in the Kaggle runtime so inference can run. I also make `qa_id` collection robust (strings/object dtype shouldn’t be `.cpu()`’d) and ensure the submission rows/columns align exactly to `sample_submission.csv`. These changes are score-positive because they allow the intended checkpoint/model weights to actually run instead of crashing.'
- What this solution (achieved 0.04729) has done: 'Your score is far below the target (0.04729 vs 0.3649), which strongly suggests the intended trained weights are not being applied correctly (either the checkpoint is incompatible with the instantiated base model, or key mismatches leave most layers randomly initialized). I make the smallest changes that keep your exact architecture/inference logic, but (1) ensure we load the *same backbone* the checkpoint expects (by auto-detecting the checkpoint’s `transformers` config name/path if present), and (2) load weights more strictly by fixing common Lightning/HF key prefix patterns and failing over to the best-available matching strategy. This should move predictions from near-random toward the checkpoint’s learned behavior, improving Spearman correlation substantially toward your target. I also keep the submission alignment/column order exactly as `sample_submission.csv`.'
- What this solution (achieved 0.04729) has done: 'Your current score is far below the target, which most likely means the checkpoint weights are not being applied to the model you instantiate (backbone mismatch and/or state_dict key mismatches), leaving most parameters effectively random. I make minimal changes to (1) infer and load the correct backbone/config from the checkpoint (without requiring internet), and (2) load the checkpoint weights more completely by handling common Lightning/HF key prefixes and alternative key layouts. I also make inference deterministic-correct by forcing `model.eval()`-consistent dropout behavior (already eval) and ensuring the tokenizer/model load use the same source (local-only when available). These changes keep your architecture and sigmoid post-processing identical, but should move predictions from near-random toward the trained checkpoint behavior, increasing Spearman toward your target band.'
- What this solution (achieved 0.04729) has done: 'Your score is far below the target, so the most likely issue is that the checkpoint’s weights are not actually being loaded into the instantiated model (backbone mismatch and/or key mismatches), leaving many layers effectively random. I make minimal changes to (1) infer the correct backbone/config from the checkpoint and instantiate the transformer from that (local-only), and (2) load the checkpoint more completely by handling common HF/Lightning key patterns, including cases where the transformer lives under `transformer.transformer.*` or similar nested prefixes. I also add a small sanity check that reports the fraction of parameters loaded and only falls back to pretrained initialization when loading coverage is clearly poor. This preserves your model architecture, tokenization, and sigmoid post-processing, but should move predictions from near-random toward the intended trained behavior and increase Spearman toward your target.'
- What this solution (achieved 0.04729) has done: 'Your score is far below the target (0.047 vs 0.365), which is consistent with the checkpoint not being loaded into the exact same architecture/backbone it was trained with (so many weights stay random or mis-mapped). I keep your model architecture and inference identical, but change checkpoint loading to (1) instantiate a pretrained backbone first, then (2) load the checkpoint weights onto it after normalizing common key-prefix patterns; this avoids accidentally building a random-from-config transformer when the checkpoint doesn’t fully match. I also expand key normalization for the most common Roberta/BERT naming patterns so more of the checkpoint lands on the right submodules, increasing correlation while preserving semantics. Finally, I keep the submission formatting/alignment unchanged.'
- What this solution (achieved 0.04729) has done: 'Your gap to target is large (0.04729 vs 0.3649, higher-is-better), which strongly indicates the checkpoint isn’t being applied to the right backbone/config and many weights remain effectively random. I keep your exact model architecture and inference flow, but make checkpoint loading backbone-aware by instantiating the transformer from the checkpoint’s saved HuggingFace `config` when available (so dimensions like hidden size/layer count match), then loading the state dict with your existing key normalization. I also ensure tokenizer/model are consistent (local-only when using a local directory), and add a light, score-positive safety check to avoid silently using a mismatched backbone when checkpoint coverage is very low. These changes should move predictions from near-random toward the intended trained checkpoint behavior and increase Spearman toward your target band without altering evaluation semantics.'
- What this solution (achieved 0.04729) has done: 'Your current score is far below the target (0.04729 vs 0.3649), so the most likely issue is that the checkpoint backbone/config is not being correctly reconstructed and only a small fraction of weights are actually loading, producing near-random predictions. I keep your exact dataset, model architecture, and sigmoid post-processing, but make checkpoint loading more “HF/Lightning-compatible” by (1) reading and using the checkpoint’s embedded HuggingFace `config` *and* `tokenizer_config` when available, and (2) expanding state-dict key normalization to cover common `transformer.embeddings.*`/`transformer.encoder.*`/`roberta.*` nesting patterns so more weights land correctly. I also add a strict, score-positive sanity check that prints loaded-parameter ratio by tensor count and by matched tensor elements; if coverage is still extremely low, we fall back exactly as you already do (pretrained backbone + partial load), keeping behavior safe. These changes are minimal and directly aimed at moving predictions from near-random toward the intended trained checkpoint behavior (higher Spearman).'
- What this solution (achieved 0.04729) has done: 'Your score (0.04729) is far below the target (0.3649), which strongly suggests the checkpoint is not actually being applied (or is being applied to the wrong backbone), yielding near-random predictions. I make the smallest changes that improve *checkpoint/backbone/tokenizer consistency* without altering your architecture or inference semantics: (1) infer the backbone directory from the checkpoint and prefer it over the hardcoded `MODEL_NAME`, (2) instantiate the transformer from that inferred backbone (pretrained) and only use the checkpoint’s HF `config` when it clearly matches the backbone, and (3) improve state_dict key normalization for common `transformer.`/`base_model.` nesting so more weights load. These changes should materially increase Spearman correlation by ensuring most learned weights are used, while keeping the same model forward and sigmoid post-processing. The submission writing/alignment stays identical and still produces a valid `submission.csv`.'
- What this solution (achieved 0.04729) has done: 'Your current score is far below the target (0.04729 vs 0.3649), which is consistent with inference running on mostly untrained/random weights because the checkpoint is not actually being loaded into the same module key layout it was saved with. I make a minimal, score-positive change to the checkpoint key normalization so it correctly maps common Lightning patterns like `transformer.*` → `transformer.transformer.*` (and related nested prefixes) used by many Kaggle QUEST checkpoints, increasing the fraction of backbone weights that load. I also add a tiny “best-of-two mappings” load attempt (same architecture) to automatically pick the mapping that yields higher parameter coverage, without changing tokenization, model forward, or post-processing. The submission writing/ordering stays identical.'
- What this solution (achieved 0.04729) has done: 'Your score gap is large (0.04729 vs target 0.3649, higher-is-better), which strongly suggests the checkpoint weights are not being applied cleanly and you’re effectively running close to a pretrained/random head. I keep your exact dataset/model/forward/sigmoid logic, but make checkpoint loading more reliable by (1) instantiating the backbone from the checkpoint’s embedded HF config when present (so layer count/hidden size match), and (2) improving state_dict key normalization to also handle `transformer.` vs `transformer.transformer.` layouts, and `*_head.*` vs `question_head/answer_head/combined_head` naming patterns. I also prevent “trying variants on the same model instance” from contaminating coverage selection by using a fresh model per variant, which should materially increase loaded-weight coverage and move Spearman toward your target. Submission formatting and column order remain identical.'
- What this solution (achieved 0.04729) has done: 'Your score gap to the target is large (0.04729 → 0.3649), which strongly suggests the checkpoint you’re loading is not actually compatible with the instantiated backbone (so most weights don’t load and predictions are near-random). I make minimal, directly score-relevant changes to (1) ensure the backbone/tokenizer come from the same local directory as the checkpoint expects by auto-searching `/kaggle/input` for a `config.json` and using that as the backbone, and (2) improve checkpoint state_dict loading by supporting an additional common key layout: `transformer.transformer.*` in the checkpoint mapping onto `transformer.*` in your model (and vice versa), selecting the mapping with best true element coverage. These changes preserve your model architecture/forward/sigmoid and only aim to make the intended trained weights actually load. Submission formatting/alignment stays identical and still writes a valid `submission.csv`.'
- What this solution (achieved 0.04729) has done: 'Your score is far below the target (0.04729 vs 0.3649), so the most likely issue is that you are not actually running the intended trained checkpoint backbone/head weights cleanly (partial/mismatched loading yields near-random ranks and low Spearman). I keep your exact architecture/forward/sigmoid and only make checkpoint/backbone resolution and state_dict mapping stricter and more complete: (1) prefer a backbone directory adjacent to the checkpoint file when present, (2) extend key normalization to handle common Lightning/HF layouts like `transformer.encoder.*`, `transformer.embeddings.*`, and `transformer.layer_weights`, and (3) select the best mapping variant by true element coverage, then verify coverage is not catastrophically low before proceeding. These changes are minimal and directly aimed at moving predictions from “mostly random weights” toward “checkpoint-loaded weights”, which should increase mean Spearman toward your target band while preserving evaluation semantics. The submission writing stays identical and still produces a valid `submission.csv`.'
- What this solution (achieved 0.04729) has done: 'Your current score (0.04729) is far below the target (0.3649), so the smallest likely score-positive fix is to ensure the checkpoint is loaded into a *matching backbone/config* instead of silently building a random-from-config transformer when shapes don’t line up. I keep your architecture and inference identical, but change checkpoint loading to always start from `from_pretrained(model_name)` (when possible) and then load the normalized checkpoint weights on top, which typically raises Spearman a lot when partial loads were leaving many tensors effectively uninitialized. I also make the “best variant” selection use element-coverage of the *actually loaded* tensors (intersection of keys with correct shapes), rather than coverage computed against the pre-load dict, so the chosen mapping is more reliable. Finally, I keep the exact same submission formatting/alignment and still write `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import google.protobuf  # noqa: F401
    from importlib.metadata import version as _pkg_version

    _pb_ver = _pkg_version("protobuf")
    if _pb_ver and _pb_ver.split(".")[0].isdigit() and int(_pb_ver.split(".")[0]) >= 5:
        import subprocess, sys, importlib

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        importlib.invalidate_caches()
except Exception:
    pass

import html
import re
import random
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from scipy.stats import spearmanr
from tqdm.auto import tqdm

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")



## === cell 1
DATA_DIR = "/kaggle/input/google-quest-challenge"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_FILE = os.path.join(DATA_DIR, "sample_submission.csv")

MODEL_NAME = "/kaggle/input/deberta-v3-large-fold3-val-spearman0-4392/pytorch/default/3/roberta-large/roberta-large"
FALLBACK_MODEL_NAME = "roberta-large"

MAX_LENGTH = 512
BATCH_SIZE = 8
SEED = 42

CHECKPOINT_PATH = "/kaggle/input/deberta-v3-large-fold3-val-spearman0-4392/pytorch/default/3/roberta_large_fold2_val_spearman0.4418.ckpt"

QUESTION_TARGET_COLS = [
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
]

ANSWER_TARGET_COLS = [
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]

TARGET_COLS = QUESTION_TARGET_COLS + ANSWER_TARGET_COLS
NUM_QUESTION_TARGETS = len(QUESTION_TARGET_COLS)
NUM_ANSWER_TARGETS = len(ANSWER_TARGET_COLS)
NUM_TARGETS = len(TARGET_COLS)

DROPOUT_RATES = [0.1, 0.15, 0.2, 0.25, 0.3]

print(f"Number of targets: {NUM_TARGETS}")
print(f"Question targets: {NUM_QUESTION_TARGETS}")
print(f"Answer targets: {NUM_ANSWER_TARGETS}")
print(f"Checkpoint: {CHECKPOINT_PATH}")




## === cell 2
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_spearman(predictions: np.ndarray, targets: np.ndarray) -> float:
    scores = []
    for i in range(predictions.shape[1]):
        score, _ = spearmanr(predictions[:, i], targets[:, i])
        if not np.isnan(score):
            scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def resolve_model_path(path_or_id: str, fallback_id: str) -> str:
    """If it's a local directory that exists, use it; otherwise use a safe fallback HF id."""
    if os.path.isdir(path_or_id):
        return path_or_id
    return fallback_id


def infer_model_name_from_checkpoint(checkpoint: dict) -> Optional[str]:
    candidates = []
    hparams = checkpoint.get("hyper_parameters", None)
    if isinstance(hparams, dict):
        for k in (
            "model_name",
            "model_name_or_path",
            "pretrained_model_name_or_path",
            "backbone",
            "encoder",
        ):
            v = hparams.get(k)
            if isinstance(v, str) and v:
                candidates.append(v)

    cfg = checkpoint.get("config", None)
    if isinstance(cfg, dict):
        for k in (
            "_name_or_path",
            "model_name_or_path",
            "pretrained_model_name_or_path",
        ):
            v = cfg.get(k)
            if isinstance(v, str) and v:
                candidates.append(v)

    for c in candidates:
        c = str(c).strip()
        if c:
            return c
    return None


def infer_local_backbone_dir_from_checkpoint(checkpoint: dict) -> Optional[str]:
    """
    If the checkpoint points to a local backbone dir, using that exact dir ensures tokenizer/config/weights match.
    """
    inferred = infer_model_name_from_checkpoint(checkpoint)
    if isinstance(inferred, str) and os.path.isdir(inferred):
        return inferred
    return None


def infer_backbone_dir_from_checkpoint_path(checkpoint_path: str) -> Optional[str]:
    """
    Change rationale (score-positive, minimal): many Kaggle model datasets store the HF backbone
    directory next to the .ckpt. Prefer this local dir to avoid backbone mismatch.
    """
    if not checkpoint_path:
        return None
    ckpt_dir = os.path.dirname(checkpoint_path)
    if not os.path.isdir(ckpt_dir):
        return None

    if os.path.exists(os.path.join(ckpt_dir, "config.json")):
        return ckpt_dir

    for dirpath, dirnames, filenames in os.walk(ckpt_dir):
        if "config.json" in filenames:
            return dirpath
    return None


def extract_hf_config_from_checkpoint(checkpoint: dict) -> Optional[Any]:
    cfg = checkpoint.get("config", None)
    if not isinstance(cfg, dict) or len(cfg) == 0:
        return None
    try:
        config = AutoConfig.from_dict(cfg)
        config.output_hidden_states = True
        return config
    except Exception:
        return None


def extract_tokenizer_kwargs_from_checkpoint(checkpoint: dict) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    tok_cfg = checkpoint.get("tokenizer_config", None)
    if isinstance(tok_cfg, dict):
        for k in ("add_prefix_space", "do_lower_case", "use_fast", "model_max_length"):
            if k in tok_cfg:
                out[k] = tok_cfg[k]
    return out


def find_best_local_backbone_dir(
    prefer_roberta: bool = True,
    roots: Tuple[str, ...] = ("/kaggle/input",),
) -> Optional[str]:
    candidates: List[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            fn = set(filenames)
            if "config.json" in fn and (
                "pytorch_model.bin" in fn or "model.safetensors" in fn
            ):
                candidates.append(dirpath)

    if not candidates:
        return None

    def score(p: str) -> Tuple[int, int, int]:
        lp = p.lower()
        s1 = 1 if (prefer_roberta and ("roberta" in lp)) else 0
        s2 = 1 if ("deberta" in lp or "bert" in lp) else 0
        s3 = len(p)
        return (s1, s2, s3)

    candidates.sort(key=score, reverse=True)
    return candidates[0]


set_seed(SEED)




## === cell 3
class QuestDataset(Dataset):
    """
    Dataset for Google QUEST Q&A Labeling.
    Two inputs for Siamese architecture:
    - Question input: (title, body)
    - Answer input: (title+body, answer)
    """

    def __init__(
        self, df: pd.DataFrame, tokenizer, max_length: int = 512, is_test: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        self.question_titles = [clean_text(t) for t in self.df["question_title"].values]
        self.question_bodies = [clean_text(t) for t in self.df["question_body"].values]
        self.answers = [clean_text(t) for t in self.df["answer"].values]

        if not is_test:
            self.targets = self.df[TARGET_COLS].values.astype(np.float32)

        self.qa_ids = self.df["qa_id"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        title = self.question_titles[idx]
        body = self.question_bodies[idx]
        answer = self.answers[idx]

        q_encoding = self.tokenizer(
            title,
            body,
            max_length=self.max_length,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt",
        )

        question_text = f"{title} {body}"
        a_encoding = self.tokenizer(
            question_text,
            answer,
            max_length=self.max_length,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt",
        )

        item = {
            "q_input_ids": q_encoding["input_ids"].squeeze(0),
            "q_attention_mask": q_encoding["attention_mask"].squeeze(0),
            "a_input_ids": a_encoding["input_ids"].squeeze(0),
            "a_attention_mask": a_encoding["attention_mask"].squeeze(0),
            "qa_id": self.qa_ids[idx],
        }

        if not self.is_test:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item




## === cell 4
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        weights = self.attention(hidden_states).squeeze(-1)
        weights = weights.masked_fill(~attention_mask.bool(), float("-inf"))
        weights = F.softmax(weights, dim=1)
        pooled = (hidden_states * weights.unsqueeze(-1)).sum(dim=1)
        return pooled


class MultiSampleDropout(nn.Module):
    def __init__(self, dropout_rates: List[float] = None):
        super().__init__()
        if dropout_rates is None:
            dropout_rates = DROPOUT_RATES
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])

    def forward(self, x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        outputs = torch.stack([linear(dropout(x)) for dropout in self.dropouts])
        return outputs.mean(dim=0)


class SiameseQuestModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        config_override: Optional[Any] = None,
    ):
        super().__init__()

        if pretrained:
            self.transformer = AutoModel.from_pretrained(
                model_name,
                output_hidden_states=True,
                local_files_only=os.path.isdir(model_name),
            )
        else:
            if config_override is not None:
                config = config_override
                try:
                    config.output_hidden_states = True
                except Exception:
                    pass
            else:
                config = AutoConfig.from_pretrained(
                    model_name, local_files_only=os.path.isdir(model_name)
                )
                config.output_hidden_states = True
            self.transformer = AutoModel.from_config(config)

        hidden_size = self.transformer.config.hidden_size
        num_layers = self.transformer.config.num_hidden_layers

        self.layer_weights = nn.Parameter(torch.ones(num_layers + 1))

        self.q_attention = AttentionPooling(hidden_size)
        self.a_attention = AttentionPooling(hidden_size)

        self.multi_dropout = MultiSampleDropout()

        self.question_head = nn.Linear(hidden_size, NUM_QUESTION_TARGETS)
        self.answer_head = nn.Linear(hidden_size, NUM_ANSWER_TARGETS)
        self.combined_head = nn.Linear(hidden_size * 2, NUM_TARGETS)

    def weighted_layer_pooling(
        self, hidden_states: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        stacked = torch.stack(hidden_states, dim=0)  # (layers, batch, seq, hidden)
        weights = F.softmax(self.layer_weights, dim=0)
        weighted = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
        return weighted

    def encode_branch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        attention_pooling: AttentionPooling,
    ) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.weighted_layer_pooling(outputs.hidden_states)
        pooled = attention_pooling(hidden, attention_mask)
        return pooled

    def forward(
        self,
        q_input_ids: torch.Tensor,
        q_attention_mask: torch.Tensor,
        a_input_ids: torch.Tensor,
        a_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        q_pooled = self.encode_branch(q_input_ids, q_attention_mask, self.q_attention)
        a_pooled = self.encode_branch(a_input_ids, a_attention_mask, self.a_attention)

        q_preds = self.multi_dropout(q_pooled, self.question_head)
        a_preds = self.multi_dropout(a_pooled, self.answer_head)

        combined = torch.cat([q_pooled, a_pooled], dim=-1)
        combined_preds = self.multi_dropout(combined, self.combined_head)

        specialized_preds = torch.cat([q_preds, a_preds], dim=-1)
        final_logits = specialized_preds * 0.5 + combined_preds * 0.5
        return final_logits




## === cell 5
def _strip_prefix_from_state_dict(
    state_dict: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    if not prefix:
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def _normalize_heads_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        nk = k
        nk = nk.replace("q_head.", "question_head.")
        nk = nk.replace("a_head.", "answer_head.")
        nk = nk.replace("question_classifier.", "question_head.")
        nk = nk.replace("answer_classifier.", "answer_head.")
        nk = nk.replace("combined_classifier.", "combined_head.")
        nk = nk.replace("qa_classifier.", "combined_head.")
        nk = nk.replace("classifier_q.", "question_head.")
        nk = nk.replace("classifier_a.", "answer_head.")
        nk = nk.replace("classifier_qa.", "combined_head.")
        if nk.startswith("classifier."):
            nk = "combined_head." + nk[len("classifier.") :]
        out[nk] = v
    return out


def _normalize_state_dict_keys_variant(
    state_dict: Dict[str, torch.Tensor], variant: int
) -> Dict[str, torch.Tensor]:
    """
    Change rationale (score-positive, minimal): extend mapping for common Lightning/HF key layouts so
    more checkpoint tensors (especially the transformer backbone + layer_weights) load with correct shapes.
    """
    sd = state_dict
    for p in (
        "model.",
        "net.",
        "module.",
        "siamese.",
        "pl_module.",
        "student.",
        "teacher.",
    ):
        sd = _strip_prefix_from_state_dict(sd, p)

    sd = _normalize_heads_keys(sd)

    remapped: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k

        if nk in ("layer_weights", "lw", "weights_layer"):
            nk = "layer_weights"

        for pfx in (
            "encoder.",
            "backbone.",
            "transformer_model.",
            "lm.",
            "language_model.",
        ):
            if nk.startswith(pfx):
                nk = "transformer." + nk[len(pfx) :]

        if nk.startswith(("embeddings.", "encoder.", "pooler.")):
            nk = "transformer." + nk

        if nk.startswith(
            ("roberta.", "bert.", "deberta.", "xlnet.", "albert.", "electra.")
        ):
            nk = "transformer." + nk

        for pfx in (
            "transformer.transformer.",
            "transformer.model.",
            "transformer.base_model.",
            "transformer.base_model.model.",
            "transformer.roberta.",
            "transformer.bert.",
            "transformer.deberta.",
        ):
            if nk.startswith(pfx):
                nk = "transformer." + nk[len(pfx) :]

        if variant == 1:
            if nk.startswith("transformer.") and not nk.startswith(
                "transformer.transformer."
            ):
                sub = nk[len("transformer.") :]
                if sub.startswith(
                    ("embeddings.", "encoder.", "pooler.", "layer.", "wte.", "wpe.")
                ):
                    nk = "transformer.transformer." + sub
        elif variant == 2:
            if nk.startswith("transformer.transformer."):
                nk = "transformer." + nk[len("transformer.transformer.") :]
        elif variant == 3:
            if nk.startswith("transformer.transformer."):
                nk = "transformer." + nk[len("transformer.transformer.") :]
            if nk.startswith("base_model."):
                nk = "transformer." + nk[len("base_model.") :]

        remapped[nk] = v
    return remapped


def _extract_state_dict_from_checkpoint_obj(ckpt_obj: dict) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        if isinstance(ckpt_obj.get("state_dict", None), dict):
            return ckpt_obj["state_dict"]
        for k in ("model_state_dict", "model", "weights"):
            v = ckpt_obj.get(k, None)
            if isinstance(v, dict) and all(isinstance(x, str) for x in v.keys()):
                return v
    return ckpt_obj


def _tensor_element_coverage(
    model: nn.Module, candidate_state_dict: Dict[str, torch.Tensor]
) -> float:
    """
    Change rationale (score-positive, minimal): choose the best key-mapping variant using the *true*
    overlap (keys with matching shapes) between model tensors and the candidate sd.
    """
    msd = model.state_dict()
    total_elems = 0
    loaded_elems = 0
    for k, t in msd.items():
        if not torch.is_tensor(t):
            continue
        n = t.numel()
        total_elems += n
        v = candidate_state_dict.get(k, None)
        if torch.is_tensor(v) and v.shape == t.shape:
            loaded_elems += n
    if total_elems == 0:
        return 0.0
    return float(loaded_elems / total_elems)


def _make_model_for_loading(
    model_name: str, ckpt_config: Optional[Any], prefer_pretrained_init: bool
) -> nn.Module:
    """
    Change rationale (score-positive, minimal): when checkpoint loading is partial, starting from
    from_pretrained() prevents large parts of the transformer from being random-initialized.
    """
    if prefer_pretrained_init:
        return SiameseQuestModel(model_name, pretrained=True)
    if ckpt_config is not None:
        try:
            return SiameseQuestModel(
                model_name, pretrained=False, config_override=ckpt_config
            )
        except Exception:
            pass
    return SiameseQuestModel(model_name, pretrained=True)


def load_model_from_checkpoint(
    checkpoint_path: str, model_name: str, device: torch.device
) -> nn.Module:
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        adj_backbone = infer_backbone_dir_from_checkpoint_path(checkpoint_path)
        if adj_backbone and os.path.isdir(adj_backbone):
            print(f"Using backbone dir inferred from checkpoint path: {adj_backbone}")
            model_name = adj_backbone

        inferred_local = infer_local_backbone_dir_from_checkpoint(checkpoint)
        if inferred_local and inferred_local != model_name:
            print(
                f"Using inferred local backbone dir from checkpoint metadata: {inferred_local}"
            )
            model_name = inferred_local

        ckpt_config = extract_hf_config_from_checkpoint(checkpoint)
        state_dict_raw = _extract_state_dict_from_checkpoint_obj(checkpoint)

        best_sd = None
        best_cov = -1.0
        best_variant = None
        best_pretrained_init = True

        for prefer_pretrained_init in (True, False):
            for variant in (0, 1, 2, 3):
                tmp_model = _make_model_for_loading(
                    model_name,
                    ckpt_config,
                    prefer_pretrained_init=prefer_pretrained_init,
                )
                sd_norm = _normalize_state_dict_keys_variant(
                    state_dict_raw, variant=variant
                )

                tmp_model.load_state_dict(sd_norm, strict=False)
                cov = _tensor_element_coverage(tmp_model, sd_norm)

                print(
                    f"pretrained_init={prefer_pretrained_init} variant={variant}: elem-coverage={cov:.3f}"
                )
                if cov > best_cov:
                    best_cov = cov
                    best_sd = sd_norm
                    best_variant = variant
                    best_pretrained_init = prefer_pretrained_init

        print(
            f"Best choice: pretrained_init={best_pretrained_init} variant={best_variant} elem-coverage={best_cov:.3f}"
        )

        if best_cov < 0.20:
            print(
                "Warning: very low checkpoint load coverage; falling back to pretrained backbone init."
            )
            model = SiameseQuestModel(model_name, pretrained=True)
        else:
            model = _make_model_for_loading(
                model_name, ckpt_config, prefer_pretrained_init=best_pretrained_init
            )
            missing, unexpected = model.load_state_dict(best_sd, strict=False)
            print(
                f"Final load: missing={len(missing)} unexpected={len(unexpected)} elem-coverage={best_cov:.3f}"
            )
    else:
        print(f"Checkpoint not found at: {checkpoint_path}")
        print(
            "Falling back to loading model weights from MODEL_NAME (pretrained=True)."
        )
        model = SiameseQuestModel(model_name, pretrained=True)

    model = model.to(device)
    model.eval()
    return model




## === cell 6
print("Loading data...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
sample_sub = pd.read_csv(SAMPLE_SUB_FILE)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Sample submission shape: {sample_sub.shape}")

SAMPLE_TARGET_COLS = [c for c in sample_sub.columns if c != "qa_id"]
if set(SAMPLE_TARGET_COLS) != set(TARGET_COLS):
    print(
        "Warning: TARGET_COLS != sample_submission target columns. Using sample_submission columns."
    )
    TARGET_COLS = SAMPLE_TARGET_COLS
    NUM_TARGETS = len(TARGET_COLS)
    NUM_QUESTION_TARGETS = len([c for c in TARGET_COLS if c.startswith("question_")])
    NUM_ANSWER_TARGETS = len([c for c in TARGET_COLS if c.startswith("answer_")])



## === cell 7
resolved_model_name = resolve_model_path(MODEL_NAME, FALLBACK_MODEL_NAME)

adj_backbone = (
    infer_backbone_dir_from_checkpoint_path(CHECKPOINT_PATH)
    if CHECKPOINT_PATH
    else None
)
if adj_backbone is not None and os.path.isdir(adj_backbone):
    print(f"Auto-using backbone dir from checkpoint path: {adj_backbone}")
    resolved_model_name = adj_backbone

if not os.path.isdir(resolved_model_name):
    auto_backbone = find_best_local_backbone_dir(
        prefer_roberta=True, roots=("/kaggle/input",)
    )
    if auto_backbone is not None:
        print(f"Auto-detected local backbone dir: {auto_backbone}")
        resolved_model_name = auto_backbone

if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
    try:
        ckpt_tmp = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        inferred_local = infer_local_backbone_dir_from_checkpoint(ckpt_tmp)
        if inferred_local:
            resolved_model_name = inferred_local
            print(
                f"Overriding MODEL_NAME with checkpoint-inferred local dir: {resolved_model_name}"
            )
    except Exception:
        pass

print(f"Requested MODEL_NAME: {MODEL_NAME}")
print(f"Resolved model name/path: {resolved_model_name}")

local_only = os.path.isdir(resolved_model_name)
print(f"Local files only: {local_only}")

tok_kwargs = {}
if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
    try:
        ckpt_tmp = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        tok_kwargs = extract_tokenizer_kwargs_from_checkpoint(ckpt_tmp)
        if "use_fast" not in tok_kwargs:
            tok_kwargs["use_fast"] = True
    except Exception:
        tok_kwargs = {"use_fast": True}
else:
    tok_kwargs = {"use_fast": True}

print(f"Loading tokenizer: {resolved_model_name} with kwargs={tok_kwargs}")
tokenizer = AutoTokenizer.from_pretrained(
    resolved_model_name, local_files_only=local_only, **tok_kwargs
)
print(f"Tokenizer loaded. Vocab size: {getattr(tokenizer, 'vocab_size', 'NA')}")



## === cell 8
test_dataset = QuestDataset(
    df=test_df, tokenizer=tokenizer, max_length=MAX_LENGTH, is_test=True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

print(f"Test dataset size: {len(test_dataset)}")
print(f"Test batches: {len(test_dataloader)}")



## === cell 9
exists = os.path.exists(CHECKPOINT_PATH)
status = "OK" if exists else "NOT FOUND"
print(f"Checkpoint: {CHECKPOINT_PATH} [{status}]")



## === cell 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = load_model_from_checkpoint(CHECKPOINT_PATH, resolved_model_name, device)

all_predictions = []
all_qa_ids = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Predicting"):
        q_input_ids = batch["q_input_ids"].to(device)
        q_attention_mask = batch["q_attention_mask"].to(device)
        a_input_ids = batch["a_input_ids"].to(device)
        a_attention_mask = batch["a_attention_mask"].to(device)

        logits = model(
            q_input_ids=q_input_ids,
            q_attention_mask=q_attention_mask,
            a_input_ids=a_input_ids,
            a_attention_mask=a_attention_mask,
        )

        preds = torch.sigmoid(logits)
        all_predictions.append(preds.cpu().numpy())

        qa_id_batch = batch["qa_id"]
        if torch.is_tensor(qa_id_batch):
            qa_id_batch = qa_id_batch.cpu().numpy()
        else:
            qa_id_batch = np.asarray(qa_id_batch)
        all_qa_ids.append(qa_id_batch)

predictions = np.concatenate(all_predictions, axis=0)
all_qa_ids = np.concatenate(all_qa_ids, axis=0)

print(f"Predictions shape: {predictions.shape}")
print(f"Num test rows: {len(test_df)}; num predictions: {predictions.shape[0]}")



## === cell 11
submission = pd.DataFrame(predictions, columns=TARGET_COLS)
submission.insert(0, "qa_id", pd.Series(all_qa_ids).astype(test_df["qa_id"].dtype))

for col in TARGET_COLS:
    submission[col] = submission[col].clip(0, 1)

submission = submission.set_index("qa_id").loc[test_df["qa_id"].values].reset_index()
submission = submission[sample_sub.columns]

print(f"Submission shape: {submission.shape}")
out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print(f"{out_path} created!")
print(submission.head())
