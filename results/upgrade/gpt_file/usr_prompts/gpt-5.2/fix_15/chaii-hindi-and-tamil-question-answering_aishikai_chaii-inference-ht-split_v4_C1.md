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
Predicting the answers to questions in Hindi and Tamil.

## Metric
Word-level Jaccard score.

A Python implementation is provided below.

```
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
```

The formula for the overall metric is:
\text{score} = \frac{1}{n} \sum_{i=1}^n \text{jaccard}(gt_i, dt_i)

where:
$n$ = number of documents

$\text{jaccard}$ = the function provided above

$gt_i$ = the ith ground truth

$dt_i$ = the ith prediction

## Submission Format
For each ID in the test set, you must predict the string that best answers the provided question based on the context. Note that the selected text needs to be quoted and complete to work correctly. Include punctuation, etc. The file should contain a header and have the following format:

```
id,PredictionString
8c8ee6504,"1"
3163c22d0,"2 string"
66aae423b,"4 word 6"
722085a7b,"1"
etc.
```

## Dataset 
**All files should be encoded as UTF-8.**

- **train.csv** - the training set, containing context, questions, and answers. Also includes the start character of the answer for disambiguation.
- **test.csv** - the test set, containing context and questions.
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - a unique identifier
- `context` - the text of the Hindi/Tamil sample from which answers should be derived
- `question` - the question, in Hindi/Tamil
- `answer_text` (train only) - the answer to the question (manual annotation) (note: for test, this is what you are attempting to predict)
- `answer_start` (train only) - the starting character in `context` for the answer (determined using substring match during data preparation)
- `language` - whether the text in question is in Tamil or Hindi

# 2. Python version

3.10

# 3. Installed packages

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
            description.md (138 lines)
            sample_submission.csv (113 lines)
            sample_submission.csv.zip (950 Bytes)
            test.csv (7173 lines)
            test.csv.zip (648.4 kB)
            train.csv (67723 lines)
            train.csv.zip (5.9 MB)
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
        input/
            description.md (138 lines)
            sample_submission.csv (113 lines)
            sample_submission.csv.zip (950 Bytes)
            test.csv (7173 lines)
            test.csv.zip (648.4 kB)
            train.csv (67723 lines)
            train.csv.zip (5.9 MB)
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
        working/
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
```

-> data/chaii-hindi-and-tamil-question-answering/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> data/chaii-hindi-and-tamil-question-answering/test.csv has 7172 rows and 4 columns.
The columns are: id, context, question, language

-> data/chaii-hindi-and-tamil-question-answering/train.csv has 67722 rows and 6 columns.
The columns are: id, context, question, answer_text, answer_start, language

-> data/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> data/test.csv has 7172 rows and 4 columns.
The columns are: id, context, question, language

-> data/train.csv has 67722 rows and 6 columns.
The columns are: id, context, question, answer_text, answer_start, language

-> input/chaii-hindi-and-tamil-question-answering/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> (stopped after 10 files for performance)

# 5. Target score

0.7131982445716858

# 6. Current score

0.03214

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.02477) has done: 'I fix the import errors caused by recent Transformers changes (AdamW moved), and make sure `AutoTokenizer/AutoConfig` are always imported so later cells can run. I also fix runtime failures due to missing Kaggle input model paths by adding a safe fallback: if the provided checkpoints aren’t available, the code still generate a valid `submission.csv` using a simple context-based baseline (score be lower, but it run end-to-end). I make `Postprocess_qa_predictions` take the tokenizer as an argument (it currently relies on a global that may not exist), and correct the Hindi/Tamil ensembling logic bug that indexed wrong arrays. Finally, I ensure the submission file is written with the correct columns and `.csv` suffix.'
- What this solution (achieved 0.03288) has done: 'I fix the immediate runtime error caused by invalid/nonexistent local Hugging Face model paths by adding a robust “offline model resolver” that searches common Kaggle input locations and falls back to a lightweight baseline when no pretrained checkpoint is available. This ensures the notebook always runs end-to-end and always writes a valid `submission.csv` with the required columns. I also prevent cascading `NameError`s by guaranteeing `fin_preds` and `test_data` are always defined, even in fallback mode. These changes are execution/stability focused; they preserve your QA postprocessing and model logic when a valid local model directory exists.'
- What this solution (achieved 0.03214) has done: 'Your current score suggests the pipeline is mostly falling back to the weak “first 80 chars of context” baseline because it can’t find the pretrained checkpoint folders you reference, and/or it’s loading the wrong base model directory. I make the model/ckpt resolution search the actual competition input tree and, if found, run inference from those checkpoints (same model architecture and postprocessing), which should move the score substantially toward your 0.713 target. I also fix a subtle bug in `Postprocess_qa_predictions` where `offset_mapping` is converted to tensors in the test dataset path in a way that can break indexing/None handling; we keep offset mappings as plain python lists for test features only. Finally, I keep submission formatting identical but stop stripping punctuation (Jaccard is token-based; stripping can remove useful tokens and hurt score).'
- What this solution (achieved 0.03214) has done: 'Your current score (0.03214) is far below target (0.7132), and the biggest cause is that the notebook is almost certainly still running in fallback mode because it can’t locate the MURIL tokenizer/model directory and/or the helper fold checkpoints. I make the model/ckpt resolution actually search the available `/kaggle/input/...` trees (and your provided `/kaggle/data/...` mirror) instead of only hardcoded `../input/...` paths, so the intended inference path runs. I also fix the ensembling feed into `Postprocess_qa_predictions` (it expects 2D arrays aligned to `features`, but the current code passes Python lists), which can silently degrade predictions even when checkpoints are found. These changes keep your core model, feature creation, and post-processing logic intact, but should move the score substantially upward toward the target by ensuring the real checkpoints are used and the logits are correctly shaped.'
- What this solution (achieved 0.03214) has done: 'Your score (0.03214) is far below the target (0.7132), so the priority is to ensure you are not silently using the weak fallback baseline and that you are actually loading a compatible MuRIL QA checkpoint for inference. I keep your model/postprocess logic intact, but (1) resolve and load the correct MuRIL tokenizer/model directory reliably, (2) fix checkpoint loading to handle common saved formats (full `state_dict`, nested `state_dict`, or a HF folder) and avoid shape-mismatch by loading with `strict=False`, and (3) fix language routing to use the per-example language from `test_df` (not per-feature, which duplicates and can misroute logits when examples create multiple features). These are minimal execution/compatibility fixes aimed at moving you toward the target by making the intended checkpoint inference actually work. The submission writing stays identical and still produces a valid `submission.csv`.'
- What this solution (achieved 0.03214) has done: 'Your score is far below target, so the smallest change likely to move it upward is to stop silently using incompatible “helper” `pytorch_model.bin` files as if they were a `state_dict` for your custom `Model` (this usually loads mostly-missing weights with `strict=False`, producing near-random logits and very low Jaccard). I minimally change checkpoint discovery/loading so that we only use checkpoints that actually match the current model (prefer `.pth/.pt` state_dict-style files; otherwise skip), and when we do use helper bins we load them as full Hugging Face QA models (`AutoModelForQuestionAnswering`) and use their native logits (same postprocess). If no compatible checkpoints are found, the script fall back to “fine-tune then infer” (already in your code) rather than producing bad random predictions, which should raise the score substantially toward your target while keeping the overall approach intact. Submission formatting and paths stay the same, and the code still runs end-to-end within constraints.'
- What this solution (achieved 0.03214) has done: 'Your current score is far below target, so the smallest likely win is to stop running inference with an incompatible head: you currently load helper checkpoints as full HF QA models (correct), but your `test_features` are built with a potentially different tokenizer (MuRIL found elsewhere), which makes offset mappings/token indices mismatched and yields near-random spans. I minimally rebuild the test features per-language using the exact tokenizer that corresponds to the Hindi and Tamil checkpoint directories, and then run postprocessing separately per-language so logits/features/tokenizer stay aligned. I also keep your existing fallback paths (baseline and finetune) intact, and keep submission formatting identical. These changes preserve your core approach (HF QA inference + same postprocess) but should move the score substantially upward toward the target.'
- What this solution (achieved 0.03214) has done: 'Your score is far below target, so the smallest high-impact fix is to ensure you are actually using the strong helper MuRIL QA checkpoints with the correct tokenizer/features, and not silently falling back to weak/garbled inference. I (1) harden checkpoint discovery to require a valid HF QA folder (config + weights), (2) rebuild Hindi/Tamil test features using each checkpoint’s own tokenizer (already intended) and ensure the tokenizers are “fast” when available (offset mappings are more reliable), and (3) fix postprocessing to use `tokenizer.cls_token_id` robustly (some tokenizers/models don’t use CLS the way your code assumes), reducing empty/invalid spans. These changes keep your model/inference approach the same (HF QA logits + same postprocess), but should move the score materially upward toward the target rather than optimizing beyond it.'
- What this solution (achieved 0.03214) has done: 'Your current score is far below target, so the most likely issue is that the “helper” MuRIL QA checkpoints are not being found/used correctly and you end up with near-random/empty spans. I make the smallest changes that improve checkpoint discovery (don’t accidentally point to a parent directory), ensure we only use real HF QA model folders (with config+weights) for Hindi/Tamil, and fix a key mismatch between feature indexing and example IDs by making postprocessing robust to non-unique ids. These changes preserve your core inference/postprocess logic (HF QA logits + same span selection) but should move the score sharply upward toward your target rather than optimizing beyond it. The script still run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 0.03214) has done: 'Your score is far below the target, so we should make the smallest change that increases real QA accuracy: ensure the helper MuRIL Hindi/Tamil QA checkpoints are actually discovered and used, rather than accidentally pointing at a parent directory where `pytorch_model.bin` exists but the *folder* is not a valid HF model directory. I minimally adjust checkpoint discovery to prefer directories that contain `config.json` plus weights (`pytorch_model.bin`/`model.safetensors`) and only then build per-language tokenizers/features aligned to those directories. I also add a very small safety fix in postprocessing to treat `sequence_ids` context tokens robustly (some tokenizers use different segment conventions), reducing empty-span fallbacks. Everything else (modeling, tokenization parameters, inference, postprocess logic, submission format/path) stays the same.'
- What this solution (achieved 0.03214) has done: 'Your score is far below the target, so the smallest likely improvement is to fix a mismatch that can silently wreck span extraction: when using the helper Hindi/Tamil HF QA checkpoints, you currently build test features with `max_seq_length=400`, which often does not match the helper checkpoints’ training/inference setup (typically 384). I keep the same HF QA inference + same postprocessing, but read each helper model’s `config.max_position_embeddings` and clamp `max_seq_length` per-language to a safe value (384 or the model limit) so offsets/logits align more like the checkpoint expects. I also ensure the feature-building uses that per-language max length without changing the model architecture/training loops, and keep the submission formatting identical. This is a minimal, execution-safe change aimed specifically at moving Jaccard up toward your target by improving span quality from the helper models rather than falling back or producing misaligned spans.'
- What this solution (achieved 0.03214) has done: 'Your current score (0.032) is far below the target (0.713), and the most likely reason is that the notebook is not actually using the strong Hindi/Tamil helper QA checkpoints correctly; in particular, it averages logits across different models even when their tokenizers produce different feature layouts, which makes the averaged logits misaligned with the offsets and yields near-random spans. I make a minimal change: for each language, only ensemble checkpoints that share the exact same tokenizer vocabulary (by comparing `tokenizer.get_vocab()`), and otherwise fall back to using the first valid checkpoint for that language rather than averaging incompatible logits. I also ensure the per-language max sequence length used to build features is clamped to the helper model’s limit (already present) but applied consistently per language without mutating the global `args` unexpectedly. These changes keep your core approach (HF QA inference + same postprocess) but should materially raise Jaccard toward your target by restoring correct span extraction.'
- What this solution (achieved 0.03214) has done: 'Your current score is far below the target, so the smallest likely improvement is to ensure inference is actually coming from the strong local helper QA checkpoints and that span postprocessing isn’t defaulting to empty/garbled answers. I minimally (1) fix the tokenizer “signature” filter that currently hashes a non-deterministic dict order (which can accidentally drop all ensemble dirs), (2) make postprocessing properly handle “no answer” by using the CLS position score (instead of always picking a span), and (3) keep offset mappings as tuples and add a tiny whitespace trim around extracted spans to avoid leading/trailing space hurting word-level Jaccard. These keep your core approach (HF QA logits + same feature building + same postprocess structure) but should move the score materially upward toward the target. The script still runs end-to-end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc

gc.enable()
import math
import json
import time
import random
import multiprocessing
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from sklearn import model_selection
from string import punctuation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler

try:
    from apex import amp

    APEX_INSTALLED = True
except ImportError:
    APEX_INSTALLED = False

import transformers
from transformers import (
    WEIGHTS_NAME,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    logging,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
)

AdamW = torch.optim.AdamW

logging.set_verbosity_warning()
logging.set_verbosity_error()


def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def optimal_num_of_loader_workers():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus * 2) if num_gpus else max(1, num_cpus - 1)
    return int(max(1, min(4, optimal_value)))


print(f"Apex AMP Installed :: {APEX_INSTALLED}")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fix_all_seeds(2021)


def _is_valid_hf_dir(path: str) -> bool:
    if not path or not isinstance(path, str):
        return False
    if not os.path.isdir(path):
        return False
    has_config = os.path.exists(os.path.join(path, "config.json"))
    has_tokenizer = any(
        os.path.exists(os.path.join(path, f))
        for f in [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "sentencepiece.bpe.model",
            "spiece.model",
        ]
    )
    return has_config and has_tokenizer


def _is_valid_hf_qa_dir(path: str) -> bool:
    if not _is_valid_hf_dir(path):
        return False
    has_weights = any(
        os.path.exists(os.path.join(path, f))
        for f in ["pytorch_model.bin", "model.safetensors"]
    )
    return has_weights


def _resolve_local_model_dir(preferred_paths):
    for p in preferred_paths:
        if p and _is_valid_hf_dir(p):
            return p
    return None


def _safe_load_tokenizer(path_or_name: str):
    for use_fast in (True, False):
        try:
            return AutoTokenizer.from_pretrained(
                path_or_name, local_files_only=True, use_fast=use_fast
            )
        except Exception:
            try:
                return AutoTokenizer.from_pretrained(path_or_name, use_fast=use_fast)
            except Exception:
                pass
    return None


def _find_ckpt_bins_under(base_dir: str, max_depth: int = 6):
    found = []
    if not base_dir or not os.path.isdir(base_dir):
        return found
    base_dir = os.path.abspath(base_dir)
    for root, dirs, files in os.walk(base_dir):
        rel_depth = os.path.relpath(root, base_dir).count(os.sep)
        if rel_depth > max_depth:
            dirs[:] = []
            continue
        if "pytorch_model.bin" in files:
            found.append(os.path.join(root, "pytorch_model.bin"))
    return found


def _pick_fold_ckpts(ckpt_bins, want_folds=5):
    fold_map = {}
    for p in ckpt_bins:
        parts = p.replace("\\", "/").split("/")
        for i, part in enumerate(parts):
            if part.startswith("checkpoint-fold-"):
                try:
                    k = int(part.split("-")[-1])
                    fold_map[k] = p
                except Exception:
                    pass
    if fold_map:
        ordered = [fold_map[k] for k in sorted(fold_map)[:want_folds]]
        return ordered
    ckpt_bins = sorted(ckpt_bins)
    return ckpt_bins[:want_folds]


def _candidate_kaggle_roots():
    roots = []
    for p in [
        "../input",
        "/kaggle/input",
        "/kaggle/data",
        "/kaggle/working",
        "/kaggle",
    ]:
        if os.path.isdir(p):
            roots.append(p)
    seen = set()
    out = []
    for r in roots:
        ar = os.path.abspath(r)
        if ar not in seen:
            out.append(ar)
            seen.add(ar)
    return out


def _find_valid_hf_dirs_by_name(
    name_contains: str, max_dirs: int = 50, max_depth: int = 6
):
    hits = []
    needle = (name_contains or "").lower()
    for root in _candidate_kaggle_roots():
        for cur_root, dirs, files in os.walk(root):
            rel_depth = os.path.relpath(cur_root, root).count(os.sep)
            if rel_depth > max_depth:
                dirs[:] = []
                continue
            if "config.json" in files:
                if needle and needle not in cur_root.lower():
                    continue
                if _is_valid_hf_dir(cur_root):
                    hits.append(cur_root)
                    if len(hits) >= max_dirs:
                        return hits
    return hits


def _find_dirs_containing_file(
    filename: str, hint: str = "", max_depth: int = 6, max_hits: int = 50
):
    hits = []
    hint_l = (hint or "").lower()
    for root in _candidate_kaggle_roots():
        for cur_root, dirs, files in os.walk(root):
            rel_depth = os.path.relpath(cur_root, root).count(os.sep)
            if rel_depth > max_depth:
                dirs[:] = []
                continue
            if filename in files:
                if hint_l and hint_l not in cur_root.lower():
                    continue
                hits.append(cur_root)
                if len(hits) >= max_hits:
                    return hits
    return hits


def _extract_state_dict(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        if any(
            isinstance(k, str) and (k.startswith("model.") or k.startswith("module."))
            for k in obj.keys()
        ):
            return obj
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    return None


def _strip_prefix_from_state_dict(sd: dict, prefixes=("module.", "model.")):
    if sd is None:
        return None
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        out[nk] = v
    return out




## === cell 1
class Configration:
    model_type = "xlm_roberta"
    XLMR_name_or_path = "../input/nothing-of-your-concern/xlm-roberta-large-squad2/xlm-roberta-large-squad2/xlm-roberta-large-squad2"
    MURIL_name_or_path = (
        "../input/nothing-of-your-concern/muril-large-cased/muril-large-cased"
    )
    MPNET2_name_or_path = "../input/nothing-of-your-concern/multi-qa-mpnet-base-cos-v1/multi-qa-mpnet-base-cos-v1"
    MPNET3_name_or_path = "../input/nothing-of-your-concern/multi-qa-mpnet-base-dot-v1/multi-qa-mpnet-base-dot-v1"
    XLMR_config_name = "../input/nothing-of-your-concern/xlm-roberta-large-squad2/xlm-roberta-large-squad2/config.json"
    MURIL_config_name = "../input/nothing-of-your-concern/muril-large-cased/muril-large-cased/config.json"
    MPNET2_config_name = "../input/nothing-of-your-concern/multi-qa-mpnet-base-cos-v1/multi-qa-mpnet-base-cos-v1/config.json"
    MPNET3_config_name = "../input/nothing-of-your-concern/multi-qa-mpnet-base-dot-v1/multi-qa-mpnet-base-dot-v1/config.json"
    fp16 = True if APEX_INSTALLED else False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2

    XLMR_tokenizer_name = "../input/nothing-of-your-concern/xlm-roberta-large-squad2/xlm-roberta-large-squad2/"
    MURIL_tokenizer_name = (
        "../input/nothing-of-your-concern/muril-large-cased/muril-large-cased/"
    )
    MPNET2_tokenizer_name = "../input/nothing-of-your-concern/multi-qa-mpnet-base-cos-v1/multi-qa-mpnet-base-cos-v1/"
    MPNET3_tokenizer_name = "../input/nothing-of-your-concern/multi-qa-mpnet-base-dot-v1/multi-qa-mpnet-base-dot-v1/"
    max_seq_length = 400
    doc_stride = 135

    epochs = 1
    train_batch_size = 4
    eval_batch_size = 128

    optimizer_type = "AdamW"
    learning_rate = 1e-5
    weight_decay = 1e-2
    epsilon = 1e-8
    max_grad_norm = 1.0

    decay_name = "linear-warmup"
    warmup_ratio = 0.1

    logging_steps = 10

    output_dir = "output"
    seed = 2021




## === cell 2
class Dataset_Retriever(Dataset):
    def __init__(self, features, mode="train"):
        super(Dataset_Retriever, self).__init__()
        self.features = features
        self.mode = mode

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature = self.features[item]
        if self.mode == "train":
            return {
                "input_ids": torch.tensor(feature["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(
                    feature["attention_mask"], dtype=torch.long
                ),
                "offset_mapping": torch.tensor(
                    feature["offset_mapping"], dtype=torch.long
                ),
                "start_position": torch.tensor(
                    feature["start_position"], dtype=torch.long
                ),
                "end_position": torch.tensor(feature["end_position"], dtype=torch.long),
            }
        else:
            return {
                "language": feature["language"],
                "input_ids": torch.tensor(feature["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(
                    feature["attention_mask"], dtype=torch.long
                ),
                "offset_mapping": feature["offset_mapping"],
                "sequence_ids": feature["sequence_ids"],
                "id": feature["example_id"],
                "context": feature["context"],
                "question": feature["question"],
            }




## === cell 3
class Model(nn.Module):
    def __init__(self, modelname_or_path, config):
        super(Model, self).__init__()
        self.config = config
        self.xlm_roberta = AutoModel.from_pretrained(modelname_or_path, config=config)
        self.linear_layer = nn.Linear(config.hidden_size, 64)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(64, 2)
        self._init_weights(self.qa_outputs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.xlm_roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        linear_output = self.linear_layer(sequence_output)
        linear_output = self.dropout(linear_output)
        qa_logits = self.qa_outputs(linear_output)

        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits




## === cell 4
def Make_Model(args):
    try:
        config = AutoConfig.from_pretrained(
            args.MURIL_name_or_path, local_files_only=True
        )
    except Exception:
        config = AutoConfig.from_pretrained(args.MURIL_name_or_path)
    tokenizer = _safe_load_tokenizer(args.MURIL_name_or_path)
    model = Model(args.MURIL_name_or_path, config=config)
    return config, tokenizer, model




## === cell 5
def Prepare_Test_Features(args, example, tokenizer):
    example["question"] = str(example["question"]).lstrip()

    tokenized_example = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=args.max_seq_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    features = []
    for i in range(len(tokenized_example["input_ids"])):
        feature = {}
        feature["language"] = example["language"]
        feature["example_id"] = example["id"]
        feature["context"] = example["context"]
        feature["question"] = example["question"]
        feature["input_ids"] = tokenized_example["input_ids"][i]
        feature["attention_mask"] = tokenized_example["attention_mask"][i]
        feature["offset_mapping"] = [
            (int(a), int(b)) for (a, b) in tokenized_example["offset_mapping"][i]
        ]
        feature["sequence_ids"] = [
            0 if x is None else x for x in tokenized_example.sequence_ids(i)
        ]
        features.append(feature)
    return features




## === cell 6
import collections


def Postprocess_qa_predictions(
    examples, features, raw_predictions, tokenizer, n_best_size=20, max_answer_length=30
):
    all_start_logits, all_end_logits = raw_predictions

    example_id_to_index = collections.defaultdict(list)
    for i, _id in enumerate(examples["id"].tolist()):
        example_id_to_index[_id].append(i)

    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        for ex_i in example_id_to_index.get(feature["example_id"], []):
            features_per_example[ex_i].append(i)

    predictions = collections.OrderedDict()
    print(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )

    cls_token_id = getattr(tokenizer, "cls_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)

    for example_index, example in examples.iterrows():
        feature_indices = features_per_example.get(example_index, [])
        valid_answers = []
        best_null_score = -1e18

        context = example["context"]
        for feature_index in feature_indices:
            start_logits = np.array(all_start_logits[feature_index])
            end_logits = np.array(all_end_logits[feature_index])

            sequence_ids = features[feature_index]["sequence_ids"]

            seg_ids = [sid for sid in sequence_ids if sid is not None]
            non_zero = [sid for sid in seg_ids if sid != 0]
            context_index = max(set(non_zero), key=non_zero.count) if non_zero else 1

            offset_mapping = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(features[feature_index]["offset_mapping"])
            ]

            input_ids = features[feature_index]["input_ids"]
            cls_index = None
            if cls_token_id is not None and cls_token_id in input_ids:
                cls_index = input_ids.index(cls_token_id)
            elif pad_token_id is not None and pad_token_id in input_ids:
                cls_index = input_ids.index(pad_token_id)
            else:
                cls_index = 0

            null_score = float(start_logits[cls_index] + end_logits[cls_index])
            if null_score > best_null_score:
                best_null_score = null_score

            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    text = context[
                        start_char:end_char
                    ].strip()  # Change: trim span whitespace to help Jaccard tokens.
                    valid_answers.append(
                        {
                            "score": float(
                                start_logits[start_index] + end_logits[end_index]
                            ),
                            "text": text,
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                0
            ]
            if best_null_score >= best_answer["score"]:
                best_text = ""
            else:
                best_text = best_answer["text"]
        else:
            best_text = ""

        predictions[example["id"]] = best_text

    return predictions




## === cell 7
DATA_DIR = "/kaggle/input/chaii-hindi-and-tamil-question-answering"
if not os.path.isdir(DATA_DIR):
    DATA_DIR = "/kaggle/data/chaii-hindi-and-tamil-question-answering"
if not os.path.isdir(DATA_DIR):
    DATA_DIR = "../input/chaii-hindi-and-tamil-question-answering"

test_path = os.path.join(DATA_DIR, "test.csv")
train_path = os.path.join(DATA_DIR, "train.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

test_df = pd.read_csv(test_path)
train_df = pd.read_csv(train_path)
sample_df = pd.read_csv(sample_path)

test_df["context"] = test_df["context"].astype(str).apply(lambda x: " ".join(x.split()))
test_df["question"] = (
    test_df["question"].astype(str).apply(lambda x: " ".join(x.split()))
)

train_df["context"] = (
    train_df["context"].astype(str).apply(lambda x: " ".join(x.split()))
)
train_df["question"] = (
    train_df["question"].astype(str).apply(lambda x: " ".join(x.split()))
)
train_df["answer_text"] = train_df["answer_text"].astype(str)

print("Using DATA_DIR:", DATA_DIR)
print("Test rows:", len(test_df), "Train rows:", len(train_df))



## === cell 8
args = Configration()

candidate_muril_dirs = [
    os.path.join(DATA_DIR, "muril-base-cased"),
    os.path.join(DATA_DIR, "muril-large-cased"),
    "/kaggle/input/muril-base-cased",
    "/kaggle/input/muril-large-cased",
    "../input/muril-base-cased",
    "../input/muril-large-cased",
    "../input/chaii-hindi-and-tamil-question-answering/muril-base-cased",
    "../input/chaii-hindi-and-tamil-question-answering/muril-large-cased",
]
resolved_muril_dir = _resolve_local_model_dir(candidate_muril_dirs)
if resolved_muril_dir is None:
    muril_hits = _find_valid_hf_dirs_by_name("muril", max_dirs=10, max_depth=6)
    resolved_muril_dir = muril_hits[0] if muril_hits else None

if resolved_muril_dir is not None:
    args.MURIL_name_or_path = resolved_muril_dir
    args.MURIL_tokenizer_name = resolved_muril_dir
    args.MURIL_config_name = resolved_muril_dir
    tokenizer = _safe_load_tokenizer(resolved_muril_dir)
else:
    tokenizer = None

test_features = []
if tokenizer is not None:
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Build test feats"):
        test_features += Prepare_Test_Features(args, row, tokenizer)

    test_dataset = Dataset_Retriever(test_features, mode="test")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        sampler=SequentialSampler(test_dataset),
        num_workers=optimal_num_of_loader_workers(),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
else:
    test_dataloader = None

print("Resolved MURIL dir:", resolved_muril_dir)
print("Tokenizer available:", tokenizer is not None)
print("Num test features:", len(test_features))



## === cell 9
base_model_hindi = "../input/chaii-helper/Muril-H/content/output/"
base_model_tamil = "../input/chaii-helper/Muril-T/content/output/"


def _safe_torch_load(path):
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")


def _is_probably_custom_state_ckpt(path: str) -> bool:
    if not path:
        return False
    p = str(path).lower()
    return (
        p.endswith(".pt") or p.endswith(".pth") or (p.endswith(".bin") and "state" in p)
    )


def Get_Predictions_From_CustomStateDict(checkpoint_path):
    config, tok, model = Make_Model(args)
    model.to(DEVICE)

    sd_obj = _safe_torch_load(checkpoint_path)
    if sd_obj is None:
        del model, tok, config
        gc.collect()
        return None, None, None

    sd = _extract_state_dict(sd_obj)
    if sd is None:
        del model, tok, config
        gc.collect()
        return None, None, None

    sd = _strip_prefix_from_state_dict(sd, prefixes=("module.", "model."))
    model.load_state_dict(sd, strict=False)

    model.eval()
    start_logits = []
    end_logits = []
    languages = []
    for batch in tqdm(test_dataloader, total=len(test_dataloader), leave=False):
        with torch.no_grad():
            outputs_start, outputs_end = model(
                batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)
            )
            start_logits.append(outputs_start.detach().cpu().numpy())
            end_logits.append(outputs_end.detach().cpu().numpy())
            languages.extend(batch["language"])
    del model, tok, config
    gc.collect()
    return np.vstack(start_logits), np.vstack(end_logits), languages


def Get_Predictions_From_HF_QA_Dir(hf_dir: str, dataloader: DataLoader):
    if not hf_dir or not os.path.isdir(hf_dir):
        return None, None
    if not _is_valid_hf_qa_dir(hf_dir):
        return None, None
    try:
        qa_model = AutoModelForQuestionAnswering.from_pretrained(
            hf_dir, local_files_only=True
        )
    except Exception:
        try:
            qa_model = AutoModelForQuestionAnswering.from_pretrained(hf_dir)
        except Exception:
            return None, None

    qa_model.to(DEVICE)
    qa_model.eval()

    start_logits = []
    end_logits = []
    for batch in tqdm(dataloader, total=len(dataloader), leave=False):
        with torch.no_grad():
            out = qa_model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
            )
            start_logits.append(out.start_logits.detach().cpu().numpy())
            end_logits.append(out.end_logits.detach().cpu().numpy())

    del qa_model
    gc.collect()
    return np.vstack(start_logits), np.vstack(end_logits)


def Prepare_Train_Features(args, example, tokenizer):
    example["question"] = str(example["question"]).lstrip()
    context = str(example["context"])
    answer_text = str(example["answer_text"])
    answer_start = int(example["answer_start"])

    tokenized_example = tokenizer(
        example["question"],
        context,
        truncation="only_second",
        max_length=args.max_seq_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    features = []
    for i in range(len(tokenized_example["input_ids"])):
        feature = {}
        feature["input_ids"] = tokenized_example["input_ids"][i]
        feature["attention_mask"] = tokenized_example["attention_mask"][i]
        feature["offset_mapping"] = tokenized_example["offset_mapping"][i]
        sequence_ids = [
            0 if x is None else x for x in tokenized_example.sequence_ids(i)
        ]
        feature["offset_mapping"] = [
            (o if sequence_ids[k] == 1 else (0, 0))
            for k, o in enumerate(feature["offset_mapping"])
        ]

        start_char = answer_start
        end_char = answer_start + len(answer_text)

        start_pos, end_pos = 0, 0
        found = False
        for t_idx, (s, e) in enumerate(feature["offset_mapping"]):
            if s == e == 0:
                continue
            if (s <= start_char < e) and not found:
                start_pos = t_idx
                found = True
            if found and (s < end_char <= e):
                end_pos = t_idx
                break

        if not found or start_pos == 0 or end_pos == 0:
            start_pos = 0
            end_pos = 0

        feature["start_position"] = start_pos
        feature["end_position"] = end_pos
        features.append(feature)

    return features


def FineTune_If_Needed(args):
    config, tok, model = Make_Model(args)
    model.to(DEVICE)

    train_features = []
    for _, row in tqdm(
        train_df.iterrows(), total=len(train_df), desc="Build train feats"
    ):
        train_features += Prepare_Train_Features(args, row, tok)

    train_dataset = Dataset_Retriever(train_features, mode="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=optimal_num_of_loader_workers(),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.epsilon
    )

    t_total = (len(train_dataloader) // args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(args.warmup_ratio * t_total)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    ce_loss_fct = nn.CrossEntropyLoss()
    model.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        pbar = tqdm(train_dataloader, desc=f"Train epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            start_positions = batch["start_position"].to(DEVICE)
            end_positions = batch["end_position"].to(DEVICE)

            start_logits, end_logits = model(input_ids, attention_mask=attention_mask)
            start_loss = ce_loss_fct(start_logits, start_positions)
            end_loss = ce_loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2.0
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if global_step % max(1, args.logging_steps) == 0:
                pbar.set_postfix({"loss": float(loss.detach().cpu().numpy())})

    model.eval()
    return model, tok, config


def _baseline_predict_from_context(df: pd.DataFrame) -> collections.OrderedDict:
    preds = collections.OrderedDict()
    for _id, ctx in df[["id", "context"]].itertuples(index=False):
        ctx = "" if pd.isna(ctx) else str(ctx)
        cut = min(len(ctx), 80)
        pred = ctx[:cut].strip()
        preds[_id] = pred
    return preds




## === cell 10
def _resolve_helper_base(default_path: str, glob_hint: str):
    if os.path.isdir(default_path):
        return default_path

    hint = (glob_hint or "").lower()
    for root in _candidate_kaggle_roots():
        if not os.path.isdir(root):
            continue
        try:
            for d in os.listdir(root):
                p = os.path.join(root, d)
                if os.path.isdir(p) and (hint in d.lower()):
                    for q in [
                        p,
                        os.path.join(p, "content", "output"),
                        os.path.join(p, "Muril-H", "content", "output"),
                        os.path.join(p, "Muril-T", "content", "output"),
                        os.path.join(p, "output"),
                    ]:
                        if os.path.isdir(q):
                            return q
        except Exception:
            pass

    return default_path


def _find_hf_qa_model_dirs_under(
    base_dir: str, max_depth: int = 10, max_dirs: int = 20
):
    hits = []
    if not base_dir or not os.path.isdir(base_dir):
        return hits
    base_dir = os.path.abspath(base_dir)
    for root, dirs, files in os.walk(base_dir):
        rel_depth = os.path.relpath(root, base_dir).count(os.sep)
        if rel_depth > max_depth:
            dirs[:] = []
            continue
        if "config.json" in files and (
            ("pytorch_model.bin" in files) or ("model.safetensors" in files)
        ):
            if _is_valid_hf_qa_dir(root):
                hits.append(root)
                if len(hits) >= max_dirs:
                    break
    return sorted(list(dict.fromkeys(hits)))


def _infer_safe_max_seq_len_for_hf_dir(hf_dir: str, default_len: int):
    if not hf_dir or not os.path.isdir(hf_dir):
        return int(default_len)
    try:
        cfg = AutoConfig.from_pretrained(hf_dir, local_files_only=True)
    except Exception:
        try:
            cfg = AutoConfig.from_pretrained(hf_dir)
        except Exception:
            return int(default_len)

    mpe = getattr(cfg, "max_position_embeddings", None)
    safe = 384
    if isinstance(mpe, int) and mpe > 0:
        safe = min(safe, mpe)
    return int(min(default_len, safe))


def _tokenizer_vocab_signature(tok) -> int:
    try:
        vocab = tok.get_vocab()
        keys = sorted(vocab.keys())
        return hash(tuple(keys[:5000]))
    except Exception:
        return -1


def _filter_dirs_by_same_tokenizer(hf_dirs, reference_tok):
    if not hf_dirs or reference_tok is None:
        return []
    ref_sig = _tokenizer_vocab_signature(reference_tok)
    kept = []
    for d in hf_dirs:
        t = _safe_load_tokenizer(d)
        if t is None:
            continue
        if _tokenizer_vocab_signature(t) == ref_sig:
            kept.append(d)
    return kept


base_model_hindi = _resolve_helper_base(base_model_hindi, "chaii-helper")
base_model_tamil = _resolve_helper_base(base_model_tamil, "chaii-helper")

hindi_bins = _find_ckpt_bins_under(base_model_hindi, max_depth=10)
tamil_bins = _find_ckpt_bins_under(base_model_tamil, max_depth=10)

hindi_ckpts = _pick_fold_ckpts(hindi_bins, want_folds=5)
tamil_ckpts = _pick_fold_ckpts(tamil_bins, want_folds=4)

hindi_dirs = _find_hf_qa_model_dirs_under(base_model_hindi, max_depth=10, max_dirs=10)
tamil_dirs = _find_hf_qa_model_dirs_under(base_model_tamil, max_depth=10, max_dirs=10)


def _bin_to_dir(p: str) -> str:
    return os.path.dirname(p) if p else p


if not hindi_dirs:
    hindi_dirs = [
        d for d in (_bin_to_dir(p) for p in hindi_ckpts) if _is_valid_hf_qa_dir(d)
    ]
if not tamil_dirs:
    tamil_dirs = [
        d for d in (_bin_to_dir(p) for p in tamil_ckpts) if _is_valid_hf_qa_dir(d)
    ]

hindi_dir0 = hindi_dirs[0] if hindi_dirs else None
tamil_dir0 = tamil_dirs[0] if tamil_dirs else None

print("Resolved Hindi ckpt base:", base_model_hindi)
print("Resolved Tamil ckpt base:", base_model_tamil)
print("Found Hindi bins:", len(hindi_bins), "Using HF dirs:", len(hindi_dirs))
print("Found Tamil bins:", len(tamil_bins), "Using HF dirs:", len(tamil_dirs))

test_features_h, test_dataloader_h, tok_h = [], None, None
test_features_t, test_dataloader_t, tok_t = [], None, None

if hindi_dir0 and os.path.isdir(hindi_dir0):
    tok_h = _safe_load_tokenizer(hindi_dir0)
    if tok_h is not None:
        df_h = test_df[test_df["language"] == "hindi"].reset_index(drop=True)
        args_h = Configration()
        args_h.__dict__.update(args.__dict__)
        args_h.max_seq_length = _infer_safe_max_seq_len_for_hf_dir(
            hindi_dir0, int(args.max_seq_length)
        )
        for _, row in tqdm(df_h.iterrows(), total=len(df_h), desc="Build H feats"):
            test_features_h += Prepare_Test_Features(args_h, row, tok_h)
        ds_h = Dataset_Retriever(test_features_h, mode="test")
        test_dataloader_h = DataLoader(
            ds_h,
            batch_size=args.eval_batch_size,
            sampler=SequentialSampler(ds_h),
            num_workers=optimal_num_of_loader_workers(),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

if tamil_dir0 and os.path.isdir(tamil_dir0):
    tok_t = _safe_load_tokenizer(tamil_dir0)
    if tok_t is not None:
        df_t = test_df[test_df["language"] != "hindi"].reset_index(drop=True)
        args_t = Configration()
        args_t.__dict__.update(args.__dict__)
        args_t.max_seq_length = _infer_safe_max_seq_len_for_hf_dir(
            tamil_dir0, int(args.max_seq_length)
        )
        for _, row in tqdm(df_t.iterrows(), total=len(df_t), desc="Build T feats"):
            test_features_t += Prepare_Test_Features(args_t, row, tok_t)
        ds_t = Dataset_Retriever(test_features_t, mode="test")
        test_dataloader_t = DataLoader(
            ds_t,
            batch_size=args.eval_batch_size,
            sampler=SequentialSampler(ds_t),
            num_workers=optimal_num_of_loader_workers(),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

can_helper_h = (
    (tok_h is not None) and (test_dataloader_h is not None) and bool(hindi_dirs)
)
can_helper_t = (
    (tok_t is not None) and (test_dataloader_t is not None) and bool(tamil_dirs)
)

if not (can_helper_h or can_helper_t):
    if tokenizer is None or resolved_muril_dir is None or test_dataloader is None:
        fin_preds = _baseline_predict_from_context(test_df)
    else:
        model, tok, config = FineTune_If_Needed(args)

        start_logits = []
        end_logits = []
        for batch in tqdm(test_dataloader, total=len(test_dataloader), desc="Infer"):
            with torch.no_grad():
                s, e = model(
                    batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)
                )
                start_logits.append(s.detach().cpu().numpy())
                end_logits.append(e.detach().cpu().numpy())
        start_logits = np.vstack(start_logits)
        end_logits = np.vstack(end_logits)

        fin_preds = Postprocess_qa_predictions(
            test_df, test_features, (start_logits, end_logits), tok
        )
else:
    fin_preds = collections.OrderedDict()

    if can_helper_h:
        hindi_dirs_ens = _filter_dirs_by_same_tokenizer(hindi_dirs, tok_h)
        if not hindi_dirs_ens:
            hindi_dirs_ens = [hindi_dir0] if hindi_dir0 else []
        s_list, e_list = [], []
        for d in hindi_dirs_ens:
            s, e = Get_Predictions_From_HF_QA_Dir(d, test_dataloader_h)
            if s is not None:
                s_list.append(s)
                e_list.append(e)
        if s_list:
            start_logits_h = np.mean(s_list, axis=0) if len(s_list) > 1 else s_list[0]
            end_logits_h = np.mean(e_list, axis=0) if len(e_list) > 1 else e_list[0]
            df_h = test_df[test_df["language"] == "hindi"].reset_index(drop=True)
            preds_h = Postprocess_qa_predictions(
                df_h, test_features_h, (start_logits_h, end_logits_h), tok_h
            )
            fin_preds.update(preds_h)

    if can_helper_t:
        tamil_dirs_ens = _filter_dirs_by_same_tokenizer(tamil_dirs, tok_t)
        if not tamil_dirs_ens:
            tamil_dirs_ens = [tamil_dir0] if tamil_dir0 else []
        s_list, e_list = [], []
        for d in tamil_dirs_ens:
            s, e = Get_Predictions_From_HF_QA_Dir(d, test_dataloader_t)
            if s is not None:
                s_list.append(s)
                e_list.append(e)
        if s_list:
            start_logits_t = np.mean(s_list, axis=0) if len(s_list) > 1 else s_list[0]
            end_logits_t = np.mean(e_list, axis=0) if len(e_list) > 1 else e_list[0]
            df_t = test_df[test_df["language"] != "hindi"].reset_index(drop=True)
            preds_t = Postprocess_qa_predictions(
                df_t, test_features_t, (start_logits_t, end_logits_t), tok_t
            )
            fin_preds.update(preds_t)

    if len(fin_preds) != len(test_df):
        base_fill = _baseline_predict_from_context(test_df)
        for _id in test_df["id"].tolist():
            if _id not in fin_preds:
                fin_preds[_id] = base_fill[_id]



## === cell 11
submission = []
for pid, pred in fin_preds.items():
    pred = " ".join(str(pred).split()).strip()
    submission.append((pid, pred))

sample = pd.DataFrame(submission, columns=["id", "PredictionString"])
test_data = pd.merge(left=test_df, right=sample, on="id", how="left")



## === cell 12
bad_starts = [".", ",", "(", ")", "-", "–", ",", ";"]
bad_endings = ["...", "-", "(", ")", "–", ",", ";"]

tamil_ad = "கி.பி"
tamil_bc = "கி.மு"
tamil_km = "கி.மீ"
hindi_ad = "ई"
hindi_bc = "ई.पू"

cleaned_preds = []
for pred, context in test_data[["PredictionString", "context"]].to_numpy():
    pred = "" if pd.isna(pred) else str(pred)
    context = "" if pd.isna(context) else str(context)

    if pred == "":
        cleaned_preds.append(pred)
        continue

    while pred and any(pred.startswith(y) for y in bad_starts):
        pred = pred[1:]
    while pred and any(pred.endswith(y) for y in bad_endings):
        if pred.endswith("..."):
            pred = pred[:-3]
        else:
            pred = pred[:-1]
    if pred.endswith("..."):
        pred = pred[:-3]

    if (
        any(
            [
                pred.endswith(tamil_ad),
                pred.endswith(tamil_bc),
                pred.endswith(tamil_km),
                pred.endswith(hindi_ad),
                pred.endswith(hindi_bc),
            ]
        )
        and (pred + ".") in context
    ):
        pred = pred + "."

    cleaned_preds.append(pred.strip())

test_data["PredictionString"] = cleaned_preds

out_path = "submission.csv"
test_data[["id", "PredictionString"]].to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape={test_data[['id','PredictionString']].shape}")
print(test_data[["id", "PredictionString"]].head())



## === cell 13
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["id", "PredictionString"]
assert len(sub) == len(test_df)
print("Submission looks valid.")
