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

0.7284508347511292

# 6. Current score

0.00431

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0022) has done: 'I speed up inference by (1) tokenizing the whole test set in large batches (instead of per-row Python loops), (2) using a custom `collate_fn` so the DataLoader doesn’t repeatedly convert Python lists to tensors per item, and (3) avoiding repeated, in-place mutation of `features` during post-processing by precomputing per-feature context-only offset mappings and CLS indices once. I also reuse the already-loaded tokenizer/config and enable fast inference settings (`inference_mode`, `torch.compile` when available) without changing model architecture or prediction semantics. These changes remove the main Python overhead hotspots while keeping identical logits, postprocessing logic, and ensembling behavior (up to negligible float differences).'
- What this solution (achieved 0.0022) has done: 'I fix the crash caused by an incompatible protobuf/transformers interaction that manifests as `MessageFactory.GetPrototype` during imports/model loading by forcing Transformers to use the pure-Python protobuf implementation before `transformers` is imported. I also make the fallback model path use a local Kaggle-provided base model (`../input/xlm-roberta-base`) if available to avoid any internet dependency (the current fallback HF id can silently fail offline and tanks score). These are minimal changes that keep your model/post-processing logic intact, but ensure the notebook runs end-to-end and produces `submission.csv` correctly. With checkpoints present, the score should move sharply upward toward the target; if checkpoints are missing, the local base-model fallback is still much better than a broken/empty pipeline.'
- What this solution (achieved 0.0022) has done: 'I fix the `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before* any `transformers` import and by ensuring the related protobuf env vars are set early and consistently. I also make checkpoint/model loading robust in offline Kaggle runs by strictly preferring existing local model directories and avoiding any accidental online fetches. Finally, I ensure the script always reaches the CSV write step and produces a valid `submission.csv` with the exact required columns and row count aligned to `test.csv`. These changes are execution/robustness fixes (and should substantially improve score vs the broken import path), while preserving your model, logits, ensembling, and post-processing logic.'
- What this solution (achieved 0.0022) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before any other imports* (especially anything that can indirectly import `google.protobuf`, like `sentencepiece`/`transformers`). Then I make checkpoint loading robust: if the listed fold checkpoints are missing, the code still run using the base model without errors and always write a valid `submission.csv`. Finally, I ensure the test/features alignment and submission merge cannot silently misalign by enforcing string `id` types and by producing predictions for every test id.'
- What this solution (achieved 0.00431) has done: 'I fix the `HFValidationError` by making `_resolve_existing_model_path` return a true existing directory (searching Kaggle’s `../input` mount) and, if nothing is found, falling back to a valid built-in HF model id (`"xlm-roberta-base"`) so Transformers no longer tries to validate a non-existent `../input/...` path. I also make cell 7 use the already-defined `make_model()` helper (so config/tokenizer/model resolution is consistent everywhere) and ensure we always define `test_data` even if something upstream fails, so the pipeline reaches the CSV write step. These are execution/robustness fixes and should substantially improve score versus “no submission”, while preserving your model forward pass and post-processing logic. Finally, I keep the submission format aligned to `test.csv` and always write `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your very low score suggests the model is effectively running with random/untrained weights because the intended fold checkpoints are not being found/loaded, so the smallest score-improving change is to make checkpoint discovery robust (without changing the model or post-processing). I add a minimal resolver that searches Kaggle’s `../input` mount for the expected `pytorch_model.bin` filenames and loads any matches, so ensembling uses the real fine-tuned weights when present. I also force `local_files_only=True` for model/config/tokenizer loading to avoid any offline fallback that silently pulls/uses mismatched assets, which can tank performance. Everything else (architecture, inference loop, logits ensembling, post-processing, and submission formatting) stays the same.'
- What this solution (achieved 0.0) has done: 'The crash is happening because `make_model()` and `_get_global_cfg_tok()` force `local_files_only=True` but your resolver falls back to the Hub id `"xlm-roberta-base"` even when no local model directory exists, which then fails in offline Kaggle. I make the resolver strictly return an existing local directory (prefer Kaggle competition dataset folders first), and if none is found, use the Kaggle-mounted HF cache if present; otherwise we fail fast with a clear error instead of attempting an online fetch. This keeps your model/inference/post-processing logic identical, but unblocks end-to-end execution so a valid `submission.csv` is written and the score can move toward the target when checkpoints/model assets are actually available locally. I also ensure `persistent_workers` is only enabled when `num_workers>0` to avoid occasional DataLoader issues.'
- What this solution (achieved 0.00431) has done: 'Your code fails because `_resolve_existing_model_path()` cannot find any local Transformers model directory with a `config.json`, so `make_model()` and `_get_global_cfg_tok()` crash before any predictions/submission can be written. I make the resolver also accept Kaggle’s pre-downloaded Hugging Face cache locations (common in offline Kaggle images), and only if that still fails, fall back to `local_files_only=False` for *just* config/tokenizer/model loading so the notebook can run end-to-end (this is required to move score above 0.0). I also make `get_predictions("__no_checkpoint__")` skip checkpoint loading cleanly (score-neutral) and keep the model/inference/post-processing logic unchanged. Finally, I ensure we always write `submission.csv` with exactly the required columns aligned to `test.csv`.'
- What this solution (achieved 0.00431) has done: 'Your current score (0.00431) is far below the target (0.72845), which strongly suggests you are not actually using the intended fine-tuned checkpoints during inference (likely `available==0`, so you’re predicting with a base model). The smallest high-impact change is to make checkpoint discovery robust by also scanning `/kaggle/input` for any `pytorch_model.bin` whose folder name contains `chaii`/`abhishek` and then ensembling those found weights (keeping your exact model forward + postprocess). I also fix a quiet but important accuracy bug in post-processing: `example_id_to_index` uses raw ids while features use `str()`, so many features can be dropped if types mismatch; forcing both sides to `str` ensures every test example receives its logits. These changes preserve your architecture, logits->span selection logic, and submission formatting, but should move the score sharply upward toward the target if any real checkpoints exist locally.'
- What this solution (achieved 0.00431) has done: 'Your current score is far below the target, which strongly indicates you’re still not loading the intended fine-tuned checkpoints (or you’re loading the wrong weights into the wrong model). I make the smallest score-critical fix: ensure that when a checkpoint is found, we instantiate the model from that checkpoint’s *own directory* (so config/tokenizer/base weights match), then load the state dict and run inference exactly as before. I also tighten checkpoint auto-discovery to prioritize fold checkpoint files and to derive their model directory correctly, while keeping the same ensembling/post-processing and still writing a valid `submission.csv`. These changes keep your architecture/forward/postprocess semantics intact but should move score sharply upward toward the target when checkpoints are present locally.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from google.protobuf import message_factory as _message_factory  # noqa: F401

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):  # noqa: N802
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            if hasattr(_message_factory, "GetMessageClass"):
                return _message_factory.GetMessageClass(descriptor)
            raise AttributeError(
                "Neither GetPrototype nor GetMessageClass is available on MessageFactory"
            )

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SequentialSampler

try:
    from apex import amp

    APEX_INSTALLED = True
except ImportError:
    APEX_INSTALLED = False

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    logging,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
)

from torch.optim import AdamW  # noqa: F401

logging.set_verbosity_warning()
logging.set_verbosity_error()


def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def optimal_num_of_loader_workers():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus * 4) if num_gpus else max(1, num_cpus - 1)
    return optimal_value


print(f"Apex AMP Installed :: {APEX_INSTALLED}")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fix_all_seeds(2021)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True




## === cell 1
class Config:
    model_type = "xlm-roberta"
    model_name_or_path = "../input/abhishek-chaii-qa-model"
    config_name = "../input/abhishek-chaii-qa-model"
    fp16 = True if APEX_INSTALLED else False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2

    tokenizer_name = "../input/abhishek-chaii-qa-model"
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
class DatasetRetriever(Dataset):
    def __init__(self, features, mode="train"):
        super(DatasetRetriever, self).__init__()
        self.features = features
        self.mode = mode

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature = self.features[item]
        if self.mode == "train":
            return {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "offset_mapping": feature["offset_mapping"],
                "start_position": feature["start_position"],
                "end_position": feature["end_position"],
            }
        else:
            return {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "offset_mapping": feature["offset_mapping"],
                "sequence_ids": feature["sequence_ids"],
                "id": feature["example_id"],
                "context": feature["context"],
                "question": feature["question"],
            }


def collate_test_fn(batch):
    input_ids = torch.as_tensor([b["input_ids"] for b in batch], dtype=torch.long)
    attention_mask = torch.as_tensor(
        [b["attention_mask"] for b in batch], dtype=torch.long
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "offset_mapping": [b["offset_mapping"] for b in batch],
        "sequence_ids": [b["sequence_ids"] for b in batch],
        "id": [b["id"] for b in batch],
        "context": [b["context"] for b in batch],
        "question": [b["question"] for b in batch],
    }




## === cell 3
class Model(nn.Module):
    def __init__(self, modelname_or_path, config, local_files_only=True):
        super(Model, self).__init__()
        self.config = config
        self.xlm_roberta = AutoModel.from_pretrained(
            modelname_or_path, config=config, local_files_only=local_files_only
        )
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
def _iter_candidate_model_dirs():
    candidates = [
        "../input/abhishek-chaii-qa-model",
        "/kaggle/input/abhishek-chaii-qa-model",
        "../input/xlm-roberta-base-squad2",
        "/kaggle/input/xlm-roberta-base-squad2",
        "../input/xlm-roberta-base",
        "/kaggle/input/xlm-roberta-base",
        "../input/xlm-roberta-large",
        "/kaggle/input/xlm-roberta-large",
    ]
    for p in candidates:
        yield p

    cache_roots = [
        os.environ.get("HF_HOME", ""),
        os.environ.get("TRANSFORMERS_CACHE", ""),
        "/kaggle/working/.cache/huggingface",
        "/kaggle/working/.cache/huggingface/transformers",
        "/root/.cache/huggingface",
        "/root/.cache/huggingface/hub",
        "/home/jovyan/.cache/huggingface",
    ]
    for cr in cache_roots:
        if not cr:
            continue
        yield cr
        yield os.path.join(cr, "hub")

    for base in ("../input", "/kaggle/input"):
        yield base


def _resolve_existing_model_path(preferred_path: str) -> str:
    """
    Resolver: find a local Transformers model directory containing config.json.
    """
    if isinstance(preferred_path, str) and os.path.isdir(preferred_path):
        if os.path.exists(os.path.join(preferred_path, "config.json")):
            return preferred_path

    if isinstance(preferred_path, str) and os.path.exists(preferred_path):
        if os.path.isdir(preferred_path) and os.path.exists(
            os.path.join(preferred_path, "config.json")
        ):
            return preferred_path

    for p in _iter_candidate_model_dirs():
        try:
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
                return p
        except Exception:
            continue

    for root_base in _iter_candidate_model_dirs():
        if not root_base or not os.path.isdir(root_base):
            continue
        try:
            for root, dirs, files in os.walk(root_base):
                if "config.json" in files:
                    return root
        except Exception:
            pass

    raise FileNotFoundError(
        "Could not find any local Transformers model directory containing config.json. "
        "If you are offline, add the required model/checkpoints as a Kaggle Dataset input "
        "or ensure the HF cache is available."
    )


def make_model(args):
    local_only = True
    try:
        resolved_config_name = _resolve_existing_model_path(args.config_name)
        resolved_tokenizer_name = _resolve_existing_model_path(args.tokenizer_name)
        resolved_model_path = _resolve_existing_model_path(args.model_name_or_path)
    except FileNotFoundError:
        local_only = False
        resolved_config_name = "xlm-roberta-base"
        resolved_tokenizer_name = "xlm-roberta-base"
        resolved_model_path = "xlm-roberta-base"

    config = AutoConfig.from_pretrained(
        resolved_config_name, local_files_only=local_only
    )
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_tokenizer_name, use_fast=True, local_files_only=local_only
    )
    model = Model(resolved_model_path, config=config, local_files_only=local_only)
    return config, tokenizer, model




## === cell 5
def prepare_test_features(args, example, tokenizer):
    example["question"] = example["question"].lstrip()

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
        feature["example_id"] = example["id"]
        feature["context"] = example["context"]
        feature["question"] = example["question"]
        feature["input_ids"] = tokenized_example["input_ids"][i]
        feature["attention_mask"] = tokenized_example["attention_mask"][i]
        feature["offset_mapping"] = tokenized_example["offset_mapping"][i]
        feature["sequence_ids"] = [
            0 if s is None else s for s in tokenized_example.sequence_ids(i)
        ]
        features.append(feature)
    return features




## === cell 6
import collections


def postprocess_qa_predictions(
    examples, features, raw_predictions, n_best_size=20, max_answer_length=30
):
    all_start_logits, all_end_logits = raw_predictions

    example_id_to_index = {str(k): i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        ex_id = str(feature["example_id"])
        if ex_id in example_id_to_index:
            features_per_example[example_id_to_index[ex_id]].append(i)

    predictions = collections.OrderedDict()
    print(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )

    for example_index, example in examples.iterrows():
        feature_indices = features_per_example[example_index]
        valid_answers = []

        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            offset_mapping = features[feature_index]["offset_mapping_ctx"]
            cls_index = features[feature_index]["cls_index"]

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
                    valid_answers.append(
                        {
                            "score": float(
                                start_logits[start_index] + end_logits[end_index]
                            ),
                            "text": context[start_char:end_char],
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                0
            ]
        else:
            best_answer = {"text": "", "score": 0.0}

        predictions[str(example["id"])] = best_answer["text"]

    return predictions




## === cell 7
test = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/test.csv")
test["id"] = test["id"].astype(str)

test["context"] = test["context"].astype(str).str.split().str.join(" ")
test["question"] = test["question"].astype(str).str.split().str.join(" ")

args = Config()

_cfg, tokenizer, _tmp_model = make_model(args)
del _tmp_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

tokenized = tokenizer(
    test["question"].tolist(),
    test["context"].tolist(),
    truncation="only_second",
    max_length=args.max_seq_length,
    stride=args.doc_stride,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding="max_length",
)

overflow_to_sample = tokenized["overflow_to_sample_mapping"]

test_features = []
for i in range(len(tokenized["input_ids"])):
    sample_idx = overflow_to_sample[i]
    feat = {
        "example_id": str(test.at[sample_idx, "id"]),
        "context": test.at[sample_idx, "context"],
        "question": test.at[sample_idx, "question"],
        "input_ids": tokenized["input_ids"][i],
        "attention_mask": tokenized["attention_mask"][i],
        "offset_mapping": tokenized["offset_mapping"][i],
    }
    seq_ids = tokenized.sequence_ids(i)
    feat["sequence_ids"] = [0 if s is None else s for s in seq_ids]
    test_features.append(feat)

cls_id = tokenizer.cls_token_id
for f in test_features:
    seq = f["sequence_ids"]
    om = f["offset_mapping"]
    f["offset_mapping_ctx"] = [(o if seq[k] == 1 else None) for k, o in enumerate(om)]
    f["cls_index"] = f["input_ids"].index(cls_id) if cls_id in f["input_ids"] else 0

test_dataset = DatasetRetriever(test_features, mode="test")
nw = optimal_num_of_loader_workers()

test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.eval_batch_size,
    sampler=SequentialSampler(test_dataset),
    num_workers=nw,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    collate_fn=collate_test_fn,
    persistent_workers=True if nw > 0 else False,
)

print("Num test examples:", len(test), "Num test features:", len(test_features))




## === cell 8
def _load_checkpoint_state_dict(checkpoint_path: str):
    if checkpoint_path in (None, "", "__no_checkpoint__"):
        return None
    if not os.path.exists(checkpoint_path):
        return None
    state = torch.load(checkpoint_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if not isinstance(state, dict):
        return None

    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    return new_state


def _get_model_dir_for_checkpoint(checkpoint_path: str) -> str:
    if checkpoint_path in (None, "", "__no_checkpoint__"):
        return None
    if not isinstance(checkpoint_path, str):
        return None
    if os.path.isdir(checkpoint_path):
        return checkpoint_path
    d = os.path.dirname(checkpoint_path)
    if os.path.exists(os.path.join(d, "config.json")):
        return d
    parent = os.path.dirname(d)
    if os.path.exists(os.path.join(parent, "config.json")):
        return parent
    return None


def get_predictions(checkpoint_path):
    ckpt_model_dir = _get_model_dir_for_checkpoint(checkpoint_path)

    if ckpt_model_dir is not None:
        local_only = True
        resolved_config_name = ckpt_model_dir
        resolved_tokenizer_name = ckpt_model_dir
        resolved_model_path = ckpt_model_dir
    else:
        local_only = True
        try:
            resolved_config_name = _resolve_existing_model_path(Config().config_name)
            resolved_tokenizer_name = _resolve_existing_model_path(
                Config().tokenizer_name
            )
            resolved_model_path = _resolve_existing_model_path(
                Config().model_name_or_path
            )
        except FileNotFoundError:
            local_only = False
            resolved_config_name = "xlm-roberta-base"
            resolved_tokenizer_name = "xlm-roberta-base"
            resolved_model_path = "xlm-roberta-base"

    config = AutoConfig.from_pretrained(
        resolved_config_name, local_files_only=local_only
    )

    _ = AutoTokenizer.from_pretrained(
        resolved_tokenizer_name, use_fast=True, local_files_only=local_only
    )

    model = Model(resolved_model_path, config=config, local_files_only=local_only)

    sd = _load_checkpoint_state_dict(checkpoint_path)
    if sd is not None:
        model.load_state_dict(sd, strict=False)

    model.to(DEVICE)
    model.eval()

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        except Exception:
            pass

    start_logits = []
    end_logits = []

    with torch.inference_mode():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            outputs_start, outputs_end = model(input_ids, attention_mask)
            start_logits.append(outputs_start.detach().cpu().numpy())
            end_logits.append(outputs_end.detach().cpu().numpy())

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.vstack(start_logits), np.vstack(end_logits)


def _find_checkpoint_by_suffix(suffix_path: str):
    if os.path.exists(suffix_path):
        return suffix_path

    suffix = suffix_path
    for prefix in ("../input", "/kaggle/input"):
        if suffix.startswith(prefix):
            rel = os.path.relpath(suffix, prefix)
        else:
            rel = suffix.replace("../input/", "").replace("/kaggle/input/", "")
        candidate = os.path.join(prefix, rel)
        if os.path.exists(candidate):
            return candidate

    fname = os.path.basename(suffix_path)
    for base in ("../input", "/kaggle/input"):
        if not os.path.isdir(base):
            continue
        try:
            for root, dirs, files in os.walk(base):
                if fname in files:
                    full = os.path.join(root, fname)
                    return full
        except Exception:
            pass
    return None


def _auto_discover_chaii_checkpoints(max_to_use: int = 8):
    """
    Change (score-critical): prioritize likely fold checkpoints and avoid grabbing unrelated
    model bins. Using the correct fine-tuned checkpoints is the main lever to move score
    from ~0.0 toward the target.
    """
    found = []
    target_names = {"pytorch_model.bin", "checkpoint.bin"}

    for base in ("../input", "/kaggle/input"):
        if not os.path.isdir(base):
            continue
        try:
            for root, dirs, files in os.walk(base):
                fset = set(files)
                hit = None
                if "pytorch_model.bin" in fset:
                    hit = "pytorch_model.bin"
                elif "checkpoint.bin" in fset:
                    hit = "checkpoint.bin"
                if hit is None:
                    continue

                rlow = root.lower()
                if ("chaii" not in rlow) and ("abhishek" not in rlow):
                    continue
                if (
                    ("checkpoint-fold" not in rlow)
                    and ("fold" not in rlow)
                    and ("output" not in rlow)
                ):
                    continue

                found.append(os.path.join(root, hit))
        except Exception:
            continue

    found = sorted(list(dict.fromkeys(found)))
    if len(found) > max_to_use:
        found = found[:max_to_use]

    return [(p, 1.0) for p in found]




## === cell 9
checkpoint_paths = [
    ("../input/chaii-abhishek-fold-0/output/checkpoint-fold-0/pytorch_model.bin", 0.05),
    ("../input/chaii-abhishek-fold-1/output/checkpoint-fold-1/pytorch_model.bin", 0.10),
    ("../input/chaii-abhishek-fold-2/output/checkpoint-fold-2/pytorch_model.bin", 0.30),
    ("../input/chaii-abhishek-fold-3/output/checkpoint-fold-3/pytorch_model.bin", 0.05),
    (
        "../input/chaii-abhishek-model-2-epochs-10folds/output/checkpoint-fold-5/pytorch_model.bin",
        0.05,
    ),
    (
        "../input/k/abhiram4572/chaii-abhishek-model-2-epochs-10folds/output/checkpoint-fold-7/pytorch_model.bin",
        0.10,
    ),
    (
        "../input/k/vineethakki/chaii-abhishek-model-2-epochs-10folds/output/checkpoint-fold-8/pytorch_model.bin",
        0.30,
    ),
    (
        "../input/k/vineethakkinapalli/chaii-abhishek-model-2-epochs-10folds/output/checkpoint-fold-9/pytorch_model.bin",
        0.05,
    ),
]

resolved_checkpoint_paths = []
for p, w in checkpoint_paths:
    rp = _find_checkpoint_by_suffix(p)
    if rp is not None and os.path.exists(rp):
        resolved_checkpoint_paths.append((rp, w))

available = [(p, w) for p, w in resolved_checkpoint_paths if os.path.exists(p)]
print(f"Available checkpoints (resolved): {len(available)}/{len(checkpoint_paths)}")
if len(available) > 0:
    print("Example resolved checkpoint path:", available[0][0])

if len(available) == 0:
    auto_ckpts = _auto_discover_chaii_checkpoints(max_to_use=8)
    if len(auto_ckpts) > 0:
        available = auto_ckpts
        print(f"Auto-discovered checkpoints: {len(available)}")
        print("Example auto checkpoint path:", available[0][0])

test_data = test.copy()
test_data["PredictionString"] = ""

if len(available) == 0:
    start_logits, end_logits = get_predictions("__no_checkpoint__")
else:
    wsum = sum(w for _, w in available)
    start_logits = None
    end_logits = None
    for p, w in available:
        sl, el = get_predictions(p)
        coef = w / wsum
        start_logits = sl * coef if start_logits is None else start_logits + sl * coef
        end_logits = el * coef if end_logits is None else end_logits + el * coef

fin_preds = postprocess_qa_predictions(test, test_features, (start_logits, end_logits))

submission = []
for pid, pred in fin_preds.items():
    pred = " ".join(str(pred).split())
    pred = pred.strip(punctuation)
    submission.append((str(pid), pred))

sample = pd.DataFrame(submission, columns=["id", "PredictionString"])
sample["id"] = sample["id"].astype(str)

test_data = pd.merge(left=test, right=sample, on="id", how="left")
test_data["PredictionString"] = test_data["PredictionString"].fillna("").astype(str)



## === cell 10
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
    while any([pred.startswith(y) for y in bad_starts]) and len(pred) > 0:
        pred = pred[1:]
    while any([pred.endswith(y) for y in bad_endings]) and len(pred) > 0:
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

    cleaned_preds.append(pred)

test_data["PredictionString"] = cleaned_preds

out_df = test_data[["id", "PredictionString"]].copy()
out_df["id"] = out_df["id"].astype(str)
out_df = out_df.merge(test[["id"]], on="id", how="right")
out_df["PredictionString"] = out_df["PredictionString"].fillna("").astype(str)

out_df.to_csv("submission.csv", index=False, encoding="utf-8")
print("Wrote submission.csv with shape:", out_df.shape)
print(out_df.head())



## === cell 11
test_data[["id", "PredictionString"]].head()
