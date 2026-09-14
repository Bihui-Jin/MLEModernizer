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

3.9

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

0.7263691425323486

# 6. Current score

0.64777

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the root cause of the failures: your notebook depends on an external Kaggle dataset (`/kaggle/input/chaii-xlmr-5-fold/...`) that isn’t present, so tokenizers/models/checkpoints can’t be loaded. To keep the core inference/postprocessing logic intact while making it run end-to-end, I add a safe fallback that uses a public HF model available in the Kaggle environment cache (`/kaggle/input`) if present, otherwise it still run by generating a valid (but low-scoring) empty-prediction submission rather than crashing. I also guard downstream cells so `test_data` is always defined and `submission.csv` is always written with the correct columns and UTF-8 encoding. No training loop/architecture changes are introduced; the only adjustments are path resolution, robust loading, and fail-safe submission generation.'
- What this solution (achieved 0.0) has done: 'Your score is 0.0 because the code usually falls back to an all-empty submission when it can’t find the external fold checkpoints under `/kaggle/input/chaii-xlmr-5-fold/output/`. To move toward the 0.726 target with minimal logic changes, I (1) allow loading a local QA model directly from any available HF model directory under `/kaggle/input` (not just that specific dataset), and (2) if fold checkpoints are missing, run a single-pass inference with the base pretrained QA head instead of producing empty strings. This preserves your exact tokenization, sliding-window feature creation, and postprocess selection logic; it just removes the “empty submission” failure mode. The output stays a valid `submission.csv` with the required columns and row alignment.'
- What this solution (achieved 0.0) has done: 'Your score is 0.0 because the current pipeline often can’t load the intended local model/checkpoints and then falls back to empty strings, which yields near-zero Jaccard. To move toward the 0.726 target with minimal semantic change, I keep your exact tokenization, windowing, model head, and post-processing, but (1) switch to loading an actual QA model head via `AutoModelForQuestionAnswering` when fold checkpoints aren’t available, and (2) remove `local_files_only=True` so the notebook can download the base model if it isn’t already cached (this is the smallest change that turns “empty submission” into real answers). I also make checkpoint loading robust to common state-dict key prefixes so it won’t silently fail due to key mismatches. The code still writes a valid `submission.csv` with the required columns and alignment.'
- What this solution (achieved 0.64777) has done: 'I remove the biggest Python-side bottlenecks without changing the model, logits, or post-processing semantics: (1) tokenize the entire test set in one batched fast-tokenizer call (instead of 7k per-row calls), (2) stop recreating the HF config/tokenizer and rebuilding the base model five times (load once, then only reload fold weights), and (3) avoid per-item tensor construction in `__getitem__` by pre-converting feature arrays to contiguous NumPy arrays and using zero-copy `torch.from_numpy` in a custom `collate_fn`. I also preallocate the full logits arrays during inference to avoid repeated `np.vstack` and reduce CPU/GPU sync overhead, while keeping identical evaluation and averaging. All changes are deterministic and preserve the exact core logic (same architecture, same checkpoints, same feature generation, same postprocess).'
- What this solution (achieved 0.64777) has done: 'The crash is coming from an upstream `protobuf`/`sentencepiece` compatibility issue triggered when `transformers` tries to load the fast tokenizer; it manifests as `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. To keep your core model/inference/postprocess logic intact and improve stability, I (1) force the pure-Python protobuf implementation (a common Kaggle fix) and (2) add a robust tokenizer loader that falls back to the slow tokenizer if the fast one fails. This is score-neutral when fast tokenizers work, but prevents the runtime failure so you always get a valid submission (and avoids the empty-submission fallback). No architecture, logits averaging, windowing, or postprocessing semantics are changed.'
- What this solution (achieved 0.64777) has done: 'The failure is coming from an incompatible `protobuf` backend being used when `transformers`/tokenizers import sentencepiece/protobuf, which triggers `MessageFactory.GetPrototype` errors; setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` *inside* the notebook is sometimes too late because protobuf may already be imported. I add an early, safe runtime guard that forces the pure-Python protobuf implementation **before** importing `transformers` (and re-exec the process once if needed), and keep your existing “slow-tokenizer fallback” intact. This is score-neutral (it doesn’t change logits/postprocessing) but makes the pipeline run end-to-end reliably and always write `submission.csv`. I also add a tiny defensive check so the inference path doesn’t proceed with an uninitialized dataloader if tokenization fails.'
- What this solution (achieved 0.64777) has done: 'Your current pipeline already runs and writes a valid `submission.csv`, but your score is “Not yielded”, so the most likely blocker is a runtime issue in Kaggle (download-disabled model fallback, tokenizer/model mismatch, or the process re-exec causing instability). I make the smallest changes that (1) remove the forced `os.execv` re-exec (keeps protobuf fix but avoids Kaggle execution issues), and (2) ensure the tokenizer used for feature creation always matches the model used for inference (especially in the pretrained QA-head fallback), which should reliably produce non-empty predictions and move the score upward toward your target. I won’t change your architecture, sliding-window feature logic, or post-processing; the only semantic change is preventing an accidental mismatch between tokenization and the model generating logits. Finally, I add a strict sanity check that the number/order of predictions matches the sample submission ids before writing.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
from torch.utils.data.distributed import DistributedSampler

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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def optimal_num_of_loader_workers():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus * 4) if num_gpus else max(1, num_cpus - 1)
    return max(1, min(4, optimal_value))


print(f"Apex AMP Installed :: {APEX_INSTALLED}")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def resolve_kaggle_input_path(p: str) -> str:
    if p is None:
        return p
    p = str(p)
    if p.startswith("../input/"):
        p2 = p.replace("../input/", "/kaggle/input/")
        return p2
    if p.startswith("../") and "/input/" in p:
        idx = p.find("/input/")
        return "/kaggle" + p[idx:]
    return p


def find_first_existing_path(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def list_input_dirs(root="/kaggle/input"):
    try:
        return [os.path.join(root, d) for d in os.listdir(root)]
    except Exception:
        return []


def infer_local_model_dir(preferred=None):
    """
    Try to locate a local transformers model directory inside /kaggle/input.
    Return a local DIRECTORY path if found; otherwise return None.
    """
    preferred = resolve_kaggle_input_path(preferred) if preferred else None
    if preferred and os.path.exists(preferred):
        return preferred

    common = [
        "/kaggle/input/chaii-xlmr-5-fold/xlm-roberta-large-squad-v2",
        "/kaggle/input/chaii-xlmr-5-fold/xlm-roberta-large",
        "/kaggle/input/xlm-roberta-large-squad2",
        "/kaggle/input/xlm-roberta-large-squad-v2",
        "/kaggle/input/xlm-roberta-large",
        "/kaggle/input/xlm-roberta-base",
    ]
    p = find_first_existing_path(common)
    if p:
        return p

    for d in list_input_dirs("/kaggle/input"):
        scan = [d]
        try:
            scan.extend([os.path.join(d, x) for x in os.listdir(d)])
        except Exception:
            pass
        for sd in scan:
            cfg = os.path.join(sd, "config.json")
            if not os.path.exists(cfg):
                continue
            w1 = os.path.join(sd, "pytorch_model.bin")
            w2 = os.path.join(sd, "model.safetensors")
            if not (os.path.exists(w1) or os.path.exists(w2)):
                continue
            tok1 = os.path.join(sd, "tokenizer.json")
            tok2 = os.path.join(sd, "sentencepiece.bpe.model")
            tok3 = os.path.join(sd, "spiece.model")
            if not (
                os.path.exists(tok1) or os.path.exists(tok2) or os.path.exists(tok3)
            ):
                continue
            return sd

    return None


def _safe_load_state_dict(model, state):
    """
    Tolerate different checkpoint formats/prefixes to avoid failing to load valid fold weights.
    """
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Loaded checkpoint is not a state_dict-like object.")

    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    return missing, unexpected


DEFAULT_HF_MODEL_ID = "deepset/xlm-roberta-large-squad2"


def resolve_model_name_or_path(maybe_path_or_id: str) -> str:
    maybe_path_or_id = resolve_kaggle_input_path(maybe_path_or_id)
    local_dir = infer_local_model_dir(maybe_path_or_id)
    if local_dir is not None:
        return local_dir
    if (
        isinstance(maybe_path_or_id, str)
        and maybe_path_or_id.startswith("/")
        and (not os.path.exists(maybe_path_or_id))
    ):
        return DEFAULT_HF_MODEL_ID
    if maybe_path_or_id is None or str(maybe_path_or_id).strip() == "":
        return DEFAULT_HF_MODEL_ID
    return maybe_path_or_id


def load_tokenizer_robust(name_or_path: str):
    name_or_path = resolve_model_name_or_path(name_or_path)
    try:
        return AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    except Exception as e_fast:
        print(
            "WARNING: fast tokenizer load failed, retrying with slow tokenizer:",
            repr(e_fast),
        )
        return AutoTokenizer.from_pretrained(name_or_path, use_fast=False)


fix_all_seeds(2021)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True




## === cell 1
class Config:
    model_type = "xlm_roberta"

    model_name_or_path = "/kaggle/input/chaii-xlmr-5-fold/xlm-roberta-large-squad-v2"
    config_name = "/kaggle/input/chaii-xlmr-5-fold/xlm-roberta-large-squad-v2"
    tokenizer_name = "/kaggle/input/chaii-xlmr-5-fold/xlm-roberta-large-squad-v2"

    fp16 = True if APEX_INSTALLED else False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2

    max_seq_length = 384
    doc_stride = 128

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
        return self.features[item]


def make_test_collate_fn():
    def _collate(batch):
        input_ids = torch.from_numpy(np.stack([b["input_ids"] for b in batch], axis=0))
        attention_mask = torch.from_numpy(
            np.stack([b["attention_mask"] for b in batch], axis=0)
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": [b["offset_mapping"] for b in batch],
            "sequence_ids": [b["sequence_ids"] for b in batch],
            "id": [b["example_id"] for b in batch],
            "context": [b["context"] for b in batch],
            "question": [b["question"] for b in batch],
        }

    return _collate




## === cell 3
class Model(nn.Module):
    def __init__(self, modelname_or_path, config):
        super(Model, self).__init__()
        self.config = config
        self.xlm_roberta = AutoModel.from_pretrained(modelname_or_path, config=config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._init_weights(self.qa_outputs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.xlm_roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits




## === cell 4
def make_model(args):
    cfg_path = resolve_model_name_or_path(args.config_name)
    tok_path = resolve_model_name_or_path(args.tokenizer_name)
    mdl_path = resolve_model_name_or_path(args.model_name_or_path)

    config = AutoConfig.from_pretrained(cfg_path)
    tokenizer = load_tokenizer_robust(tok_path)
    model = Model(mdl_path, config=config)
    return config, tokenizer, model


def make_qa_model_fallback(args):
    """
    When fold checkpoints aren't present, use a real QA head (AutoModelForQuestionAnswering).

    Change (score improvement + correctness): return BOTH tokenizer and the resolved model path,
    so we can ensure test features are built with the same tokenizer family as the model used
    to produce logits. This avoids tokenizer/model mismatch that can severely degrade Jaccard.
    """
    resolved = resolve_model_name_or_path(args.model_name_or_path)
    tokenizer = load_tokenizer_robust(resolved)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(resolved)
    return resolved, tokenizer, qa_model




## === cell 5
def prepare_test_features_batched(args, df, tokenizer):
    questions = [q.lstrip() for q in df["question"].tolist()]
    contexts = df["context"].tolist()
    ids = df["id"].tolist()

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=args.max_seq_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized["overflow_to_sample_mapping"]
    features = []
    input_ids_all = tokenized["input_ids"]
    attention_mask_all = tokenized["attention_mask"]
    offset_mapping_all = tokenized["offset_mapping"]
    for i in range(len(input_ids_all)):
        sample_idx = sample_mapping[i]
        feature = {
            "example_id": ids[sample_idx],
            "context": contexts[sample_idx],
            "question": questions[sample_idx],
            "input_ids": np.asarray(input_ids_all[i], dtype=np.int64),
            "attention_mask": np.asarray(attention_mask_all[i], dtype=np.int64),
            "offset_mapping": offset_mapping_all[i],
            "sequence_ids": [0 if x is None else x for x in tokenized.sequence_ids(i)],
        }
        features.append(feature)
    return features




## === cell 6
import collections


def postprocess_qa_predictions(
    examples, features, raw_predictions, tokenizer, n_best_size=20, max_answer_length=30
):
    all_start_logits, all_end_logits = raw_predictions

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

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

            sequence_ids = features[feature_index]["sequence_ids"]
            context_index = 1

            offset_mapping = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(features[feature_index]["offset_mapping"])
            ]

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

        predictions[example["id"]] = best_answer["text"]

    return predictions




## === cell 7
test_path = "/kaggle/input/chaii-hindi-and-tamil-question-answering/test.csv"
test = pd.read_csv(test_path)

test["context"] = test["context"].apply(lambda x: " ".join(str(x).split()))
test["question"] = test["question"].apply(lambda x: " ".join(str(x).split()))

args = Config()

tokenizer_name = resolve_model_name_or_path(args.tokenizer_name)

CAN_PREDICT = True
try:
    tokenizer = load_tokenizer_robust(tokenizer_name)
except Exception as e:
    print("WARNING: tokenizer load failed:", repr(e))
    CAN_PREDICT = False
    tokenizer = None

test_features = []
if CAN_PREDICT:
    try:
        test_features = prepare_test_features_batched(
            args, test[["id", "context", "question"]], tokenizer
        )
    except Exception as e:
        print("WARNING: feature preparation failed:", repr(e))
        CAN_PREDICT = False
        test_features = []

test_dataset = DatasetRetriever(test_features, mode="test") if CAN_PREDICT else None
test_dataloader = (
    DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        sampler=SequentialSampler(test_dataset),
        num_workers=optimal_num_of_loader_workers(),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=make_test_collate_fn(),
    )
    if CAN_PREDICT
    else None
)



## === cell 8
base_model = "/kaggle/input/chaii-xlmr-5-fold/output/"

_GLOBALS = {}


def _get_or_build_base():
    key = "base"
    if key in _GLOBALS:
        return _GLOBALS[key]
    config, tok_local, model = make_model(Config())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    _GLOBALS[key] = (config, tok_local, model, device)
    return _GLOBALS[key]


def _predict_with_model(model, device, dataloader):
    n_features = len(dataloader.dataset)
    start_logits = np.empty((n_features, Config.max_seq_length), dtype=np.float32)
    end_logits = np.empty((n_features, Config.max_seq_length), dtype=np.float32)

    idx = 0
    for batch in dataloader:
        bs = batch["input_ids"].shape[0]
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            outputs_start, outputs_end = model(input_ids, attention_mask)
            start_logits[idx : idx + bs] = outputs_start.detach().cpu().numpy()
            end_logits[idx : idx + bs] = outputs_end.detach().cpu().numpy()
        idx += bs
    return start_logits, end_logits


def get_predictions(checkpoint_path):
    config, tok_local, model, device = _get_or_build_base()

    ckpt_full = os.path.join(base_model, checkpoint_path)
    if not os.path.exists(ckpt_full):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_full}\n"
            f"Attach the Kaggle dataset 'chaii-xlmr-5-fold' (or update base_model/checkpoint paths)."
        )

    state = torch.load(ckpt_full, map_location="cpu")
    missing, unexpected = _safe_load_state_dict(model, state)
    if len(unexpected) > 0:
        print(
            f"WARNING: unexpected keys while loading {checkpoint_path}: {unexpected[:5]} ..."
        )
    if len(missing) > 0:
        print(
            f"WARNING: missing keys while loading {checkpoint_path}: {missing[:5]} ..."
        )

    return _predict_with_model(model, device, test_dataloader)


def get_predictions_pretrained_qa_head_and_retokenize_if_needed():
    """
    Change (score improvement): if we fall back to a pretrained QA head, ensure the test features
    (input_ids/offset_mapping) were created with the SAME tokenizer as that QA model.
    Otherwise offsets/token indices can misalign and produce poor/empty answer spans.
    """
    resolved, tok_local, qa_model = make_qa_model_fallback(Config())

    global tokenizer, test_features, test_dataset, test_dataloader
    need_retokenize = False
    try:
        if tokenizer is None:
            need_retokenize = True
        else:
            if hasattr(tokenizer, "vocab_size") and hasattr(tok_local, "vocab_size"):
                if int(tokenizer.vocab_size) != int(tok_local.vocab_size):
                    need_retokenize = True
    except Exception:
        need_retokenize = True

    if need_retokenize:
        tokenizer = tok_local
        test_features = prepare_test_features_batched(
            args, test[["id", "context", "question"]], tokenizer
        )
        test_dataset = DatasetRetriever(test_features, mode="test")
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            sampler=SequentialSampler(test_dataset),
            num_workers=optimal_num_of_loader_workers(),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=make_test_collate_fn(),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qa_model.to(device)
    qa_model.eval()

    n_features = len(test_dataloader.dataset)
    start_logits = np.empty((n_features, Config.max_seq_length), dtype=np.float32)
    end_logits = np.empty((n_features, Config.max_seq_length), dtype=np.float32)

    idx = 0
    for batch in test_dataloader:
        bs = batch["input_ids"].shape[0]
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            out = qa_model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits[idx : idx + bs] = out.start_logits.detach().cpu().numpy()
            end_logits[idx : idx + bs] = out.end_logits.detach().cpu().numpy()
        idx += bs

    del qa_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return start_logits, end_logits


test_data = test.copy()
test_data["PredictionString"] = ""

if CAN_PREDICT and test_dataloader is not None and len(test_features) > 0:
    try:
        start_logits1, end_logits1 = get_predictions(
            "checkpoint-fold-0/pytorch_model.bin"
        )
        start_logits2, end_logits2 = get_predictions(
            "checkpoint-fold-1/pytorch_model.bin"
        )
        start_logits3, end_logits3 = get_predictions(
            "checkpoint-fold-2/pytorch_model.bin"
        )
        start_logits4, end_logits4 = get_predictions(
            "checkpoint-fold-3/pytorch_model.bin"
        )
        start_logits5, end_logits5 = get_predictions(
            "checkpoint-fold-4/pytorch_model.bin"
        )

        start_logits = (
            start_logits1
            + start_logits2
            + start_logits3
            + start_logits4
            + start_logits5
        ) / 5.0
        end_logits = (
            end_logits1 + end_logits2 + end_logits3 + end_logits4 + end_logits5
        ) / 5.0

        fin_preds = postprocess_qa_predictions(
            test, test_features, (start_logits, end_logits), tokenizer=tokenizer
        )

        submission = []
        for p1, p2 in fin_preds.items():
            p2 = " ".join(str(p2).split())
            submission.append((p1, p2))

        sample = pd.DataFrame(submission, columns=["id", "PredictionString"])
        test_data = pd.merge(left=test, right=sample, on="id", how="left")
        test_data["PredictionString"] = test_data["PredictionString"].fillna("")
    except FileNotFoundError as e:
        print("WARNING:", str(e))
        print("Falling back to pretrained QA-head inference (no fold checkpoints).")
        try:
            start_logits, end_logits = (
                get_predictions_pretrained_qa_head_and_retokenize_if_needed()
            )

            fin_preds = postprocess_qa_predictions(
                test, test_features, (start_logits, end_logits), tokenizer=tokenizer
            )

            submission = []
            for p1, p2 in fin_preds.items():
                p2 = " ".join(str(p2).split())
                submission.append((p1, p2))

            sample = pd.DataFrame(submission, columns=["id", "PredictionString"])
            test_data = pd.merge(left=test, right=sample, on="id", how="left")
            test_data["PredictionString"] = test_data["PredictionString"].fillna("")
        except Exception as e2:
            print("WARNING: Pretrained QA-head inference failed:", repr(e2))
            print(
                "Proceeding with empty PredictionString for all rows (valid submission)."
            )
            test_data = test.copy()
            test_data["PredictionString"] = ""
else:
    test_data = test.copy()
    test_data["PredictionString"] = ""

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

sub = test_data[["id", "PredictionString"]].copy()

sample_sub_path = (
    "/kaggle/input/chaii-hindi-and-tamil-question-answering/sample_submission.csv"
)
sample_sub = pd.read_csv(sample_sub_path)
sub = sample_sub[["id"]].merge(sub, on="id", how="left")
sub["PredictionString"] = sub["PredictionString"].fillna("")

assert (
    sub.shape[0] == sample_sub.shape[0]
), "Submission row count mismatch vs sample_submission."
assert (
    sub["id"].tolist() == sample_sub["id"].tolist()
), "Submission id order mismatch vs sample_submission."

sub.to_csv("submission.csv", index=False, encoding="utf-8")
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv saved at:", os.path.abspath("submission.csv"))

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
