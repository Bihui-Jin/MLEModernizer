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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0022) has done: 'I speed up inference by (1) tokenizing the whole test set in large batches (instead of per-row Python loops), (2) using a custom `collate_fn` so the DataLoader doesn’t repeatedly convert Python lists to tensors per item, and (3) avoiding repeated, in-place mutation of `features` during post-processing by precomputing per-feature context-only offset mappings and CLS indices once. I also reuse the already-loaded tokenizer/config and enable fast inference settings (`inference_mode`, `torch.compile` when available) without changing model architecture or prediction semantics. These changes remove the main Python overhead hotspots while keeping identical logits, postprocessing logic, and ensembling behavior (up to negligible float differences).'
- What this solution (achieved 0.0022) has done: 'I fix the crash caused by an incompatible protobuf/transformers interaction that manifests as `MessageFactory.GetPrototype` during imports/model loading by forcing Transformers to use the pure-Python protobuf implementation before `transformers` is imported. I also make the fallback model path use a local Kaggle-provided base model (`../input/xlm-roberta-base`) if available to avoid any internet dependency (the current fallback HF id can silently fail offline and tanks score). These are minimal changes that keep your model/post-processing logic intact, but ensure the notebook runs end-to-end and produces `submission.csv` correctly. With checkpoints present, the score should move sharply upward toward the target; if checkpoints are missing, the local base-model fallback is still much better than a broken/empty pipeline.'
- What this solution (achieved 0.0022) has done: 'I fix the `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before* any `transformers` import and by ensuring the related protobuf env vars are set early and consistently. I also make checkpoint/model loading robust in offline Kaggle runs by strictly preferring existing local model directories and avoiding any accidental online fetches. Finally, I ensure the script always reaches the CSV write step and produces a valid `submission.csv` with the exact required columns and row count aligned to `test.csv`. These changes are execution/robustness fixes (and should substantially improve score vs the broken import path), while preserving your model, logits, ensembling, and post-processing logic.'
- What this solution (achieved 0.0022) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before any other imports* (especially anything that can indirectly import `google.protobuf`, like `sentencepiece`/`transformers`). Then I make checkpoint loading robust: if the listed fold checkpoints are missing, the code still run using the base model without errors and always write a valid `submission.csv`. Finally, I ensure the test/features alignment and submission merge cannot silently misalign by enforcing string `id` types and by producing predictions for every test id.'

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
def _resolve_existing_model_path(preferred_path: str) -> str:
    """
    ROBUSTNESS:
    - Prefer existing local paths; avoid any online HF fallback in offline Kaggle.
    - If the preferred dataset isn't mounted, fall back to other common local model dirs.
    """
    if isinstance(preferred_path, str) and os.path.exists(preferred_path):
        return preferred_path

    local_fallbacks = [
        "../input/abhishek-chaii-qa-model",
        "../input/xlm-roberta-base-squad2",
        "../input/xlm-roberta-base",
        "../input/xlm-roberta-large",
    ]
    for p in local_fallbacks:
        if os.path.exists(p):
            return p

    return preferred_path


def make_model(args):
    resolved_config_name = _resolve_existing_model_path(args.config_name)
    resolved_tokenizer_name = _resolve_existing_model_path(args.tokenizer_name)
    resolved_model_path = _resolve_existing_model_path(args.model_name_or_path)

    config = AutoConfig.from_pretrained(resolved_config_name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_tokenizer_name, use_fast=True, local_files_only=True
    )
    model = Model(resolved_model_path, config=config)
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

        predictions[example["id"]] = best_answer["text"]

    return predictions




## === cell 7
test = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/test.csv")
test["id"] = test["id"].astype(str)

test["context"] = test["context"].astype(str).str.split().str.join(" ")
test["question"] = test["question"].astype(str).str.split().str.join(" ")

args = Config()

tokenizer = AutoTokenizer.from_pretrained(
    _resolve_existing_model_path(args.tokenizer_name),
    use_fast=True,
    local_files_only=True,
)

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




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/abhishek-chaii-qa-model'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_55/4025098165.py in <cell line: 0>()
      7 args = Config()
      8 
----> 9 tokenizer = AutoTokenizer.from_pretrained(
     10     _resolve_existing_model_path(args.tokenizer_name),
     11     use_fast=True,

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
    981 
    982         # Next, let's try to use the tokenizer_config file to get the tokenizer class.
--> 983         tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
    984         if "_commit_hash" in tokenizer_config:
    985             kwargs["_commit_hash"] = tokenizer_config["_commit_hash"]

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in get_tokenizer_config(pretrained_model_name_or_path, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, **kwargs)
    813 
    814     commit_hash = kwargs.get("_commit_hash", None)
--> 815     resolved_config_file = cached_file(
    816         pretrained_model_name_or_path,
    817         TOKENIZER_CONFIG_FILE,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    520 
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    521         # Now we try to recover if we can find all files correctly in the cache
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]
    525         if all(file is not None for file in resolved_files):

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    138 ):
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
    142         return resolved_file

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    104         ):
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 
    108             elif arg_name == "token" and arg_value is not None:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    152 
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"
    156             f" '{repo_id}'. Use `repo_type` argument if needed."

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/abhishek-chaii-qa-model'. Use `repo_type` argument if needed.

## === cell 8
def _load_checkpoint_state_dict(checkpoint_path: str):
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


_GLOBAL_CFG = None
_GLOBAL_TOK = None


def _get_global_cfg_tok():
    global _GLOBAL_CFG, _GLOBAL_TOK
    if _GLOBAL_CFG is None or _GLOBAL_TOK is None:
        resolved_config_name = _resolve_existing_model_path(Config().config_name)
        resolved_tokenizer_name = _resolve_existing_model_path(Config().tokenizer_name)
        _GLOBAL_CFG = AutoConfig.from_pretrained(
            resolved_config_name, local_files_only=True
        )
        _GLOBAL_TOK = AutoTokenizer.from_pretrained(
            resolved_tokenizer_name, use_fast=True, local_files_only=True
        )
    return _GLOBAL_CFG, _GLOBAL_TOK


def get_predictions(checkpoint_path):
    config, _tok = _get_global_cfg_tok()
    resolved_model_path = _resolve_existing_model_path(Config().model_name_or_path)
    model = Model(resolved_model_path, config=config)

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

available = [(p, w) for p, w in checkpoint_paths if os.path.exists(p)]
print(f"Available checkpoints: {len(available)}/{len(checkpoint_paths)}")

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
test_data = pd.merge(left=test, right=sample, on="id", how="left")

test_data["PredictionString"] = test_data["PredictionString"].fillna("").astype(str)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/abhishek-chaii-qa-model'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/abhishek-chaii-qa-model'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/3988578980.py in <cell line: 0>()
     26 
     27 if len(available) == 0:
---> 28     start_logits, end_logits = get_predictions("__no_checkpoint__")
     29 else:
     30     wsum = sum(w for _, w in available)

/tmp/ipykernel_55/393441171.py in get_predictions(checkpoint_path)
     38 
     39 def get_predictions(checkpoint_path):
---> 40     config, _tok = _get_global_cfg_tok()
     41     resolved_model_path = _resolve_existing_model_path(Config().model_name_or_path)
     42     model = Model(resolved_model_path, config=config)

/tmp/ipykernel_55/393441171.py in _get_global_cfg_tok()
     28         resolved_config_name = _resolve_existing_model_path(Config().config_name)
     29         resolved_tokenizer_name = _resolve_existing_model_path(Config().tokenizer_name)
---> 30         _GLOBAL_CFG = AutoConfig.from_pretrained(
     31             resolved_config_name, local_files_only=True
     32         )

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/configuration_auto.py in from_pretrained(cls, pretrained_model_name_or_path, **kwargs)
   1195         code_revision = kwargs.pop("code_revision", None)
   1196 
-> 1197         config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
   1198         has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
   1199         has_local_code = "model_type" in config_dict and config_dict["model_type"] in CONFIG_MAPPING

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    606         original_kwargs = copy.deepcopy(kwargs)
    607         # Get config dict associated with the base config file
--> 608         config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
    609         if config_dict is None:
    610             return {}, kwargs

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    688             except Exception:
    689                 # For any other exception, we throw a generic error.
--> 690                 raise OSError(
    691                     f"Can't load the configuration of '{pretrained_model_name_or_path}'. If you were trying to load it"
    692                     " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"

OSError: Can't load the configuration of '../input/abhishek-chaii-qa-model'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '../input/abhishek-chaii-qa-model' is the correct path to a directory containing a config.json file

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
out_df.to_csv("submission.csv", index=False, encoding="utf-8")
print("Wrote submission.csv with shape:", out_df.shape)
print(out_df.head())



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3658966074.py in <cell line: 0>()
      9 
     10 cleaned_preds = []
---> 11 for pred, context in test_data[["PredictionString", "context"]].to_numpy():
     12     pred = "" if pd.isna(pred) else str(pred)
     13     context = "" if pd.isna(context) else str(context)

NameError: name 'test_data' is not defined

## === cell 11
test_data[["id", "PredictionString"]].head()

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2282797748.py in <cell line: 0>()
----> 1 test_data[["id", "PredictionString"]].head()

NameError: name 'test_data' is not defined
