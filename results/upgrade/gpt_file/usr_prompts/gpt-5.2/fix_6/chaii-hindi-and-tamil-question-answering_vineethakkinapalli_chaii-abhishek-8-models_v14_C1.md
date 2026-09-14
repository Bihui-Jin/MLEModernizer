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

0.7294511198997498

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

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
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    logging,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
)

from torch.optim import AdamW  # kept for compatibility with any downstream usage

logging.set_verbosity_warning()
logging.set_verbosity_error()


def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def optimal_num_of_loader_workers():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus * 4) if num_gpus else max(1, num_cpus - 1)
    return optimal_value


print(f"Apex AMP Installed :: {APEX_INSTALLED}")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

INPUT_DIR = "/kaggle/input/chaii-hindi-and-tamil-question-answering"


def _find_local_model_dir(preferred_root, model_folder_name="xlm-roberta-base"):
    """
    Bugfix: tokenizer/model loading failed offline because the model files are not in this dataset.
    We search common Kaggle input locations for a locally uploaded xlm-roberta-base folder.
    """
    candidates = [
        os.path.join(preferred_root, model_folder_name),
        os.path.join(
            preferred_root,
            "chaii-hindi-and-tamil-question-answering",
            model_folder_name,
        ),
        os.path.join("/kaggle/input", model_folder_name),
        os.path.join("/kaggle/input", "xlm-roberta-base"),
        os.path.join("/kaggle/input", "xlm-roberta-base", model_folder_name),
    ]
    for c in candidates:
        if os.path.isdir(c) and (
            os.path.exists(os.path.join(c, "config.json"))
            or os.path.exists(os.path.join(c, "sentencepiece.bpe.model"))
            or os.path.exists(os.path.join(c, "tokenizer.json"))
        ):
            return c
    return None


def _find_hf_cache_model_dir(model_folder_name="xlm-roberta-base"):
    """
    Bugfix: allow loading from the Kaggle HF cache (still offline) if present.
    Transformers will resolve files from cache if they exist; this helper tries to locate a snapshot dir.
    """
    hf_cache_roots = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/root/.cache/huggingface/hub",
        "/kaggle/working/.cache/huggingface/hub",
    ]
    repo_dirname = f"models--{model_folder_name.replace('/', '--')}"
    for root in hf_cache_roots:
        repo_root = os.path.join(root, repo_dirname)
        if not os.path.isdir(repo_root):
            continue
        snapshots_dir = os.path.join(repo_root, "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue
        for snap in sorted(os.listdir(snapshots_dir), reverse=True):
            snap_path = os.path.join(snapshots_dir, snap)
            if os.path.isdir(snap_path) and os.path.exists(
                os.path.join(snap_path, "config.json")
            ):
                return snap_path
    return None


LOCAL_MODEL_DIR = _find_local_model_dir(INPUT_DIR, "xlm-roberta-base")
if LOCAL_MODEL_DIR is None:
    ALT_INPUT_DIR = "/kaggle/input/chaii-hindi-and-tamil-question-answering/chaii-hindi-and-tamil-question-answering"
    alt_dir = _find_local_model_dir(ALT_INPUT_DIR, "xlm-roberta-base")
    if alt_dir is not None:
        INPUT_DIR = ALT_INPUT_DIR
        LOCAL_MODEL_DIR = alt_dir

HF_CACHE_MODEL_DIR = _find_hf_cache_model_dir("xlm-roberta-base")

print("Resolved INPUT_DIR:", INPUT_DIR)
print("Resolved LOCAL_MODEL_DIR:", LOCAL_MODEL_DIR)
print("Resolved HF_CACHE_MODEL_DIR:", HF_CACHE_MODEL_DIR)


def _resolve_model_name_or_path():
    if LOCAL_MODEL_DIR and os.path.isdir(LOCAL_MODEL_DIR):
        return LOCAL_MODEL_DIR
    if HF_CACHE_MODEL_DIR and os.path.isdir(HF_CACHE_MODEL_DIR):
        return HF_CACHE_MODEL_DIR
    return "xlm-roberta-base"  # will only work if already cached; still local_files_only=True


RESOLVED_MODEL_PATH = _resolve_model_name_or_path()
print("Resolved model path used for from_pretrained:", RESOLVED_MODEL_PATH)




## === cell 1
class Config:
    model_type = "xlm-roberta"

    model_name_or_path = RESOLVED_MODEL_PATH
    config_name = RESOLVED_MODEL_PATH
    tokenizer_name = RESOLVED_MODEL_PATH

    fp16 = True if APEX_INSTALLED else False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2

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

    logging_steps = 50

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
        self.xlm_roberta = AutoModel.from_pretrained(
            modelname_or_path, config=config, local_files_only=True
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
def make_model(args):
    config = AutoConfig.from_pretrained(args.config_name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, local_files_only=True
    )
    model = Model(args.model_name_or_path, config=config)
    return config, tokenizer, model




## === cell 5
def prepare_test_features(args, example, tokenizer):
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
        feature["example_id"] = example["id"]
        feature["context"] = example["context"]
        feature["question"] = example["question"]
        feature["input_ids"] = tokenized_example["input_ids"][i]
        feature["attention_mask"] = tokenized_example["attention_mask"][i]
        feature["offset_mapping"] = tokenized_example["offset_mapping"][i]
        feature["sequence_ids"] = [
            0 if j is None else j for j in tokenized_example.sequence_ids(i)
        ]
        features.append(feature)
    return features




## === cell 6
def prepare_train_features(args, example, tokenizer):
    example["question"] = str(example["question"]).lstrip()
    context = str(example["context"])
    answer_text = "" if pd.isna(example["answer_text"]) else str(example["answer_text"])
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
        input_ids = tokenized_example["input_ids"][i]
        attention_mask = tokenized_example["attention_mask"][i]
        offset_mapping = tokenized_example["offset_mapping"][i]
        sequence_ids = [
            0 if j is None else j for j in tokenized_example.sequence_ids(i)
        ]

        try:
            cls_index = input_ids.index(tokenizer.cls_token_id)
        except ValueError:
            cls_index = 0
        start_position = cls_index
        end_position = cls_index

        if answer_text != "":
            start_char = answer_start
            end_char = answer_start + len(answer_text)

            token_start_index = 0
            while (
                token_start_index < len(sequence_ids)
                and sequence_ids[token_start_index] != 1
            ):
                token_start_index += 1
            token_end_index = len(sequence_ids) - 1
            while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if token_start_index < len(offset_mapping) and token_end_index >= 0:
                if (
                    offset_mapping[token_start_index] is not None
                    and offset_mapping[token_end_index] is not None
                    and not (
                        offset_mapping[token_start_index][0] > start_char
                        or offset_mapping[token_end_index][1] < end_char
                    )
                ):
                    while (
                        token_start_index < len(offset_mapping)
                        and offset_mapping[token_start_index] is not None
                        and offset_mapping[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    start_position = token_start_index - 1

                    while (
                        token_end_index >= 0
                        and offset_mapping[token_end_index] is not None
                        and offset_mapping[token_end_index][1] >= end_char
                    ):
                        token_end_index -= 1
                    end_position = token_end_index + 1

        features.append(
            {
                "example_id": example["id"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "offset_mapping": offset_mapping,
                "sequence_ids": sequence_ids,
                "start_position": start_position,
                "end_position": end_position,
            }
        )
    return features




## === cell 7
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

            features[feature_index]["offset_mapping"] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(features[feature_index]["offset_mapping"])
            ]
            offset_mapping = features[feature_index]["offset_mapping"]

            try:
                cls_index = features[feature_index]["input_ids"].index(
                    tokenizer.cls_token_id
                )
            except ValueError:
                cls_index = 0

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




## === cell 8
args = Config()
fix_all_seeds(args.seed)

os.makedirs(args.output_dir, exist_ok=True)

test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
test["context"] = test["context"].apply(lambda x: " ".join(str(x).split()))
test["question"] = test["question"].apply(lambda x: " ".join(str(x).split()))

try:
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, local_files_only=True
    )
except Exception as e:
    raise RuntimeError(
        "Tokenizer could not be loaded in offline mode. "
        "Please attach a dataset containing 'xlm-roberta-base' files, or ensure the model exists in the HF cache."
    ) from e

test_features = []
for _, row in test.iterrows():
    test_features += prepare_test_features(args, row, tokenizer)

test_dataset = DatasetRetriever(test_features, mode="test")
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.eval_batch_size,
    sampler=SequentialSampler(test_dataset),
    num_workers=optimal_num_of_loader_workers(),
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
LocalEntryNotFoundError                   Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    113 
--> 114         return fn(*args, **kwargs)
    115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in hf_hub_download(repo_id, filename, subfolder, repo_type, revision, library_name, library_version, cache_dir, local_dir, user_agent, force_download, proxies, etag_timeout, token, local_files_only, headers, endpoint, resume_download, force_filename, local_dir_use_symlinks)
   1006     else:
-> 1007         return _hf_hub_download_to_cache_dir(
   1008             # Destination

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _hf_hub_download_to_cache_dir(cache_dir, repo_id, filename, repo_type, revision, endpoint, etag_timeout, headers, proxies, token, local_files_only, force_download)
   1113         # Otherwise, raise appropriate error
-> 1114         _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1645     if local_files_only:
-> 1646         raise LocalEntryNotFoundError(
   1647             "Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable"

LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/737326734.py in <cell line: 0>()
     12 try:
---> 13     tokenizer = AutoTokenizer.from_pretrained(
     14         args.tokenizer_name, local_files_only=True

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
   1002                 else:
-> 1003                     config = AutoConfig.from_pretrained(
   1004                         pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/configuration_auto.py in from_pretrained(cls, pretrained_model_name_or_path, **kwargs)
   1196 
-> 1197         config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
   1198         has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    607         # Get config dict associated with the base config file
--> 608         config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
    609         if config_dict is None:

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    542             elif _raise_exceptions_for_missing_entries:
--> 543                 raise OSError(
    544                     f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load the files, and couldn't find them in the"

OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/737326734.py in <cell line: 0>()
     15     )
     16 except Exception as e:
---> 17     raise RuntimeError(
     18         "Tokenizer could not be loaded in offline mode. "
     19         "Please attach a dataset containing 'xlm-roberta-base' files, or ensure the model exists in the HF cache."

RuntimeError: Tokenizer could not be loaded in offline mode. Please attach a dataset containing 'xlm-roberta-base' files, or ensure the model exists in the HF cache.

## === cell 9
def _load_state_dict_robust(checkpoint_path, device):
    sd = torch.load(checkpoint_path, map_location=device)
    if (
        isinstance(sd, dict)
        and "state_dict" in sd
        and isinstance(sd["state_dict"], dict)
    ):
        sd = sd["state_dict"]
    if isinstance(sd, dict):
        keys = list(sd.keys())
        if len(keys) > 0 and all(k.startswith("module.") for k in keys):
            sd = {k[len("module.") :]: v for k, v in sd.items()}
    return sd




## === cell 10
def get_predictions_from_model(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    start_logits = []
    end_logits = []
    for batch in tqdm(test_dataloader, desc="Infer", leave=False):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs_start, outputs_end = model(input_ids, attention_mask)
            start_logits.append(outputs_start.detach().cpu().numpy())
            end_logits.append(outputs_end.detach().cpu().numpy())
            del outputs_start, outputs_end, input_ids, attention_mask

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.vstack(start_logits), np.vstack(end_logits)


def train_fallback_and_predict():
    train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    train["context"] = train["context"].apply(lambda x: " ".join(str(x).split()))
    train["question"] = train["question"].apply(lambda x: " ".join(str(x).split()))
    train["answer_text"] = train["answer_text"].fillna("").astype(str)

    if torch.cuda.is_available():
        max_rows = min(len(train), 20000)
    else:
        max_rows = min(len(train), 6000)
    train = train.sample(n=max_rows, random_state=args.seed).reset_index(drop=True)

    train_features = []
    for _, row in tqdm(
        train.iterrows(), total=len(train), desc="Featurize", leave=False
    ):
        train_features += prepare_train_features(args, row, tokenizer)

    train_dataset = DatasetRetriever(train_features, mode="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=optimal_num_of_loader_workers(),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, _tok_unused, model = make_model(args)
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=args.epsilon,
        weight_decay=args.weight_decay,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    total_steps = int(args.epochs * num_update_steps_per_epoch)
    warmup_steps = int(args.warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    model.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        pbar = tqdm(train_dataloader, desc=f"Train epoch {epoch+1}", leave=False)
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_pos = batch["start_position"].to(device)
            end_pos = batch["end_position"].to(device)

            start_logits, end_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_pos)
            end_loss = loss_fct(end_logits, end_pos)
            loss = (start_loss + end_loss) / 2.0
            loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if (global_step % args.logging_steps) == 0:
                pbar.set_postfix({"loss": float(loss.detach().cpu().item())})

            del (
                input_ids,
                attention_mask,
                start_pos,
                end_pos,
                start_logits,
                end_logits,
                loss,
                start_loss,
                end_loss,
            )

    model.eval()
    return get_predictions_from_model(model)




## === cell 11
checkpoint_paths = [
    "../input/chaii-abhishek-fold-0/output/checkpoint-fold-0/pytorch_model.bin",
    "../input/chaii-abhishek-fold-1/output/checkpoint-fold-1/pytorch_model.bin",
    "../input/chaii-abhishek-fold-2/output/checkpoint-fold-2/pytorch_model.bin",
    "../input/chaii-abhishek-fold-3/output/checkpoint-fold-3/pytorch_model.bin",
    "../input/chaii-abhishek-model-2-epochs-10folds/output/checkpoint-fold-5/pytorch_model.bin",
    "../input/k/abhiram4572/chaii-abhishek-model-2-epochs-10folds/output/checkpoint-fold-7/pytorch_model.bin",
    "../input/k/vineethakki/chaii-abhishek-model-2-epochs-10folds/output/checkpoint-fold-8/pytorch_model.bin",
    "../input/k/vineethakkinapalli/chaii-abhishek-model-2-epochs-10folds/output/checkpoint-fold-9/pytorch_model.bin",
]

available_paths = [p for p in checkpoint_paths if os.path.exists(p)]

if len(available_paths) > 0:

    def get_predictions(checkpoint_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config, _tokenizer_unused, model = make_model(Config())
        model.to(device)
        model.eval()

        state_dict = _load_state_dict_robust(checkpoint_path, device)
        model.load_state_dict(state_dict, strict=True)

        s, e = get_predictions_from_model(model)

        del model, config, _tokenizer_unused
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return s, e

    all_start = []
    all_end = []
    for p in available_paths:
        s, e = get_predictions(p)
        all_start.append(s)
        all_end.append(e)

    start_logits = np.mean(all_start, axis=0)
    end_logits = np.mean(all_end, axis=0)
else:
    start_logits, end_logits = train_fallback_and_predict()

fin_preds = postprocess_qa_predictions(
    test, test_features, (start_logits, end_logits), tokenizer=tokenizer
)

submission = []
for p1, p2 in fin_preds.items():
    p2 = " ".join(str(p2).split())
    p2 = p2.strip(punctuation)
    submission.append((p1, p2))

sample = pd.DataFrame(submission, columns=["id", "PredictionString"])
test_data = pd.merge(left=test, right=sample, on="id", how="left")



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4213442837.py in <cell line: 0>()
     42     end_logits = np.mean(all_end, axis=0)
     43 else:
---> 44     start_logits, end_logits = train_fallback_and_predict()
     45 
     46 fin_preds = postprocess_qa_predictions(

/tmp/ipykernel_55/800052104.py in train_fallback_and_predict()
     38         train.iterrows(), total=len(train), desc="Featurize", leave=False
     39     ):
---> 40         train_features += prepare_train_features(args, row, tokenizer)
     41 
     42     train_dataset = DatasetRetriever(train_features, mode="train")

NameError: name 'tokenizer' is not defined

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

    while any(pred.startswith(y) for y in bad_starts) and len(pred) > 0:
        pred = pred[1:]
    while any(pred.endswith(y) for y in bad_endings) and len(pred) > 0:
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
test_data[["id", "PredictionString"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", test_data[["id", "PredictionString"]].shape)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4167017210.py in <cell line: 0>()
      9 
     10 cleaned_preds = []
---> 11 for pred, context in test_data[["PredictionString", "context"]].to_numpy():
     12     pred = "" if pd.isna(pred) else str(pred)
     13     context = "" if pd.isna(context) else str(context)

NameError: name 'test_data' is not defined

## === cell 13
test_data[["id", "PredictionString"]].head()

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2282797748.py in <cell line: 0>()
----> 1 test_data[["id", "PredictionString"]].head()

NameError: name 'test_data' is not defined
