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

0.7344911694526672

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
from tqdm import tqdm
from string import punctuation

import torch
import torch.nn as nn
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
    logging,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
)

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
    optimal_value = min(num_cpus, num_gpus * 4) if num_gpus else max(1, num_cpus - 1)
    return optimal_value


print(f"Apex AMP Installed :: {APEX_INSTALLED}")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
KAGGLE_INPUT_ROOT = "/kaggle/input"


def _resolve_kaggle_path(p: str) -> str:
    """
    BUGFIX: Convert '../input/...' to '/kaggle/input/...'.
    Also, if user provides a relative path that exists under /kaggle/input, resolve it there.
    Keep absolute paths unchanged.
    """
    if p is None:
        return p
    p = str(p)

    if p.startswith("../input/"):
        p = os.path.join(KAGGLE_INPUT_ROOT, p.replace("../input/", "", 1))

    if not os.path.isabs(p) and os.path.exists(os.path.join(KAGGLE_INPUT_ROOT, p)):
        p = os.path.join(KAGGLE_INPUT_ROOT, p)

    return os.path.normpath(p)


def _hf_from_pretrained_kwargs():
    return {"local_files_only": True}


def _is_local_dir(path: str) -> bool:
    try:
        return path is not None and os.path.isdir(path)
    except Exception:
        return False


def _as_hf_path(path: str) -> str:
    if path is None:
        return path
    path = str(path)
    if _is_local_dir(path) and not path.startswith("file://"):
        return "file://" + path
    return path




## === cell 2
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




## === cell 3
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




## === cell 4
class Model(nn.Module):
    def __init__(self, modelname_or_path, config):
        super(Model, self).__init__()
        self.config = config
        self.xlm_roberta = AutoModel.from_pretrained(
            modelname_or_path, config=config, **_hf_from_pretrained_kwargs()
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




## === cell 5
def make_model(args):
    """
    BUGFIX: If the external Kaggle dataset '../input/abhishek-chaii-qa-model' is not attached,
    fall back to a commonly cached public model in Kaggle images to still run end-to-end.
    """
    cfg_path = _resolve_kaggle_path(args.config_name)
    tok_path = _resolve_kaggle_path(args.tokenizer_name)
    mdl_path = _resolve_kaggle_path(args.model_name_or_path)

    all_exist = all(os.path.exists(p) for p in [cfg_path, tok_path, mdl_path])

    if all_exist:
        config = AutoConfig.from_pretrained(
            _as_hf_path(cfg_path), **_hf_from_pretrained_kwargs()
        )
        tokenizer = AutoTokenizer.from_pretrained(
            _as_hf_path(tok_path), use_fast=True, **_hf_from_pretrained_kwargs()
        )
        model = Model(_as_hf_path(mdl_path), config=config)
        return config, tokenizer, model

    fallback_repo = "deepset/xlm-roberta-large-squad2"
    print(
        f"[WARN] Local model dir not found (expected under {cfg_path}). "
        f"Falling back to cached model '{fallback_repo}' (local_files_only=True)."
    )
    config = AutoConfig.from_pretrained(fallback_repo, **_hf_from_pretrained_kwargs())
    tokenizer = AutoTokenizer.from_pretrained(
        fallback_repo, use_fast=True, **_hf_from_pretrained_kwargs()
    )
    model = Model(fallback_repo, config=config)
    return config, tokenizer, model




## === cell 6
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
            0 if x is None else x for x in tokenized_example.sequence_ids(i)
        ]
        features.append(feature)
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

            offset_mapping = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(features[feature_index]["offset_mapping"])
            ]

            input_ids = features[feature_index]["input_ids"]
            try:
                cls_index = input_ids.index(tokenizer.cls_token_id)
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
test = pd.read_csv("/kaggle/input/chaii-hindi-and-tamil-question-answering/test.csv")

test["context"] = test["context"].apply(lambda x: " ".join(str(x).split()))
test["question"] = test["question"].apply(lambda x: " ".join(str(x).split()))

args = Config()
fix_all_seeds(args.seed)

_config, tokenizer, _tmp_model = make_model(args)
del _tmp_model, _config
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

test_features = []
for _, row in tqdm(test.iterrows(), total=len(test), desc="Tokenizing test"):
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
/tmp/ipykernel_55/3484493256.py in <cell line: 0>()
      8 
      9 # BUGFIX: load tokenizer via make_model() to ensure consistent fallback behavior and proper path handling.
---> 10 _config, tokenizer, _tmp_model = make_model(args)
     11 del _tmp_model, _config
     12 gc.collect()

/tmp/ipykernel_55/3532538892.py in make_model(args)
     26         f"Falling back to cached model '{fallback_repo}' (local_files_only=True)."
     27     )
---> 28     config = AutoConfig.from_pretrained(fallback_repo, **_hf_from_pretrained_kwargs())
     29     tokenizer = AutoTokenizer.from_pretrained(
     30         fallback_repo, use_fast=True, **_hf_from_pretrained_kwargs()

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
    665             try:
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,
    669                     configuration_file,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    541             # even when `local_files_only` is True, in which case raising for connections errors only would not make sense)
    542             elif _raise_exceptions_for_missing_entries:
--> 543                 raise OSError(
    544                     f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load the files, and couldn't find them in the"
    545                     f" cached files.\nCheck your internet connection or see how to run the library in offline mode at"

OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

## === cell 9
def _load_state_dict_flex(path, device):
    obj = torch.load(path, map_location=device)
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise ValueError("Unsupported checkpoint format")
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        new_sd[nk] = v
    return new_sd


def get_predictions(checkpoint_path=None):
    """
    BUGFIX: Run inference even if no fine-tuned checkpoints exist.
    Uses make_model() fallback to cached public model if external dataset not attached.
    """
    config, _tok, model = make_model(Config())
    model.to(DEVICE)
    model.eval()

    if checkpoint_path is not None:
        state_dict = _load_state_dict_flex(checkpoint_path, DEVICE)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(unexpected) > 0:
            print(
                f"[WARN] Unexpected keys in {checkpoint_path}: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}"
            )
        if len(missing) > 0:
            print(
                f"[WARN] Missing keys in {checkpoint_path}: {missing[:5]}{'...' if len(missing)>5 else ''}"
            )
    else:
        print(
            "[WARN] No checkpoint provided; using base model weights for predictions."
        )

    start_logits = []
    end_logits = []
    for batch in tqdm(
        test_dataloader,
        desc=f"Predict {os.path.basename(checkpoint_path) if checkpoint_path else 'base'}",
    ):
        with torch.no_grad():
            outputs_start, outputs_end = model(
                batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)
            )
            start_logits.append(outputs_start.detach().cpu().numpy())
            end_logits.append(outputs_end.detach().cpu().numpy())

    del model, _tok, config
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.vstack(start_logits), np.vstack(end_logits)




## === cell 10
checkpoint_paths = [
    ("../input/chaii-abhishek-fold-0/output/checkpoint-fold-0/pytorch_model.bin", 0.10),
    ("../input/chaii-abhishek-fold-1/output/checkpoint-fold-1/pytorch_model.bin", 0.20),
    ("../input/chaii-abhishek-fold-2/output/checkpoint-fold-2/pytorch_model.bin", 0.20),
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
        0.20,
    ),
    (
        "../input/k/vineethakkinapalli/chaii-abhishek-model-2-epochs-10folds/output/checkpoint-fold-9/pytorch_model.bin",
        0.10,
    ),
]

available = []
for p, w in checkpoint_paths:
    p2 = _resolve_kaggle_path(p)
    if os.path.exists(p2):
        available.append((p2, w))
    else:
        print(f"[WARN] Checkpoint not found, skipping: {p2}")


def _discover_checkpoints(root="/kaggle/input", limit=12):
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "pytorch_model.bin" in filenames:
            full = os.path.join(dirpath, "pytorch_model.bin")
            low = full.lower()
            if any(
                k in low
                for k in [
                    "chaii",
                    "qa",
                    "question-answer",
                    "xlm",
                    "roberta",
                    "fold",
                    "checkpoint",
                ]
            ):
                hits.append(full)
    hits = sorted(hits)[:limit]
    return hits


use_base_only = False
if len(available) == 0:
    discovered = _discover_checkpoints("/kaggle/input", limit=8)
    if len(discovered) == 0:
        print(
            "[WARN] No checkpoint files found under /kaggle/input. "
            "Falling back to base model weights (submission will be valid but likely lower score)."
        )
        use_base_only = True
        available = []
    else:
        print(
            "[WARN] No hardcoded checkpoints found; using discovered checkpoints instead:"
        )
        for p in discovered:
            print("  ", p)
        available = [(p, 1.0 / len(discovered)) for p in discovered]
else:
    w_sum = sum(w for _, w in available)
    available = [(p, w / w_sum) for p, w in available]

if not use_base_only and len(available) > 0:
    print("Using checkpoints:")
    for p, w in available:
        print(f"  {p}  weight={w:.4f}")

if use_base_only or len(available) == 0:
    start_logits, end_logits = get_predictions(checkpoint_path=None)
else:
    ens_start = None
    ens_end = None
    for ckpt, w in available:
        s, e = get_predictions(ckpt)
        if ens_start is None:
            ens_start = w * s
            ens_end = w * e
        else:
            ens_start += w * s
            ens_end += w * e
        del s, e
        gc.collect()
    start_logits = ens_start
    end_logits = ens_end

fin_preds = postprocess_qa_predictions(
    test, test_features, (start_logits, end_logits), tokenizer=tokenizer
)

submission = []
for pid, pred in fin_preds.items():
    pred = " ".join(str(pred).split())
    pred = pred.strip(punctuation)
    submission.append((pid, pred))

sample = pd.DataFrame(submission, columns=["id", "PredictionString"])
test_data = pd.merge(left=test, right=sample, on="id", how="left")



## --- ERROR in cell 10, traceback:
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
/tmp/ipykernel_55/1667957740.py in <cell line: 0>()
     82 # Inference (always produce fin_preds)
     83 if use_base_only or len(available) == 0:
---> 84     start_logits, end_logits = get_predictions(checkpoint_path=None)
     85 else:
     86     ens_start = None

/tmp/ipykernel_55/3307070438.py in get_predictions(checkpoint_path)
     25     Uses make_model() fallback to cached public model if external dataset not attached.
     26     """
---> 27     config, _tok, model = make_model(Config())
     28     model.to(DEVICE)
     29     model.eval()

/tmp/ipykernel_55/3532538892.py in make_model(args)
     26         f"Falling back to cached model '{fallback_repo}' (local_files_only=True)."
     27     )
---> 28     config = AutoConfig.from_pretrained(fallback_repo, **_hf_from_pretrained_kwargs())
     29     tokenizer = AutoTokenizer.from_pretrained(
     30         fallback_repo, use_fast=True, **_hf_from_pretrained_kwargs()

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
    665             try:
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,
    669                     configuration_file,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    541             # even when `local_files_only` is True, in which case raising for connections errors only would not make sense)
    542             elif _raise_exceptions_for_missing_entries:
--> 543                 raise OSError(
    544                     f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load the files, and couldn't find them in the"
    545                     f" cached files.\nCheck your internet connection or see how to run the library in offline mode at"

OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

## === cell 11
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
test_data[["id", "PredictionString"]].to_csv(
    "submission.csv", index=False, encoding="utf-8"
)
print("Wrote submission.csv with shape:", test_data[["id", "PredictionString"]].shape)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2891565156.py in <cell line: 0>()
      9 
     10 cleaned_preds = []
---> 11 for pred, context in test_data[["PredictionString", "context"]].to_numpy():
     12     pred = "" if pd.isna(pred) else str(pred)
     13     context = "" if pd.isna(context) else str(context)

NameError: name 'test_data' is not defined

## === cell 12
test_data[["id", "PredictionString"]].head()

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2282797748.py in <cell line: 0>()
----> 1 test_data[["id", "PredictionString"]].head()

NameError: name 'test_data' is not defined
