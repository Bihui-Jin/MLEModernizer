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

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the root cause of the failures: your notebook depends on an external Kaggle dataset (`/kaggle/input/chaii-xlmr-5-fold/...`) that isn’t present, so tokenizers/models/checkpoints can’t be loaded. To keep the core inference/postprocessing logic intact while making it run end-to-end, I add a safe fallback that uses a public HF model available in the Kaggle environment cache (`/kaggle/input`) if present, otherwise it still run by generating a valid (but low-scoring) empty-prediction submission rather than crashing. I also guard downstream cells so `test_data` is always defined and `submission.csv` is always written with the correct columns and UTF-8 encoding. No training loop/architecture changes are introduced; the only adjustments are path resolution, robust loading, and fail-safe submission generation.'
- What this solution (achieved 0.0) has done: 'Your score is 0.0 because the code usually falls back to an all-empty submission when it can’t find the external fold checkpoints under `/kaggle/input/chaii-xlmr-5-fold/output/`. To move toward the 0.726 target with minimal logic changes, I (1) allow loading a local QA model directly from any available HF model directory under `/kaggle/input` (not just that specific dataset), and (2) if fold checkpoints are missing, run a single-pass inference with the base pretrained QA head instead of producing empty strings. This preserves your exact tokenization, sliding-window feature creation, and postprocess selection logic; it just removes the “empty submission” failure mode. The output stays a valid `submission.csv` with the required columns and row alignment.'
- What this solution (achieved 0.0) has done: 'Your score is 0.0 because the current pipeline often can’t load the intended local model/checkpoints and then falls back to empty strings, which yields near-zero Jaccard. To move toward the 0.726 target with minimal semantic change, I keep your exact tokenization, windowing, model head, and post-processing, but (1) switch to loading an actual QA model head via `AutoModelForQuestionAnswering` when fold checkpoints aren’t available, and (2) remove `local_files_only=True` so the notebook can download the base model if it isn’t already cached (this is the smallest change that turns “empty submission” into real answers). I also make checkpoint loading robust to common state-dict key prefixes so it won’t silently fail due to key mismatches. The code still writes a valid `submission.csv` with the required columns and alignment.'

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
    AutoModelForQuestionAnswering,  # change: enables fallback to a real QA head if fold ckpts absent
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
    Priority:
      1) preferred if exists
      2) common chaii/xlmr directories
      3) any directory containing config.json + (tokenizer.json or sentencepiece model) + model weights
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
    change: tolerate different checkpoint formats/prefixes to avoid failing to load valid fold weights.
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
    cfg_path = infer_local_model_dir(args.config_name) or args.config_name
    tok_path = infer_local_model_dir(args.tokenizer_name) or args.tokenizer_name
    mdl_path = infer_local_model_dir(args.model_name_or_path) or args.model_name_or_path

    config = AutoConfig.from_pretrained(cfg_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    model = Model(mdl_path, config=config)
    return config, tokenizer, model


def make_qa_model_fallback(args):
    """
    change: when fold checkpoints aren't present, use a real QA head (AutoModelForQuestionAnswering)
    instead of our randomly initialized Linear head, which is a minimal but crucial boost from ~0 Jaccard.
    """
    mdl_path = infer_local_model_dir(args.model_name_or_path) or args.model_name_or_path
    tok_path = infer_local_model_dir(args.tokenizer_name) or args.tokenizer_name
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(mdl_path)
    return tokenizer, qa_model




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
            0 if x is None else x for x in tokenized_example.sequence_ids(i)
        ]
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

tokenizer_path = infer_local_model_dir(args.tokenizer_name) or args.tokenizer_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
CAN_PREDICT = True

test_records = test[["id", "context", "question"]].to_dict(orient="records")

test_features = []
for row in tqdm(test_records, total=len(test_records), desc="Tokenizing test"):
    test_features.extend(prepare_test_features(args, row, tokenizer))

test_dataset = DatasetRetriever(test_features, mode="test")
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.eval_batch_size,
    sampler=SequentialSampler(test_dataset),
    num_workers=optimal_num_of_loader_workers(),
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)



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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/chaii-xlmr-5-fold/xlm-roberta-large-squad-v2'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_55/2005781357.py in <cell line: 0>()
      9 # change: allow download if local not present (prevents CAN_PREDICT=False causing empty submission)
     10 tokenizer_path = infer_local_model_dir(args.tokenizer_name) or args.tokenizer_name
---> 11 tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
     12 CAN_PREDICT = True
     13 

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/chaii-xlmr-5-fold/xlm-roberta-large-squad-v2'. Use `repo_type` argument if needed.

## === cell 8
base_model = "/kaggle/input/chaii-xlmr-5-fold/output/"


def get_predictions(checkpoint_path):
    config, tok_local, model = make_model(Config())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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

    model.eval()

    start_logits = []
    end_logits = []
    for batch in tqdm(test_dataloader, desc=f"Predict {checkpoint_path}"):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs_start, outputs_end = model(input_ids, attention_mask)
            start_logits.append(outputs_start.detach().cpu().numpy())
            end_logits.append(outputs_end.detach().cpu().numpy())
            del outputs_start, outputs_end, input_ids, attention_mask

    del model, tok_local, config
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.vstack(start_logits), np.vstack(end_logits)


def get_predictions_base_model_only():
    config, tok_local, model = make_model(Config())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    start_logits = []
    end_logits = []
    for batch in tqdm(test_dataloader, desc="Predict base model (random QA head)"):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs_start, outputs_end = model(input_ids, attention_mask)
            start_logits.append(outputs_start.detach().cpu().numpy())
            end_logits.append(outputs_end.detach().cpu().numpy())
            del outputs_start, outputs_end, input_ids, attention_mask

    del model, tok_local, config
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.vstack(start_logits), np.vstack(end_logits)


def get_predictions_pretrained_qa_head():
    tok_local, qa_model = make_qa_model_fallback(Config())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qa_model.to(device)
    qa_model.eval()

    start_logits = []
    end_logits = []
    for batch in tqdm(test_dataloader, desc="Predict pretrained QA head (fallback)"):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = qa_model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits.append(out.start_logits.detach().cpu().numpy())
            end_logits.append(out.end_logits.detach().cpu().numpy())
            del out, input_ids, attention_mask

    del qa_model, tok_local
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.vstack(start_logits), np.vstack(end_logits)




## === cell 9
test_data = test.copy()
test_data["PredictionString"] = ""

if CAN_PREDICT:
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
            p2 = p2.strip(punctuation)
            submission.append((p1, p2))

        sample = pd.DataFrame(submission, columns=["id", "PredictionString"])
        test_data = pd.merge(left=test, right=sample, on="id", how="left")
        test_data["PredictionString"] = test_data["PredictionString"].fillna("")
    except FileNotFoundError as e:
        print("WARNING:", str(e))
        print("Falling back to pretrained QA-head inference (no fold checkpoints).")
        try:
            start_logits, end_logits = get_predictions_pretrained_qa_head()

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
            test_data["PredictionString"] = test_data["PredictionString"].fillna("")
        except Exception as e2:
            print("WARNING: Pretrained QA-head inference failed:", repr(e2))
            print(
                "Proceeding with empty PredictionString for all rows (valid submission)."
            )
            test_data = test.copy()
            test_data["PredictionString"] = ""



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2518925353.py in <cell line: 0>()
      2 test_data["PredictionString"] = ""
      3 
----> 4 if CAN_PREDICT:
      5     try:
      6         start_logits1, end_logits1 = get_predictions(

NameError: name 'CAN_PREDICT' is not defined

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

sub = test_data[["id", "PredictionString"]].copy()

sample_sub_path = (
    "/kaggle/input/chaii-hindi-and-tamil-question-answering/sample_submission.csv"
)
sample_sub = pd.read_csv(sample_sub_path)
sub = sample_sub[["id"]].merge(sub, on="id", how="left")
sub["PredictionString"] = sub["PredictionString"].fillna("")

sub.to_csv("submission.csv", index=False, encoding="utf-8")
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv saved at:", os.path.abspath("submission.csv"))
