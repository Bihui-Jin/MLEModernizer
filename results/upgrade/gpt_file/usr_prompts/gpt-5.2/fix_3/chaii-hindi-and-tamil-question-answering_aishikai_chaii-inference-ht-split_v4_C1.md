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

0.7131982445716858

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.02477) has done: 'I fix the import errors caused by recent Transformers changes (AdamW moved), and make sure `AutoTokenizer/AutoConfig` are always imported so later cells can run. I also fix runtime failures due to missing Kaggle input model paths by adding a safe fallback: if the provided checkpoints aren’t available, the code still generate a valid `submission.csv` using a simple context-based baseline (score be lower, but it run end-to-end). I make `Postprocess_qa_predictions` take the tokenizer as an argument (it currently relies on a global that may not exist), and correct the Hindi/Tamil ensembling logic bug that indexed wrong arrays. Finally, I ensure the submission file is written with the correct columns and `.csv` suffix.'

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
    return optimal_value


print(f"Apex AMP Installed :: {APEX_INSTALLED}")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fix_all_seeds(2021)




## === cell 1
class Configration:
    model_type = "xlm_roberta"
    XLMR_name_or_path = "../input/nothing-of-your-concern/xlm-roberta-large-squad2/xlm-roberta-large-squad2"
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
    config = AutoConfig.from_pretrained(args.MURIL_config_name)
    tokenizer = AutoTokenizer.from_pretrained(args.MURIL_tokenizer_name)
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
        feature["offset_mapping"] = tokenized_example["offset_mapping"][i]
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
            start_logits = np.array(all_start_logits[feature_index])
            end_logits = np.array(all_end_logits[feature_index])

            sequence_ids = features[feature_index]["sequence_ids"]
            context_index = 1

            offset_mapping = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(features[feature_index]["offset_mapping"])
            ]
            cls_index = features[feature_index]["input_ids"].index(
                tokenizer.cls_token_id
            )

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



## === cell 8
args = Configration()

offline_muril_dir = os.path.join(DATA_DIR, "muril-base-cased")
if os.path.exists(offline_muril_dir):
    args.MURIL_name_or_path = offline_muril_dir
    args.MURIL_tokenizer_name = offline_muril_dir
    args.MURIL_config_name = offline_muril_dir

tokenizer = AutoTokenizer.from_pretrained(args.MURIL_tokenizer_name)

test_features = []
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
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



## --- ERROR in cell 8, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/nothing-of-your-concern/muril-large-cased/muril-large-cased/'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_55/2566911348.py in <cell line: 0>()
     10     args.MURIL_config_name = offline_muril_dir
     11 
---> 12 tokenizer = AutoTokenizer.from_pretrained(args.MURIL_tokenizer_name)
     13 
     14 test_features = []

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/nothing-of-your-concern/muril-large-cased/muril-large-cased/'. Use `repo_type` argument if needed.

## === cell 9
base_model_hindi = "../input/chaii-helper/Muril-H/content/output/"
base_model_tamil = "../input/chaii-helper/Muril-T/content/output/"


def _safe_torch_load(path):
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")


def Get_Predictions(checkpoint_path):
    config, tok, model = Make_Model(args)
    model.to(DEVICE)
    sd = _safe_torch_load(checkpoint_path)
    if sd is None:
        return None, None, None
    model.load_state_dict(sd)
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




## === cell 10
hindi_ckpts = [
    os.path.join(base_model_hindi, f"checkpoint-fold-{i}", "pytorch_model.bin")
    for i in range(5)
]
tamil_ckpts = [
    os.path.join(base_model_tamil, f"checkpoint-fold-{i}", "pytorch_model.bin")
    for i in range(4)
]

have_any_ckpt = any(os.path.exists(p) for p in (hindi_ckpts + tamil_ckpts))

if have_any_ckpt:
    start_list_h, end_list_h, languages = [], [], None
    for p in hindi_ckpts:
        s, e, langs = Get_Predictions(p)
        if s is not None:
            start_list_h.append(s)
            end_list_h.append(e)
            languages = langs  # per-feature languages

    start_list_t, end_list_t = [], []
    for p in tamil_ckpts:
        s, e, _ = Get_Predictions(p)
        if s is not None:
            start_list_t.append(s)
            end_list_t.append(e)

    start_logits_h = np.mean(start_list_h, axis=0) if len(start_list_h) else None
    end_logits_h = np.mean(end_list_h, axis=0) if len(end_list_h) else None
    start_logits_t = np.mean(start_list_t, axis=0) if len(start_list_t) else None
    end_logits_t = np.mean(end_list_t, axis=0) if len(end_list_t) else None

    start, end = [], []
    for i, lang in enumerate(languages):
        if lang == "hindi" and start_logits_h is not None:
            start.append(start_logits_h[i])
            end.append(end_logits_h[i])
        elif lang != "hindi" and start_logits_t is not None:
            start.append(start_logits_t[i])
            end.append(end_logits_t[i])
        else:
            if start_logits_h is not None:
                start.append(start_logits_h[i])
                end.append(end_logits_h[i])
            else:
                start.append(start_logits_t[i])
                end.append(end_logits_t[i])

    fin_preds = Postprocess_qa_predictions(
        test_df, test_features, (start, end), tokenizer
    )
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



## --- ERROR in cell 10, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/nothing-of-your-concern/muril-large-cased/muril-large-cased/config.json'. Use `repo_type` argument if needed.

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/nothing-of-your-concern/muril-large-cased/muril-large-cased/config.json'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/4104395340.py in <cell line: 0>()
     56     # Change (score-relevant): instead of a weak context-first baseline, fine-tune the SAME model for 1 epoch
     57     # using provided train labels, then run normal QA postprocessing.
---> 58     model, tok, config = FineTune_If_Needed(args)
     59 
     60     start_logits = []

/tmp/ipykernel_55/3776393724.py in FineTune_If_Needed(args)
     98     # Change (score-relevant): if no external finetuned checkpoints exist, do a minimal 1-epoch finetune
     99     # on the provided training set using the SAME model, head, and CE losses implied by start/end logits.
--> 100     config, tok, model = Make_Model(args)
    101     model.to(DEVICE)
    102 

/tmp/ipykernel_55/3585602275.py in Make_Model(args)
      1 def Make_Model(args):
----> 2     config = AutoConfig.from_pretrained(args.MURIL_config_name)
      3     tokenizer = AutoTokenizer.from_pretrained(args.MURIL_tokenizer_name)
      4     model = Model(args.MURIL_name_or_path, config=config)
      5     return config, tokenizer, model

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

OSError: Can't load the configuration of '../input/nothing-of-your-concern/muril-large-cased/muril-large-cased/config.json'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '../input/nothing-of-your-concern/muril-large-cased/muril-large-cased/config.json' is the correct path to a directory containing a config.json file

## === cell 11
submission = []
for pid, pred in fin_preds.items():
    pred = " ".join(str(pred).split())
    pred = pred.strip(punctuation)
    submission.append((pid, pred))

sample = pd.DataFrame(submission, columns=["id", "PredictionString"])
test_data = pd.merge(left=test_df, right=sample, on="id", how="left")



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4288137622.py in <cell line: 0>()
      1 submission = []
----> 2 for pid, pred in fin_preds.items():
      3     pred = " ".join(str(pred).split())
      4     pred = pred.strip(punctuation)
      5     submission.append((pid, pred))

NameError: name 'fin_preds' is not defined

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

    cleaned_preds.append(pred)

test_data["PredictionString"] = cleaned_preds

out_path = "submission.csv"
test_data[["id", "PredictionString"]].to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape={test_data[['id','PredictionString']].shape}")
print(test_data[["id", "PredictionString"]].head())



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3780225020.py in <cell line: 0>()
      9 
     10 cleaned_preds = []
---> 11 for pred, context in test_data[["PredictionString", "context"]].to_numpy():
     12     pred = "" if pd.isna(pred) else str(pred)
     13     context = "" if pd.isna(context) else str(context)

NameError: name 'test_data' is not defined

## === cell 13
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["id", "PredictionString"]
assert len(sub) == len(test_df)
print("Submission looks valid.")

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1379982644.py in <cell line: 0>()
----> 1 sub = pd.read_csv("submission.csv")
      2 assert list(sub.columns) == ["id", "PredictionString"]
      3 assert len(sub) == len(test_df)
      4 print("Submission looks valid.")

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'
