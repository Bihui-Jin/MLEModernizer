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

0.727562665939331

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the pipeline so it can run without any external Kaggle model datasets by switching the pretrained directory resolution to a robust local path (the competition dataset folder) and disabling `local_files_only` only when needed. This unblocks `Config` creation and prevents the downstream `NameError` cascade by ensuring cell 1 does not raise and by making later cells resilient if model checkpoints are unavailable. I also add a safe fallback that produces a valid `submission.csv` even when checkpoints can’t be loaded (predicting an empty string, which is score-poor but yields a valid file), while keeping the original model/feature/postprocess logic intact when checkpoints exist. Finally, I ensure `test_data` is always defined before cleaning and writing the submission.'
- What this solution (achieved 0.0) has done: 'I fix the tokenizer/model directory resolution so `AutoTokenizer.from_pretrained()` never points at the competition dataset folder (which lacks a `config.json`), and instead falls back to a valid HF model id when no local pretrained model is available. I also make `base_model` reuse the already-resolved `BASE_MODEL_DIR` so checkpoints are found when present, and keep the existing 5-fold averaging and post-processing unchanged. These changes should eliminate the runtime `ValueError` and allow real model inference (instead of the empty-string fallback), moving the score up toward the target while preserving the core logic. The script still always write a valid `submission.csv` in the required format.'
- What this solution (achieved 0.0) has done: 'The crash is coming from an incompatibility between `transformers` and the installed `protobuf` (it manifests as `MessageFactory.GetPrototype` missing), which can be fixed by forcing the pure-Python protobuf implementation before importing `transformers`. I add that environment flag at the very top (before `transformers` is imported) and keep the rest of the pipeline/model logic unchanged. This should allow the pretrained tokenizer/model and the 5-fold checkpoints (if present) to load and run inference instead of falling back to empty strings, moving the score up toward the target. I also keep the existing safe fallback and ensure `submission.csv` is always written in the required format.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf incompatibility that triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by forcing the pure-Python protobuf implementation early and (critically) restarting the protobuf module state before importing `transformers`. I also ensure `protobuf` uses the supported API by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, which is the most common fix for this exact `GetPrototype` breakage in Kaggle runtimes. These changes are execution-only and do not alter your model, folds, post-processing, or submission formatting; they just unblock loading the tokenizer/model so you can get a non-zero score. The script still fall back to empty predictions and write a valid `submission.csv` if checkpoints aren’t available, but in the normal case it should now run full inference.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf/transformers incompatibility that is still triggering `MessageFactory.GetPrototype` by forcing the pure-Python protobuf implementation and importing `google.protobuf` early (before `transformers`) so the correct code path is used. I also make checkpoint loading more robust by using `weights_only=True` when available and by correctly unwrapping common checkpoint dict formats, which avoids silent load failures without changing the model. Finally, I keep your existing 5-fold averaging/postprocess logic intact, but ensure we always write a valid `submission.csv` even if a fold is missing or inference fails.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

import google.protobuf  # noqa: F401

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
    optimal_value = min(num_cpus, num_gpus * 4) if num_gpus else max(0, num_cpus - 1)
    return optimal_value


def resolve_path(p: str) -> str:
    if p is None:
        return p
    if os.path.isabs(p) and os.path.exists(p):
        return p

    if os.path.exists(p):
        return p

    if p.startswith("../input/"):
        alt = os.path.join("/kaggle/input", p[len("../input/") :])
        if os.path.exists(alt):
            return alt
    if p.startswith("input/"):
        alt = os.path.join("/kaggle", p)
        if os.path.exists(alt):
            return alt

    if p.startswith("../data/"):
        alt = os.path.join("/kaggle/data", p[len("../data/") :])
        if os.path.exists(alt):
            return alt
    if p.startswith("data/"):
        alt = os.path.join("/kaggle", p)
        if os.path.exists(alt):
            return alt

    return p


print(f"Apex AMP Installed :: {APEX_INSTALLED}")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



## === cell 1
BASE_MODEL_DIR = resolve_path("../input/chaii-xlmr-5-fold/output/")

FALLBACK_MODEL_DIRS = [
    resolve_path("../input/xlm-roberta-large-squad-v2"),
    resolve_path("/kaggle/input/xlm-roberta-large-squad-v2"),
    BASE_MODEL_DIR,
]


def _is_valid_hf_dir(p: str) -> bool:
    if not p or not os.path.isdir(p):
        return False
    return os.path.exists(os.path.join(p, "config.json"))


PRETRAINED_DIR = next((p for p in FALLBACK_MODEL_DIRS if _is_valid_hf_dir(p)), None)

print("Resolved PRETRAINED_DIR:", PRETRAINED_DIR)
print("Resolved BASE_MODEL_DIR:", BASE_MODEL_DIR)


class Config:
    model_type = "xlm_roberta"
    model_name_or_path = (
        PRETRAINED_DIR if PRETRAINED_DIR is not None else "xlm-roberta-large"
    )
    config_name = PRETRAINED_DIR if PRETRAINED_DIR is not None else "xlm-roberta-large"
    fp16 = True if APEX_INSTALLED else False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2

    tokenizer_name = (
        PRETRAINED_DIR if PRETRAINED_DIR is not None else "xlm-roberta-large"
    )
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
        local_only = os.path.isdir(str(modelname_or_path))
        self.xlm_roberta = AutoModel.from_pretrained(
            modelname_or_path, config=config, local_files_only=local_only
        )
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._init_weights(self.qa_outputs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        attention_mask=None,
    ):
        outputs = self.xlm_roberta(
            input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1] if len(outputs) > 1 else None

        qa_logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits




## === cell 4
def make_model(args):
    local_only_cfg = os.path.isdir(str(args.config_name))
    local_only_tok = os.path.isdir(str(args.tokenizer_name))
    config = AutoConfig.from_pretrained(
        args.config_name, local_files_only=local_only_cfg
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, local_files_only=local_only_tok
    )
    model = Model(args.model_name_or_path, config=config)
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
            0 if j is None else j for j in tokenized_example.sequence_ids(i)
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

            input_ids = features[feature_index]["input_ids"]
            cls_index = input_ids.index(tokenizer.cls_token_id)

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
test_path = resolve_path("../input/chaii-hindi-and-tamil-question-answering/test.csv")
if not os.path.exists(test_path):
    test_path = resolve_path(
        "/kaggle/data/chaii-hindi-and-tamil-question-answering/test.csv"
    )
test = pd.read_csv(test_path)

test["context"] = test["context"].apply(lambda x: " ".join(str(x).split()))
test["question"] = test["question"].apply(lambda x: " ".join(str(x).split()))

args = Config()
local_only_tok = os.path.isdir(str(args.tokenizer_name))
tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_name, local_files_only=local_only_tok
)

test_features = []
for _, row in test.iterrows():
    test_features += prepare_test_features(args, row, tokenizer)

test_dataset = DatasetRetriever(test_features, mode="test")
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.eval_batch_size,
    sampler=SequentialSampler(test_dataset),
    num_workers=optimal_num_of_loader_workers(),
    pin_memory=True,
    drop_last=False,
)



## === cell 8
base_model = BASE_MODEL_DIR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 9
def _torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _extract_state_dict(state):
    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            return state["state_dict"]
        if "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
            return state["model_state_dict"]
    return state


def get_predictions(checkpoint_path):
    config, _tok, model = make_model(Config())
    model.to(device)
    model.eval()

    ckpt_file = os.path.join(base_model, checkpoint_path)
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    state = _torch_load_compat(ckpt_file, map_location="cpu")
    state = _extract_state_dict(state)
    model.load_state_dict(state, strict=True)

    start_logits = []
    end_logits = []
    for batch in test_dataloader:
        with torch.no_grad():
            outputs_start, outputs_end = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            start_logits.append(outputs_start.detach().cpu().numpy())
            end_logits.append(outputs_end.detach().cpu().numpy())
            del outputs_start, outputs_end

    del model, _tok, config
    gc.collect()

    return np.concatenate(start_logits, axis=0), np.concatenate(end_logits, axis=0)




## === cell 10
fin_preds = None
try:
    start_logits1, end_logits1 = get_predictions("checkpoint-fold-0/pytorch_model.bin")
    start_logits2, end_logits2 = get_predictions("checkpoint-fold-1/pytorch_model.bin")
    start_logits3, end_logits3 = get_predictions("checkpoint-fold-2/pytorch_model.bin")
    start_logits4, end_logits4 = get_predictions("checkpoint-fold-3/pytorch_model.bin")
    start_logits5, end_logits5 = get_predictions("checkpoint-fold-4/pytorch_model.bin")

    start_logits = (
        start_logits1 + start_logits2 + start_logits3 + start_logits4 + start_logits5
    ) / 5.0
    end_logits = (
        end_logits1 + end_logits2 + end_logits3 + end_logits4 + end_logits5
    ) / 5.0

    fin_preds = postprocess_qa_predictions(
        test, test_features, (start_logits, end_logits), tokenizer=tokenizer
    )
except Exception as e:
    print(
        "WARNING: Could not load checkpoints / run inference. Falling back to empty predictions."
    )
    print("Reason:", repr(e))
    fin_preds = collections.OrderedDict((i, "") for i in test["id"].tolist())

submission = []
for p1, p2 in fin_preds.items():
    p2 = " ".join(str(p2).split())
    p2 = p2.strip(punctuation)
    submission.append((p1, p2))

sample = pd.DataFrame(submission, columns=["id", "PredictionString"])
test_data = pd.merge(left=test, right=sample, on="id", how="left")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 11
bad_starts = [".", ",", "(", ")", "-", "–", ",", ";"]
bad_endings = ["...", "-", "(", ")", "–", ",", ";"]

tamil_ad = "கி.பி"
tamil_bc = "கி.மு"
tamil_km = "கி.மீ"
hindi_ad = "ई"
hindi_bc = "ई.पू"

if "PredictionString" not in test_data.columns:
    test_data["PredictionString"] = ""

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
