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

0.727562665939331

# 6. Current score

0.0358

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the offline model loading failures by making the code automatically fall back to the provided `sample_submission.csv` when the required HuggingFace model/checkpoints are not available locally (which is what currently stops execution). This keeps the original inference/postprocess logic intact when assets exist, but guarantees an end-to-end run that always writes a valid `submission.csv`. I also fix the `test_data` NameError by ensuring `test_data` is created in both the normal and fallback paths. Finally, I make the tokenizer/model loading robust by trying local paths first and only using `local_files_only=True` when those paths exist, preventing crashes in Kaggle’s no-internet environment.'
- What this solution (achieved 0.0358) has done: 'Your current 0.0 score is caused by the fallback path writing essentially empty predictions when the local tokenizer/checkpoints aren’t present; to move toward the 0.727 target we need non-empty, context-derived answers even without the model assets. I keep your existing model/inference/postprocess unchanged when checkpoints are available, but replace the fallback with a minimal heuristic extractor that selects a short span from the context based on keyword overlap with the question (which directly optimizes for word-level Jaccard). I also make the tokenizer/model loading consistently use the same resolved local directory (when it exists) to reduce accidental fallback. These are minimal changes focused on producing meaningful predictions and a valid submission.csv in all cases.'

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
    optimal_value = min(num_cpus, num_gpus * 4) if num_gpus else max(1, num_cpus - 1)
    return optimal_value


print(f"Apex AMP Installed :: {APEX_INSTALLED}")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

DATA_DIR_CANDIDATES = [
    "/kaggle/input/chaii-hindi-and-tamil-question-answering",
    "../input/chaii-hindi-and-tamil-question-answering",
    "/kaggle/data/chaii-hindi-and-tamil-question-answering",
    "/kaggle/data",
    "../input",
]
DATA_DIR = None
for p in DATA_DIR_CANDIDATES:
    if p and os.path.exists(os.path.join(p, "test.csv")):
        DATA_DIR = p
        break
if DATA_DIR is None:
    DATA_DIR = "."

print("Resolved DATA_DIR =", DATA_DIR)
fix_all_seeds(2021)




## === cell 1
class Config:
    model_type = "xlm_roberta"
    model_name_or_path = "../input/xlm-roberta-large-squad-v2"
    config_name = "../input/xlm-roberta-large-squad-v2"
    fp16 = True if APEX_INSTALLED else False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2

    tokenizer_name = "../input/xlm-roberta-large-squad-v2"
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


def _resolve_first_existing_dir(paths):
    for p in paths:
        if p and os.path.isdir(p):
            return p
    return None


_cfg = Config()
local_model_dir = _resolve_first_existing_dir(
    [
        _cfg.model_name_or_path,
        "/kaggle/input/xlm-roberta-large-squad-v2",
        "/kaggle/input/xlm-roberta-large",
        "/kaggle/input/xlm-roberta-base",
        "../input/xlm-roberta-large-squad-v2",
        "../input/xlm-roberta-large",
        "../input/xlm-roberta-base",
    ]
)

if local_model_dir is not None:
    _cfg.model_name_or_path = local_model_dir
    _cfg.config_name = local_model_dir
    _cfg.tokenizer_name = local_model_dir
else:
    _cfg.model_name_or_path = "xlm-roberta-large"
    _cfg.config_name = "xlm-roberta-large"
    _cfg.tokenizer_name = "xlm-roberta-large"

Config = lambda: _cfg

print("Resolved model_name_or_path =", _cfg.model_name_or_path)
print("Resolved tokenizer_name     =", _cfg.tokenizer_name)
print("Resolved config_name        =", _cfg.config_name)




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
        qa_logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits




## === cell 4
def _safe_local_files_only(path_or_repo_id: str) -> bool:
    return bool(path_or_repo_id) and os.path.isdir(path_or_repo_id)


def make_model(args):
    local_only_cfg = _safe_local_files_only(args.config_name)
    local_only_tok = _safe_local_files_only(args.tokenizer_name)

    config = AutoConfig.from_pretrained(
        args.config_name, local_files_only=local_only_cfg
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, local_files_only=local_only_tok, use_fast=True
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
            0 if s is None else s for s in tokenized_example.sequence_ids(i)
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

            features[feature_index]["offset_mapping"] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(features[feature_index]["offset_mapping"])
            ]
            offset_mapping = features[feature_index]["offset_mapping"]

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
test_path = os.path.join(DATA_DIR, "test.csv")
test = pd.read_csv(test_path)

test["context"] = test["context"].apply(lambda x: " ".join(str(x).split()))
test["question"] = test["question"].apply(lambda x: " ".join(str(x).split()))

args = Config()

tokenizer = None
tokenizer_available = False
try:
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        local_files_only=_safe_local_files_only(args.tokenizer_name),
        use_fast=True,
    )
    tokenizer_available = True
except Exception as e:
    print(
        "Tokenizer not available locally; will use fallback submission. Error:", repr(e)
    )
    tokenizer_available = False

test_features = []
test_dataset = None
test_dataloader = None
if tokenizer_available:
    for _, row in tqdm(test.iterrows(), total=len(test), desc="Tokenizing test"):
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
    print("Num test examples =", len(test), "Num test features =", len(test_features))



## === cell 8
base_model = "../input/chaii-xlmr-5-fold/output/"
candidates = [
    base_model,
    "/kaggle/input/chaii-xlmr-5-fold/output/",
    "/kaggle/data/chaii-xlmr-5-fold/output/",
    "/kaggle/input/chaii-xlmr-5-fold/chaii-xlmr-5-fold/output/",
]
resolved = None
for c in candidates:
    if c and os.path.isdir(c):
        resolved = c
        break

base_model = resolved
print("Resolved base_model =", base_model)

checkpoints_available = base_model is not None and os.path.isdir(base_model)




## === cell 9
def get_predictions(checkpoint_path):
    config, tok, model = make_model(Config())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    state_path = os.path.join(base_model, checkpoint_path)
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Missing checkpoint file: {state_path}")
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    start_logits = []
    end_logits = []
    for batch in tqdm(
        test_dataloader, desc=f"Infer {os.path.dirname(checkpoint_path)}"
    ):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs_start, outputs_end = model(input_ids, attention_mask)
            start_logits.append(outputs_start.detach().cpu().numpy())
            end_logits.append(outputs_end.detach().cpu().numpy())
            del outputs_start, outputs_end, input_ids, attention_mask

    del model, tok, config
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.vstack(start_logits), np.vstack(end_logits)




## === cell 10
def _normalize_for_match(s: str) -> str:
    s = "" if s is None else str(s)
    return " ".join(s.replace("\n", " ").replace("\t", " ").split()).strip()


def _simple_tokenize_ws(s: str):
    s = _normalize_for_match(s).lower()
    return [t for t in s.split() if t]


def _heuristic_extract_answer(context: str, question: str, max_words: int = 14) -> str:
    context = _normalize_for_match(context)
    question = _normalize_for_match(question)

    ctx_tokens = context.split()
    if len(ctx_tokens) == 0:
        return ""

    q_tokens = set(_simple_tokenize_ws(question))
    if len(q_tokens) == 0:
        return " ".join(ctx_tokens[: min(max_words, len(ctx_tokens))])

    best_i = 0
    best_len = min(max_words, len(ctx_tokens))
    best_score = -1
    for wlen in range(2, min(max_words, len(ctx_tokens)) + 1):
        for i in range(0, len(ctx_tokens) - wlen + 1):
            window = ctx_tokens[i : i + wlen]
            overlap = len(set(t.lower() for t in window) & q_tokens)
            if overlap > best_score:
                best_score = overlap
                best_i = i
                best_len = wlen

    ans = " ".join(ctx_tokens[best_i : best_i + best_len])
    ans = " ".join(ans.split()).strip(punctuation)
    return ans


test_data = None

if tokenizer_available and checkpoints_available:
    start_logits1, end_logits1 = get_predictions("checkpoint-fold-0/pytorch_model.bin")
    start_logits2, end_logits2 = get_predictions("checkpoint-fold-1/pytorch_model.bin")
    start_logits3, end_logits3 = get_predictions("checkpoint-fold-2/pytorch_model.bin")
    start_logits4, end_logits4 = get_predictions("checkpoint-fold-3/pytorch_model.bin")
    start_logits5, end_logits5 = get_predictions("checkpoint-fold-4/pytorch_model.bin")

    start_logits = (
        start_logits1 + start_logits2 + start_logits3 + start_logits4 + start_logits5
    ) / 5
    end_logits = (
        end_logits1 + end_logits2 + end_logits3 + end_logits4 + end_logits5
    ) / 5

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
    print("test_data shape =", test_data.shape)
else:
    preds = []
    for rid, ctx, q in tqdm(
        test[["id", "context", "question"]].itertuples(index=False),
        total=len(test),
        desc="Heuristic fallback predict",
    ):
        preds.append((rid, _heuristic_extract_answer(ctx, q)))

    test_data = pd.DataFrame(preds, columns=["id", "PredictionString"])
    test_data = pd.merge(test[["id", "context"]], test_data, on="id", how="left")
    test_data["PredictionString"] = test_data["PredictionString"].fillna("")
    print(
        "Using heuristic fallback submission. shape =",
        test_data[["id", "PredictionString"]].shape,
    )



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
    pred = "" if pred is None else str(pred)
    context = "" if context is None else str(context)
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

out_path = "submission.csv"
test_data[["id", "PredictionString"]].to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape={test_data[['id','PredictionString']].shape}")
print(test_data[["id", "PredictionString"]].head())
