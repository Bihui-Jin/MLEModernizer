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

0.7240094542503357

# 6. Current score

0.62857

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.03396) has done: 'Main runtime issues are (1) loading the large XLM-R model via `Make_Model()` even though a tokenizer is already created, (2) slow Python-level post-processing that repeatedly sorts full-length logits and iterates pandas rows, and (3) avoidable DataLoader overhead (too many workers / heavy per-item tensor creation). I keep the exact same model, tokenization parameters, and prediction semantics, but speed up by caching/avoiding redundant tokenizer creation, limiting DataLoader workers to a safe small value for inference, and making post-processing equivalent-but-faster by using `np.argpartition` for top-k selection and iterating over numpy arrays instead of `DataFrame.iterrows()`. I also avoid building intermediate `start/end` Python lists (no-op language branch in your code) and pass numpy arrays directly into post-processing. These are provably equivalent in outputs aside from negligible floating-point tie-order differences.'
- What this solution (achieved 0.62857) has done: 'I fix the runtime crash caused by an incompatible `protobuf`/`sentencepiece` stack being imported transitively (it triggers the `MessageFactory.GetPrototype` error) by setting the safe protobuf implementation environment variables before any `transformers` import. I also ensure inference actually uses the intended pretrained QA weights (right now it reinitializes the QA head randomly because it loads `AutoModel` instead of `AutoModelForQuestionAnswering`), which is the main reason your score is extremely low; this keeps the same underlying backbone and QA objective but restores correct pretrained behavior. Finally, I keep the existing tokenization, feature creation, and post-processing semantics unchanged, only making small robustness tweaks to submission creation to guarantee a valid `submission.csv`.'
- What this solution (achieved 0.62857) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before any* `transformers/sentencepiece` import happens (your current setting is in cell 0, but in Kaggle notebooks imports can occur earlier; we enforce it at the very top and re-import safely). Then I make the submission creation robust even if inference fails mid-way (so a valid `submission.csv` is always written), while keeping the same model/tokenization/post-processing logic that produced your 0.62857 score. Finally, I ensure `test_df` indexing and merge alignment are stable and that `PredictionString` is always a proper UTF-8 string column.'
- What this solution (achieved 0.62857) has done: 'I fix the protobuf/sentencepiece crash by forcing the pure-Python protobuf implementation *before* any `transformers` import, and by explicitly using the Python protobuf backend in-process; this removes the `MessageFactory.GetPrototype` runtime error that currently prevents real inference. I keep your model/inference/post-processing logic the same, but ensure we actually run it (so we don’t fall back to empty strings), which should move the score up substantially toward the target. I also make the inference path use `local_files_only=True` as a safe fallback when Kaggle has the model cached, while still working normally when it can download. Finally, I keep the same submission formatting and guarantee `submission.csv` is always written.'
- What this solution (achieved 0.62857) has done: 'I fix the protobuf/sentencepiece crash by forcing the pure-Python protobuf implementation *before any other imports* and by proactively reloading `google.protobuf` if it was already imported in the notebook runtime. Then I keep your inference + postprocessing logic the same, but remove the try/except fallback that currently masks the failure and yields empty predictions (hurting score), so the pipeline actually runs end-to-end. Finally, I make the submission writing robust and aligned to `sample_submission.csv` ids to guarantee the correct row count and ordering without changing prediction semantics.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys

if "google.protobuf" in sys.modules:
    mods = [m for m in list(sys.modules.keys()) if m.startswith("google.protobuf")]
    for m in mods:
        sys.modules.pop(m, None)

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

try:
    from apex import amp

    APEX_INSTALLED = True
except ImportError:
    APEX_INSTALLED = False

try:
    from google.protobuf.internal import api_implementation

    try:
        api_implementation._SetType("python")
    except Exception:
        pass
except Exception:
    pass

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
    optimal_value = min(num_cpus, num_gpus * 4) if num_gpus else max(0, num_cpus - 1)
    return optimal_value


print(f"Apex AMP Installed :: {APEX_INSTALLED}")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fix_all_seeds(2021)
torch.backends.cudnn.benchmark = True




## === cell 1
class Configration:
    model_type = "xlm_roberta"

    XLMR_name_or_path = "deepset/xlm-roberta-large-squad2"
    XLMR_config_name = "deepset/xlm-roberta-large-squad2"
    XLMR_tokenizer_name = "deepset/xlm-roberta-large-squad2"

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
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "offset_mapping": feature["offset_mapping"],
                "start_position": feature["start_position"],
                "end_position": feature["end_position"],
            }
        else:
            return {
                "language": feature["language"],
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "offset_mapping": feature["offset_mapping"],
                "sequence_ids": feature["sequence_ids"],
                "id": feature["example_id"],
                "context": feature["context"],
                "question": feature["question"],
            }


def _fast_collate_test(batch):
    input_ids = torch.as_tensor([b["input_ids"] for b in batch], dtype=torch.long)
    attention_mask = torch.as_tensor(
        [b["attention_mask"] for b in batch], dtype=torch.long
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "language": [b["language"] for b in batch],
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
def Make_Model(args):
    config = AutoConfig.from_pretrained(args.XLMR_config_name)
    tokenizer = AutoTokenizer.from_pretrained(args.XLMR_tokenizer_name, use_fast=True)
    model = Model(args.XLMR_name_or_path, config=config)
    return config, tokenizer, model




## === cell 5
def Prepare_Test_Features(args, example, tokenizer):
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
        feature["language"] = example["language"]
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




## === cell 7
def Postprocess_qa_predictions(
    examples, features, raw_predictions, tokenizer, n_best_size=20, max_answer_length=30
):
    all_start_logits, all_end_logits = raw_predictions

    example_id_to_index = {k: i for i, k in enumerate(examples["id"].to_numpy())}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    cls_token_id = (
        tokenizer.cls_token_id
        if tokenizer.cls_token_id is not None
        else tokenizer.bos_token_id
    )

    masked_offsets = [None] * len(features)
    for fi, feat in enumerate(features):
        seq_ids = feat["sequence_ids"]
        offs = feat["offset_mapping"]
        masked_offsets[fi] = [
            o if seq_ids[k] == 1 else None for k, o in enumerate(offs)
        ]

    predictions = collections.OrderedDict()
    print(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )

    ex_ids = examples["id"].to_numpy()
    ex_contexts = examples["context"].to_numpy()

    for example_index in range(len(ex_ids)):
        ex_id = ex_ids[example_index]
        context = ex_contexts[example_index]
        feature_indices = features_per_example[example_index]
        valid_answers = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = masked_offsets[feature_index]

            if n_best_size < len(start_logits):
                s_idx = np.argpartition(start_logits, -n_best_size)[-n_best_size:]
            else:
                s_idx = np.arange(len(start_logits))
            if n_best_size < len(end_logits):
                e_idx = np.argpartition(end_logits, -n_best_size)[-n_best_size:]
            else:
                e_idx = np.arange(len(end_logits))

            s_idx = s_idx[np.argsort(start_logits[s_idx])[::-1]]
            e_idx = e_idx[np.argsort(end_logits[e_idx])[::-1]]

            for start_index in s_idx.tolist():
                if (
                    start_index >= len(offset_mapping)
                    or offset_mapping[start_index] is None
                ):
                    continue
                start_char = offset_mapping[start_index][0]
                for end_index in e_idx.tolist():
                    if (
                        end_index >= len(offset_mapping)
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index:
                        continue
                    if end_index - start_index + 1 > max_answer_length:
                        continue
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
            best_answer = max(valid_answers, key=lambda x: x["score"])
        else:
            best_answer = {"text": "", "score": 0.0}

        predictions[ex_id] = best_answer["text"]

    return predictions




## === cell 8
test_df = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/test.csv")
test_df = test_df.reset_index(drop=True)



## === cell 9
test_df["context"] = test_df["context"].apply(lambda x: " ".join(str(x).split()))
test_df["question"] = test_df["question"].apply(lambda x: " ".join(str(x).split()))



## === cell 10
args = Configration()

try:
    tokenizer = AutoTokenizer.from_pretrained(
        args.XLMR_tokenizer_name, use_fast=True, local_files_only=True
    )
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(args.XLMR_tokenizer_name, use_fast=True)

questions = test_df["question"].str.lstrip().tolist()
contexts = test_df["context"].tolist()

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

overflow_to_sample = tokenized["overflow_to_sample_mapping"]
input_ids_all = tokenized["input_ids"]
attn_all = tokenized["attention_mask"]
offs_all = tokenized["offset_mapping"]

test_features = []
for i in range(len(input_ids_all)):
    sample_idx = overflow_to_sample[i]
    row = test_df.iloc[int(sample_idx)]
    feature = {
        "language": row["language"],
        "example_id": row["id"],
        "context": row["context"],
        "question": str(row["question"]).lstrip(),
        "input_ids": input_ids_all[i],
        "attention_mask": attn_all[i],
        "offset_mapping": offs_all[i],
        "sequence_ids": [0 if j is None else j for j in tokenized.sequence_ids(i)],
    }
    test_features.append(feature)

test_dataset = Dataset_Retriever(test_features, mode="test")

_num_workers = optimal_num_of_loader_workers()
_num_workers = min(_num_workers, 4)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.eval_batch_size,
    sampler=SequentialSampler(test_dataset),
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    persistent_workers=(_num_workers > 0),
    collate_fn=_fast_collate_test,
)



## === cell 11
base_model_hindi = args.XLMR_name_or_path
base_model_tamil = args.XLMR_name_or_path




## === cell 12
@torch.inference_mode()
def Get_Predictions(checkpoint_path=None):
    cfg = Configration()

    try:
        qa_model = AutoModelForQuestionAnswering.from_pretrained(
            cfg.XLMR_name_or_path, local_files_only=True
        )
    except Exception:
        qa_model = AutoModelForQuestionAnswering.from_pretrained(cfg.XLMR_name_or_path)

    qa_model.to(DEVICE)
    qa_model.eval()

    if (
        checkpoint_path is not None
        and isinstance(checkpoint_path, str)
        and os.path.exists(checkpoint_path)
    ):
        state = torch.load(checkpoint_path, map_location="cpu")
        missing, unexpected = qa_model.load_state_dict(state, strict=False)
        print(
            f"Loaded checkpoint: {checkpoint_path} | missing={len(missing)} unexpected={len(unexpected)}"
        )

    start_logits_chunks = []
    end_logits_chunks = []
    languages = []
    for batch in tqdm(test_dataloader, desc="Infer", leave=False):
        out = qa_model(
            input_ids=batch["input_ids"].to(DEVICE, non_blocking=True),
            attention_mask=batch["attention_mask"].to(DEVICE, non_blocking=True),
        )
        outputs_start = out.start_logits
        outputs_end = out.end_logits
        start_logits_chunks.append(outputs_start.detach().cpu().numpy())
        end_logits_chunks.append(outputs_end.detach().cpu().numpy())
        languages.extend(batch["language"])

    del qa_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.vstack(start_logits_chunks), np.vstack(end_logits_chunks), languages




## === cell 13
start_logits, end_logits, languages = Get_Predictions(None)

fin_preds = Postprocess_qa_predictions(
    test_df, test_features, (start_logits, end_logits), tokenizer=tokenizer
)

submission = []
for p1, p2 in fin_preds.items():
    p2 = " ".join(str(p2).split())
    p2 = p2.strip(punctuation)
    submission.append((p1, p2))

sample_pred_df = pd.DataFrame(submission, columns=["id", "PredictionString"])
test_data = pd.merge(left=test_df, right=sample_pred_df, on="id", how="left")
test_data["PredictionString"] = test_data["PredictionString"].fillna("")



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 14
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

sample_sub = pd.read_csv(
    "../input/chaii-hindi-and-tamil-question-answering/sample_submission.csv"
)
out_df = sample_sub[["id"]].merge(
    test_data[["id", "PredictionString"]], on="id", how="left"
)
out_df["PredictionString"] = out_df["PredictionString"].fillna("").astype(str)
out_df.to_csv("submission.csv", index=False, encoding="utf-8")
print("Wrote submission.csv with shape:", out_df.shape)



## === cell 15
test_data.head()
