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

0.727562665939331

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
APEX_INSTALLED = False
print(f"Apex AMP Installed :: {APEX_INSTALLED}")

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

from torch.optim import AdamW

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

logging.set_verbosity_warning()
logging.set_verbosity_error()

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(MODEL_CONFIG_CLASSES)

torch.backends.cudnn.benchmark = True

torch.set_num_threads(os.cpu_count())
torch.set_float32_matmul_precision("high")




## === cell 1
class Config:
    model_type = "xlm-roberta-base"
    model_name_or_path = "xlm-roberta-base"
    config_name = "xlm-roberta-base"
    fp16 = True if APEX_INSTALLED else False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2

    tokenizer_name = "xlm-roberta-base"
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
def optimal_num_of_loader_workers():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus * 4) if num_gpus else max(num_cpus - 1, 1)
    return optimal_value




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
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "offset_mapping": feature["offset_mapping_context"],
                "cls_index": feature["cls_index"],
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
        self.xlm_roberta = AutoModel.from_pretrained(modelname_or_path, config=config)
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




## === cell 5
def make_model(args):
    config = AutoConfig.from_pretrained(args.model_name_or_path, local_files_only=False)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, local_files_only=False
    )
    model = Model(args.model_name_or_path, config=config)
    return config, tokenizer, model




## === cell 6
def prepare_test_features(args, tokenizer, df):
    """
    Batch‑tokenize the whole test dataframe and return a flat list of features.
    Pre‑computes context‑only offset mapping and CLS token index to avoid
    recomputation during post‑processing.
    """
    tokenized = tokenizer(
        df["question"].tolist(),
        df["context"].tolist(),
        truncation="only_second",
        max_length=args.max_seq_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    features = []
    for i in range(len(tokenized["input_ids"])):
        example_index = sample_mapping[i]
        example = df.iloc[example_index]

        input_ids = tokenized["input_ids"][i]
        attention_mask = tokenized["attention_mask"][i]
        offsets = tokenized["offset_mapping"][i]
        seq_ids = tokenized.sequence_ids(i)

        offset_mapping_context = [
            (o if seq_ids[k] == 1 else None) for k, o in enumerate(offsets)
        ]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        feature = {
            "example_id": example["id"],
            "context": example["context"],
            "question": example["question"].lstrip(),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "offset_mapping_context": offset_mapping_context,
            "sequence_ids": [0 if idx is None else idx for idx in seq_ids],
            "cls_index": cls_index,
        }
        features.append(feature)
    return features




## === cell 7
import collections


def _top_k_indices(arr, k):
    """Return indices of the top‑k values in descending order using argpartition."""
    if k >= len(arr):
        return np.argsort(arr)[::-1]
    part = np.argpartition(arr, -k)[-k:]
    return part[np.argsort(arr[part])[::-1]]


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

        min_null_score = None
        valid_answers = []
        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            offset_mapping = features[feature_index]["offset_mapping_context"]
            cls_index = features[feature_index]["cls_index"]

            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = _top_k_indices(start_logits, n_best_size)
            end_indexes = _top_k_indices(end_logits, n_best_size)
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
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char:end_char],
                        }
                    )
        if valid_answers:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                0
            ]
        else:
            best_answer = {"text": "", "score": 0.0}
        predictions[example["id"]] = best_answer["text"]
    return predictions




## === cell 8
test = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/test.csv")
test["context"] = test["context"].apply(lambda x: " ".join(str(x).split()))
test["question"] = test["question"].apply(lambda x: " ".join(str(x).split()))

args = Config()
_, tokenizer, _ = make_model(args)  # tokenizer is needed for feature creation

test_features = prepare_test_features(args, tokenizer, test)

test_dataset = DatasetRetriever(test_features, mode="test")
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.eval_batch_size,
    sampler=SequentialSampler(test_dataset),
    num_workers=0,  # avoid overhead of many workers during inference
    pin_memory=True,
    drop_last=False,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
base_model = (
    "../input/chaii-xlmr-5-fold/output/"  # kept for compatibility; not used for loading
)


def get_ensemble_predictions(checkpoint_paths):
    """
    Loads the model once, iterates over the provided checkpoint files,
    runs inference on the full test dataloader for each checkpoint,
    and returns the averaged start/end logits.
    """
    config, tokenizer_local, model = make_model(Config())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if hasattr(torch, "compile"):
        model = torch.compile(model)

    model.eval()

    total_examples = len(test_dataset)
    seq_len = args.max_seq_length  # token length used for padding

    sum_start = torch.zeros(
        (total_examples, seq_len), device=device, dtype=torch.float32
    )
    sum_end = torch.zeros((total_examples, seq_len), device=device, dtype=torch.float32)

    for checkpoint_path in checkpoint_paths:
        checkpoint_file = os.path.join(base_model, checkpoint_path)
        if os.path.isfile(checkpoint_file):
            model.load_state_dict(torch.load(checkpoint_file, map_location=device))
        else:
            print(
                f"Warning: checkpoint {checkpoint_file} not found – using random weights."
            )

        offset = 0
        for batch in test_dataloader:
            with torch.inference_mode():
                inputs_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                outputs_start, outputs_end = model(inputs_ids, attention_mask)

                batch_sz = outputs_start.size(0)
                sum_start[offset : offset + batch_sz] += outputs_start
                sum_end[offset : offset + batch_sz] += outputs_end
                offset += batch_sz

        torch.cuda.empty_cache()

    avg_start = (sum_start / len(checkpoint_paths)).cpu().numpy()
    avg_end = (sum_end / len(checkpoint_paths)).cpu().numpy()

    del model, tokenizer_local, config
    gc.collect()
    return avg_start, avg_end




## === cell 10
checkpoint_files = [
    "checkpoint-fold-0/pytorch_model.bin",
    "checkpoint-fold-1/pytorch_model.bin",
    "checkpoint-fold-2/pytorch_model.bin",
    "checkpoint-fold-3/pytorch_model.bin",
    "checkpoint-fold-4/pytorch_model.bin",
]

start_logits, end_logits = get_ensemble_predictions(checkpoint_files)

fin_preds = postprocess_qa_predictions(
    test, test_features, (start_logits, end_logits), tokenizer
)

submission = []
for pid, pred in fin_preds.items():
    pred_clean = " ".join(pred.split())
    pred_clean = pred_clean.strip(punctuation)
    submission.append((pid, pred_clean))

sample = pd.DataFrame(submission, columns=["id", "PredictionString"])
test_data = pd.merge(left=test, right=sample, on="id")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in collate(batch, collate_fn_map)
    222                     for i, samples in enumerate(transposed):
--> 223                         clone[i] = collate(samples, collate_fn_map=collate_fn_map)
    224                     return clone

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in collate(batch, collate_fn_map)
    239 
--> 240     raise TypeError(default_collate_err_msg_format.format(elem_type))
    241 

TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'NoneType'>

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in collate(batch, collate_fn_map)
    170                 clone.update(
--> 171                     {
    172                         key: collate(

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in <dictcomp>(.0)
    171                     {
--> 172                         key: collate(
    173                             [d[key] for d in batch], collate_fn_map=collate_fn_map

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in collate(batch, collate_fn_map)
    234                 # or `__init__(iterable)` (e.g., `range`).
--> 235                 return [
    236                     collate(samples, collate_fn_map=collate_fn_map)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in <listcomp>(.0)
    235                 return [
--> 236                     collate(samples, collate_fn_map=collate_fn_map)
    237                     for samples in transposed

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in collate(batch, collate_fn_map)
    239 
--> 240     raise TypeError(default_collate_err_msg_format.format(elem_type))
    241 

TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'NoneType'>

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in collate(batch, collate_fn_map)
    222                     for i, samples in enumerate(transposed):
--> 223                         clone[i] = collate(samples, collate_fn_map=collate_fn_map)
    224                     return clone

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in collate(batch, collate_fn_map)
    239 
--> 240     raise TypeError(default_collate_err_msg_format.format(elem_type))
    241 

TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'NoneType'>

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3335785020.py in <cell line: 0>()
      7 ]
      8 
----> 9 start_logits, end_logits = get_ensemble_predictions(checkpoint_files)
     10 
     11 fin_preds = postprocess_qa_predictions(

/tmp/ipykernel_55/332099160.py in get_ensemble_predictions(checkpoint_paths)
     37 
     38         offset = 0
---> 39         for batch in test_dataloader:
     40             with torch.inference_mode():
     41                 inputs_ids = batch["input_ids"].to(device, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     53         else:
     54             data = self.dataset[possibly_batched_index]
---> 55         return self.collate_fn(data)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in default_collate(batch)
    396         >>> default_collate(batch)  # Handle `CustomType` automatically
    397     """
--> 398     return collate(batch, collate_fn_map=default_collate_fn_map)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in collate(batch, collate_fn_map)
    189             # The mapping type may not support `copy()` / `update(mapping)`
    190             # or `__init__(iterable)`.
--> 191             return {
    192                 key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map)
    193                 for key in elem

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in <dictcomp>(.0)
    190             # or `__init__(iterable)`.
    191             return {
--> 192                 key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map)
    193                 for key in elem
    194             }

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in collate(batch, collate_fn_map)
    233                 # The sequence type may not support `copy()` / `__setitem__(index, item)`
    234                 # or `__init__(iterable)` (e.g., `range`).
--> 235                 return [
    236                     collate(samples, collate_fn_map=collate_fn_map)
    237                     for samples in transposed

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in <listcomp>(.0)
    234                 # or `__init__(iterable)` (e.g., `range`).
    235                 return [
--> 236                     collate(samples, collate_fn_map=collate_fn_map)
    237                     for samples in transposed
    238                 ]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in collate(batch, collate_fn_map)
    238                 ]
    239 
--> 240     raise TypeError(default_collate_err_msg_format.format(elem_type))
    241 
    242 

TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'NoneType'>

## === cell 11
bad_starts = [".", ",", "(", ")", "-", "–", ";"]
bad_endings = ["...", "-", "(", ")", "–", ",", ";"]

tamil_ad = "கி.பி"
tamil_bc = "கி.மு"
tamil_km = "கி.மீ"
hindi_ad = "ई"
hindi_bc = "ई.पू"

cleaned_preds = []
for pred, context in test_data[["PredictionString", "context"]].to_numpy():
    if not isinstance(pred, str):
        cleaned_preds.append("")
        continue
    if pred == "":
        cleaned_preds.append(pred)
        continue
    while any(pred.startswith(y) for y in bad_starts):
        pred = pred[1:]
    while any(pred.endswith(y) for y in bad_endings):
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

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3085637239.py in <cell line: 0>()
      9 
     10 cleaned_preds = []
---> 11 for pred, context in test_data[["PredictionString", "context"]].to_numpy():
     12     if not isinstance(pred, str):
     13         cleaned_preds.append("")

NameError: name 'test_data' is not defined
