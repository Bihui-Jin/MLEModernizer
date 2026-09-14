# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
)

transformers.logging.set_verbosity_warning()
transformers.logging.set_verbosity_error()


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
    optimal_value = min(num_cpus, num_gpus * 4) if num_gpus else num_cpus - 1
    return optimal_value


print(f"Apex AMP Installed :: {APEX_INSTALLED}")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

torch.backends.cudnn.benchmark = True




## === cell 1
class Config:
    model_type = "xlm-roberta"
    model_name_or_path = "deepset/xlm-roberta-base-squad2"
    config_name = "deepset/xlm-roberta-base-squad2"
    fp16 = True if APEX_INSTALLED else False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2

    tokenizer_name = "deepset/xlm-roberta-base-squad2"
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

    def forward(
        self,
        input_ids,
        attention_mask=None,
    ):
        outputs = self.xlm_roberta(input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        linear_output = self.linear_layer(sequence_output)
        linear_output = self.dropout(linear_output)
        qa_logits = self.qa_outputs(linear_output)

        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits




## === cell 4
def make_model(args):
    try:
        config = AutoConfig.from_pretrained(args.config_name)
    except Exception as e:
        print(
            f"Config load failed ({e}); falling back to 'deepset/xlm-roberta-base-squad2'"
        )
        config = AutoConfig.from_pretrained("deepset/xlm-roberta-base-squad2")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    except Exception as e:
        print(
            f"Tokenizer load failed ({e}); falling back to 'deepset/xlm-roberta-base-squad2'"
        )
        tokenizer = AutoTokenizer.from_pretrained("deepset/xlm-roberta-base-squad2")
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
            0 if i is None else i for i in tokenized_example.sequence_ids(i)
        ]
        features.append(feature)
    return features




## === cell 6
import collections
from functools import partial


def _process_single_example(
    example_idx,
    example_row,
    features,
    start_logits,
    end_logits,
    tokenizer,
    n_best_size,
    max_answer_length,
):
    feature_indices = features[example_idx]

    min_null_score = None
    valid_answers = []

    context = example_row["context"]
    for feature_index in feature_indices:
        start_log = start_logits[feature_index]
        end_log = end_logits[feature_index]

        sequence_ids = features["sequence_ids"][feature_index]
        context_index = 1

        offset_mapping = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(features["offset_mapping"][feature_index])
        ]

        cls_index = features["input_ids"][feature_index].index(tokenizer.cls_token_id)
        feature_null_score = start_log[cls_index] + end_log[cls_index]
        if (min_null_score is None) or (min_null_score < feature_null_score):
            min_null_score = feature_null_score

        start_indexes = np.argsort(start_log)[-1 : -n_best_size - 1 : -1].tolist()
        end_indexes = np.argsort(end_log)[-1 : -n_best_size - 1 : -1].tolist()
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
                        "score": start_log[start_index] + end_log[end_index],
                        "text": context[start_char:end_char],
                    }
                )

    if len(valid_answers) > 0:
        best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
    else:
        best_answer = {"text": "", "score": 0.0}
    return example_row["id"], best_answer["text"]


def postprocess_qa_predictions(
    examples, features, raw_predictions, n_best_size=20, max_answer_length=30
):
    """
    Sequential version of post‑processing to avoid multiprocessing overhead,
    preserving the exact original logic.
    """
    all_start_logits, all_end_logits = raw_predictions

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, f in enumerate(features):
        features_per_example[example_id_to_index[f["example_id"]]].append(i)

    feature_store = {
        "input_ids": [f["input_ids"] for f in features],
        "offset_mapping": [f["offset_mapping"] for f in features],
        "sequence_ids": [f["sequence_ids"] for f in features],
    }

    results = []
    for example_index, example_row in examples.iterrows():
        pred_id, pred_text = _process_single_example(
            example_index,
            example_row,
            features_per_example,
            all_start_logits,
            all_end_logits,
            tokenizer,
            n_best_size,
            max_answer_length,
        )
        results.append((pred_id, pred_text))

    predictions = collections.OrderedDict(results)
    print(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )
    return predictions




## === cell 7
test = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/test.csv")

test["context"] = test["context"].apply(lambda x: " ".join(x.split()))
test["question"] = test["question"].apply(lambda x: " ".join(x.split()))

args = Config()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

examples = test[["id", "context", "question"]].to_dict(orient="records")
questions = [ex["question"] for ex in examples]
contexts = [ex["context"] for ex in examples]

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

test_features = []
overflow_to_sample = tokenized["overflow_to_sample_mapping"]
for i in range(len(tokenized["input_ids"])):
    sample_idx = overflow_to_sample[i]
    example = examples[sample_idx]
    feature = {
        "example_id": example["id"],
        "context": example["context"],
        "question": example["question"],
        "input_ids": tokenized["input_ids"][i],
        "attention_mask": tokenized["attention_mask"][i],
        "offset_mapping": tokenized["offset_mapping"][i],
        "sequence_ids": [
            0 if sid is None else sid for sid in tokenized.sequence_ids(i)
        ],
    }
    test_features.append(feature)

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
def get_predictions(checkpoint_path):
    config, tokenizer, model = make_model(Config())
    model.cuda()
    model.eval()  # ensure eval mode for speed
    if os.path.isfile(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            print("Loaded checkpoint.")
        except Exception as e:
            print(f"Failed to load checkpoint ({e}); using pretrained model.")
    else:
        print("Checkpoint not found; using pretrained model.")

    start_logits = []
    end_logits = []
    amp_enabled = Config().fp16 and torch.cuda.is_available()
    for batch in test_dataloader:
        with torch.no_grad():
            if amp_enabled:
                with torch.cuda.amp.autocast():
                    outputs_start, outputs_end = model(
                        batch["input_ids"].cuda(),
                        batch["attention_mask"].cuda(),
                    )
            else:
                outputs_start, outputs_end = model(
                    batch["input_ids"].cuda(),
                    batch["attention_mask"].cuda(),
                )
            start_logits.append(outputs_start.cpu().numpy())
            end_logits.append(outputs_end.cpu().numpy())
            del outputs_start, outputs_end
    del model, tokenizer, config
    gc.collect()
    torch.cuda.empty_cache()
    return np.concatenate(start_logits, axis=0), np.concatenate(end_logits, axis=0)




## === cell 9
start_logits1, end_logits1 = get_predictions(
    "../input/chaii-abhishek-model-2-epochs-10folds/output/checkpoint-fold-9/pytorch_model.bin"
)

fin_preds = postprocess_qa_predictions(
    test, test_features, (start_logits1, end_logits1)
)

submission = []
for p1, p2 in fin_preds.items():
    p2 = " ".join(p2.split())
    p2 = p2.strip(punctuation)
    submission.append((p1, p2))

sample = pd.DataFrame(submission, columns=["id", "PredictionString"])

test_data = pd.merge(left=test, right=sample, on="id")



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
    if pred == "":
        cleaned_preds.append(pred)
        continue
    while any([pred.startswith(y) for y in bad_starts]):
        pred = pred[1:]
    while any([pred.endswith(y) for y in bad_endings]):
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
        and pred + "." in context
    ):
        pred = pred + "."

    cleaned_preds.append(pred)

test_data["PredictionString"] = cleaned_preds
test_data[["id", "PredictionString"]].to_csv("submission.csv", index=False)



## === cell 11
test_data[["id", "PredictionString"]]
