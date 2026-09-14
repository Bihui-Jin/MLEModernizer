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
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

APEX_INSTALLED = False
print(f"Apex AMP Installed :: {APEX_INSTALLED}")

import gc

gc.enable()
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm
from string import punctuation
import multiprocessing

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    logging,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
)

logging.set_verbosity_error()
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
    def __init__(self, features, mode="test"):
        self.features = features
        self.mode = mode

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        f = self.features[idx]
        if self.mode == "test":
            return {
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"],
            }
        else:
            return {
                "input_ids": torch.tensor(f["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(f["attention_mask"], dtype=torch.long),
                "offset_mapping": torch.tensor(f["offset_mapping"], dtype=torch.long),
                "start_position": torch.tensor(f["start_position"], dtype=torch.long),
                "end_position": torch.tensor(f["end_position"], dtype=torch.long),
            }




## === cell 4
class Model(nn.Module):
    def __init__(self, modelname_or_path, config):
        super().__init__()
        self.config = config  # store config for weight init
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
        return start_logits.squeeze(-1), end_logits.squeeze(-1)




## === cell 5
def make_model(args):
    config = AutoConfig.from_pretrained(args.model_name_or_path, local_files_only=False)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, local_files_only=False, use_fast=False
    )
    model = Model(args.model_name_or_path, config=config)
    return config, tokenizer, model




## === cell 6
def prepare_test_features(args, tokenizer, df):
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

        features.append(
            {
                "example_id": example["id"],
                "context": example["context"],
                "question": example["question"].lstrip(),
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "offset_mapping_context": offset_mapping_context,
                "sequence_ids": [0 if idx is None else idx for idx in seq_ids],
                "cls_index": cls_index,
            }
        )
    return features




## === cell 7
test = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/test.csv")
test["context"] = test["context"].apply(lambda x: " ".join(str(x).split()))
test["question"] = test["question"].apply(lambda x: " ".join(str(x).split()))

args = Config()
_, tokenizer, _ = make_model(args)  # we only need the tokenizer here

test_features = prepare_test_features(args, tokenizer, test)

test_dataset = DatasetRetriever(test_features, mode="test")
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.eval_batch_size,
    sampler=SequentialSampler(test_dataset),
    num_workers=optimal_num_of_loader_workers(),
    pin_memory=True,
    collate_fn=lambda batch: {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    },
    drop_last=False,
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
base_model = "../input/chaii-xlmr-5-fold/output/"


def get_ensemble_predictions(checkpoint_paths):
    config, tokenizer_local, model = make_model(Config())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_examples = len(test_dataset)
    seq_len = args.max_seq_length

    sum_start = torch.zeros(
        (total_examples, seq_len), device=device, dtype=torch.float32
    )
    sum_end = torch.zeros((total_examples, seq_len), device=device, dtype=torch.float32)

    for ckpt in checkpoint_paths:
        ckpt_file = os.path.join(base_model, ckpt)
        if os.path.isfile(ckpt_file):
            model.load_state_dict(torch.load(ckpt_file, map_location=device))
        else:
            print(f"Warning: {ckpt_file} not found – using random weights.")

        offset = 0
        for batch in test_dataloader:
            with torch.inference_mode():
                ids = batch["input_ids"].to(device, non_blocking=True)
                mask = batch["attention_mask"].to(device, non_blocking=True)
                start_logits, end_logits = model(ids, mask)

                bs = start_logits.size(0)
                sum_start[offset : offset + bs] += start_logits
                sum_end[offset : offset + bs] += end_logits
                offset += bs

        torch.cuda.empty_cache()

    avg_start = (sum_start / len(checkpoint_paths)).cpu().numpy()
    avg_end = (sum_end / len(checkpoint_paths)).cpu().numpy()

    del model, tokenizer_local, config
    gc.collect()
    return avg_start, avg_end




## === cell 9
def postprocess_qa_predictions(
    examples,
    features,
    raw_predictions,
    tokenizer,
    n_best_size=20,
    max_answer_length=30,
):
    """
    Simple post‑processing that maps start/end logits back to answer strings.
    """
    start_logits, end_logits = raw_predictions
    example_id_to_index = {ex_id: i for i, ex_id in enumerate(examples["id"])}
    features_per_example = {}
    for i, f in enumerate(features):
        eid = f["example_id"]
        features_per_example.setdefault(eid, []).append(i)

    predictions = {}
    for example_id, feature_indices in features_per_example.items():
        prelims = []
        for idx in feature_indices:
            start_log = start_logits[idx]
            end_log = end_logits[idx]
            offsets = features[idx]["offset_mapping_context"]
            cls_index = features[idx]["cls_index"]

            start_log[cls_index] = -1e9
            end_log[cls_index] = -1e9

            start_top = np.argsort(start_log)[-n_best_size:]
            end_top = np.argsort(end_log)[-n_best_size:]
            for s in start_top:
                for e in end_top:
                    if s > e:
                        continue
                    if e - s + 1 > max_answer_length:
                        continue
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    prelims.append(
                        {
                            "score": start_log[s] + end_log[e],
                            "start": s,
                            "end": e,
                        }
                    )
        if prelims:
            best = max(prelims, key=lambda x: x["score"])
            s_offset, e_offset = offsets[best["start"]][0], offsets[best["end"]][1]
            context = features[feature_indices[0]]["context"]
            pred_text = context[s_offset:e_offset]
        else:
            pred_text = ""
        predictions[example_id] = pred_text
    return predictions




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

cleaned = []
for pid, pred in fin_preds.items():
    pred_clean = " ".join(str(pred).split())
    pred_clean = pred_clean.strip(punctuation)
    cleaned.append((pid, pred_clean))

submission_df = pd.DataFrame(cleaned, columns=["id", "PredictionString"])
submission_df.to_csv("submission.csv", index=False)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4136659239.py in <cell line: 0>()
      7 ]
      8 
----> 9 start_logits, end_logits = get_ensemble_predictions(checkpoint_files)
     10 
     11 fin_preds = postprocess_qa_predictions(

/tmp/ipykernel_55/161680726.py in get_ensemble_predictions(checkpoint_paths)
      8     model.eval()
      9 
---> 10     total_examples = len(test_dataset)
     11     seq_len = args.max_seq_length
     12 

NameError: name 'test_dataset' is not defined
