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

0.7240094542503357

# 6. Current score

0.57411

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00699) has done: 'I fix the tokenizer call by removing the unsupported argument, make checkpoint loading optional so the script can run with the pretrained model, and ensure the post‑processing function has access to the tokenizer. These changes resolve the runtime errors and let the notebook generate a valid `submission.csv` file.'
- What this solution (achieved 0.57411) has done: 'I fixed the runtime crash by simplifying the prediction routine: the model is run on the available device without CUDA‑specific calls or AMP, and the outputs are accessed via the proper `start_logits` / `end_logits` fields. This removes the protobuf/AMP error, ensures `fin_preds` is defined, and lets the notebook write a valid `submission.csv` file. No core‑logic changes were made, so the scoring behavior remains unchanged.'
- What this solution (achieved 0.57411) has done: 'The fix handles the crash caused by passing a `None` config name to `AutoConfig.from_pretrained`. We now only load a config when a valid name/path is provided and fall back to the model name otherwise. This prevents the protobuf `MessageFactory` error and lets the script run end‑to‑end, producing a proper `submission.csv` while keeping the core logic unchanged.'
- What this solution (achieved 0.57411) has done: 'Implemented fixes to resolve runtime errors and ensure a valid submission is generated. Added a proper model loading routine in the prediction function, removed the undefined `Make_Model` call, and kept the rest of the workflow unchanged so predictions are produced and written to `submission.csv`. This correction enables end‑to‑end execution and creates the required output file.'

# 9. Code solution

## === cell 0
class Configration:
    model_type = "xlm_roberta"
    XLMR_name_or_path = "deepset/xlm-roberta-base-squad2"
    MURIL_name_or_path = "google/muril-base-cased"
    MPNET2_name_or_path = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
    MPNET3_name_or_path = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

    XLMR_config_name = None
    MURIL_config_name = None
    MPNET2_config_name = None
    MPNET3_config_name = None

    fp16 = False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2

    XLMR_tokenizer_name = None
    MURIL_tokenizer_name = None
    MPNET2_tokenizer_name = None
    MPNET3_tokenizer_name = None

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




## === cell 1
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass  # If protobuf is not available, the later imports will raise appropriately

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
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torch.optim as optim

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    logging,
)

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
    optimal_value = min(num_cpus, num_gpus * 4) if num_gpus else num_cpus - 1
    return optimal_value


print(f"CUDA available: {torch.cuda.is_available()}")
MODEL_CONFIG_CLASSES = list(transformers.MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

torch.backends.cudnn.benchmark = True




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
            0 if idx is None else idx for idx in tokenized_example.sequence_ids(i)
        ]
        features.append(feature)
    return features




## === cell 4
def Postprocess_qa_predictions(
    examples, features, raw_predictions, n_best_size=20, max_answer_length=30
):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()
    print(
        f"Post-processing {len(examples)} examples split into {len(features)} features."
    )
    for example_index, example in examples.iterrows():
        feature_indices = features_per_example[example_index]
        min_null_score = None
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
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score
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




## === cell 5
test_df = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/test.csv")
test_df["context"] = test_df["context"].apply(lambda x: " ".join(str(x).split()))
test_df["question"] = test_df["question"].apply(lambda x: " ".join(str(x).split()))




## === cell 6
args = Configration()
try:
    tokenizer = AutoTokenizer.from_pretrained(args.XLMR_tokenizer_name)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(args.XLMR_name_or_path)

questions = test_df["question"].apply(lambda x: x.lstrip()).tolist()
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

test_features = []
overflow_to_sample = tokenized["overflow_to_sample_mapping"]
for i in range(len(tokenized["input_ids"])):
    orig_idx = overflow_to_sample[i]
    feature = {
        "language": test_df.iloc[orig_idx]["language"],
        "example_id": test_df.iloc[orig_idx]["id"],
        "context": test_df.iloc[orig_idx]["context"],
        "question": test_df.iloc[orig_idx]["question"],
        "input_ids": tokenized["input_ids"][i],
        "attention_mask": tokenized["attention_mask"][i],
        "offset_mapping": tokenized["offset_mapping"][i],
        "sequence_ids": [
            0 if idx is None else idx for idx in tokenized.sequence_ids(i)
        ],
    }
    test_features.append(feature)

test_dataset = Dataset_Retriever(test_features, mode="test")
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.eval_batch_size,
    sampler=SequentialSampler(test_dataset),
    num_workers=0,
    pin_memory=True,
    drop_last=False,
)




## === cell 7
def Get_Predictions(checkpoint_path=None):
    """
    Load the pretrained QA model, run it on the test set, and
    collect start/end logits. This implementation avoids the
    undefined Make_Model call and works on CPU/GPU without AMP.
    """
    args = Configration()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForQuestionAnswering.from_pretrained(args.XLMR_name_or_path)
    model.to(device)
    model.eval()

    if checkpoint_path and os.path.isfile(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        except Exception as e:
            print(f"Failed to load checkpoint {checkpoint_path}: {e}")

    start_logits_list, end_logits_list = [], []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits.detach().cpu().numpy()
            end_logits = outputs.end_logits.detach().cpu().numpy()

            start_logits_list.append(start_logits)
            end_logits_list.append(end_logits)

    start_arr = np.concatenate(start_logits_list, axis=0)
    end_arr = np.concatenate(end_logits_list, axis=0)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return start_arr, end_arr




## === cell 8
start_logits, end_logits = Get_Predictions()  # no checkpoint path

fin_preds = Postprocess_qa_predictions(
    test_df, test_features, (start_logits, end_logits)
)




## === cell 9
submission_rows = []
for pid, pred in fin_preds.items():
    pred_clean = " ".join(str(pred).split())
    pred_clean = pred_clean.strip(punctuation)
    submission_rows.append((pid, pred_clean))

submission_df = pd.DataFrame(submission_rows, columns=["id", "PredictionString"])
submission_df.to_csv("submission.csv", index=False)




## === cell 10
print("Submission file saved to 'submission.csv'.")
