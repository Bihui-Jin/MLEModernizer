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




## === cell 1
class Configration:
    model_type = "xlm_roberta"
    XLMR_name_or_path = "xlm-roberta-base"
    MURIL_name_or_path = "google/muril-base-cased"
    MPNET2_name_or_path = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
    MPNET3_name_or_path = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

    XLMR_config_name = "../input/nothing-of-your-concern/xlm-roberta-large-squad2/xlm-roberta-large-squad2/config.json"
    MURIL_config_name = "../input/nothing-of-your-concern/muril-large-cased/muril-large-cased/config.json"
    MPNET2_config_name = "../input/nothing-of-your-concern/multi-qa-mpnet-base-cos-v1/multi-qa-mpnet-base-cos-v1/config.json"
    MPNET3_config_name = "../input/nothing-of-your-concern/multi-qa-mpnet-base-dot-v1/multi-qa-mpnet-base-dot-v1/config.json"

    fp16 = False
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
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._init_weights(self.qa_outputs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None):
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
def Make_Model(args):
    """
    Load model configuration, tokenizer and the model itself.
    If the supplied config/tokenizer paths are invalid, fall back to the
    HuggingFace model name provided in args.XLMR_name_or_path.
    """
    try:
        config = AutoConfig.from_pretrained(args.XLMR_config_name)
    except Exception:
        config = AutoConfig.from_pretrained(args.XLMR_name_or_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.XLMR_tokenizer_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.XLMR_name_or_path)

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
            0 if idx is None else idx for idx in tokenized_example.sequence_ids(i)
        ]
        features.append(feature)
    return features




## === cell 6
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




## === cell 7
test_df = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/test.csv")
test_df["context"] = test_df["context"].apply(lambda x: " ".join(str(x).split()))
test_df["question"] = test_df["question"].apply(lambda x: " ".join(str(x).split()))




## === cell 8
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
    return_overflow_to_sample_mapping=True,
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
    num_workers=0,  # no extra workers needed; data is already in RAM
    pin_memory=True,
    drop_last=False,
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3067362376.py in <cell line: 0>()
      8 questions = test_df["question"].apply(lambda x: x.lstrip()).tolist()
      9 contexts = test_df["context"].tolist()
---> 10 tokenized = tokenizer(
     11     questions,
     12     contexts,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in __call__(self, text, text_pair, text_target, text_pair_target, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, **kwargs)
   2853             if not self._in_target_context_manager:
   2854                 self._switch_to_input_mode()
-> 2855             encodings = self._call_one(text=text, text_pair=text_pair, **all_kwargs)
   2856         if text_target is not None:
   2857             self._switch_to_target_mode()

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in _call_one(self, text, text_pair, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens, **kwargs)
   2941                 )
   2942             batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
-> 2943             return self.batch_encode_plus(
   2944                 batch_text_or_text_pairs=batch_text_or_text_pairs,
   2945                 add_special_tokens=add_special_tokens,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in batch_encode_plus(self, batch_text_or_text_pairs, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens, **kwargs)
   3142         )
   3143 
-> 3144         return self._batch_encode_plus(
   3145             batch_text_or_text_pairs=batch_text_or_text_pairs,
   3146             add_special_tokens=add_special_tokens,

TypeError: PreTrainedTokenizerFast._batch_encode_plus() got an unexpected keyword argument 'return_overflow_to_sample_mapping'

## === cell 9
def Get_Predictions(checkpoint_path):
    config, tokenizer_local, model = Make_Model(Configration())
    model.cuda()
    if os.path.isfile(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        except Exception as e:
            print(f"Failed to load checkpoint {checkpoint_path}: {e}")
    else:
        print(f"Checkpoint not found: {checkpoint_path} – using pretrained weights.")
    start_logits, end_logits, languages = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs_start, outputs_end = model(input_ids, attention_mask)
            start_logits.append(outputs_start.cpu().numpy())
            end_logits.append(outputs_end.cpu().numpy())
            languages.extend(batch["language"])
    start_arr = np.concatenate(start_logits, axis=0)
    end_arr = np.concatenate(end_logits, axis=0)
    del model, tokenizer_local, config
    torch.cuda.empty_cache()
    gc.collect()
    return start_arr, end_arr, languages




## === cell 10
base_model_hindi = "../input/chaii-helper/XLM-Roberta-H/output/"
base_model_tamil = "../input/nothing-of-your-concern/XLM-Roberta-T/output/"




## === cell 11
start_logits1, end_logits1, languages = Get_Predictions(
    base_model_hindi + "checkpoint-fold-0/pytorch_model.bin"
)
start_logits2, end_logits2, _ = Get_Predictions(
    base_model_hindi + "checkpoint-fold-1/pytorch_model.bin"
)
start_logits3, end_logits3, _ = Get_Predictions(
    base_model_hindi + "checkpoint-fold-2/pytorch_model.bin"
)
start_logits4, end_logits4, _ = Get_Predictions(
    base_model_hindi + "checkpoint-fold-3/pytorch_model.bin"
)
start_logits5, end_logits5, _ = Get_Predictions(
    base_model_hindi + "checkpoint-fold-4/pytorch_model.bin"
)
start_logits1_, end_logits1_, _ = Get_Predictions(
    base_model_tamil + "checkpoint-fold-0/pytorch_model.bin"
)
start_logits2_, end_logits2_, _ = Get_Predictions(
    base_model_tamil + "checkpoint-fold-1/pytorch_model.bin"
)
start_logits3_, end_logits3_, _ = Get_Predictions(
    base_model_tamil + "checkpoint-fold-2/pytorch_model.bin"
)
start_logits4_, end_logits4_, _ = Get_Predictions(
    base_model_tamil + "checkpoint-fold-3/pytorch_model.bin"
)
start_logits5_, end_logits5_, _ = Get_Predictions(
    base_model_tamil + "checkpoint-fold-4/pytorch_model.bin"
)

avg_hindi_start = (
    start_logits1 + start_logits2 + start_logits3 + start_logits4 + start_logits5
) / 5
avg_hindi_end = (
    end_logits1 + end_logits2 + end_logits3 + end_logits4 + end_logits5
) / 5
avg_tamil_start = (
    start_logits1_ + start_logits2_ + start_logits3_ + start_logits4_ + start_logits5_
) / 5
avg_tamil_end = (
    end_logits1_ + end_logits2_ + end_logits3_ + end_logits4_ + end_logits5_
) / 5

selected_start, selected_end = [], []
for idx, lang in enumerate(languages):
    if lang == "hindi":
        selected_start.append(avg_hindi_start[idx])
        selected_end.append(avg_hindi_end[idx])
    else:
        selected_start.append(avg_tamil_start[idx])
        selected_end.append(avg_tamil_end[idx])

selected_start = np.stack(selected_start)
selected_end = np.stack(selected_end)

fin_preds = Postprocess_qa_predictions(
    test_df, test_features, (selected_start, selected_end)
)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 12
submission_rows = []
for pid, pred in fin_preds.items():
    pred_clean = " ".join(str(pred).split())
    pred_clean = pred_clean.strip(punctuation)
    submission_rows.append((pid, pred_clean))

submission_df = pd.DataFrame(submission_rows, columns=["id", "PredictionString"])
submission_df.to_csv("submission.csv", index=False)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2787250849.py in <cell line: 0>()
      1 submission_rows = []
----> 2 for pid, pred in fin_preds.items():
      3     pred_clean = " ".join(str(pred).split())
      4     pred_clean = pred_clean.strip(punctuation)
      5     submission_rows.append((pid, pred_clean))

NameError: name 'fin_preds' is not defined

## === cell 13
print("Submission file saved to 'submission.csv'.")
