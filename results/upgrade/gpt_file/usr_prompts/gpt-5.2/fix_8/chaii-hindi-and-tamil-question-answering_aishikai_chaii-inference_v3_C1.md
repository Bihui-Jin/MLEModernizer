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
    if torch.cuda.is_available():
        return min(4, max(1, num_cpus // 2))
    return min(2, max(1, num_cpus - 1))


print(f"Apex AMP Installed :: {APEX_INSTALLED}")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

fix_all_seeds(2021)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

_CAN_COMPILE = hasattr(torch, "compile")




## === cell 1
class Configration:
    model_type = "xlm_roberta"

    _KAGGLE_DEFAULT_MODEL_DIR = "../input/nothing-of-your-concern/xlm-roberta-large-squad2/xlm-roberta-large-squad2"
    _HF_FALLBACK_MODEL_ID = "deepset/xlm-roberta-large-squad2"

    XLMR_name_or_path = (
        _KAGGLE_DEFAULT_MODEL_DIR
        if os.path.exists(_KAGGLE_DEFAULT_MODEL_DIR)
        else _HF_FALLBACK_MODEL_ID
    )
    XLMR_config_name = (
        os.path.join(_KAGGLE_DEFAULT_MODEL_DIR, "config.json")
        if os.path.exists(_KAGGLE_DEFAULT_MODEL_DIR)
        else _HF_FALLBACK_MODEL_ID
    )
    XLMR_tokenizer_name = (
        _KAGGLE_DEFAULT_MODEL_DIR
        if os.path.exists(_KAGGLE_DEFAULT_MODEL_DIR)
        else _HF_FALLBACK_MODEL_ID
    )

    MURIL_name_or_path = (
        "../input/nothing-of-your-concern/muril-large-cased/muril-large-cased"
    )
    MPNET2_name_or_path = "../input/nothing-of-your-concern/multi-qa-mpnet-base-cos-v1/multi-qa-mpnet-base-cos-v1"
    MPNET3_name_or_path = "../input/nothing-of-your-concern/multi-qa-mpnet-base-dot-v1/multi-qa-mpnet-base-dot-v1"
    MURIL_config_name = "../input/nothing-of-your-concern/muril-large-cased/muril-large-cased/config.json"
    MPNET2_config_name = "../input/nothing-of-your-concern/multi-qa-mpnet-base-cos-v1/multi-qa-mpnet-base-cos-v1/config.json"
    MPNET3_config_name = "../input/nothing-of-your-concern/multi-qa-mpnet-base-dot-v1/multi-qa-mpnet-base-dot-v1/config.json"
    MURIL_tokenizer_name = (
        "../input/nothing-of-your-concern/muril-large-cased/muril-large-cased/"
    )
    MPNET2_tokenizer_name = "../input/nothing-of-your-concern/multi-qa-mpnet-base-cos-v1/multi-qa-mpnet-base-cos-v1/"
    MPNET3_tokenizer_name = "../input/nothing-of-your-concern/multi-qa-mpnet-base-dot-v1/multi-qa-mpnet-base-dot-v1/"

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
def Make_Model(args):
    config = AutoConfig.from_pretrained(args.XLMR_config_name)
    tokenizer = AutoTokenizer.from_pretrained(args.XLMR_tokenizer_name, use_fast=True)
    model = Model(args.XLMR_name_or_path, config=config)
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




## === cell 7
def _build_feature_cache(features, cls_token_id):
    feat_example_ids = np.empty(len(features), dtype=object)
    feat_ctx_offsets = [None] * len(features)
    feat_cls_index = np.empty(len(features), dtype=np.int32)
    feat_valid_ctx_idx = [None] * len(features)

    for i, f in enumerate(features):
        feat_example_ids[i] = f["example_id"]
        seq_ids = f["sequence_ids"]
        offs = f["offset_mapping"]

        ctx_offs = [o if seq_ids[k] == 1 else None for k, o in enumerate(offs)]
        feat_ctx_offsets[i] = ctx_offs

        feat_valid_ctx_idx[i] = np.fromiter(
            (k for k, o in enumerate(ctx_offs) if o is not None),
            dtype=np.int32,
        )

        if cls_token_id is not None:
            try:
                feat_cls_index[i] = f["input_ids"].index(cls_token_id)
            except ValueError:
                feat_cls_index[i] = 0
        else:
            feat_cls_index[i] = 0

    return feat_example_ids, feat_ctx_offsets, feat_cls_index, feat_valid_ctx_idx


def _topk_indices_desc(arr1d, k):
    if k >= arr1d.shape[0]:
        return np.argsort(arr1d)[::-1]
    idx = np.argpartition(arr1d, -k)[-k:]
    return idx[np.argsort(arr1d[idx])[::-1]]


def Postprocess_qa_predictions(
    examples, features, raw_predictions, tokenizer, n_best_size=20, max_answer_length=30
):
    all_start_logits, all_end_logits = raw_predictions

    ex_ids = examples["id"].tolist()
    example_id_to_index = {k: i for i, k in enumerate(ex_ids)}

    features_per_example = collections.defaultdict(list)
    for i, f_ex_id in enumerate(_feat_example_ids):
        features_per_example[example_id_to_index[f_ex_id]].append(i)

    predictions = collections.OrderedDict()
    print(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )

    feat_ctx_offsets = _feat_ctx_offsets
    feat_valid_idx = _feat_valid_ctx_idx
    start_logits_all = all_start_logits
    end_logits_all = all_end_logits
    topk = _topk_indices_desc

    for ex in examples.itertuples(index=True):
        example_index = ex.Index
        ex_id = ex.id
        context = ex.context

        feature_indices = features_per_example[example_index]
        best_score = None
        best_text = ""

        for feature_index in feature_indices:
            start_logits = start_logits_all[feature_index]
            end_logits = end_logits_all[feature_index]
            offset_mapping = feat_ctx_offsets[feature_index]
            valid_idx = feat_valid_idx[feature_index]
            if valid_idx.size == 0:
                continue

            s_idx_local = topk(
                start_logits[valid_idx], min(n_best_size, valid_idx.size)
            )
            e_idx_local = topk(end_logits[valid_idx], min(n_best_size, valid_idx.size))
            start_indexes = valid_idx[s_idx_local]
            end_indexes = valid_idx[e_idx_local]

            for start_index in start_indexes.tolist():
                start_char = offset_mapping[start_index][0]
                for end_index in end_indexes.tolist():
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    end_char = offset_mapping[end_index][1]
                    score = float(start_logits[start_index] + end_logits[end_index])
                    if best_score is None or score > best_score:
                        best_score = score
                        best_text = context[start_char:end_char]

        predictions[ex_id] = best_text if best_score is not None else ""

    return predictions




## === cell 8
test_path = "../input/chaii-hindi-and-tamil-question-answering/test.csv"
if not os.path.exists(test_path):
    test_path = "/kaggle/input/chaii-hindi-and-tamil-question-answering/test.csv"
test_df = pd.read_csv(test_path)



## === cell 9
test_df["context"] = test_df["context"].astype(str).str.split().str.join(" ")
test_df["question"] = test_df["question"].astype(str).str.split().str.join(" ")



## === cell 10
args = Configration()
tokenizer = AutoTokenizer.from_pretrained(args.XLMR_tokenizer_name, use_fast=True)

questions = test_df["question"].astype(str).str.lstrip().tolist()
contexts = test_df["context"].astype(str).tolist()
example_ids = test_df["id"].tolist()

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
test_features = []
for i in range(len(tokenized["input_ids"])):
    sample_idx = int(overflow_to_sample[i])
    ex_id = example_ids[sample_idx]
    seq_ids = tokenized.sequence_ids(i)
    test_features.append(
        {
            "example_id": ex_id,
            "context": contexts[sample_idx],
            "question": questions[sample_idx],
            "input_ids": tokenized["input_ids"][i],
            "attention_mask": tokenized["attention_mask"][i],
            "offset_mapping": tokenized["offset_mapping"][i],
            "sequence_ids": [0 if s is None else s for s in seq_ids],
        }
    )

_feat_example_ids, _feat_ctx_offsets, _feat_cls_index, _feat_valid_ctx_idx = (
    _build_feature_cache(test_features, tokenizer.cls_token_id)
)

test_dataset = Dataset_Retriever(test_features, mode="test")

nw = optimal_num_of_loader_workers()
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.eval_batch_size,
    sampler=SequentialSampler(test_dataset),
    num_workers=nw,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(nw > 0),
    prefetch_factor=2 if nw > 0 else None,
    drop_last=False,
)

print("Num test examples:", len(test_df), "Num test features:", len(test_features))



## === cell 11
base_model = "../input/chaii-helper/XLM-Roberta-HT/content/drive/MyDrive/output/"




## === cell 12
@torch.inference_mode()
def Get_Predictions_Ensemble(checkpoint_paths):
    config = AutoConfig.from_pretrained(args.XLMR_config_name)
    model = Model(args.XLMR_name_or_path, config=config).to(DEVICE)
    model.eval()

    if _CAN_COMPILE:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

    n_feat = len(test_dataset)
    seq_len = args.max_seq_length

    if torch.cuda.is_available():
        sum_start_t = torch.zeros(
            (n_feat, seq_len), dtype=torch.float32, pin_memory=True
        )
        sum_end_t = torch.zeros((n_feat, seq_len), dtype=torch.float32, pin_memory=True)
    else:
        sum_start_t = torch.zeros((n_feat, seq_len), dtype=torch.float32)
        sum_end_t = torch.zeros((n_feat, seq_len), dtype=torch.float32)

    use_cuda_amp = torch.cuda.is_available()
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_cuda_amp
        else None
    )

    cpu_states = []
    for checkpoint_path in checkpoint_paths:
        ckpt_full = os.path.join(base_model, checkpoint_path)
        if os.path.exists(ckpt_full):
            cpu_states.append(torch.load(ckpt_full, map_location="cpu"))
        else:
            print(f"Warning: checkpoint not found: {ckpt_full}. Using current weights.")
            cpu_states.append(None)

    for st in cpu_states:
        if st is not None:
            model.load_state_dict(st, strict=False)

        write_pos = 0
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            bs = input_ids.size(0)

            if amp_ctx is not None:
                with amp_ctx:
                    s, e = model(input_ids, attention_mask)
            else:
                s, e = model(input_ids, attention_mask)

            sum_start_t[write_pos : write_pos + bs].add_(
                s.detach().to(device="cpu", dtype=torch.float32)
            )
            sum_end_t[write_pos : write_pos + bs].add_(
                e.detach().to(device="cpu", dtype=torch.float32)
            )
            write_pos += bs

    n_models = float(len(checkpoint_paths))
    out_start = (sum_start_t / n_models).numpy()
    out_end = (sum_end_t / n_models).numpy()

    del model, config, cpu_states, sum_start_t, sum_end_t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_start, out_end




## === cell 13
start_logits, end_logits = Get_Predictions_Ensemble(
    [
        "checkpoint-fold-0/pytorch_model.bin",
        "checkpoint-fold-1/pytorch_model.bin",
        "checkpoint-fold-2/pytorch_model.bin",
        "checkpoint-fold-3/pytorch_model.bin",
        "checkpoint-fold-4/pytorch_model.bin",
    ]
)

fin_preds = Postprocess_qa_predictions(
    test_df, test_features, (start_logits, end_logits), tokenizer=tokenizer
)

submission = []
strip_punct = punctuation  # minor speed: localize
for ex_id, pred in fin_preds.items():
    pred = " ".join(str(pred).split())
    pred = pred.strip(strip_punct)
    submission.append((ex_id, pred))

sample = pd.DataFrame(submission, columns=["id", "PredictionString"])

test_data = test_df[["id", "context"]].merge(sample, on="id", how="left")
test_data["PredictionString"] = test_data["PredictionString"].fillna("")

print("Predictions prepared:", test_data.shape)



## === cell 14
bad_starts = [".", ",", "(", ")", "-", "–", ",", ";"]
bad_endings = ["...", "-", "(", ")", "–", ",", ";"]

tamil_ad = "கி.பி"
tamil_bc = "கி.மு"
tamil_km = "கி.மீ"
hindi_ad = "ई"
hindi_bc = "ई.पू"

bad_starts_t = tuple(bad_starts)
bad_endings_t = tuple([x for x in bad_endings if x != "..."])  # handle "..." separately

cleaned_preds = []
for pred, context in test_data[["PredictionString", "context"]].to_numpy():
    pred = str(pred)
    context = str(context)
    if pred == "":
        cleaned_preds.append(pred)
        continue

    while pred and pred.startswith(bad_starts_t):
        pred = pred[1:]

    while pred and (pred.endswith("...") or pred.endswith(bad_endings_t)):
        if pred.endswith("..."):
            pred = pred[:-3]
        else:
            pred = pred[:-1]

    if pred.endswith("..."):
        pred = pred[:-3]

    if (
        pred.endswith(tamil_ad)
        or pred.endswith(tamil_bc)
        or pred.endswith(tamil_km)
        or pred.endswith(hindi_ad)
        or pred.endswith(hindi_bc)
    ) and (pred + ".") in context:
        pred = pred + "."

    cleaned_preds.append(pred)

test_data["PredictionString"] = cleaned_preds

out_path = "submission.csv"
test_data[["id", "PredictionString"]].to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(test_data))



## === cell 15
test_data.head()
