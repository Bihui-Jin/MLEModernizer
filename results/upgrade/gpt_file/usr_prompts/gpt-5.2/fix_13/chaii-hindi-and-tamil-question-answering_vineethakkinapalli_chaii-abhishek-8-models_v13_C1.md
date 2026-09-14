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

0.7053247094154358

# 6. Current score

0.06384

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00936) has done: 'You’re timing out mainly because you run full XLM-R inference over many sliding-window “features” (overflow chunks) without any speed-focused settings, and you also keep extra per-item fields in the test Dataset that force Python-object collation overhead. I keep the exact model and post-processing logic, but (1) enable safe, deterministic inference accelerations (`torch.compile` on Torch 2.6 when available, TF32 on Ampere+), (2) remove unused fields from the test Dataset’s `__getitem__` so DataLoader batches don’t carry large Python objects, and (3) make host→device transfers faster with a larger prefetch and pinned memory while keeping determinism. These changes do not alter architecture, tokenization, logits, or post-processing semantics (only negligible float-level differences may occur due to TF32, which matches your allowance).'
- What this solution (achieved 0.0372) has done: 'I fix the runtime crash caused by an incompatibility between `transformers` and the pure-Python protobuf fallback triggered by `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`; the safest minimal fix is to stop forcing the python protobuf implementation so the default (C++/upb) implementation is used. Then I ensure the pipeline actually trains before inference (your current run does zero-shot QA, which explains the very low score), keeping the same model, loss, and training loop—just wiring in feature building, a train DataLoader, and calling `train_one_epoch`. Finally, I keep your existing post-processing and submission writing unchanged so the output format remains valid.'
- What this solution (achieved 0.07143) has done: 'I fix the `MessageFactory.GetPrototype` crash by forcing a protobuf version that is compatible with `transformers` in this Kaggle image (the error is coming from an incompatible `google-protobuf` API at import time). Then I make feature caching robust to model names containing slashes (so caching never breaks) and ensure the script always completes and writes `submission.csv`. These changes are execution/stability fixes and do not alter your model architecture, loss, tokenization settings, training loop semantics, or post-processing logic, so they should improve score mainly by allowing the training+inference pipeline to actually run end-to-end.'
- What this solution (achieved 0.06384) has done: 'Your current score (0.07143) is far below the target (0.7053), so we need a real-but-minimal quality improvement without changing the model architecture, loss, or training loop. The biggest issue is that training on all overflowed chunks equally injects many “no-answer-in-this-window” negatives, so the model learns to predict CLS/empty; we filter training features to keep only windows that actually contain the answer span (plus a small, fixed fraction of CLS windows for stability). This keeps the same tokenization, same labels, same model, and same training code, but improves the signal-to-noise ratio in training and should move the score sharply upward toward your target. I also ensure we don’t accidentally break test feature alignment and still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver
    except Exception:
        pb_ver = None

    need_install = False
    if pb_ver is None:
        need_install = True
    else:
        major = int(str(pb_ver).split(".")[0])
        if major >= 5:
            need_install = True

    if need_install:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )


_ensure_protobuf_compat()

import gc

gc.enable()
import math
import random
import multiprocessing
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm
from string import punctuation

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler

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
    logging,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
)

try:
    from transformers import AdamW  # older versions
except Exception:
    from torch.optim import AdamW  # compatible replacement

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def optimal_num_of_loader_workers():
    num_cpus = multiprocessing.cpu_count()
    return max(1, min(4, num_cpus - 1))


print(f"Apex AMP Installed :: {APEX_INSTALLED}")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)




## === cell 1
class Config:
    model_type = "xlm-roberta"

    model_name_or_path = "xlm-roberta-base"
    config_name = "xlm-roberta-base"
    tokenizer_name = "xlm-roberta-base"

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

    keep_no_answer_frac = 0.10  # fixed fraction of CLS windows to retain


fix_all_seeds(Config.seed)




## === cell 2
class DatasetRetriever(Dataset):
    def __init__(self, features, mode="train"):
        super(DatasetRetriever, self).__init__()
        self.features = features
        self.mode = mode

        if self.mode == "train":
            self.input_ids = torch.as_tensor(
                np.asarray([f["input_ids"] for f in features], dtype=np.int64),
                dtype=torch.long,
            )
            self.attention_mask = torch.as_tensor(
                np.asarray([f["attention_mask"] for f in features], dtype=np.int64),
                dtype=torch.long,
            )
            self.offset_mapping = torch.as_tensor(
                np.asarray([f["offset_mapping"] for f in features], dtype=np.int64),
                dtype=torch.long,
            )
            self.start_position = torch.as_tensor(
                np.asarray([f["start_position"] for f in features], dtype=np.int64),
                dtype=torch.long,
            )
            self.end_position = torch.as_tensor(
                np.asarray([f["end_position"] for f in features], dtype=np.int64),
                dtype=torch.long,
            )
        else:
            self.input_ids = torch.as_tensor(
                np.asarray([f["input_ids"] for f in features], dtype=np.int64),
                dtype=torch.long,
            )
            self.attention_mask = torch.as_tensor(
                np.asarray([f["attention_mask"] for f in features], dtype=np.int64),
                dtype=torch.long,
            )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        if self.mode == "train":
            return {
                "input_ids": self.input_ids[item],
                "attention_mask": self.attention_mask[item],
                "offset_mapping": self.offset_mapping[item],
                "start_position": self.start_position[item],
                "end_position": self.end_position[item],
            }
        else:
            return {
                "input_ids": self.input_ids[item],
                "attention_mask": self.attention_mask[item],
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
def make_model(args):
    try:
        config = AutoConfig.from_pretrained(args.config_name, local_files_only=True)
    except Exception:
        config = AutoConfig.from_pretrained(args.config_name, local_files_only=False)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, local_files_only=True, use_fast=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, local_files_only=False, use_fast=True
        )

    model = Model(args.model_name_or_path, config=config)
    return config, tokenizer, model




## === cell 5
def _batched_tokenize_qa(args, questions, contexts, tokenizer):
    return tokenizer(
        list(questions),
        list(contexts),
        truncation="only_second",
        max_length=args.max_seq_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )


def _feature_cache_paths(args, split_name: str):
    os.makedirs(args.output_dir, exist_ok=True)
    key = f"{split_name}_{args.model_name_or_path}_len{args.max_seq_length}_stride{args.doc_stride}"
    safe_key = "".join([c if c.isalnum() or c in "._-" else "_" for c in key])
    return os.path.join(args.output_dir, f"{safe_key}.npz")


def build_test_features_batched(args, df, tokenizer, chunk_size=4096, use_cache=True):
    cache_path = _feature_cache_paths(args, "test_features")
    if use_cache and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return data["features"].tolist()

    questions_all = df["question"].astype(str).str.lstrip().to_numpy()
    contexts_all = df["context"].astype(str).to_numpy()
    ids_all = df["id"].to_numpy()

    features = []
    n = len(df)
    for start in tqdm(range(0, n, chunk_size), desc="Tokenize test", leave=False):
        end = min(n, start + chunk_size)
        questions = questions_all[start:end]
        contexts = contexts_all[start:end]
        ids = ids_all[start:end]

        enc = _batched_tokenize_qa(args, questions, contexts, tokenizer)
        sample_mapping = enc["overflow_to_sample_mapping"]

        for i in range(len(enc["input_ids"])):
            sample_idx = int(sample_mapping[i])
            features.append(
                {
                    "example_id": ids[sample_idx],
                    "context": contexts[sample_idx],
                    "question": questions[sample_idx],
                    "input_ids": enc["input_ids"][i],
                    "attention_mask": enc["attention_mask"][i],
                    "offset_mapping": enc["offset_mapping"][i],
                    "sequence_ids": [
                        0 if j is None else j for j in enc.sequence_ids(i)
                    ],
                }
            )

    if use_cache:
        np.savez_compressed(cache_path, features=np.array(features, dtype=object))
    return features


def build_train_features_batched(args, df, tokenizer, chunk_size=4096, use_cache=True):
    cache_path = _feature_cache_paths(args, "train_features")
    if use_cache and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return data["features"].tolist()

    questions_all = df["question"].astype(str).str.lstrip().to_numpy()
    contexts_all = df["context"].astype(str).to_numpy()
    answer_texts_all = df["answer_text"].fillna("").astype(str).to_numpy()
    answer_starts_all = df["answer_start"].fillna(0).astype(int).to_numpy()

    cls_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0

    features = []
    n = len(df)
    for start in tqdm(range(0, n, chunk_size), desc="Tokenize train", leave=False):
        end = min(n, start + chunk_size)

        questions = questions_all[start:end]
        contexts = contexts_all[start:end]
        answer_texts = answer_texts_all[start:end]
        answer_starts = answer_starts_all[start:end]

        enc = _batched_tokenize_qa(args, questions, contexts, tokenizer)
        sample_mapping = enc["overflow_to_sample_mapping"]

        for i in range(len(enc["input_ids"])):
            sample_idx = int(sample_mapping[i])
            input_ids = enc["input_ids"][i]
            attention_mask = enc["attention_mask"][i]
            offset_mapping = enc["offset_mapping"][i]
            sequence_ids = [0 if j is None else j for j in enc.sequence_ids(i)]

            answer_text = answer_texts[sample_idx]
            answer_start = int(answer_starts[sample_idx])
            answer_end = answer_start + len(answer_text)

            context_index = 1
            token_start_index = 0
            while (
                token_start_index < len(sequence_ids)
                and sequence_ids[token_start_index] != context_index
            ):
                token_start_index += 1

            token_end_index = len(sequence_ids) - 1
            while (
                token_end_index >= 0 and sequence_ids[token_end_index] != context_index
            ):
                token_end_index -= 1

            if cls_token_id in input_ids:
                cls_index = input_ids.index(cls_token_id)
            else:
                cls_index = 0

            start_position = cls_index
            end_position = cls_index

            if token_start_index <= token_end_index:
                start_char = offset_mapping[token_start_index][0]
                end_char = offset_mapping[token_end_index][1]
                if not (answer_start >= start_char and answer_end <= end_char):
                    start_position = cls_index
                    end_position = cls_index
                else:
                    while (
                        token_start_index < len(offset_mapping)
                        and offset_mapping[token_start_index][0] <= answer_start
                    ):
                        token_start_index += 1
                    start_position = token_start_index - 1

                    while (
                        token_end_index >= 0
                        and offset_mapping[token_end_index][1] >= answer_end
                    ):
                        token_end_index -= 1
                    end_position = token_end_index + 1

                    start_position = int(np.clip(start_position, 0, len(input_ids) - 1))
                    end_position = int(np.clip(end_position, 0, len(input_ids) - 1))
                    if end_position < start_position:
                        start_position = cls_index
                        end_position = cls_index

            features.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "offset_mapping": offset_mapping,
                    "start_position": start_position,
                    "end_position": end_position,
                }
            )

    if use_cache:
        np.savez_compressed(cache_path, features=np.array(features, dtype=object))
    return features




## === cell 6
import collections


def postprocess_qa_predictions(
    examples, features, raw_predictions, tokenizer, n_best_size=20, max_answer_length=30
):
    all_start_logits, all_end_logits = raw_predictions
    n_features = len(features)

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    print(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )

    cls_token_id = tokenizer.cls_token_id

    feat_cls_index = np.zeros(n_features, dtype=np.int32)
    feat_ctx_offsets = [None] * n_features
    for fi, f in enumerate(features):
        inp = f["input_ids"]
        if cls_token_id is not None:
            try:
                feat_cls_index[fi] = inp.index(cls_token_id)
            except ValueError:
                feat_cls_index[fi] = 0
        else:
            feat_cls_index[fi] = 0

        si = f["sequence_ids"]
        off = f["offset_mapping"]
        feat_ctx_offsets[fi] = [o if (sid == 1) else None for sid, o in zip(si, off)]

    k = n_best_size
    feat_topk_start = [None] * n_features
    feat_topk_end = [None] * n_features
    for fs in range(n_features):
        s = np.asarray(all_start_logits[fs])
        e = np.asarray(all_end_logits[fs])

        kk = k if k < s.shape[0] else s.shape[0]
        if kk == s.shape[0]:
            s_idx = np.argsort(s)[::-1]
            e_idx = np.argsort(e)[::-1]
        else:
            s_part = np.argpartition(s, -kk)[-kk:]
            e_part = np.argpartition(e, -kk)[-kk:]
            s_idx = s_part[np.argsort(s[s_part])[::-1]]
            e_idx = e_part[np.argsort(e[e_part])[::-1]]

        feat_topk_start[fs] = s_idx.tolist()
        feat_topk_end[fs] = e_idx.tolist()

    predictions = collections.OrderedDict()
    for example in examples.itertuples(index=False):
        ex_id = example.id
        context = example.context
        example_index = example_id_to_index[ex_id]
        feature_indices = features_per_example[example_index]

        best_text = ""
        best_score = -1e30

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = feat_ctx_offsets[feature_index]

            start_indexes = feat_topk_start[feature_index]
            end_indexes = feat_topk_end[feature_index]

            for start_index in start_indexes:
                if (
                    start_index >= len(offset_mapping)
                    or offset_mapping[start_index] is None
                ):
                    continue
                for end_index in end_indexes:
                    if (
                        end_index >= len(offset_mapping)
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
                    score = float(start_logits[start_index] + end_logits[end_index])

                    if score > best_score:
                        best_score = score
                        best_text = context[start_char:end_char]

        predictions[ex_id] = best_text if best_score > -1e29 else ""

    return predictions




## === cell 7
DATA_DIR = "/kaggle/input/chaii-hindi-and-tamil-question-answering"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")

train["context"] = train["context"].astype(str).str.split().str.join(" ")
train["question"] = train["question"].astype(str).str.split().str.join(" ")
train["answer_text"] = train["answer_text"].fillna("").astype(str)
train["answer_start"] = train["answer_start"].fillna(0).astype(int)

test["context"] = test["context"].astype(str).str.split().str.join(" ")
test["question"] = test["question"].astype(str).str.split().str.join(" ")

args = Config()
config, tokenizer, model = make_model(args)

train_features = build_train_features_batched(
    args, train, tokenizer, chunk_size=4096, use_cache=True
)

cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0
pos_features = []
neg_features = []
for f in train_features:
    cls_index = f["input_ids"].index(cls_id) if cls_id in f["input_ids"] else 0
    if int(f["start_position"]) == cls_index and int(f["end_position"]) == cls_index:
        neg_features.append(f)
    else:
        pos_features.append(f)

rng = np.random.RandomState(args.seed)
keep_neg = int(len(neg_features) * float(getattr(args, "keep_no_answer_frac", 0.10)))
if keep_neg > 0:
    neg_idx = rng.choice(len(neg_features), size=keep_neg, replace=False)
    neg_keep = [neg_features[i] for i in neg_idx]
else:
    neg_keep = []

train_features = pos_features + neg_keep
rng.shuffle(train_features)

test_features = build_test_features_batched(
    args, test, tokenizer, chunk_size=4096, use_cache=True
)

train_dataset = DatasetRetriever(train_features, mode="train")
test_dataset = DatasetRetriever(test_features, mode="test")
_nw = optimal_num_of_loader_workers()

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.train_batch_size,
    sampler=RandomSampler(train_dataset),
    num_workers=_nw,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_nw > 0),
    prefetch_factor=4 if _nw > 0 else None,
    drop_last=False,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.eval_batch_size,
    sampler=SequentialSampler(test_dataset),
    num_workers=_nw,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_nw > 0),
    prefetch_factor=4 if _nw > 0 else None,
    drop_last=False,
)

print("Num train examples:", len(train))
print("Num train features (raw):", len(pos_features) + len(neg_features))
print("Num train features (used):", len(train_features))
print(
    "Pos features:",
    len(pos_features),
    "Neg features kept:",
    len(neg_keep),
    "Neg raw:",
    len(neg_features),
)
print("Num test examples:", len(test))
print("Num test features:", len(test_features))




## === cell 8
def train_one_epoch(args, model, train_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.epsilon,
    )

    total_steps = (
        math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        * args.epochs
    )
    warmup_steps = int(args.warmup_ratio * total_steps)

    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    ce_loss = nn.CrossEntropyLoss()

    if args.fp16 and APEX_INSTALLED:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_dataloader, desc="Train", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        start_positions = batch["start_position"].to(device, non_blocking=True)
        end_positions = batch["end_position"].to(device, non_blocking=True)

        start_logits, end_logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        loss_start = ce_loss(start_logits, start_positions)
        loss_end = ce_loss(end_logits, end_positions)
        loss = (loss_start + loss_end) / 2.0
        loss = loss / args.gradient_accumulation_steps

        if args.fp16 and APEX_INSTALLED:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if not (args.fp16 and APEX_INSTALLED):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.logging_steps == 0:
                pbar.set_postfix(
                    {
                        "loss": float(
                            loss.detach().cpu().item()
                            * args.gradient_accumulation_steps
                        )
                    }
                )

    return model


def get_predictions_from_model(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model.to(device)
    model.eval()

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune", fullgraph=False)
        except Exception:
            pass

    n = len(test_dataloader.dataset)
    seq_len = test_dataloader.dataset.input_ids.shape[1]
    start_logits = np.empty((n, seq_len), dtype=np.float32)
    end_logits = np.empty((n, seq_len), dtype=np.float32)

    ofs = 0
    for batch in tqdm(test_dataloader, desc="Inference", leave=False):
        bs = batch["input_ids"].shape[0]
        with torch.inference_mode():
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            outputs_start, outputs_end = model(input_ids, attention_mask)
            start_logits[ofs : ofs + bs] = outputs_start.detach().cpu().numpy()
            end_logits[ofs : ofs + bs] = outputs_end.detach().cpu().numpy()
        ofs += bs

    return start_logits, end_logits


model = train_one_epoch(args, model, train_dataloader)

start_logits1, end_logits1 = get_predictions_from_model(model, test_dataloader)

fin_preds = postprocess_qa_predictions(
    test, test_features, (start_logits1, end_logits1), tokenizer=tokenizer
)

submission_rows = []
for _id, pred in fin_preds.items():
    pred = " ".join(str(pred).split())
    pred = pred.strip(punctuation)
    submission_rows.append((_id, pred))

sample = pd.DataFrame(submission_rows, columns=["id", "PredictionString"])

test_data = test[["id", "context"]].merge(sample, on="id", how="left")
test_data["PredictionString"] = test_data["PredictionString"].fillna("")

bad_starts = [".", ",", "(", ")", "-", "–", ",", ";"]
bad_endings = ["...", "-", "(", ")", "–", ",", ";"]

tamil_ad = "கி.பி"
tamil_bc = "கி.மு"
tamil_km = "கி.மீ"
hindi_ad = "ई"
hindi_bc = "ई.पू"

cleaned_preds = []
for pred, context in test_data[["PredictionString", "context"]].to_numpy():
    pred = str(pred)
    context = str(context)
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

sub_path = "submission.csv"
test_data[["id", "PredictionString"]].to_csv(sub_path, index=False)
print("Wrote submission.csv with shape:", test_data[["id", "PredictionString"]].shape)
print(test_data[["id", "PredictionString"]].head())
