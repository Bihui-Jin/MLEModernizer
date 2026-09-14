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

0.1928532570600509

# 6. Current score

0.22755

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.02084) has done: 'The timeout is dominated by running 10 full forward passes of a large Transformer over all test features (5 Hindi + 5 Tamil folds), plus heavy Python work in post-processing. I keep the exact ensemble logic and model forward semantics, but cut overhead by (1) pre-stacking all feature tensors once and using a faster TensorDataset DataLoader, (2) compiling the model (PyTorch 2.x) for faster inference, (3) using pinned memory and a prefetching DataLoader setup, and (4) optimizing post-processing by iterating over numpy arrays / pre-grouped indices instead of Pandas row iteration and repeated Python list work. These changes are computationally equivalent (same logits, same argmax selection, same text slicing), only reducing Python overhead and improving GPU throughput.'
- What this solution (achieved 0.02084) has done: 'I fix the immediate runtime crash coming from a protobuf/transformers incompatibility triggered during tokenizer/model loading, by enforcing the pure-Python protobuf implementation early (before importing transformers). I also make the pretrained-path fallback robust in Kaggle’s offline environment by preferring locally available competition/input model folders and only using HF IDs if they can be resolved (keeping model architecture and inference logic unchanged). Finally, I keep the existing ensemble/post-processing untouched, ensuring the pipeline completes and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.22755) has done: 'We fix the runtime crash caused by an incompatibility between `protobuf` and `transformers` in this environment by forcing the compatible protobuf API implementation *before* importing `transformers`, and by pinning the Python protobuf version behavior more robustly. Then we make model/tokenizer loading fully offline-safe by preferring locally available model folders and falling back to a known Kaggle-available XLM-R QA model if the MPNet path is missing, since the current setup likely loads a non-QA backbone and hurts score severely. Finally, we keep the ensemble/inference/postprocess logic the same, but ensure the correct QA model is actually used so the score moves toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.22755) has done: 'I fix the runtime crash in tokenizer/model loading caused by an incompatibility between `transformers` and the installed `protobuf` by forcing the pure-Python protobuf implementation *and* disabling the C++ descriptors implementation before importing `transformers`. This is a correctness/stability fix and should not change the model/inference logic or intended score behavior. I also make the test CSV path offline-robust (preferring `/kaggle/input/...` but falling back to the provided `../input/...` if present) so the notebook runs in your described filesystem. No changes are made to the model architecture, ensembling, or post-processing, so the score should remain close to the current 0.22755 (already above target).'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PROTOCOL_BUFFERS_DISABLE_CPP_IMPLEMENTATION", "1")
os.environ.setdefault("PROTOBUF_DISABLE_UPB", "1")

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
from torch.utils.data import (
    Dataset,
    DataLoader,
    SequentialSampler,
    RandomSampler,
    TensorDataset,
)

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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def optimal_num_of_loader_workers():
    try:
        num_cpus = multiprocessing.cpu_count()
    except Exception:
        num_cpus = 2
    if torch.cuda.is_available():
        return min(4, max(1, num_cpus // 2))
    return min(2, max(0, num_cpus - 1))


print(f"Apex AMP Installed :: {APEX_INSTALLED}")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def resolve_pretrained_path(preferred_path: str, fallback_repo_id: str) -> str:
    """
    Prefer existing local paths (Kaggle offline), otherwise fall back to HF id.
    """
    if isinstance(preferred_path, str) and os.path.exists(preferred_path):
        return preferred_path
    candidates = [
        preferred_path,
        str(preferred_path).replace("../input/", "/kaggle/input/"),
        str(preferred_path).replace("../input/", "/kaggle/data/"),
    ]
    for c in candidates:
        if isinstance(c, str) and os.path.exists(c):
            return c
    return fallback_repo_id


def resolve_competition_file(rel_path_under_comp: str) -> str:
    """
    BUGFIX: Robustly resolve dataset files across common Kaggle mount points.
    """
    candidates = [
        f"../input/chaii-hindi-and-tamil-question-answering/{rel_path_under_comp}",
        f"/kaggle/input/chaii-hindi-and-tamil-question-answering/{rel_path_under_comp}",
        f"/kaggle/data/chaii-hindi-and-tamil-question-answering/{rel_path_under_comp}",
        f"../kaggle/input/chaii-hindi-and-tamil-question-answering/{rel_path_under_comp}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]




## === cell 1
class Configration:
    model_type = "xlm_roberta"
    XLMR_name_or_path = "../input/nothing-of-your-concern/xlm-roberta-large-squad2/xlm-roberta-large-squad2"
    MURIL_name_or_path = (
        "../input/nothing-of-your-concern/muril-large-cased/muril-large-cased"
    )
    MPNET2_name_or_path = "../input/nothing-of-your-concern/multi-qa-mpnet-base-cos-v1/multi-qa-mpnet-base-cos-v1"
    MPNET3_name_or_path = "../input/nothing-of-your-concern/multi-qa-mpnet-base-dot-v1/multi-qa-mpnet-base-dot-v1"
    XLMR_config_name = "../input/nothing-of-your-concern/xlm-roberta-large-squad2/xlm-roberta-large-squad2/config.json"
    MURIL_config_name = "../input/nothing-of-your-concern/muril-large-cased/muril-large-cased/config.json"
    MPNET2_config_name = "../input/nothing-of-your-concern/multi-qa-mpnet-base-cos-v1/multi-qa-mpnet-base-cos-v1/config.json"
    MPNET3_config_name = "../input/nothing-of-your-concern/multi-qa-mpnet-base-dot-v1/multi-qa-mpnet-base-dot-v1/config.json"
    fp16 = True if APEX_INSTALLED else False
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
fix_all_seeds(Configration().seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

try:
    torch.set_num_threads(max(1, multiprocessing.cpu_count() // 2))
except Exception:
    pass

try:
    torch.backends.cudnn.deterministic = False
except Exception:
    pass




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

    def forward(self, input_ids, attention_mask=None):
        outputs = self.xlm_roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits




## === cell 5
_ARGS = Configration()

_FALLBACK_MODEL_ID = "deepset/xlm-roberta-large-squad2"

_CFG_SRC = resolve_pretrained_path(_ARGS.XLMR_config_name, _FALLBACK_MODEL_ID)
_TOK_SRC = resolve_pretrained_path(_ARGS.XLMR_tokenizer_name, _FALLBACK_MODEL_ID)
_MDL_SRC = resolve_pretrained_path(_ARGS.XLMR_name_or_path, _FALLBACK_MODEL_ID)

_CONFIG_CACHE = None
_TOKENIZER_CACHE = None


def Make_Model(args):
    global _CONFIG_CACHE, _TOKENIZER_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = AutoConfig.from_pretrained(_CFG_SRC)
    if _TOKENIZER_CACHE is None:
        _TOKENIZER_CACHE = AutoTokenizer.from_pretrained(_TOK_SRC, use_fast=True)
    config = _CONFIG_CACHE
    tokenizer = _TOKENIZER_CACHE
    model = Model(_MDL_SRC, config=config)
    return config, tokenizer, model




## === cell 6
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
            0 if s is None else s for s in tokenized_example.sequence_ids(i)
        ]
        features.append(feature)
    return features




## === cell 7
import collections


def Postprocess_qa_predictions(
    examples, features, raw_predictions, tokenizer, n_best_size=20, max_answer_length=30
):
    all_start_logits, all_end_logits = raw_predictions

    example_ids = examples["id"].to_numpy()
    contexts = examples["context"].to_numpy()

    example_id_to_index = {k: i for i, k in enumerate(example_ids)}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    context_index = 1
    cls_id = tokenizer.cls_token_id

    feat_offset_ctx = [None] * len(features)
    feat_cls_index = np.zeros(len(features), dtype=np.int32)
    for fi, feat in enumerate(features):
        seq_ids = feat["sequence_ids"]
        om = feat["offset_mapping"]
        feat_offset_ctx[fi] = [
            (o if seq_ids[k] == context_index else None) for k, o in enumerate(om)
        ]
        try:
            feat_cls_index[fi] = feat["input_ids"].index(cls_id)
        except Exception:
            feat_cls_index[fi] = 0

    predictions = collections.OrderedDict()
    print(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )

    k = int(n_best_size)
    for example_index in range(len(example_ids)):
        feature_indices = features_per_example[example_index]
        valid_answers = []
        context = contexts[example_index]

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = feat_offset_ctx[feature_index]

            if k >= start_logits.shape[0]:
                start_top = np.arange(start_logits.shape[0], dtype=np.int32)
            else:
                start_top = np.argpartition(start_logits, -k)[-k:]
            if k >= end_logits.shape[0]:
                end_top = np.arange(end_logits.shape[0], dtype=np.int32)
            else:
                end_top = np.argpartition(end_logits, -k)[-k:]

            start_top = start_top[np.argsort(start_logits[start_top])[::-1]]
            end_top = end_top[np.argsort(end_logits[end_top])[::-1]]

            for start_index in start_top:
                start_index = int(start_index)
                om_s = offset_mapping[start_index]
                if om_s is None:
                    continue
                for end_index in end_top:
                    end_index = int(end_index)
                    if end_index < start_index:
                        continue
                    if end_index - start_index + 1 > max_answer_length:
                        continue
                    om_e = offset_mapping[end_index]
                    if om_e is None:
                        continue
                    start_char = om_s[0]
                    end_char = om_e[1]
                    valid_answers.append(
                        {
                            "score": float(
                                start_logits[start_index] + end_logits[end_index]
                            ),
                            "text": context[start_char:end_char],
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = max(valid_answers, key=lambda x: x["score"])["text"]
        else:
            best_answer = ""

        predictions[example_ids[example_index]] = best_answer

    return predictions




## === cell 8
test_path = resolve_competition_file("test.csv")
test_df = pd.read_csv(test_path)
test_df["context"] = test_df["context"].astype(str).str.split().str.join(" ")
test_df["question"] = test_df["question"].astype(str).str.split().str.join(" ")

args = Configration()
_, tokenizer, _ = Make_Model(args)

questions = test_df["question"].astype(str).str.lstrip().tolist()
contexts = test_df["context"].astype(str).tolist()
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
_ids = test_df["id"].to_numpy()
_langs = test_df["language"].to_numpy()
_ctxs = test_df["context"].to_numpy()
_qs = test_df["question"].to_numpy()

for i in range(len(tokenized["input_ids"])):
    sample_idx = int(overflow_to_sample[i])
    test_features.append(
        {
            "language": _langs[sample_idx],
            "example_id": _ids[sample_idx],
            "context": _ctxs[sample_idx],
            "question": _qs[sample_idx],
            "input_ids": tokenized["input_ids"][i],
            "attention_mask": tokenized["attention_mask"][i],
            "offset_mapping": tokenized["offset_mapping"][i],
            "sequence_ids": [0 if s is None else s for s in tokenized.sequence_ids(i)],
        }
    )

input_ids_arr = np.asarray(tokenized["input_ids"], dtype=np.int64)
attn_arr = np.asarray(tokenized["attention_mask"], dtype=np.int64)
input_ids_t = torch.from_numpy(input_ids_arr)
attn_t = torch.from_numpy(attn_arr)
test_tensor_dataset = TensorDataset(input_ids_t, attn_t)

_nw = optimal_num_of_loader_workers()
test_dataloader = DataLoader(
    test_tensor_dataset,
    batch_size=args.eval_batch_size,
    sampler=SequentialSampler(test_tensor_dataset),
    num_workers=_nw,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    persistent_workers=True if _nw > 0 else False,
    prefetch_factor=2 if _nw > 0 else None,
)

print("num test examples:", len(test_df), "num features:", len(test_features))



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
base_model_hindi = "../input/nothing-of-your-concern/MPNet3-H/output/"
base_model_tamil = "../input/nothing-of-your-concern/MPNet3-T/output/"


def _safe_load_state_dict(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu")
    try:
        model.load_state_dict(state, strict=True)
        return True
    except RuntimeError:
        new_state = {}
        for k, v in state.items():
            nk = k.replace("module.", "") if k.startswith("module.") else k
            new_state[nk] = v
        model.load_state_dict(new_state, strict=False)
        return True


def _accumulate_logits_into_sums(model, dataloader, start_sum, end_sum):
    model.eval()
    idx = 0
    with torch.inference_mode():
        for input_ids, attention_mask in dataloader:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            os_, oe_ = model(input_ids, attention_mask)
            bs = os_.shape[0]
            start_sum[idx : idx + bs] += os_.cpu().numpy()
            end_sum[idx : idx + bs] += oe_.cpu().numpy()
            idx += bs


languages = [feat["language"] for feat in test_features]
seq_len = args.max_seq_length
n_feat = len(test_tensor_dataset)

start_sum_h = np.zeros((n_feat, seq_len), dtype=np.float32)
end_sum_h = np.zeros((n_feat, seq_len), dtype=np.float32)
start_sum_t = np.zeros((n_feat, seq_len), dtype=np.float32)
end_sum_t = np.zeros((n_feat, seq_len), dtype=np.float32)

config, tok_tmp, model = Make_Model(Configration())
del tok_tmp  # tokenizer not needed here
model.to(device)

if hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        print("torch.compile enabled")
    except Exception as e:
        print("torch.compile unavailable, continuing without it:", repr(e))

for fold in range(5):
    h_ckpt = base_model_hindi + f"checkpoint-fold-{fold}/pytorch_model.bin"
    if os.path.exists(h_ckpt):
        _safe_load_state_dict(model, h_ckpt)
    else:
        print(
            f"WARNING: checkpoint not found: {h_ckpt}. Using base pretrained weights."
        )
    _accumulate_logits_into_sums(model, test_dataloader, start_sum_h, end_sum_h)

for fold in range(5):
    t_ckpt = base_model_tamil + f"checkpoint-fold-{fold}/pytorch_model.bin"
    if os.path.exists(t_ckpt):
        _safe_load_state_dict(model, t_ckpt)
    else:
        print(
            f"WARNING: checkpoint not found: {t_ckpt}. Using base pretrained weights."
        )
    _accumulate_logits_into_sums(model, test_dataloader, start_sum_t, end_sum_t)

del model, config
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

start_logits_h = start_sum_h / 5.0
end_logits_h = end_sum_h / 5.0
start_logits_t = start_sum_t / 5.0
end_logits_t = end_sum_t / 5.0

langs = np.asarray(languages, dtype=object)
is_hindi = np.char.lower(langs.astype(str)) == "hindi"
start = np.empty_like(start_logits_h)
end = np.empty_like(end_logits_h)
start[is_hindi] = start_logits_h[is_hindi]
end[is_hindi] = end_logits_h[is_hindi]
start[~is_hindi] = start_logits_t[~is_hindi]
end[~is_hindi] = end_logits_t[~is_hindi]

fin_preds = Postprocess_qa_predictions(
    test_df, test_features, (start, end), tokenizer=tokenizer
)

submission = []
for ex_id, pred in fin_preds.items():
    pred = " ".join(str(pred).split())
    pred = pred.strip(punctuation)
    submission.append((ex_id, pred))

sample = pd.DataFrame(submission, columns=["id", "PredictionString"])
test_data = pd.merge(left=test_df, right=sample, on="id", how="left")



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
    pred = "" if pd.isna(pred) else str(pred)
    context = "" if pd.isna(context) else str(context)

    if pred == "":
        cleaned_preds.append(pred)
        continue

    while any(pred.startswith(y) for y in bad_starts) and len(pred) > 0:
        pred = pred[1:]
    while any(pred.endswith(y) for y in bad_endings) and len(pred) > 0:
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
test_data[["id", "PredictionString"]].to_csv(out_path, index=False, encoding="utf-8")
print("Wrote:", out_path, "rows:", len(test_data))



## === cell 11
print(test_data[["id", "PredictionString"]].head())
print("submission shape:", pd.read_csv("submission.csv").shape)
