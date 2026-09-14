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

0.1951206475496292

# 6. Current score

0.26903

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.26903) has done: 'I remove the biggest Python overheads while keeping the exact same model, checkpoints, feature logic, and postprocessing semantics. The main speedups come from batching tokenization for the whole test set (instead of per-row calls), eliminating repeated CPU→NumPy copies by preallocating and copying efficiently, and making the DataLoader cheaper (fewer worker overheads + faster host→GPU transfer). I also avoid recomputing per-feature offset filtering inside the postprocess loop by caching it once (equivalent computation, just moved out of inner loops). All changes preserve the same inputs/outputs, folds, logits averaging, and final text selection logic.'
- What this solution (achieved 0.26903) has done: 'The crash in cell 8 comes from a known `protobuf`/`transformers` incompatibility in some Kaggle images (it surfaces as `MessageFactory.GetPrototype` missing during fast-tokenizer initialization). I fix this by forcing the pure-Python protobuf implementation via an environment variable set *before* importing `transformers`, which avoids that code path and keeps your exact tokenization/model logic unchanged. I also add a small safety fallback to load checkpoints with `weights_only=False` for Torch 2.6+ environments where the default can break some older checkpoint formats, without changing any scoring logic. No model/postprocess/calibration changes are introduced, so score should remain essentially the same while the notebook runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.26903) has done: 'The crash in cell 8 happens during `transformers` fast-tokenizer initialization due to a protobuf API mismatch (`MessageFactory.GetPrototype`), and setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` alone is not sufficient unless we also force the pure-Python implementation version and do it before importing `transformers`. I make that environment fix stricter and earlier, and add a safe fallback to `use_fast=False` if the fast tokenizer still fails, which preserves the same model and overall pipeline while restoring end-to-end execution. I also ensure the tokenizer used for feature creation and for postprocessing are the same object (score-neutral, avoids subtle mismatches). No model architecture/training/postprocessing logic is changed, so the score should remain essentially the same (already above the target band), but the notebook reliably produce a valid `submission.csv`.'
- What this solution (achieved 0.26903) has done: 'We fix the protobuf/transformers crash by forcing the pure-Python protobuf implementation *before any transformers/tokenizers import* and additionally setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` via direct assignment (not `setdefault`) so it can’t be ignored if already set in the environment. To keep scoring semantics stable (your current score is already above the target band), we won’t change the model, folds, logits averaging, or postprocessing logic; we only make the tokenizer initialization robust by retrying with `use_fast=False` if the fast tokenizer still fails. We also ensure the same tokenizer object is used consistently in feature creation and postprocessing (score-neutral, avoids subtle mismatches). Finally, we keep the output as a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.26903) has done: 'I fix the crash in cell 8 caused by the protobuf `MessageFactory.GetPrototype` incompatibility by forcing the pure-Python protobuf runtime *before* any `transformers/tokenizers` import, and by adding a robust tokenizer construction helper that falls back to `use_fast=False` if needed. This is purely a stability/runtime fix and does not intentionally change your model, folds, logits averaging, or post-processing, so the score should remain essentially the same (still above the target band). I also remove the redundant second tokenizer creation in cell 9 to ensure the exact same tokenizer object is used for feature creation and postprocess (score-neutral but avoids subtle mismatches). Finally, I keep the same output path and ensure `submission.csv` is always written with the required columns.'
- What this solution (achieved 0.26903) has done: 'I fix the runtime crash in cell 8 caused by a protobuf/fast-tokenizer incompatibility by forcing the pure-Python protobuf implementation before any `transformers/tokenizers` import and adding a safe fallback to `use_fast=False` if fast tokenizer initialization still fails. This is a stability fix and does not change your model architecture, checkpoints, folding/averaging, or post-processing, so it should keep the score essentially the same (your current score is already above the target band, so we avoid any score-changing tweaks). I also make sure the environment variables are set unconditionally (not ignored if preset) and keep the submission writing unchanged so a valid `submission.csv` is always produced.'
- What this solution (achieved 0.26903) has done: 'We fix the crash in cell 8 by ensuring the protobuf runtime is forced to the pure-Python implementation *before* any `transformers/tokenizers` import occurs, which is what triggers the `MessageFactory.GetPrototype` error in some Kaggle images. To keep your achieved score behavior stable (already above the target band), we won’t change any model checkpoints, fold ensembling, feature creation parameters, or postprocessing logic—only make tokenizer initialization robust by falling back to `use_fast=False` if the fast tokenizer still fails. We also keep the same submission writing path/name and verify the output columns match `sample_submission.csv`. These changes are runtime/stability fixes and should not materially change your score.'
- What this solution (achieved 0.26903) has done: 'I fix the `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before* any `transformers/tokenizers` import, which is when the fast tokenizer triggers the incompatible protobuf API. To keep your score behavior stable (you’re already above the target band), I won’t change any model checkpoints, folds, averaging, feature parameters, or postprocessing; only tokenizer initialization be made robust by retrying with `use_fast=False` if needed. I also make the environment-variable enforcement unconditional (not overridable by a pre-set value) and keep the submission writing exactly as `submission.csv` with `id,PredictionString`.'
- What this solution (achieved 0.26903) has done: 'I fix the runtime crash in cell 8 caused by a protobuf/fast-tokenizer incompatibility by enforcing the pure-Python protobuf runtime *before* importing anything from `transformers/tokenizers`, and by making tokenizer construction robust with a fallback to `use_fast=False` if the fast tokenizer still fails. This is a stability fix that preserves your exact model, checkpoints, feature parameters, fold-averaging, and postprocessing semantics, so the score should remain essentially unchanged (and since your current score is already above the target band, we avoid any score-increasing tweaks). I also adjust the cell numbering to start at 1 (your current script starts at cell 0), without changing execution order. Finally, the script still write a valid `submission.csv` with the required `id,PredictionString` columns.'
- What this solution (achieved 0.26903) has done: 'We fix the runtime crash in cell 8 caused by a protobuf/fast-tokenizer incompatibility by enforcing the pure-Python protobuf implementation before importing `transformers`, and by constructing the tokenizer via a small helper that safely falls back to `use_fast=False` if the fast tokenizer still triggers the protobuf error. These are stability-only changes and won’t alter your model checkpoints, fold averaging, feature parameters, or postprocessing logic—so the score should remain essentially the same (and since your current score is already above the target band, we avoid any score-changing tweaks). We also adjust the cell numbering to start at 1 (your script currently starts at cell 0) while preserving execution order. The pipeline then run end-to-end and reliably write a valid `submission.csv` with `id,PredictionString`.'
- What this solution (achieved 0.26903) has done: 'We fix the protobuf/fast-tokenizer crash by enforcing the pure-Python protobuf runtime *before any transformers/tokenizers import* and by building the tokenizer via a helper that retries with `use_fast=False` on the specific `MessageFactory.GetPrototype` failure. We also make cell 8’s exception handling robust (catching the actual exception type) so it reliably falls back instead of stopping execution. These are runtime/stability fixes only and do not change the model, checkpoints, folding/averaging, or postprocessing semantics, so your score should remain essentially unchanged (still above the target band). Finally, we renumber the cells to start at 1 as required while preserving the original execution order and ensure `submission.csv` is written with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SequentialSampler

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

from torch.optim import AdamW

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
    num_gpus = torch.cuda.device_count()
    if num_gpus:
        return 0
    return 0


def build_tokenizer(name_or_path):
    try:
        tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
        _ = tok("test", "context", truncation="only_second", max_length=8)
        return tok
    except Exception as e:
        print(
            f"[WARN] Fast tokenizer failed ({type(e).__name__}: {e}); falling back to slow tokenizer."
        )
        tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=False)
        return tok


print(f"Apex AMP Installed :: {APEX_INSTALLED}")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fix_all_seeds(2021)




## === cell 1
class Configration:
    model_type = "xlm_roberta"

    HF_name_or_path = "deepset/xlm-roberta-base-squad2"

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
    config = AutoConfig.from_pretrained(args.HF_name_or_path)
    tokenizer = build_tokenizer(args.HF_name_or_path)
    model = Model(args.HF_name_or_path, config=config)
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


def Postprocess_qa_predictions(
    examples, features, raw_predictions, tokenizer, n_best_size=20, max_answer_length=30
):
    all_start_logits, all_end_logits = raw_predictions

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    cls_id = tokenizer.cls_token_id
    cached_offsets = [None] * len(features)
    for i, f in enumerate(features):
        seq_ids = f["sequence_ids"]
        offsets = f["offset_mapping"]
        cached_offsets[i] = [
            (o if seq_ids[k] == 1 else None) for k, o in enumerate(offsets)
        ]
        if "cls_index" not in f:
            try:
                f["cls_index"] = f["input_ids"].index(cls_id)
            except ValueError:
                f["cls_index"] = 0

    predictions = collections.OrderedDict()
    print(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )

    ex_ids = examples["id"].to_numpy()
    ex_contexts = examples["context"].to_numpy()

    def _topk_desc(x, k):
        if k >= x.shape[0]:
            return np.argsort(x)[::-1]
        idx = np.argpartition(x, -k)[-k:]
        return idx[np.argsort(x[idx])[::-1]]

    for example_index in range(len(ex_ids)):
        feature_indices = features_per_example[example_index]
        valid_answers = []
        context = ex_contexts[example_index]

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offsets = cached_offsets[feature_index]

            start_indexes = _topk_desc(start_logits, n_best_size)
            end_indexes = _topk_desc(end_logits, n_best_size)

            for start_index in start_indexes.tolist():
                if start_index >= len(offsets) or offsets[start_index] is None:
                    continue
                start_char = offsets[start_index][0]
                for end_index in end_indexes.tolist():
                    if (
                        end_index >= len(offsets)
                        or offsets[end_index] is None
                        or end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    end_char = offsets[end_index][1]
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

        predictions[ex_ids[example_index]] = best_answer["text"]

    return predictions




## === cell 7
DATA_DIR_CANDIDATES = [
    "../input/chaii-hindi-and-tamil-question-answering",
    "/kaggle/input/chaii-hindi-and-tamil-question-answering",
    "../kaggle/input/chaii-hindi-and-tamil-question-answering",
]
DATA_DIR = None
for d in DATA_DIR_CANDIDATES:
    if os.path.exists(os.path.join(d, "test.csv")):
        DATA_DIR = d
        break
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate chaii-hindi-and-tamil-question-answering/test.csv in expected paths."
    )

test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
test_df["context"] = test_df["context"].apply(lambda x: " ".join(str(x).split()))
test_df["question"] = test_df["question"].apply(lambda x: " ".join(str(x).split()))

print(test_df.shape)
test_df.head()



## === cell 8
args = Configration()
config, tokenizer, _ = Make_Model(args)  # build tokenizer once for feature generation

questions = test_df["question"].astype(str).str.lstrip().tolist()
contexts = test_df["context"].astype(str).tolist()
ids = test_df["id"].tolist()

try:
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
except Exception as e:
    if "GetPrototype" in str(e) or "MessageFactory" in str(e):
        print(
            "[WARN] Tokenizer call hit protobuf GetPrototype issue; rebuilding slow tokenizer."
        )
        tokenizer = AutoTokenizer.from_pretrained(args.HF_name_or_path, use_fast=False)
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
    else:
        raise

sample_mapping = tokenized["overflow_to_sample_mapping"]
test_features = []
for i in range(len(tokenized["input_ids"])):
    sample_idx = sample_mapping[i]
    test_features.append(
        {
            "example_id": ids[sample_idx],
            "context": contexts[sample_idx],
            "question": questions[sample_idx],
            "input_ids": tokenized["input_ids"][i],
            "attention_mask": tokenized["attention_mask"][i],
            "offset_mapping": tokenized["offset_mapping"][i],
            "sequence_ids": [0 if s is None else s for s in tokenized.sequence_ids(i)],
        }
    )

test_dataset = Dataset_Retriever(test_features, mode="test")

num_workers = optimal_num_of_loader_workers()
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.eval_batch_size,
    sampler=SequentialSampler(test_dataset),
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    persistent_workers=False,
)

print(f"Num test examples: {len(test_df)}, num features: {len(test_features)}")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
base_model = "../input/nothing-of-your_concern/MPNet3-HT/output/".replace("_", "-")
base_model = "../input/nothing-of-your-concern/MPNet3-HT/output/"


def _safe_load_state_dict(model, ckpt_path):
    if ckpt_path is None:
        return False
    if not os.path.exists(ckpt_path):
        return False
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return True


_global_args = args
_global_config = config
_global_tokenizer = tokenizer

_global_model = Model(_global_args.HF_name_or_path, config=_global_config).to(DEVICE)
_global_model.eval()


def Get_Predictions_with_model(model, checkpoint_rel_path=None):
    ckpt_abs = (
        os.path.join(base_model, checkpoint_rel_path)
        if checkpoint_rel_path is not None
        else None
    )
    loaded = _safe_load_state_dict(model, ckpt_abs)
    print(f"Checkpoint loaded: {loaded} ({ckpt_abs if ckpt_abs else 'None'})")

    n_feats = len(test_dataset)
    seq_len = args.max_seq_length

    start_logits = np.empty((n_feats, seq_len), dtype=np.float32)
    end_logits = np.empty((n_feats, seq_len), dtype=np.float32)

    offset = 0
    total_steps = math.ceil(n_feats / args.eval_batch_size)

    use_amp = DEVICE.type == "cuda"
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else torch.autocast(device_type="cpu", enabled=False)
    )

    with torch.inference_mode():
        for batch in tqdm(test_dataloader, total=total_steps):
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)

            with autocast_ctx:
                outputs_start, outputs_end = model(input_ids, attention_mask)

            bs = outputs_start.shape[0]
            s_cpu = outputs_start.float().cpu().numpy()
            e_cpu = outputs_end.float().cpu().numpy()
            start_logits[offset : offset + bs, :] = s_cpu
            end_logits[offset : offset + bs, :] = e_cpu
            offset += bs

    return start_logits, end_logits




## === cell 10
fold_paths = [
    "checkpoint-fold-0/pytorch_model.bin",
    "checkpoint-fold-1/pytorch_model.bin",
    "checkpoint-fold-2/pytorch_model.bin",
    "checkpoint-fold-3/pytorch_model.bin",
    "checkpoint-fold-4/pytorch_model.bin",
]

start_mean = None
end_mean = None
n_folds = 0

for p in fold_paths:
    s, e = Get_Predictions_with_model(_global_model, p)
    if start_mean is None:
        start_mean = s.astype(np.float32, copy=False)
        end_mean = e.astype(np.float32, copy=False)
        n_folds = 1
    else:
        n_folds += 1
        start_mean += (s - start_mean) / n_folds
        end_mean += (e - end_mean) / n_folds

start_logits = start_mean
end_logits = end_mean

del _global_model, _global_config
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

fin_preds = Postprocess_qa_predictions(
    test_df, test_features, (start_logits, end_logits), tokenizer=tokenizer
)

submission = []
for pid, pred in fin_preds.items():
    pred = " ".join(str(pred).split())
    pred = pred.strip(punctuation)
    submission.append((pid, pred))

sample = pd.DataFrame(submission, columns=["id", "PredictionString"])
test_data = pd.merge(left=test_df, right=sample, on="id", how="left")

print(sample.shape)
sample.head()



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
print(f"Wrote {out_path} with shape {test_data[['id','PredictionString']].shape}")



## === cell 12
sub = pd.read_csv("submission.csv")
print(sub.head())
print(sub.shape)
print(sub.columns.tolist())
print("Missing predictions:", sub["PredictionString"].isna().sum())
