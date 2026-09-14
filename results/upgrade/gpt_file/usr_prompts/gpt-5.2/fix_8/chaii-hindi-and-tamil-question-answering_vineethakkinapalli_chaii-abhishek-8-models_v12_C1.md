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

0.0054513025097548

# 6. Current score

0.00298

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00298) has done: 'I fix the runtime crash caused by an incompatible protobuf version (the `MessageFactory.GetPrototype` error) by forcing the pure-Python protobuf implementation before importing `transformers`/tokenizers. I also make the checkpoint loading robust: if the referenced Kaggle input model path is missing, the code fall back to a deterministic baseline prediction (empty string) so it still produces a valid `submission.csv`. Finally, I ensure `test_data` is always created and the submission file is always written with the exact required columns and row count, unblocking end-to-end execution.'
- What this solution (achieved 0.00298) has done: 'Your current score (0.00298) is below the target (0.0054513), so we need a small, safe boost without changing the model/training core. The biggest issue is that your code is very likely falling back to empty predictions because the referenced checkpoint path doesn’t exist in this environment; fixing the checkpoint discovery so an actual QA checkpoint is loaded should move the score upward toward the target. I add a minimal “find an existing `pytorch_model.bin` under `../input`” fallback (only used if the hardcoded path is missing), keep the exact same inference/postprocess logic, and ensure we still always write a valid `submission.csv`. I also add a tiny safety fix for `cls_token_id` on XLM-R (it’s typically `None`), avoiding accidental behavior differences in postprocessing.'
- What this solution (achieved 0.00298) has done: 'Your current score (0.00298) is below the target (0.0054513), so we should make a small, safe improvement without changing the model/training core. The biggest low-risk gain here is to make inference robustly use the best available checkpoint under `../input`, instead of potentially loading a mismatched/non-QA `pytorch_model.bin` (or a partially compatible state) and producing near-random spans. I (1) improve checkpoint discovery to prefer directories that look like *chaii/qa/xlm-roberta* and ensure we load a fully compatible state dict, and (2) add a minimal “no-answer” handling consistent with standard QA postprocessing by allowing empty prediction when the [CLS] score is better than any span. These changes keep your architecture and inference loop intact but should raise the score toward the target band.'
- What this solution (achieved 0.00298) has done: 'Your current score is below the target, so the safest way to move it upward without changing your model/training core is to (1) make sure you actually load a compatible QA checkpoint (not just any `pytorch_model.bin`) by filtering candidates using the presence of `config.json` and by preferring paths that look like chaii QA checkpoints, and (2) fix the biggest silent mismatch: you build your model using a base config (no `qa_outputs`) but then load QA weights; instead, we keep your exact custom head, but we initialize the backbone with `AutoModel.from_pretrained(checkpoint_dir)` when available so embeddings/backbone weights match the checkpoint. Finally, we keep your postprocess logic intact but set a small positive `null_score_diff_threshold` to reduce empty answers (empty answers tend to hurt Jaccard), which should give a modest lift toward the target without trying to over-optimize.'
- What this solution (achieved 0.00298) has done: 'Your current score (0.00298) is below the target (0.0054513), so we should make a small, low-risk improvement that preserves your exact model/inference core. The biggest likely gap is checkpoint mismatch: you initialize the backbone from a checkpoint dir but the tokenizer used for feature creation remains fixed to `xlm-roberta-base`, which can cause tokenization/offset misalignment (especially if the checkpoint is from a different XLM-R variant) and hurt Jaccard. I minimally change the code to (1) select/load the tokenizer from the same checkpoint directory that the model backbone is loaded from (while keeping your architecture, weights loading, and postprocess intact), and (2) keep the same tokenizer instance consistently for feature creation and postprocessing. This should improve answer span alignment and nudge the score upward toward your target without altering the modeling approach.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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




## === cell 1
class Config:
    model_type = "xlm-roberta"
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
        pooled_output = outputs[1] if len(outputs) > 1 else None

        linear_output = self.linear_layer(sequence_output)
        linear_output = self.dropout(linear_output)
        qa_logits = self.qa_outputs(linear_output)

        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits




## === cell 4
def make_model(args, backbone_init_path=None):
    if backbone_init_path is None:
        config = AutoConfig.from_pretrained(args.config_name)
        model_path = args.model_name_or_path
    else:
        config = AutoConfig.from_pretrained(backbone_init_path)
        model_path = backbone_init_path

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    model = Model(model_path, config=config)
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
            0 if x is None else x for x in tokenized_example.sequence_ids(i)
        ]
        features.append(feature)
    return features




## === cell 6
import collections


def postprocess_qa_predictions(
    examples,
    features,
    raw_predictions,
    tokenizer,
    n_best_size=20,
    max_answer_length=30,
    null_score_diff_threshold=0.0,
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

        valid_answers = []
        context = example["context"]

        best_null_score = None

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

            cls_token_id = tokenizer.cls_token_id
            if (
                cls_token_id is not None
                and cls_token_id in features[feature_index]["input_ids"]
            ):
                cls_index = features[feature_index]["input_ids"].index(cls_token_id)
            else:
                cls_index = 0

            null_score = float(start_logits[cls_index] + end_logits[cls_index])
            if best_null_score is None or null_score > best_null_score:
                best_null_score = null_score

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
                            "score": float(
                                start_logits[start_index] + end_logits[end_index]
                            ),
                            "text": context[start_char:end_char],
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                0
            ]
        else:
            best_answer = {"text": "", "score": -1e9}

        if best_null_score is None:
            best_null_score = -1e9
        score_diff = best_null_score - float(best_answer["score"])
        if score_diff > null_score_diff_threshold:
            predictions[example["id"]] = ""
        else:
            predictions[example["id"]] = best_answer["text"]

    return predictions




## === cell 7
args = Config()
fix_all_seeds(args.seed)

test = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/test.csv")

test["context"] = test["context"].apply(lambda x: " ".join(str(x).split()))
test["question"] = test["question"].apply(lambda x: " ".join(str(x).split()))

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

print("Loaded test:", test.shape)



## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)




## === cell 9
def _find_checkpoint_bin(
    search_root="../input", filename="pytorch_model.bin", max_hits=400
):
    hits = []
    for root, _dirs, files in os.walk(search_root):
        if filename in files:
            hits.append(os.path.join(root, filename))
            if len(hits) >= max_hits:
                break
    return hits


def _score_checkpoint_path(p: str) -> int:
    s = p.lower()
    score = 0
    for kw, w in [
        ("chaii", 80),
        ("hindi", 10),
        ("tamil", 10),
        ("qa", 25),
        ("question", 10),
        ("answer", 10),
        ("xlm", 15),
        ("roberta", 15),
        ("squad", 10),
        ("checkpoint", 5),
        ("fold", 3),
        ("output", 2),
    ]:
        if kw in s:
            score += w
    if "bert" in s and "xlm" not in s:
        score -= 5
    return score


def _checkpoint_dir_from_bin(p: str) -> str:
    return os.path.dirname(p)


def _looks_like_hf_checkpoint_dir(d: str) -> bool:
    return os.path.exists(os.path.join(d, "config.json"))


def _select_best_checkpoint(candidates):
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda p: (
            int(_looks_like_hf_checkpoint_dir(_checkpoint_dir_from_bin(p))),
            _score_checkpoint_path(p),
            len(p),
        ),
        reverse=True,
    )
    return ranked[0]


def get_predictions(checkpoint_path, tokenizer_for_inference):
    if not os.path.exists(checkpoint_path):
        print(f"WARNING: checkpoint not found: {checkpoint_path}")
        candidates = _find_checkpoint_bin("../input", "pytorch_model.bin")
        chosen = _select_best_checkpoint(candidates)
        if chosen is not None:
            print("Found alternative checkpoint. Using:", chosen)
            checkpoint_path = chosen
        else:
            print(
                "No checkpoint found under ../input. Falling back to empty-string predictions (baseline)."
            )
            n_feat = len(test_features)
            seq_len = Config().max_seq_length
            return np.zeros((n_feat, seq_len), dtype=np.float32), np.zeros(
                (n_feat, seq_len), dtype=np.float32
            )

    checkpoint_dir = os.path.dirname(checkpoint_path)

    config, _tok_unused, model = make_model(Config(), backbone_init_path=checkpoint_dir)
    model.to(device)
    model.eval()

    state = torch.load(checkpoint_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]

    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            new_state[nk] = v
        state = new_state

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(unexpected) > 0:
        print("WARNING: unexpected keys when loading state_dict:", unexpected[:5])
    if len(missing) > 0:
        print("WARNING: missing keys when loading state_dict:", missing[:5])

    start_logits = []
    end_logits = []
    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            outputs_start, outputs_end = model(input_ids, attention_mask)
            start_logits.append(outputs_start.detach().cpu().numpy())
            end_logits.append(outputs_end.detach().cpu().numpy())
            del outputs_start, outputs_end, input_ids, attention_mask
    del model, _tok_unused, config, state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.vstack(start_logits), np.vstack(end_logits)




## === cell 10
checkpoint_path = "../input/chaii-abhishek-model-2-epochs-10folds/output/checkpoint-fold-4/pytorch_model.bin"

if not os.path.exists(checkpoint_path):
    candidates = _find_checkpoint_bin("../input", "pytorch_model.bin")
    chosen = _select_best_checkpoint(candidates)
    if chosen is not None:
        print("Primary checkpoint missing; using discovered checkpoint:", chosen)
        checkpoint_path = chosen
    else:
        print(
            "Primary checkpoint missing and none discovered; will fall back to baseline."
        )

ckpt_dir = os.path.dirname(checkpoint_path) if os.path.exists(checkpoint_path) else None
if ckpt_dir is not None and os.path.exists(os.path.join(ckpt_dir, "config.json")):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
    print("Tokenizer loaded from checkpoint dir:", ckpt_dir)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    print("Tokenizer loaded from default:", args.tokenizer_name)

test_features = []
for _, row in test.iterrows():
    test_features += prepare_test_features(args, row, tokenizer)

test_dataset = DatasetRetriever(test_features, mode="test")
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.eval_batch_size,
    sampler=SequentialSampler(test_dataset),
    num_workers=optimal_num_of_loader_workers(),
    pin_memory=True,
    drop_last=False,
)

print("Test examples:", len(test), "Test features:", len(test_features))

start_logits1, end_logits1 = get_predictions(
    checkpoint_path, tokenizer_for_inference=tokenizer
)

fin_preds = postprocess_qa_predictions(
    test,
    test_features,
    (start_logits1, end_logits1),
    tokenizer=tokenizer,
    null_score_diff_threshold=1.0,
)

submission = []
for p1, p2 in fin_preds.items():
    p2 = " ".join(str(p2).split())
    p2 = p2.strip(punctuation)
    submission.append((p1, p2))

sample = pd.DataFrame(submission, columns=["id", "PredictionString"])
test_data = pd.merge(left=test, right=sample, on="id", how="left")
print(
    "Merged rows:",
    len(test_data),
    "Null preds:",
    test_data["PredictionString"].isna().sum(),
)



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

out_path = "submission.csv"
sub_df = test_data[["id", "PredictionString"]].copy()

sub_df["PredictionString"] = sub_df["PredictionString"].fillna("")

sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(sub_df))
print(sub_df.head())



## === cell 12
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["id", "PredictionString"]
assert len(chk) == len(test)
print("Submission OK. Head:")
print(chk.head())
