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

0.7344911694526672

# 6. Current score

0.02925

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.02594) has done: 'I fix the root cause of the crash: the notebook assumes an attached HuggingFace model/checkpoint exists under `/kaggle/input`, but in your environment none are present, so `make_model()` raises before any submission is produced. To keep the pipeline end-to-end and offline-safe, I add a deterministic “extractive baseline” fallback that does not require transformers weights (it selects a likely answer span from the context using simple token overlap with the question). I also make cell ordering consistent (your “cell 0” becomes cell 1) and ensure `test_data` is always defined so the submission-writing cell cannot NameError. This produces a valid `submission.csv` and, while it won’t hit the target without the pretrained QA checkpoints, it run reliably and is the minimal legitimate way to “increase score” from “no score” to a non-zero score.'
- What this solution (achieved 0.02594) has done: 'Your current low score is coming from the offline fallback baseline, which is too weak for this QA task. To move toward the 0.734 target without changing the model architecture/training, I (1) make the transformer path actually usable by loading an attached local model/checkpoint when available and (2) add a light, metric-aligned answer cleaning step (whitespace/punctuation trimming) for both transformer and baseline outputs. Additionally, I fix the key mismatch where `Config` says “QA model” but `AutoModel` (not QA head) is used, by ensuring we only ever load base encoders plus your existing linear QA head from checkpoints when present. These are minimal changes that keep your core inference/postprocess logic intact but should substantially raise the score whenever the checkpoint files exist under `/kaggle/input`.'
- What this solution (achieved 0.02504) has done: 'Your current score suggests the run is effectively using the weak baseline or (worse) a randomly-initialized QA head on top of a base encoder, which stay near-zero on Jaccard. To move toward the 0.734 target with minimal core-logic changes, I (1) force a safe fallback to the baseline whenever no compatible fine-tuned checkpoint is found (instead of predicting with an untrained QA head), and (2) strengthen the baseline slightly in a metric-aligned way by selecting a window that maximizes Jaccard overlap (not just raw overlap count) and by preferring shorter spans when tied. This keeps the architecture/training/postprocess intact for the transformer path, but avoids catastrophic low-scoring “random head” submissions and should materially increase the score toward the target. The script still runs end-to-end offline and always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.02983) has done: 'I keep your transformer/ensemble pipeline unchanged, but improve the fallback baseline so it produces much more extractive, context-faithful spans, which should raise Jaccard substantially from ~0.025 toward your 0.734 target when no fine-tuned checkpoints are available. Specifically, I make the baseline search operate on character spans (so it returns exact substrings from the original context, preserving punctuation/spacing) and use a lightweight token normalization for scoring that better matches Hindi/Tamil text. I also add an ultra-cheap “sentence/segment first” narrowing step to avoid O(n²) window scans on very long contexts, keeping runtime within limits. Submission writing and all paths remain the same, and transformer behavior is untouched.'
- What this solution (achieved 0.02967) has done: 'Your current score (0.02983) is far below the target (0.73449), so we should improve performance while keeping your transformer path untouched and making only minimal, score-relevant changes to the fallback baseline (which is what’s effectively being used without attached fine-tuned checkpoints). I strengthen the baseline by extracting spans directly from the original context using token-span windows (no lossy normalization for locating), and switch the window scoring from set-Jaccard to a bag-of-words Jaccard (multiset) which matches repeated tokens better and is still aligned with the competition metric. I also add a tiny language-aware window length tweak (Hindi tends to have slightly longer answer phrases) while keeping runtime bounded via the existing segment-first narrowing. These changes should move the baseline submission materially upward toward the target without altering your model architecture/training/inference pipeline.'
- What this solution (achieved 0.02967) has done: 'Your current score (0.02967) is far below the target (0.73449), and in this environment you’re effectively always using the fallback baseline (no fine-tuned checkpoints found). To move the score upward while keeping the overall approach intact, I minimally strengthen only the baseline extractor by (1) adding a very light IDF-like weighting so rare question tokens matter more than common ones and (2) allowing slightly longer answer windows with a small length penalty, which better matches typical chaii answer spans. I also add a tiny “question tail” boost (last tokens often carry the specific entity) when scoring windows, without changing any transformer path logic. Everything still runs offline, end-to-end, and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.02925) has done: 'Your current score is far below the target, and in this environment you’re almost certainly running the baseline extractor (no fine-tuned checkpoints). To move the score upward with minimal changes and without touching the transformer path, I strengthen only the baseline span selection: add a lightweight character n-gram overlap term (helps Hindi/Tamil where word tokenization is brittle) and a small “best exact-ish match” shortcut for short questions/entities. I also make the baseline’s segment selection use the same combined score (BoW Jaccard + weighted coverage + char n-gram) so it more often searches in the right sentence. These tweaks keep the same extractive heuristic core (segment → window search → substring from context) but should materially increase Jaccard toward your target while remaining deterministic and fast.'

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
from tqdm import tqdm
from string import punctuation

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler

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
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus * 4) if num_gpus else max(1, num_cpus - 1)
    return optimal_value


print(f"Apex AMP Installed :: {APEX_INSTALLED}")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 1
KAGGLE_INPUT_ROOT = "/kaggle/input"


def _resolve_kaggle_path(p: str) -> str:
    """
    Convert '../input/...' to '/kaggle/input/...'.
    If a relative path exists under /kaggle/input, resolve it there.
    Keep absolute paths unchanged.
    """
    if p is None:
        return p
    p = str(p)

    if p.startswith("../input/"):
        p = os.path.join(KAGGLE_INPUT_ROOT, p.replace("../input/", "", 1))

    if not os.path.isabs(p) and os.path.exists(os.path.join(KAGGLE_INPUT_ROOT, p)):
        p = os.path.join(KAGGLE_INPUT_ROOT, p)

    return os.path.normpath(p)


def _hf_from_pretrained_kwargs():
    return {"local_files_only": True}


def _discover_hf_model_dirs(root="/kaggle/input", limit=20):
    """
    Find a usable local HF model dir under /kaggle/input.
    A usable dir typically contains: config.json and tokenizer.json (or vocab files).
    """
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        fn = set(filenames)
        if "config.json" in fn and (
            "tokenizer.json" in fn
            or "sentencepiece.bpe.model" in fn
            or "vocab.json" in fn
            or "vocab.txt" in fn
            or "vocab.txt" in fn
        ):
            low = dirpath.lower()
            if any(
                k in low for k in ["chaii", "xlm", "roberta", "mbert", "bert", "qa"]
            ):
                hits.append(dirpath)
    hits = sorted(hits, key=lambda p: (-p.count(os.sep), p))[:limit]
    return hits




## === cell 2
class Config:
    model_type = "xlm-roberta"
    model_name_or_path = "../input/abhishek-chaii-qa-model"
    config_name = "../input/abhishek-chaii-qa-model"
    fp16 = True if APEX_INSTALLED else False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2

    tokenizer_name = "../input/abhishek-chaii-qa-model"
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




## === cell 4
class Model(nn.Module):
    def __init__(self, modelname_or_path, config):
        super(Model, self).__init__()
        self.config = config
        self.xlm_roberta = AutoModel.from_pretrained(
            modelname_or_path, config=config, **_hf_from_pretrained_kwargs()
        )
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




## === cell 5
def make_model(args):
    """
    Offline-only loader. Returns (config, tokenizer, model).
    If nothing is available locally, raises FileNotFoundError (handled by caller to fall back).

    Change (score-relevant, minimal): prefer a discovered *model dir* that matches attached chaii assets
    to avoid accidentally picking a tokenizer-only folder.
    """
    cfg_path = _resolve_kaggle_path(args.config_name)
    tok_path = _resolve_kaggle_path(args.tokenizer_name)
    mdl_path = _resolve_kaggle_path(args.model_name_or_path)

    all_exist = all(os.path.isdir(p) for p in [cfg_path, tok_path, mdl_path])

    if all_exist:
        chosen_dir = mdl_path
        cfg_dir = cfg_path
        tok_dir = tok_path
        config = AutoConfig.from_pretrained(cfg_dir, **_hf_from_pretrained_kwargs())
        tokenizer = AutoTokenizer.from_pretrained(
            tok_dir, use_fast=True, **_hf_from_pretrained_kwargs()
        )
        model = Model(chosen_dir, config=config)
        return config, tokenizer, model

    candidates = _discover_hf_model_dirs("/kaggle/input", limit=30)
    if len(candidates) > 0:
        chosen_dir = candidates[0]
        print(
            f"[WARN] Local model dir not found at configured path. Using discovered model dir: {chosen_dir}"
        )
        config = AutoConfig.from_pretrained(chosen_dir, **_hf_from_pretrained_kwargs())
        tokenizer = AutoTokenizer.from_pretrained(
            chosen_dir, use_fast=True, **_hf_from_pretrained_kwargs()
        )
        model = Model(chosen_dir, config=config)
        return config, tokenizer, model

    fallback_ids = [
        "xlm-roberta-base",
        "bert-base-multilingual-cased",
    ]
    last_err = None
    for mid in fallback_ids:
        try:
            config = AutoConfig.from_pretrained(mid, **_hf_from_pretrained_kwargs())
            tokenizer = AutoTokenizer.from_pretrained(
                mid, use_fast=True, **_hf_from_pretrained_kwargs()
            )
            model = Model(mid, config=config)
            print(
                f"[WARN] No attached model dir found. Using offline-cached base model: {mid}"
            )
            return config, tokenizer, model
        except Exception as e:
            last_err = e

    raise FileNotFoundError(
        "No local HuggingFace model directory found under /kaggle/input and no suitable cached model available. "
        "Attach a model dataset (e.g. '../input/abhishek-chaii-qa-model') or ensure a cached model like "
        "'xlm-roberta-base' exists. Last error: " + repr(last_err)
    )




## === cell 6
def prepare_test_features(args, example, tokenizer):
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
            0 if x is None else x for x in tokenized_example.sequence_ids(i)
        ]
        features.append(feature)
    return features




## === cell 7
import collections


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
        valid_answers = []
        context = example["context"]

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            sequence_ids = features[feature_index]["sequence_ids"]
            context_index = 1

            offset_mapping = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(features[feature_index]["offset_mapping"])
            ]

            input_ids = features[feature_index]["input_ids"]
            try:
                cls_index = input_ids.index(tokenizer.cls_token_id)
            except Exception:
                cls_index = 0

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
            best_answer = {"text": "", "score": 0.0}

        predictions[example["id"]] = best_answer["text"]

    return predictions




## === cell 8
import re
from collections import Counter


def _normalize_ws(s: str) -> str:
    return " ".join(str(s).split())


_TOKEN_RE = re.compile(r"[\w\u0900-\u097F\u0B80-\u0BFF]+", flags=re.UNICODE)


def _norm_for_match(s: str) -> str:
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokens_for_scoring(s: str):
    s = _norm_for_match(s)
    return _TOKEN_RE.findall(s)


def _jaccard_bow(a_tokens, b_tokens) -> float:
    ca = Counter(a_tokens)
    cb = Counter(b_tokens)
    if len(ca) == 0 and len(cb) == 0:
        return 1.0
    if len(ca) == 0 or len(cb) == 0:
        return 0.0
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return float(inter) / float(union) if union > 0 else 0.0


def _split_segments_with_spans(context: str):
    ctx = (
        ""
        if context is None or (isinstance(context, float) and np.isnan(context))
        else str(context)
    )
    spans = []
    start = 0
    for m in re.finditer(r"[\n\r]+|[।!?]|[\.]{2,}|[;]", ctx):
        end = m.end()
        if end > start:
            seg = ctx[start:end]
            if seg.strip():
                spans.append((start, end, seg))
        start = end
    if start < len(ctx):
        seg = ctx[start:]
        if seg.strip():
            spans.append((start, len(ctx), seg))
    if len(spans) == 0:
        spans = [(0, len(ctx), ctx)]
    return spans


def _token_spans_in_text(text: str):
    spans = []
    for m in _TOKEN_RE.finditer(text):
        spans.append((m.start(), m.end(), m.group(0)))
    return spans


def _q_token_weights(q_tokens):
    cnt = Counter(q_tokens)
    w = {t: 1.0 / math.sqrt(c) for t, c in cnt.items()}
    return w


def _weighted_overlap_score(window_tokens, q_tokens, q_w, q_tail_set):
    if not q_tokens:
        return 0.0
    covered = set(window_tokens)
    num = 0.0
    den = 0.0
    for t in set(q_tokens):
        wt = q_w.get(t, 1.0)
        den += wt
        if t in covered:
            num += wt
    base = (num / den) if den > 0 else 0.0
    if q_tail_set and any(t in q_tail_set for t in covered):
        base += 0.05
    return base


def _char_ngrams(s: str, n: int = 3):
    s = _norm_for_match(s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) < n:
        return [s] if s else []
    return [s[i : i + n] for i in range(0, len(s) - n + 1)]


def _jaccard_set(a_list, b_list) -> float:
    a = set(a_list)
    b = set(b_list)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    c = a.intersection(b)
    return float(len(c)) / float(len(a) + len(b) - len(c))


def baseline_extract_answer(
    context: str,
    question: str,
    max_answer_tokens: int = 20,
    language: str | None = None,
) -> str:
    """
    Deterministic extractive heuristic:
    1) pick best context segment by similarity(question, segment)
    2) within that segment, pick best *character* window built from token spans.

    Change (score-relevant, minimal): keep the same segment->window search, but:
      - score segments with the same combined score used for windows (BoW + weighted coverage + char n-gram)
      - add an "exact-ish short phrase" shortcut for short questions/entities (often improves Jaccard cheaply)
    """
    context = "" if pd.isna(context) else str(context)
    question = "" if pd.isna(question) else str(question)
    q_toks = _tokens_for_scoring(question)

    if len(context.strip()) == 0:
        return ""
    if len(q_toks) == 0:
        return context.strip()[: min(len(context.strip()), 60)].strip()

    if language is not None:
        lang = str(language).lower()
        if "hindi" in lang:
            max_answer_tokens = max(max_answer_tokens, 28)
        elif "tamil" in lang:
            max_answer_tokens = max(max_answer_tokens, 24)
    max_answer_tokens = min(max_answer_tokens, 32)

    q_w = _q_token_weights(q_toks)
    q_tail = q_toks[-6:] if len(q_toks) >= 6 else q_toks
    q_tail_set = set(q_tail)

    q_clean = _norm_for_match(question)
    q_clean = re.sub(r"[^\w\u0900-\u097F\u0B80-\u0BFF\s]+", " ", q_clean).strip()
    q_clean = re.sub(r"\s+", " ", q_clean).strip()
    if 0 < len(q_clean) <= 28:
        pos = _norm_for_match(context).find(q_clean)
        if pos != -1:
            pos2 = context.find(q_clean)
            if pos2 != -1:
                return context[pos2 : pos2 + len(q_clean)].strip()

    q_tri = _char_ngrams(question, n=3)

    best_seg = (0, len(context), context)
    best_seg_score = -1e9
    for s0, s1, seg in _split_segments_with_spans(context):
        seg_toks = _tokens_for_scoring(seg)
        if len(seg_toks) == 0:
            continue
        j = _jaccard_bow(seg_toks, q_toks)
        cov = _weighted_overlap_score(seg_toks, q_toks, q_w, q_tail_set)
        cj = _jaccard_set(_char_ngrams(seg, n=3), q_tri) if q_tri else 0.0
        seg_score = (j + 0.20 * cov + 0.25 * cj) - 0.0005 * max(0, len(seg_toks) - 80)
        if seg_score > best_seg_score:
            best_seg_score = seg_score
            best_seg = (s0, s1, seg)

    seg_offset, _, seg_text = best_seg

    raw_spans = _token_spans_in_text(seg_text)
    if len(raw_spans) == 0:
        return seg_text.strip()[: min(len(seg_text.strip()), 80)].strip()

    max_len = min(max_answer_tokens, len(raw_spans))

    qset = set(q_toks)
    best_score = -1e9
    best_span = (raw_spans[0][0], raw_spans[0][1])
    best_L = 1

    for L in range(1, max_len + 1):
        for i in range(0, len(raw_spans) - L + 1):
            window_tokens = [_norm_for_match(raw_spans[k][2]) for k in range(i, i + L)]
            if best_score > 0.0 and not any(t in qset for t in window_tokens):
                continue

            j = _jaccard_bow(window_tokens, q_toks)
            cov = _weighted_overlap_score(window_tokens, q_toks, q_w, q_tail_set)

            start_char = raw_spans[i][0]
            end_char = raw_spans[i + L - 1][1]
            cand_text = seg_text[start_char:end_char]
            cj = _jaccard_set(_char_ngrams(cand_text, n=3), q_tri) if q_tri else 0.0

            score = (j + 0.25 * cov + 0.30 * cj) - 0.004 * L

            if (score > best_score) or (score == best_score and L < best_L):
                best_score = score
                best_span = (start_char, end_char)
                best_L = L

    ans = seg_text[best_span[0] : best_span[1]].strip()
    if ans == "":
        ans = seg_text.strip()[: min(len(seg_text.strip()), 80)].strip()
    return ans


def clean_pred_text(pred: str) -> str:
    pred = (
        ""
        if pred is None or (isinstance(pred, float) and np.isnan(pred))
        else str(pred)
    )
    pred = " ".join(pred.split()).strip()
    pred = pred.strip(punctuation)
    return pred


test = pd.read_csv("/kaggle/input/chaii-hindi-and-tamil-question-answering/test.csv")

test["context"] = test["context"].apply(lambda x: " ".join(str(x).split()))
test["question"] = test["question"].apply(lambda x: " ".join(str(x).split()))

args = Config()
fix_all_seeds(args.seed)

tokenizer = None
use_transformer = True
try:
    _config, tokenizer, _tmp_model = make_model(args)
    del _tmp_model, _config
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except FileNotFoundError as e:
    print(
        "[WARN] No offline HF model available; switching to baseline extractive submission."
    )
    print("       Details:", str(e)[:300], "...")
    use_transformer = False

test_features = []
test_dataloader = None
if use_transformer:
    for _, row in tqdm(test.iterrows(), total=len(test), desc="Tokenizing test"):
        test_features += prepare_test_features(args, row, tokenizer)

    test_dataset = DatasetRetriever(test_features, mode="test")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        sampler=SequentialSampler(test_dataset),
        num_workers=optimal_num_of_loader_workers(),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )




## === cell 9
def _load_state_dict_flex(path, device):
    obj = torch.load(path, map_location=device)
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise ValueError("Unsupported checkpoint format")
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        new_sd[nk] = v
    return new_sd


def get_predictions(checkpoint_path=None):
    """
    Run inference. Uses make_model() which is offline-safe.
    """
    config, _tok, model = make_model(Config())
    model.to(DEVICE)
    model.eval()

    if checkpoint_path is not None:
        state_dict = _load_state_dict_flex(checkpoint_path, DEVICE)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(unexpected) > 0:
            print(
                f"[WARN] Unexpected keys in {checkpoint_path}: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}"
            )
        if len(missing) > 0:
            print(
                f"[WARN] Missing keys in {checkpoint_path}: {missing[:5]}{'...' if len(missing)>5 else ''}"
            )
    else:
        print(
            "[WARN] No checkpoint provided; using base model weights for predictions."
        )

    start_logits = []
    end_logits = []
    for batch in tqdm(
        test_dataloader,
        desc=f"Predict {os.path.basename(checkpoint_path) if checkpoint_path else 'base'}",
    ):
        with torch.no_grad():
            outputs_start, outputs_end = model(
                batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)
            )
            start_logits.append(outputs_start.detach().cpu().numpy())
            end_logits.append(outputs_end.detach().cpu().numpy())

    del model, _tok, config
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.vstack(start_logits), np.vstack(end_logits)




## === cell 10
test_data = None

if not use_transformer:
    preds = []
    for _id, ctx, q, lang in tqdm(
        test[["id", "context", "question", "language"]].itertuples(
            index=False, name=None
        ),
        total=len(test),
        desc="Baseline predict",
    ):
        pred = baseline_extract_answer(ctx, q, max_answer_tokens=20, language=lang)
        pred = clean_pred_text(pred)
        preds.append((_id, pred))
    test_data = test.merge(
        pd.DataFrame(preds, columns=["id", "PredictionString"]), on="id", how="left"
    )
else:
    checkpoint_paths = [
        (
            "../input/chaii-abhishek-fold-0/output/checkpoint-fold-0/pytorch_model.bin",
            0.10,
        ),
        (
            "../input/chaii-abhishek-fold-1/output/checkpoint-fold-1/pytorch_model.bin",
            0.20,
        ),
        (
            "../input/chaii-abhishek-fold-2/output/checkpoint-fold-2/pytorch_model.bin",
            0.20,
        ),
        (
            "../input/chaii-abhishek-fold-3/output/checkpoint-fold-3/pytorch_model.bin",
            0.05,
        ),
        (
            "../input/chaii-abhishek-model-2-epochs-10folds/output/checkpoint-fold-5/pytorch_model.bin",
            0.05,
        ),
        (
            "../input/k/abhiram4572/chaii-abhishek-model-2-epochs-10folds/output/checkpoint-fold-7/pytorch_model.bin",
            0.10,
        ),
        (
            "../input/k/vineethakki/chaii-abhishek-model-2-epochs-10folds/output/checkpoint-fold-8/pytorch_model.bin",
            0.20,
        ),
        (
            "../input/k/vineethakkinapalli/chaii-abhishek-model-2-epochs-10folds/output/checkpoint-fold-9/pytorch_model.bin",
            0.10,
        ),
    ]

    available = []
    for p, w in checkpoint_paths:
        p2 = _resolve_kaggle_path(p)
        if os.path.exists(p2):
            available.append((p2, w))
        else:
            print(f"[WARN] Checkpoint not found, skipping: {p2}")

    def _discover_checkpoints(root="/kaggle/input", limit=12):
        hits = []
        for dirpath, dirnames, filenames in os.walk(root):
            if "pytorch_model.bin" in filenames:
                full = os.path.join(dirpath, "pytorch_model.bin")
                low = full.lower()
                if any(
                    k in low
                    for k in [
                        "chaii",
                        "qa",
                        "question-answer",
                        "xlm",
                        "roberta",
                        "fold",
                        "checkpoint",
                    ]
                ):
                    hits.append(full)
        hits = sorted(hits)[:limit]
        return hits

    if len(available) == 0:
        discovered = _discover_checkpoints("/kaggle/input", limit=8)
        if len(discovered) == 0:
            print(
                "[WARN] No fine-tuned checkpoint files found under /kaggle/input. "
                "Falling back to baseline extractive submission (better than random QA head)."
            )
            use_transformer = False
        else:
            print(
                "[WARN] No hardcoded checkpoints found; using discovered checkpoints instead:"
            )
            for p in discovered:
                print("  ", p)
            available = [(p, 1.0 / len(discovered)) for p in discovered]
    else:
        w_sum = sum(w for _, w in available)
        available = [(p, w / w_sum) for p, w in available]

    if not use_transformer:
        preds = []
        for _id, ctx, q, lang in tqdm(
            test[["id", "context", "question", "language"]].itertuples(
                index=False, name=None
            ),
            total=len(test),
            desc="Baseline predict (no ckpt)",
        ):
            pred = baseline_extract_answer(ctx, q, max_answer_tokens=20, language=lang)
            pred = clean_pred_text(pred)
            preds.append((_id, pred))
        test_data = test.merge(
            pd.DataFrame(preds, columns=["id", "PredictionString"]), on="id", how="left"
        )
    else:
        print("Using checkpoints:")
        for p, w in available:
            print(f"  {p}  weight={w:.4f}")

        ens_start = None
        ens_end = None
        for ckpt, w in available:
            s, e = get_predictions(ckpt)
            if ens_start is None:
                ens_start = w * s
                ens_end = w * e
            else:
                ens_start += w * s
                ens_end += w * e
            del s, e
            gc.collect()
        start_logits = ens_start
        end_logits = ens_end

        fin_preds = postprocess_qa_predictions(
            test, test_features, (start_logits, end_logits), tokenizer=tokenizer
        )

        submission = []
        for pid, pred in fin_preds.items():
            pred = clean_pred_text(pred)
            submission.append((pid, pred))

        sample = pd.DataFrame(submission, columns=["id", "PredictionString"])
        test_data = pd.merge(left=test, right=sample, on="id", how="left")




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

    pred = " ".join(pred.split()).strip()
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
test_data[["id", "PredictionString"]].to_csv(
    "submission.csv", index=False, encoding="utf-8"
)
print("Wrote submission.csv with shape:", test_data[["id", "PredictionString"]].shape)
print(test_data[["id", "PredictionString"]].head())
