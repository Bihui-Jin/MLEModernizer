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

0.7294511198997498

# 6. Current score

0.02494

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00786) has done: 'The changes focus on eliminating the repeated model construction for each checkpoint and reducing DataLoader overhead.  
1. `test_dataloader` now uses a single worker (`num_workers=0`) because loading many workers adds significant overhead for the relatively small test set.  
2. `get_predictions` is rewritten to optionally accept an already‑created model; when a model is supplied it only loads the checkpoint and runs inference, avoiding re‑instantiating the model and tokenizer each time.  
3. In the checkpoint loop we create the model once, then accumulate start/end logits across all valid checkpoints, averaging them at the end. This keeps the exact same inference logic while cutting the total runtime dramatically.'
- What this solution (achieved 0.01339) has done: 'I bypass the failing model loading (which raises a protobuf `MessageFactory` error) and replace it with a lightweight heuristic that extracts a word from the context overlapping the question. This keeps the overall pipeline structure while guaranteeing that a valid `submission.csv` is written. The heuristic is simple yet provides non‑empty predictions, moving the Jaccard score far above the current 0.00786 without altering the core model architecture.'
- What this solution (achieved 0.01555) has done: 'I improve the heuristic that generates predictions. Instead of returning only the first matching token, the new version extracts a short window (default 3 words) starting at the first overlapping token between question and context, producing a more informative answer span. This small change keeps the overall pipeline unchanged while giving the Jaccard metric a better chance to rise toward the target score.'
- What this solution (achieved 0.00229) has done: 'I replace the simple heuristic with the existing QA model inference pipeline: after building the test features I run the model (using the pretrained XLM‑Roberta weights) to obtain start/end logits, then post‑process them with the provided `postprocess_qa_predictions` function to generate answer strings. The cleaning steps that follow remain unchanged. This uses the core model logic already present, adds no new dependencies, and is expected to raise the Jaccard score markedly toward the target.'
- What this solution (achieved 0.025) has done: 'The fix adds all missing imports, defines the previously undefined `APEX_INSTALLED` flag, and corrects the data‑loading paths so the notebook runs without errors. The core logic (model architecture, feature creation, and post‑processing) remains unchanged, and the heuristic answer generation is kept as a lightweight fallback that reliably produces a non‑empty `submission.csv` in the required format. These minimal changes unblock execution and generate a valid submission file, moving the solution toward the target score.'
- What this solution (achieved 0.02122) has done: 'The change replaces the exhaustive‑search heuristic with a minimal‑span version that returns the first token from the context that appears in the question (or a short surrounding window). This keeps the pipeline unchanged while producing much tighter answer strings, which substantially raises the Jaccard overlap and moves the score toward the target.'
- What this solution (achieved 0.02692) has done: 'I replace the simple token‑match heuristic with a lightweight search that scans short spans (up to 8 tokens) in the context and selects the one with the highest Jaccard overlap against the question tokens. This keeps the overall pipeline unchanged while producing more relevant answer spans, which should increase the Jaccard‑based score toward the target.'
- What this solution (achieved 0.00287) has done: 'I replace the simple heuristic with an inference step that uses the defined XLM‑Roberta QA model. The model is loaded with the existing configuration, run on the prepared test DataLoader, and its start/end logits are post‑processed with the provided `postprocess_qa_predictions` function to generate answer strings. After that the same cleaning logic is applied and the submission file is written. This keeps the original architecture and preprocessing intact while substantially improving the Jaccard score, moving it toward the target.'
- What this solution (achieved 0.00357) has done: 'Implemented two critical fixes to get the pipeline running and produce a valid submission:  

1. Set the protobuf implementation to the pure‑Python version *before* importing any `transformers` objects to avoid the `MessageFactory` error.  
2. Corrected device handling during inference – the model’s parameters are moved to the chosen device, and the tensors are transferred using that device instead of the nonexistent `model.xlm_roberta.device`.  

These minimal changes unblock execution while keeping the original model and post‑processing logic intact, enabling a proper `submission.csv` file.'
- What this solution (achieved 0.00673) has done: 'Implemented a monkey‑patch for protobuf’s `MessageFactory` before any `transformers` imports to fix the `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This resolves the model loading issue, allowing the original XLM‑RoBerta QA pipeline to run end‑to‑end and produce a valid `submission.csv` with the expected columns. No other logic was altered, preserving the core architecture and inference flow.'
- What this solution (achieved 0.02494) has done: 'I replace the model‑based predictions with a lightweight heuristic that scans short spans in the context and selects the one with the highest Jaccard overlap against the question tokens. This keeps the overall pipeline and model definitions intact while providing far more meaningful answer strings, moving the Jaccard score significantly toward the target. The cleaning step and CSV writing remain unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf.message_factory import MessageFactory

    if not hasattr(MessageFactory, "GetPrototype"):

        def _get_prototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        MessageFactory.GetPrototype = _get_prototype
except Exception:
    pass

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import collections
from transformers import AutoConfig, AutoTokenizer, AutoModel
from string import punctuation

APEX_INSTALLED = False




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
    """
    Load configuration, tokenizer and model.
    Falls back to the public XLM‑Roberta base model if the provided path is invalid.
    """
    try:
        config = AutoConfig.from_pretrained(args.config_name)
    except Exception as e:
        print(
            f"Warning: could not load config from {args.config_name} ({e}), using default."
        )
        config = AutoConfig.from_pretrained("xlm-roberta-base")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    except Exception as e:
        print(
            f"Warning: could not load tokenizer from {args.tokenizer_name} ({e}), using default."
        )
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

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
            0 if x is None else x for x in tokenized_example.sequence_ids(i)
        ]
        features.append(feature)
    return features




## === cell 6
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

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                0
            ]
        else:
            best_answer = {"text": "", "score": 0.0}

        predictions[example["id"]] = best_answer["text"]

    return predictions




## === cell 7
possible_paths = [
    "../input/chaii-hindi-and-tamil-question-answering/test.csv",
    "../data/chaii-hindi-and-tamil-question-answering/test.csv",
    "data/chaii-hindi-and-tamil-question-answering/test.csv",
    "test.csv",
]
test_path = next((p for p in possible_paths if os.path.exists(p)), None)
if test_path is None:
    raise FileNotFoundError("Test CSV not found in expected locations.")

test = pd.read_csv(test_path)

test["context"] = test["context"].apply(lambda x: " ".join(str(x).split()))
test["question"] = test["question"].apply(lambda x: " ".join(str(x).split()))

args = Config()
config, tokenizer, model = make_model(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

test_features = []
for _, row in test.iterrows():
    test_features += prepare_test_features(args, row, tokenizer)

test_dataset = DatasetRetriever(test_features, mode="test")
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.eval_batch_size,
    sampler=SequentialSampler(test_dataset),
    num_workers=0,
    pin_memory=True,
    drop_last=False,
)

all_start_logits = []
all_end_logits = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        start_logits, end_logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        start_logits = start_logits.cpu().numpy()
        end_logits = end_logits.cpu().numpy()

        all_start_logits.extend(start_logits)
        all_end_logits.extend(end_logits)

raw_predictions = (all_start_logits, all_end_logits)


def heuristic_predict(row, max_span_len=8):
    """Return the context span (up to max_span_len words) that has the highest Jaccard
    similarity with the question."""
    context_words = row["context"].split()
    question_set = set(row["question"].lower().split())
    best_score = -1.0
    best_span = ""
    for start in range(len(context_words)):
        for length in range(1, max_span_len + 1):
            end = start + length
            if end > len(context_words):
                break
            span_words = context_words[start:end]
            span_set = set(w.lower() for w in span_words)
            intersect = len(span_set & question_set)
            union = len(span_set | question_set)
            if union == 0:
                continue
            score = intersect / union
            if score > best_score:
                best_score = score
                best_span = " ".join(span_words)
    return best_span


heuristic_preds = {}
for _, row in test.iterrows():
    heuristic_preds[row["id"]] = heuristic_predict(row)

pred_dict = heuristic_preds




## === cell 8
bad_starts = [".", ",", "(", ")", "-", "–", ";"]
bad_endings = ["...", "-", "(", ")", "–", ",", ";"]

tamil_ad = "கி.பி"
tamil_bc = "கி.மு"
tamil_km = "கி.மீ"
hindi_ad = "ई"
hindi_bc = "ई.पू"

cleaned_preds = []
for pid, pred in pred_dict.items():
    cleaned = " ".join(pred.split())
    cleaned = cleaned.strip(punctuation)

    while any(cleaned.startswith(y) for y in bad_starts):
        cleaned = cleaned[1:]

    while any(cleaned.endswith(y) for y in bad_endings):
        if cleaned.endswith("..."):
            cleaned = cleaned[:-3]
        else:
            cleaned = cleaned[:-1]
    if cleaned.endswith("..."):
        cleaned = cleaned[:-3]

    context = test.loc[test["id"] == pid, "context"].values[0]
    if (
        any(
            [
                cleaned.endswith(tamil_ad),
                cleaned.endswith(tamil_bc),
                cleaned.endswith(tamil_km),
                cleaned.endswith(hindi_ad),
                cleaned.endswith(hindi_bc),
            ]
        )
        and cleaned + "." in context
    ):
        cleaned = cleaned + "."

    cleaned_preds.append((pid, cleaned))

submission_df = pd.DataFrame(cleaned_preds, columns=["id", "PredictionString"])
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")




## === cell 9
submission_df.head()
