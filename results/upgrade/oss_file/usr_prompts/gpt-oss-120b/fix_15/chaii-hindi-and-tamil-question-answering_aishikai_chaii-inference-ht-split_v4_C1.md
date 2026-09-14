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
import re
import string
import collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForQuestionAnswering,
)

from sentence_transformers import SentenceTransformer, util

APEX_INSTALLED = False


def optimal_num_of_loader_workers():
    return 0




## === cell 1
class Configration:
    model_type = "bert"
    XLMR_name_or_path = "bert-base-multilingual-cased"
    XLMR_tokenizer_name = "bert-base-multilingual-cased"

    MURIL_name_or_path = "google/muril-base-cased"
    MPNET2_name_or_path = "sentence-transformers/all-mpnet-base-v2"
    MPNET3_name_or_path = "sentence-transformers/all-mpnet-base-v2"
    XLMR_config_name = "bert-base-multilingual-cased"
    MURIL_config_name = "google/muril-base-cased"
    MPNET2_config_name = "sentence-transformers/all-mpnet-base-v2"
    MPNET3_config_name = "sentence-transformers/all-mpnet-base-v2"
    fp16 = True if APEX_INSTALLED else False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2

    max_seq_length = 400
    doc_stride = 135

    epochs = 1
    train_batch_size = 4
    eval_batch_size = 256  # larger batch for faster inference

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
def Make_Model(args):
    """
    Load the tokenizer and a ready‑to‑use QA model.
    Preference is given to the XLMR multilingual model defined in the config.
    If loading fails (e.g., protobuf incompatibility), fall back to a safe multilingual QA model.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.XLMR_tokenizer_name)
        model = AutoModelForQuestionAnswering.from_pretrained(args.XLMR_name_or_path)
    except Exception as e:
        print(
            f"Primary model load failed ({e}), falling back to a safe multilingual QA model."
        )
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        model = AutoModelForQuestionAnswering.from_pretrained(
            "bert-base-multilingual-cased"
        )
    return tokenizer, model




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
            0 if s is None else s for s in tokenized_example.sequence_ids(i)
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
            features[feature_index]["offset_mapping"] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(features[feature_index]["offset_mapping"])
            ]
            offset_mapping = features[feature_index]["offset_mapping"]
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
test_path = os.path.join(
    "..", "input", "chaii-hindi-and-tamil-question-answering", "test.csv"
)
test_df = pd.read_csv(test_path)




## === cell 8
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def improved_heuristic_predictions(df, max_span_len=30):
    """
    For each example, embed the question once and then evaluate every
    contiguous span (up to `max_span_len` tokens) in the context.
    The span with the highest cosine similarity to the question is chosen.
    """
    preds = {}
    for _, row in df.iterrows():
        question = row["question"]
        ctx_tokens = row["context"].split()
        n = len(ctx_tokens)

        q_emb = embedder.encode(question, convert_to_tensor=True)

        best_score = -1.0
        best_span = ""

        for start in range(n):
            for end in range(start, min(start + max_span_len, n)):
                span_text = " ".join(ctx_tokens[start : end + 1])
                span_emb = embedder.encode(span_text, convert_to_tensor=True)
                sim = util.cos_sim(q_emb, span_emb).item()
                if sim > best_score:
                    best_score = sim
                    best_span = span_text

        if not best_span:
            best_span = ctx_tokens[0] if ctx_tokens else ""
        preds[row["id"]] = best_span.strip()
    return preds


pred_dict = improved_heuristic_predictions(test_df)




## === cell 9
bad_starts = [".", ",", "(", ")", "-", "–", ";"]
bad_endings = ["...", "-", "(", ")", "–", ",", ";"]
tamil_ad = "கி.பி"
tamil_bc = "கி.மு"
tamil_km = "கி.மீ"
hindi_ad = "ई"
hindi_bc = "ஈ.पू"

cleaned_preds = []
for row_id, pred in pred_dict.items():
    pred_clean = " ".join(pred.split())
    pred_clean = pred_clean.strip(string.punctuation)
    while any(pred_clean.startswith(b) for b in bad_starts):
        pred_clean = pred_clean[1:].lstrip()
    while any(pred_clean.endswith(b) for b in bad_endings):
        if pred_clean.endswith("..."):
            pred_clean = pred_clean[:-3]
        else:
            pred_clean = pred_clean[:-1]
    if (
        any(
            pred_clean.endswith(suf)
            for suf in [tamil_ad, tamil_bc, tamil_km, hindi_ad, hindi_bc]
        )
        and pred_clean + "."
        in test_df.loc[test_df["id"] == row_id, "context"].values[0]
    ):
        pred_clean = pred_clean + "."
    cleaned_preds.append((row_id, pred_clean))

heuristic_dict = dict(cleaned_preds)




## === cell 10
final_predictions = {}
for idx in test_df["id"]:
    final_predictions[idx] = heuristic_dict.get(idx, "")

submission = pd.DataFrame(
    [(idx, final_predictions[idx]) for idx in test_df["id"]],
    columns=["id", "PredictionString"],
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
print(submission.head())
