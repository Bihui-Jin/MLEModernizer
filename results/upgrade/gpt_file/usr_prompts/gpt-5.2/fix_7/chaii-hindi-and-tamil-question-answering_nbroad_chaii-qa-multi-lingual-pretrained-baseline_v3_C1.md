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

3.9

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
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

0.5439859628677368

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    del os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]

import numpy as np
import pandas as pd
import torch

CANDIDATE_BASE_PATHS = [
    "/kaggle/input/chaii-hindi-and-tamil-question-answering",
    "/kaggle/data/chaii-hindi-and-tamil-question-answering",
    "../input/chaii-hindi-and-tamil-question-answering",
    "/kaggle/input",
    "/kaggle/data",
]

BASE_PATH = None
for p in CANDIDATE_BASE_PATHS:
    if os.path.isfile(os.path.join(p, "test.csv")) and os.path.isfile(
        os.path.join(p, "train.csv")
    ):
        BASE_PATH = p
        break

if BASE_PATH is None:
    raise FileNotFoundError(
        "Could not find train.csv/test.csv in expected competition input directories. "
        f"Tried: {CANDIDATE_BASE_PATHS}"
    )

test_path = os.path.join(BASE_PATH, "test.csv")
sample_path = os.path.join(BASE_PATH, "sample_submission.csv")

test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

required_cols = {"id", "context", "question"}
missing = required_cols - set(test_df.columns)
if missing:
    raise ValueError(
        f"test.csv is missing required columns: {missing}. Columns: {list(test_df.columns)}"
    )

print("BASE_PATH:", BASE_PATH)
print("test_df shape:", test_df.shape)
print("sample_submission shape:", sample_sub.shape)



## === cell 1
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

CANDIDATE_MODEL_DIRS = [
    "/kaggle/input/pretrained-xlm-models-for-squad/deepset/xlm-roberta-large-squad2",
    "/kaggle/input/pretrained-xlm-models-for-squad/xlm-roberta-large-squad2",
    "../input/pretrained-xlm-models-for-squad/deepset/xlm-roberta-large-squad2",
    "../input/pretrained-xlm-models-for-squad/xlm-roberta-large-squad2",
    "/kaggle/input/pretrained-xlm-models-for-squad",
    "../input/pretrained-xlm-models-for-squad",
]

MODEL_PATH = None
for p in CANDIDATE_MODEL_DIRS:
    if os.path.isdir(p):
        if os.path.isfile(os.path.join(p, "config.json")):
            MODEL_PATH = p
            break
        try:
            for name in os.listdir(p):
                cand = os.path.join(p, name)
                if os.path.isdir(cand) and os.path.isfile(
                    os.path.join(cand, "config.json")
                ):
                    MODEL_PATH = cand
                    break
        except Exception:
            pass
    if MODEL_PATH is not None:
        break

FALLBACK_HF_MODEL = "deepset/xlm-roberta-large-squad2"
LOCAL_ONLY = MODEL_PATH is not None
MODEL_ID = MODEL_PATH if LOCAL_ONLY else FALLBACK_HF_MODEL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, use_fast=True, local_files_only=LOCAL_ONLY
)
model = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_ID, local_files_only=LOCAL_ONLY
)

model.to(device)
model.eval()
torch.set_grad_enabled(False)

print("MODEL_ID:", MODEL_ID)
print("device:", device)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2

predictions = []

max_length = 384
doc_stride = 128
n_best = 20
max_answer_length = 30

for ctx, q in test_df[["context", "question"]].to_numpy():
    enc = tokenizer(
        q,
        ctx,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt",
        padding=False,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    token_type_ids = enc.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    start_logits = outputs.start_logits.detach().cpu().numpy()
    end_logits = outputs.end_logits.detach().cpu().numpy()

    best_score = -1e18
    best_span = (0, 0)

    num_features = start_logits.shape[0]
    for feat_idx in range(num_features):
        seq_ids = enc.sequence_ids(feat_idx)
        ctx_token_mask = np.array([sid == 1 for sid in seq_ids], dtype=bool)

        s_logits = np.where(ctx_token_mask, start_logits[feat_idx], -1e9)
        e_logits = np.where(ctx_token_mask, end_logits[feat_idx], -1e9)

        start_indexes = np.argsort(s_logits)[-n_best:][::-1]
        end_indexes = np.argsort(e_logits)[-n_best:][::-1]

        offsets = enc["offset_mapping"][feat_idx].tolist()

        for s in start_indexes:
            s_off = offsets[s]
            if s_off is None:
                continue
            for e in end_indexes:
                if e < s:
                    continue
                if (e - s + 1) > max_answer_length:
                    continue
                e_off = offsets[e]
                if e_off is None:
                    continue

                start_char, _ = s_off
                _, end_char = e_off
                if start_char is None or end_char is None:
                    continue
                if start_char == 0 and end_char == 0:
                    continue

                score = float(s_logits[s] + e_logits[e])
                if score > best_score:
                    best_score = score
                    best_span = (int(start_char), int(end_char))

    if best_score <= -1e17:
        pred = ""
    else:
        pred = ctx[best_span[0] : best_span[1]]

    predictions.append(pred)

print("n_predictions:", len(predictions))
print("first_pred:", predictions[0] if predictions else None)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in convert_to_tensors(self, tensor_type, prepend_batch_axis)
    766                 if not is_tensor(value):
--> 767                     tensor = as_tensor(value)
    768 

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in as_tensor(value, dtype)
    728                     return torch.from_numpy(np.array(value))
--> 729                 return torch.tensor(value)
    730 

ValueError: expected sequence of length 384 at dim 1 (got 325)

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1505413441.py in <cell line: 0>()
     10 
     11 for ctx, q in test_df[["context", "question"]].to_numpy():
---> 12     enc = tokenizer(
     13         q,
     14         ctx,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in __call__(self, text, text_pair, text_target, text_pair_target, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, **kwargs)
   2853             if not self._in_target_context_manager:
   2854                 self._switch_to_input_mode()
-> 2855             encodings = self._call_one(text=text, text_pair=text_pair, **all_kwargs)
   2856         if text_target is not None:
   2857             self._switch_to_target_mode()

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in _call_one(self, text, text_pair, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens, **kwargs)
   2963             )
   2964         else:
-> 2965             return self.encode_plus(
   2966                 text=text,
   2967                 text_pair=text_pair,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in encode_plus(self, text, text_pair, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, **kwargs)
   3038         )
   3039 
-> 3040         return self._encode_plus(
   3041             text=text,
   3042             text_pair=text_pair,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_fast.py in _encode_plus(self, text, text_pair, add_special_tokens, padding_strategy, truncation_strategy, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens, **kwargs)
    625     ) -> BatchEncoding:
    626         batched_input = [(text, text_pair)] if text_pair else [text]
--> 627         batched_output = self._batch_encode_plus(
    628             batched_input,
    629             is_split_into_words=is_split_into_words,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_fast.py in _batch_encode_plus(self, batch_text_or_text_pairs, add_special_tokens, padding_strategy, truncation_strategy, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens)
    599         for input_ids in sanitized_tokens["input_ids"]:
    600             self._eventual_warn_about_too_long_sequence(input_ids, max_length, verbose)
--> 601         return BatchEncoding(sanitized_tokens, sanitized_encodings, tensor_type=return_tensors)
    602 
    603     def _encode_plus(

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in __init__(self, data, encoding, tensor_type, prepend_batch_axis, n_sequences)
    238         self._n_sequences = n_sequences
    239 
--> 240         self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)
    241 
    242     @property

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in convert_to_tensors(self, tensor_type, prepend_batch_axis)
    781                         "Please see if a fast version of this tokenizer is available to have this feature available."
    782                     ) from e
--> 783                 raise ValueError(
    784                     "Unable to create tensor, you should probably activate truncation and/or padding with"
    785                     " 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your"

ValueError: Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your features (`input_ids` in this case) have excessive nesting (inputs type `list` where type `int` is expected).

## === cell 3
assert len(predictions) == len(
    test_df
), f"Predictions ({len(predictions)}) != test rows ({len(test_df)})"

submission_df = pd.DataFrame(
    {"id": test_df["id"].astype(str), "PredictionString": predictions}
)

if set(sample_sub["id"].astype(str)) == set(submission_df["id"].astype(str)):
    pass

submission_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_df.shape)
submission_df.head()

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/3143324900.py in <cell line: 0>()
      1 # Create submission with required columns and correct row count.
----> 2 assert len(predictions) == len(
      3     test_df
      4 ), f"Predictions ({len(predictions)}) != test rows ({len(test_df)})"
      5 

AssertionError: Predictions (0) != test rows (112)
