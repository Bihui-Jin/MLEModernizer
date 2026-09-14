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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

# 3. Installed packages

datasets==4.4.1
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
transformers==4.53.3
vega-datasets==0.9.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.7753155262158988

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd

import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
model_path = "roberta-base"
MAX_LEN = 512

test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
test_df = pd.read_csv(test_path)



## === cell 2
CHAR_CAP = 8000  # conservative cap to reduce tokenizer work; still far above what's needed for 512 BPE tokens
s = test_df["full_text"].astype("string")
test_df["full_text"] = s.str.slice(0, CHAR_CAP).astype(str)



## === cell 3
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

try:
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

texts = test_df["full_text"].tolist()
n = len(texts)
num_labels = int(model.config.num_labels)

batch_size = 512 if device.type == "cuda" else 64

preds = np.empty((n, num_labels), dtype=np.float32)

use_cuda = device.type == "cuda"

autocast_ctx = (
    torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    if use_cuda
    else torch.amp.autocast(device_type="cpu", enabled=False)
)


def _batch_iter(seq, bs):
    for i in range(0, len(seq), bs):
        yield i, seq[i : i + bs]


with torch.inference_mode():
    for start, chunk in _batch_iter(texts, batch_size):
        batch = tokenizer(
            chunk,
            truncation=True,
            truncation_strategy="longest_first",
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        )

        if use_cuda:
            batch = {k: v.pin_memory() for k, v in batch.items()}

        batch = {k: v.to(device, non_blocking=use_cuda) for k, v in batch.items()}

        with autocast_ctx:
            out = model(**batch)

        end = start + len(chunk)
        preds[start:end] = out.logits.detach().float().cpu().numpy()

scores = (np.argmax(preds, axis=1) + 1).astype(int)
scores = np.clip(scores, 1, 6)

sub = test_df[["essay_id"]].copy()
sub["score"] = scores

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)

print("Wrote:", sub_path)
print(sub.head())
print("Rows:", len(sub), "Columns:", list(sub.columns))

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/581056339.py in <cell line: 0>()
     43     for start, chunk in _batch_iter(texts, batch_size):
     44         # Speed fix: explicitly use tokenizer truncation strategy; padding remains dynamic.
---> 45         batch = tokenizer(
     46             chunk,
     47             truncation=True,

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

TypeError: transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast._batch_encode_plus() got multiple values for keyword argument 'truncation_strategy'
