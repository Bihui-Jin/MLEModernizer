# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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
transformers==4.53.3

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

# 5. Code solution

## === cell 0
from time import time
import os
import sys
from pathlib import Path

import torch

torch.set_grad_enabled(False)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TRANSFORMERS_NO_PROTOBUF", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import transformers  # re-import here so the env vars are in effect for this cell

llama_path = Path("/kaggle/input/llama-3/transformers/8b-chat-hf/1")
gpt2_path = Path("/kaggle/input/gpt2")

if llama_path.exists():
    model = str(llama_path)
    _local_files_only = True
elif gpt2_path.exists():
    model = str(gpt2_path)
    _local_files_only = True
else:
    model = "gpt2"
    _local_files_only = False  # last-resort: prevents immediate LocalEntryNotFoundError

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model,
    local_files_only=_local_files_only,
    trust_remote_code=True,
    use_fast=True,
)
model_obj = transformers.AutoModelForCausalLM.from_pretrained(
    model,
    local_files_only=_local_files_only,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

model_obj.eval()

pipeline = transformers.pipeline(
    "text-generation",
    model=model_obj,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)


## === cell 1
import re

system_message = """
You are an AI assistant designed to score student essay.
The score is a value from 1 to 6.
You must answer only the score.
"""

_HAS_CHAT_TEMPLATE = bool(getattr(pipeline.tokenizer, "chat_template", None))
_TERMINATORS = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

_SCORE_RE = re.compile(r"Score:\s*([1-6])")


def _make_prompt(system_message: str, user_message: str) -> str:
    user_message = "Essay: " + user_message + " Score:"
    if _HAS_CHAT_TEMPLATE:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        prompt = pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = f"{system_message.strip()}\n\n{user_message}"
    return prompt


@torch.inference_mode()
def query_model_batch_from_prompts(
    prompts, temperature=0.7, max_length=5, batch_size=32
):
    out = []
    n = len(prompts)

    gen_kwargs = dict(
        do_sample=True,
        top_p=0.9,
        temperature=temperature,
        eos_token_id=_TERMINATORS,
        max_new_tokens=max_length,
        pad_token_id=pipeline.model.config.eos_token_id,
        use_cache=True,
    )

    tok = pipeline.tokenizer
    mdl = pipeline.model
    device = mdl.device

    for i in range(0, n, batch_size):
        batch_prompts = prompts[i : i + batch_size]

        enc = tok(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
        )

        if device.type == "cuda":
            enc = {
                k: v.pin_memory().to(device, non_blocking=True) for k, v in enc.items()
            }
        else:
            enc = {k: v.to(device) for k, v in enc.items()}

        gen = mdl.generate(**enc, **gen_kwargs)

        in_len = enc["input_ids"].shape[1]
        gen_new = gen[:, in_len:]
        texts = tok.batch_decode(gen_new, skip_special_tokens=True)
        out.extend(texts)

    return out


def query_model_batch(
    system_message, user_messages, temperature=0.7, max_length=5, batch_size=32
):
    prompts = [_make_prompt(system_message, um) for um in user_messages]
    return query_model_batch_from_prompts(
        prompts, temperature=temperature, max_length=max_length, batch_size=batch_size
    )


def _parse_score_from_response(response: str) -> int:
    m = _SCORE_RE.search(response)
    if m:
        return int(m.group(1))
    return 3




## === cell 2
import pandas as pd



## === cell 3
test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)



## === cell 4
if getattr(pipeline.tokenizer, "pad_token_id", None) is None:
    pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
    pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id

texts = test["full_text"].tolist()

uniq_texts = list(dict.fromkeys(texts))  # stable unique (preserves first-seen order)
uniq_prompts = [_make_prompt(system_message, t) for t in uniq_texts]

responses_uniq = query_model_batch_from_prompts(
    uniq_prompts, max_length=5, batch_size=32
)

resp_map = dict(zip(uniq_texts, responses_uniq))
responses = [resp_map[t] for t in texts]
prediction = [_parse_score_from_response(r) for r in responses]



## === cell 5
submission = test[["essay_id"]].copy()
submission["score"] = prediction



## === cell 6
submission



## === cell 7
submission.to_csv("submission.csv", index=False)
