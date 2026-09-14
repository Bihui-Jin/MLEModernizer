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
tensorflow-datasets==4.9.9
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

0.6669591221714222

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the import-time crash by avoiding the `datasets`/protobuf path that’s triggering the `MessageFactory.GetPrototype` error, and instead use a lightweight PyTorch `Dataset` for tokenized inputs. I also fix the `from_pretrained` path issue by loading a public DeBERTa checkpoint (available without extra Kaggle inputs) and run it in pure inference mode to produce a valid `submission.csv`. Because no score was yielded, the primary objective is to get an end-to-end working pipeline that writes the required columns and row alignment; the core “Transformer sequence classifier + argmax to discrete 1–6 scores” logic is preserved. Finally, I ensure predicted labels are mapped into the competition’s 1–6 score range and match the test `essay_id` order.'
- What this solution (achieved 0.0) has done: 'I fix the import-time crash causing the pipeline to stop before generating `submission.csv` by preventing the `datasets`/protobuf incompatibility from being triggered (it can be imported indirectly by `transformers.Trainer` in this environment). Then I keep your core inference logic (DeBERTa sequence classifier → logits → argmax → map to 1–6) but run prediction with a simple PyTorch `DataLoader` loop instead of `Trainer` to avoid that problematic dependency path. I also make sure the model is instantiated with a valid `num_labels=6` head so the score mapping is correct and stable, and that the output CSV has exactly `essay_id,score` aligned to the test row order. These changes are execution/stability fixes and should move the score up from 0.0 by producing a valid submission with nontrivial predictions.'
- What this solution (achieved 0.0) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by forcing the Python protobuf implementation before importing `transformers`, which avoids the C++-protobuf/TF/datasets incompatibility path that triggers this error in Kaggle environments. I also add a safe fallback to the pure-PyTorch (slow) tokenizer if the fast tokenizer import path still triggers protobuf issues. These changes keep your core logic identical (DeBERTa sequence classifier → logits → argmax → map to 1–6) and are score-improving only insofar as they allow producing a valid, non-empty `submission.csv` instead of scoring 0.0 due to a crash. Finally, I keep I/O paths and the submission schema exactly as required.'
- What this solution (achieved 0.0) has done: 'The timeout is dominated by per-batch tokenization in Python inside the DataLoader loop (re-tokenizing 15k essays on the fly) plus multi-worker overhead; we can preserve identical inference logic by tokenizing the entire test set once with the same tokenizer settings and then feeding pre-tokenized tensors via a fast TensorDataset-style DataLoader. This removes repeated tokenizer work and reduces Python overhead while keeping the same model, max_length, padding/truncation semantics, and argmax-to-score mapping. We also enable `torch.compile` when available (safe for inference correctness) and tune DataLoader options to minimize host/device stalls without changing outputs. All file paths and prediction semantics remain unchanged.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf-related crash (`MessageFactory.GetPrototype`) by avoiding the `transformers` import path that triggers the incompatible `datasets/protobuf` stack, and instead run the same “Transformer logits → argmax → map to 1–6” inference using `sentence-transformers`’ underlying HF model/tokenizer (which is already installed and typically avoids that crash path here). I keep your model choice and fallback logic, preserve max_length/padding/truncation semantics, and keep the pre-tokenize-once TensorDataset-style DataLoader to ensure it finishes within the time limit. Finally, I ensure the submission is written as `submission.csv` with exactly `essay_id,score` aligned to the test row order.'
- What this solution (achieved 0.0) has done: 'I fix the crash in cell 2 by avoiding the `sentence-transformers` import path that triggers the protobuf `MessageFactory.GetPrototype` error in this environment, while keeping the same core inference semantics (Transformer encoder → classification head → logits → argmax → map to 1–6). Concretely, I load the same DeBERTa checkpoint directly via `transformers` with TF/Flax disabled (already set in env), and use `AutoModelForSequenceClassification` so logits are produced consistently without needing to manually attach a classifier. I keep the pre-tokenize-once approach and the same mapping/clipping for submission, ensuring the script runs end-to-end and writes a valid `submission.csv`. These changes should move the score up from 0.0 by producing a non-empty, valid submission.'
- What this solution (achieved 0.0) has done: 'I fix the `MessageFactory.GetPrototype` crash by forcing a compatible protobuf runtime early and (crucially) avoiding importing the `transformers` stack in this environment, which is where the failure is triggered. To preserve your core inference logic (Transformer → logits → argmax → map to 1–6), I switch to the installed `sentence-transformers` backend to load the same Hugging Face DeBERTa encoder and attach a minimal classification head so we still produce 6 logits per essay. I keep the same tokenization settings (padding/truncation/max_length) and the same post-processing to discrete scores, and ensure the final `submission.csv` is written with `essay_id,score` aligned to test order. These changes are purely to unblock execution and produce a non-empty valid submission (raising score from 0.0 toward the target).'
- What this solution (achieved 0.0) has done: 'I fix the crash in the model-loading cell by avoiding `sentence-transformers`, which is triggering the protobuf `MessageFactory.GetPrototype` error in this Kaggle environment. To keep the same core semantics (Transformer encoder → 6-class logits → argmax → map to 1–6), I load the same DeBERTa checkpoint directly with `transformers` using a plain PyTorch inference loop. I also keep your “tokenize once, then DataLoader over tensors” approach so it finishes quickly and deterministically. Finally, I ensure the submission is written as `submission.csv` with exactly `essay_id,score` aligned to the test order.'
- What this solution (achieved 0.0) has done: 'I fix the `MessageFactory.GetPrototype` crash by preventing the incompatible protobuf C++ runtime from being used and by avoiding importing `transformers` in a way that triggers the failing `google.protobuf` path. Concretely, I force the pure-Python protobuf implementation *before any protobuf-related imports* and also set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` via both `os.environ` and `google.protobuf.internal.api_implementation`, then retry loading the same DeBERTa model/tokenizer. I keep your exact inference semantics (HF sequence classification logits → argmax → map to 1–6) and the same pre-tokenize-once + DataLoader loop so runtime stays under the limit. Finally, I ensure `submission.csv` is always written with `essay_id,score` aligned to the test order so the score is no longer 0.0 due to a crash.'
- What this solution (achieved 0.0) has done: 'You’re crashing during `transformers` import/model load due to a protobuf runtime mismatch (`MessageFactory.GetPrototype`), so I make the fix deterministic by pinning protobuf to the pure-Python implementation **before any protobuf/transformers imports** and by avoiding the internal `_SetImplementationType` call that can backfire. To keep your core inference logic identical (HF sequence classifier → logits → argmax → map to 1–6), I keep the same DeBERTa model choices and the same DataLoader-based prediction loop, only making the import path more robust and adding a safe second fallback model that’s widely cached on Kaggle. This should move your score up from 0.0 by reliably producing a non-empty, valid `submission.csv` instead of crashing. No training, feature, or post-processing logic is changed beyond stability.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass



## === cell 1
test_data = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
test_data.head()



## === cell 2
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PRIMARY_MODEL_NAME = "microsoft/deberta-v3-xsmall"
FALLBACK_MODEL_NAME = "microsoft/deberta-v3-base"  # broadly available
SECOND_FALLBACK_MODEL_NAME = "microsoft/deberta-v3-small"  # extra robustness

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(name, num_labels=6)
    return tok, mdl


_last_err = None
for _name in (PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME, SECOND_FALLBACK_MODEL_NAME):
    try:
        tokenizer, model = load_model_and_tokenizer(_name)
        _last_err = None
        break
    except Exception as e:
        _last_err = e

if _last_err is not None:
    raise _last_err

model = model.to(device)
model.eval()

if torch.cuda.is_available():
    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception:
        pass

if hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="max-autotune", fullgraph=False)
    except Exception:
        pass



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
model_max_len = getattr(tokenizer, "model_max_length", 512)
if model_max_len is None or model_max_len > 1024:
    model_max_len = 512

max_length = min(512, int(model_max_len))

len(test_data), max_length



## === cell 4
texts = test_data["full_text"].astype(str).tolist()
test_enc = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=max_length,
    return_tensors="pt",
)


class EncodedDataset(TorchDataset):
    def __init__(self, encodings):
        self.encodings = encodings
        self.keys = tuple(encodings.keys())

    def __len__(self):
        return self.encodings[self.keys[0]].shape[0]

    def __getitem__(self, idx):
        return {k: self.encodings[k][idx] for k in self.keys}


def collate_encoded(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


encoded_test_dataset = EncodedDataset(test_enc)

batch_size = 192 if torch.cuda.is_available() else 48
num_workers = 0

test_loader = DataLoader(
    encoded_test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
    collate_fn=collate_encoded,
)

n_test = len(encoded_test_dataset)
num_labels = 6
logits = np.empty((n_test, num_labels), dtype=np.float32)

model.eval()
row = 0
with torch.inference_mode():
    for batch in test_loader:
        batch = {
            k: v.to(device, non_blocking=torch.cuda.is_available())
            for k, v in batch.items()
            if k in ("input_ids", "attention_mask")
        }

        outputs = model(**batch)
        out = outputs.logits  # [bs, 6]
        out = out.detach().to("cpu").numpy()
        bsz = out.shape[0]

        if out.shape[-1] != 6:
            if out.shape[-1] > 6:
                out = out[:, :6]
            else:
                pad = np.zeros((bsz, 6 - out.shape[-1]), dtype=out.dtype)
                out = np.concatenate([out, pad], axis=1)

        logits[row : row + bsz] = out
        row += bsz

logits = logits[:row]
logits.shape



## === cell 5
pred_class = np.argmax(logits, axis=-1)

num_labels = logits.shape[-1]
if num_labels == 6:
    predicted_scores = (pred_class + 1).astype(np.int32)
else:
    predicted_scores = np.rint(1 + (pred_class / max(1, num_labels - 1)) * 5).astype(
        np.int32
    )

predicted_scores = np.clip(predicted_scores, 1, 6)

predicted_scores[:10], predicted_scores.min(), predicted_scores.max(), num_labels



## === cell 6
submission = pd.DataFrame(
    {
        "essay_id": test_data["essay_id"].values,
        "score": predicted_scores,
    }
)

submission = submission[["essay_id", "score"]]
submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 7
submission
