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
Predict which chatbot response a user will prefer in a competition between two chatbots.

## Metric
Log loss with "eps=auto"

## Submission Format
For each id in the test set, you must predict the probability for each target class. The file should contain a header and have the following format:

```
 id,winner_model_a,winner_model_b,winner_tie
 136060,0.33,0,33,0.33
 211333,0.33,0,33,0.33
 1233961,0.33,0,33,0.33
 etc
```

## Dataset
**train.csv**

- `id` - A unique identifier for the row.
- `model_[a/b]` - The identity of model_[a/b]. Included in train.csv but not test.csv.
- `prompt` - The prompt that was given as an input (to both models).
- `response_[a/b]` - The response from model_[a/b] to the given prompt.
- `winner_model_[a/b/tie]` - Binary columns marking the judge's selection. The ground truth target column.

**test.csv**

- `id`
- `prompt`
- `response_[a/b]`

**sample_submission.csv** A submission file in the correct format.

- `id`
- `winner_model_[a/b/tie]` - This is what is predicted from the test set.

# 2. Python version

3.14

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
peft==0.16.0
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
tqdm==4.67.1
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        input/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        working/
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
```

-> data/lmsys-chatbot-arena/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/lmsys-chatbot-arena/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/lmsys-chatbot-arena/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> data/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> (stopped after 10 files for performance)

# 5. Target score

1.033805603403759

# 6. Current score

2.43066

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 2.43066) has done: 'I fix the two runtime blockers while keeping the same inference-only approach and submission format. First, I avoid the `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation before importing `transformers` (this is a known incompatibility in some Kaggle images). Second, I prevent the DistilBERT max-position error by dynamically capping `max_length` to the loaded model/tokenizer’s supported context length (512 for DistilBERT), while leaving the rest of the batching, temperature scaling, and probability post-processing unchanged. These changes should make the notebook run end-to-end and reliably write a valid `submission.csv`.'
- What this solution (achieved 2.43066) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing a compatible protobuf runtime selection *before* importing `transformers`, and by explicitly importing `google.protobuf` early so the env var takes effect. I also make the model loading robust to “no internet” by setting `local_files_only=True` when a local model directory is found, while keeping the same fallback behavior otherwise. Finally, I keep your inference logic intact but add a safe fallback to generate a valid submission even if the model cannot be loaded (so you always get a `.csv`), which is score-neutral versus crashing.'
- What this solution (achieved 2.43066) has done: 'I fix the protobuf crash that prevents the model from loading by pinning a compatible protobuf runtime early (including disabling C++ implementation) and only importing `transformers` after that takes effect. Then I keep your exact inference approach (same combined text, batching, temperature scaling, and post-processing) but ensure the code always loads the dataset from the provided paths and always writes a valid `submission.csv`. This should both unblock end-to-end execution and improve the score versus the current uniform-fallback behavior (which is why you’re seeing very high log loss).'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import google.protobuf  # noqa: F401
except Exception:
    pass

import numpy as np
import pandas as pd
import torch

torch.set_grad_enabled(False)
torch.manual_seed(0)
np.random.seed(0)

print("Base imports OK. Torch:", torch.__version__)




## === cell 1
def _find_local_model_dir():
    candidates = [
        "/kaggle/input/inference-fine-tuned-mistral-1/transformers/default/1",
        "/kaggle/input/inference-fine-tuned-mistral-1",
        "/kaggle/input",
    ]
    for c in candidates:
        if os.path.isdir(c):
            if os.path.isfile(os.path.join(c, "config.json")):
                return c
            try:
                for root, dirs, files in os.walk(c):
                    if "config.json" in files:
                        return root
                    if root.count(os.sep) - c.count(os.sep) >= 4:
                        dirs[:] = []
            except Exception:
                pass
    return None


MODEL_PATH = _find_local_model_dir()
FALLBACK_MODEL = (
    "distilbert-base-uncased-finetuned-sst-2-english"  # offline-friendly baseline
)

model_source = MODEL_PATH if MODEL_PATH is not None else FALLBACK_MODEL
print("Model source:", model_source)

use_cuda = torch.cuda.is_available()
dtype = torch.float16 if use_cuda else torch.float32
device_map = "auto" if use_cuda else None
local_only = MODEL_PATH is not None

quant_model = None
tokenizer = None
MAX_LEN = 512

from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, use_fast=True, local_files_only=local_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
            if tokenizer.eos_token is not None
            else tokenizer.unk_token
        )
    tokenizer.padding_side = "right"

    quant_model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        torch_dtype=dtype,
        device_map=device_map,
        local_files_only=local_only,
    )
    quant_model.eval()

    try:
        num_labels = int(getattr(quant_model.config, "num_labels", 0))
    except Exception:
        num_labels = 0

    print(
        "Loaded model. num_labels:",
        num_labels,
        "device:",
        quant_model.device if hasattr(quant_model, "device") else "unknown",
    )

    model_max_pos = getattr(
        getattr(quant_model, "config", None), "max_position_embeddings", None
    )
    tok_max = getattr(tokenizer, "model_max_length", None)

    candidates = []
    for v in [model_max_pos, tok_max]:
        if isinstance(v, int) and v > 0 and v < 10**6:
            candidates.append(v)
    MAX_LEN = min(candidates) if candidates else 512
    MAX_LEN = int(min(MAX_LEN, 1536))

    print(
        "Using MAX_LEN:",
        MAX_LEN,
        "| model_max_pos:",
        model_max_pos,
        "| tok_max:",
        tok_max,
    )
except Exception as e:
    print("WARNING: model/tokenizer failed to load; will output uniform predictions.")
    print("Load error:", repr(e))
    quant_model = None
    tokenizer = None
    MAX_LEN = 512



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
from tqdm import tqdm

test_path = "/kaggle/input/lmsys-chatbot-arena/test.csv"
if not os.path.exists(test_path):
    test_path = "/kaggle/input/test.csv"
if not os.path.exists(test_path):
    test_path = "/kaggle/data/lmsys-chatbot-arena/test.csv"
if not os.path.exists(test_path):
    test_path = "/kaggle/data/test.csv"

test_df = pd.read_csv(test_path)

test_df["combined_text"] = (
    test_df["prompt"].fillna("").astype(str)
    + "\n\nResponse A:\n"
    + test_df["response_a"].fillna("").astype(str)
    + "\n\nResponse B:\n"
    + test_df["response_b"].fillna("").astype(str)
)

print("Loaded test:", test_df.shape, "from", test_path)



## === cell 3
temperature = 1.5
batch_size = 16

preds = []

if quant_model is None or tokenizer is None:
    preds = np.full((len(test_df), 3), 1.0 / 3.0, dtype=np.float64)
else:
    for i in tqdm(range(0, len(test_df), batch_size), desc="Infer"):
        batch = test_df.iloc[i : i + batch_size]["combined_text"].tolist()
        inputs = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        model_device = (
            quant_model.device
            if hasattr(quant_model, "device")
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            out = quant_model(**inputs)
            logits = out.logits

            if logits.shape[-1] == 3:
                adj_logits = logits / temperature
                probs = torch.softmax(adj_logits, dim=-1)
            elif logits.shape[-1] == 2:
                adj_logits = logits / temperature
                p2 = torch.softmax(adj_logits, dim=-1)  # [p0, p1]
                probs = torch.stack([p2[:, 0], 0.5 * p2[:, 1], 0.5 * p2[:, 1]], dim=-1)
            else:
                probs = torch.full(
                    (logits.shape[0], 3),
                    1.0 / 3.0,
                    device=logits.device,
                    dtype=torch.float32,
                )

            preds.extend(probs.detach().float().cpu().numpy().tolist())

    preds = np.asarray(preds, dtype=np.float64)

row_sums = preds.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
preds = preds / row_sums
preds = np.clip(preds, 1e-15, 1 - 1e-15)
preds = preds / preds.sum(axis=1, keepdims=True)

submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "winner_model_a": preds[:, 0],
        "winner_model_b": preds[:, 1],
        "winner_tie": preds[:, 2],
    }
)

submission = submission[["id", "winner_model_a", "winner_model_b", "winner_tie"]]
submission.to_csv("submission.csv", index=False)

print("submission.csv ready:", submission.shape)
print(submission.head())
