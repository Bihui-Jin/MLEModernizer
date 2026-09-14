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

1.284226737812286

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, desc):
            return self.GetMessageClass(desc)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from tqdm import tqdm

print("torch:", torch.__version__)
print("pandas:", pd.__version__)
print("numpy:", np.__version__)




## === cell 1
TEST_PATH = "/kaggle/input/lmsys-chatbot-arena/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/lmsys-chatbot-arena/sample_submission.csv"

_CANDIDATE_MODEL_DIRS = [
    "/kaggle/input/qwen2.5/transformers/3b-instruct/1",  # user-provided (may already be correct if it's a directory)
    "/kaggle/input/qwen2.5/transformers/3b-instruct",  # fallback
]
MODEL_ID = None
for p in _CANDIDATE_MODEL_DIRS:
    if os.path.isdir(p) and (
        os.path.isfile(os.path.join(p, "config.json"))
        or os.path.isdir(os.path.join(p, "snapshots"))
    ):
        snap_root = os.path.join(p, "snapshots")
        if os.path.isdir(snap_root):
            snaps = sorted(
                [
                    os.path.join(snap_root, d)
                    for d in os.listdir(snap_root)
                    if os.path.isdir(os.path.join(snap_root, d))
                ]
            )
            if len(snaps) > 0:
                MODEL_ID = snaps[-1]
                break
        if os.path.isfile(os.path.join(p, "config.json")):
            MODEL_ID = p
            break

if MODEL_ID is None:
    raise FileNotFoundError(
        "Could not locate a usable local model directory under /kaggle/input/qwen2.5. "
        "Expected config.json or a snapshots/ folder."
    )

ADAPTER_ID = "/kaggle/input/qwen-lora/qwen25_arena_rm"

print("Resolved MODEL_ID:", MODEL_ID)
print("Adapter:", ADAPTER_ID)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, trust_remote_code=True, local_files_only=True
)

base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=1,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
)

if not hasattr(base_model, "prepare_inputs_for_generation"):
    base_model.prepare_inputs_for_generation = lambda *args, **kwargs: None

model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
model.eval()

_MODEL_DEVICE = next(model.parameters()).device
print(
    f"Model loaded. Using device: {_MODEL_DEVICE}, dtype: {next(model.parameters()).dtype}"
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2197078306.py in <cell line: 0>()
     34 
     35 if MODEL_ID is None:
---> 36     raise FileNotFoundError(
     37         "Could not locate a usable local model directory under /kaggle/input/qwen2.5. "
     38         "Expected config.json or a snapshots/ folder."

FileNotFoundError: Could not locate a usable local model directory under /kaggle/input/qwen2.5. Expected config.json or a snapshots/ folder.

## === cell 2
def calculate_probs(s_a, s_b, epsilon=0.6, temperature=0.7):
    """
    Same core logic as provided: map score difference to win/tie probabilities.
    Adds only numerical stability (finite + normalization) without changing intent.
    """
    diff = (s_a - s_b) / temperature

    if diff >= 0:
        z = np.exp(-diff)
        p_a_raw = 1.0 / (1.0 + z)
    else:
        z = np.exp(diff)
        p_a_raw = z / (1.0 + z)

    p_b_raw = 1.0 - p_a_raw
    p_tie = np.exp(-np.abs(diff) / epsilon) * 0.25  # cap at 0.25 as in original code

    total = p_a_raw * (1 - p_tie) + p_b_raw * (1 - p_tie) + p_tie
    winner_a = (p_a_raw * (1 - p_tie)) / total
    winner_b = (p_b_raw * (1 - p_tie)) / total
    winner_tie = p_tie / total

    probs = np.array([winner_a, winner_b, winner_tie], dtype=np.float64)
    probs = np.nan_to_num(probs, nan=1.0 / 3.0, posinf=1.0 / 3.0, neginf=1.0 / 3.0)
    probs = np.clip(probs, 1e-9, 1.0)
    probs = probs / probs.sum()

    return float(probs[0]), float(probs[1]), float(probs[2])


def get_score(prompt, response):
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(_MODEL_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
        score = out.logits.view(-1)[0].float().item()
    return score




## === cell 3
test_df = pd.read_csv(TEST_PATH)

results = []
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    s_a = get_score(row["prompt"], row["response_a"])
    s_b = get_score(row["prompt"], row["response_b"])

    prob_a, prob_b, prob_tie = calculate_probs(s_a, s_b)

    results.append(
        {
            "id": row["id"],
            "winner_model_a": prob_a,
            "winner_model_b": prob_b,
            "winner_tie": prob_tie,
        }
    )

pred_df = pd.DataFrame(results)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sub = sample_sub[["id"]].merge(pred_df, on="id", how="left")

for c in ["winner_model_a", "winner_model_b", "winner_tie"]:
    if c not in sub.columns:
        sub[c] = 1.0 / 3.0
sub[["winner_model_a", "winner_model_b", "winner_tie"]] = sub[
    ["winner_model_a", "winner_model_b", "winner_tie"]
].fillna(1.0 / 3.0)

p = sub[["winner_model_a", "winner_model_b", "winner_tie"]].to_numpy(dtype=np.float64)
p = np.clip(p, 1e-9, 1.0)
p = p / p.sum(axis=1, keepdims=True)
sub[["winner_model_a", "winner_model_b", "winner_tie"]] = p

sub.to_csv("submission.csv", index=False)
print("submission.csv written:", sub.shape)
print(sub.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3645292201.py in <cell line: 0>()
      3 results = []
      4 for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
----> 5     s_a = get_score(row["prompt"], row["response_a"])
      6     s_b = get_score(row["prompt"], row["response_b"])
      7 

/tmp/ipykernel_55/485978547.py in get_score(prompt, response)
     31 def get_score(prompt, response):
     32     text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
---> 33     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
     34     inputs = {k: v.to(_MODEL_DEVICE) for k, v in inputs.items()}
     35 

NameError: name 'tokenizer' is not defined
