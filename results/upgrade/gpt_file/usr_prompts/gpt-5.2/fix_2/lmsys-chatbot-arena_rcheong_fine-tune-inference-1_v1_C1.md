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

1.0501584368984174

# 6. Current score

4.66623

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 4.66623) has done: 'I remove the failing bitsandbytes/peft imports that trigger the protobuf `MessageFactory` error, since they are not used in the inference path. Then I fix model loading so it works with a local Kaggle dataset folder by using `local_files_only=True` and falling back to the best available base model (`microsoft/deberta-v3-base`) if the provided fine-tuned path is not present. Finally, I ensure `tokenizer`/`model` are always defined, run batched inference safely, and write a properly formatted `submission.csv` with the required columns and row order.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


torch.manual_seed(0)
np.random.seed(0)

print("Torch:", torch.__version__)



## === cell 1
TEST_PATH = "/kaggle/input/lmsys-chatbot-arena/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/lmsys-chatbot-arena/sample_submission.csv"

test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

test_df["combined_text"] = (
    test_df["prompt"].fillna("").astype(str)
    + "\n\nResponse A:\n"
    + test_df["response_a"].fillna("").astype(str)
    + "\n\nResponse B:\n"
    + test_df["response_b"].fillna("").astype(str)
)

assert "id" in test_df.columns
assert set(sample_sub.columns) == {
    "id",
    "winner_model_a",
    "winner_model_b",
    "winner_tie",
}
print("Loaded test:", test_df.shape, "sample_submission:", sample_sub.shape)



## === cell 2

PREFERRED_LOCAL_MODEL_PATH = (
    "/kaggle/input/inference-fine-tuned-mistral-1/transformers/default/1"
)
FALLBACK_MODEL_ID = "microsoft/deberta-v3-base"  # widely used for text classification


def load_model_and_tokenizer():
    if os.path.isdir(PREFERRED_LOCAL_MODEL_PATH):
        try:
            tok = AutoTokenizer.from_pretrained(
                PREFERRED_LOCAL_MODEL_PATH, use_fast=True, local_files_only=True
            )
            mdl = AutoModelForSequenceClassification.from_pretrained(
                PREFERRED_LOCAL_MODEL_PATH,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                device_map="auto",
                local_files_only=True,
            )
            return mdl, tok, PREFERRED_LOCAL_MODEL_PATH
        except Exception as e:
            print("Local model load failed, will fall back. Error:", repr(e))

    tok = AutoTokenizer.from_pretrained(FALLBACK_MODEL_ID, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        FALLBACK_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    return mdl, tok, FALLBACK_MODEL_ID


model, tokenizer, loaded_from = load_model_and_tokenizer()

if tokenizer.pad_token is None:
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    )
tokenizer.padding_side = "right"

model.eval()
print("Loaded model from:", loaded_from)
print("Num labels:", getattr(model.config, "num_labels", None))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
from tqdm import tqdm

preds = []
batch_size = 16

model_max_len = getattr(tokenizer, "model_max_length", 1024)
if model_max_len is None or model_max_len > 4096:
    model_max_len = 1024
max_length = min(1024, int(model_max_len))

device = model.device

with torch.no_grad():
    for i in tqdm(range(0, len(test_df), batch_size), desc="Infer"):
        batch = test_df.iloc[i : i + batch_size]["combined_text"].tolist()
        inputs = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)

        if probs.shape[-1] != 3:
            if probs.shape[-1] > 3:
                probs = probs[:, :3]
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            else:
                pad = torch.full(
                    (probs.shape[0], 3 - probs.shape[-1]),
                    1e-6,
                    device=probs.device,
                    dtype=probs.dtype,
                )
                probs = torch.cat([probs, pad], dim=-1)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        preds.extend(probs.detach().cpu().numpy())

preds = np.asarray(preds, dtype=np.float64)
print("Preds shape:", preds.shape)



## === cell 4
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "winner_model_a": preds[:, 0],
        "winner_model_b": preds[:, 1],
        "winner_tie": preds[:, 2],
    }
)

submission = sample_sub[["id"]].merge(submission, on="id", how="left")

for col in ["winner_model_a", "winner_model_b", "winner_tie"]:
    if submission[col].isna().any():
        submission[col] = submission[col].fillna(1.0 / 3.0)

row_sum = (
    submission[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1).values
)
row_sum = np.clip(row_sum, 1e-12, None)
submission[["winner_model_a", "winner_model_b", "winner_tie"]] = submission[
    ["winner_model_a", "winner_model_b", "winner_tie"]
].div(row_sum, axis=0)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print(out_path, "ready:", submission.shape)
print(submission.head())
