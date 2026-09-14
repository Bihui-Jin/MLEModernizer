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

1.3105775659942334

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()
logging.basicConfig(level=logging.ERROR)

TEST_PATH = "data/lmsys-chatbot-arena/test.csv"

model_id = "/kaggle/input/qwen2.5/transformers/3b/1"
adapter_id = "/kaggle/input/qwen2-5-3b-lora/qwen25_arena_rm/qwen25_arena_rm"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = None
try:
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=1,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    if not hasattr(base_model, "prepare_inputs_for_generation"):
        base_model.prepare_inputs_for_generation = lambda *args, **kwargs: None
    model = PeftModel.from_pretrained(base_model, adapter_id)
    model.eval()
except Exception as e:
    logging.error(f"Model loading failed, falling back to heuristic scoring: {e}")
    model = None


def score_with_model(prompt: str, response: str) -> float:
    """Compute a raw score using the loaded model (if available)."""
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits[0].item()


def fallback_score(prompt: str, response: str) -> float:
    """Simple deterministic fallback: negative absolute length difference."""
    return -abs(len(response) - len(prompt))


def get_score(prompt: str, response: str) -> float:
    """Unified interface – uses the model when possible, otherwise fallback."""
    if model is not None:
        try:
            return score_with_model(prompt, response)
        except Exception as e:
            logging.error(f"Runtime error during model inference: {e}")
    return fallback_score(prompt, response)


def calculate_probs(
    s_a: float, s_b: float, epsilon: float = 0.6, temperature: float = 0.7
):
    """
    Map raw score differences to a three‑class probability distribution.
    """
    diff = (s_a - s_b) / temperature
    p_a_raw = 1 / (1 + np.exp(-diff))
    p_b_raw = 1 - p_a_raw
    p_tie = np.exp(-np.abs(diff) / epsilon) * 0.25  # max tie prob ≈0.25

    total = p_a_raw * (1 - p_tie) + p_b_raw * (1 - p_tie) + p_tie
    winner_a = (p_a_raw * (1 - p_tie)) / total
    winner_b = (p_b_raw * (1 - p_tie)) / total
    winner_tie = p_tie / total
    return winner_a, winner_b, winner_tie


test_df = pd.read_csv(TEST_PATH)
results = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df), disable=False):
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

pd.DataFrame(results).to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created.")

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
