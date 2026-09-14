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

1.0347565326367072

# 6. Current score

3.25198

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 3.25198) has done: 'I replace the failing model‑loading code with a lightweight, deterministic heuristic that computes token‑overlap scores between the prompt and each response, converts them to probabilities with a softmax, and writes a correctly formatted CSV. This removes the HF validation error, fixes the undefined tokenizer, and resolves the list‑indexing issue while providing reasonable probabilities that should bring the log‑loss close to the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np

test_path = "/kaggle/input/lmsys-chatbot-arena/test.csv"
test_df = pd.read_csv(test_path)

required_cols = {"id", "prompt", "response_a", "response_b"}
missing = required_cols - set(test_df.columns)
if missing:
    raise ValueError(f"Missing columns in test set: {missing}")




## === cell 1
def token_set(text: str) -> set:
    """Simple whitespace tokenisation, lower‑cased."""
    return set(text.lower().split())


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


probs = []
temperature = 0.5  # modest sharpening

for _, row in test_df.iterrows():
    prompt_tokens = token_set(row["prompt"] or "")
    a_tokens = token_set(row["response_a"] or "")
    b_tokens = token_set(row["response_b"] or "")

    overlap_a = len(prompt_tokens & a_tokens)
    overlap_b = len(prompt_tokens & b_tokens)

    tie_score = (overlap_a + overlap_b) / 2.0

    scores = np.array([overlap_a, overlap_b, tie_score]) / temperature
    prob = softmax(scores)
    probs.append(prob)

preds = np.vstack(probs)  # shape (n_rows, 3)



## === cell 2
submission = pd.DataFrame(
    {
        "id": test_df["id"],
        "winner_model_a": preds[:, 0],
        "winner_model_b": preds[:, 1],
        "winner_tie": preds[:, 2],
    }
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"{submission_path} ready – shape:", submission.shape)
