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

# 5. Target score

0.08701

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the root cause of the crash by loading the model/tokenizer from a valid local Kaggle path (or gracefully falling back to a lightweight baseline if the LLaMA folder isn’t present), which also fix the downstream `tokenizer not defined` errors. I keep your generation/parsing logic intact, but add a necessary runtime guard: if the LLM is available, only score a small deterministic subset and fill the rest with the same fallback you already use (3), so the notebook finishes within Kaggle’s time limits and always produces `submission.csv`. I also make the data path more robust by using the known competition directory and a fallback to the flat `/kaggle/input/...` copies. Finally, I ensure the submission has exactly the required columns and valid integer scores clipped to 1–6.'

# 9. Code solution

## === cell 0
from time import time
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CANDIDATE_MODEL_DIRS = [
    "/kaggle/input/llama-3/transformers/8b-chat-hf/1",
    "/kaggle/input/llama-3/transformers/Meta-Llama-3-8B-Instruct/1",
    "/kaggle/input/llama-3/transformers/Meta-Llama-3-8B/1",
    "/kaggle/input/llama-3/Meta-Llama-3-8B-Instruct",
    "/kaggle/input/llama-3",
]


def _find_model_dir(candidates):
    for p in candidates:
        if isinstance(p, str) and os.path.isdir(p):
            if os.path.exists(os.path.join(p, "config.json")):
                return p
    return None


model_path = _find_model_dir(CANDIDATE_MODEL_DIRS)

tokenizer = None
model = None
LLM_AVAILABLE = False

if model_path is not None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        LLM_AVAILABLE = True
        print("Loaded LLM from:", model_path)
    except Exception as e:
        print("Could not load LLM; falling back to baseline. Error:", repr(e))
        tokenizer = None
        model = None
        LLM_AVAILABLE = False
else:
    print("LLM model directory not found in /kaggle/input; falling back to baseline.")
    LLM_AVAILABLE = False




## === cell 1
def query_model(
    system_message,
    user_message,
    temperature=0.7,
    max_length=5,  # interpreted as max_new_tokens; keep behavior close to original usage
):
    if not LLM_AVAILABLE or tokenizer is None or model is None:
        return f"Essay: {'' if user_message is None else str(user_message)} Score: 3"

    start_time = time()

    user_prompt = (
        "Essay: " + ("" if user_message is None else str(user_message)) + " Score:"
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    terminators = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id != tokenizer.unk_token_id:
        terminators.append(eot_id)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            top_p=0.9,
            temperature=float(temperature),
            eos_token_id=terminators,
            max_new_tokens=int(max_length),
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True)

    end_time = time()
    ttime = f"Total time: {round(end_time-start_time, 2)} sec."

    return user_prompt + " " + answer + " " + ttime


system_message = """
You are an AI assistant designed to score student essay.
The score is a value from 1 to 6.
You must answer only the score.
""".strip()



## === cell 2
import pandas as pd



## === cell 3
TEST_CANDIDATES = [
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv",
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/learning-agency-lab-automated-essay-scoring-2/test.csv",
]
test_path = next((p for p in TEST_CANDIDATES if os.path.exists(p)), None)
if test_path is None:
    raise FileNotFoundError(f"Could not find test.csv. Tried: {TEST_CANDIDATES}")

test = pd.read_csv(test_path)
print("Loaded test:", test.shape, "from:", test_path)



## === cell 4
_score_re = re.compile(r"\b([1-6])\b")


def predict_score(text):
    response = query_model(system_message, user_message=text, max_length=5)

    idx = response.rfind("Score:")
    tail = response[idx:] if idx != -1 else response
    m = _score_re.search(tail)
    if not m:
        m = _score_re.search(response)

    if m:
        return int(m.group(1))
    return 3  # fallback consistent with original default




## === cell 5
N = len(test)
prediction = pd.Series([3] * N, index=test.index, dtype="int64")

if LLM_AVAILABLE:
    K = min(64, N)
    print(
        f"LLM available: scoring first {K} essays with LLM, rest default to 3 for runtime safety."
    )
    prediction.iloc[:K] = (
        test["full_text"].iloc[:K].apply(predict_score).astype("int64")
    )
else:
    print("LLM not available: using default score=3 for all essays.")



## === cell 6
submission = pd.DataFrame(
    {
        "essay_id": test["essay_id"].values,
        "score": prediction.astype(int).clip(1, 6).values,
    }
)



## === cell 7
assert list(submission.columns) == ["essay_id", "score"]
assert submission["score"].between(1, 6).all()
assert submission["essay_id"].notnull().all()

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
print(submission.head())
