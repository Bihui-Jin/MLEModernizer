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

0.233718277572957

# 6. Current score

0.32738

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.03368) has done: 'I fix the root cause of the initial failure by switching the model loading to a local Kaggle input path that actually exists (and fall back safely if it doesn’t), which also resolve the downstream `tokenizer`/`prediction` `NameError`s. Because running an 8B LLM over 15k essays won’t finish within Kaggle’s 600s constraint and currently prevents any valid CSV from being produced, I keep your generation/parsing logic intact but add a minimal, deterministic fallback scorer that produces valid 1–6 predictions when the LLM can’t be loaded or used. Finally, I make submission creation always use the required `essay_id,score` columns and write `submission.csv` reliably.'
- What this solution (achieved 0.06967) has done: 'Your current score is low because you’re always using the length-based heuristic (USE_LLM defaults off), which has weak correlation with true rubric scores under QWK. To move toward the target with minimal core-logic change, I keep your same inference pipeline and fallback structure, but replace the heuristic with a slightly richer deterministic rubric proxy using only inexpensive text features (length + sentence count + word sophistication + punctuation/paragraph structure). This stays within the “fallback scorer” role you already added, runs fast under 600s for 15k essays, and should materially improve QWK versus pure length binning. I also keep the exact submission schema and clipping to [1,6].'
- What this solution (achieved 0.32738) has done: 'To move your QWK score closer to the target with minimal changes, I keep your exact fallback/LLM structure but make the fallback scorer train-calibrated using the provided `train.csv` labels. Specifically, I fit a tiny ordinal mapping from your existing heuristic score (1–6) to the best label distribution on training data (a monotonic “confusion-matrix” remap), which usually boosts QWK a lot without changing model architecture or adding heavy training. I also keep clipping to [1,6] and ensure submission alignment and schema remain identical. LLM behavior remains unchanged; only the deterministic fallback path is strengthened.'

# 9. Code solution

## === cell 0
from time import time
import os
import re
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print("Transformers:", transformers.__version__)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())




## === cell 1
def _find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


candidate_model_paths = [
    "/kaggle/input/llama-3/transformers/8b-chat-hf/1",
    "/kaggle/input/llama-3/transformers/8b-chat-hf",
    "/kaggle/input/llama-3-8b-instruct/transformers/llama-3-8b-instruct-hf/1",
    "/kaggle/input/llama-3-8b-instruct/transformers/llama-3-8b-instruct-hf",
    "/kaggle/input/meta-llama/Meta-Llama-3-8B-Instruct",
    "/kaggle/input/llama-3/Meta-Llama-3-8B-Instruct",
]

model_path = _find_first_existing(candidate_model_paths)
print("Resolved model_path:", model_path)

tokenizer = None
model = None
eos_ids = None
LLM_AVAILABLE = False

if model_path is not None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, local_files_only=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
        )
        model.eval()

        eos_ids = [tokenizer.eos_token_id]
        try:
            eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if (
                isinstance(eot_id, int)
                and tokenizer.unk_token_id is not None
                and eot_id != tokenizer.unk_token_id
            ):
                eos_ids.append(eot_id)
        except Exception:
            pass

        LLM_AVAILABLE = True
        print("Loaded model/tokenizer. eos_ids:", eos_ids)
    except Exception as e:
        print(
            "WARNING: Could not load LLM; falling back to deterministic heuristic scorer.\nError:",
            repr(e),
        )
        tokenizer, model, eos_ids = None, None, None
        LLM_AVAILABLE = False
else:
    print(
        "WARNING: No local LLM path found; falling back to deterministic heuristic scorer."
    )
    LLM_AVAILABLE = False




## === cell 2
def query_model(system_message, user_message, temperature=0.7, max_length=128):
    """
    Generates a short answer that should contain only the score.

    Note: If LLM isn't available, raises RuntimeError so caller can fall back.
    """
    if not LLM_AVAILABLE:
        raise RuntimeError("LLM not available")

    start_time = time()

    user_message = (
        "Essay: " + (user_message if isinstance(user_message, str) else "") + "\nScore:"
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen = out[0, inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(gen, skip_special_tokens=True)

    end_time = time()
    ttime = f"Total time: {round(end_time - start_time, 2)} sec."
    return user_message + " " + answer + " " + ttime


system_message = """
You are an AI assistant designed to score student essay.
The score is a value from 1 to 6. Meaning of each score:
6: Clear mastery with few errors, outstanding critical thinking, appropriate evidence, well-organized, skilled language use.
5: Reasonable mastery with occasional errors, strong critical thinking, generally appropriate evidence, well-organized, good language use.
4: Adequate mastery with some lapses, competent critical thinking, adequate evidence, generally organized, fair language use.
3: Developing mastery with weaknesses, limited critical thinking, inconsistent evidence, limited organization, fair language use with weaknesses.
2: Little mastery with serious flaws, weak critical thinking, insufficient evidence, poor organization, limited language use with frequent errors.
1: Very little or no mastery, severely flawed, no viable point of view, disorganized, fundamental language flaws, pervasive grammar/mechanics errors.

You must answer only the score.
""".strip()



## === cell 3
test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
print(test.shape)
print(test.columns)



## === cell 4
_score_re = re.compile(r"\b([1-6])\b")

_word_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_sentence_split_re = re.compile(r"[.!?]+")
_paragraph_split_re = re.compile(r"\n\s*\n+")
_punct_re = re.compile(r"[,;:]")
_digit_re = re.compile(r"\d")


def _clamp_int(x, lo=1, hi=6):
    try:
        x = int(round(float(x)))
    except Exception:
        x = 3
    return max(lo, min(hi, x))


def _heuristic_score(text: str) -> int:
    if not isinstance(text, str):
        return 3

    t = text.strip()
    if not t:
        return 1

    n_chars = len(t)
    words = _word_re.findall(t)
    n_words = len(words)

    n_sent = len([s for s in _sentence_split_re.split(t) if s.strip()])
    n_para = len([p for p in _paragraph_split_re.split(t) if p.strip()])

    avg_wlen = (sum(len(w) for w in words) / n_words) if n_words else 0.0
    long_word_ratio = (
        (sum(1 for w in words if len(w) >= 7) / n_words) if n_words else 0.0
    )
    uniq_ratio = (len(set(w.lower() for w in words)) / n_words) if n_words else 0.0

    punct_var = len(set(ch for ch in t if ch in ".,;:!?\"'()-"))
    n_commasemicol = len(_punct_re.findall(t))
    digit_penalty = 1.0 if _digit_re.search(t) else 0.0

    wps = (n_words / n_sent) if n_sent else n_words
    sents_per_para = (n_sent / n_para) if n_para else n_sent

    score = 1.0

    if n_words < 60:
        score = 1.0
    elif n_words < 120:
        score = 2.0
    elif n_words < 200:
        score = 3.0
    elif n_words < 300:
        score = 4.0
    elif n_words < 420:
        score = 5.0
    else:
        score = 6.0

    if n_para >= 2:
        score += 0.3
    if n_para >= 3:
        score += 0.2
    if n_sent >= 6:
        score += 0.2
    if n_sent >= 10:
        score += 0.2

    if n_sent <= 2 and n_words >= 80:
        score -= 0.4
    if wps >= 35:
        score -= 0.3
    if wps <= 6 and n_sent >= 8:
        score -= 0.2

    if avg_wlen >= 4.3:
        score += 0.2
    if avg_wlen >= 4.8:
        score += 0.2
    if long_word_ratio >= 0.12:
        score += 0.2
    if long_word_ratio >= 0.18:
        score += 0.2

    if n_words >= 120 and uniq_ratio >= 0.55:
        score += 0.2
    if n_words >= 200 and uniq_ratio >= 0.60:
        score += 0.2
    if n_words >= 120 and uniq_ratio <= 0.40:
        score -= 0.2

    if punct_var >= 6:
        score += 0.2
    if n_commasemicol >= 6:
        score += 0.1

    score -= 0.1 * digit_penalty

    if n_chars < 250:
        score = min(score, 2.0)
    if n_chars < 120:
        score = 1.0

    return _clamp_int(score, 1, 6)


def predict_score(text):
    """
    Parses LLM output to extract the first standalone digit 1-6; fallback to heuristic score.
    """
    try:
        response = query_model(system_message, user_message=text)
        if "Score:" in response:
            tail = response.split("Score:", 1)[1]
            m = _score_re.search(tail)
            if m:
                return int(m.group(1))
        m = _score_re.search(response)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return _heuristic_score(text)




## === cell 5
train = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
train["score"] = train["score"].astype(int).clip(1, 6)

h_train = train["full_text"].map(_heuristic_score).astype(int).clip(1, 6)

ct = pd.crosstab(h_train, train["score"]).reindex(
    index=range(1, 7), columns=range(1, 7), fill_value=0
)
raw_map = {}
for k in range(1, 7):
    row = ct.loc[k]
    raw_map[k] = int(row.idxmax()) if row.sum() > 0 else k

mono_map = {}
prev = 1
for k in range(1, 7):
    mapped = raw_map[k]
    if mapped < prev:
        mapped = prev
    prev = mapped
    mono_map[k] = int(mapped)

print("Heuristic->Label raw_map:", raw_map)
print("Heuristic->Label mono_map:", mono_map)


def _calibrated_heuristic_score(text: str) -> int:
    h = _heuristic_score(text)
    return int(mono_map.get(int(h), int(h)))




## === cell 6
USE_LLM = (os.environ.get("USE_LLM", "0") == "1") and LLM_AVAILABLE
print("USE_LLM:", USE_LLM, "| LLM_AVAILABLE:", LLM_AVAILABLE)

if USE_LLM:
    prediction = test["full_text"].apply(predict_score)
else:
    prediction = test["full_text"].map(_calibrated_heuristic_score)

print("Pred distribution:\n", prediction.value_counts(dropna=False).sort_index())



## === cell 7
submission = test[["essay_id"]].copy()
submission["score"] = pd.Series(prediction, index=test.index).astype(int).clip(1, 6)

assert list(submission.columns) == ["essay_id", "score"]
assert submission["score"].between(1, 6).all()

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.dtypes)
print(submission.head())
