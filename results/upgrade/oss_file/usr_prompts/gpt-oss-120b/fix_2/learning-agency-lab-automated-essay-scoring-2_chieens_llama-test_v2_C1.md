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

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I add a safe fallback that avoids the failing LLM pipeline by loading the training data, computing the overall mean score, and using this constant as the prediction for every essay. The pipeline creation is wrapped in a try/except so the script continues even if the model cannot be loaded. This fixes the NameError issues, ensures the “score” column is present, and produces a valid `submission.csv` that should achieve a baseline QWK close to the target.'

# 9. Code solution

## === cell 0
from time import time
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd



## === cell 1
model_path = "/kaggle/input/llama-3/transformers/8b-chat-hf/1"
try:
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
except Exception as e:
    print(f"Pipeline creation failed ({e}); falling back to constant predictions.")
    pipeline = None



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
train_df = pd.read_csv(train_path)
default_score = int(round(train_df["score"].mean()))
print(f"Default constant prediction will be: {default_score}")



## === cell 3
test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
test = pd.read_csv(test_path)



## === cell 4
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
"""


def query_model(system_message, user_message, temperature=0.7, max_length=8000):
    """
    Wrapper around the LLM pipeline. If the pipeline could not be created,
    simply return a constant answer containing the default_score.
    """
    if pipeline is None:
        return f"{system_message} Score: {default_score}"

    start_time = time()
    user_message = "Essay: " + user_message + " Score:"
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_p=0.9,
        temperature=temperature,
        eos_token_id=terminators,
        max_new_tokens=max_length,
        return_full_text=False,
        pad_token_id=pipeline.model.config.eos_token_id,
    )
    answer = sequences[0]["generated_text"]
    end_time = time()
    return (
        user_message
        + " "
        + answer
        + f" Total time: {round(end_time - start_time, 2)} sec."
    )


def predict_score(text):
    response = query_model(system_message, user_message=text)
    try:
        score_str = response.split("Score:")[1].strip().split()[0]
        score = int(score_str)
        if 1 <= score <= 6:
            return score
    except Exception:
        pass
    return default_score




## === cell 5
prediction = test["full_text"].apply(predict_score)



## === cell 6
submission = test[["essay_id"]].copy()
submission["score"] = prediction



## === cell 7
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv with", len(submission), "rows.")
