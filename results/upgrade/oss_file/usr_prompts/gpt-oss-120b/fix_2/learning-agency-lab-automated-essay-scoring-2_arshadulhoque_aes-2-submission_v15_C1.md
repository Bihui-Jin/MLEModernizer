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

0.741923748724992

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'The current notebook crashes while importing the HuggingFace model because the provided path is not a valid repo and the protobuf version used by Torch/Transformers causes an import error. To unblock the pipeline we wrap the heavy model loading in a try/except and fall back to a very simple baseline: predict the overall mean score from the training set (rounded to the nearest integer). All file paths are corrected to point to the competition’s data folder, and the script now always creates a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import pandas as pd
import torch
from pathlib import Path

try:
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    print(f"Transformers import failed ({e}), will use fallback model.")
    TRANSFORMERS_AVAILABLE = False



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = Path("/kaggle/input/learning-agency-lab-automated-essay-scoring-2")

train_path = DATA_ROOT / "train.csv"
train_df = pd.read_csv(train_path)

baseline_score = int(round(train_df["score"].mean()))
print(f"Baseline constant score: {baseline_score}")



## === cell 2
test_path = DATA_ROOT / "test.csv"
test_df = pd.read_csv(test_path)

if TRANSFORMERS_AVAILABLE:
    try:
        model_dir = Path("/kaggle/input/trained-deberta-xsmall")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        test_encodings = tokenizer(
            test_df["full_text"].tolist(),
            truncation=True,
            padding=True,
            max_length=1536,
        )

        test_dataset = Dataset.from_dict(test_encodings)
        test_dataset = test_dataset.add_column("essay_id", test_df["essay_id"].tolist())

        predict_args = TrainingArguments(
            output_dir=".",
            per_device_eval_batch_size=4,
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=predict_args,
            tokenizer=tokenizer,
        )

        predictions = trainer.predict(test_dataset)
        predicted_labels = (
            torch.argmax(torch.tensor(predictions.predictions), dim=-1).cpu().numpy()
            + 1
        )  # assuming model outputs 0‑5 logits

        print("Model inference completed.")
    except Exception as e:
        print(f"Model loading/prediction failed ({e}), falling back to baseline.")
        predicted_labels = None
else:
    predicted_labels = None



## === cell 3
if predicted_labels is None:
    predicted_labels = [baseline_score] * len(test_df)

submission = pd.DataFrame(
    {
        "essay_id": test_df["essay_id"],
        "score": pd.Series(predicted_labels, dtype="int32"),
    }
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")



## === cell 4
submission.head()
