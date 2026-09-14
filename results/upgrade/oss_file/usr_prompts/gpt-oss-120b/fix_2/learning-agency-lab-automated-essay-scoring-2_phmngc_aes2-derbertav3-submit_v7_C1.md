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

3.14

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8017657349561782

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import warnings

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore", category=UserWarning)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = DebertaV2ForSequenceClassification.from_pretrained(
    MODEL_PATH, num_labels=NUM_LABELS, local_files_only=True
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1447719294.py in <cell line: 0>()
      1 # Load tokenizer and model from the provided local path (no remote download)
----> 2 tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
      3 model = DebertaV2ForSequenceClassification.from_pretrained(
      4     MODEL_PATH, num_labels=NUM_LABELS, local_files_only=True
      5 )

NameError: name 'MODEL_PATH' is not defined

## === cell 2
def predict_essay_score(model, dataset, num_labels, batch_size, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()

    preds, essay_ids_all = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            essay_ids = batch["essay_id"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            pred_labels = torch.argmax(logits, dim=1).cpu().numpy() + 1

            preds.extend(pred_labels)
            essay_ids_all.extend(essay_ids)

    preds = np.array(preds)
    essay_ids_all = np.array(essay_ids_all)

    chunk_results = pd.DataFrame({"essay_id": essay_ids_all, "raw_prediction": preds})

    aggregated_predictions = chunk_results.groupby("essay_id")["raw_prediction"].mean()
    y_pred_aggregated = np.rint(aggregated_predictions.values)
    final_scores = np.clip(y_pred_aggregated, 1, num_labels).astype(int)

    final_results_df = pd.DataFrame(
        {"essay_id": aggregated_predictions.index.values, "score": final_scores}
    )

    return final_results_df, preds




## === cell 3
df_test = pd.read_csv(TEST_PATH)
test_dataset = TestEssayDataset(df_test, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4133861034.py in <cell line: 0>()
----> 1 df_test = pd.read_csv(TEST_PATH)
      2 test_dataset = TestEssayDataset(df_test, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO)
      3 

NameError: name 'TEST_PATH' is not defined

## === cell 4
test_results, _ = predict_essay_score(
    dataset=test_dataset,
    model=model,
    num_labels=NUM_LABELS,
    batch_size=BATCH_SIZE,
    device="cuda",
)

print("Submission test:")
print(test_results.head())



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/307034990.py in <cell line: 0>()
      1 test_results, _ = predict_essay_score(
----> 2     dataset=test_dataset,
      3     model=model,
      4     num_labels=NUM_LABELS,
      5     batch_size=BATCH_SIZE,

NameError: name 'test_dataset' is not defined

## === cell 5
submission_path = os.path.join(OUTPUT_PATH, "submission.csv")
test_results.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1154655980.py in <cell line: 0>()
      1 # Write submission to working directory with the correct filename
----> 2 submission_path = os.path.join(OUTPUT_PATH, "submission.csv")
      3 test_results.to_csv(submission_path, index=False)
      4 print(f"Submission saved to {submission_path}")

NameError: name 'OUTPUT_PATH' is not defined
