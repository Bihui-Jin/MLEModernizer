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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
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

0.6623544245383161

# 6. Current score

0.46202

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.46202) has done: 'The fix removes the unnecessary package reinstallations and kernel restart, loads the data directly, uses a lightweight Sentence‑Transformer to embed the essays, trains a simple Ridge regression on the training embeddings, predicts scores for the test set, rounds them to the 1‑6 range, and writes a correctly formatted `submission.csv` file.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sentence_transformers import SentenceTransformer



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df["full_text"] = train_df["full_text"].str.replace("\n", " ", regex=False)
test_df["full_text"] = test_df["full_text"].str.replace("\n", " ", regex=False)



## === cell 2
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding training essays...")
train_embeddings = model.encode(
    train_df["full_text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
)

print("Encoding test essays...")
test_embeddings = model.encode(
    test_df["full_text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
)



## === cell 3
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(train_embeddings, train_df["score"].astype(float))

test_pred_cont = ridge.predict(test_embeddings)



## === cell 4
test_pred_clipped = np.clip(test_pred_cont, 1, 6)
test_pred_rounded = np.rint(test_pred_clipped).astype(int)

test_pred_final = np.clip(test_pred_rounded, 1, 6)



## === cell 5
submission = pd.DataFrame({"essay_id": test_df["essay_id"], "score": test_pred_final})

output_path = os.path.join("/kaggle/working", "submission.csv")
submission.to_csv(output_path, index=False)

print(f"Submission file written to {output_path}")
