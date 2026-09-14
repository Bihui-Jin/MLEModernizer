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

0.7859064152834561

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the import-time `MessageFactory.GetPrototype` crash by setting a compatible protobuf implementation before importing `transformers`/`sentencepiece`-dependent code. Then I make model/tokenizer loading robust by detecting whether `MODEL_PATH` exists locally (Kaggle dataset mount) and falling back to a safe public DeBERTa checkpoint if it doesn’t, using `local_files_only=True` when appropriate. I also fix the cell numbering and ensure `tokenizer`, `model`, and `test_dataset` are defined before use, add a safe CPU fallback if CUDA isn’t available, and always write a valid `submission.csv` with the exact required columns and row alignment to `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the import-time `MessageFactory.GetPrototype` crash by forcing the protobuf runtime to use the pure-Python implementation before any `transformers` import, and by importing `google.protobuf` early to ensure the env var takes effect. I also make the model/tokenizer class selection robust for both DeBERTa v2 and v3 checkpoints (some v3 checkpoints require `AutoTokenizer/AutoModelForSequenceClassification`), while keeping the same inference logic and submission formatting. Finally, I add a safe fallback to load from the Kaggle-mounted model directory if present, otherwise use a public checkpoint (only if available locally), and always write `/kaggle/working/submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
TRAIN_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
TEST_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
MODEL_PATH = "/kaggle/input/aes2-debertav3-large/transformers/default/1"
OUTPUT_PATH = "/kaggle/working/"

MAX_LEN = 512
OVERLAP = 64
MIN_CHUNK_RATIO = 0.3  # Skip if this chunk < RATIO * MAX_LEN
BATCH_SIZE = 2
NUM_LABELS = 6



## === cell 1
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



## === cell 2
import google.protobuf  # noqa: F401

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)




## === cell 3
def _find_local_hf_model_dir(search_root: str = "/kaggle/input") -> str | None:
    """
    Find a local Hugging Face Transformers model directory by looking for a folder
    containing a config.json. Prefer DeBERTa-like directories if multiple exist.
    """
    candidates = []
    for root, dirs, files in os.walk(search_root):
        if "config.json" in files:
            candidates.append(root)

    if not candidates:
        return None

    preferred_keywords = ["deberta", "aes", "essay", "transformers"]

    def score_path(p: str) -> tuple[int, int]:
        pl = p.lower()
        kw_hits = sum(k in pl for k in preferred_keywords)
        return (kw_hits, -len(p))

    candidates.sort(key=score_path, reverse=True)
    return candidates[0]


def _resolve_model_path(model_path: str) -> tuple[str | None, bool]:
    """
    Returns (resolved_local_path_or_repo_id_or_None, local_files_only).

    In Kaggle (no internet), we must only load locally.
    If given MODEL_PATH doesn't exist, try to locate any local model folder under /kaggle/input.
    If still not found, return (None, True) and we'll use an offline sklearn fallback.
    """
    if (
        model_path
        and os.path.isdir(model_path)
        and os.path.exists(os.path.join(model_path, "config.json"))
    ):
        return model_path, True

    guessed = _find_local_hf_model_dir("/kaggle/input")
    if guessed is not None:
        return guessed, True

    return None, True


resolved_model_path, use_local_files_only = _resolve_model_path(MODEL_PATH)
print(
    f"Resolved model path: {resolved_model_path} (local_files_only={use_local_files_only})"
)

tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

if resolved_model_path is not None:
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_path,
        local_files_only=use_local_files_only,
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        resolved_model_path,
        num_labels=NUM_LABELS,
        local_files_only=use_local_files_only,
    )
    model.to(device)
    print(f"Loaded transformer model from: {resolved_model_path}")
    print(f"Device: {device}")
else:
    print(
        "No local transformer model found under /kaggle/input; will use offline sklearn fallback."
    )




## === cell 4
def plot_confusion_matrix(preds, labels, title="Confusion Matrix"):
    preds = np.array(preds)
    labels = np.array(labels)

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greens", values_format="d")
    plt.title(title)
    plt.show()




## === cell 5
def compute_metrics(preds, labels, device="cuda"):
    if len(preds.shape) > 1:
        preds = np.argmax(preds, axis=-1)
    qwk_score = cohen_kappa_score(labels, preds, weights="quadratic")
    acc_score = accuracy_score(labels, preds)
    return {
        "quadratic_weighted_kappa": qwk_score,
        "accuracy": acc_score,
    }




## === cell 6
def predict_essay_score(model, dataset, num_labels, batch_size, device):
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
            pred_labels = torch.argmax(logits, axis=1).cpu().numpy() + 1

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




## === cell 7
class TestEssayDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        max_len: int = 512,
        overlap: int = 128,
        min_chunk_ratio: float = 0.3,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.overlap = overlap
        self.min_chunk_ratio = min_chunk_ratio

        num_special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        self.max_content_len = max_len - num_special_tokens

        for _, row in df.iterrows():
            essay_id = row["essay_id"]
            text = row["full_text"]
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]

            step = self.max_content_len - overlap
            start = 0

            while start < len(tokens):
                end = start + self.max_content_len
                chunk_content = tokens[start:end]
                if len(chunk_content) < self.max_content_len * self.min_chunk_ratio:
                    break

                processed_tokens = tokenizer.build_inputs_with_special_tokens(
                    chunk_content
                )
                padding_len = self.max_len - len(processed_tokens)
                if padding_len > 0:
                    processed_tokens += [tokenizer.pad_token_id] * padding_len

                self.samples.append((processed_tokens, essay_id))
                start += step

                if end >= len(tokens):
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, essay_id = self.samples[idx]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "essay_id": essay_id,
        }




## === cell 8
df_test = pd.read_csv(TEST_PATH)

test_results = None

if tokenizer is not None and model is not None:
    test_dataset = TestEssayDataset(
        df_test, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO
    )
    print(f"Test rows: {len(df_test)}, test chunks: {len(test_dataset)}")

    test_results, _ = predict_essay_score(
        dataset=test_dataset,
        model=model,
        num_labels=NUM_LABELS,
        batch_size=BATCH_SIZE,
        device=device,
    )

    print("Submission preview (transformer):")
    print(test_results.head())
else:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    df_train = pd.read_csv(TRAIN_PATH)

    pipe = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=200000,
                    strip_accents="unicode",
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            ("ridge", Ridge(alpha=1.0, random_state=42)),
        ]
    )

    X_train = df_train["full_text"].astype(str).fillna("")
    y_train = df_train["score"].astype(float).values

    X_test = df_test["full_text"].astype(str).fillna("")

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    y_pred_round = np.rint(y_pred)
    y_pred_round = np.clip(y_pred_round, 1, NUM_LABELS).astype(int)

    test_results = pd.DataFrame(
        {"essay_id": df_test["essay_id"].values, "score": y_pred_round}
    )

    print("Submission preview (sklearn fallback):")
    print(test_results.head())



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    114             try:
--> 115                 coef, info = sp_linalg.cg(C, y_column, tol=tol, atol="legacy")
    116             except TypeError:

TypeError: cg() got an unexpected keyword argument 'tol'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/303589029.py in <cell line: 0>()
     51     X_test = df_test["full_text"].astype(str).fillna("")
     52 
---> 53     pipe.fit(X_train, y_train)
     54     y_pred = pipe.predict(X_test)
     55 

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in fit(self, X, y, **fit_params)
    403             if self._final_estimator != "passthrough":
    404                 fit_params_last_step = fit_params_steps[self.steps[-1][0]]
--> 405                 self._final_estimator.fit(Xt, y, **fit_params_last_step)
    406 
    407         return self

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in fit(self, X, y, sample_weight)
   1132             y_numeric=True,
   1133         )
-> 1134         return super().fit(X, y, sample_weight=sample_weight)
   1135 
   1136 

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in fit(self, X, y, sample_weight)
    898                 params = {}
    899 
--> 900             self.coef_, self.n_iter_ = _ridge_regression(
    901                 X,
    902                 y,

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _ridge_regression(X, y, alpha, sample_weight, solver, max_iter, tol, verbose, positive, random_state, return_n_iter, return_intercept, X_scale, X_offset, check_input, fit_intercept)
    669     n_iter = None
    670     if solver == "sparse_cg":
--> 671         coef = _solve_sparse_cg(
    672             X,
    673             y,

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    116             except TypeError:
    117                 # old scipy
--> 118                 coef, info = sp_linalg.cg(C, y_column, tol=tol)
    119             coefs[i] = X1.rmatvec(coef)
    120         else:

TypeError: cg() got an unexpected keyword argument 'tol'

## === cell 9
sample_path = (
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
df_sample = pd.read_csv(sample_path)

sub = df_sample[["essay_id"]].merge(test_results, on="essay_id", how="left")

sub["score"] = sub["score"].fillna(3)
sub["score"] = np.clip(np.rint(sub["score"].astype(float)), 1, NUM_LABELS).astype(int)
sub = sub[["essay_id", "score"]]

out_file = os.path.join(OUTPUT_PATH, "submission.csv")
sub.to_csv(out_file, index=False)

print(f"Wrote submission: {out_file} | shape={sub.shape}")
print(sub.head())
print(sub["score"].value_counts().sort_index())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3623502336.py in <cell line: 0>()
      5 
      6 # Ensure ordering and completeness exactly match sample_submission.
----> 7 sub = df_sample[["essay_id"]].merge(test_results, on="essay_id", how="left")
      8 
      9 sub["score"] = sub["score"].fillna(3)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10830         from pandas.core.reshape.merge import merge
  10831 
> 10832         return merge(
  10833             self,
  10834             right,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    151 ) -> DataFrame:
    152     left_df = _validate_operand(left)
--> 153     right_df = _validate_operand(right)
    154     if how == "cross":
    155         return _cross_merge(

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _validate_operand(obj)
   2690         return obj.to_frame()
   2691     else:
-> 2692         raise TypeError(
   2693             f"Can only merge Series or DataFrame objects, a {type(obj)} was passed"
   2694         )

TypeError: Can only merge Series or DataFrame objects, a <class 'NoneType'> was passed
