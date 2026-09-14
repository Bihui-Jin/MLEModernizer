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

0.7924246284831522

# 6. Current score

0.71076

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'We redirect the output directory to a writable location (`/kaggle/working`) and simplify the model loading step so that any failure to load the heavy Deberta model cleanly falls back to the lightweight `SimpleTokenizer` and `DummyModel`. This eliminates the protobuf import error and the permission error when writing the submission file, allowing the script to run end‑to‑end and produce a valid `submission.csv` while preserving the original logic for prediction and aggregation.'
- What this solution (achieved 0.00915) has done: 'I remove the problematic transformers import to avoid the protobuf error and add a simple length‑based baseline using the training data to replace the constant dummy predictions. This keeps the overall pipeline unchanged while producing a valid submission.csv and nudges the quadratic weighted kappa score toward the target.'
- What this solution (achieved 0.01515) has done: 'I add a cheap length‑based regression that complements the existing bin‑average mapping. By fitting a low‑degree polynomial on essay length vs. score from the training set and blending its predictions with the current length‑bin averages, we can raise the quadratic weighted kappa toward the target without altering the core model logic.'
- What this solution (achieved 0.68183) has done: 'I fixed the runtime errors by adding a lightweight `SimpleTokenizer` implementation (so the dummy model can work) and by switching the Ridge regression solver to one that handles sparse data without triggering the SciPy `cg` bug. I also tweaked the final blending weights slightly (favoring the Ridge model) to modestly improve the quadratic weighted kappa score while keeping the original logic intact. The script now runs end‑to‑end and writes a proper `submission.csv`.'
- What this solution (achieved 0.68183) has done: 'I add a quick validation step that evaluates a few sensible blend ratios between the Ridge TF‑IDF model and the length‑based baseline on the held‑out validation set. The ratio that yields the higher QWK be stored in global variables and then used for the final test‑set blending, replacing the fixed 0.6/0.4 weights. This small change keeps the core pipeline untouched while nudging the overall score toward the target.'
- What this solution (achieved 0.6955) has done: 'I add a small grid‑search on the validation set to pick the ridge‑baseline blend weight that gives the highest quadratic weighted kappa (instead of only checking 0.6/0.4 and 0.7/0.3). This minor change keeps the core pipeline untouched while likely improving the validation QWK and therefore moving the final score closer to the target.'
- What this solution (achieved 0.70371) has done: 'I slightly improve the TF‑IDF ridge model and search the blend weight with a finer grid.  
- Increase `max_features` to 30 000 so the ridge model captures more useful n‑grams.  
- Reduce the Ridge regularisation (`alpha=0.5`) which often boosts predictive power on this dataset.  
- Refine the blend‑weight search from 0.5–0.9 step 0.05 to 0.4–1.0 step 0.02, keeping the weight that gives the highest validation QWK.  
These minimal tweaks keep the overall pipeline unchanged while nudging the validation QWK closer to the target.'
- What this solution (achieved 0.69855) has done: 'I slightly strengthen the TF‑IDF representation (more n‑grams and a larger vocabulary) and lower the Ridge regularisation strength. These tweaks keep the same model pipeline but give it a richer feature set, which should improve the validation QWK and move the final score nearer the target. The blend‑weight search remains unchanged and automatically pick the best ridge‑vs‑baseline mix after the updates.'
- What this solution (achieved 0.69511) has done: 'I slightly expand the TF‑IDF representation (increase max_features to 80 000) and reduce the Ridge regularisation strength (alpha = 0.1). These minimal tweaks give the ridge model richer features and a bit more flexibility, which should raise the validation QWK and therefore move the final Kaggle score closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.71076) has done: 'I increased the TF‑IDF representation by enabling sub‑linear term frequency scaling and expanding the vocabulary, and I made the Ridge regression a little less regularized (alpha = 0.05). I also refined the blend‑weight search resolution to 0.01, which lets the validation split pick a more optimal mixture of the ridge model and the length‑based baseline. These modest tweaks keep the original pipeline intact while moving the validation QWK—and consequently the competition score—closer to the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", category=UserWarning)

DATA_ROOT = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
OUTPUT_PATH = "/kaggle/working"
MODEL_PATH = os.path.join(DATA_ROOT, "model")  # kept for compatibility; may not exist
NUM_LABELS = 6
MAX_LEN = 512
OVERLAP = 64
MIN_CHUNK_RATIO = 0.3
BATCH_SIZE = 32

os.makedirs(OUTPUT_PATH, exist_ok=True)

df_train = pd.read_csv(TRAIN_PATH)
train_lengths = df_train["full_text"].str.len()
bin_edges = np.arange(0, train_lengths.max() + 101, 100)
train_bins = pd.cut(train_lengths, bins=bin_edges, include_lowest=True)
avg_score_per_bin = df_train.groupby(train_bins)["score"].mean()
length_bin_edges = bin_edges

poly_degree = 2
clipped_lengths = np.clip(train_lengths, a_min=0, a_max=train_lengths.max())
poly_coeffs = np.polyfit(clipped_lengths, df_train["score"], deg=poly_degree)

vectorizer = TfidfVectorizer(
    max_features=100000,  # larger vocabulary for richer representation
    ngram_range=(1, 3),
    min_df=1,
    stop_words="english",
    sublinear_tf=True,  # sub‑linear TF scaling often boosts linear models
)

X_full = df_train["full_text"].astype(str)
y_full = df_train["score"]

X_vec_full = vectorizer.fit_transform(X_full)

X_train, X_val, y_train, y_val = train_test_split(
    X_vec_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

ridge_model = Ridge(alpha=0.05, random_state=42, solver="lsqr")
ridge_model.fit(X_train, y_train)

val_pred = np.rint(ridge_model.predict(X_val)).astype(int)
val_pred = np.clip(val_pred, 1, NUM_LABELS)
val_qwk = cohen_kappa_score(y_val, val_pred, weights="quadratic")
print(f"Validation QWK (ridge model): {val_qwk:.5f}")

val_lengths = df_train.loc[y_val.index, "full_text"].str.len()
val_bins = pd.cut(val_lengths, bins=length_bin_edges, include_lowest=True)
mapped_scores_val = val_bins.map(avg_score_per_bin).fillna(3)

poly_pred_val = np.rint(np.polyval(poly_coeffs, val_lengths.values)).astype(int)

baseline_blend_val = np.rint(0.5 * mapped_scores_val + 0.5 * poly_pred_val).astype(int)
baseline_blend_val = np.clip(baseline_blend_val, 1, NUM_LABELS)

ridge_val_pred = val_pred

best_w = 0.6
best_qwk = -1.0
for w in np.arange(0.30, 1.001, 0.01):  # test weights 0.30 … 1.00 by 0.01
    blend = np.rint(w * ridge_val_pred + (1 - w) * baseline_blend_val).astype(int)
    blend = np.clip(blend, 1, NUM_LABELS)
    qwk = cohen_kappa_score(y_val, blend, weights="quadratic")
    if qwk > best_qwk:
        best_qwk = qwk
        best_w = w
RIDGE_WEIGHT = best_w
BASELINE_WEIGHT = 1 - best_w
print(
    f"Using blend weight RIDGE_WEIGHT={RIDGE_WEIGHT:.2f}, BASELINE_WEIGHT={BASELINE_WEIGHT:.2f} with QWK={best_qwk:.5f}"
)


class SimpleTokenizer:
    """A minimal whitespace tokenizer compatible with the expected API."""

    def __init__(self):
        self.pad_token_id = 0
        self._vocab = {}
        self._next_id = 1

    def _get_id(self, token):
        if token not in self._vocab:
            self._vocab[token] = self._next_id
            self._next_id += 1
        return self._vocab[token]

    def __call__(self, text, add_special_tokens=False):
        tokens = text.split()
        ids = [self._get_id(tok) for tok in tokens]
        return {"input_ids": ids}

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def build_inputs_with_special_tokens(self, token_ids):
        return [101] + token_ids + [102]


try:
    raise ImportError("Skip heavy transformer import in this environment.")
except Exception:
    tokenizer = SimpleTokenizer()

    class DummyModel:
        def __init__(self, num_labels, avg_score_map, bin_edges):
            self.num_labels = num_labels
            self.avg_score_map = avg_score_map
            self.bin_edges = bin_edges
            self.pad_token_id = tokenizer.pad_token_id

        def to(self, device):
            return self

        def eval(self):
            pass

        def __call__(self, input_ids, attention_mask):
            lengths = (input_ids != self.pad_token_id).sum(dim=1).cpu().numpy()
            preds = np.digitize(lengths, self.bin_edges, right=False) - 1
            preds = np.clip(preds, 0, self.num_labels - 1).astype(int)

            batch_size = input_ids.shape[0]
            logits = torch.full((batch_size, self.num_labels), -10.0)
            logits[range(batch_size), preds] = 10.0
            return type("Output", (), {"logits": logits})

    model = DummyModel(NUM_LABELS, avg_score_per_bin, length_bin_edges)




## === cell 1
def plot_confusion_matrix(preds, labels, title="Confusion Matrix"):
    preds = np.array(preds)
    labels = np.array(labels)

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greens", values_format="d")
    plt.title(title)
    plt.show()




## === cell 2
def compute_metrics(preds, labels, device="cuda"):
    if len(preds.shape) > 1:
        preds = np.argmax(preds, axis=-1)
    qwk_score = cohen_kappa_score(labels, preds, weights="quadratic")
    acc_score = accuracy_score(labels, preds)
    return {"quadratic_weighted_kappa": qwk_score, "accuracy": acc_score}




## === cell 3
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




## === cell 4
class TestEssayDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        max_len: int = 512,
        overlap: int = 64,
        min_chunk_ratio: float = 0.3,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.overlap = overlap
        self.min_chunk_ratio = min_chunk_ratio

        self.num_special = tokenizer.num_special_tokens_to_add(pair=False)

        tag_tokens = tokenizer("[A]", add_special_tokens=False)["input_ids"]
        len_tag = len(tag_tokens)

        self.max_content_len = max_len - self.num_special - len_tag

        for _, row in df.iterrows():
            essay_id = row["essay_id"]
            base_tokens = tokenizer(row["full_text"], add_special_tokens=False)[
                "input_ids"
            ]

            step = self.max_content_len - overlap
            start = 0

            while start < len(base_tokens):
                end = start + self.max_content_len
                chunk = base_tokens[start:end]

                if (
                    len(chunk) < self.max_content_len * self.min_chunk_ratio
                    and start != 0
                ):
                    break

                chunk_with_tag = tag_tokens + chunk

                processed = tokenizer.build_inputs_with_special_tokens(chunk_with_tag)

                if len(processed) > self.max_len:
                    processed = processed[: self.max_len]

                pad_len = self.max_len - len(processed)
                if pad_len > 0:
                    processed += [self.tokenizer.pad_token_id] * pad_len

                self.samples.append((processed, essay_id))

                start += step
                if end >= len(base_tokens):
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




## === cell 5
df_test = pd.read_csv(TEST_PATH)

test_dataset = TestEssayDataset(df_test, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO)

device = "cuda" if torch.cuda.is_available() else "cpu"

_, _ = predict_essay_score(
    dataset=test_dataset,
    model=model,
    num_labels=NUM_LABELS,
    batch_size=BATCH_SIZE,
    device=device,
)

X_test_vec = vectorizer.transform(df_test["full_text"].astype(str))
model_pred = np.rint(ridge_model.predict(X_test_vec)).astype(int)
model_pred = np.clip(model_pred, 1, NUM_LABELS)

test_lengths = df_test["full_text"].str.len()
test_bins = pd.cut(test_lengths, bins=length_bin_edges, include_lowest=True)
mapped_scores = test_bins.map(avg_score_per_bin).fillna(3)

poly_pred = np.rint(np.polyval(poly_coeffs, test_lengths.values)).astype(int)

baseline_blend = np.rint(0.5 * mapped_scores + 0.5 * poly_pred).astype(int)
baseline_blend = np.clip(baseline_blend, 1, NUM_LABELS)

final_score = np.rint(
    RIDGE_WEIGHT * model_pred + BASELINE_WEIGHT * baseline_blend
).astype(int)
final_score = np.clip(final_score, 1, NUM_LABELS)

test_results = pd.DataFrame({"essay_id": df_test["essay_id"], "score": final_score})

print("Submission preview:")
print(test_results.head())




## === cell 6
output_file = os.path.join(OUTPUT_PATH, "submission.csv")
test_results.to_csv(output_file, index=False)
print(f"Submission written to {output_file}")
