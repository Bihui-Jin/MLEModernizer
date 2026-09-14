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
numpy==1.26.4
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

0.8053301059475746

# 6. Current score

-0.01867

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved -0.01867) has done: 'The import crash comes from `transformers` trying to import TensorFlow and protobuf C-extensions that aren’t compatible in this environment; we prevent TF/JAX imports and force pure-Python protobuf before importing `transformers`. `DataCollatorWithPadding` is also triggering the problematic `transformers.data` import path, so we replace it with a tiny local padding collator that keeps the same batching semantics. We also make tokenizer loading robust by preferring local fold directories (if present) and otherwise falling back to a model that can load without sentencepiece conversion, so inference completes. Finally, we ensure `submission.csv` is always written with the required `essay_id,score` columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")



## === cell 1
import random
import glob
import warnings
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.simplefilter("ignore")




## === cell 2
class PATHS:
    test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
    test_path_alt = (
        "/kaggle/data/learning-agency-lab-automated-essay-scoring-2/test.csv"
    )
    model_dir = "/kaggle/input/groupkfold-deberta-aes2-0/"




## === cell 3
class CFG:
    max_length = 512
    num_labels = 6




## === cell 4
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


seed_everything(42)




## === cell 5
class Tokenize(object):
    def __init__(self, test, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.test = test

    def __call__(self):
        texts = self.test["full_text"].tolist()
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=CFG.max_length,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors=None,
            padding=False,
        )
        return enc, self.tokenizer




## === cell 6
if os.path.exists(PATHS.test_path):
    test = pd.read_csv(PATHS.test_path)
elif os.path.exists(PATHS.test_path_alt):
    test = pd.read_csv(PATHS.test_path_alt)
else:
    raise FileNotFoundError(
        f"Could not find test.csv at {PATHS.test_path} or {PATHS.test_path_alt}"
    )

model_paths = []
if os.path.exists(PATHS.model_dir):
    cand = glob.glob(os.path.join(PATHS.model_dir, "*fold*"))
    cand = [p for p in cand if os.path.isdir(p)]
    cand.sort()
    model_paths = cand

if len(model_paths) == 0:
    fallback_model = "distilbert-base-uncased"
    model_paths = [fallback_model]

model_paths



## === cell 7
use_fp16 = torch.cuda.is_available()



## === cell 8
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenize = Tokenize(test, model_paths[0])
tokenized_test, tokenizer = tokenize()


class EncodedDataset(Dataset):
    def __init__(self, encodings):
        self.enc = encodings
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.n = len(self.input_ids)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


class SimplePadCollator:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, features):
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [
            torch.tensor(f["attention_mask"], dtype=torch.long) for f in features
        ]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tok.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}


per_device_eval_bs = 32 if torch.cuda.is_available() else 8
data_collator = SimplePadCollator(tokenizer=tokenizer)

num_workers = 4 if torch.cuda.is_available() else 0
dl = DataLoader(
    EncodedDataset(tokenized_test),
    batch_size=per_device_eval_bs,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)


def predict_logits(model, dataloader):
    model.eval()
    all_logits = []
    with torch.inference_mode():
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if use_fp16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(**batch)
            else:
                out = model(**batch)
            all_logits.append(out.logits.detach().cpu().numpy())
    return np.concatenate(all_logits, axis=0)


n_test = len(test)
final_pred_logits = None
n_models = 0

for i, model_path in enumerate(model_paths):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=CFG.num_labels
    ).to(device)

    pre_preds = predict_logits(model, dl)  # (n_test, num_labels)

    if final_pred_logits is None:
        if (
            pre_preds.ndim != 2
            or pre_preds.shape[0] != n_test
            or pre_preds.shape[1] != CFG.num_labels
        ):
            raise ValueError(
                f"Unexpected prediction shape at model {i}: {pre_preds.shape}"
            )
        final_pred_logits = np.zeros_like(pre_preds, dtype=np.float64)

    final_pred_logits += pre_preds.astype(np.float64, copy=False)
    n_models += 1

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if n_models == 0 or final_pred_logits is None:
    raise RuntimeError(
        "No predictions were produced; check model_paths and inference pipeline."
    )

n_models, final_pred_logits.shape



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
final_pred_logits /= n_models
final_pred = final_pred_logits.argmax(axis=1) + 1  # class 0-5 -> score 1-6

submission = pd.DataFrame(
    {"essay_id": test["essay_id"].values, "score": final_pred.astype(int)}
)
submission.to_csv("submission.csv", index=False)

submission.shape, submission.dtypes, submission.head()
