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

0.6518295764981217

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the crash caused by an incompatible `protobuf`/`sentencepiece` stack (which triggers `MessageFactory.GetPrototype` errors) by removing the forced pure-Python protobuf setting and instead explicitly using the fast Hugging Face tokenizer implementation. I also make the model/tokenizer paths robust by falling back to the competition dataset paths when the custom `/kaggle/input/bert-model/...` directory is not present, and add a safe fallback model (public `bert-base-uncased`) so the notebook always produces a valid submission CSV. Finally, I harden the state-dict loading to handle common key mismatches (e.g., `module.` prefix) without changing the architecture, ensuring inference runs end-to-end and writes `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

from transformers import BertTokenizerFast, BertModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class EssayDataset(Dataset):
    """Custom Dataset class for essays"""

    def __init__(self, tokenizer, essays, max_length):
        self.tokenizer = tokenizer
        self.texts = essays
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].flatten()
        attention_mask = inputs["attention_mask"].flatten()
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class BertRegressor(nn.Module):
    """BERT Model for Regression Tasks"""

    def __init__(self, pre_trained_model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_name)
        self.out = nn.Linear(
            self.bert.config.hidden_size, 1
        )  # keep name 'out' for state_dict compatibility

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.out(pooled_output)


def _exists_dir(p: str) -> bool:
    return isinstance(p, str) and len(p) > 0 and os.path.isdir(p)


def _exists_file(p: str) -> bool:
    return isinstance(p, str) and len(p) > 0 and os.path.isfile(p)


def load_state_dict_forgiving(model: nn.Module, state_path: str) -> None:
    """Load weights while handling common mismatches (e.g., DataParallel 'module.' prefix)."""
    state = torch.load(state_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(
            f"Missing keys when loading state_dict (showing up to 20): {missing[:20]}"
        )
    if unexpected:
        logger.warning(
            f"Unexpected keys when loading state_dict (showing up to 20): {unexpected[:20]}"
        )




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
LOCAL_MODEL_PATH = "/kaggle/input/bert-model/pytorch/bertmodel/2/bert_regressor (2)/local_bert_base_uncased_model"
TOKENIZER_PATH = "/kaggle/input/bert-model/pytorch/bertmodel/2/bert_regressor (2)/local_bert_base_uncased_tokenizer"
MODEL_WEIGHTS_PATH = (
    "/kaggle/input/bert-model/pytorch/bertmodel/2/bert_regressor/bert_regressor.pth"
)

test_data_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
sample_sub_path = (
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)

BATCH_SIZE = 16
MAX_LEN = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

base_model_name = None
tokenizer_source = None

if _exists_dir(LOCAL_MODEL_PATH) and _exists_dir(TOKENIZER_PATH):
    base_model_name = LOCAL_MODEL_PATH
    tokenizer_source = TOKENIZER_PATH
    logger.info("Using provided local BERT model/tokenizer directories.")
else:
    base_model_name = "bert-base-uncased"
    tokenizer_source = "bert-base-uncased"
    logger.warning(
        "Custom local model/tokenizer path not found. Falling back to 'bert-base-uncased' "
        "(submission will still be valid; score may differ)."
    )

tokenizer = BertTokenizerFast.from_pretrained(tokenizer_source)

model = BertRegressor(base_model_name)

if _exists_file(MODEL_WEIGHTS_PATH):
    logger.info(f"Loading model weights from: {MODEL_WEIGHTS_PATH}")
    load_state_dict_forgiving(model, MODEL_WEIGHTS_PATH)
else:
    logger.warning(
        "Model weights file not found; using randomly-initialized regression head (submission will be valid)."
    )

model.to(device)
model.eval()

test_df = pd.read_csv(test_data_path)
assert "essay_id" in test_df.columns and "full_text" in test_df.columns

test_dataset = EssayDataset(tokenizer, test_df["full_text"].tolist(), MAX_LEN)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

predictions = []
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions.extend(output.squeeze(-1).detach().cpu().numpy().tolist())

sub = test_df[["essay_id"]].copy()
sub["score"] = predictions

sub["score"] = pd.to_numeric(sub["score"], errors="coerce").fillna(3.0)
sub["score"] = sub["score"].clip(1, 6).round(0).astype(int)

sample_sub = pd.read_csv(sample_sub_path)
sub = sample_sub[["essay_id"]].merge(sub, on="essay_id", how="left")
sub["score"] = sub["score"].fillna(3).astype(int)

output_file_path = "/kaggle/working/submission.csv"
sub.to_csv(output_file_path, index=False)
logger.info(f"Wrote submission to: {output_file_path}")
logger.info(f"Submission shape: {sub.shape}")
logger.info(f"Submission head:\n{sub.head()}")




## === cell 2
submission = pd.read_csv("/kaggle/working/submission.csv")
assert list(submission.columns) == ["essay_id", "score"]
assert submission["essay_id"].isna().sum() == 0
submission["score"] = submission["score"].astype(int)
assert submission["score"].between(1, 6).all()
submission.head()
