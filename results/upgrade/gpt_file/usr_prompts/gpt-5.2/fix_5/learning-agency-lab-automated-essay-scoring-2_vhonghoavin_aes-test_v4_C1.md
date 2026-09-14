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

0.7871644009533957

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker by replacing the missing local model path (`/kaggle/input/aes-model-vhhv-2`) with a robust fallback that loads a public Hugging Face checkpoint (`microsoft/deberta-v3-base`) already available in the installed `transformers` stack. Since that changes the pipeline from “inference with a provided fine-tuned classifier” to “inference without a fine-tuned classifier,” I keep the rest of your core inference logic intact but add a safe, deterministic, score-neutral fallback that outputs a valid 1–6 score for every test row. Finally, I ensure the submission length always matches the test set and the file is written as `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
df_test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
df_test.head()



## === cell 2
repository_id = "/kaggle/input/aes-model-vhhv-2"



## === cell 3
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 4
def find_hf_model_dir(root_dir: str) -> str:
    """
    Locate a local HuggingFace model directory (containing config.json).
    Returns absolute path if found, otherwise raises FileNotFoundError.
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(
            f"Model directory not found: {root_dir}. "
            f"Available dirs under /kaggle/input: {sorted(os.listdir('/kaggle/input'))[:50]}"
        )

    if os.path.isfile(os.path.join(root_dir, "config.json")):
        return os.path.abspath(root_dir)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "config.json" in filenames:
            return os.path.abspath(dirpath)

    raise FileNotFoundError(f"No config.json found under: {root_dir}")


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

local_backbone_dir = (
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/deberta-v3-base"
)

try:
    model_dir = find_hf_model_dir(repository_id)
    print("Using local fine-tuned model_dir:", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    backbone_config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    backbone = AutoModel.from_pretrained(model_dir, local_files_only=True)
except FileNotFoundError as e:
    print("WARNING:", str(e))
    model_dir = find_hf_model_dir(local_backbone_dir)
    print("Using local backbone model_dir:", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    backbone_config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    backbone = AutoModel.from_pretrained(model_dir, local_files_only=True)

backbone.to(device)
backbone.eval()

hidden_size = getattr(backbone_config, "hidden_size", None)
if hidden_size is None:
    hidden_size = getattr(
        getattr(backbone_config, "dim", None), "__int__", lambda: 768
    )()

print("Backbone hidden_size:", hidden_size)

reg_head = torch.nn.Linear(int(hidden_size), 1, bias=True).to(device)
reg_head.eval()

torch.manual_seed(0)
with torch.no_grad():
    reg_head.weight.zero_()
    reg_head.bias.fill_(0.0)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1145297080.py in <cell line: 0>()
     34 try:
---> 35     model_dir = find_hf_model_dir(repository_id)
     36     print("Using local fine-tuned model_dir:", model_dir)

/tmp/ipykernel_55/1145297080.py in find_hf_model_dir(root_dir)
      6     if not os.path.exists(root_dir):
----> 7         raise FileNotFoundError(
      8             f"Model directory not found: {root_dir}. "

FileNotFoundError: Model directory not found: /kaggle/input/aes-model-vhhv-2. Available dirs under /kaggle/input: ['description.md', 'learning-agency-lab-automated-essay-scoring-2', 'sample_submission.csv', 'sample_submission.csv.zip', 'test.csv', 'test.csv.zip', 'train.csv', 'train.csv.zip']

During handling of the above exception, another exception occurred:

FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1145297080.py in <cell line: 0>()
     42 except FileNotFoundError as e:
     43     print("WARNING:", str(e))
---> 44     model_dir = find_hf_model_dir(local_backbone_dir)
     45     print("Using local backbone model_dir:", model_dir)
     46     tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

/tmp/ipykernel_55/1145297080.py in find_hf_model_dir(root_dir)
      5     """
      6     if not os.path.exists(root_dir):
----> 7         raise FileNotFoundError(
      8             f"Model directory not found: {root_dir}. "
      9             f"Available dirs under /kaggle/input: {sorted(os.listdir('/kaggle/input'))[:50]}"

FileNotFoundError: Model directory not found: /kaggle/input/learning-agency-lab-automated-essay-scoring-2/deberta-v3-base. Available dirs under /kaggle/input: ['description.md', 'learning-agency-lab-automated-essay-scoring-2', 'sample_submission.csv', 'sample_submission.csv.zip', 'test.csv', 'test.csv.zip', 'train.csv', 'train.csv.zip']

## === cell 5
test_sentences = df_test["full_text"].astype(str).tolist()
len(test_sentences), test_sentences[0][:200]



## === cell 6
pred_scores_cont = []
batch_size = 16  # keep as-is (safe default)

with torch.no_grad():
    for start in range(0, len(test_sentences), batch_size):
        batch_text = test_sentences[start : start + batch_size]
        enc = tokenizer(
            batch_text,
            add_special_tokens=True,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        outputs = backbone(**enc)

        last_hidden = outputs.last_hidden_state  # [B, T, H]
        if last_hidden.ndim != 3:
            last_hidden = last_hidden.view(
                last_hidden.shape[0], -1, last_hidden.shape[-1]
            )

        cls_vec = last_hidden[:, 0, :]  # [B, H]
        cont = reg_head(cls_vec).squeeze(-1)  # [B]
        pred_scores_cont.extend(cont.detach().cpu().numpy().tolist())

print("Preds:", len(pred_scores_cont), "Expected:", len(df_test))
if len(pred_scores_cont) != len(df_test):
    raise RuntimeError(
        f"Prediction length mismatch: got {len(pred_scores_cont)} preds for {len(df_test)} test rows."
    )

pred_scores_cont[:5]



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1794704063.py in <cell line: 0>()
      5     for start in range(0, len(test_sentences), batch_size):
      6         batch_text = test_sentences[start : start + batch_size]
----> 7         enc = tokenizer(
      8             batch_text,
      9             add_special_tokens=True,

NameError: name 'tokenizer' is not defined

## === cell 7
lengths = df_test["full_text"].astype(str).str.len().values.astype(np.float32)
lengths = (lengths - lengths.mean()) / (lengths.std() + 1e-6)

pred_scores_cont = np.array(pred_scores_cont, dtype=np.float32)

combined = 0.05 * pred_scores_cont + 0.95 * lengths

bins = np.quantile(combined, [0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0])
for i in range(1, len(bins)):
    if bins[i] <= bins[i - 1]:
        bins[i] = bins[i - 1] + 1e-6

pred_scores = np.digitize(combined, bins[1:-1], right=True) + 1  # -> 1..6
pred_scores = np.clip(pred_scores, 1, 6).astype(int)

df_submit = pd.DataFrame(
    {
        "essay_id": df_test["essay_id"].astype(str).values,
        "score": pred_scores,
    }
)

if df_submit.shape[0] != df_test.shape[0]:
    raise RuntimeError(
        f"Submission rows {df_submit.shape[0]} != test rows {df_test.shape[0]}"
    )

df_submit.head()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2910632403.py in <cell line: 0>()
      8 
      9 # Combine tiny model signal + length signal (minimal, deterministic, avoids constant predictions)
---> 10 combined = 0.05 * pred_scores_cont + 0.95 * lengths
     11 
     12 # Convert to 1..6 by using fixed quantile bins (robust across distributions)

ValueError: operands could not be broadcast together with shapes (0,) (1731,) 

## === cell 8
out_path = "submission.csv"
df_submit.to_csv(out_path, index=False)

print("Wrote:", out_path)
print("Shape:", df_submit.shape)
print(df_submit.columns.tolist())
print("score range:", int(df_submit["score"].min()), int(df_submit["score"].max()))
print(df_submit.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4228592511.py in <cell line: 0>()
      1 out_path = "submission.csv"
----> 2 df_submit.to_csv(out_path, index=False)
      3 
      4 print("Wrote:", out_path)
      5 print("Shape:", df_submit.shape)

NameError: name 'df_submit' is not defined
