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

3.13

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

0.781539931926555

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import torch

print(f"PyTorch version: {torch.__version__}")

import transformers

print(f"Hugging Face Transformers version: {transformers.__version__}")

import datasets

print(f"Hugging Face Datasets version: {datasets.__version__}")



## === cell 1
import os
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_NAME = "bert-base-cased"
INPUT_DIR = "/kaggle/input/"
MODEL_DIR = "/kaggle/input/aes2-bertbase-5ep-results/aes2_bertbase_5ep_results/"
CHECKPOINT_SUBDIR = "checkpoint-1000"

MAX_LENGTH = 1024
RANDOM_SEED = 42
EVAL_USE_PRETRAIN = 1
SUBMISSION = 1

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:", device)
if torch.cuda.is_available():
    print("cuda current_device:", torch.cuda.current_device())
else:
    print("cuda not available; running on CPU")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)



## === cell 2
checkpoint_path = os.path.join(MODEL_DIR, CHECKPOINT_SUBDIR)
use_path = checkpoint_path if os.path.isdir(checkpoint_path) else MODEL_NAME

if os.path.isdir(checkpoint_path):
    print("Loading local checkpoint from:", checkpoint_path)
else:
    print("WARNING: checkpoint not found at:", checkpoint_path)
    print("Falling back to base model:", MODEL_NAME)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    use_path,
    local_files_only=os.path.isdir(checkpoint_path),
    use_fast=True,
)
print("Loaded tokenizer successfully")

model = transformers.AutoModelForSequenceClassification.from_pretrained(
    use_path,
    local_files_only=os.path.isdir(checkpoint_path),
)
print("Loaded model successfully")

use_fp16 = bool(torch.cuda.is_available())

training_args = transformers.TrainingArguments(
    output_dir=".",
    per_device_eval_batch_size=32,
    report_to="none",
    fp16=use_fp16,
)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
df = pd.read_csv(
    os.path.join(INPUT_DIR, "learning-agency-lab-automated-essay-scoring-2", "test.csv")
)
texts = df["full_text"].astype(str).values.tolist()
print("Read test.csv successfully:", df.shape)

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
)
print("Encoded texts successfully; keys:", list(encodings.keys()))

dataset = datasets.Dataset.from_dict(encodings)
print("Built Dataset successfully:", dataset)

submission_logits = trainer.predict(dataset).predictions
submission_preds = torch.argmax(torch.tensor(submission_logits), dim=1) + 1
submission_preds = submission_preds.cpu().numpy().astype(int)
print("Predicted successfully; preds shape:", submission_preds.shape)

submission_csv = pd.read_csv(
    os.path.join(
        INPUT_DIR,
        "learning-agency-lab-automated-essay-scoring-2",
        "sample_submission.csv",
    )
)
if len(submission_csv) != len(submission_preds):
    raise ValueError(
        f"Prediction length mismatch: sample_submission={len(submission_csv)} vs preds={len(submission_preds)}"
    )

submission_csv["score"] = submission_preds
submission_csv["score"] = submission_csv["score"].astype(int)
submission_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", submission_csv.shape)
print(submission_csv.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/569995581.py in <cell line: 0>()
     18 
     19 # Predictions: keep original core logic (argmax over logits + 1).
---> 20 submission_logits = trainer.predict(dataset).predictions
     21 submission_preds = torch.argmax(torch.tensor(submission_logits), dim=1) + 1
     22 submission_preds = submission_preds.cpu().numpy().astype(int)

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in predict(self, test_dataset, ignore_keys, metric_key_prefix)
   4275 
   4276         eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
-> 4277         output = eval_loop(
   4278             test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
   4279         )

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in evaluation_loop(self, dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
   4392 
   4393             # Prediction step
-> 4394             losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
   4395             main_input_name = getattr(self.model, "main_input_name", "input_ids")
   4396             inputs_decode = (

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in prediction_step(self, model, inputs, prediction_loss_only, ignore_keys)
   4618                     loss = None
   4619                     with self.compute_loss_context_manager():
-> 4620                         outputs = model(**inputs)
   4621                     if isinstance(outputs, dict):
   4622                         logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/accelerate/utils/operations.py in forward(*args, **kwargs)
    816 
    817     def forward(*args, **kwargs):
--> 818         return model_forward(*args, **kwargs)
    819 
    820     # To act like a decorator so that it can be popped when doing `extract_model_from_parallel`

/usr/local/lib/python3.11/dist-packages/accelerate/utils/operations.py in __call__(self, *args, **kwargs)
    804 
    805     def __call__(self, *args, **kwargs):
--> 806         return convert_to_fp32(self.model_forward(*args, **kwargs))
    807 
    808     def __getstate__(self):

/usr/local/lib/python3.11/dist-packages/torch/amp/autocast_mode.py in decorate_autocast(*args, **kwargs)
     42     def decorate_autocast(*args, **kwargs):
     43         with autocast_instance:
---> 44             return func(*args, **kwargs)
     45 
     46     decorate_autocast.__script_unsupported = "@autocast() decorator is not supported in script mode"  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/transformers/models/bert/modeling_bert.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
   1481         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
   1482 
-> 1483         outputs = self.bert(
   1484             input_ids,
   1485             attention_mask=attention_mask,

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/transformers/models/bert/modeling_bert.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask, past_key_values, use_cache, output_attentions, output_hidden_states, return_dict)
    930                 token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    931 
--> 932         embedding_output = self.embeddings(
    933             input_ids=input_ids,
    934             position_ids=position_ids,

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/transformers/models/bert/modeling_bert.py in forward(self, input_ids, token_type_ids, position_ids, inputs_embeds, past_key_values_length)
    183         if self.position_embedding_type == "absolute":
    184             position_embeddings = self.position_embeddings(position_ids)
--> 185             embeddings += position_embeddings
    186         embeddings = self.LayerNorm(embeddings)
    187         embeddings = self.dropout(embeddings)

RuntimeError: The size of tensor a (1024) must match the size of tensor b (512) at non-singleton dimension 1
