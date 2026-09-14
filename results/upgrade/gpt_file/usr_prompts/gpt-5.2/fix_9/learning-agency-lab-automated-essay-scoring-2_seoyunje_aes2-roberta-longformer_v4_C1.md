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

cudf-polars-cu12==25.6.0
datasets==4.4.1
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
polars==1.25.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
tqdm==4.67.1
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

0.7968195210060658

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the protobuf-related crash by forcing the compatible Python protobuf implementation before importing `transformers/datasets`, and I also disable evaluation/save strategies when running inference-only so `Trainer` can be instantiated without an `eval_dataset`. Then I ensure a `trainer` object always exists by loading the model from `CFG.preset` and using it directly for test-time prediction. Finally, I keep the same regression-to-1..6 rounding logic and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf crash that prevents the notebook from running by pinning a compatible `protobuf` major version at runtime and forcing the pure-Python implementation before importing `transformers/datasets`. Then I make sure inference mode creates a usable `trainer` by attaching a minimal `TrainingArguments` and a dummy dataset (so `Trainer.predict()` works reliably across HF versions), without changing your model, preprocessing, or rounding logic. Finally, I ensure the submission is always written as `submission.csv` with the exact required columns and correct row alignment.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the submission rows not matching Kaggle’s expected `essay_id` order/contents (row count mismatch vs sample_submission is a red flag), or with predictions being produced by an effectively “untrained/random” head because the loaded checkpoint isn’t actually being used as intended for regression. I keep your exact model and inference approach, but (1) force the model to run in pure inference mode (`model.eval()` + `torch.no_grad()` via `Trainer` settings), (2) ensure we load a sequence-classification checkpoint robustly even if the preset folder is nested, and (3) write the submission by left-joining onto `sample_submission.csv` so the output has exactly the expected `essay_id` rows and order. These are minimal changes that typically move a broken/invalid-alignment submission from ~0.0 toward a reasonable QWK without changing core modeling logic.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a prediction/label scale mismatch caused by loading a classification head (6 logits) as if it were a regression head (1 logit), which makes `trainer.predict()` outputs meaningless and then rounding/clipping collapses to near-constant predictions. I keep your Longformer + Trainer inference flow, but make the minimal fix: detect whether the loaded model outputs 1 logit (regression) or 6 logits (classification) and convert to a continuous score accordingly (expected value over classes for 6 logits, raw value for 1 logit), then apply the same 1–6 rounding/clipping. I also ensure the submission order exactly matches `sample_submission.csv` via the same merge you already do, so Kaggle gets the expected ids/row count. These changes preserve your architecture and evaluation semantics but should move the score upward toward the target instead of producing degenerate predictions.'
- What this solution (achieved 0.0) has done: 'I fix the pipeline so it can actually load a local HF checkpoint in the Kaggle environment instead of crashing on a missing `/kaggle/input/aes2-longformer/...` directory, by automatically falling back to a known available base model (`allenai/longformer-base-4096`) when the preset path doesn’t exist. I also make `INFERENCE=True` work end-to-end by ensuring `tokenizer`/`trainer` are always created and by using a minimal dummy train_dataset (required by `Trainer` in some versions) without changing your core model/Trainer inference flow. Finally, I keep your existing 1–6 rounding/clipping and “classification head → expected value” conversion, and always write a correctly ordered `submission.csv` by merging onto `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely coming from using an unfine-tuned fallback base Longformer checkpoint, which produces near-random/degenerate predictions even though the submission CSV is valid. To move the score upward toward your target, the smallest legitimate change is to actually train the existing model briefly (same architecture, same preprocessing, same loss/Trainer loop) and then use that trained model for test predictions. I switch `CFG.INFERENCE` to training mode, keep your fold split, and keep the same rounding/clipping post-processing so evaluation semantics stay the same. I also ensure the Trainer uses the real training dataset (not the 1-row dummy) when training, and still writes `submission.csv` in the required format.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import sys
import subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            raise RuntimeError(f"Incompatible protobuf version detected: {pb_ver}")
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        import importlib

        importlib.invalidate_caches()


_ensure_protobuf_compatible()

import gc
import re
import ctypes
import random
from tqdm import tqdm
import polars as pl

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    LongformerForSequenceClassification,
    DataCollatorWithPadding,
    AutoConfig,
)
from transformers import Trainer, TrainingArguments
from datasets import Dataset

import warnings

warnings.filterwarnings("ignore")

print("Python:", sys.version)
import google.protobuf

print("protobuf:", google.protobuf.__version__)




## === cell 1
class CFG:
    SEED = 2024
    VER = 1

    INFERENCE = None

    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"
    preset = "/kaggle/input/aes2-longformer/DeBerta_BASE_v1"

    MAX_LEN = 1024
    TRAIN_BATCH = 4
    EVAL_BATCH = 4
    EPOCHS = 4




## === cell 2
Clean = True


def clean_memory():
    if Clean:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        gc.collect()


clean_memory()




## === cell 3
def seed_everything():
    random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)
    np.random.seed(CFG.SEED)
    torch.manual_seed(CFG.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CFG.SEED)
        torch.cuda.manual_seed_all(CFG.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything()




## === cell 4
def resolve_model_path_or_fallback(preset_path: str) -> str:
    """
    Bugfix: the provided CFG.preset points to a Kaggle input dataset that is not present,
    which crashes with FileNotFoundError and prevents any submission from being produced.

    Minimal, robust behavior:
    - If preset_path is a local HF directory (contains config.json), use it.
    - Else, fall back to a known-available HF model id for Longformer.
      (This preserves the same core architecture family and lets the script run end-to-end.)
    """
    preset_path = (preset_path or "").rstrip("/")

    if preset_path and os.path.isdir(preset_path):
        if os.path.exists(os.path.join(preset_path, "config.json")):
            return preset_path
        for root, _, files in os.walk(preset_path):
            if "config.json" in files:
                return root

    fallback = "allenai/longformer-base-4096"
    return fallback


CFG.preset = resolve_model_path_or_fallback(CFG.preset)
print("Using preset:", CFG.preset)



## === cell 5
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values(by="essay_id")
df_train["label"] = df_train["score"] - 1

skf = StratifiedKFold(n_splits=5, random_state=CFG.SEED, shuffle=True)
for i, (_, val_index) in enumerate(skf.split(df_train, df_train["score"])):
    df_train.loc[val_index, "fold"] = i

print("Shape of Train: ", df_train.shape)
print(df_train.head())



## === cell 6
df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values(by="essay_id")
print("Shape of Test: ", df_test.shape)
print(df_test.head())



## === cell 7
cList = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would have",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have",
}



## === cell 8
c_re = re.compile("(%s)" % "|".join(cList.keys()))


def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]

    return c_re.sub(replace, text)


def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)


def dataPreprocessing(x):
    x = x.lower()
    x = removeHTML(x)
    x = re.sub(r"@\w+", "", x)
    x = re.sub(r"'\d+", "", x)
    x = re.sub(r"\d+", "", x)
    x = re.sub(r"http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = expandContractions(x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = re.sub(r'[^\w\s.,;:""\'\'?!]', "", x)
    x = x.strip()
    return x




## === cell 9
train = pl.from_pandas(df_train)
test = pl.from_pandas(df_test)

train = train.with_columns(
    pl.col("full_text").map_elements(
        lambda x: dataPreprocessing(x), return_dtype=pl.Utf8
    )
)
df_train = train.to_pandas()

test = test.with_columns(
    pl.col("full_text").map_elements(
        lambda x: dataPreprocessing(x), return_dtype=pl.Utf8
    )
)
df_test = test.to_pandas()



## === cell 10
df_train["label"] = df_train["label"].astype("float32")



## === cell 11
tokenizer = AutoTokenizer.from_pretrained(
    CFG.preset,
    clean_up_tokenization_spaces=False,
    use_fast=True,
)
print(tokenizer.__class__.__name__)




## === cell 12
def preprocess(sample):
    return tokenizer(
        sample["full_text"],
        truncation=True,
        max_length=CFG.MAX_LEN,
    )




## === cell 13
if CFG.INFERENCE is None:
    train_df = df_train[df_train["fold"] != 0]
    valid_df = df_train[df_train["fold"] == 0]

    dataset_v = Dataset.from_pandas(valid_df, preserve_index=False)
    dataset_t = Dataset.from_pandas(train_df, preserve_index=False)

    tokenized_dataset_v = dataset_v.map(
        preprocess,
        num_proc=None,
        remove_columns=["essay_id", "full_text", "score", "fold"],
    )
    tokenized_dataset_t = dataset_t.map(
        preprocess,
        num_proc=None,
        remove_columns=["essay_id", "full_text", "score", "fold"],
    )
else:
    dataset_train = Dataset.from_pandas(df_train, preserve_index=False)
    tokenized_dataset_train = dataset_train.map(
        preprocess,
        num_proc=None,
        remove_columns=["essay_id", "full_text", "score", "fold"],
    )



## === cell 14
if CFG.INFERENCE is None:
    eval_strategy = "epoch"
    save_strategy = "epoch"
    load_best_model_at_end = True
else:
    eval_strategy = "no"
    save_strategy = "no"
    load_best_model_at_end = False

training_args = TrainingArguments(
    output_dir=f"/kaggle/working/output_v{CFG.VER}",
    per_device_train_batch_size=CFG.TRAIN_BATCH,
    per_device_eval_batch_size=CFG.EVAL_BATCH,
    num_train_epochs=CFG.EPOCHS,
    eval_strategy=eval_strategy,
    save_strategy=save_strategy,
    load_best_model_at_end=load_best_model_at_end,
    weight_decay=1e-3,
    fp16=torch.cuda.is_available(),
    learning_rate=1e-5,
    lr_scheduler_type="linear",
    optim="adamw_torch",
    metric_for_best_model="qwk",
    save_total_limit=1,
    report_to="none",
    gradient_checkpointing=True,
    gradient_accumulation_steps=2,
    logging_steps=50,
    do_train=(CFG.INFERENCE is None),
    do_eval=(CFG.INFERENCE is None),
)




## === cell 15
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    qwk = cohen_kappa_score(
        labels, np.clip(predictions, 0, 5).round(), weights="quadratic"
    )
    return {"qwk": qwk}




## === cell 16
model_config = AutoConfig.from_pretrained(CFG.preset)

model_config.attention_probs_dropout_prob = 0.0
model_config.hidden_dropout_prob = 0.0

num_labels = getattr(model_config, "num_labels", None)
if num_labels is None:
    num_labels = 1
if int(num_labels) == 1:
    model_config.problem_type = "regression"
else:
    model_config.problem_type = "single_label_classification"

model = LongformerForSequenceClassification.from_pretrained(
    CFG.preset, config=model_config
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    train_dataset=(
        tokenized_dataset_t if CFG.INFERENCE is None else tokenized_dataset_train
    ),
    eval_dataset=tokenized_dataset_v if CFG.INFERENCE is None else None,
)

if CFG.INFERENCE is not None:
    trainer.model.eval()

if CFG.INFERENCE is None:
    trainer.train()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3611921168.py in <cell line: 0>()
     33 
     34 if CFG.INFERENCE is None:
---> 35     trainer.train()
     36 

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in train(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
   2204                 hf_hub_utils.enable_progress_bars()
   2205         else:
-> 2206             return inner_training_loop(
   2207                 args=args,
   2208                 resume_from_checkpoint=resume_from_checkpoint,

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _inner_training_loop(self, batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
   2546                     )
   2547                     with context():
-> 2548                         tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
   2549 
   2550                     if (

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in training_step(self, model, inputs, num_items_in_batch)
   3747 
   3748         with self.compute_loss_context_manager():
-> 3749             loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
   3750 
   3751         del inputs

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in compute_loss(self, model, inputs, return_outputs, num_items_in_batch)
   3834                 loss_kwargs["num_items_in_batch"] = num_items_in_batch
   3835             inputs = {**inputs, **loss_kwargs}
-> 3836         outputs = model(**inputs)
   3837         # Save past state if it exists
   3838         # TODO: this needs to be fixed and made cleaner later.

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

/usr/local/lib/python3.11/dist-packages/transformers/models/longformer/modeling_longformer.py in forward(self, input_ids, attention_mask, global_attention_mask, head_mask, token_type_ids, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
   1828             elif self.config.problem_type == "single_label_classification":
   1829                 loss_fct = CrossEntropyLoss()
-> 1830                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
   1831             elif self.config.problem_type == "multi_label_classification":
   1832                 loss_fct = BCEWithLogitsLoss()

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/loss.py in forward(self, input, target)
   1293 
   1294     def forward(self, input: Tensor, target: Tensor) -> Tensor:
-> 1295         return F.cross_entropy(
   1296             input,
   1297             target,

/usr/local/lib/python3.11/dist-packages/torch/nn/functional.py in cross_entropy(input, target, weight, size_average, ignore_index, reduce, reduction, label_smoothing)
   3492     if size_average is not None or reduce is not None:
   3493         reduction = _Reduction.legacy_get_string(size_average, reduce)
-> 3494     return torch._C._nn.cross_entropy_loss(
   3495         input,
   3496         target,

RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Float'

## === cell 17
if CFG.INFERENCE is None:
    y_true = valid_df["score"].values
    predictions = trainer.predict(tokenized_dataset_v).predictions

    pred_arr = np.asarray(predictions)
    if pred_arr.ndim == 2 and pred_arr.shape[1] > 1:
        probs = F.softmax(torch.from_numpy(pred_arr), dim=1).numpy()
        exp_label = (probs * np.arange(probs.shape[1])[None, :]).sum(axis=1)
        pred_score = exp_label + 1
    else:
        pred_score = pred_arr.reshape(-1) + 1

    cm = confusion_matrix(
        y_true, np.clip(pred_score, 1, 6).round(), labels=[x for x in range(1, 7)]
    )
    draw_cm = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[x for x in range(1, 7)]
    )
    draw_cm.plot()
    plt.show()

    trainer.save_model(f"/kaggle/working/DeBerta_BASE_v{CFG.VER}")



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/564879130.py in <cell line: 0>()
      1 if CFG.INFERENCE is None:
      2     y_true = valid_df["score"].values
----> 3     predictions = trainer.predict(tokenized_dataset_v).predictions
      4 
      5     pred_arr = np.asarray(predictions)

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
   4608                 if has_labels or loss_without_labels:
   4609                     with self.compute_loss_context_manager():
-> 4610                         loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
   4611                     loss = loss.detach().mean()
   4612 

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in compute_loss(self, model, inputs, return_outputs, num_items_in_batch)
   3834                 loss_kwargs["num_items_in_batch"] = num_items_in_batch
   3835             inputs = {**inputs, **loss_kwargs}
-> 3836         outputs = model(**inputs)
   3837         # Save past state if it exists
   3838         # TODO: this needs to be fixed and made cleaner later.

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

/usr/local/lib/python3.11/dist-packages/transformers/models/longformer/modeling_longformer.py in forward(self, input_ids, attention_mask, global_attention_mask, head_mask, token_type_ids, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
   1828             elif self.config.problem_type == "single_label_classification":
   1829                 loss_fct = CrossEntropyLoss()
-> 1830                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
   1831             elif self.config.problem_type == "multi_label_classification":
   1832                 loss_fct = BCEWithLogitsLoss()

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/loss.py in forward(self, input, target)
   1293 
   1294     def forward(self, input: Tensor, target: Tensor) -> Tensor:
-> 1295         return F.cross_entropy(
   1296             input,
   1297             target,

/usr/local/lib/python3.11/dist-packages/torch/nn/functional.py in cross_entropy(input, target, weight, size_average, ignore_index, reduce, reduction, label_smoothing)
   3492     if size_average is not None or reduce is not None:
   3493         reduction = _Reduction.legacy_get_string(size_average, reduce)
-> 3494     return torch._C._nn.cross_entropy_loss(
   3495         input,
   3496         target,

RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Float'

## === cell 18
dataset_test = Dataset.from_pandas(df_test, preserve_index=False)
tokenized_test = dataset_test.map(
    preprocess,
    num_proc=None,
    remove_columns=["essay_id", "full_text"],
)



## === cell 19
preds_test = trainer.predict(tokenized_test).predictions

pred_arr = np.asarray(preds_test)
if pred_arr.ndim == 2 and pred_arr.shape[1] > 1:
    probs = F.softmax(torch.from_numpy(pred_arr), dim=1).numpy()
    exp_label = (probs * np.arange(probs.shape[1])[None, :]).sum(axis=1)
    preds_score = exp_label + 1
else:
    preds_score = pred_arr.reshape(-1) + 1



## === cell 20
sample_sub = pd.read_csv(CFG.BASE_PATH + "sample_submission.csv")

pred_df = pd.DataFrame({"essay_id": df_test["essay_id"].values})
pred_df["score"] = np.clip(preds_score, 1, 6).round().astype(int)

sub = sample_sub[["essay_id"]].merge(pred_df, on="essay_id", how="left")
sub["score"] = sub["score"].fillna(3).astype(int)

sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
print(sub.head())
print("Saved to:", os.path.abspath("submission.csv"))
print("Missing scores after merge:", int(sub["score"].isna().sum()))
print("Pred score value counts:\n", sub["score"].value_counts().sort_index())
