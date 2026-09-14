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
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.10

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
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

0.8198699774283027

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.42395) has done: 'I fix the early protobuf/transformers crash by forcing the C++ protobuf backend (the current setting triggers the `MessageFactory.GetPrototype` error in this environment). Then I fix the tokenization bug by not requesting PyTorch tensors before padding; instead I return variable-length lists and let `DataCollatorWithPadding` dynamically pad batches (this removes the “expected sequence of length …” error). Finally, I keep your model/training logic the same, but make prediction use a numerically-stable softmax and always write a valid `submission.csv` with `id,score`.'
- What this solution (achieved 0.42395) has done: 'I fix the early protobuf/transformers crash by switching the protobuf implementation to the C++ backend (the “python” backend is what triggers the `MessageFactory.GetPrototype` issue in this environment). Then I fix the tokenizer input type error by ensuring we pass plain Python `list[str]` (not numpy object arrays) into `tokenizer(...)`. Finally, I keep your training/inference logic intact but make the pipeline robust to missing models by cleanly falling back and always writing a valid `submission.csv` with `id,score`.'
- What this solution (achieved 0.44529) has done: 'The timeout is primarily coming from full fine-tuning a transformer with `Trainer` (plus extra overhead from Python protobufs, dataset wrapping, and slow CPU tokenization/list conversions). I keep the exact same model, objective, and 1-epoch training loop, but remove avoidable overhead: switch protobuf to the fast C++ implementation, avoid repeated pandas/Series work, tokenize from already-built Python lists without extra conversions, use a zero-copy dataset that returns tensors directly (no per-item dict rebuilding from nested datasets), and enable faster matmul/TF32 on GPU while keeping deterministic settings intact. I also explicitly set `torch.set_num_threads` on CPU to reduce oversubscription stalls, and keep `torch.compile` only when it’s safe. These changes are equivalent in semantics (same inputs, same labels, same Trainer training) and focus purely on reducing constant-factor runtime.'

# 9. Code solution

## === cell 0
import os

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["DISABLE_TRANSFORMERS_IMAGE_TRANSFORMS"] = "1"
os.environ["TRANSFORMERS_DISABLE_VISION"] = "1"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

set_seed(42)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

try:
    if not torch.cuda.is_available():
        torch.set_num_threads(min(4, (os.cpu_count() or 4)))
except Exception:
    pass

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def resolve_competition_dir() -> str:
    """
    Kaggle sometimes exposes the dataset under /kaggle/input/<slug>/...
    and your file tree also shows /kaggle/data/... mirrors.
    """
    candidates = [
        "/kaggle/input/us-patent-phrase-to-phrase-matching",
        "/kaggle/data/us-patent-phrase-to-phrase-matching",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for base in candidates:
        if os.path.isdir(base) and (
            os.path.exists(os.path.join(base, "train.csv"))
            or os.path.exists(
                os.path.join(base, "us-patent-phrase-to-phrase-matching", "train.csv")
            )
        ):
            if os.path.exists(os.path.join(base, "train.csv")):
                return base
            nested = os.path.join(base, "us-patent-phrase-to-phrase-matching")
            if os.path.exists(os.path.join(nested, "train.csv")):
                return nested
    return "/kaggle/input/us-patent-phrase-to-phrase-matching"


DATA_DIR = resolve_competition_dir()
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

print("Resolved DATA_DIR:", DATA_DIR)
print("Train exists:", os.path.exists(TRAIN_PATH), TRAIN_PATH)
print("Test exists :", os.path.exists(TEST_PATH), TEST_PATH)
print("Sample exists:", os.path.exists(SAMPLE_PATH), SAMPLE_PATH)




## === cell 2
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)

MODEL_CANDIDATES = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
    "microsoft/deberta-v3-small",
]


def try_load_model_and_tokenizer(name_or_path: str):
    tok = AutoTokenizer.from_pretrained(
        name_or_path, local_files_only=False, use_fast=True
    )
    mdl = AutoModelForSequenceClassification.from_pretrained(
        name_or_path,
        num_labels=1,
        problem_type="regression",
        local_files_only=False,
    )
    return mdl, tok


model = None
tokenizer = None
backbone_name = None
last_err = None

local_model_dir = os.path.join(DATA_DIR, "model")
if os.path.isdir(local_model_dir):
    try:
        model, tokenizer = try_load_model_and_tokenizer(local_model_dir)
        backbone_name = local_model_dir
    except Exception as e:
        last_err = e
        model = None
        tokenizer = None

if model is None or tokenizer is None:
    preferred = MODEL_CANDIDATES[0]  # distilbert-base-uncased
    try:
        model, tokenizer = try_load_model_and_tokenizer(preferred)
        backbone_name = preferred
    except Exception as e:
        last_err = e
        model = None
        tokenizer = None

if model is None or tokenizer is None:
    for name in MODEL_CANDIDATES[1:]:
        try:
            model, tokenizer = try_load_model_and_tokenizer(name)
            backbone_name = name
            break
        except Exception as e:
            last_err = e
            model = None
            tokenizer = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model is not None:
    try:
        model.config.use_cache = False
    except Exception:
        pass
    model.to(device)
    print("Loaded backbone:", backbone_name)
else:
    print(
        "WARNING: Could not load any transformer model. Will use fallback predictor.\n"
        f"Last error: {last_err!r}"
    )
print("Device:", device)




## === cell 3
class MyDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        self._len = (
            int(self.encodings["input_ids"].shape[0])
            if hasattr(self.encodings["input_ids"], "shape")
            else int(len(self.encodings["input_ids"]))
        )

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        out = {}
        for k, v in self.encodings.items():
            out[k] = v[idx]
        return out




## === cell 4
def build_pair_text_arrays(df: pd.DataFrame):
    c0 = df["context"].astype("string").str.slice(0, 1).fillna("")
    anchor = df["anchor"].astype("string").fillna("")
    target = df["target"].astype("string").fillna("")
    t1 = (c0 + " " + anchor).astype(str).to_numpy()
    t2 = target.astype(str).to_numpy()
    return t1, t2


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_text1, train_text2 = build_pair_text_arrays(train_df)
test_text1, test_text2 = build_pair_text_arrays(test_df)

print("Train rows:", len(train_df), "Test rows:", len(test_df))




## === cell 5
from difflib import SequenceMatcher

if tokenizer is None or model is None:

    def sim(a, b):
        return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

    sims = np.array(
        [sim(a, b) for a, b in zip(test_df["anchor"].values, test_df["target"].values)],
        dtype=np.float32,
    )

    pred = np.clip(sims, 0.0, 1.0).astype(np.float32)

    submit = pd.DataFrame({"id": test_df["id"].values, "score": pred})
    submit.to_csv("submission.csv", index=False)
    print(submit.head())
    print("Wrote submission.csv with shape:", submit.shape)
    print("Submission columns:", submit.columns.tolist())

else:
    y = train_df["score"].to_numpy(dtype=np.float32, copy=False)

    max_len = 128

    def fast_encode_pairs_to_tensors(t1, t2):
        enc = tokenizer(
            list(t1),  # ensure stable iteration type without per-element Python work
            list(t2),
            truncation=True,
            padding=False,  # dynamic padding in the collator
            max_length=max_len,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        return enc

    train_enc = fast_encode_pairs_to_tensors(train_text1, train_text2)
    train_enc["labels"] = y.astype(np.float32)

    test_enc = fast_encode_pairs_to_tensors(test_text1, test_text2)

    from transformers import default_data_collator

    class DictTensorDataset(torch.utils.data.Dataset):
        def __init__(self, enc: dict, include_labels: bool):
            self.enc = enc
            self.include_labels = include_labels
            self._len = len(self.enc["input_ids"])

        def __len__(self):
            return int(self._len)

        def __getitem__(self, idx):
            if self.include_labels:
                return {
                    "input_ids": torch.tensor(
                        self.enc["input_ids"][idx], dtype=torch.long
                    ),
                    "attention_mask": torch.tensor(
                        self.enc["attention_mask"][idx], dtype=torch.long
                    ),
                    "labels": torch.tensor(
                        self.enc["labels"][idx], dtype=torch.float32
                    ).view(1),
                }
            else:
                return {
                    "input_ids": torch.tensor(
                        self.enc["input_ids"][idx], dtype=torch.long
                    ),
                    "attention_mask": torch.tensor(
                        self.enc["attention_mask"][idx], dtype=torch.long
                    ),
                }

    trainset_w = DictTensorDataset(train_enc, include_labels=True)
    testset_w = DictTensorDataset(test_enc, include_labels=False)

    print("Train examples:", len(trainset_w), "Test examples:", len(testset_w))

    is_cuda = torch.cuda.is_available()
    per_device_bs = 64 if is_cuda else 16
    grad_accum = 4 if (is_cuda and per_device_bs == 64) else 1

    num_workers = 2 if is_cuda else 0
    prefetch_factor = 4 if (is_cuda and num_workers > 0) else None
    persistent_workers = True if (is_cuda and num_workers > 0) else False

    args = TrainingArguments(
        output_dir="/kaggle/working/tmp_model",
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=200,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        fp16=is_cuda,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=is_cuda,
        dataloader_prefetch_factor=prefetch_factor,
        dataloader_persistent_workers=persistent_workers,
        disable_tqdm=True,
        remove_unused_columns=False,
    )


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=trainset_w,
        data_collator=default_data_collator,
    )

    trainer.train()

    outputs = trainer.predict(testset_w)
    preds = np.asarray(outputs.predictions, dtype=np.float32).reshape(-1)

    pred = np.clip(preds, 0.0, 1.0).astype(np.float32)

    submit = pd.DataFrame({"id": test_df["id"].values, "score": pred})
    submit.to_csv("submission.csv", index=False)

    print(submit.head())
    print("Wrote submission.csv with shape:", submit.shape)
    print("Submission columns:", submit.columns.tolist())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/724653208.py in <cell line: 0>()
    130     )
    131 
--> 132     trainer.train()
    133 
    134     outputs = trainer.predict(testset_w)

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in train(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
   2204                 hf_hub_utils.enable_progress_bars()
   2205         else:
-> 2206             return inner_training_loop(
   2207                 args=args,
   2208                 resume_from_checkpoint=resume_from_checkpoint,

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _inner_training_loop(self, batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
   2500                 update_step += 1
   2501                 num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
-> 2502                 batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
   2503                 for i, inputs in enumerate(batch_samples):
   2504                     step += 1

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in get_batch_samples(self, epoch_iterator, num_batches, device)
   5298         for _ in range(num_batches):
   5299             try:
-> 5300                 batch_samples.append(next(epoch_iterator))
   5301             except StopIteration:
   5302                 break

/usr/local/lib/python3.11/dist-packages/accelerate/data_loader.py in __iter__(self)
    565         # We iterate one batch ahead to check when we are at the end
    566         try:
--> 567             current_batch = next(dataloader_iter)
    568         except StopIteration:
    569             self.end()

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     53         else:
     54             data = self.dataset[possibly_batched_index]
---> 55         return self.collate_fn(data)

/usr/local/lib/python3.11/dist-packages/transformers/data/data_collator.py in default_data_collator(features, return_tensors)
     91 
     92     if return_tensors == "pt":
---> 93         return torch_default_data_collator(features)
     94     elif return_tensors == "tf":
     95         return tf_default_data_collator(features)

/usr/local/lib/python3.11/dist-packages/transformers/data/data_collator.py in torch_default_data_collator(features)
    153         if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
    154             if isinstance(v, torch.Tensor):
--> 155                 batch[k] = torch.stack([f[k] for f in features])
    156             elif isinstance(v, np.ndarray):
    157                 batch[k] = torch.from_numpy(np.stack([f[k] for f in features]))

RuntimeError: stack expects each tensor to be equal size, but got [10] at entry 0 and [8] at entry 1
