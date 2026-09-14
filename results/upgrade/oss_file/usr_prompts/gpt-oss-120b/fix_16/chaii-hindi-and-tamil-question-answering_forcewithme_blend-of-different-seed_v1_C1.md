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
Predicting the answers to questions in Hindi and Tamil.

## Metric
Word-level Jaccard score.

A Python implementation is provided below.

```
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
```

The formula for the overall metric is:
\text{score} = \frac{1}{n} \sum_{i=1}^n \text{jaccard}(gt_i, dt_i)

where:
$n$ = number of documents

$\text{jaccard}$ = the function provided above

$gt_i$ = the ith ground truth

$dt_i$ = the ith prediction

## Submission Format
For each ID in the test set, you must predict the string that best answers the provided question based on the context. Note that the selected text needs to be quoted and complete to work correctly. Include punctuation, etc. The file should contain a header and have the following format:

```
id,PredictionString
8c8ee6504,"1"
3163c22d0,"2 string"
66aae423b,"4 word 6"
722085a7b,"1"
etc.
```

## Dataset 
**All files should be encoded as UTF-8.**

- **train.csv** - the training set, containing context, questions, and answers. Also includes the start character of the answer for disambiguation.
- **test.csv** - the test set, containing context and questions.
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - a unique identifier
- `context` - the text of the Hindi/Tamil sample from which answers should be derived
- `question` - the question, in Hindi/Tamil
- `answer_text` (train only) - the answer to the question (manual annotation) (note: for test, this is what you are attempting to predict)
- `answer_start` (train only) - the starting character in `context` for the answer (determined using substring match during data preparation)
- `language` - whether the text in question is in Tamil or Hindi

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
            description.md (138 lines)
            sample_submission.csv (113 lines)
            sample_submission.csv.zip (950 Bytes)
            test.csv (7173 lines)
            test.csv.zip (648.4 kB)
            train.csv (67723 lines)
            train.csv.zip (5.9 MB)
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
        input/
            description.md (138 lines)
            sample_submission.csv (113 lines)
            sample_submission.csv.zip (950 Bytes)
            test.csv (7173 lines)
            test.csv.zip (648.4 kB)
            train.csv (67723 lines)
            train.csv.zip (5.9 MB)
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
        working/
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
```

-> data/chaii-hindi-and-tamil-question-answering/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> data/chaii-hindi-and-tamil-question-answering/test.csv has 7172 rows and 4 columns.
The columns are: id, context, question, language

-> data/chaii-hindi-and-tamil-question-answering/train.csv has 67722 rows and 6 columns.
The columns are: id, context, question, answer_text, answer_start, language

-> data/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> data/test.csv has 7172 rows and 4 columns.
The columns are: id, context, question, language

-> data/train.csv has 67722 rows and 6 columns.
The columns are: id, context, question, answer_text, answer_start, language

-> input/chaii-hindi-and-tamil-question-answering/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> (stopped after 10 files for performance)

# 5. Target score

0.727562665939331

# 6. Current score

0.00497

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0035) has done: 'The fix corrects the data file path so the test set loads correctly, then replaces the placeholder prediction with a real inference pipeline that uses the loaded XLM‑RoBERTa QA model to extract answer spans from each context‑question pair. This enables generation of a valid `submission.csv` and should raise the Jaccard score toward the target while keeping the original model architecture intact.'
- What this solution (achieved 0.00497) has done: 'I fix the crash caused by loading the XLM‑RoBERTa model (which fails due to protobuf incompatibility) and add a lightweight fallback that uses the training set to directly answer questions when a matching question exists. This keeps the original inference pipeline but avoids the failing model load, and it can dramatically improve the Jaccard score by re‑using known answers. The script now loads `train.csv`, builds a question‑to‑answer map, attempts to load the model (gracefully handling failure), and for each test example either uses the model predictions or the fallback map. The final submission CSV is written with the correct column names.'

# 9. Code solution

## === cell 0
import os
import gc
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering

APEX_INSTALLED = False


class Config:
    model_type = "xlm-roberta-base"
    model_name_or_path = "xlm-roberta-base"
    config_name = "xlm-roberta-base"
    fp16 = True if APEX_INSTALLED else False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2

    tokenizer_name = "xlm-roberta-base"
    max_seq_length = 400
    doc_stride = 135

    epochs = 1
    train_batch_size = 4
    eval_batch_size = 128

    optimizer_type = "AdamW"
    learning_rate = 1e-5
    weight_decay = 1e-2
    epsilon = 1e-8
    max_grad_norm = 1.0

    decay_name = "linear-warmup"
    warmup_ratio = 0.1

    logging_steps = 10

    output_dir = "output"
    seed = 2021


def make_model(args):
    """
    Load a pretrained QA model. If loading fails (e.g., due to protobuf issues),
    return None for config, tokenizer and model so that a fallback can be used.
    """
    try:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path, local_files_only=False
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, local_files_only=False, use_fast=True
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            args.model_name_or_path, config=config
        )
        return config, tokenizer, model
    except Exception as e:
        print(f"Model loading failed ({e}); will use fallback QA mapping.")
        return None, None, None


def optimal_num_of_loader_workers():
    return 0


possible_paths = [
    os.path.join("input", "chaii-hindi-and-tamil-question-answering", "test.csv"),
    os.path.join(
        "/kaggle", "input", "chaii-hindi-and-tamil-question-answering", "test.csv"
    ),
    os.path.join(
        "kaggle", "input", "chaii-hindi-and-tamil-question-answering", "test.csv"
    ),
]
test_path = None
for p in possible_paths:
    if os.path.exists(p):
        test_path = p
        break
if test_path is None:
    raise FileNotFoundError("Test CSV not found in expected locations.")

test = pd.read_csv(test_path)
test["context"] = test["context"].apply(lambda x: " ".join(str(x).split()))
test["question"] = test["question"].apply(lambda x: " ".join(str(x).split()))

train_path = None
possible_train_paths = [
    os.path.join("input", "chaii-hindi-and-tamil-question-answering", "train.csv"),
    os.path.join(
        "/kaggle", "input", "chaii-hindi-and-tamil-question-answering", "train.csv"
    ),
    os.path.join(
        "kaggle", "input", "chaii-hindi-and-tamil-question-answering", "train.csv"
    ),
]
for p in possible_train_paths:
    if os.path.exists(p):
        train_path = p
        break
if train_path is None:
    raise FileNotFoundError("Train CSV not found in expected locations.")

train = pd.read_csv(train_path)
train["question"] = train["question"].apply(lambda x: " ".join(str(x).split()))
train["answer_text"] = train["answer_text"].apply(lambda x: " ".join(str(x).split()))

question_to_answer = {}
for _, row in train.iterrows():
    q = row["question"]
    if q not in question_to_answer:
        question_to_answer[q] = row["answer_text"]

args = Config()
config, tokenizer, model = make_model(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model is not None:
    model.to(device)
    model.eval()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class QADataset(Dataset):
    def __init__(self, df, tokenizer, max_len, doc_stride):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.doc_stride = doc_stride

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = str(row["question"])
        context = str(row["context"])

        inputs = self.tokenizer(
            question,
            context,
            max_length=self.max_len,
            truncation="only_second",
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "example_id": row["id"],
        }


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    example_ids = [item["example_id"] for item in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "example_ids": example_ids,
    }


def predict_batch(batch, model, tokenizer, device):
    """
    If a model is available, use it to predict answer spans.
    Otherwise fall back to the question‑to‑answer dictionary built from training data.
    """
    if model is None:
        preds = []
        for ex_id in batch["example_ids"]:
            q = test.loc[test["id"] == ex_id, "question"].values
            if len(q) == 0:
                preds.append("")
            else:
                preds.append(question_to_answer.get(q[0], ""))
        return preds

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_idx = torch.argmax(start_logits, dim=1)
    end_idx = torch.argmax(end_logits, dim=1)

    predictions = []
    for i in range(start_idx.size(0)):
        s = start_idx[i].item()
        e = end_idx[i].item()
        if s > e:
            e = s
        token_ids = input_ids[i][s : e + 1]
        pred = tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        predictions.append(pred)
    return predictions


if tokenizer is not None and model is not None:
    dataset = QADataset(test, tokenizer, args.max_seq_length, args.doc_stride)
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        sampler=SequentialSampler(dataset),
        num_workers=optimal_num_of_loader_workers(),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    preds = []
    ids = []
    for batch in loader:
        batch_preds = predict_batch(batch, model, tokenizer, device)
        preds.extend(batch_preds)
        ids.extend(batch["example_ids"])
else:
    preds = [question_to_answer.get(q, "") for q in test["question"]]
    ids = test["id"].tolist()

submission_df = pd.DataFrame({"id": ids, "PredictionString": preds})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file written to {submission_path} with {len(submission_df)} rows.")
