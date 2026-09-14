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

0.7179869413375854

# 6. Current score

0.57068

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00306) has done: 'I fixed the undefined variables and missing imports, replaced the broken custom model code with HuggingFace’s question‑answering pipeline (which works out‑of‑the‑box for the provided data), and added a straightforward loop that builds the required `id,PredictionString` submission file without altering the overall modelling intent.'
- What this solution (achieved 0.52515) has done: 'The fix adds an environment variable before any library imports to avoid the protobuf `MessageFactory` error, and switches the model to a multilingual QA checkpoint that is already fine‑tuned (`deepset/xlm-roberta-large-squad2`). This change resolves the runtime crash and provides much better answer predictions, moving the Jaccard score toward the target while keeping the original pipeline logic unchanged.'
- What this solution (achieved 0.54152) has done: 'I fix the runtime error by keeping the protobuf environment variable and then improve the prediction quality by configuring the QA pipeline with the intended `max_seq_length` and `doc_stride` (the values defined in the Configration). These settings let the model see a larger portion of each context, which generally raises the Jaccard score while preserving the original architecture and workflow.'
- What this solution (achieved 0.57068) has done: 'The changes switch inference to GPU (when available), batch the 7 k test examples into one pipeline call, and enable cuDNN benchmarking for faster CUDA kernels. Using the pipeline’s native list‑input and `batch_size` removes the Python‑level loop and repetitive tokenization, keeping the exact same model and answer‑extraction logic while staying within the 600 s limit.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import gc
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch


class Configration:
    XLMR_name_or_path = "deepset/xlm-roberta-large-squad2"
    max_seq_length = 512  # larger context window
    doc_stride = 64  # smaller stride for better overlap
    epochs = 1
    train_batch_size = 4
    eval_batch_size = 128
    fp16 = False
    gradient_accumulation_steps = 2
    output_dir = "output"
    seed = 2021


np.random.seed(Configration.seed)
torch.manual_seed(Configration.seed)
torch.backends.cudnn.benchmark = True  # faster CUDA kernels when input size is fixed



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_dir = "/kaggle/input/chaii-hindi-and-tamil-question-answering"
train_path = os.path.join(data_dir, "train.csv")
test_path = os.path.join(data_dir, "test.csv")
sample_sub_path = os.path.join(data_dir, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)



## === cell 2
tokenizer = AutoTokenizer.from_pretrained(Configration.XLMR_name_or_path)
model = AutoModelForQuestionAnswering.from_pretrained(Configration.XLMR_name_or_path)

device_id = 0 if torch.cuda.is_available() else -1  # use GPU if possible
qa_pipeline = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    device=device_id,
    max_seq_length=Configration.max_seq_length,
    doc_stride=Configration.doc_stride,
)



## === cell 3
inputs = [
    {"question": str(q), "context": str(c)}
    for q, c in zip(test_df["question"], test_df["context"])
]

batch_size = 32
results = qa_pipeline(inputs, top_k=1, batch_size=batch_size)

predictions = []
for idx, result in enumerate(results):
    answer = result["answer"] if isinstance(result, dict) else result[0]["answer"]
    predictions.append({"id": test_df.iloc[idx]["id"], "PredictionString": answer})

submission_df = pd.DataFrame(predictions)[["id", "PredictionString"]]

output_path = os.path.join("/kaggle/working", "submission.csv")
submission_df.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
