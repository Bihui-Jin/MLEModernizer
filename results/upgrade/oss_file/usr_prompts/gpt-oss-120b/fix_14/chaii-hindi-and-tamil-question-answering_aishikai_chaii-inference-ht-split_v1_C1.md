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

0.7240094542503357

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00699) has done: 'I fix the tokenizer call by removing the unsupported argument, make checkpoint loading optional so the script can run with the pretrained model, and ensure the post‑processing function has access to the tokenizer. These changes resolve the runtime errors and let the notebook generate a valid `submission.csv` file.'
- What this solution (achieved 0.57411) has done: 'I fixed the runtime crash by simplifying the prediction routine: the model is run on the available device without CUDA‑specific calls or AMP, and the outputs are accessed via the proper `start_logits` / `end_logits` fields. This removes the protobuf/AMP error, ensures `fin_preds` is defined, and lets the notebook write a valid `submission.csv` file. No core‑logic changes were made, so the scoring behavior remains unchanged.'
- What this solution (achieved 0.57411) has done: 'The fix handles the crash caused by passing a `None` config name to `AutoConfig.from_pretrained`. We now only load a config when a valid name/path is provided and fall back to the model name otherwise. This prevents the protobuf `MessageFactory` error and lets the script run end‑to‑end, producing a proper `submission.csv` while keeping the core logic unchanged.'
- What this solution (achieved 0.57411) has done: 'Implemented fixes to resolve runtime errors and ensure a valid submission is generated. Added a proper model loading routine in the prediction function, removed the undefined `Make_Model` call, and kept the rest of the workflow unchanged so predictions are produced and written to `submission.csv`. This correction enables end‑to‑end execution and creates the required output file.'
- What this solution (achieved 0.56994) has done: 'Implemented modest, score‑directed tweaks while keeping the original pipeline intact:

1. **Configration** – extended `max_seq_length` to 512 and tightened the overlap (`doc_stride` = 64) so the model sees more context without altering the model architecture.  
2. **Postprocess_qa_predictions** – added a null‑answer check: if the best span score does not exceed the highest null score, the prediction is set to an empty string. Also raised the defaults for `n_best_size` and `max_answer_length` to capture slightly longer answers.  
3. **Prediction call** – supplied the new `n_best_size` and `max_answer_length` arguments.

These minimal changes are expected to raise the Jaccard score toward the target while preserving all core logic.'
- What this solution (achieved 0.53274) has done: 'I slightly adjust the retrieval parameters to give the model more context when splitting long passages (increase `doc_stride`), and I allow the post‑processing step to consider a larger set of candidate spans and longer answers (increase `n_best_size` and `max_answer_length`). These changes keep the overall pipeline unchanged while giving the model a better chance to locate the correct answer, which should raise the Jaccard score toward the target.'

# 9. Code solution

## === cell 0
import collections
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from tqdm.auto import tqdm


class Configration:
    model_type = "xlm_roberta"
    XLMR_name_or_path = "deepset/xlm-roberta-base-squad2"
    MURIL_name_or_path = "google/muril-base-cased"
    MPNET2_name_or_path = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
    MPNET3_name_or_path = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

    XLMR_config_name = None
    MURIL_config_name = None
    MPNET2_config_name = None
    MPNET3_config_name = None

    fp16 = False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2

    XLMR_tokenizer_name = None
    MURIL_tokenizer_name = None
    MPNET2_tokenizer_name = None
    MPNET3_tokenizer_name = None

    max_seq_length = 512
    doc_stride = 64

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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_root = Path("./data/chaii-hindi-and-tamil-question-answering")
train_path = data_root / "train.csv"
test_path = data_root / "test.csv"
sample_sub_path = data_root / "sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

model_name = Configration.XLMR_name_or_path
tokenizer = AutoTokenizer.from_pretrained(
    Configration.XLMR_tokenizer_name or model_name,
    use_fast=True,
)
model = AutoModelForQuestionAnswering.from_pretrained(
    Configration.XLMR_name_or_path,
    torch_dtype=torch.float16 if Configration.fp16 else torch.float32,
)

device = 0 if torch.cuda.is_available() else -1
qa_pipe = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_seq_len=Configration.max_seq_length,
    doc_stride=Configration.doc_stride,
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2461574757.py in <cell line: 0>()
      6 
      7 # Load data
----> 8 train_df = pd.read_csv(train_path)
      9 test_df = pd.read_csv(test_path)
     10 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'data/chaii-hindi-and-tamil-question-answering/train.csv'

## === cell 2
def get_predictions(df: pd.DataFrame) -> dict:
    """
    Run the QA pipeline on each test row and return a dict:
    {example_id: predicted_string}
    """
    preds = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        try:
            output = qa_pipe(
                question=row["question"],
                context=row["context"],
                top_k=1,
                handle_impossible_answer=False,
            )
            answer = (
                output["answer"] if isinstance(output, dict) else output[0]["answer"]
            )
        except Exception as e:
            answer = ""
        preds[row["id"]] = answer
    return preds


fin_preds = get_predictions(test_df)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1302791719.py in <cell line: 0>()
     26 
     27 # Generate predictions for the test set
---> 28 fin_preds = get_predictions(test_df)
     29 

NameError: name 'test_df' is not defined

## === cell 3
submission = pd.DataFrame(
    {
        "id": list(fin_preds.keys()),
        "PredictionString": list(fin_preds.values()),
    }
)

submission = submission[["id", "PredictionString"]]

output_path = Path("./submission.csv")
submission.to_csv(
    output_path,
    index=False,
    encoding="utf-8",
    quoting=pd.io.common.csv.QUOTE_ALL,
)

print(f"Submission file written to {output_path.resolve()}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1250606411.py in <cell line: 0>()
      2 submission = pd.DataFrame(
      3     {
----> 4         "id": list(fin_preds.keys()),
      5         "PredictionString": list(fin_preds.values()),
      6     }

NameError: name 'fin_preds' is not defined
