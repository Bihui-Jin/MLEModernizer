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
Predict the word or phrase from tweets that exemplifies the labelled sentiment.

## Metric
Word-level Jaccard score.

## Submission Format
For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

## Dataset
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        input/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        working/
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
```

-> data/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> data/tweet-sentiment-extraction/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/tweet-sentiment-extraction/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/tweet-sentiment-extraction/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> input/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> (stopped after 10 files for performance)

# 5. Target score

0.6553906798362732

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I remove the TensorFlow GPU check that is crashing due to an incompatibility in the Kaggle image (it’s not needed since the rest of the notebook uses PyTorch). I fix the Transformers import error by using `torch.optim.AdamW` (Transformers v4.53 removed `AdamW` from the top-level) and remove the unnecessary `pip install transformers` cell to avoid time/network issues. I also make the Roberta QA forward call compatible with the current Transformers outputs, and adjust tokenization for RoBERTa (no `token_type_ids`, and replace deprecated `pad_to_max_length` with `padding='max_length'`). Finally, I ensure predictions align to `sample_submission.csv` order by using `textID` mapping, guaranteeing a correctly sized `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
print("Skipping TensorFlow GPU check (not required for this PyTorch solution).")



## === cell 2
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")



## === cell 3
import transformers

print("transformers version:", transformers.__version__)



## === cell 4
from transformers import RobertaForQuestionAnswering, RobertaTokenizer

output_dir = "/kaggle/input/roberta-2"

model = RobertaForQuestionAnswering.from_pretrained(output_dir)
tokenizer = RobertaTokenizer.from_pretrained(output_dir)

model.to(device)
model.eval()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler

max_len = 192

df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
print("Number of test sentences: {:,}\n".format(df_test.shape[0]))
df_test["id_num"] = np.arange(len(df_test), dtype=np.int64)

input_ids = []
attention_masks = []
textID = []

for i in range(len(df_test["text"])):
    question = (
        "what portion of texts best reflect "
        + str(df_test["sentiment"][i])
        + " sentiment"
    )
    text = str(df_test["text"][i])

    encoded_dict = tokenizer.encode_plus(
        question,
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids.append(encoded_dict["input_ids"])
    attention_masks.append(encoded_dict["attention_mask"])
    textID.append(int(df_test["id_num"][i]))

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
textID = torch.tensor(textID, dtype=torch.long)

batch_size = 32
prediction_data = TensorDataset(input_ids, attention_masks, textID)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(
    prediction_data, sampler=prediction_sampler, batch_size=batch_size
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1601103653.py in <cell line: 0>()
     23     text = str(df_test["text"][i])
     24 
---> 25     encoded_dict = tokenizer.encode_plus(
     26         question,
     27         text,

NameError: name 'tokenizer' is not defined

## === cell 6
def get_start_end(start_score, end_score):
    starts = np.zeros(len(start_score), dtype=np.int64)
    ends = np.zeros(len(start_score), dtype=np.int64)
    for i in range(len(start_score)):
        total_score = []
        arg = []
        for a in range(len(start_score[i])):
            for b in range(a, len(end_score[i])):
                total_score.append(start_score[i][a] + end_score[i][b])
                arg.append((a, b))

        total_score = torch.tensor(total_score)
        max_idx = torch.argmax(total_score).item()
        start, end = arg[max_idx]
        starts[i] = start
        ends[i] = end
    return starts, ends




## === cell 7
predictions = []
pred_idnums = []

for batch in prediction_dataloader:
    b_input_ids, b_input_mask, b_textID = (t.to(device) for t in batch)

    with torch.no_grad():
        outputs = model(
            input_ids=b_input_ids,
            attention_mask=b_input_mask,
        )
        start_score = outputs.start_logits
        end_score = outputs.end_logits

        start_score_pred = start_score.detach().cpu().numpy()
        end_score_pred = end_score.detach().cpu().numpy()

        b_input_ids_cpu = b_input_ids.detach().cpu().numpy()
        b_textID_cpu = b_textID.detach().cpu().numpy()

        start_pred, end_pred = get_start_end(start_score_pred, end_score_pred)

        for i in range(len(b_input_ids_cpu)):
            s = int(start_pred[i])
            e = int(end_pred[i])
            if e < s:
                e = s
            predicted_text = tokenizer.decode(
                b_input_ids_cpu[i][s:e], skip_special_tokens=True
            ).strip()
            predictions.append(predicted_text)
            pred_idnums.append(int(b_textID_cpu[i]))



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4085805772.py in <cell line: 0>()
      4 # BUGFIX: Transformers QA models now return a ModelOutput; access .start_logits/.end_logits.
      5 # Also, RoBERTa forward should not receive token_type_ids.
----> 6 for batch in prediction_dataloader:
      7     b_input_ids, b_input_mask, b_textID = (t.to(device) for t in batch)
      8 

NameError: name 'prediction_dataloader' is not defined

## === cell 8
sub = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")

pred_map = {i: p for i, p in zip(pred_idnums, predictions)}

ordered_preds = [pred_map.get(int(i), "") for i in df_test["id_num"].values]

n = len(sub)
if len(ordered_preds) != n:
    if len(ordered_preds) > n:
        ordered_preds = ordered_preds[:n]
    else:
        ordered_preds = ordered_preds + [""] * (n - len(ordered_preds))

sub["selected_text"] = ordered_preds
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv with shape:", sub.shape)



## === cell 9
sub.head(20)
