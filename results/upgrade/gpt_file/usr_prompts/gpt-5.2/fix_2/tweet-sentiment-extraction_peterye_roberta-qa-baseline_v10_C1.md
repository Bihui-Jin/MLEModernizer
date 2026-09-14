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

0.4975326359272003

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")



## === cell 2
pass



## === cell 3
from transformers import RobertaForQuestionAnswering, RobertaTokenizer

output_dir = "/kaggle/input/roberta-sentiment-extraction"

model = RobertaForQuestionAnswering.from_pretrained(output_dir)
tokenizer = RobertaTokenizer.from_pretrained(output_dir)

model.to(device)
model.eval()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler

max_len = 110

df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
print("Number of test sentences: {:,}\n".format(df_test.shape[0]))
df_test["id_num"] = np.arange(len(df_test))

input_ids = []
attention_masks = []
token_type_ids = []
textID = []

for i in range(len(df_test["text"])):
    question = (
        "what portion of texts best reflect " + df_test["sentiment"][i] + "sentiment"
    )
    text = df_test["text"][i]

    encoded_dict = tokenizer.encode_plus(
        text,
        question,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=True,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids.append(encoded_dict["input_ids"])
    attention_masks.append(encoded_dict["attention_mask"])
    token_type_ids.append(
        encoded_dict.get("token_type_ids", torch.zeros_like(encoded_dict["input_ids"]))
    )
    textID.append(df_test["id_num"][i])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
token_type_ids = torch.cat(token_type_ids, dim=0)
textID = torch.tensor([int(x) for x in textID], dtype=torch.long)

batch_size = 32
prediction_data = TensorDataset(input_ids, attention_masks, token_type_ids, textID)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(
    prediction_data, sampler=prediction_sampler, batch_size=batch_size
)

print("Prepared prediction dataloader with", len(prediction_data), "examples.")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2549863119.py in <cell line: 0>()
     23     text = df_test["text"][i]
     24 
---> 25     encoded_dict = tokenizer.encode_plus(
     26         text,
     27         question,

NameError: name 'tokenizer' is not defined

## === cell 5
def id_to_word(answer_start, answer_end, input_ids_1d):
    s = int(answer_start)
    e = int(answer_end)
    if s < 0:
        s = 0
    if e < 0:
        e = 0
    if e < s:
        e = s

    idx = input_ids_1d[s + 1 : e + 1]
    answer = tokenizer.decode(idx, skip_special_tokens=True)
    return str(answer).strip()


def get_start_end(start_score, end_score):
    starts = np.zeros(len(start_score))
    ends = np.zeros(len(start_score))
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




## === cell 6
predictions = []
text_ids = []

for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_input_type_ids, b_textID = batch

    with torch.no_grad():
        out = model(b_input_ids, attention_mask=b_input_mask)
        start_score = out.start_logits.detach().cpu().numpy()
        end_score = out.end_logits.detach().cpu().numpy()

        b_input_ids_np = b_input_ids.detach().cpu().numpy()
        textID_np = b_textID.detach().cpu().numpy()

        start_pred, end_pred = get_start_end(start_score, end_score)

        for i in range(len(b_input_ids_np)):
            text_ids.append(int(textID_np[i]))
            predictions.append(
                id_to_word(start_pred[i], end_pred[i], b_input_ids_np[i])
            )

print("Generated predictions:", len(predictions))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/549791740.py in <cell line: 0>()
      2 text_ids = []
      3 
----> 4 for batch in prediction_dataloader:
      5     batch = tuple(t.to(device) for t in batch)
      6     b_input_ids, b_input_mask, b_input_type_ids, b_textID = batch

NameError: name 'prediction_dataloader' is not defined

## === cell 7
sub = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
sub_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
sub_test["id_num"] = np.arange(len(sub_test))

pred_df = pd.DataFrame({"id_num": text_ids, "selected_text": predictions})
pred_df = pred_df.sort_values("id_num").reset_index(drop=True)

if len(pred_df) != len(sub):
    raise ValueError(
        f"Prediction length ({len(pred_df)}) does not match submission length ({len(sub)})."
    )

sub["selected_text"] = pred_df["selected_text"].values

sub["selected_text"] = sub["selected_text"].apply(
    lambda x: x.replace("!!!!", "!") if len(str(x).split()) == 1 else x
)
sub["selected_text"] = sub["selected_text"].apply(
    lambda x: x.replace("..", ".") if len(str(x).split()) == 1 else x
)
sub["selected_text"] = sub["selected_text"].apply(
    lambda x: x.replace("...", ".") if len(str(x).split()) == 1 else x
)

test_texts = sub_test["text"].tolist()
sub["selected_text"] = [
    (sel if isinstance(sel, str) and sel.strip() != "" else test_texts[i])
    for i, sel in enumerate(sub["selected_text"].tolist())
]

sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3524527342.py in <cell line: 0>()
      8 
      9 if len(pred_df) != len(sub):
---> 10     raise ValueError(
     11         f"Prediction length ({len(pred_df)}) does not match submission length ({len(sub)})."
     12     )

ValueError: Prediction length (0) does not match submission length (2749).

## === cell 8
print(sub.head())
print(sub.isna().sum())
print("submission shape:", sub.shape)
print("Unique textIDs:", sub["textID"].nunique())
