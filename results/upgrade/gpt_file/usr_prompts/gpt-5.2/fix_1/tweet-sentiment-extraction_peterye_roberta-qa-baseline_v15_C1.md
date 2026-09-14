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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import tensorflow as tf

device_name = tf.test.gpu_device_name()

if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
import torch

if torch.cuda.is_available():    

    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

## === cell 4
!pip install transformers

## === cell 5
from transformers import BertForQuestionAnswering, AdamW, BertConfig, RobertaForQuestionAnswering, RobertaTokenizer, BertTokenizer


output_dir = '/kaggle/input/roberta-2'





model = RobertaForQuestionAnswering.from_pretrained(output_dir)
tokenizer = RobertaTokenizer.from_pretrained(output_dir)

model.to(device)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1614913494.py in <cell line: 0>()
----> 1 from transformers import BertForQuestionAnswering, AdamW, BertConfig, RobertaForQuestionAnswering, RobertaTokenizer, BertTokenizer
      2 
      3 
      4 output_dir = '/kaggle/input/roberta-2'
      5 

ImportError: cannot import name 'AdamW' from 'transformers' (/usr/local/lib/python3.11/dist-packages/transformers/__init__.py)

## === cell 6
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

max_len = 192


df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")

print('Number of test sentences: {:,}\n'.format(df_test.shape[0]))
df_test['id_num'] = np.arange(len(df_test))

input_ids = []
attention_masks = []
token_type_ids = []
textID = []

for i in range(len(df_test['text'])):
    question = 'what portion of texts best reflect ' + df_test['sentiment'][i] + 'sentiment'
    text = df_test['text'][i]
    encoded_dict = tokenizer.encode_plus(
                        question,
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_len,           # Pad & truncate all sentences.
                        return_token_type_ids = True,
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    input_ids.append(encoded_dict['input_ids'])
    
    attention_masks.append(encoded_dict['attention_mask'])

    token_type_ids.append(encoded_dict['token_type_ids'])

    textID.append(df_test['id_num'][i])


input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
token_type_ids = torch.cat(token_type_ids, dim=0)
textID = torch.tensor([int(x) for x in textID])


batch_size = 32  


prediction_data = TensorDataset(input_ids, attention_masks,  token_type_ids, textID)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2051067397.py in <cell line: 0>()
     31     #   (6) Create attention masks for [PAD] tokens.
     32     text = df_test['text'][i]
---> 33     encoded_dict = tokenizer.encode_plus(
     34                         question,
     35                         text,                      # Sentence to encode.

NameError: name 'tokenizer' is not defined

## === cell 7
def id_to_word(answer_start, answer_end, input_ids):
    idx = input_ids[int(answer_start)+1:int(answer_end)+1]
    answer = tokenizer.decode(idx)
    return str(answer)


def get_start_end(start_score, end_score):
    starts = np.zeros(len(start_score))
    ends = np.zeros(len(start_score))
    for i in range(len(start_score)):
        total_score = []
        arg = []
        for a in range(len(start_score[i])):
            for b in range(a, len(end_score[i])):
                total_score.append( start_score[i][a] + end_score[i][b])
                arg.append((a,b))

        total_score = torch.tensor(total_score)

        max_idx = torch.argmax(total_score)
        start, end = arg[max_idx]
        starts[i] = start
        ends[i] = end
    return starts, ends

## === cell 8
predictions  = []
textID = []
for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
  
    b_input_ids, b_input_mask, b_input_type_ids, b_textID = batch
  
    with torch.no_grad():
        start_score, end_score, hidden_state = model(b_input_ids, 
                                   token_type_ids = b_input_type_ids,
                                   attention_mask=b_input_mask
                                   )

        start_score_pred = start_score.detach().cpu().numpy()
        end_score_pred = end_score.detach().cpu().numpy()

        b_input_ids = b_input_ids.to('cpu').numpy()
        b_textID = b_textID.detach().cpu().numpy()
        start_pred, end_pred = get_start_end(start_score_pred, end_score_pred)
        for i in range(len(b_input_ids)):
            predicted_text = tokenizer.decode(b_input_ids[i][int(start_pred[i]):int(end_pred[i])]) 
            predictions.append(predicted_text)
            textID.append(b_textID[i])



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/929025495.py in <cell line: 0>()
      2 textID = []
      3 # Predict
----> 4 for batch in prediction_dataloader:
      5   # Add batch to GPU
      6     batch = tuple(t.to(device) for t in batch)

NameError: name 'prediction_dataloader' is not defined

## === cell 9
df = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
df['selected_text'] = predictions

df.to_csv('/kaggle/working/submission.csv', index=False)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/141789377.py in <cell line: 0>()
      1 df = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
----> 2 df['selected_text'] = predictions
      3 # df['selected_text'] = df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)
      4 # df['selected_text'] = df['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)
      5 # df['selected_text'] = df['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (0) does not match length of index (2749)

## === cell 10
df.head(20)
