# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.9

# 2. Installed packages

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
pyarrow==19.0.1
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

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
!pip install transformers --quiet
!pip install datasets transformers[sentencepiece] --quiet
!pip install "transformers[sentencepiece]" --quiet


## === cell 2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns 
import time
import datetime


## === cell 3
df = pd.read_csv(
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
)
test_csv = pd.read_csv(
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip"
)

test_labels_path = (
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip"
)
if os.path.exists(test_labels_path):
    test_csv_labels = pd.read_csv(test_labels_path)
else:
    test_csv_labels = None

print(df.columns)
print(df.shape)
target_col = df.columns[2:]
feature_col = df.columns[1:2]
df.head()


## === cell 4
df = df.rename(columns={"id": "idx"})


## === cell 5
categories = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']


## === cell 6
import os

import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--quiet", "protobuf==3.20.*"]
)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler, AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_scheduler,
    get_linear_schedule_with_warmup,
)
import pyarrow as pa
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

try:
    import datasets
except Exception as e:
    datasets = None
    print(f"Warning: failed to import `datasets` due to: {type(e).__name__}: {e}")

import random
from sklearn.metrics import classification_report, hamming_loss, accuracy_score


## === cell 7
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)


## === cell 8
train_val_df, test_df = train_test_split(df[["idx", "comment_text"] + categories], test_size=0.2, random_state=seed_value)


## === cell 9
train_df, val_df, = train_test_split(train_val_df[["idx", "comment_text"] + categories], test_size=0.25, random_state=seed_value)


## === cell 10
print(f"Size of train, validation and test are {len(train_df)}, {len(val_df)}, {len(test_df)} respectively.")
print(f"Proportion of train, validation and test are {round(len(train_df)/len(df),2)}, {round(len(val_df)/len(df),2)}, {round(len(test_df)/len(df),2)} respectively.")


## === cell 11
train_df.reset_index(inplace=True)
train_df.drop("index", axis=1, inplace=True)

val_df.reset_index(inplace=True)
val_df.drop("index", axis=1, inplace=True)

test_df.reset_index(inplace=True)
test_df.drop("index", axis=1, inplace=True)


## === cell 12
train_df.head()


## === cell 13
checkpoint = "huawei-noah/TinyBERT_General_4L_312D"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


## === cell 14
train_tokens = tokenizer.batch_encode_plus(
    train_df["comment_text"].tolist(),
    max_length=200,
    padding="max_length",
    truncation=True,
    return_token_type_ids=False,
)

val_tokens = tokenizer.batch_encode_plus(
    val_df["comment_text"].tolist(),
    max_length=200,
    padding="max_length",
    truncation=True,
    return_token_type_ids=False,
)

test_tokens = tokenizer.batch_encode_plus(
    test_df["comment_text"].tolist(),
    max_length=200,
    padding="max_length",
    truncation=True,
    return_token_type_ids=False,
)


## === cell 15
train_seq = torch.tensor(train_tokens['input_ids'])
train_mask = torch.tensor(train_tokens['attention_mask'])
train_y = torch.tensor(np.array(train_df[categories]))  

val_seq = torch.tensor(val_tokens['input_ids'])
val_mask = torch.tensor(val_tokens['attention_mask'])
val_y = torch.tensor(np.array(val_df[categories]))

test_seq = torch.tensor(test_tokens['input_ids'])
test_mask = torch.tensor(test_tokens['attention_mask'])
test_y = torch.tensor(np.array(test_df[categories]))


## === cell 16
train_y


## === cell 17
train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)

val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)

test_data = TensorDataset(test_seq, test_mask, test_y)
test_sampler = SequentialSampler(test_data)


## === cell 18
batch_size = 32

train_dataloader = DataLoader(train_data, 
                              sampler=train_sampler, 
                              batch_size=batch_size,
                              )

val_dataloader = DataLoader(val_data, 
                            sampler = val_sampler, 
                            batch_size=batch_size,
                            )


## === cell 19
for step,batch in enumerate(train_dataloader):
    break
print(batch[0])
print(batch[1])
print(batch[2])


## === cell 20
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 6)





## === cell 21
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


## === cell 22
LEARN_RATE = 3e-5
optimizer = AdamW(model.parameters(),
                  lr = LEARN_RATE, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                  )





## === cell 23
epochs = 3

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # default value
                                            num_training_steps = total_steps
                                            )


## === cell 24
criterion = nn.BCEWithLogitsLoss()




## === cell 25
def accuracy_thresh(y_pred, y_true, thresh:float=0.4, sigmoid:bool=True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: 
        y_pred = y_pred.sigmoid()
    return np.mean(((y_pred>thresh).float()==y_true.float()).float().cpu().numpy(), axis=1).sum()


## === cell 26
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    
    return str(datetime.timedelta(seconds=elapsed_rounded))


## === cell 27
training_stats = []

total_t0 = time.time()

for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_train_loss = 0
    total_train_accuracy = 0

    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        

        output = model(b_input_ids, 
                       attention_mask=b_input_mask, 
                       )
        logits = output.logits
        loss = criterion(logits, b_labels.float())

        total_train_loss += loss.item()
        
        total_train_accuracy += accuracy_thresh(logits,b_labels.float())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

    train_accuracy = total_train_accuracy / len(train_df)
    print("  Accuracy: {0:.5f}".format(train_accuracy))
    
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
        

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in val_dataloader:
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():        

            output = model(b_input_ids, 
                           attention_mask=b_input_mask, 
                           )
            
            logits = output.logits
            loss = criterion(logits, b_labels.float())
            
        total_eval_loss += loss.item()
        
        total_eval_accuracy += accuracy_thresh(logits,b_labels.float())

        logits = logits.detach().cpu().numpy() # maybe irrelevant over here
        label_ids = b_labels.to('cpu').numpy() # maybe irrelevant over here

    val_accuracy = total_eval_accuracy / len(val_df)
    print("  Accuracy: {0:.5f}".format(val_accuracy))

    avg_val_loss = total_eval_loss / len(val_dataloader)
    
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Accur': train_accuracy,
            'Valid. Accur.': val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


## === cell 28
pd.set_option("display.precision", 5)

df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index("epoch")
df_stats


## === cell 29
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4])

plt.show()


## === cell 30
plt.plot(df_stats['Training Accur'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Accur.'], 'g-o', label="Validation")

plt.title("Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4])

plt.show()


## === cell 31
thresh = 0.4
with torch.no_grad():
    outputs = model(test_seq[0:1000].to(device), test_mask[0:1000].to(device))
    pred_probs = torch.sigmoid(outputs.logits)
    pred_probs = pred_probs.cpu().detach().numpy()


## === cell 32
pred_probs


## === cell 33
y_pred = (pred_probs > thresh).astype(int)


## === cell 34
y_pred


## === cell 35
y_true = np.array(test_df[categories])[0:1000]


## === cell 36
def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    
    Take in np.array for y_true and y_pred. E.g.
    y_true = np.array([[0,1,0],
                       [0,1,1],
                       [1,0,1],
                       [0,0,1]])

    y_pred = np.array([[0,1,1],
                       [0,1,1],
                       [0,1,0],
                       [0,0,0]])
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)


## === cell 37
print("accuracy_score:", accuracy_score(y_true, y_pred))
print("Hamming_score:", hamming_score(y_true, y_pred))
print("Hamming_loss:", hamming_loss(y_true, y_pred))


## === cell 38
PATH = "./toxic_tinyBERT_multilabel"
torch.save(model.state_dict(), PATH)


## === cell 39
len(test_csv)


## === cell 40
test_csv.head()


## === cell 41
sub_tokens = tokenizer.batch_encode_plus(test_csv["comment_text"].tolist(),
                                         max_length = 200,
                                         pad_to_max_length=True,
                                         truncation=True,
                                         return_token_type_ids=False
                                         )


## --- ERROR in cell 41, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1332018539.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m# tokenize and encode sequences in the actual test set[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m sub_tokens = tokenizer.batch_encode_plus(test_csv["comment_text"].tolist(),
[0m[1;32m      3[0m                                          [0mmax_length[0m [0;34m=[0m [0;36m200[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m                                          [0mpad_to_max_length[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m                                          [0mtruncation[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py[0m in [0;36mbatch_encode_plus[0;34m(self, batch_text_or_text_pairs, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens, **kwargs)[0m
[1;32m   3142[0m         )
[1;32m   3143[0m [0;34m[0m[0m
[0;32m-> 3144[0;31m         return self._batch_encode_plus(
[0m[1;32m   3145[0m             [0mbatch_text_or_text_pairs[0m[0;34m=[0m[0mbatch_text_or_text_pairs[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3146[0m             [0madd_special_tokens[0m[0;34m=[0m[0madd_special_tokens[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: PreTrainedTokenizerFast._batch_encode_plus() got an unexpected keyword argument 'pad_to_max_length'

## === cell 42
sub_seq = torch.tensor(sub_tokens['input_ids'])
sub_mask = torch.tensor(sub_tokens['attention_mask'])
