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
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

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

# 4. Data file paths

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

# 5. Target score

0.98459

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random, time, datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)




## === cell 1
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip"

df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train shape:", df.shape)
print("Test shape :", test_df.shape)




## === cell 2
categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
df = df.rename(columns={"id": "idx"})  # internal generic id name
test_df = test_df.rename(columns={"id": "idx"})




## === cell 3
train_val_df, holdout_df = train_test_split(
    df[["idx", "comment_text"] + categories],
    test_size=0.2,
    random_state=seed_value,
    stratify=df[categories].sum(axis=1),  # simple stratification proxy
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.25,  # results in 5% of original data as validation
    random_state=seed_value,
)

print(
    f"Sizes -> train: {len(train_df)}, val: {len(val_df)}, holdout: {len(holdout_df)}"
)




## === cell 4
from transformers import AutoTokenizer

checkpoint = "google/mobilebert-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def encode_texts(texts):
    return tokenizer(
        texts,
        max_length=200,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors="pt",
    )


train_enc = encode_texts(train_df["comment_text"].tolist())
val_enc = encode_texts(val_df["comment_text"].tolist())
test_enc = encode_texts(test_df["comment_text"].tolist())




## === cell 5
batch_size = 512

train_seq = train_enc["input_ids"]
train_mask = train_enc["attention_mask"]
train_labels = torch.tensor(train_df[categories].values, dtype=torch.float)

val_seq = val_enc["input_ids"]
val_mask = val_enc["attention_mask"]
val_labels = torch.tensor(val_df[categories].values, dtype=torch.float)

test_seq = test_enc["input_ids"]
test_mask = test_enc["attention_mask"]

train_data = TensorDataset(train_seq, train_mask, train_labels)
val_data = TensorDataset(val_seq, val_mask, val_labels)
test_data = TensorDataset(test_seq, test_mask)  # no labels for final test

worker_cnt = min(8, os.cpu_count() or 4)

train_loader = DataLoader(
    train_data,
    sampler=RandomSampler(train_data),
    batch_size=batch_size,
    num_workers=worker_cnt,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_data,
    sampler=SequentialSampler(val_data),
    batch_size=batch_size,
    num_workers=worker_cnt,
    pin_memory=True,
    persistent_workers=True,
)
test_loader = DataLoader(
    test_data,
    sampler=SequentialSampler(test_data),
    batch_size=batch_size,
    num_workers=worker_cnt,
    pin_memory=True,
    persistent_workers=True,
)




## === cell 6
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # keep nondeterministic cudnn for speed

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=len(categories)
)
model.to(device)

LEARN_RATE = 3e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARN_RATE, eps=1e-8)

epochs = 3
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

criterion = nn.BCEWithLogitsLoss()

scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None


def accuracy_thresh(y_pred, y_true, thresh: float = 0.4):
    """Binary accuracy with a threshold (used for quick monitoring)."""
    y_pred = torch.sigmoid(y_pred) > thresh
    return (y_pred == y_true.byte()).float().mean().item()




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
for epoch in range(epochs):
    print(f"\n=== Epoch {epoch+1}/{epochs} ===")
    model.train()
    epoch_loss, epoch_acc = 0.0, 0.0

    for step, batch in enumerate(train_loader):
        b_input_ids, b_input_mask, b_labels = [
            b.to(device, non_blocking=True) for b in batch
        ]

        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(b_input_ids, attention_mask=b_input_mask)
                loss = criterion(outputs.logits, b_labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            loss = criterion(outputs.logits, b_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        epoch_acc += accuracy_thresh(outputs.logits, b_labels)

    avg_train_loss = epoch_loss / len(train_loader)
    avg_train_acc = epoch_acc / len(train_loader)

    model.eval()
    val_loss, val_acc = 0.0, 0.0
    for batch in val_loader:
        b_input_ids, b_input_mask, b_labels = [
            b.to(device, non_blocking=True) for b in batch
        ]
        with torch.no_grad():
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(b_input_ids, attention_mask=b_input_mask)
                    loss = criterion(outputs.logits, b_labels)
            else:
                outputs = model(b_input_ids, attention_mask=b_input_mask)
                loss = criterion(outputs.logits, b_labels)
        val_loss += loss.item()
        val_acc += accuracy_thresh(outputs.logits, b_labels)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)

    print(f"Train loss: {avg_train_loss:.4f}  acc: {avg_train_acc:.4f}")
    print(f"Val   loss: {avg_val_loss:.4f}  acc: {avg_val_acc:.4f}")




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/2773685891.py in <cell line: 0>()
     12         if scaler:
     13             with torch.cuda.amp.autocast():
---> 14                 outputs = model(b_input_ids, attention_mask=b_input_mask)
     15                 loss = criterion(outputs.logits, b_labels)
     16             scaler.scale(loss).backward()

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

/usr/local/lib/python3.11/dist-packages/transformers/models/mobilebert/modeling_mobilebert.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
   1155         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
   1156 
-> 1157         outputs = self.mobilebert(
   1158             input_ids,
   1159             attention_mask=attention_mask,

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

/usr/local/lib/python3.11/dist-packages/transformers/models/mobilebert/modeling_mobilebert.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, output_hidden_states, output_attentions, return_dict)
    788             input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
    789         )
--> 790         encoder_outputs = self.encoder(
    791             embedding_output,
    792             attention_mask=extended_attention_mask,

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

/usr/local/lib/python3.11/dist-packages/transformers/models/mobilebert/modeling_mobilebert.py in forward(self, hidden_states, attention_mask, head_mask, output_attentions, output_hidden_states, return_dict)
    551                 all_hidden_states = all_hidden_states + (hidden_states,)
    552 
--> 553             layer_outputs = layer_module(
    554                 hidden_states,
    555                 attention_mask,

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

/usr/local/lib/python3.11/dist-packages/transformers/models/mobilebert/modeling_mobilebert.py in forward(self, hidden_states, attention_mask, head_mask, output_attentions)
    494             query_tensor, key_tensor, value_tensor, layer_input = [hidden_states] * 4
    495 
--> 496         self_attention_outputs = self.attention(
    497             query_tensor,
    498             key_tensor,

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

/usr/local/lib/python3.11/dist-packages/transformers/models/mobilebert/modeling_mobilebert.py in forward(self, query_tensor, key_tensor, value_tensor, layer_input, attention_mask, head_mask, output_attentions)
    328         output_attentions: Optional[bool] = None,
    329     ) -> tuple[torch.Tensor]:
--> 330         self_outputs = self.self(
    331             query_tensor,
    332             key_tensor,

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

/usr/local/lib/python3.11/dist-packages/transformers/models/mobilebert/modeling_mobilebert.py in forward(self, query_tensor, key_tensor, value_tensor, attention_mask, head_mask, output_attentions)
    264         # This is actually dropping out entire tokens to attend to, which might
    265         # seem a bit unusual, but is taken from the original Transformer paper.
--> 266         attention_probs = self.dropout(attention_probs)
    267         # Mask heads if we want to
    268         if head_mask is not None:

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/dropout.py in forward(self, input)
     68 
     69     def forward(self, input: Tensor) -> Tensor:
---> 70         return F.dropout(input, self.p, self.training, self.inplace)
     71 
     72 

/usr/local/lib/python3.11/dist-packages/torch/nn/functional.py in dropout(input, p, training, inplace)
   1423         raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
   1424     return (
-> 1425         _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)
   1426     )
   1427 

OutOfMemoryError: CUDA out of memory. Tried to allocate 314.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 2.88 MiB is free. Process 3725554 has 47.52 GiB memory in use. Of the allocated memory 46.02 GiB is allocated by PyTorch, and 1.18 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 8
from tqdm.auto import tqdm

model.eval()
all_predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        b_input_ids, b_input_mask = [b.to(device, non_blocking=True) for b in batch]
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(b_input_ids, attention_mask=b_input_mask)
        else:
            outputs = model(b_input_ids, attention_mask=b_input_mask)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        all_predictions.append(probs)

pred_array = np.concatenate(all_predictions, axis=0)

submission = pd.DataFrame(pred_array, columns=categories)
submission.insert(0, "id", test_df["idx"].values)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/2952887642.py in <cell line: 0>()
      9         if scaler:
     10             with torch.cuda.amp.autocast():
---> 11                 outputs = model(b_input_ids, attention_mask=b_input_mask)
     12         else:
     13             outputs = model(b_input_ids, attention_mask=b_input_mask)

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

/usr/local/lib/python3.11/dist-packages/transformers/models/mobilebert/modeling_mobilebert.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
   1155         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
   1156 
-> 1157         outputs = self.mobilebert(
   1158             input_ids,
   1159             attention_mask=attention_mask,

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

/usr/local/lib/python3.11/dist-packages/transformers/models/mobilebert/modeling_mobilebert.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, output_hidden_states, output_attentions, return_dict)
    772             attention_mask = torch.ones(input_shape, device=device)
    773         if token_type_ids is None:
--> 774             token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    775 
    776         # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]

OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 896.00 KiB is free. Process 3725554 has 47.52 GiB memory in use. Of the allocated memory 45.95 GiB is allocated by PyTorch, and 1.26 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 9
print("Submission shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2536056748.py in <cell line: 0>()
----> 1 print("Submission shape:", submission.shape)
      2 print(submission.head())

NameError: name 'submission' is not defined
