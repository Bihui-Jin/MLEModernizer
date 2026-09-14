# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.98446

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime



## === cell 2
TRAIN_PATH = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
TEST_PATH = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip"
SAMPLE_SUB_PATH = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip"

df = pd.read_csv(TRAIN_PATH)
test_csv = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print(df.columns)
print(df.shape)
target_col = df.columns[2:]
feature_col = df.columns[1:2]
df.head()



## === cell 3
for col in target_col:
    print(f"The unique value for {col} are {df[col].unique()}")



## === cell 4
print(target_col)
print(feature_col)



## === cell 5
df.dtypes



## === cell 6
categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
sns.barplot(x=categories, y=df[categories].sum())
plt.show()



## === cell 7
condition = (
    (df["toxic"] == 1)
    | (df["severe_toxic"] == 1)
    | (df["obscene"] == 1)
    | (df["threat"] == 1)
    | (df["insult"] == 1)
    | (df["identity_hate"] == 1)
)
df.loc[condition, "y"] = 1
df.loc[~condition, "y"] = 0
df["y"] = df["y"].astype(int)



## === cell 8
agg_df = df["y"].value_counts(normalize=True).rename("Proportion").reset_index()
agg_df.columns = ["y", "Proportion"]



## === cell 9
splot = sns.barplot(x="y", y="Proportion", data=agg_df)
plt.ylim(0, 1)
plt.title("Proportion of Non-Toxic/Toxic Post")
for p in splot.patches:
    splot.annotate(
        format(p.get_height(), ".2f"),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 9),
        textcoords="offset points",
    )



## === cell 10
df["len"] = df["comment_text"].fillna("").str.split().str.len()



## === cell 11
print(f"Average length is {df['len'].mean()} while max length is {df['len'].max()}.")
df["len"].hist(bins=20)
plt.show()



## === cell 12
sns.histplot(data=df, x="len", hue="y", bins=50, alpha=0.2)



## === cell 13
sns.histplot(
    data=df,
    x="len",
    hue="y",
    log_scale=True,
    element="step",
    fill=False,
    cumulative=True,
    stat="density",
    common_norm=False,
)



## === cell 14
df = df.rename(columns={"id": "idx"})



## === cell 15
df.head()



## === cell 16
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random
import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler



## === cell 17
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)



## === cell 18
train_val_df, test_df = train_test_split(
    df[["idx", "comment_text"] + categories], test_size=0.2, random_state=seed_value
)



## === cell 19
train_df, val_df = train_test_split(
    train_val_df[["idx", "comment_text"] + categories],
    test_size=0.25,
    random_state=seed_value,
)



## === cell 20
print(
    f"Size of train, validation and test are {len(train_df)}, {len(val_df)}, {len(test_df)} respectively."
)
print(
    f"Proportion of train, validation and test are {round(len(train_df)/len(df),2)}, {round(len(val_df)/len(df),2)}, {round(len(test_df)/len(df),2)} respectively."
)



## === cell 21
train_df.reset_index(inplace=True)
train_df.drop("index", axis=1, inplace=True)

val_df.reset_index(inplace=True)
val_df.drop("index", axis=1, inplace=True)

test_df.reset_index(inplace=True)
test_df.drop("index", axis=1, inplace=True)



## === cell 22
train_df.head()



## === cell 23
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)



## === cell 24
MAX_LEN = 200

train_tokens = tokenizer(
    train_df["comment_text"].fillna("").tolist(),
    max_length=MAX_LEN,
    padding="max_length",
    truncation=True,
    return_token_type_ids=False,
)

val_tokens = tokenizer(
    val_df["comment_text"].fillna("").tolist(),
    max_length=MAX_LEN,
    padding="max_length",
    truncation=True,
    return_token_type_ids=False,
)

test_tokens = tokenizer(
    test_df["comment_text"].fillna("").tolist(),
    max_length=MAX_LEN,
    padding="max_length",
    truncation=True,
    return_token_type_ids=False,
)



## === cell 25
train_seq = torch.tensor(train_tokens["input_ids"], dtype=torch.long)
train_mask = torch.tensor(train_tokens["attention_mask"], dtype=torch.long)
train_y = torch.tensor(np.array(train_df[categories]), dtype=torch.float32)

val_seq = torch.tensor(val_tokens["input_ids"], dtype=torch.long)
val_mask = torch.tensor(val_tokens["attention_mask"], dtype=torch.long)
val_y = torch.tensor(np.array(val_df[categories]), dtype=torch.float32)

test_seq = torch.tensor(test_tokens["input_ids"], dtype=torch.long)
test_mask = torch.tensor(test_tokens["attention_mask"], dtype=torch.long)
test_y = torch.tensor(np.array(test_df[categories]), dtype=torch.float32)



## === cell 26
train_y



## === cell 27
train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)

val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)

test_data = TensorDataset(test_seq, test_mask, test_y)
test_sampler = SequentialSampler(test_data)



## === cell 28
batch_size = 32

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)



## === cell 29
for step, batch in enumerate(train_dataloader):
    break
print(batch[0].shape)
print(batch[1].shape)
print(batch[2].shape)



## === cell 30
try:
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=6)
except Exception as e:
    raise RuntimeError(
        "Failed to load transformer model. This is often due to protobuf/transformers "
        "binary incompat in the Kaggle image. The script sets "
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python to mitigate this, but it may "
        "still fail in some environments."
    ) from e

try:
    model.config.problem_type = "multi_label_classification"
except Exception:
    pass



## === cell 31
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)



## === cell 32
LEARN_RATE = 3e-5
optimizer = AdamW(model.parameters(), lr=LEARN_RATE, eps=1e-8)



## === cell 33
epochs = 3
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps,
)



## === cell 34
criterion = nn.BCEWithLogitsLoss()




## === cell 35
def accuracy_thresh(y_pred, y_true, thresh: float = 0.4, sigmoid: bool = True):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    return np.mean(
        ((y_pred > thresh).float() == y_true.float()).float().cpu().numpy(), axis=1
    ).sum()




## === cell 36
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))




## === cell 37
training_stats = []
total_t0 = time.time()

for epoch_i in range(0, epochs):
    print("")
    print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
    print("Training...")

    t0 = time.time()
    total_train_loss = 0.0
    total_train_accuracy = 0.0

    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 400 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print(
                "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                    step, len(train_dataloader), elapsed
                )
            )

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        output = model(b_input_ids, attention_mask=b_input_mask)
        logits = output.logits
        loss = criterion(logits, b_labels)

        total_train_loss += loss.item()
        total_train_accuracy += accuracy_thresh(logits, b_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    train_accuracy = total_train_accuracy / len(train_df)
    print("  Accuracy: {0:.5f}".format(train_accuracy))

    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.4f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))

    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()

    total_eval_accuracy = 0.0
    total_eval_loss = 0.0

    for batch in val_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            output = model(b_input_ids, attention_mask=b_input_mask)
            logits = output.logits
            loss = criterion(logits, b_labels)

        total_eval_loss += loss.item()
        total_eval_accuracy += accuracy_thresh(logits, b_labels)

    val_accuracy = total_eval_accuracy / len(val_df)
    print("  Accuracy: {0:.5f}".format(val_accuracy))

    avg_val_loss = total_eval_loss / len(val_dataloader)
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.4f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            "epoch": epoch_i + 1,
            "Training Loss": avg_train_loss,
            "Valid. Loss": avg_val_loss,
            "Training Accur": train_accuracy,
            "Valid. Accur.": val_accuracy,
            "Training Time": training_time,
            "Validation Time": validation_time,
        }
    )

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))



## === cell 38
pd.set_option("display.precision", 5)
df_stats = pd.DataFrame(data=training_stats).set_index("epoch")
df_stats



## === cell 39
plt.plot(df_stats["Training Loss"], "b-o", label="Training")
plt.plot(df_stats["Valid. Loss"], "g-o", label="Validation")
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks(list(df_stats.index))
plt.show()



## === cell 40
plt.plot(df_stats["Training Accur"], "b-o", label="Training")
plt.plot(df_stats["Valid. Accur."], "g-o", label="Validation")
plt.title("Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.xticks(list(df_stats.index))
plt.show()



## === cell 41
thresh = 0.4
with torch.no_grad():
    outputs = model(
        test_seq[0:1000].to(device), attention_mask=test_mask[0:1000].to(device)
    )
    pred_probs = torch.sigmoid(outputs.logits).cpu().numpy()



## === cell 42
pred_probs[:3]



## === cell 43
y_pred = (pred_probs > thresh).astype(int)



## === cell 44
y_pred[:3]



## === cell 45
y_true = np.array(test_df[categories])[0:1000]



## === cell 46
from sklearn.metrics import hamming_loss, accuracy_score




## === cell 47
def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / float(
                len(set_true.union(set_pred))
            )
        acc_list.append(tmp_a)
    return np.mean(acc_list)




## === cell 48
print("accuracy_score:", accuracy_score(y_true, y_pred))
print("Hamming_score:", hamming_score(y_true, y_pred))
print("Hamming_loss:", hamming_loss(y_true, y_pred))



## === cell 49
PATH = "./toxic_distilBERT_multilabel.pt"
torch.save(model.state_dict(), PATH)



## === cell 50
test_csv.head()



## === cell 51
len(test_csv)



## === cell 52
sub_tokens = tokenizer(
    test_csv["comment_text"].fillna("").tolist(),
    max_length=MAX_LEN,
    padding="max_length",
    truncation=True,
    return_token_type_ids=False,
)



## === cell 53
sub_seq = torch.tensor(sub_tokens["input_ids"], dtype=torch.long)
sub_mask = torch.tensor(sub_tokens["attention_mask"], dtype=torch.long)



## === cell 54
sub_data = TensorDataset(sub_seq, sub_mask)



## === cell 55
sub_dataloader = DataLoader(sub_data, batch_size=batch_size)



## === cell 56
model.eval()
t0 = time.time()

pred_chunks = []
for step, batch in enumerate(sub_dataloader):
    if step % 400 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print(
            "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                step, len(sub_dataloader), elapsed
            )
        )

    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)

    with torch.no_grad():
        outputs = model(b_input_ids, attention_mask=b_input_mask)
        pred_probs = torch.sigmoid(outputs.logits).cpu().numpy()
        pred_chunks.append(pred_probs)

predictions = np.vstack(pred_chunks)



## === cell 57
predictions_df = pd.DataFrame(predictions, columns=categories)



## === cell 58
len(predictions_df), predictions_df.head()



## === cell 59
submission = pd.concat([test_csv["id"], predictions_df], axis=1)
submission = submission[["id"] + categories]



## === cell 60
submission.head()



## === cell 61
submission.to_csv("submission.csv", index=False, header=True)
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
