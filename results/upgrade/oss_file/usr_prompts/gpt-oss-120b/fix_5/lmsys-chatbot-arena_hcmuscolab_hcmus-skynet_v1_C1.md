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
Predict which chatbot response a user will prefer in a competition between two chatbots.

## Metric
Log loss with "eps=auto"

## Submission Format
For each id in the test set, you must predict the probability for each target class. The file should contain a header and have the following format:

```
 id,winner_model_a,winner_model_b,winner_tie
 136060,0.33,0,33,0.33
 211333,0.33,0,33,0.33
 1233961,0.33,0,33,0.33
 etc
```

## Dataset
**train.csv**

- `id` - A unique identifier for the row.
- `model_[a/b]` - The identity of model_[a/b]. Included in train.csv but not test.csv.
- `prompt` - The prompt that was given as an input (to both models).
- `response_[a/b]` - The response from model_[a/b] to the given prompt.
- `winner_model_[a/b/tie]` - Binary columns marking the judge's selection. The ground truth target column.

**test.csv**

- `id`
- `prompt`
- `response_[a/b]`

**sample_submission.csv** A submission file in the correct format.

- `id`
- `winner_model_[a/b/tie]` - This is what is predicted from the test set.

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
plotly==5.24.1
plotly-express==0.4.1
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tf_keras==2.18.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        input/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        working/
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
```

-> data/lmsys-chatbot-arena/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/lmsys-chatbot-arena/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/lmsys-chatbot-arena/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> data/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> (stopped after 10 files for performance)

# 5. Target score

1.1289065843520338

# 6. Current score

1.25878

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.25878) has done: 'Implemented two key fixes:

1. **Environment & Imports** – Set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` to `"python"` to avoid the protobuf import error, then import TensorFlow, Keras, and Keras‑NLP in the correct order.
2. **Dataset Shuffling** – Replaced the incorrect boolean buffer size in `tf.data.Dataset.shuffle` with a safe integer buffer (`1024`) while keeping the seed for reproducibility.

These changes eliminate the import failures and allow the full training‑inference pipeline to run, producing a valid `submission.csv` file.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
import keras_nlp

import numpy as np
import pandas as pd
from tqdm import tqdm
import json

import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("KerasNLP:", keras_nlp.__version__)




## === cell 2
class CFG:
    seed = 69  # Random seed
    preset = "bert_tiny_en_uncased"  # Name of pretrained models
    sequence_length = 512  # Input sequence length
    epochs = 8  # Training epochs
    batch_size = 16  # Batch size
    scheduler = "cosine"  # Learning rate scheduler
    label2name = {0: "winner_model_a", 1: "winner_model_b", 2: "winner_tie"}
    name2label = {v: k for k, v in label2name.items()}
    class_labels = list(label2name.keys())
    class_names = list(label2name.values())




## === cell 3
keras.utils.set_random_seed(CFG.seed)



## === cell 4
keras.mixed_precision.set_global_policy("mixed_float16")



## === cell 5
BASE_PATH = "/kaggle/input/lmsys-chatbot-arena"



## === cell 6
df = pd.read_csv(f"{BASE_PATH}/train.csv", on_bad_lines="skip", engine="python")

df["prompt"] = df.prompt.map(lambda x: eval(x.lower())[0])
df["response_a"] = df.response_a.map(lambda x: eval(x.replace("null", "''").lower())[0])
df["response_b"] = df.response_b.map(lambda x: eval(x.replace("null", "''").lower())[0])

df["class_name"] = df[["winner_model_a", "winner_model_b", "winner_tie"]].idxmax(axis=1)
df["class_label"] = df.class_name.map(CFG.name2label)

df.head()



## === cell 7
test_df = pd.read_csv(f"{BASE_PATH}/test.csv")

test_df["prompt"] = test_df.prompt.map(lambda x: eval(x.lower())[0])
test_df["response_a"] = test_df.response_a.map(
    lambda x: eval(x.replace("null", "''").lower())[0]
)
test_df["response_b"] = test_df.response_b.map(
    lambda x: eval(x.replace("null", "''").lower())[0]
)

test_df.head()




## === cell 8
def make_pairs(row):
    row["encode_fail"] = False
    try:
        prompt = row.prompt.encode("utf-8").decode("utf-8")
    except:
        prompt = ""
        row["encode_fail"] = True

    try:
        response_a = row.response_a.encode("utf-8").decode("utf-8")
    except:
        response_a = ""
        row["encode_fail"] = True

    try:
        response_b = row.response_b.encode("utf-8").decode("utf-8")
    except:
        response_b = ""
        row["encode_fail"] = True

    row["options"] = [
        f"Prompt: {prompt}\n\nResponse: {response_a}",
        f"Prompt: {prompt}\n\nResponse: {response_b}",
    ]
    return row




## === cell 9
df = df.apply(make_pairs, axis=1)
display(df.head())

test_df = test_df.apply(make_pairs, axis=1)
display(test_df.head())



## === cell 10
df.encode_fail.value_counts(normalize=False)



## === cell 11
model_df = pd.concat([df.model_a, df.model_b])
counts = model_df.value_counts().reset_index()
counts.columns = ["LLM", "Count"]

fig = px.bar(
    counts,
    x="LLM",
    y="Count",
    title="Distribution of LLMs",
    color="Count",
    color_continuous_scale="viridis",
)

fig.update_layout(xaxis_tickangle=-45)
fig.show()



## === cell 12
counts = df["class_name"].value_counts().reset_index()
counts.columns = ["Winner", "Win Count"]

fig = px.bar(
    counts,
    x="Winner",
    y="Win Count",
    title="Winner distribution for Train Data",
    labels={"Winner": "Winner", "Win Count": "Win Count"},
    color="Winner",
    color_continuous_scale="viridis",
)

fig.update_layout(xaxis_title="Winner", yaxis_title="Win Count")
fig.show()



## === cell 13
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(
    df, test_size=0.2, stratify=df["class_label"], random_state=CFG.seed
)



## === cell 14
preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
    preset=CFG.preset,
    sequence_length=CFG.sequence_length,
)



## === cell 15
outs = preprocessor(df.options.iloc[0])
for k, v in outs.items():
    print(k, ":", v.shape)




## === cell 16
def preprocess_fn(text, label=None):
    text = preprocessor(text)
    return (text, label) if label is not None else text




## === cell 17
def build_dataset(texts, labels=None, batch_size=32, cache=True, shuffle=True):
    AUTO = tf.data.AUTOTUNE
    slices = (
        (texts,)
        if labels is None
        else (texts, keras.utils.to_categorical(labels, num_classes=3))
    )
    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.cache() if cache else ds
    ds = ds.map(preprocess_fn, num_parallel_calls=AUTO)
    opt = tf.data.Options()
    if shuffle:
        ds = ds.shuffle(buffer_size=1024, seed=CFG.seed)
        opt.experimental_deterministic = False
    ds = ds.with_options(opt)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds




## === cell 18
train_texts = train_df.options.tolist()
train_labels = train_df.class_label.tolist()
train_ds = build_dataset(
    train_texts, train_labels, batch_size=CFG.batch_size, shuffle=True
)

valid_texts = valid_df.options.tolist()
valid_labels = valid_df.class_label.tolist()
valid_ds = build_dataset(
    valid_texts, valid_labels, batch_size=CFG.batch_size, shuffle=False
)



## === cell 19
import math


def get_lr_callback(batch_size=8, mode="cos", epochs=10, plot=False):
    lr_start, lr_max, lr_min = 1.0e-6, 0.6e-6 * batch_size, 1e-6
    lr_ramp_ep, lr_sus_ep, lr_decay = 2, 0, 0.8

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        elif mode == "exp":
            lr = (lr_max - lr_min) * lr_decay ** (
                epoch - lr_ramp_ep - lr_sus_ep
            ) + lr_min
        elif mode == "step":
            lr = lr_max * lr_decay ** ((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == "cos":
            decay_total_epochs = epochs - lr_ramp_ep - lr_sus_ep + 3
            decay_epoch_index = epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(epochs), [lrfn(e) for e in np.arange(epochs)], marker="o")
        plt.xlabel("epoch")
        plt.ylabel("lr")
        plt.title("LR Scheduler")
        plt.show()

    return keras.callbacks.LearningRateScheduler(lrfn, verbose=False)




## === cell 20
lr_cb = get_lr_callback(CFG.batch_size, plot=False)



## === cell 21
ckpt_cb = keras.callbacks.ModelCheckpoint(
    "best_model.weights.h5",
    monitor="val_log_loss",
    save_best_only=True,
    save_weights_only=True,
    mode="min",
)



## === cell 22
log_loss = keras.metrics.CategoricalCrossentropy(name="log_loss")



## === cell 23
inputs = {
    "token_ids": keras.Input(shape=(2, None), dtype=tf.int32, name="token_ids"),
    "padding_mask": keras.Input(shape=(2, None), dtype=tf.int32, name="padding_mask"),
    "segment_ids": keras.Input(shape=(2, None), dtype=tf.int32, name="segment_ids"),
}
backbone = keras_nlp.models.BertBackbone.from_preset(CFG.preset)

response_a = {k: v[:, 0, :] for k, v in inputs.items()}
embed_a = backbone(response_a)

response_b = {k: v[:, 1, :] for k, v in inputs.items()}
embed_b = backbone(response_b)

embeds = keras.layers.Concatenate(axis=-1)(
    [embed_a["sequence_output"], embed_b["sequence_output"]]
)
embeds = keras.layers.GlobalAveragePooling1D()(embeds)
outputs = keras.layers.Dense(3, activation="softmax", name="classifier")(embeds)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(5e-6),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
    metrics=[log_loss, keras.metrics.CategoricalAccuracy(name="accuracy")],
)



## === cell 24
model.summary()



## === cell 25
history = model.fit(
    train_ds,
    epochs=CFG.epochs,
    validation_data=valid_ds,
    callbacks=[lr_cb, ckpt_cb],
)



## === cell 26
model.load_weights("best_model.weights.h5")



## === cell 27
test_texts = test_df.options.tolist()
test_ds = build_dataset(
    test_texts, batch_size=min(len(test_df), CFG.batch_size), shuffle=False
)

test_preds = model.predict(test_ds, verbose=1).astype(np.float32)

row_sums = np.clip(test_preds.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)
test_preds = test_preds / row_sums

preds_df = pd.DataFrame(test_preds, columns=CFG.class_names)
sub_df = pd.concat([test_df[["id"]].reset_index(drop=True), preds_df], axis=1)

sub_df.to_csv("submission.csv", index=False)
sub_df.head()
